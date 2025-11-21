"""
自定义Video Diffusion模型实现
完全支持5D视频输入 (B, C, T, H, W)，不依赖imagen-pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ============================================================================
# 基础模块
# ============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """时间步的正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock3D(nn.Module):
    """3D残差块，支持时间维度"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # 动态计算num_groups以适应不同的通道数
        num_groups_1 = min(8, in_channels) if in_channels >= 8 else 1
        num_groups_2 = min(8, out_channels) if out_channels >= 8 else 1
        self.norm1 = nn.GroupNorm(num_groups_1, in_channels)
        self.norm2 = nn.GroupNorm(num_groups_2, out_channels)
        
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        # 残差连接
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        """
        x: (B, C, T, H, W)
        time_emb: (B, time_emb_dim)
        """
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        # 添加时间嵌入
        time_emb = self.time_mlp(time_emb)
        # (B, C) -> (B, C, 1, 1, 1) 用于广播
        h = h + time_emb[:, :, None, None, None]
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class DownBlock3D(nn.Module):
    """下采样块（空间下采样，保持时间维度）"""
    def __init__(self, in_channels, out_channels, time_emb_dim, num_res_blocks=2):
        super().__init__()
        
        self.res_blocks = nn.ModuleList([
            ResBlock3D(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim
            ) for i in range(num_res_blocks)
        ])
        
        # 空间下采样（只在H和W维度）
        self.downsample = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=(1, 3, 3),  # 时间维度kernel=1
            stride=(1, 2, 2),       # 时间维度stride=1
            padding=(0, 1, 1)
        )
    
    def forward(self, x, time_emb):
        for res_block in self.res_blocks:
            x = res_block(x, time_emb)
        h = x
        x = self.downsample(x)
        return x, h  # 返回下采样后的x和跳跃连接h


class UpBlock3D(nn.Module):
    """上采样块（空间上采样，保持时间维度）"""
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim, num_res_blocks=2):
        super().__init__()
        
        # 上采样
        self.upsample = nn.ConvTranspose3d(
            in_channels, in_channels,
            kernel_size=(1, 4, 4),  # 时间维度kernel=1
            stride=(1, 2, 2),       # 时间维度stride=1
            padding=(0, 1, 1)
        )
        
        # 第一个ResBlock接收连接后的输入（in_channels + skip_channels）
        # 后续的ResBlock接收out_channels
        self.res_blocks = nn.ModuleList([
            ResBlock3D(
                in_channels + skip_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim
            ) for i in range(num_res_blocks)
        ])
    
    def forward(self, x, skip, time_emb):
        x = self.upsample(x)
        # 连接跳跃连接
        x = torch.cat([x, skip], dim=1)
        for res_block in self.res_blocks:
            x = res_block(x, time_emb)
        return x


class MiddleBlock3D(nn.Module):
    """中间瓶颈块"""
    def __init__(self, channels, time_emb_dim, num_res_blocks=2):
        super().__init__()
        
        self.res_blocks = nn.ModuleList([
            ResBlock3D(channels, channels, time_emb_dim)
            for _ in range(num_res_blocks)
        ])
    
    def forward(self, x, time_emb):
        for res_block in self.res_blocks:
            x = res_block(x, time_emb)
        return x


# ============================================================================
# Video UNet3D
# ============================================================================

class VideoUNet3D(nn.Module):
    """
    自定义3D UNet用于视频扩散
    完全支持5D输入 (B, C, T, H, W)
    
    参考forecast-video-diffmodels的配置：
    - dim = 32 (base_channels)
    - dim_mults = (1, 2, 4, 8)
    - num_resnet_blocks = 3
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,  # ConvLSTM条件特征的通道数
        base_channels: int = 32,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 3,
        time_emb_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4),
        )
        time_emb_dim = time_emb_dim * 4
        
        # 条件投影（从ConvLSTM特征投影到与输入相同的通道数）
        num_groups_cond = min(8, in_channels) if in_channels >= 8 else 1
        self.cond_proj = nn.Sequential(
            nn.Conv3d(cond_channels, in_channels, kernel_size=1),
            nn.GroupNorm(num_groups_cond, in_channels),
            nn.SiLU(),
        )
        
        # 初始卷积
        self.init_conv = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # 计算各层通道数
        channels = [base_channels * mult for mult in channel_mults]
        
        # 编码器（下采样）
        self.down_blocks = nn.ModuleList([])
        ch_in = base_channels
        for ch_out in channels:
            self.down_blocks.append(
                DownBlock3D(ch_in, ch_out, time_emb_dim, num_res_blocks)
            )
            ch_in = ch_out
        
        # 瓶颈
        self.middle_block = MiddleBlock3D(channels[-1], time_emb_dim, num_res_blocks)
        
        # 解码器（上采样）
        # 需要反向遍历channels，同时跳跃连接来自对应的down block
        self.up_blocks = nn.ModuleList([])
        for i in range(len(channels) - 1, 0, -1):
            self.up_blocks.append(
                UpBlock3D(
                    in_channels=channels[i],      # 来自更深层或middle block
                    skip_channels=channels[i-1],   # 来自对应的down block
                    out_channels=channels[i-1],    # 输出到下一层
                    time_emb_dim=time_emb_dim,
                    num_res_blocks=num_res_blocks
                )
            )
        
        # 最后一个上采样块（从channels[0]回到base_channels）
        self.up_blocks.append(
            UpBlock3D(
                in_channels=channels[0],
                skip_channels=base_channels,  # 来自init_conv
                out_channels=base_channels,
                time_emb_dim=time_emb_dim,
                num_res_blocks=num_res_blocks
            )
        )
        
        # 最终输出卷积
        num_groups_final = min(8, base_channels) if base_channels >= 8 else 1
        self.final_conv = nn.Sequential(
            nn.GroupNorm(num_groups_final, base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond_video: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) - noisy future frames
            time: (B,) - diffusion timestep
            cond_video: (B, C_cond, T, H, W) - condition from ConvLSTM
        
        Returns:
            predicted_noise: (B, C, T, H, W)
        """
        # 时间嵌入
        time_emb = self.time_embed(time)  # (B, time_emb_dim)
        
        # 投影条件
        cond = self.cond_proj(cond_video)  # (B, C, T, H, W)
        
        # 将条件添加到输入
        x = x + cond
        
        # 初始卷积
        x = self.init_conv(x)
        
        # 编码器
        skip_connections = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, time_emb)
            skip_connections.append(skip)
        
        # 瓶颈
        x = self.middle_block(x, time_emb)
        
        # 解码器
        for up_block in self.up_blocks:
            skip = skip_connections.pop()
            x = up_block(x, skip, time_emb)
        
        # 最终输出
        x = self.final_conv(x)
        
        return x


# ============================================================================
# Video Diffusion Trainer
# ============================================================================

class VideoDiffusionTrainer:
    """
    简化的视频扩散训练器
    实现DDPM (Denoising Diffusion Probabilistic Models)
    """
    def __init__(
        self,
        unet: VideoUNet3D,
        timesteps: int = 1000,
        beta_schedule: str = 'linear',
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = 'cuda'
    ):
        self.unet = unet
        self.timesteps = timesteps
        self.device = device
        
        # 设置noise schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        elif beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(timesteps, device)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 用于采样的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # 后验分布的方差
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps, device, s=0.008):
        """余弦noise schedule"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=device)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        给干净的图像添加噪声
        
        Args:
            x_start: (B, C, T, H, W) - 干净的视频
            t: (B,) - 时间步
            noise: 可选的噪声，如果为None则生成
        
        Returns:
            x_noisy: (B, C, T, H, W) - 加噪后的视频
            noise: (B, C, T, H, W) - 使用的噪声
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 获取alpha值
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t]
        
        # 调整形状用于广播 (B, 1, 1, 1, 1)
        sqrt_alpha_prod = sqrt_alpha_prod[:, None, None, None, None]
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[:, None, None, None, None]
        
        # 添加噪声: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_noisy = sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise
        
        return x_noisy, noise
    
    def train_step(
        self,
        future_frames: torch.Tensor,
        cond_video: torch.Tensor
    ) -> torch.Tensor:
        """
        单步训练
        
        Args:
            future_frames: (B, C, T, H, W) - 目标未来帧
            cond_video: (B, C_cond, T, H, W) - 条件视频（从ConvLSTM）
        
        Returns:
            loss: 标量损失
        """
        B = future_frames.shape[0]
        device = future_frames.device
        
        # 随机采样时间步
        t = torch.randint(0, self.timesteps, (B,), device=device).long()
        
        # 添加噪声
        x_noisy, noise = self.add_noise(future_frames, t)
        
        # 预测噪声
        predicted_noise = self.unet(x_noisy, t, cond_video)
        
        # 计算MSE损失
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        cond_video: torch.Tensor,
        num_frames: int,
        guidance_scale: float = 3.0,
        return_all_timesteps: bool = False
    ) -> torch.Tensor:
        """
        DDPM采样过程
        
        Args:
            cond_video: (B, C_cond, T, H, W) - 条件视频
            num_frames: 生成的帧数（应该与T匹配）
            guidance_scale: classifier-free guidance的强度
            return_all_timesteps: 是否返回所有时间步的结果
        
        Returns:
            x: (B, C, T, H, W) - 生成的视频
        """
        B = cond_video.shape[0]
        C = self.unet.out_channels
        T = num_frames
        H, W = cond_video.shape[-2:]
        device = cond_video.device
        
        # 从纯噪声开始
        x = torch.randn(B, C, T, H, W, device=device)
        
        all_frames = [x] if return_all_timesteps else []
        
        # 逆向扩散过程
        for t_idx in reversed(range(self.timesteps)):
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            
            # 预测噪声
            predicted_noise = self.unet(x, t, cond_video)
            
            # 计算前一步的均值
            alpha = self.alphas[t][:, None, None, None, None]
            alpha_cumprod = self.alphas_cumprod[t][:, None, None, None, None]
            beta = self.betas[t][:, None, None, None, None]
            
            # 去噪
            if t_idx > 0:
                noise = torch.randn_like(x)
                posterior_variance = self.posterior_variance[t][:, None, None, None, None]
            else:
                noise = 0
                posterior_variance = 0
            
            # x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1-alpha_bar_t)) * predicted_noise) + sqrt(posterior_variance) * noise
            x = (
                self.sqrt_recip_alphas[t][:, None, None, None, None] *
                (x - beta / self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None, None] * predicted_noise)
                + torch.sqrt(posterior_variance) * noise
            )
            
            if return_all_timesteps:
                all_frames.append(x)
        
        if return_all_timesteps:
            return torch.stack(all_frames, dim=0)  # (timesteps, B, C, T, H, W)
        else:
            return x
    
    @torch.no_grad()
    def ddim_sample(
        self,
        cond_video: torch.Tensor,
        num_frames: int,
        ddim_steps: int = 50,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        DDIM采样（更快的采样方法）
        
        Args:
            cond_video: (B, C_cond, T, H, W)
            num_frames: 生成的帧数
            ddim_steps: DDIM步数（少于总timesteps）
            eta: DDIM参数（0=确定性，1=DDPM）
        
        Returns:
            x: (B, C, T, H, W)
        """
        B = cond_video.shape[0]
        C = self.unet.out_channels
        T = num_frames
        H, W = cond_video.shape[-2:]
        device = cond_video.device
        
        # 选择采样时间步的子集
        c = self.timesteps // ddim_steps
        ddim_timesteps = torch.arange(0, self.timesteps, c, device=device)
        ddim_timesteps = list(reversed(ddim_timesteps.tolist()))
        
        # 从纯噪声开始
        x = torch.randn(B, C, T, H, W, device=device)
        
        for i, t_idx in enumerate(ddim_timesteps):
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            
            # 预测噪声
            predicted_noise = self.unet(x, t, cond_video)
            
            # 获取下一个时间步
            prev_t_idx = ddim_timesteps[i + 1] if i < len(ddim_timesteps) - 1 else 0
            
            alpha_cumprod = self.alphas_cumprod[t_idx]
            alpha_cumprod_prev = self.alphas_cumprod[prev_t_idx] if prev_t_idx > 0 else torch.tensor(1.0, device=device)
            
            # 预测x0
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha_cumprod)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # 方向指向x_t
            dir_xt = torch.sqrt(1 - alpha_cumprod_prev - eta**2 * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * (1 - alpha_cumprod / alpha_cumprod_prev)) * predicted_noise
            
            # 随机项
            if eta > 0 and prev_t_idx > 0:
                noise = torch.randn_like(x)
                sigma = eta * torch.sqrt((1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * (1 - alpha_cumprod / alpha_cumprod_prev))
            else:
                noise = 0
                sigma = 0
            
            # 更新x
            x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt + sigma * noise
        
        return x


if __name__ == "__main__":
    # Test code
    print("Testing VideoUNet3D and VideoDiffusionTrainer...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    unet = VideoUNet3D(
        in_channels=24,      # ERA5 channels
        out_channels=24,
        cond_channels=64,    # ConvLSTM hidden channels
        base_channels=32,    # Consistent with forecast-video-diffmodels
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=3
    ).to(device)
    
    # Create trainer
    trainer = VideoDiffusionTrainer(
        unet=unet,
        timesteps=250,  # Consistent with forecast-video-diffmodels
        beta_schedule='linear',
        device=device
    )
    
    # Test data
    B, T = 2, 12
    H, W = 64, 64
    
    future_frames = torch.randn(B, 24, T, H, W, device=device)
    cond_video = torch.randn(B, 64, T, H, W, device=device)
    
    print(f"\nInput shapes:")
    print(f"  future_frames: {future_frames.shape}")
    print(f"  cond_video: {cond_video.shape}")
    
    # Test training step
    print("\nTesting training step...")
    loss = trainer.train_step(future_frames, cond_video)
    print(f"  Training loss: {loss.item():.4f}")
    
    # Test sampling
    print("\nTesting DDPM sampling...")
    sampled = trainer.sample(cond_video, num_frames=T)
    print(f"  Sampled output shape: {sampled.shape}")
    
    # Test DDIM sampling
    print("\nTesting DDIM sampling...")
    sampled_ddim = trainer.ddim_sample(cond_video, num_frames=T, ddim_steps=50)
    print(f"  DDIM sampled output shape: {sampled_ddim.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"\nTotal model parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    print("\n[OK] All tests passed!")

