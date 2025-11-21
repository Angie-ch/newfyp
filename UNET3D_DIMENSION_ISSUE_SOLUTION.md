# Unet3D 5D维度问题 - 解决方案

## 🔍 问题分析

根据对 [forecast-video-diffmodels](https://github.com/Ren-creater/forecast-video-diffmodels) 仓库的深入分析：

### 关键发现

1. **他们使用本地修改的imagen-pytorch**
   - 训练脚本 `cx2_dim64.pbs` 第14-16行：
     ```bash
     cd imagen-pytorch
     python setup.py develop
     cd ..
     ```
   - 这个 `imagen-pytorch` 目录**不在公开仓库中**

2. **Unet3D配置**
   - 从 `train64.py`:
     ```python
     from imagen_pytorch import Unet3D, Imagen, ImagenTrainer
     
     unet1 = Unet3D(
         dim = 32,
         cond_dim = 1024,
         dim_mults = (1, 2, 4, 8),
         num_resnet_blocks = 3,
         layer_attns = (False, True, True, True),
     )
     ```

3. **训练时使用cond_video_frames**
   - 从 `v_m01w_woERA5_64_FC.py` 第112行:
     ```python
     loss = trainer(vid_64,
                    cond_video_frames=vid_cond,
                    unet_number=k,
                    ignore_time=False)
     ```

## 🚨 根本问题

**我们使用的标准 `imagen-pytorch==2.1.0` 与他们的本地修改版本不同！**

当前错误：
```
einops.EinopsError: Wrong shape: expected 4 dims. Received 5-dim tensor.
Input tensor shape: torch.Size([2, 256, 20, 32, 32])
Pattern: "b (h c) x y -> (b h) (x y) c"
```

这表明标准版本的 `Unet3D` 内部的注意力层（第1003行）期望4D输入 `(B, C, H, W)`，但收到了5D输入 `(B, C, T, H, W)`。

## 💡 解决方案

### 方案A: 使用简化的自定义Video Diffusion UNet（推荐）

**不依赖有问题的imagen-pytorch，自己实现一个支持5D视频的UNet3D**

优点：
- ✅ 完全控制实现
- ✅ 避免依赖未公开的修改版本
- ✅ 可以针对typhoon预测优化
- ✅ 减少不必要的复杂度

缺点：
- ⚠️ 需要从头实现diffusion训练逻辑
- ⚠️ 需要时间调试和验证

### 方案B: 克隆并尝试找到作者的imagen-pytorch版本

联系仓库作者询问：
1. 他们使用的 `imagen-pytorch` 版本/分支
2. 是否有公开的修改版本
3. 如何解决5D维度问题

优点：
- ✅ 使用作者验证过的代码
- ✅ 可能获得额外的实现细节

缺点：
- ⚠️ 需要等待回复
- ⚠️ 可能没有公开该版本
- ⚠️ 依赖外部维护

### 方案C: 修改标准imagen-pytorch

尝试修改标准 `imagen-pytorch==2.1.0` 的 `Unet3D` 注意力层：

1. 找到问题的 `einops.rearrange` 调用（`imagen_video.py` 第1003行）
2. 修改以支持5D输入
3. 可能需要重构多个注意力模块

优点：
- ✅ 保留imagen框架的其他功能

缺点：
- ⚠️ 需要深入理解imagen-pytorch源码
- ⚠️ 可能引入其他bug
- ⚠️ 维护困难

## 🎯 建议的实现路径（方案A）

### 1. 创建简化的Video Diffusion UNet

```python
class VideoUNet3D(nn.Module):
    """简化的3D UNet用于视频扩散，完全支持5D输入"""
    def __init__(self, 
                 in_channels,
                 out_channels,
                 cond_channels,
                 base_channels=64,
                 channel_mults=(1, 2, 4, 8),
                 num_res_blocks=2):
        super().__init__()
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, base_channels * 4),
            nn.SiLU(),
            nn.Linear(base_channels * 4, base_channels * 4)
        )
        
        # 条件投影（从ConvLSTM特征）
        self.cond_proj = nn.Conv3d(cond_channels, in_channels, 1)
        
        # 编码器、瓶颈、解码器（使用3D卷积）
        # ... 使用Conv3d, GroupNorm, 3D残差块
        
    def forward(self, x, time, cond_video):
        # x: (B, C, T, H, W) - noisy future frames
        # time: (B,) - diffusion timestep
        # cond_video: (B, C_cond, T, H, W) - condition from ConvLSTM
        
        # 投影condition
        cond = self.cond_proj(cond_video)
        
        # 连接输入和条件
        x = x + cond  # 或 torch.cat([x, cond], dim=1)
        
        # U-Net forward pass with 3D convolutions
        # ...
```

### 2. 实现自定义Diffusion Trainer

```python
class VideoDiffusionTrainer:
    """简化的视频扩散训练器"""
    def __init__(self, unet, timesteps=1000, beta_schedule='linear'):
        self.unet = unet
        self.timesteps = timesteps
        # 设置noise schedule
        self.setup_noise_schedule(beta_schedule)
    
    def setup_noise_schedule(self, schedule_type):
        if schedule_type == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, self.timesteps)
        elif schedule_type == 'cosine':
            # ... cosine schedule
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x, t):
        # 添加噪声到x
        noise = torch.randn_like(x)
        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise, noise
    
    def train_step(self, future_frames, cond_video):
        B = future_frames.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=future_frames.device)
        
        noisy_frames, noise = self.add_noise(future_frames, t)
        predicted_noise = self.unet(noisy_frames, t, cond_video)
        
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    @torch.no_grad()
    def sample(self, cond_video, num_frames):
        # DDPM采样
        x = torch.randn(cond_video.shape[0], self.out_channels, 
                       num_frames, cond_video.shape[-2], cond_video.shape[-1])
        
        for t in reversed(range(self.timesteps)):
            # ... denoising步骤
        
        return x
```

### 3. 集成到HybridTyphoonPredictor

```python
class HybridTyphoonPredictor(nn.Module):
    def __init__(self, ...):
        # ConvLSTM encoder
        self.convlstm = ConvLSTM(...)
        
        # 自定义Video Diffusion (替代imagen-pytorch)
        self.video_unet = VideoUNet3D(...)
        self.diffusion_trainer = VideoDiffusionTrainer(self.video_unet)
        
        # Track predictor
        self.track_predictor = TrackPredictor(...)
```

## 📊 参考实现

可以参考以下项目的3D UNet实现：
1. **Video Diffusion Models** (Google Research): https://github.com/google-research/vdm
2. **Make-A-Video**: 3D U-Net for video generation
3. **PyTorch Video Models**: https://github.com/pytorch/vision

## 🔄 下一步

1. **立即行动**: 实现方案A（自定义Video UNet）
2. **同时进行**: 在forecast-video-diffmodels的GitHub上开issue询问imagen-pytorch版本
3. **备选方案**: 如果方案A遇到困难，考虑使用更简单的baseline（纯ConvLSTM或ConvGRU）

## 📝 时间估计

- **方案A实现**: 2-4小时（UNet结构 + diffusion逻辑）
- **调试和训练**: 4-8小时
- **总计**: 1-2天可以有初步结果

## ⚡ 关键要点

**forecast-video-diffmodels使用的不是标准的imagen-pytorch 2.1.0！**

他们有一个本地修改的版本，能够正确处理5D视频输入的attention layers。由于这个版本未公开，我们需要：

1. **自己实现** 一个支持5D的Video Diffusion UNet (推荐)
2. **或者联系作者** 获取他们的imagen-pytorch版本
3. **或者简化模型** 使用更简单的baseline避免这个问题

选择方案A可以让我们完全控制实现，避免依赖未公开的第三方修改。

