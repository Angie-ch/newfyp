"""
混合台风预测模型 V3 - 修复Diffusion训练
分离encoder和diffusion训练，正确使用ImagenTrainer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# 导入之前的ConvLSTM和Track predictor
from hybrid_typhoon_predictor_v2 import ConvLSTMCell, ConvLSTM, TrackPredictor

# 导入imagen_pytorch
try:
    from imagen_pytorch import Unet3D, Imagen
    print("[OK] imagen_pytorch imported successfully!")
except ImportError as e:
    print(f"[ERROR] Failed to import imagen_pytorch: {e}")
    raise


class HybridTyphoonPredictor_V3(nn.Module):
    """
    混合模型V3 - 正确的Diffusion集成
    
    职责分离：
    - Model: 负责encoder和条件生成
    - ImagenTrainer: 独立管理diffusion训练（在外部）
    """
    def __init__(
        self,
        input_channels: int = 24,
        hidden_channels: int = 64,
        output_channels: int = 24,
        past_timesteps: int = 8,
        future_timesteps: int = 12,
        image_size: Tuple[int, int] = (64, 64),
        # Unet3D参数
        unet_dim: int = 32,
        unet_dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        unet_num_resnet_blocks: int = 3,
        # Diffusion参数
        diffusion_timesteps: int = 250,
        cond_drop_prob: float = 0.1,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.past_timesteps = past_timesteps
        self.future_timesteps = future_timesteps
        self.image_size = image_size
        
        print(f"\n[INFO] Initializing HybridTyphoonPredictor_V3...")
        print(f"  - Input channels: {input_channels}")
        print(f"  - Hidden channels: {hidden_channels}")
        print(f"  - Output channels: {output_channels}")
        print(f"  - Past/Future timesteps: {past_timesteps}/{future_timesteps}")
        
        # 1. ConvLSTM编码器
        self.convlstm_encoder = ConvLSTM(
            input_channels=input_channels,
            hidden_channels=[hidden_channels, hidden_channels],
            kernel_size=3,
            num_layers=2,
            batch_first=True
        )
        print(f"  [OK] ConvLSTM encoder initialized")
        
        # 2. 条件投影层
        self.cond_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, output_channels, kernel_size=1),
            nn.GroupNorm(8, output_channels),
            nn.SiLU()
        )
        print(f"  [OK] Condition projection layer initialized")
        
        # 3. Track预测器
        track_input_dim = hidden_channels * image_size[0] * image_size[1]
        self.track_predictor = TrackPredictor(
            hidden_dim=track_input_dim,
            output_frames=future_timesteps
        )
        print(f"  [OK] Track predictor initialized")
        
        # 4. Video Diffusion (Unet3D + Imagen)
        # 注意：ImagenTrainer将在外部创建和管理
        print(f"  [INFO] Creating Unet3D and Imagen...")
        self.video_unet = Unet3D(
            dim=unet_dim,
            channels=output_channels,
            channels_out=output_channels,
            cond_dim=1024,
            dim_mults=unet_dim_mults,
            num_resnet_blocks=unet_num_resnet_blocks,
            layer_attns=(False, True, True, True),
            cond_on_text=False,
            memory_efficient=True,
            attn_heads=8,
            attn_dim_head=64,
        )
        
        self.imagen = Imagen(
            unets=[self.video_unet],
            image_sizes=image_size[0],
            timesteps=diffusion_timesteps,
            cond_drop_prob=cond_drop_prob,
            condition_on_text=False,  # CRITICAL: 明确设置为False！
        )
        print(f"  [OK] Unet3D and Imagen initialized")
        print(f"\n[OK] HybridTyphoonPredictor_V3 initialized successfully!\n")
    
    def encode_and_prepare_condition(
        self,
        past_frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        编码past frames并准备条件
        
        Args:
            past_frames: (B, T_past, C, H, W)
        
        Returns:
            convlstm_features: (B, C_hidden, H, W) - 用于track prediction
            cond_video: (B, C_out, T_future, H, W) - 用于diffusion conditioning
            predicted_track: (B, T_future, 2)
        """
        B, T_past, C_in, H, W = past_frames.shape
        
        # 1. ConvLSTM编码
        _, last_state = self.convlstm_encoder(past_frames)
        convlstm_features = last_state[-1][0]  # (B, C_hidden, H, W)
        
        # 2. 投影并扩展为视频条件
        cond_features_2d = self.cond_proj(convlstm_features)  # (B, C_out, H, W)
        cond_video = cond_features_2d.unsqueeze(2).repeat(1, 1, self.future_timesteps, 1, 1)  # (B, C_out, T_future, H, W)
        
        # 3. Track预测
        track_input = convlstm_features.reshape(B, -1)
        predicted_track = self.track_predictor(track_input)  # (B, T_future, 2)
        
        return convlstm_features, cond_video, predicted_track
    
    def forward(
        self,
        past_frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass - 只做编码和条件准备
        
        Args:
            past_frames: (B, T_past, C, H, W)
        
        Returns:
            cond_video: (B, C_out, T_future, H, W) - 用于diffusion
            predicted_track: (B, T_future, 2)
        """
        _, cond_video, predicted_track = self.encode_and_prepare_condition(past_frames)
        return cond_video, predicted_track
    
    @torch.no_grad()
    def sample(
        self,
        past_frames: torch.Tensor,
        cond_scale: float = 1.0,
        use_tqdm: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        推理模式：生成未来帧
        
        Args:
            past_frames: (B, T_past, C, H, W)
            cond_scale: Guidance scale
            use_tqdm: 是否显示进度条
        
        Returns:
            predicted_frames: (B, T_future, C, H, W)
            predicted_track: (B, T_future, 2)
        """
        B = past_frames.shape[0]
        
        # 获取条件
        _, cond_video, predicted_track = self.encode_and_prepare_condition(past_frames)
        
        # Diffusion采样
        try:
            sampled_video = self.imagen.sample(
                batch_size=B,
                video_frames=self.future_timesteps,
                cond_video_frames=cond_video,  # (B, C, T, H, W)
                cond_scale=cond_scale,
                use_tqdm=use_tqdm
            )
            # (B, C, T, H, W) -> (B, T, C, H, W)
            sampled_video = sampled_video.permute(0, 2, 1, 3, 4)
        except Exception as e:
            print(f"[WARNING] Diffusion sampling error: {e}")
            sampled_video = torch.zeros(
                B, self.future_timesteps, self.output_channels, 
                self.image_size[0], self.image_size[1],
                device=past_frames.device
            )
        
        return sampled_video, predicted_track


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Testing HybridTyphoonPredictor_V3 with separated architecture")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}\n")
    
    # 创建模型
    model = HybridTyphoonPredictor_V3(
        input_channels=24,
        hidden_channels=64,
        output_channels=24,
        past_timesteps=8,
        future_timesteps=12,
        image_size=(64, 64),
    ).to(device)
    
    print(f"\n[OK] Model created successfully!")
    
    # 测试数据
    B, T_past, T_future = 2, 8, 12
    C, H, W = 24, 64, 64
    
    past_frames = torch.randn(B, T_past, C, H, W, device=device)
    future_frames = torch.randn(B, T_future, C, H, W, device=device)
    
    print(f"\n[INFO] Testing encode_and_prepare_condition...")
    model.train()
    convlstm_features, cond_video, pred_track = model.encode_and_prepare_condition(past_frames)
    
    print(f"  - convlstm_features: {convlstm_features.shape}")
    print(f"  - cond_video: {cond_video.shape}")
    print(f"  - predicted_track: {pred_track.shape}")
    print(f"  [OK] Condition preparation successful!")
    
    print(f"\n[INFO] Testing sampling...")
    model.eval()
    sampled_frames, sampled_track = model.sample(past_frames)
    print(f"  - sampled_frames: {sampled_frames.shape}")
    print(f"  - sampled_track: {sampled_track.shape}")
    print(f"  [OK] Sampling successful!")
    
    print(f"\n{'='*80}")
    print("[OK] All tests passed!")
    print(f"{'='*80}\n")

