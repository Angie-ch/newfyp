# 使用imagen-pytorch的Unet3D - 最终解决方案

## ✅ 关键发现

从克隆的 [lucidrains/imagen-pytorch](https://github.com/lucidrains/imagen-pytorch) 仓库分析：

### 1. Unet3D **原生支持**5D视频输入！

从 `imagen-pytorch/imagen_pytorch/imagen_video.py` 第1681-1697行：

```python
def forward(
    self,
    x,
    time,
    *,
    lowres_cond_img = None,
    lowres_noise_times = None,
    text_embeds = None,
    text_mask = None,
    cond_images = None,
    cond_video_frames = None,  # ← 支持条件视频帧！
    post_cond_video_frames = None,
    self_cond = None,
    cond_drop_prob = 0.,
    ignore_time = False
):
    assert x.ndim == 5, 'input to 3d unet must have 5 dimensions (batch, channels, time, height, width)'
```

### 2. 我们之前的问题根源

**我们使用的是通过`pip install imagen-pytorch`安装的标准版本**，它确实应该支持5D输入。但是：

1. **forecast-video-diffmodels使用`python setup.py develop`安装了本地修改版本**
2. 他们可能修改了某些内部实现以避免5D attention的问题
3. 或者他们使用了特定的参数配置来避开问题区域

### 3. 正确的使用方式

从forecast-video-diffmodels和forecast-diffmodels的代码，他们使用：

```python
from imagen_pytorch import Unet3D, Imagen, ImagenTrainer

unet1 = Unet3D(
    dim = 32,
    cond_dim = 1024,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 3,
    layer_attns = (False, True, True, True),  # 关键配置
)

# 训练时
loss = trainer(
    vid_64,
    cond_video_frames=vid_cond,  # 使用cond_video_frames参数
    unet_number=k,
    ignore_time=False
)
```

## 🎯 解决方案

### 方案1：直接使用imagen-pytorch的Unet3D（推荐）⭐

**不需要自己实现，直接使用标准库！**

```python
from imagen_pytorch import Unet3D, Imagen, ImagenTrainer
import torch
import torch.nn as nn

class HybridTyphoonPredictor_v2(nn.Module):
    def __init__(self, ...):
        super().__init__()
        
        # ConvLSTM编码器（Model A）
        self.convlstm_encoder = ConvLSTM(...)
        
        # Video Diffusion使用imagen-pytorch的Unet3D（Model B）
        self.video_unet = Unet3D(
            dim=32,
            channels=24,  # ERA5 channels
            channels_out=24,
            cond_dim=1024,
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks=3,
            layer_attns=(False, True, True, True),
            cond_on_text=False,  # 不使用文本条件
            memory_efficient=True,
        )
        
        # Imagen包装器
        self.imagen = Imagen(
            unets=[self.video_unet],
            image_sizes=64,
            timesteps=250,
            cond_drop_prob=0.1
        )
        
        # ImagenTrainer处理diffusion训练
        self.diffusion_trainer = ImagenTrainer(self.imagen, lr=3e-4)
        
        # Track预测器
        self.track_predictor = TrackPredictor(...)
    
    def forward(self, past_frames, future_frames=None):
        B, T_past, C_in, H, W = past_frames.shape
        
        # 1. ConvLSTM编码past frames
        _, last_state = self.convlstm_encoder(past_frames)
        convlstm_features = last_state[-1][0]  # (B, C_hidden, H, W)
        
        # 2. 扩展为视频维度用于conditioning
        # 需要将(B, C, H, W)扩展为(B, C, T, H, W)
        T_future = future_frames.shape[1] if future_frames is not None else 12
        cond_video = convlstm_features.unsqueeze(2).repeat(1, 1, T_future, 1, 1)
        
        # 3. Track预测
        track_input = convlstm_features.reshape(B, -1)
        predicted_track = self.track_predictor(track_input)
        
        if self.training and future_frames is not None:
            # 训练时：使用ImagenTrainer
            # 注意：future_frames需要是(B, C, T, H, W)格式
            loss = self.diffusion_trainer(
                future_frames,  # 目标视频
                cond_video_frames=cond_video,  # 条件视频
                unet_number=1
            )
            return loss, predicted_track
        else:
            # 推理时：使用Imagen采样
            sampled_video = self.imagen.sample(
                batch_size=B,
                video_frames=T_future,
                cond_video_frames=cond_video,
                cond_scale=3.0
            )
            return sampled_video, predicted_track
```

**优势：**
- ✅ 使用经过充分测试的库
- ✅ 不需要自己实现复杂的diffusion逻辑
- ✅ 自动处理5D视频输入
- ✅ 包含采样、训练等完整功能
- ✅ 有社区支持和文档

### 方案2：继续完善我们的自定义实现

如果想要更多控制，可以修复我们的`custom_video_diffusion.py`中的GroupNorm问题。

**主要修复：**
1. 确保所有ResBlock的in_channels和out_channels正确传递
2. 动态计算GroupNorm的num_groups
3. 修复UpBlock的跳跃连接通道计算

但这需要更多调试时间。

## 📝 推荐行动计划

1. **立即**：使用imagen-pytorch的Unet3D重写混合模型
2. **测试**：确保可以正确forward和训练
3. **训练**：在我们的72个样本上训练混合模型
4. **评估**：与ConvLSTM baseline比较

## ⚠️ 重要注意事项

1. **数据格式**：imagen-pytorch期望视频格式为 `(B, C, T, H, W)`
2. **条件视频**：使用`cond_video_frames`参数传递ConvLSTM特征
3. **文本条件**：设置`cond_on_text=False`因为我们不需要文本
4. **训练**：使用`ImagenTrainer`而不是手动实现training loop

## 🚀 下一步

实现`HybridTyphoonPredictor_v2`，使用imagen-pytorch的Unet3D！

