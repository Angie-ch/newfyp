# 接口修复总结

## ✅ 已修复的问题

1. **Unet3D条件参数**: 使用`cond_video_frames`而不是`cond`
2. **通道数匹配**: 添加`condition_video_proj`将64通道投影到24通道
3. **Imagen参数**: 移除了不支持的`condition_on_continuous`

## ⚠️ 当前问题

**Unet3D内部维度不匹配**: 
- Unet3D的某些注意力层期望4D输入 (B, C, H, W)
- 但我们传入的是5D视频 (B, C, T, H, W)

**错误信息**:
```
einops.EinopsError: Wrong shape: expected 4 dims. Received 5-dim tensor.
Input tensor shape: torch.Size([2, 256, 24, 32, 32])
```

## 🔍 分析

从forecast-video-diffmodels的代码看，他们使用`cond_video_frames`来传递条件视频。但Unet3D内部可能对视频的处理方式不同。

## 💡 解决方案

### 选项1: 检查Unet3D的视频处理
- Unet3D应该支持视频输入
- 可能需要特定的参数配置

### 选项2: 使用Unet而不是Unet3D
- forecast-video-diffmodels的某些脚本使用`Unet`而不是`Unet3D`
- 但我们需要视频支持

### 选项3: 修改条件传递方式
- 不使用`cond_video_frames`
- 改用其他条件方式（如text_embeds，但需要正确维度）

## 📝 下一步

需要检查forecast-video-diffmodels中Unet3D的正确使用方式，特别是如何处理5D视频输入。

