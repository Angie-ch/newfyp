# 混合模型架构说明: ConvLSTM + Video-to-Video Diffusion

## 🎯 设计理念

结合两个模型的优势:
- **ConvLSTM**: 高效的特征提取,理解过去帧的时空模式
- **Video Diffusion**: 高质量的视频生成,保持时间一致性

参考LT3P论文: 使用扩散模型进行轨迹预测,但改为**Video-to-Video**方式

---

## 🏗️ 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Past Frames                       │
│              (B, T_past=8, C=24, H=64, W=64)                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         PART 1: ConvLSTM Encoder (特征提取)                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Input Conv (24→64 channels)                        │   │
│  │  ConvLSTM Layers (2层, 处理8个时间步)               │   │
│  │  Output Projection (64 channels)                    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
┌──────────────────────┐            ┌──────────────────────────┐
│  Condition Features  │            │   Final Hidden State    │
│ (B, T_past, 64, H, W)│            │    (B, 64, H, W)        │
└──────────────────────┘            └──────────────────────────┘
        │                                       │
        │                                       ▼
        │                            ┌──────────────────────────┐
        │                            │   Track Predictor (MLP)  │
        │                            │   → (B, T_future, 2)     │
        │                            └──────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│    PART 2: Video-to-Video Diffusion (未来帧生成)          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Input: Noisy Future Frames                         │   │
│  │         (B, T_future=12, C=24, H, W)                 │   │
│  │  Condition: Past Features (from ConvLSTM)           │   │
│  │            (B, T_past, 64, H, W)                     │   │
│  │                                                       │   │
│  │  Unet3D (with temporal layers)                       │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  3D Convolutions (spatial + temporal)        │   │   │
│  │  │  Temporal Attention                          │   │   │
│  │  │  Self-Attention                              │   │   │
│  │  │  Residual Blocks                             │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                       │   │
│  │  Output: Predicted Noise                            │   │
│  │          (B, T_future, C, H, W)                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              OUTPUT: Future Frames + Track                  │
│  Future Frames: (B, T_future=12, C=24, H, W)               │
│  Track:        (B, T_future=12, 2)  [lon, lat]             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔑 关键创新点

### 1. **Video-to-Video Diffusion** (vs 传统Video Diffusion)

**传统Video Diffusion**:
```
Noise → Denoise → Video
(从纯噪声生成视频)
```

**Video-to-Video Diffusion** (我们的方法):
```
Past Video (条件) + Noisy Future Video → Denoise → Future Video
(以过去视频为条件,生成未来视频)
```

**优势**:
- ✅ 更好的时间连续性
- ✅ 条件信息更丰富 (ConvLSTM提取的特征)
- ✅ 符合LT3P的扩散预测思路

### 2. **ConvLSTM作为条件编码器**

ConvLSTM不仅提取特征,还:
- 理解时空动态
- 捕获台风运动模式
- 为扩散模型提供强条件信号

### 3. **多任务学习**

同时预测:
- **未来帧** (ERA5气象场): Video Diffusion
- **轨迹** (经纬度): MLP from ConvLSTM features
- **强度** (可选): 可以添加

---

## 📊 与LT3P的对比

| 特性 | LT3P | 我们的混合模型 |
|------|------|---------------|
| **扩散类型** | 轨迹扩散 | **Video-to-Video扩散** |
| **输入** | 过去轨迹 | 过去ERA5帧 + 轨迹 |
| **输出** | 未来轨迹 | 未来ERA5帧 + 轨迹 |
| **条件** | 物理约束 | ConvLSTM特征 |
| **架构** | 单一扩散模型 | ConvLSTM + Diffusion |

**我们的优势**:
- ✅ 同时预测气象场和轨迹
- ✅ Video-to-Video更直观
- ✅ ConvLSTM提供强条件

---

## 🚀 训练流程

### 阶段1: 扩散前向过程
```python
# 1. 编码过去帧
condition_features, final_hidden = encoder(past_frames)

# 2. 添加噪声到未来帧
timestep = random(0, 1000)
noise = randn_like(future_frames)
noisy_future = sqrt(alpha_t) * future_frames + sqrt(1-alpha_t) * noise

# 3. 预测噪声
predicted_noise = diffusion(noisy_future, condition_features, timestep)

# 4. 损失
loss_diffusion = MSE(predicted_noise, noise)
```

### 阶段2: 轨迹预测
```python
# 从ConvLSTM特征预测轨迹
predicted_track = track_predictor(final_hidden)

# 损失
loss_track = MSE(predicted_track, true_track)
```

### 总损失
```python
total_loss = loss_diffusion + 10.0 * loss_track
```

---

## 🎯 预期性能

基于论文和架构分析:

| 指标 | ConvLSTM | Video Diffusion | **混合模型 (预期)** |
|------|----------|-----------------|-------------------|
| **72h轨迹误差** | 695 km | ~550 km | **~500 km** ⬇️ |
| **时间一致性** | 中等 | 高 | **很高** ⬆️ |
| **图像质量** | 中等 | 高 | **很高** ⬆️ |
| **训练时间** | 15分钟 | 2-4小时 | **2-4小时** |
| **推理时间** | 0.1秒 | 2-5秒 | **2-5秒** |

**预期改进**:
- 轨迹误差: ⬇️ 28% (vs ConvLSTM)
- 时间一致性: ⬆️ 40%+ (SSIM)
- 图像质量: ⬆️ 20%+ (PSNR)

---

## 📝 代码结构

### 核心类

1. **ConvLSTMEncoder**: 提取过去帧特征
2. **VideoToVideoDiffusion**: Video-to-Video扩散模型
3. **HybridTyphoonPredictor**: 完整混合模型

### 训练脚本

`hybrid_convlstm_video_diffusion.py`:
- 数据加载
- 模型训练
- 损失计算
- 模型保存

---

## ✅ 优势总结

1. **结合优势**: ConvLSTM特征提取 + Diffusion高质量生成
2. **Video-to-Video**: 更直观的条件生成
3. **多任务**: 同时预测帧和轨迹
4. **LT3P启发**: 使用扩散模型,但改进为Video-to-Video
5. **时间一致性**: 扩散模型保证更好的时间连贯性

---

## 🚀 下一步

1. **训练混合模型**: `python hybrid_convlstm_video_diffusion.py`
2. **评估性能**: 对比ConvLSTM、Video Diffusion、混合模型
3. **可视化**: 生成预测视频和轨迹对比

**预期**: 混合模型将取得最佳性能! 🎯

