# 📚 手动DDPM实现说明

## 🎯 核心目标
完全手动实现DDPM训练循环，确保：
- ✅ Diffusion loss > 0
- ✅ 完全控制训练流程
- ✅ Video-to-Video Diffusion
- ✅ 条件生成（基于过去帧）

---

## 🏗️ 架构概览

```
Input: past_frames (B, 8, 24, 64, 64)
         ↓
   [ConvLSTM Encoder] → convlstm_features (B, 64, 64, 64)
         ↓
   [Condition Projection] → cond_video (B, 24, 12, 64, 64)
         ↓                              ↓
   [Track Predictor]           [Manual DDPM Training]
         ↓                              ↓
   predicted_track              future_frames (B, 12, 24, 64, 64)
```

---

## 🔬 DDPM实现细节

### 1. Noise Schedule (Beta Schedule)

```python
# Linear schedule
betas = linspace(1e-4, 0.02, timesteps=250)

# Pre-compute useful quantities
alphas = 1 - betas
alphas_cumprod = cumprod(alphas)
sqrt_alphas_cumprod = sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = sqrt(1 - alphas_cumprod)
```

**为什么预计算？**
- 训练时需要随机采样t，每次都计算太慢
- 预计算所有timesteps的系数，直接查表即可

---

### 2. Forward Diffusion (q_sample)

```python
def q_sample(x_start, t, noise):
    """
    给干净数据加噪声
    x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
    """
    sqrt_alpha_prod = sqrt_alphas_cumprod[t]
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alphas_cumprod[t]
    
    return sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise
```

**输入**：
- `x_start`: (B, C, T, H, W) - 干净的未来帧
- `t`: (B,) - 每个样本的时间步
- `noise`: (B, C, T, H, W) - 随机噪声

**输出**：
- `x_noisy`: (B, C, T, H, W) - 加噪声后的未来帧

**关键点**：
- `_extract` 函数将1D系数扩展为5D以匹配video tensor
- 支持batch内每个样本不同的t

---

### 3. Training Loss (p_losses)

```python
def p_losses(unet, x_start, t, cond_video, noise):
    """
    DDPM训练核心：预测噪声
    """
    # 1. 加噪声
    x_noisy = q_sample(x_start, t, noise)
    
    # 2. UNet预测噪声（条件是cond_video）
    predicted_noise = unet(
        x_noisy,
        time=t,
        cond_video_frames=cond_video
    )
    
    # 3. MSE loss
    loss = MSE(predicted_noise, noise)
    
    return loss
```

**这是DDPM的精髓！**
- 训练UNet学会：给定 `x_t` 和条件 `cond_video`，预测噪声 `ε`
- Loss是预测噪声与真实噪声的MSE
- 这个loss **必定 > 0**（除非完美预测）

---

### 4. Sampling (p_sample_loop)

```python
@torch.no_grad()
def p_sample_loop(unet, shape, cond_video, device):
    """
    推理：从噪声生成干净数据
    x_T → x_{T-1} → ... → x_0
    """
    # 从纯噪声开始
    x = randn(shape, device=device)
    
    # 逐步去噪
    for t in reversed(range(timesteps)):
        x = p_sample(unet, x, t, cond_video)
    
    return x  # x_0: 干净的future_frames
```

**推理流程**：
1. 从 `x_T ~ N(0, I)` 开始（纯噪声）
2. 对于 t = T-1, T-2, ..., 0：
   - 用UNet预测噪声
   - 计算 `x_{t-1}` = f(x_t, predicted_noise, t)
3. 最终得到 `x_0`（干净数据）

---

## 🎓 完整训练循环

```python
for epoch in epochs:
    for batch in train_loader:
        # 1. 编码过去帧
        _, cond_video, predicted_track = model.encode_past_frames(past_frames)
        
        # 2. 转换为 (B, C, T, H, W) for diffusion
        future_frames_perm = future_frames.permute(0, 2, 1, 3, 4)
        
        # 3. 随机采样timesteps
        t = randint(0, 250, size=(B,))
        
        # 4. 计算DDPM loss（这是关键！）
        diffusion_loss = model.diffusion.p_losses(
            model.video_unet,
            future_frames_perm,  # x_start (干净数据)
            t,                   # timesteps
            cond_video           # condition
        )
        
        # 5. 计算track loss
        track_loss = MSE(predicted_track, track_future)
        
        # 6. 总loss
        total_loss = diffusion_loss + track_loss
        
        # 7. 反向传播
        total_loss.backward()
        optimizer.step()
```

---

## ✨ 与ImagenTrainer的区别

| 特性 | ImagenTrainer | 手动DDPM |
|------|---------------|----------|
| **loss计算** | 内部处理，返回float | 返回tensor，可backward |
| **控制力** | 黑盒，难调试 | 完全透明，易调试 |
| **灵活性** | 受限于库接口 | 可自由定制 |
| **loss是否>0** | ❌ 经常是0 | ✅ 保证>0 |
| **梯度传播** | ❌ 内部调用backward | ✅ 外部控制 |

---

## 🔍 关键创新点

### 1. Video-to-Video Diffusion
- 输入/输出都是5D tensor `(B, C, T, H, W)`
- 使用 `Unet3D` 处理时空数据
- 条件 `cond_video` 也是5D tensor

### 2. 条件生成
```python
# ConvLSTM features → 条件video
cond_video_expanded = convlstm_features.unsqueeze(2).repeat(1, 1, T_future, 1, 1)
cond_video = conv3d(cond_video_expanded)  # (B, C, T, H, W)

# 传给Unet3D
predicted_noise = unet(x_noisy, time=t, cond_video_frames=cond_video)
```

### 3. Multi-task Learning
- 同时预测气象场（diffusion）和轨迹（track）
- `total_loss = λ_diffusion * diffusion_loss + λ_track * track_loss`

---

## 🚀 使用方法

### 训练
```bash
cd C:\Users\fyp\Desktop\fyp\typhoon_prediction
. .\pytorch_gpu\Scripts\Activate.ps1
python train_hybrid_manual_ddpm.py
```

### 推理
```python
model.eval()
with torch.no_grad():
    sampled_frames, predicted_track = model.sample(past_frames)
```

---

## 📊 预期结果

### 训练输出
```
Epoch 1/50 [Train]: 100%|██████████| 36/36 [00:45<00:00]
  D_loss: 0.2543, T_loss: 0.1234, Total: 0.3777

Epoch 1/50 Summary:
  Train - Diffusion: 0.2543, Track: 0.1234, Total: 0.3777
  Val   - Diffusion: 0.2891, Track: 0.1456, Total: 0.4347
  [OK] Saved best model (val_loss=0.4347)
```

**关键指标**：
- ✅ `Diffusion loss` 应该在 0.1 - 0.5 之间（**不再是0！**）
- ✅ 随epoch增加应该下降
- ✅ 可以正常backward和更新参数

---

## 🎯 技术优势

1. **透明性**：每一步都清楚发生了什么
2. **可调试**：可以打印中间结果验证
3. **可扩展**：易于添加新功能（如DDIM采样）
4. **稳定性**：不依赖第三方trainer的黑盒逻辑

---

## 📖 参考文献

- [DDPM Original Paper](https://arxiv.org/abs/2006.11239)
- [Video Diffusion Models](https://arxiv.org/abs/2204.03458)
- [Imagen Video](https://arxiv.org/abs/2210.02303)

---

**准备好了吗？让我们开始训练，见证 loss > 0 的时刻！** 🎉

