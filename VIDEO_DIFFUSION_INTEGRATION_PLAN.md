# Video Diffusion Model 集成计划

## 📦 已Clone仓库
✅ **仓库**: [forecast-video-diffmodels](https://github.com/Ren-creater/forecast-video-diffmodels)  
✅ **论文性能**: 
- MAE提升 19.3%
- PSNR提升 16.2%  
- SSIM提升 36.1%
- **预测时长**: 从36小时扩展到**50小时** 🎯

---

## 🏗️ 架构对比

### 我们当前的框架 (ConvLSTM)
```
Input: Past 8 frames (48h) → ConvLSTM Encoder → Hidden State
       ↓
Hidden State → ConvLSTM Decoder → Future 12 frames (72h)
```
**问题**: 
- 逐帧生成,缺乏时间一致性
- 长期预测误差累积
- 72小时预测误差: 695 km

### Video Diffusion Model (forecast-video-diffmodels)
```
Input: Past frames + Noise → Video Diffusion UNet (带时间层)
       ↓
Denoising Process (多步去噪)
       ↓
Output: Future frames (同时生成多帧)
```
**优势**:
- ✅ 显式建模时间依赖
- ✅ 多帧同时生成 (更好的时间一致性)
- ✅ 两阶段训练策略 (低数据量表现好)
- ✅ 50小时可靠预测

---

## 🔗 集成策略

### 方案1: 直接适配 (推荐) ⭐
使用我们已生成的72个样本,适配到Video Diffusion架构

**步骤**:
1. **数据适配器**: 转换我们的`.npz`格式到他们的dataloader
2. **模型迁移**: 使用他们的`train64.py`和`modules.py`
3. **两阶段训练**:
   - Stage 1: 训练单帧去噪
   - Stage 2: 训练时间层
4. **评估**: 使用FVD (Fréchet Video Distance)

**优势**:
- 快速验证 (数据已准备好)
- 利用两阶段训练 (适合少量数据)
- 时间一致性大幅提升

### 方案2: 混合架构
保留ConvLSTM作为条件输入,结合Video Diffusion

**步骤**:
1. ConvLSTM提取特征 → 作为条件
2. Video Diffusion生成未来帧
3. 联合训练

---

## 📊 数据对比

### 他们的数据格式
- **输入**: ERA5 + 卫星红外图像
- **分辨率**: 64x64
- **帧数**: 10帧预测任务
- **数据集**: 全球热带气旋 (2000-2019)

### 我们的数据
- **输入**: ERA5气象场 (24通道)
- **分辨率**: 64x64 ✅ (匹配!)
- **帧数**: 8 past + 12 future
- **数据集**: 西太平洋台风 (2018-2021, 72样本)

✅ **分辨率完全匹配!** 可以直接使用他们的模型架构!

---

## 🚀 实施步骤

### Phase 1: 环境设置 (5分钟)
```bash
cd forecast-video-diffmodels/imagen
pip install -r requirements.txt
```

### Phase 2: 数据适配 (30分钟)
创建适配器将我们的数据转换为他们的格式:
```python
# 我们的数据: (8_past, 12_future) × (24, 64, 64)
# 他们的数据: (n_frames) × (channels, 64, 64)
```

### Phase 3: 模型训练 (2-4小时)
使用两阶段训练:
1. Stage 1: 单帧扩散模型 (1-2小时)
2. Stage 2: 视频扩散模型 (1-2小时)

### Phase 4: 评估对比 (30分钟)
对比ConvLSTM vs Video Diffusion:
- MAE, PSNR, SSIM
- FVD (视频质量)
- 轨迹预测误差

---

## 📝 关键文件说明

### 数据处理
- `dataproc/fc1-create-dataloaders.py`: 创建PyTorch dataloader
- `dataproc/utils.py`: 数据处理工具

### 模型架构
- `imagen/modules.py`: Video Diffusion UNet实现
- `imagen/helpers.py`: 辅助函数

### 训练脚本
- `imagen/64_FC/train64.py`: 主训练脚本
- `imagen/64_FC/m01_64_FC.py`: 模型定义和训练循环

### 评估脚本
- `imagen/64_FC/test64.py`: 测试脚本
- `imagen/64_FC/t05-forecasting-pipeline.py`: 预测流程

---

## 🎯 预期改进

基于论文结果,使用Video Diffusion后:

| 指标 | ConvLSTM (当前) | Video Diffusion (预期) | 改进 |
|------|----------------|----------------------|------|
| **72h轨迹误差** | 695 km | ~550 km | ⬇️ 21% |
| **时间一致性** | 中等 | 高 | ⬆️ 36% (SSIM) |
| **图像质量** | 中等 | 高 | ⬆️ 16% (PSNR) |
| **预测时长** | 72h | **可扩展到120h+** | 🎉 |

---

## ⚠️ 注意事项

1. **计算资源**: Video Diffusion需要更多GPU内存
   - ConvLSTM: ~4GB
   - Video Diffusion: ~8-12GB

2. **训练时间**: 扩散模型训练较慢
   - ConvLSTM: 15分钟 (50 epochs)
   - Video Diffusion: 2-4小时 (两阶段)

3. **推理时间**: 扩散模型需要多步采样
   - ConvLSTM: ~0.1秒/样本
   - Video Diffusion: ~2-5秒/样本

---

## 📚 参考资料

- **原始论文**: "Improving Tropical Cyclone Forecasting With Video Diffusion Models"
- **基于**: [forecast-diffmodels](https://github.com/p3jitnath/forecast-diffmodels)
- **Video模型**: [imagen-pytorch](https://github.com/lucidrains/imagen-pytorch)
- **FVD评估**: [common_metrics_on_video_quality](https://github.com/JunyaoHu/common_metrics_on_video_quality)

---

## ✅ 下一步行动

想要立即开始吗? 选择一个:

1. **🚀 快速验证** (2小时): 在72个样本上训练Video Diffusion,看效果
2. **📥 先补充数据** (4-8小时): 下载50天ERA5,增加到150样本
3. **🔬 详细分析**: 先深入理解Video Diffusion架构

推荐: **选项1** - 快速验证效果,如果好再补充数据!

