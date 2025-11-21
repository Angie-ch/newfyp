# Checkpoint Validation Guide

## 概述

`check_best_model.py` 是一个全面的模型检查点验证脚本，用于验证训练好的模型文件的完整性和正确性。

## 功能特性

✅ **多模型支持**: 支持检查所有训练模型类型
- `joint_autoencoder` - 联合自编码器模型
- `autoencoder` - 空间自编码器模型  
- `diffusion` - 扩散模型
- `joint_diffusion` - 联合扩散模型

✅ **全面验证**:
- 文件存在性和大小检查
- Checkpoint 结构验证（必需键检查）
- 模型状态字典验证
- 模型组件完整性检查
- 参数统计信息
- 模型加载测试

✅ **详细报告**: 提供清晰的验证报告，包括错误、警告和信息

## 使用方法

### 基本用法

```bash
# 检查默认模型 (joint_autoencoder) 的 best.pth
python check_best_model.py

# 检查特定模型
python check_best_model.py --model autoencoder

# 检查所有 checkpoint 文件（不仅仅是 best.pth）
python check_best_model.py --check-all

# 列出所有可用的 checkpoints
python check_best_model.py --list
```

### 检查所有模型

```bash
# 检查所有模型类型
for model in joint_autoencoder autoencoder diffusion joint_diffusion; do
    python check_best_model.py --model $model
done
```

## 输出说明

脚本会提供以下信息：

1. **验证信息** (✓): 成功的验证步骤
2. **警告** (⚠️): 需要注意但不致命的问题
3. **错误** (❌): 导致验证失败的问题
4. **Checkpoint 信息**: Epoch、验证损失、训练配置
5. **模型统计**: 参数数量、模型大小、层数

## 关于 Checkpoints 目录

⚠️ **重要**: `checkpoints/` 目录在 `.gitignore` 中，因此不会被 Git 跟踪。

### 为什么 checkpoints 不在仓库中？

- Checkpoint 文件通常很大（几十到几百 MB）
- GitHub 有文件大小限制（100 MB）
- 训练过程中会生成大量 checkpoint 文件

### 如何获取 Checkpoints？

#### 方法 1: 本地训练生成

```bash
# 训练模型以生成 checkpoints
python train_joint_pipeline.py --config configs/joint_autoencoder_config.yaml
```

#### 方法 2: 从云存储下载

如果 checkpoints 存储在云存储（如 Google Drive, AWS S3, 等），可以下载到本地：

```bash
# 示例：从云存储下载（根据实际情况调整）
# gsutil -m cp -r gs://your-bucket/checkpoints ./checkpoints
# aws s3 sync s3://your-bucket/checkpoints ./checkpoints
```

#### 方法 3: 从其他位置复制

如果 checkpoints 在其他位置，可以复制到项目目录：

```bash
cp -r /path/to/checkpoints ./checkpoints
```

#### 方法 4: 使用 Git LFS（如果配置）

如果项目使用 Git LFS 管理大文件：

```bash
git lfs pull
```

## 验证标准

脚本会检查以下内容：

### 必需检查项

- ✅ 文件存在
- ✅ 文件大小合理（> 0.1 MB）
- ✅ 可以成功加载 PyTorch checkpoint
- ✅ 包含所有必需键：
  - `epoch`
  - `model_state_dict`
  - `optimizer_state_dict`
  - `scheduler_state_dict`
  - `val_loss`
  - `config`
- ✅ 模型状态字典非空
- ✅ 包含预期的模型组件

### 可选检查项

- ⚠️ 模型可以成功实例化和加载
- ⚠️ 文件大小合理（不会过大）
- ⚠️ 所有预期组件都存在

## 故障排除

### 问题: "Checkpoint directory does not exist"

**解决方案**:
1. 确认训练已完成
2. 检查 checkpoint 保存路径是否正确
3. 查看训练日志确认保存位置

### 问题: "Failed to load checkpoint"

**可能原因**:
- 文件损坏
- PyTorch 版本不兼容
- 文件格式错误

**解决方案**:
- 重新训练模型
- 检查 PyTorch 版本兼容性
- 验证文件完整性

### 问题: "Missing required keys"

**可能原因**:
- Checkpoint 格式不正确
- 使用了旧版本的保存格式

**解决方案**:
- 使用最新版本的训练脚本重新保存
- 检查训练器代码中的保存逻辑

### 问题: "Cannot load model state dict"

**可能原因**:
- 模型架构不匹配
- 参数名称不匹配
- 模型配置不一致

**解决方案**:
- 确认使用相同的模型配置
- 检查模型定义是否更改
- 使用 `strict=False` 模式加载（如果兼容）

## 示例输出

```
================================================================================
VALIDATION REPORT: JOINT_AUTOENCODER
================================================================================

File: checkpoints/joint_autoencoder/best.pth
Absolute path: /path/to/project/checkpoints/joint_autoencoder/best.pth

--------------------------------------------------------------------------------
VALIDATION INFO
--------------------------------------------------------------------------------
  ✓ File found: checkpoints/joint_autoencoder/best.pth
  ✓ File size: 45.23 MB
  ✓ Checkpoint loaded successfully
  ✓ All required checkpoint keys present
  ✓ Model state dict contains 234 parameters
  ✓ Found all expected components: encoder, decoder, ibtracs_embedder
  ✓ Model state dict can be loaded (strict=False)

--------------------------------------------------------------------------------
CHECKPOINT INFORMATION
--------------------------------------------------------------------------------
  Epoch: 3
  Validation Loss: 950.660000

  Training Config:
    Learning Rate: 0.0001
    Batch Size: 8
    Weight Decay: 0.0001
    Epochs: 10

--------------------------------------------------------------------------------
MODEL STATISTICS
--------------------------------------------------------------------------------
  Total Parameters: 12,345,678
  Model Size: 47.12 MB
  Number of Layers: 234

================================================================================
✅ VALIDATION PASSED - Checkpoint appears to be OK!
================================================================================
```

## 集成到 CI/CD

可以将此脚本集成到 CI/CD 流程中：

```yaml
# .github/workflows/validate-checkpoints.yml
name: Validate Checkpoints

on:
  workflow_dispatch:

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install torch
      - name: Validate checkpoints
        run: python check_best_model.py --check-all
```

## 相关文件

- `training/trainers/joint_autoencoder_trainer.py` - 训练器代码
- `models/autoencoder/joint_autoencoder.py` - 模型定义
- `.gitignore` - Git 忽略规则

## 贡献

如果发现验证脚本的问题或有改进建议，请提交 Issue 或 Pull Request。

