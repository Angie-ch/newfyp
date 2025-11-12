# 按年份划分数据集 - 避免数据泄漏

## ⚠️ 问题说明

之前的数据划分方法会导致**严重的数据泄漏问题**：

```python
# ❌ 错误的方法 - 按样本随机划分
所有样本 = [
    舒力基_样本1, 舒力基_样本2, 舒力基_样本3,
    查帕卡_样本1, 查帕卡_样本2, ...
]
训练集 = 样本[0:70%]    # 可能包含舒力基_样本1
测试集 = 样本[85%:]    # 可能包含舒力基_样本2
```

**问题**：同一个台风的不同时间窗口会同时出现在训练集和测试集中！

模型在训练时已经见过这个台风的行为模式，测试时只是预测同一个台风的不同时间段。这会导致：
- 虚高的性能指标
- 无法评估模型对新台风的泛化能力
- 不符合真实预测场景

## ✅ 正确的方法 - 按年份划分

### LT3P 论文和其他研究的标准做法

根据 [LT3P (ICLR 2024)](https://github.com/iclr2024submit/LT3P) 和台风预测领域的最佳实践，应该使用**时间顺序划分**：

```python
# ✅ 正确的方法 - 按年份划分
训练集 = 2021-2022年的所有台风
验证集 = 2023年的所有台风
测试集 = 2024年的所有台风
```

**优点**：
- ✅ 避免数据泄漏
- ✅ 模拟真实预测场景（用历史预测未来）
- ✅ 测试模型对新台风的泛化能力
- ✅ 符合时间序列预测的标准实践

## 🚀 使用方法

### 1. 生成按年份划分的数据集

```bash
cd /Volumes/data/fyp/typhoon_prediction/data
python generate_data_by_year.py
```

这会创建以下目录结构：

```
data/processed_temporal_split/
├── train/
│   └── cases/
│       ├── 2021_2021WP01_s0.npz
│       ├── 2021_2021WP01_s1.npz
│       ├── 2022_2022WP05_s0.npz
│       └── ...
├── val/
│   └── cases/
│       ├── 2023_2023WP02_s0.npz
│       └── ...
├── test/
│   └── cases/
│       ├── 2024_2024WP03_s0.npz
│       └── ...
└── dataset_metadata.pkl
```

### 2. 在训练代码中使用

```python
from torch.utils.data import DataLoader
from typhoon_prediction.data.datasets.typhoon_dataset import TyphoonDataset

# 创建数据集（使用时间划分）
train_dataset = TyphoonDataset(
    data_dir='data/processed_temporal_split',
    split='train',
    use_temporal_split=True  # ← 重要！使用时间划分
)

val_dataset = TyphoonDataset(
    data_dir='data/processed_temporal_split',
    split='val',
    use_temporal_split=True
)

test_dataset = TyphoonDataset(
    data_dir='data/processed_temporal_split',
    split='test',
    use_temporal_split=True
)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
```

### 3. 配置参数

可以在 `generate_data_by_year.py` 中修改配置：

```python
# 配置参数
START_YEAR = 2021          # 起始年份
END_YEAR = 2024            # 结束年份
SAMPLES_PER_STORM = 2      # 每个台风生成的样本数
PAST_TIMESTEPS = 8         # 过去时间步（48小时）
FUTURE_TIMESTEPS = 12      # 未来时间步（72小时）

# 划分配置
TRAIN_YEARS = [2021, 2022] # 训练年份
VAL_YEARS = [2023]         # 验证年份
TEST_YEARS = [2024]        # 测试年份
```

## 📊 数据统计示例

运行脚本后会显示类似输出：

```
================================================================================
DATASET SUMMARY
================================================================================
Training samples:   84 (years [2021, 2022])
Validation samples: 38 (years [2023])
Test samples:       42 (years [2024])

Total: 164 samples
```

## 🔬 为什么这很重要？

### 场景对比

**错误的随机划分**：
- 训练：舒力基台风 2021年8月1-5日
- 测试：舒力基台风 2021年8月6-10日
- 结果：模型已经见过舒力基的行为模式 → 性能虚高

**正确的时间划分**：
- 训练：2021-2022年的所有台风
- 测试：2024年的全新台风（模型从未见过）
- 结果：真实评估模型的泛化能力

### 研究诚信

使用正确的数据划分方法是：
- ✅ 学术诚信的要求
- ✅ 公平比较的基础
- ✅ 真实场景模拟

## 📚 参考文献

1. **LT3P (ICLR 2024)**:
   - Paper: "Long-Term Typhoon Trajectory Prediction: A Physics-Conditioned Approach Without Reanalysis Data"
   - GitHub: https://github.com/iclr2024submit/LT3P
   - 使用时间顺序划分，避免数据泄漏

2. **时间序列预测最佳实践**:
   - 使用过去的数据预测未来
   - 避免未来信息泄漏
   - 按时间或ID划分，不按样本随机划分

## ⚠️ 不要使用旧方法

如果使用 `use_temporal_split=False`，会触发警告：

```python
# ❌ 不推荐
dataset = TyphoonDataset(
    data_dir='data/processed',
    split='train',
    use_temporal_split=False  # 会导致数据泄漏！
)
# 警告：Using random split - this can cause data leakage! Use temporal split instead.
```

## 🎯 下一步

1. 运行 `python generate_data_by_year.py` 生成数据
2. 使用新的数据目录训练模型
3. 观察性能指标的真实变化（可能会比之前低，这是正常的！）
4. 现在的结果才是模型真正的泛化能力

---

**记住**：更低的性能指标 + 正确的评估方法 > 虚高的性能指标 + 错误的评估方法！

