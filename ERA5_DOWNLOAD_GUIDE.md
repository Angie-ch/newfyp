# ERA5数据下载指南

## 📋 概述
需要下载**50个缺失日期**的ERA5数据(约3.7 GB),预计可将样本数从72增加到100-150个。

---

## 🔧 设置步骤

### 1. 安装CDS API
```bash
pip install cdsapi
```

### 2. 注册Copernicus Climate Data Store账号
访问: https://cds.climate.copernicus.eu/
- 点击右上角 "Register" 注册账号
- 验证邮箱

### 3. 获取API密钥
访问: https://cds.climate.copernicus.eu/api-how-to
- 登录后可以看到你的 UID 和 API Key

### 4. 配置API凭据

#### Windows:
在用户目录创建文件 `C:\Users\你的用户名\.cdsapirc`
```
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_UID:YOUR_API_KEY
```

**例如:**
```
url: https://cds.climate.copernicus.eu/api/v2
key: 12345:abcdef12-3456-7890-abcd-ef1234567890
```

#### 注意:
- 文件名是 `.cdsapirc` (前面有个点)
- 替换 `YOUR_UID:YOUR_API_KEY` 为你的真实凭据
- Windows可能需要用命令创建(因为不允许以点开头):
  ```powershell
  New-Item -Path $env:USERPROFILE\.cdsapirc -ItemType File
  ```

---

## 🚀 下载ERA5数据

### 方式1: 自动下载脚本 (推荐)
```bash
# 激活虚拟环境
. .\pytorch_gpu\Scripts\Activate.ps1

# 安装cdsapi
pip install cdsapi

# 运行下载脚本
python download_missing_era5.py
```

### 方式2: 手动下载
如果自动脚本有问题,可以手动从网站下载:
1. 访问: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels
2. 选择日期、变量、气压层
3. 下载并放到 `data/era5/ERA5_YYYY_26data/` 目录

---

## 📊 缺失日期清单

已生成缺失日期列表: `missing_era5_dates.txt`

### 按年份统计:
- **2018年**: 27个日期 (最多)
- **2019年**: 11个日期
- **2020年**: 4个日期
- **2021年**: 8个日期

---

## 💾 下载参数

脚本将下载以下数据:

### 变量 (Variables):
- Geopotential (位势高度)
- Temperature (温度)
- U/V wind components (风速分量)
- Relative humidity (相对湿度)
- Vertical velocity (垂直速度)

### 气压层 (Pressure Levels):
- 200, 300, 500, 700, 850, 925 hPa

### 空间范围:
- 北纬60°到南纬10°
- 东经90°到180° (西太平洋区域)

### 时间分辨率:
- 每小时 (00:00 - 23:00)

---

## ⏱️ 预计时间

- **每个文件下载时间**: 5-15分钟
- **总下载时间**: 约4-12小时 (取决于网速和服务器负载)
- **建议**: 后台运行,或分批下载

---

## ✅ 下载完成后

1. **验证下载**:
   ```bash
   python find_missing_era5_dates.py
   ```
   应该显示: "All required ERA5 files already exist!"

2. **重新生成数据集**:
   ```bash
   python data/generate_data_by_year.py
   ```
   预计生成 100-150 个样本

3. **验证新数据集**:
   ```bash
   python final_dataset_summary.py
   ```

---

## 🔍 常见问题

### Q: 下载速度很慢?
A: CDS服务器在欧洲,亚洲访问可能较慢。建议:
- 选择非高峰时段(欧洲夜间)
- 使用稳定网络连接
- 分批下载

### Q: API请求失败?
A: 检查:
- API凭据是否正确配置
- 是否接受了CDS服务条款 (首次使用需要)
- 网络连接是否正常

### Q: 下载文件损坏?
A: 脚本会自动删除失败的下载,可以重新运行。

### Q: 需要多少磁盘空间?
A: 至少需要 5 GB 可用空间 (3.7 GB 数据 + 缓存)

---

## 📞 获取帮助

- CDS API文档: https://cds.climate.copernicus.eu/api-how-to
- CDS论坛: https://forum.ecmwf.int/
- ERA5数据集说明: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels

---

**祝下载顺利! 🌊⛈️**

