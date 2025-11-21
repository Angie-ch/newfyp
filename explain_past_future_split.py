"""
演示如何分离过去帧和未来帧

示例: 假设一个台风有42个时间步 (6小时间隔)
配置:
  - PAST_TIMESTEPS = 2 (过去2步 = 12小时)
  - FUTURE_TIMESTEPS = 4 (未来4步 = 24小时)
  - STRIDE = 1 (每次滑动1步)
  - SKIP_EARLY_TIMESTEPS = 4 (跳过前4步确保有ERA5历史数据)
"""

# 模拟台风时间序列
total_timesteps = 42
PAST_TIMESTEPS = 2
FUTURE_TIMESTEPS = 4
STRIDE = 1
SKIP_EARLY_TIMESTEPS = 4

print("="*80)
print("过去帧和未来帧分离示例")
print("="*80)

print(f"\n台风配置:")
print(f"  总时间步: {total_timesteps}")
print(f"  过去时间步: {PAST_TIMESTEPS} (12小时历史)")
print(f"  未来时间步: {FUTURE_TIMESTEPS} (24小时预测)")
print(f"  滑动窗口步长: {STRIDE}")
print(f"  跳过前: {SKIP_EARLY_TIMESTEPS}步")

print(f"\n生成样本:")
print("-"*80)

sample_count = 0
for start_idx in range(SKIP_EARLY_TIMESTEPS, 
                       total_timesteps - PAST_TIMESTEPS - FUTURE_TIMESTEPS + 1, 
                       STRIDE):
    
    # 计算索引范围
    past_start = start_idx
    past_end = start_idx + PAST_TIMESTEPS
    future_start = past_end
    future_end = future_start + FUTURE_TIMESTEPS
    
    sample_count += 1
    
    if sample_count <= 5 or sample_count > 30:  # 只显示前5个和最后几个
        print(f"\n样本 #{sample_count}:")
        print(f"  start_idx = {start_idx}")
        print(f"  过去时间步索引: [{past_start}:{past_end}]  (时间步 {past_start}, {past_start+1})")
        print(f"  未来时间步索引: [{future_start}:{future_end}]  (时间步 {future_start}, {future_start+1}, {future_start+2}, {future_start+3})")
        print(f"  总跨度: 时间步 {past_start} 到 {future_end-1} (共{future_end-past_start}步 = {(future_end-past_start)*6}小时)")
    elif sample_count == 6:
        print(f"\n  ... (中间样本省略) ...")

print(f"\n总共生成样本: {sample_count}")

print("\n" + "="*80)
print("实际代码中的实现")
print("="*80)

print("""
在 data/real_data_loader.py 的 create_training_sample() 函数中:

1. 定义时间索引:
   ```python
   past_start = start_idx
   past_end = start_idx + past_timesteps
   future_start = past_end
   future_end = future_start + future_timesteps
   ```

2. 切片台风数据:
   ```python
   # 过去的轨迹和强度
   past_lats = storm_data['lats'][past_start:past_end]
   past_lons = storm_data['lons'][past_start:past_end]
   past_times = storm_data['times'][past_start:past_end]
   
   # 未来的轨迹和强度
   future_lats = storm_data['lats'][future_start:future_end]
   future_lons = storm_data['lons'][future_start:future_end]
   future_times = storm_data['times'][future_start:future_end]
   ```

3. 提取ERA5气象场:
   ```python
   # 提取过去的ERA5帧
   past_frames = era5_loader.extract_frames_at_times(
       center_lons=past_lons,
       center_lats=past_lats,
       times=past_times,
       crop_size=64
   )
   # 形状: (2, 24, 64, 64)
   #       ↑  ↑   ↑   ↑
   #       │  │   │   └─ 宽度 64像素
   #       │  │   └───── 高度 64像素
   #       │  └────────── 24个气象通道
   #       └────────────── 2个过去时间步
   
   # 提取未来的ERA5帧
   future_frames = era5_loader.extract_frames_at_times(
       center_lons=future_lons,
       center_lats=future_lats,
       times=future_times,
       crop_size=64
   )
   # 形状: (4, 24, 64, 64)
   #       ↑  ↑   ↑   ↑
   #       │  │   │   └─ 宽度 64像素
   #       │  │   └───── 高度 64像素
   #       │  └────────── 24个气象通道
   #       └────────────── 4个未来时间步
   ```

4. 保存到.npz文件:
   ```python
   np.savez(
       output_file,
       past_frames=past_frames,      # (2, 24, 64, 64)
       future_frames=future_frames,  # (4, 24, 64, 64)
       track_past=track_past,        # (2, 2) - [lat, lon]
       track_future=track_future,    # (4, 2) - [lat, lon]
       intensity_past=intensity_past,
       intensity_future=intensity_future
   )
   ```
""")

print("\n" + "="*80)
print("关键点")
print("="*80)
print("""
1. **时间连续性**: 
   - 过去帧结束时间 = 未来帧开始时间
   - 没有时间间隙
   - 例如: 过去[0,1], 未来[2,3,4,5]

2. **滑动窗口**: 
   - STRIDE=1: 每次向前移动1个时间步
   - 创建重叠样本增加训练数据量
   - 例如: 样本1[0:6], 样本2[1:7], 样本3[2:8]...

3. **空间对齐**:
   - 每个时间步的ERA5帧以台风中心为中心裁剪
   - 64×64像素对应~16°×16°区域
   - 跟随台风移动

4. **用途**:
   - 模型输入: past_frames (历史气象场)
   - 模型输出: future_frames (预测目标)
   - 或者用于轨迹预测: track_past → track_future
""")

print("="*80)

