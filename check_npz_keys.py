import numpy as np
from pathlib import Path

sample_file = Path("D:/typhoon_data_2018_2021_full/train/cases/2018_2018082N04147_w00.npz")
data = np.load(sample_file)

print("Keys in .npz file:")
for k in data.keys():
    print(f"  {k}: {data[k].shape if hasattr(data[k], 'shape') else type(data[k])}")

