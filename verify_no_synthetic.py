"""Quick verification that code is configured for ERA5-only (no synthetic)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'data'))

# Check generate_data_by_year.py
with open('data/generate_data_by_year.py', 'r', encoding='utf-8', errors='ignore') as f:
    gen_code = f.read()

checks = []
checks.append(('use_era5 = True (hardcoded)', 'use_era5 = True' in gen_code and 'use_era5 = False' not in gen_code))
checks.append(('Skip storms without ERA5', 'Skip this storm if no ERA5' in gen_code))
checks.append(('32x32 crop size', 'image_size=(32, 32)' in gen_code))
checks.append(('.npy format', 'np.save' in gen_code and 'f"{base_name}_past_frames.npy"' in gen_code))
checks.append(('REQUIRED, no synthetic', 'REQUIRED, no synthetic data allowed' in gen_code))

# Check real_data_loader.py
with open('data/real_data_loader.py', 'r', encoding='utf-8', errors='ignore') as f:
    loader_code = f.read()

checks.append(('ONLY use real ERA5', 'ONLY use real ERA5 data, never synthetic' in loader_code))
checks.append(('Returns None if no ERA5', 'return None' in loader_code and 'No ERA5 data available - return None instead of using synthetic data' in loader_code))
checks.append(('xarray merge fixed', "xr.merge(datasets, join='outer')" in loader_code))

print("=== CODE VERIFICATION ===")
print()
all_pass = True
for check, result in checks:
    status = "✓ PASS" if result else "✗ FAIL"
    color = "\033[92m" if result else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} {check}")
    if not result:
        all_pass = False

print()
if all_pass:
    print("✓ All checks passed - Code is configured for ERA5-only (no synthetic)")
else:
    print("✗ Some checks failed - Review code configuration")
