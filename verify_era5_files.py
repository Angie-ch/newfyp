"""
Verify if data/era5 contains real ERA5 reanalysis data
"""
import xarray as xr
from pathlib import Path
import numpy as np

def check_era5_file(file_path):
    """Check if an ERA5 file contains real data"""
    try:
        print(f"  Checking: {file_path.name}")
        print(f"    Size: {file_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Open dataset
        ds = xr.open_dataset(file_path)
        
        # Check variables
        vars_list = list(ds.data_vars.keys())
        print(f"    Variables: {len(vars_list)} ({', '.join(vars_list[:5])}...)")
        print(f"    Dimensions: {dict(ds.dims)}")
        
        # Check if it has data
        if vars_list:
            # Check first variable
            first_var = vars_list[0]
            data = ds[first_var]
            
            # Get actual values (handle different data types)
            if hasattr(data, 'values'):
                values = data.values
                non_zero = (values != 0).sum() / values.size * 100
                min_val = float(np.nanmin(values))
                max_val = float(np.nanmax(values))
                mean_val = float(np.nanmean(values))
                
                print(f"    Sample data ({first_var}):")
                print(f"      Range: [{min_val:.2f}, {max_val:.2f}]")
                print(f"      Mean: {mean_val:.2f}")
                print(f"      Non-zero: {non_zero:.2f}%")
                
                if non_zero > 1.0:
                    print(f"    Status: REAL DATA" + " " * 30 + "[PASS]")
                    return True
                else:
                    print(f"    Status: EMPTY/SUSPICIOUS" + " " * 20 + "[FAIL]")
                    return False
            else:
                print(f"    Status: Cannot read data values")
                return False
        else:
            print(f"    Status: No variables found" + " " * 25 + "[FAIL]")
            return False
            
    except Exception as e:
        print(f"    ERROR: {str(e)[:100]}")
        return False
    finally:
        if 'ds' in locals():
            ds.close()

def main():
    print("="*80)
    print("VERIFYING ERA5 DATA IN data/era5")
    print("="*80)
    print()
    
    era5_base = Path("data/era5")
    if not era5_base.exists():
        print("ERROR: data/era5 directory not found!")
        return False
    
    print("Checking ERA5 directories...")
    print()
    
    all_good = True
    total_files = 0
    
    for year in [2018, 2019, 2020, 2021]:
        year_dir = era5_base / f"ERA5_{year}_26data"
        
        if not year_dir.exists():
            print(f"ERA5_{year}_26data/: NOT FOUND" + " " * 30 + "[FAIL]")
            all_good = False
            continue
        
        # Count files
        files = list(year_dir.glob("era5_pl_*.nc"))
        total_files += len(files)
        
        print(f"ERA5_{year}_26data/:")
        print(f"  Files found: {len(files)}")
        
        if len(files) > 0:
            # Calculate total size
            total_size = sum(f.stat().st_size for f in files) / (1024**3)
            print(f"  Total size: {total_size:.2f} GB")
            
            # Check a sample file
            sample_file = files[0]
            print(f"  Sample file check:")
            is_valid = check_era5_file(sample_file)
            
            if not is_valid:
                all_good = False
            
            # Check a few more files to be sure
            if len(files) > 1:
                print(f"  Checking additional files...")
                for sample_file in files[1:4]:  # Check 3 more
                    is_valid = check_era5_file(sample_file)
                    if not is_valid:
                        all_good = False
        else:
            print(f"  No .nc files found!" + " " * 30 + "[FAIL]")
            all_good = False
        
        print()
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total ERA5 files: {total_files}")
    print(f"Status: {'REAL ERA5 DATA FOUND' if all_good else 'ISSUES DETECTED'}")
    print("="*80)
    
    return all_good

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)










