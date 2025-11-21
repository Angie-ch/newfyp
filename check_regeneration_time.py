"""Check regeneration progress and estimate time"""
import pandas as pd
from datetime import datetime
from pathlib import Path

# Check storm count
csv_file = Path("data/raw/interpolated_typhoon_tracks_2018_2021.csv")
if csv_file.exists():
    df = pd.read_csv(csv_file)
    unique_storms = df['typhoon_id'].nunique()
    total_records = len(df)
    print(f"Data Statistics:")
    print(f"  Unique storms: {unique_storms}")
    print(f"  Total records: {total_records}")
    print(f"  Avg records per storm: {total_records / unique_storms:.1f}")
else:
    print("CSV file not found")
    unique_storms = None

# Estimate time
if unique_storms:
    print(f"\nTime Estimation:")
    print(f"  Storms to process: ~{unique_storms}")
    print(f"  ERA5 loading: ~2-5 seconds per storm × {unique_storms} = {2*unique_storms}-{5*unique_storms} seconds")
    print(f"  Sample generation: ~10-30 minutes")
    print(f"  Saving: ~5-15 minutes")
    
    total_minutes = (2*unique_storms/60) + 10 + 5
    total_max = (5*unique_storms/60) + 30 + 15
    print(f"\n  Estimated total: {total_minutes:.0f}-{total_max:.0f} minutes")
    print(f"  (Best case: ~{total_minutes:.0f} min, Worst case: ~{total_max:.0f} min)")





