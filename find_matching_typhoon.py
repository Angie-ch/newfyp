"""Find which 2018 typhoon matches the ERA5 file coverage"""
import pandas as pd
from pathlib import Path

# Load tracks
tracks_file = Path("data/raw/interpolated_typhoon_tracks_2018_2021_6h.csv")
tracks_df = pd.read_csv(tracks_file)
tracks_df['ISO_TIME'] = pd.to_datetime(tracks_df['ISO_TIME'])

# Filter for 2018 February
feb_2018_typhoons = tracks_df[
    (tracks_df['ISO_TIME'] >= '2018-02-01') & 
    (tracks_df['ISO_TIME'] <= '2018-02-28')
]

print("="*80)
print("TYPHOONS IN FEBRUARY 2018")
print("="*80)

if len(feb_2018_typhoons) > 0:
    for typhoon_id in feb_2018_typhoons['typhoon_id'].unique():
        storm = feb_2018_typhoons[feb_2018_typhoons['typhoon_id'] == typhoon_id]
        print(f"\n{typhoon_id}:")
        print(f"  Time: {storm['ISO_TIME'].min()} to {storm['ISO_TIME'].max()}")
        print(f"  Lat: {storm['lat'].min():.2f} to {storm['lat'].max():.2f}")
        print(f"  Lon: {storm['lon'].min():.2f} to {storm['lon'].max():.2f}")
        print(f"  Records: {len(storm)}")
        
        # Check if matches ERA5 coverage (Feb 8-16, lon 110-155, lat 1-16)
        matches_time = (storm['ISO_TIME'].min() <= pd.Timestamp('2018-02-16')) and \
                      (storm['ISO_TIME'].max() >= pd.Timestamp('2018-02-08'))
        matches_space = (storm['lat'].min() >= 1 and storm['lat'].max() <= 16 and \
                        storm['lon'].min() >= 110 and storm['lon'].max() <= 155)
        
        if matches_time and matches_space:
            print(f"  >>> MATCHES ERA5 FILES! <<<")
else:
    print("\nNo typhoons found in February 2018.")
    print("\nChecking early March instead...")
    
    # Try early March (ERA5 has files for March 23)
    march_typhoons = tracks_df[
        (tracks_df['ISO_TIME'] >= '2018-03-01') & 
        (tracks_df['ISO_TIME'] <= '2018-03-31')
    ]
    
    for typhoon_id in march_typhoons['typhoon_id'].unique():
        storm = march_typhoons[march_typhoons['typhoon_id'] == typhoon_id]
        print(f"\n{typhoon_id}:")
        print(f"  Time: {storm['ISO_TIME'].min()} to {storm['ISO_TIME'].max()}")
        print(f"  Lat: {storm['lat'].min():.2f} to {storm['lat'].max():.2f}")
        print(f"  Lon: {storm['lon'].min():.2f} to {storm['lon'].max():.2f}")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
print("Test data generation with ONLY the matching typhoon(s) above.")
print("This will verify the pipeline works before addressing the missing ERA5 data.")

