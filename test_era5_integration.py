#!/usr/bin/env python3
"""
Test script for IBTrACS WP + ERA5 integration

Tests:
1. IBTrACS WP data loading
2. ERA5 setup detection
3. Data sample generation
4. Channel verification
"""

import sys
import numpy as np
from pathlib import Path

def test_ibtracs_wp():
    """Test IBTrACS Western Pacific data loading"""
    print("\n" + "="*80)
    print("TEST 1: IBTrACS WP Data Loading")
    print("="*80)
    
    from data.real_data_loader import IBTrACSLoader
    
    loader = IBTrACSLoader(data_dir="data/raw")
    
    try:
        df = loader.load_ibtracs()
        print(f"✓ Loaded {len(df)} IBTrACS records")
        
        # Filter for recent typhoons
        storm_ids = loader.filter_typhoons(df, start_year=2018, end_year=2023)
        print(f"✓ Found {len(storm_ids)} strong typhoons (2018-2023)")
        
        if storm_ids:
            # Get data for one storm
            storm_data = loader.get_storm_data(df, storm_ids[0])
            print(f"\nExample storm: {storm_data['name']} ({storm_data['storm_id']})")
            print(f"  Track points: {len(storm_data['lats'])}")
            print(f"  Max wind: {np.nanmax(storm_data['winds']):.1f} m/s")
            print(f"  Min pressure: {np.nanmin(storm_data['pressures']):.0f} hPa")
            
            return True, storm_ids[0], df
        else:
            print("✗ No storms found")
            return False, None, None
            
    except Exception as e:
        print(f"✗ Error loading IBTrACS: {e}")
        return False, None, None


def test_era5_setup():
    """Test ERA5 setup and availability"""
    print("\n" + "="*80)
    print("TEST 2: ERA5 Setup Detection")
    print("="*80)
    
    try:
        import cdsapi
        print("✓ cdsapi installed")
        
        # Check for API key
        cdsapirc = Path.home() / ".cdsapirc"
        if cdsapirc.exists():
            print(f"✓ Found API key configuration: {cdsapirc}")
            
            # Try to read it
            with open(cdsapirc) as f:
                content = f.read()
                if 'url' in content and 'key' in content:
                    print("✓ API key format looks correct")
                    return True
                else:
                    print("✗ API key file format incorrect")
                    print("  Should contain 'url' and 'key' lines")
                    return False
        else:
            print(f"✗ No API key found at {cdsapirc}")
            print("  See ERA5_SETUP.md for configuration instructions")
            return False
            
    except ImportError:
        print("✗ cdsapi not installed")
        print("  Install with: pip install cdsapi")
        return False


def test_sample_generation(storm_id, df, use_era5=False):
    """Test sample generation with/without ERA5"""
    print("\n" + "="*80)
    print(f"TEST 3: Sample Generation ({'with ERA5' if use_era5 else 'without ERA5'})")
    print("="*80)
    
    from data.real_data_loader import IBTrACSLoader, ERA5Loader
    
    loader = IBTrACSLoader(data_dir="data/raw")
    storm_data = loader.get_storm_data(df, storm_id)
    
    era5_loader = None
    era5_dataset = None
    
    if use_era5:
        era5_loader = ERA5Loader()
        storm_id_clean = storm_id.replace(' ', '_')
        era5_file = era5_loader.data_dir / f"{storm_id_clean}_era5.nc"
        
        if era5_file.exists():
            print(f"✓ Found cached ERA5 data: {era5_file}")
            try:
                era5_dataset = era5_loader.load_era5(era5_file)
                print(f"✓ Loaded ERA5 dataset")
                print(f"  Variables: {list(era5_dataset.data_vars)}")
            except Exception as e:
                print(f"✗ Error loading ERA5: {e}")
                use_era5 = False
        else:
            print(f"✗ No cached ERA5 data found at {era5_file}")
            print("  Run with --download-era5 to download")
            use_era5 = False
    
    try:
        sample = loader.create_training_sample(
            storm_data,
            past_timesteps=12,
            future_timesteps=8,
            era5_dataset=era5_dataset,
            era5_loader=era5_loader,
            use_era5=use_era5
        )
        
        if sample:
            print(f"✓ Generated training sample")
            print(f"  Storm: {sample['storm_name']} ({sample['storm_id']})")
            print(f"  Past frames: {sample['past_frames'].shape}")
            print(f"  Future frames: {sample['future_frames'].shape}")
            print(f"  Past track: {sample['past_track'].shape}")
            print(f"  Future track: {sample['future_track'].shape}")
            
            n_channels = sample['past_frames'].shape[1]
            if use_era5:
                print(f"  Using {n_channels} ERA5 channels")
            else:
                print(f"  Using {n_channels} synthetic channels")
            
            # Check for NaN values
            if np.any(np.isnan(sample['past_frames'])):
                print("  ⚠ Warning: NaN values in frames")
            else:
                print("  ✓ No NaN values")
            
            return True
        else:
            print("✗ Failed to generate sample")
            return False
            
    except Exception as e:
        print(f"✗ Error generating sample: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("IBTRACS WP + ERA5 INTEGRATION TEST")
    print("="*80)
    print("\nData Sources:")
    print("  - IBTrACS: Western Pacific typhoon tracks")
    print("  - ERA5: ECMWF reanalysis meteorological data")
    
    # Test 1: IBTrACS
    success1, storm_id, df = test_ibtracs_wp()
    
    if not success1:
        print("\n" + "!"*80)
        print("IBTrACS loading failed. Cannot continue tests.")
        print("!"*80)
        return False
    
    # Test 2: ERA5 setup
    has_era5 = test_era5_setup()
    
    # Test 3: Sample generation without ERA5
    success3 = test_sample_generation(storm_id, df, use_era5=False)
    
    # Test 4: Sample generation with ERA5 (if available)
    success4 = False
    if has_era5:
        success4 = test_sample_generation(storm_id, df, use_era5=True)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"1. IBTrACS WP loading:     {'✓ PASS' if success1 else '✗ FAIL'}")
    print(f"2. ERA5 setup:             {'✓ PASS' if has_era5 else '✗ FAIL (optional)'}")
    print(f"3. Synthetic samples:      {'✓ PASS' if success3 else '✗ FAIL'}")
    print(f"4. ERA5 samples:           {'✓ PASS' if success4 else '✗ FAIL (optional)' if has_era5 else 'SKIP (ERA5 not set up)'}")
    
    all_critical_pass = success1 and success3
    
    if all_critical_pass:
        print("\n✓ All critical tests passed!")
        print("\nYou can now:")
        print("  - Run demo: python demo_visualizations.py")
        print("  - Train model: python test_with_real_data.py")
        if not has_era5:
            print("\nOptional: Set up ERA5 for real meteorological data")
            print("  See ERA5_SETUP.md for instructions")
    else:
        print("\n✗ Some critical tests failed")
        print("Check error messages above")
    
    return all_critical_pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test IBTrACS WP + ERA5 integration')
    args = parser.parse_args()
    
    success = main()
    sys.exit(0 if success else 1)

