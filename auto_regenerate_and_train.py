"""
Automatically regenerate dataset with real ERA5 data and start training
"""
import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime

def check_regeneration_complete():
    """Check if regeneration is complete"""
    info_file = Path("data/processed_temporal_split/dataset_info.json")
    if not info_file.exists():
        return False, None
    
    try:
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        # Check if it's using ERA5 data
        if info.get('meteorological_data') == 'ERA5':
            return True, info
        else:
            return False, info
    except:
        return False, None

def wait_for_regeneration(timeout_minutes=120):
    """Wait for regeneration to complete"""
    print("="*80)
    print("WAITING FOR DATASET REGENERATION TO COMPLETE")
    print("="*80)
    print(f"Timeout: {timeout_minutes} minutes")
    print("Checking every 30 seconds...")
    print()
    
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    while time.time() - start_time < timeout_seconds:
        complete, info = check_regeneration_complete()
        
        if complete:
            print("="*80)
            print("✓ DATASET REGENERATION COMPLETE!")
            print("="*80)
            print(f"  Meteorological data: {info['meteorological_data']}")
            print(f"  Total samples: {info.get('total_samples', 'N/A')}")
            print(f"  Train samples: {info['splits']['train']['n_samples']}")
            print(f"  Val samples: {info['splits']['val']['n_samples']}")
            print(f"  Test samples: {info['splits']['test']['n_samples']}")
            print()
            return True
        
        if info:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Still regenerating... (current: {info.get('meteorological_data', 'unknown')})")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for regeneration to start...")
        
        time.sleep(30)
    
    print("="*80)
    print("⚠ TIMEOUT: Regeneration did not complete within timeout period")
    print("="*80)
    return False

def start_training():
    """Start autoencoder training"""
    print("="*80)
    print("STARTING AUTOENCODER TRAINING")
    print("="*80)
    print("Using REAL ERA5 data from: data/processed_temporal_split")
    print("="*80)
    print()
    
    # Use virtual environment Python if available
    venv_python = Path("pytorch_gpu/Scripts/python.exe")
    if venv_python.exists():
        python_cmd = str(venv_python)
        print(f"Using virtual environment Python: {python_cmd}")
    else:
        python_cmd = sys.executable
        print(f"Using system Python: {python_cmd}")
    
    cmd = [
        python_cmd,
        "train_autoencoder.py",
        "--config",
        "configs/autoencoder_config.yaml"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    print("Training output will appear below:")
    print("="*80)
    print()
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            sys.stdout.flush()
        
        process.wait()
        return process.returncode
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        if 'process' in locals():
            process.terminate()
            process.wait()
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1

def main():
    """Main function"""
    print("="*80)
    print("AUTOMATED REGENERATION AND TRAINING")
    print("="*80)
    print()
    
    # Step 1: Check if regeneration is already complete
    complete, info = check_regeneration_complete()
    
    if complete:
        print("✓ Dataset already regenerated with REAL ERA5 data!")
        print(f"  Total samples: {info.get('total_samples', 'N/A')}")
        print()
        print("Starting training immediately...")
        print()
    else:
        # Step 2: Wait for regeneration
        if not wait_for_regeneration():
            print("ERROR: Regeneration did not complete. Please check manually.")
            return 1
    
    # Step 3: Start training
    return start_training()

if __name__ == "__main__":
    sys.exit(main())











