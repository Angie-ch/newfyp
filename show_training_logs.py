"""
Show training logs in real-time
"""
import time
import subprocess
import sys
from pathlib import Path

def main():
    print("="*80)
    print("AUTOENCODER TRAINING - REAL-TIME LOG VIEWER")
    print("="*80)
    print("Using REAL ERA5 data from: data/processed_temporal_split")
    print("="*80)
    print()
    
    # Activate virtual environment and run training
    venv_python = Path("pytorch_gpu/Scripts/python.exe")
    if not venv_python.exists():
        print("ERROR: Virtual environment not found at pytorch_gpu/Scripts/python.exe")
        print("Please activate the virtual environment first or install dependencies.")
        return 1
    
    print(f"Using Python: {venv_python}")
    print("Starting training...")
    print("="*80)
    print()
    
    # Run training with real-time output
    cmd = [
        str(venv_python),
        "train_autoencoder.py",
        "--config",
        "configs/autoencoder_config.yaml"
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output line by line in real-time
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

if __name__ == "__main__":
    sys.exit(main())











