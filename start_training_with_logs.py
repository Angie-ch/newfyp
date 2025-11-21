"""
Start autoencoder training and show logs in real-time
"""
import subprocess
import sys
from pathlib import Path

def main():
    print("="*80)
    print("STARTING AUTOENCODER TRAINING WITH REAL-TIME LOGS")
    print("="*80)
    print("Using REAL ERA5 data from: data/processed_temporal_split")
    print("="*80)
    print()
    
    # Start training process with real-time output
    cmd = [sys.executable, "train_autoencoder.py", "--config", "configs/autoencoder_config.yaml"]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Print output in real-time
    try:
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        process.terminate()
        process.wait()
    
    process.wait()
    return process.returncode

if __name__ == "__main__":
    sys.exit(main())











