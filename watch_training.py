"""
Watch training logs in real-time
"""
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def get_latest_tensorboard_log(log_dir):
    """Get the most recent TensorBoard log file"""
    log_path = Path(log_dir)
    if not log_path.exists():
        return None
    
    # Find most recent events file
    event_files = list(log_path.glob("events.out.tfevents.*"))
    if not event_files:
        return None
    
    # Sort by modification time
    latest = max(event_files, key=lambda p: p.stat().st_mtime)
    return latest

def monitor_training():
    """Monitor training progress"""
    log_dir = Path("logs/autoencoder")
    
    print("="*80)
    print("MONITORING AUTOENCODER TRAINING")
    print("="*80)
    print(f"Log directory: {log_dir.absolute()}")
    print(f"Press Ctrl+C to stop monitoring\n")
    
    last_size = {}
    last_epoch = None
    
    try:
        while True:
            # Check for latest TensorBoard log
            latest_log = get_latest_tensorboard_log(log_dir)
            if latest_log:
                current_size = latest_log.stat().st_size
                if latest_log not in last_size or current_size > last_size[latest_log]:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training active - log file updated")
                    last_size[latest_log] = current_size
            
            # Check checkpoint directory for progress
            checkpoint_dir = Path("checkpoints/autoencoder")
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("*.pth"))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                    checkpoint_time = datetime.fromtimestamp(latest_checkpoint.stat().st_mtime)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Latest checkpoint: {latest_checkpoint.name} (saved at {checkpoint_time.strftime('%H:%M:%S')})")
            
            time.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    monitor_training()











