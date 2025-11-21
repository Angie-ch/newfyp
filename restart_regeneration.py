"""
Restart regeneration with better error handling and monitoring
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def main():
    print("="*80)
    print("RESTARTING REGENERATION WITH MONITORING")
    print("="*80)
    print()
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"regeneration_log_{timestamp}.txt"
    
    print(f"Log file: {log_file}")
    print(f"Starting regeneration...")
    print()
    
    # Get the virtual environment Python
    venv_python = Path("pytorch_gpu/Scripts/python.exe")
    if not venv_python.exists():
        print("[ERROR] Virtual environment not found!")
        print(f"  Expected: {venv_python}")
        return 1
    
    # Run the regeneration script
    script_path = Path("data/generate_data_by_year.py")
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return 1
    
    try:
        # Run with output to both console and log file
        with open(log_file, 'w', encoding='utf-8') as log:
            # Write header
            log.write(f"Regeneration started at {datetime.now().isoformat()}\n")
            log.write("="*80 + "\n\n")
            log.flush()
            
            # Run the script
            process = subprocess.Popen(
                [str(venv_python), str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output to both console and log
            for line in process.stdout:
                print(line, end='')
                log.write(line)
                log.flush()
            
            # Wait for completion
            return_code = process.wait()
            
            # Write footer
            log.write("\n" + "="*80 + "\n")
            log.write(f"Regeneration finished at {datetime.now().isoformat()}\n")
            log.write(f"Exit code: {return_code}\n")
            
            if return_code == 0:
                print("\n" + "="*80)
                print("[OK] Regeneration completed successfully!")
                print("="*80)
            else:
                print("\n" + "="*80)
                print(f"[ERROR] Regeneration failed with exit code {return_code}")
                print("="*80)
                print(f"Check log file: {log_file}")
            
            return return_code
            
    except KeyboardInterrupt:
        print("\n[WARNING] Regeneration interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Failed to run regeneration: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())





