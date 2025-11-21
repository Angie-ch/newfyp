"""
Check disk space and permissions for regeneration
"""
import shutil
from pathlib import Path
import os
import sys
from typing import Tuple

def check_disk_space(path: Path, min_gb: float = 10.0) -> Tuple[bool, str]:
    """Check if there's enough disk space"""
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024**3)
        if free_gb >= min_gb:
            return True, f"{free_gb:.2f} GB free (>= {min_gb} GB required)"
        else:
            return False, f"{free_gb:.2f} GB free (< {min_gb} GB required)"
    except Exception as e:
        return False, f"Error checking disk space: {str(e)}"

def check_write_permission(path: Path) -> Tuple[bool, str]:
    """Check if we can write to a directory"""
    try:
        # Try to create a test file
        test_file = path / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        return True, "Write permission OK"
    except PermissionError:
        return False, "Permission denied - cannot write to directory"
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    print("="*80)
    print("DISK SPACE AND PERMISSIONS CHECK")
    print("="*80)
    print()
    
    # Check project directory
    project_dir = Path.cwd()
    print(f"Project directory: {project_dir}")
    
    # Check disk space
    print("\n1. Disk Space Check:")
    has_space, space_msg = check_disk_space(project_dir, min_gb=10.0)
    if has_space:
        print(f"  [OK] {space_msg}")
    else:
        print(f"  [ERROR] {space_msg}")
        print("  Please free up disk space before regenerating.")
        return 1
    
    # Check write permissions for output directory
    print("\n2. Write Permissions Check:")
    output_dir = project_dir / "data" / "processed_temporal_split"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    
    has_write, write_msg = check_write_permission(output_dir.parent)
    if has_write:
        print(f"  [OK] {write_msg}")
        print(f"  Output directory: {output_dir}")
    else:
        print(f"  [ERROR] {write_msg}")
        print(f"  Directory: {output_dir.parent}")
        return 1
    
    # Check ERA5 directory read permission
    print("\n3. ERA5 Data Directory Check:")
    era5_dir = project_dir / "data" / "era5"
    if era5_dir.exists():
        has_read = os.access(era5_dir, os.R_OK)
        if has_read:
            print(f"  [OK] Can read ERA5 directory: {era5_dir}")
        else:
            print(f"  [ERROR] Cannot read ERA5 directory: {era5_dir}")
            return 1
    else:
        print(f"  [WARNING] ERA5 directory not found: {era5_dir}")
        print("  This is OK if ERA5 data is in a different location.")
    
    print()
    print("="*80)
    print("[OK] All checks passed!")
    print("="*80)
    return 0

if __name__ == "__main__":
    from typing import Tuple
    sys.exit(main())

