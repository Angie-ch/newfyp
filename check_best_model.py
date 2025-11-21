"""
Comprehensive Checkpoint Validation Script

This script validates model checkpoints for all trained models in the project.
It checks multiple checkpoint locations and provides detailed validation reports.

Usage:
    python check_best_model.py [--model MODEL_NAME] [--check-all]
    
Examples:
    python check_best_model.py                    # Check joint_autoencoder (default)
    python check_best_model.py --model autoencoder # Check autoencoder
    python check_best_model.py --check-all         # Check all models
"""

import torch
from pathlib import Path
import sys
import argparse
from typing import Dict, List, Tuple, Optional
import json

# Define all possible checkpoint locations
# Note: Check both 'checkpoint' (singular) and 'checkpoints' (plural) directories
CHECKPOINT_LOCATIONS = {
    'joint_autoencoder': ['checkpoint/joint_autoencoder', 'checkpoints/joint_autoencoder'],
    'autoencoder': ['checkpoint/autoencoder', 'checkpoints/autoencoder'],
    'diffusion': ['checkpoint/diffusion', 'checkpoints/diffusion'],
    'joint_diffusion': ['checkpoint/joint_diffusion', 'checkpoints/joint_diffusion'],
}

# Expected model components for each model type
EXPECTED_COMPONENTS = {
    'joint_autoencoder': ['encoder', 'decoder', 'ibtracs_embedder'],
    'autoencoder': ['encoder', 'decoder'],
    'diffusion': ['model', 'diffusion'],
    'joint_diffusion': ['model', 'diffusion', 'autoencoder'],
}


class CheckpointValidator:
    """Validates PyTorch model checkpoints"""
    
    def __init__(self, checkpoint_dir: Path, model_name: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.errors = []
        self.warnings = []
        self.info = []
        
    def validate_file_exists(self, filepath: Path) -> bool:
        """Check if checkpoint file exists"""
        if not filepath.exists():
            self.errors.append(f"❌ File not found: {filepath}")
            return False
        self.info.append(f"✓ File found: {filepath}")
        return True
    
    def validate_file_size(self, filepath: Path) -> bool:
        """Check if file size is reasonable"""
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        self.info.append(f"✓ File size: {file_size_mb:.2f} MB")
        
        if file_size_mb < 0.1:
            self.errors.append(f"❌ File size suspiciously small: {file_size_mb:.2f} MB")
            return False
        elif file_size_mb > 5000:
            self.warnings.append(f"⚠️  File size very large: {file_size_mb:.2f} MB")
        
        return True
    
    def validate_checkpoint_structure(self, ckpt: Dict) -> bool:
        """Validate checkpoint contains required keys"""
        required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict', 
                        'scheduler_state_dict', 'val_loss', 'config']
        missing_keys = [key for key in required_keys if key not in ckpt]
        
        if missing_keys:
            self.errors.append(f"❌ Missing required keys: {missing_keys}")
            return False
        
        self.info.append("✓ All required checkpoint keys present")
        return True
    
    def validate_model_state_dict(self, model_state: Dict) -> bool:
        """Validate model state dictionary"""
        if not isinstance(model_state, dict):
            self.errors.append("❌ model_state_dict is not a dictionary")
            return False
        
        if len(model_state) == 0:
            self.errors.append("❌ model_state_dict is empty")
            return False
        
        self.info.append(f"✓ Model state dict contains {len(model_state)} parameters")
        
        # Check for expected components
        if self.model_name in EXPECTED_COMPONENTS:
            expected = EXPECTED_COMPONENTS[self.model_name]
            found_components = []
            for key in model_state.keys():
                for comp in expected:
                    if comp in key and comp not in found_components:
                        found_components.append(comp)
            
            missing_components = set(expected) - set(found_components)
            if missing_components:
                self.warnings.append(f"⚠️  Missing expected components: {missing_components}")
            else:
                self.info.append(f"✓ Found all expected components: {', '.join(found_components)}")
        
        return True
    
    def calculate_model_stats(self, model_state: Dict) -> Dict:
        """Calculate model parameter statistics"""
        total_params = 0
        param_info = {}
        
        for key, value in model_state.items():
            if isinstance(value, torch.Tensor):
                num_params = value.numel()
                total_params += num_params
                param_info[key] = {
                    'shape': list(value.shape),
                    'params': num_params,
                    'size_mb': num_params * 4 / (1024 * 1024)  # Assuming float32
                }
        
        return {
            'total_params': total_params,
            'total_size_mb': total_params * 4 / (1024 * 1024),
            'num_layers': len(param_info),
            'param_info': param_info
        }
    
    def validate_model_loading(self, model_state: Dict) -> bool:
        """Try to load model state dict into actual model"""
        try:
            # Try to import the appropriate model
            if self.model_name == 'joint_autoencoder':
                from models.autoencoder.joint_autoencoder import JointAutoencoder
                
                # Get config for model initialization
                model_config = {}
                if hasattr(self, 'config') and isinstance(self.config, dict):
                    model_config = self.config.get('model', {})
                
                era5_channels = model_config.get('era5_channels', 40)
                latent_channels = model_config.get('latent_channels', 8)
                
                model = JointAutoencoder(
                    era5_channels=era5_channels,
                    latent_channels=latent_channels
                )
                
            elif self.model_name == 'autoencoder':
                from models.autoencoder import SpatialAutoencoder
                model = SpatialAutoencoder(
                    in_channels=40,
                    latent_channels=8,
                    hidden_dims=[64, 128, 256]
                )
            else:
                # For diffusion models, skip loading test
                self.warnings.append("⚠️  Model loading test skipped (diffusion models)")
                return True
            
            # Try to load state dict
            try:
                model.load_state_dict(model_state, strict=False)
                self.info.append("✓ Model state dict can be loaded (strict=False)")
                return True
            except Exception as e:
                try:
                    model.load_state_dict(model_state, strict=True)
                    self.info.append("✓ Model state dict can be loaded (strict=True)")
                    return True
                except Exception as e2:
                    self.errors.append(f"❌ Cannot load model state dict: {e2}")
                    return False
                    
        except ImportError as e:
            self.warnings.append(f"⚠️  Could not import model class: {e}")
            return True  # Not a critical error
        except Exception as e:
            self.warnings.append(f"⚠️  Could not test model instantiation: {e}")
            return True  # Not a critical error
    
    def validate_checkpoint(self, filepath: Path) -> Tuple[bool, Dict]:
        """Validate a single checkpoint file"""
        self.errors = []
        self.warnings = []
        self.info = []
        
        # Check file exists
        if not self.validate_file_exists(filepath):
            return False, {}
        
        # Check file size
        if not self.validate_file_size(filepath):
            return False, {}
        
        # Try to load checkpoint
        try:
            ckpt = torch.load(filepath, map_location='cpu', weights_only=False)
            self.info.append("✓ Checkpoint loaded successfully")
        except Exception as e:
            self.errors.append(f"❌ Failed to load checkpoint: {e}")
            return False, {}
        
        # Store config for later use
        if 'config' in ckpt:
            self.config = ckpt['config']
        
        # Validate structure
        if not self.validate_checkpoint_structure(ckpt):
            return False, ckpt
        
        # Validate model state dict
        model_state = ckpt['model_state_dict']
        if not self.validate_model_state_dict(model_state):
            return False, ckpt
        
        # Calculate statistics
        stats = self.calculate_model_stats(model_state)
        
        # Try to load into model
        self.validate_model_loading(model_state)
        
        # Compile results
        result = {
            'valid': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info,
            'stats': stats,
            'checkpoint_info': {
                'epoch': ckpt.get('epoch', 'N/A'),
                'val_loss': ckpt.get('val_loss', 'N/A'),
                'config': ckpt.get('config', {})
            }
        }
        
        return result['valid'], result


<<<<<<< HEAD










=======
def find_all_checkpoints(checkpoint_dir: Path) -> List[Path]:
    """Find all checkpoint files in a directory"""
    checkpoints = []
    
    if not checkpoint_dir.exists():
        return checkpoints
    
    # Look for best.pth
    best_path = checkpoint_dir / 'best.pth'
    if best_path.exists():
        checkpoints.append(best_path)
    
    # Look for epoch checkpoints
    for filepath in checkpoint_dir.glob('checkpoint_epoch_*.pth'):
        checkpoints.append(filepath)
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return checkpoints


def print_validation_report(model_name: str, results: Dict, filepath: Path):
    """Print a formatted validation report"""
    print("\n" + "=" * 80)
    print(f"VALIDATION REPORT: {model_name.upper()}")
    print("=" * 80)
    print(f"\nFile: {filepath}")
    print(f"Absolute path: {filepath.absolute()}")
    
    # Print info messages
    if results['info']:
        print("\n" + "-" * 80)
        print("VALIDATION INFO")
        print("-" * 80)
        for msg in results['info']:
            print(f"  {msg}")
    
    # Print warnings
    if results['warnings']:
        print("\n" + "-" * 80)
        print("WARNINGS")
        print("-" * 80)
        for msg in results['warnings']:
            print(f"  {msg}")
    
    # Print errors
    if results['errors']:
        print("\n" + "-" * 80)
        print("ERRORS")
        print("-" * 80)
        for msg in results['errors']:
            print(f"  {msg}")
    
    # Print checkpoint information
    if 'checkpoint_info' in results:
        print("\n" + "-" * 80)
        print("CHECKPOINT INFORMATION")
        print("-" * 80)
        info = results['checkpoint_info']
        print(f"  Epoch: {info['epoch']}")
        if isinstance(info['val_loss'], (int, float)):
            print(f"  Validation Loss: {info['val_loss']:.6f}")
        else:
            print(f"  Validation Loss: {info['val_loss']}")
        
        if isinstance(info['config'], dict):
            print(f"\n  Training Config:")
            config = info['config']
            print(f"    Learning Rate: {config.get('learning_rate', 'N/A')}")
            print(f"    Batch Size: {config.get('batch_size', 'N/A')}")
            print(f"    Weight Decay: {config.get('weight_decay', 'N/A')}")
            print(f"    Epochs: {config.get('epochs', 'N/A')}")
    
    # Print model statistics
    if 'stats' in results:
        print("\n" + "-" * 80)
        print("MODEL STATISTICS")
        print("-" * 80)
        stats = results['stats']
        print(f"  Total Parameters: {stats['total_params']:,}")
        print(f"  Model Size: {stats['total_size_mb']:.2f} MB")
        print(f"  Number of Layers: {stats['num_layers']}")
    
    # Print final status
    print("\n" + "=" * 80)
    if results['valid']:
        print("✅ VALIDATION PASSED - Checkpoint appears to be OK!")
    else:
        print("❌ VALIDATION FAILED - Checkpoint has errors!")
    print("=" * 80)


def check_model_checkpoints(model_name: str, check_all_files: bool = False):
    """Check checkpoints for a specific model"""
    if model_name not in CHECKPOINT_LOCATIONS:
        print(f"❌ Unknown model name: {model_name}")
        print(f"Available models: {', '.join(CHECKPOINT_LOCATIONS.keys())}")
        return False
    
    # Try to find the checkpoint directory (check both singular and plural)
    possible_dirs = CHECKPOINT_LOCATIONS[model_name]
    checkpoint_dir = None
    
    for dir_path in possible_dirs:
        path = Path(dir_path)
        if path.exists():
            checkpoint_dir = path
            break
    
    print("\n" + "=" * 80)
    print(f"CHECKING CHECKPOINTS FOR: {model_name.upper()}")
    print("=" * 80)
    
    if checkpoint_dir is None:
        print(f"\n❌ Checkpoint directory does not exist in any of these locations:")
        for dir_path in possible_dirs:
            print(f"  - {dir_path}")
        print("\nPossible solutions:")
        print("  1. Run training to generate checkpoints")
        print("  2. Download checkpoints from cloud storage (if available)")
        print("  3. Copy checkpoints from another location")
        print("  4. Check if checkpoints are in a different location")
        return False
    
    print(f"\nCheckpoint directory: {checkpoint_dir}")
    print(f"Absolute path: {checkpoint_dir.absolute()}")
    
    # Find all checkpoints
    all_checkpoints = find_all_checkpoints(checkpoint_dir)
    
    if not all_checkpoints:
        print(f"\n❌ No checkpoint files found in {checkpoint_dir}")
        print("\nPossible solutions:")
        print("  1. Run training to generate checkpoints")
        print("  2. Check if checkpoints are saved with different naming")
        return False
    
    print(f"\nFound {len(all_checkpoints)} checkpoint file(s):")
    for ckpt in all_checkpoints:
        print(f"  - {ckpt.name}")
    
    # Validate checkpoints
    validator = CheckpointValidator(checkpoint_dir, model_name)
    
    if check_all_files:
        # Validate all checkpoints
        for ckpt_path in all_checkpoints:
            is_valid, results = validator.validate_checkpoint(ckpt_path)
            print_validation_report(model_name, results, ckpt_path)
    else:
        # Only validate best.pth or latest checkpoint
        best_path = checkpoint_dir / 'best.pth'
        if best_path.exists():
            target_path = best_path
        else:
            target_path = all_checkpoints[0]  # Latest checkpoint
        
        is_valid, results = validator.validate_checkpoint(target_path)
        print_validation_report(model_name, results, target_path)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Validate model checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_best_model.py                    # Check joint_autoencoder best model
  python check_best_model.py --model autoencoder # Check autoencoder best model
  python check_best_model.py --check-all         # Check all models and all files
  python check_best_model.py --list              # List all available checkpoints
        """
    )
    parser.add_argument('--model', type=str, default='joint_autoencoder',
                        choices=list(CHECKPOINT_LOCATIONS.keys()),
                        help='Model name to check (default: joint_autoencoder)')
    parser.add_argument('--check-all', action='store_true',
                        help='Check all checkpoint files, not just best.pth')
    parser.add_argument('--list', action='store_true',
                        help='List all available checkpoints without validation')
    
    args = parser.parse_args()
    
    if args.list:
        # List all checkpoints
        print("\n" + "=" * 80)
        print("AVAILABLE CHECKPOINTS")
        print("=" * 80)
        for model_name, possible_dirs in CHECKPOINT_LOCATIONS.items():
            print(f"\n{model_name}:")
            found = False
            for dir_path in possible_dirs:
                dir_path_obj = Path(dir_path)
                if dir_path_obj.exists():
                    print(f"  Directory: {dir_path} ✓")
                    checkpoints = find_all_checkpoints(dir_path_obj)
                    if checkpoints:
                        print(f"  Found {len(checkpoints)} checkpoint(s):")
                        for ckpt in checkpoints:
                            size_mb = ckpt.stat().st_size / (1024 * 1024)
                            mtime = ckpt.stat().st_mtime
                            from datetime import datetime
                            mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                            print(f"    - {ckpt.name} ({size_mb:.2f} MB, {mtime_str})")
                        found = True
                    else:
                        print("  No checkpoints found")
                    break
            if not found:
                print(f"  Checked locations: {', '.join(possible_dirs)}")
                print("  ✗ No checkpoint directory found")
        return
    
    # Validate checkpoints
    success = check_model_checkpoints(args.model, args.check_all)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
>>>>>>> 70dc4f9 (chore: sync local typhoon updates)
