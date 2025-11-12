"""
Helper Functions and Utilities
"""

import torch
import numpy as np
import random
import yaml
from pathlib import Path
import logging


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, output_path: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        output_path: Path to save YAML file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Create logger
    
    Args:
        name: Logger name
        log_file: Optional log file path
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def denormalize_coordinates(coords: torch.Tensor, lat_range: tuple = (-90, 90), lon_range: tuple = (-180, 180)) -> torch.Tensor:
    """
    Denormalize coordinates from [0, 1] to lat/lon ranges
    
    Args:
        coords: (B, T, 2) normalized coordinates
        lat_range: (min, max) latitude range
        lon_range: (min, max) longitude range
    
    Returns:
        Denormalized coordinates
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    
    denorm_coords = coords.clone()
    denorm_coords[..., 0] = coords[..., 0] * (lat_max - lat_min) + lat_min
    denorm_coords[..., 1] = coords[..., 1] * (lon_max - lon_min) + lon_min
    
    return denorm_coords


def normalize_coordinates(coords: torch.Tensor, lat_range: tuple = (-90, 90), lon_range: tuple = (-180, 180)) -> torch.Tensor:
    """
    Normalize coordinates from lat/lon to [0, 1] range
    
    Args:
        coords: (B, T, 2) lat/lon coordinates
        lat_range: (min, max) latitude range
        lon_range: (min, max) longitude range
    
    Returns:
        Normalized coordinates
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    
    norm_coords = coords.clone()
    norm_coords[..., 0] = (coords[..., 0] - lat_min) / (lat_max - lat_min)
    norm_coords[..., 1] = (coords[..., 1] - lon_min) / (lon_max - lon_min)
    
    return norm_coords

