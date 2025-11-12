"""
Inference Script for Typhoon Prediction

Loads trained autoencoder and diffusion models to generate predictions
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

from models.autoencoder.autoencoder import SpatialAutoencoder
from models.diffusion.physics_diffusion import PhysicsInformedDiffusionModel
from data.datasets.typhoon_dataset import TyphoonDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TyphoonPredictor:
    """
    Prediction pipeline for typhoon forecasting
    """
    
    def __init__(
        self,
        autoencoder_path: str,
        diffusion_path: str,
        autoencoder_config: dict,
        diffusion_config: dict,
        device: str = 'cpu'
    ):
        """
        Initialize predictor
        
        Args:
            autoencoder_path: Path to trained autoencoder checkpoint
            diffusion_path: Path to trained diffusion checkpoint
            autoencoder_config: Autoencoder configuration
            diffusion_config: Diffusion configuration
            device: Device to run inference on
        """
        self.device = device
        
        # Load autoencoder
        logger.info(f"Loading autoencoder from {autoencoder_path}")
        self.autoencoder = self._load_autoencoder(autoencoder_path, autoencoder_config)
        self.autoencoder.eval()
        
        # Load diffusion model
        logger.info(f"Loading diffusion model from {diffusion_path}")
        self.diffusion = self._load_diffusion(diffusion_path, diffusion_config)
        self.diffusion.eval()
        
        logger.info("Models loaded successfully")
    
    def _load_autoencoder(self, path: str, config: dict) -> SpatialAutoencoder:
        """Load autoencoder model"""
        model = SpatialAutoencoder(
            in_channels=config['model']['in_channels'],
            latent_channels=config['model']['latent_channels'],
            hidden_dims=config['model']['hidden_dims'],
            use_attention=config['model'].get('use_attention', False)
        ).to(self.device)
        
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def _load_diffusion(self, path: str, config: dict) -> PhysicsInformedDiffusionModel:
        """Load diffusion model"""
        model = PhysicsInformedDiffusionModel(
            latent_channels=config['model']['latent_channels'],
            hidden_dim=config['model'].get('hidden_dim', 128),
            num_heads=config['model'].get('num_heads', 4),
            depth=config['model'].get('depth', 3),
            output_frames=config['model'].get('output_frames', 12)
        ).to(self.device)
        
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    @torch.no_grad()
    def predict(
        self,
        past_frames: torch.Tensor,
        num_future_steps: int = 8,
        ddim_steps: int = 50
    ) -> dict:
        """
        Generate predictions
        
        Args:
            past_frames: (B, T_past, C, H, W) past atmospheric frames
            num_future_steps: Number of future timesteps to predict
            ddim_steps: Number of DDIM sampling steps
        
        Returns:
            Dictionary containing:
            - future_frames: (B, T_future, C, H, W) predicted frames
            - future_latents: (B, T_future, C_latent, H_latent, W_latent)
            - track: (B, T_future, 2) predicted positions
            - intensity: (B, T_future,) predicted wind speeds
            - pressure: (B, T_future,) predicted pressures
        """
        B, T_past, C, H, W = past_frames.shape
        
        # Move to device
        past_frames = past_frames.to(self.device)
        
        # Encode past frames to latent space
        logger.info("Encoding past frames to latent space...")
        past_frames_flat = past_frames.reshape(B * T_past, C, H, W)
        past_latents = self.autoencoder.encode(past_frames_flat)
        _, C_latent, H_latent, W_latent = past_latents.shape
        past_latents = past_latents.reshape(B, T_past, C_latent, H_latent, W_latent)
        
        # Predict future latents using diffusion
        logger.info(f"Predicting {num_future_steps} future timesteps...")
        predictions = self.diffusion.predict(
            past_latents,
            num_future_steps=num_future_steps,
            ddim_steps=ddim_steps
        )
        
        future_latents = predictions['future_latents']
        
        # Decode future latents to atmospheric frames
        logger.info("Decoding future latents to atmospheric frames...")
        future_latents_flat = future_latents.reshape(B * num_future_steps, C_latent, H_latent, W_latent)
        future_frames = self.autoencoder.decode(future_latents_flat)
        future_frames = future_frames.reshape(B, num_future_steps, C, H, W)
        
        # Add decoded frames to predictions
        predictions['future_frames'] = future_frames
        
        return predictions
    
    def predict_batch(
        self,
        dataloader: DataLoader,
        output_dir: str,
        num_future_steps: int = 8
    ):
        """
        Generate predictions for a batch of samples
        
        Args:
            dataloader: DataLoader with test samples
            output_dir: Directory to save predictions
            num_future_steps: Number of future timesteps to predict
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_predictions = []
        all_ground_truth = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Predicting")):
            past_frames = batch['past_frames']
            future_frames_gt = batch['future_frames']
            
            # Generate predictions
            predictions = self.predict(past_frames, num_future_steps)
            
            # Store results
            all_predictions.append({
                'future_frames': predictions['future_frames'].cpu(),
                'track': predictions.get('track', None),
                'intensity': predictions.get('intensity', None),
                'pressure': predictions.get('pressure', None),
                'case_id': batch.get('case_id', f'sample_{batch_idx}')
            })
            
            all_ground_truth.append({
                'future_frames': future_frames_gt,
                'track': batch.get('track_future', None),
                'intensity': batch.get('intensity_future', None),
                'pressure': batch.get('pressure_future', None)
            })
        
        # Save predictions
        predictions_file = output_dir / 'predictions.npz'
        logger.info(f"Saving predictions to {predictions_file}")
        
        np.savez(
            predictions_file,
            predictions=all_predictions,
            ground_truth=all_ground_truth
        )
        
        logger.info(f"Predictions saved successfully!")
        
        return all_predictions, all_ground_truth


def main():
    parser = argparse.ArgumentParser(description='Typhoon Prediction Inference')
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True,
                      help='Path to autoencoder checkpoint')
    parser.add_argument('--diffusion_checkpoint', type=str, required=True,
                      help='Path to diffusion checkpoint')
    parser.add_argument('--autoencoder_config', type=str, default='configs/autoencoder_config.yaml',
                      help='Path to autoencoder config')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml',
                      help='Path to diffusion config')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                      help='Directory with processed data')
    parser.add_argument('--output_dir', type=str, default='results/predictions',
                      help='Directory to save predictions')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device to run inference on')
    parser.add_argument('--split', type=str, default='test',
                      help='Dataset split to use (train/val/test)')
    
    args = parser.parse_args()
    
    # Load configs
    logger.info("Loading configurations...")
    with open(args.autoencoder_config, 'r') as f:
        autoencoder_config = yaml.safe_load(f)
    
    with open(args.diffusion_config, 'r') as f:
        diffusion_config = yaml.safe_load(f)
    
    # Create predictor
    predictor = TyphoonPredictor(
        autoencoder_path=args.autoencoder_checkpoint,
        diffusion_path=args.diffusion_checkpoint,
        autoencoder_config=autoencoder_config,
        diffusion_config=diffusion_config,
        device=args.device
    )
    
    # Load test dataset
    logger.info(f"Loading {args.split} dataset...")
    dataset = TyphoonDataset(
        data_dir=args.data_dir,
        split=args.split,
        normalize=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Dataset size: {len(dataset)} samples")
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions, ground_truth = predictor.predict_batch(
        dataloader,
        output_dir=args.output_dir,
        num_future_steps=8
    )
    
    logger.info("Inference complete!")


if __name__ == '__main__':
    main()
