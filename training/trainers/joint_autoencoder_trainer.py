"""
Joint Autoencoder Trainer

Trains the joint autoencoder that encodes ERA5 + IBTrACS together
and decodes them separately
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import logging

from models.autoencoder.joint_autoencoder import JointAutoencoder, JointAutoencoderLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JointAutoencoderTrainer:
    """
    Trainer for joint autoencoder (ERA5 + IBTrACS)
    """
    
    def __init__(
        self,
        model: JointAutoencoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = 'cuda'
    ):
        """
        Initialize trainer
        
        Args:
            model: Joint autoencoder model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function with configurable weights
        self.criterion = JointAutoencoderLoss(
            era5_weight=config.get('era5_weight', 1.0),
            track_weight=config.get('track_weight', 10.0),
            intensity_weight=config.get('intensity_weight', 5.0)
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 50)
        )
        
        # Logging
        self.log_dir = Path(config.get('log_dir', 'logs/joint_autoencoder'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Checkpointing
        self.save_dir = Path(config.get('save_dir', 'checkpoints/joint_autoencoder'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.global_step = 0
    
    def train(self, epochs: int = None):
        """
        Main training loop
        
        Args:
            epochs: Number of epochs to train
        """
        if epochs is None:
            epochs = self.config.get('epochs', 50)
        
        logger.info(f"Starting joint autoencoder training for {epochs} epochs")
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss_dict = self.train_epoch(epoch)
            
            # Validate
            val_loss_dict = self.validate(epoch)
            
            # Step scheduler
            self.scheduler.step()
            
            # Log
            logger.info(f"Train Loss: {train_loss_dict['total']:.6f} "
                       f"(ERA5: {train_loss_dict['era5']:.6f}, "
                       f"Track: {train_loss_dict['track']:.6f}, "
                       f"Intensity: {train_loss_dict['intensity']:.6f})")
            logger.info(f"Val Loss: {val_loss_dict['total']:.6f} "
                       f"(ERA5: {val_loss_dict['era5']:.6f}, "
                       f"Track: {val_loss_dict['track']:.6f}, "
                       f"Intensity: {val_loss_dict['intensity']:.6f})")
            
            # TensorBoard logging
            for key in ['total', 'era5', 'track', 'intensity']:
                self.writer.add_scalar(f'Loss/train_{key}', train_loss_dict[key], epoch)
                self.writer.add_scalar(f'Loss/val_{key}', val_loss_dict[key], epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_interval', 5) == 0:
                self.save_checkpoint(epoch, val_loss_dict['total'])
            
            # Save best model
            if val_loss_dict['total'] < self.best_val_loss:
                self.best_val_loss = val_loss_dict['total']
                self.save_checkpoint(epoch, val_loss_dict['total'], is_best=True)
                logger.info(f"New best model saved with val loss: {val_loss_dict['total']:.6f}")
        
        logger.info("Training complete!")
        self.writer.close()
    
    def train_epoch(self, epoch: int) -> dict:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        total_losses = {'total': 0, 'era5': 0, 'track': 0, 'intensity': 0}
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Get ALL data (ERA5 + IBTrACS)
            past_frames = batch['past_frames'].to(self.device)      # (B, T_past, C, H, W)
            future_frames = batch['future_frames'].to(self.device)  # (B, T_future, C, H, W)
            past_track = batch['track_past'].to(self.device)        # (B, T_past, 2)
            future_track = batch['track_future'].to(self.device)    # (B, T_future, 2)
            past_intensity = batch['intensity_past'].to(self.device)    # (B, T_past)
            future_intensity = batch['intensity_future'].to(self.device)  # (B, T_future)
            
            # Get expected ERA5 channels from model config
            # Model expects era5_channels, but data might have more (e.g., 48 vs 40)
            # Use only the first era5_channels channels
            era5_channels = self.model.era5_channels
            past_frames = past_frames[:, :, :era5_channels, :, :]  # Slice to expected channels
            future_frames = future_frames[:, :, :era5_channels, :, :]
            
            # Combine past and future
            frames = torch.cat([past_frames, future_frames], dim=1)  # (B, T_total, C, H, W)
            tracks = torch.cat([past_track, future_track], dim=1)    # (B, T_total, 2)
            intensities = torch.cat([past_intensity, future_intensity], dim=1)  # (B, T_total)
            
            B, T, C, H, W = frames.shape
            
            # Flatten temporal dimension for processing
            frames_flat = frames.reshape(B * T, C, H, W)
            tracks_flat = tracks.reshape(B * T, 2)
            intensities_flat = intensities.reshape(B * T)
            
            # Forward pass through JOINT autoencoder
            outputs = self.model(frames_flat, tracks_flat, intensities_flat)
            
            # Prepare targets
            targets = {
                'era5': frames_flat,
                'track': tracks_flat,
                'intensity': intensities_flat
            }
            
            # Compute loss
            loss, loss_dict = self.criterion(outputs, targets)
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss detected at batch {batch_idx}, skipping batch")
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Check for NaN gradients
            has_nan = False
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        logger.warning(f"NaN/Inf gradient in {name}, skipping batch")
                        has_nan = True
                        break
            
            if has_nan:
                self.optimizer.zero_grad()
                continue
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('gradient_clip', 1.0)
            )
            
            # Log gradient norm occasionally
            if batch_idx % self.config.get('log_interval', 100) == 0:
                self.writer.add_scalar('Gradients/norm', grad_norm.item(), self.global_step)
            
            self.optimizer.step()
            
            # Update metrics
            for key in loss_dict:
                total_losses[key] += loss_dict[key]
            
            # Log
            if batch_idx % self.config.get('log_interval', 100) == 0:
                for key in loss_dict:
                    self.writer.add_scalar(f'Loss/train_batch_{key}', loss_dict[key], self.global_step)
                pbar.set_postfix({
                    'loss': loss_dict['total'],
                    'era5': loss_dict['era5'],
                    'track': loss_dict['track'],
                    'int': loss_dict['intensity']
                })
            
            self.global_step += 1
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= len(self.train_loader)
        
        return total_losses
    
    def validate(self, epoch: int) -> dict:
        """
        Validate model
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary of average validation losses
        """
        self.model.eval()
        total_losses = {'total': 0, 'era5': 0, 'track': 0, 'intensity': 0}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Get data
                past_frames = batch['past_frames'].to(self.device)
                future_frames = batch['future_frames'].to(self.device)
                past_track = batch['track_past'].to(self.device)
                future_track = batch['track_future'].to(self.device)
                past_intensity = batch['intensity_past'].to(self.device)
                future_intensity = batch['intensity_future'].to(self.device)
                
                # Combine
                frames = torch.cat([past_frames, future_frames], dim=1)
                tracks = torch.cat([past_track, future_track], dim=1)
                intensities = torch.cat([past_intensity, future_intensity], dim=1)
                
                B, T, C, H, W = frames.shape
                
                # Flatten
                frames_flat = frames.reshape(B * T, C, H, W)
                tracks_flat = tracks.reshape(B * T, 2)
                intensities_flat = intensities.reshape(B * T)
                
                # Forward pass
                outputs = self.model(frames_flat, tracks_flat, intensities_flat)
                
                # Targets
                targets = {
                    'era5': frames_flat,
                    'track': tracks_flat,
                    'intensity': intensities_flat
                }
                
                # Compute loss
                loss, loss_dict = self.criterion(outputs, targets)
                
                # Update metrics
                for key in loss_dict:
                    total_losses[key] += loss_dict[key]
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= len(self.val_loader)
        
        return total_losses
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        if is_best:
            path = self.save_dir / 'best.pth'
            logger.info(f"Saving best checkpoint to {path}")
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            logger.info(f"Saving checkpoint to {path}")
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint
        
        Args:
            path: Path to checkpoint file
        """
        logger.info(f"Loading checkpoint from {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint['epoch']

