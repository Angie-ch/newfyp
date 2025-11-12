"""
Autoencoder Trainer

Trains the spatial autoencoder for frame compression
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import logging

from models.autoencoder.autoencoder import SpatialAutoencoder, AutoencoderLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoencoderTrainer:
    """
    Trainer for spatial autoencoder
    """
    
    def __init__(
        self,
        model: SpatialAutoencoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = 'cuda'
    ):
        """
        Initialize trainer
        
        Args:
            model: Autoencoder model
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
        
        # Loss function
        self.criterion = AutoencoderLoss()
        
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
        self.log_dir = Path(config.get('log_dir', 'logs/autoencoder'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Checkpointing
        self.save_dir = Path(config.get('save_dir', 'checkpoints/autoencoder'))
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
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Step scheduler
            self.scheduler.step()
            
            # Log
            logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_interval', 5) == 0:
                self.save_checkpoint(epoch, val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
                logger.info(f"New best model saved with val loss: {val_loss:.6f}")
        
        logger.info("Training complete!")
        self.writer.close()
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Get data - use both past and future frames for autoencoder training
            past_frames = batch['past_frames'].to(self.device)  # (B, T_past, C, H, W)
            future_frames = batch['future_frames'].to(self.device)  # (B, T_future, C, H, W)
            frames = torch.cat([past_frames, future_frames], dim=1)  # (B, T_past+T_future, C, H, W)
            
            B, T, C, H, W = frames.shape
            
            # Process all frames independently
            frames_flat = frames.reshape(B * T, C, H, W)
            
            # Forward pass
            recon, latent = self.model(frames_flat)
            
            # Compute loss
            loss, loss_dict = self.criterion(recon, frames_flat)
            
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
            total_loss += loss.item()
            
            # Log
            if batch_idx % self.config.get('log_interval', 100) == 0:
                self.writer.add_scalar('Loss/train_batch', loss.item(), self.global_step)
                pbar.set_postfix({'loss': loss.item()})
            
            self.global_step += 1
        
        return total_loss / len(self.train_loader)
    
    def validate(self, epoch: int) -> float:
        """
        Validate model
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                past_frames = batch['past_frames'].to(self.device)
                future_frames = batch['future_frames'].to(self.device)
                frames = torch.cat([past_frames, future_frames], dim=1)
                
                B, T, C, H, W = frames.shape
                frames_flat = frames.reshape(B * T, C, H, W)
                
                # Forward pass
                recon, latent = self.model(frames_flat)
                
                # Compute loss
                loss, _ = self.criterion(recon, frames_flat)
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
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

