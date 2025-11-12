"""
Joint Diffusion Model Trainer

Trains the diffusion model using joint autoencoder that encodes ERA5 + IBTrACS together
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import logging
import copy

from models.diffusion.physics_diffusion import PhysicsInformedDiffusionModel, DiffusionSchedule
from models.autoencoder.joint_autoencoder import JointAutoencoder
from training.losses.multitask_loss import MultiTaskDiffusionLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JointDiffusionTrainer:
    """
    Trainer for diffusion model with joint autoencoder (ERA5 + IBTrACS)
    
    Key difference from standard diffusion trainer:
    - Uses joint autoencoder that encodes ERA5 + IBTrACS together
    - Latent space contains both atmospheric and track/intensity information
    - Decoder separates them back for evaluation
    """
    
    def __init__(
        self,
        model: PhysicsInformedDiffusionModel,
        autoencoder: JointAutoencoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = 'cuda'
    ):
        """
        Initialize trainer
        
        Args:
            model: Diffusion model
            autoencoder: Pretrained JOINT autoencoder
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.autoencoder = autoencoder.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Freeze autoencoder
        self.autoencoder.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        
        # Diffusion schedule
        self.diffusion = DiffusionSchedule(
            num_timesteps=config.get('timesteps', 1000),
            beta_start=config.get('beta_start', 1e-4),
            beta_end=config.get('beta_end', 0.02),
            schedule_type=config.get('beta_schedule', 'linear')
        ).to(device)
        
        # Loss function
        loss_weights = config.get('loss_weights', {})
        physics_weights = config.get('physics_weights', {})
        
        self.criterion = MultiTaskDiffusionLoss(
            autoencoder=self.autoencoder,
            diffusion_weight=loss_weights.get('diffusion', 1.0),
            track_weight=loss_weights.get('track', 0.5),
            intensity_weight=loss_weights.get('intensity', 0.3),
            physics_weight=loss_weights.get('physics', 0.2),
            consistency_weight=loss_weights.get('consistency', 0.1),
            physics_weights=physics_weights
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 2e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 100)
        )
        
        # EMA model
        self.ema_model = copy.deepcopy(self.model)
        self.ema_decay = config.get('ema_decay', 0.9999)
        
        # Logging
        self.log_dir = Path(config.get('log_dir', 'logs/joint_diffusion'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Checkpointing
        self.save_dir = Path(config.get('save_dir', 'checkpoints/joint_diffusion'))
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
            epochs = self.config.get('epochs', 100)
        
        logger.info(f"Starting joint diffusion training for {epochs} epochs")
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Step scheduler
            self.scheduler.step()
            
            # Log
            logger.info(f"Train Loss: {train_metrics['total']:.6f}, Val Loss: {val_metrics['total']:.6f}")
            
            for key in train_metrics:
                self.writer.add_scalar(f'Train/{key}', train_metrics[key], epoch)
            
            for key in val_metrics:
                self.writer.add_scalar(f'Val/{key}', val_metrics[key], epoch)
            
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_interval', 5) == 0:
                self.save_checkpoint(epoch, val_metrics['total'])
            
            # Save best model
            if val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                self.save_checkpoint(epoch, val_metrics['total'], is_best=True)
                logger.info(f"New best model saved with val loss: {val_metrics['total']:.6f}")
        
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
        
        metrics_accum = {}
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Get data
            past_frames = batch['past_frames'].to(self.device)      # (B, T_in, C, H, W)
            future_frames = batch['future_frames'].to(self.device)  # (B, T_out, C, H, W)
            past_track = batch['track_past'].to(self.device)        # (B, T_in, 2)
            target_track = batch['track_future'].to(self.device)    # (B, T_out, 2)
            past_intensity = batch['intensity_past'].to(self.device)    # (B, T_in)
            target_intensity = batch['intensity_future'].to(self.device)  # (B, T_out)
            
            B, T_in, C, H, W = past_frames.shape
            T_out = future_frames.shape[1]
            
            # ═════════════════════════════════════════════════════════════
            # JOINT ENCODING: Encode ERA5 + IBTrACS together
            # ═════════════════════════════════════════════════════════════
            
            with torch.no_grad():
                # Flatten temporal dimension for encoding
                past_frames_flat = past_frames.reshape(B * T_in, C, H, W)
                past_track_flat = past_track.reshape(B * T_in, 2)
                past_intensity_flat = past_intensity.reshape(B * T_in)
                
                # Joint encode past (ERA5 + IBTrACS → unified latent)
                past_latents = self.autoencoder.encode(
                    past_frames_flat, 
                    past_track_flat, 
                    past_intensity_flat
                )
                past_latents = past_latents.reshape(B, T_in, -1, H // 8, W // 8)
                
                # Joint encode future (ERA5 + IBTrACS → unified latent)
                future_frames_flat = future_frames.reshape(B * T_out, C, H, W)
                future_track_flat = target_track.reshape(B * T_out, 2)
                future_intensity_flat = target_intensity.reshape(B * T_out)
                
                future_latents = self.autoencoder.encode(
                    future_frames_flat,
                    future_track_flat,
                    future_intensity_flat
                )
                future_latents = future_latents.reshape(B, T_out, -1, H // 8, W // 8)
            
            # Sample random diffusion timestep
            t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)
            
            # Add noise to target latents
            noise = torch.randn_like(future_latents)
            noisy_latents = self.diffusion.add_noise(future_latents, t, noise)
            
            # ═════════════════════════════════════════════════════════════
            # CONDITIONING: Only need past latents (already contains track/intensity!)
            # ═════════════════════════════════════════════════════════════
            
            condition_dict = {
                'past_latents': past_latents,
                # No need for separate past_track and past_intensity!
                # They're already encoded in past_latents
            }
            
            # Forward pass through diffusion model
            predictions = self.model(noisy_latents, t, condition_dict, return_x0=True)
            
            # Prepare targets
            targets = {
                'noise': noise,
                'track': target_track,
                'intensity': target_intensity,
                'frames': future_latents
            }
            
            # Compute loss
            loss, loss_dict = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('gradient_clip', 1.0)
            )
            
            self.optimizer.step()
            
            # Update EMA model
            self.update_ema()
            
            # Accumulate metrics
            for key, value in loss_dict.items():
                if key not in metrics_accum:
                    metrics_accum[key] = 0
                metrics_accum[key] += value
            
            # Log
            if batch_idx % self.config.get('log_interval', 50) == 0:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'Loss/{key}_batch', value, self.global_step)
                
                pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})
            
            self.global_step += 1
        
        # Average metrics
        for key in metrics_accum:
            metrics_accum[key] /= len(self.train_loader)
        
        return metrics_accum
    
    def validate(self, epoch: int) -> dict:
        """
        Validate model
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary of average validation metrics
        """
        self.model.eval()
        
        metrics_accum = {}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Get data
                past_frames = batch['past_frames'].to(self.device)
                future_frames = batch['future_frames'].to(self.device)
                past_track = batch['track_past'].to(self.device)
                target_track = batch['track_future'].to(self.device)
                past_intensity = batch['intensity_past'].to(self.device)
                target_intensity = batch['intensity_future'].to(self.device)
                
                B, T_in, C, H, W = past_frames.shape
                T_out = future_frames.shape[1]
                
                # Joint encode
                past_frames_flat = past_frames.reshape(B * T_in, C, H, W)
                past_track_flat = past_track.reshape(B * T_in, 2)
                past_intensity_flat = past_intensity.reshape(B * T_in)
                
                past_latents = self.autoencoder.encode(
                    past_frames_flat, past_track_flat, past_intensity_flat
                )
                past_latents = past_latents.reshape(B, T_in, -1, H // 8, W // 8)
                
                future_frames_flat = future_frames.reshape(B * T_out, C, H, W)
                future_track_flat = target_track.reshape(B * T_out, 2)
                future_intensity_flat = target_intensity.reshape(B * T_out)
                
                future_latents = self.autoencoder.encode(
                    future_frames_flat, future_track_flat, future_intensity_flat
                )
                future_latents = future_latents.reshape(B, T_out, -1, H // 8, W // 8)
                
                # Sample timestep
                t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)
                
                # Add noise
                noise = torch.randn_like(future_latents)
                noisy_latents = self.diffusion.add_noise(future_latents, t, noise)
                
                # Conditioning
                condition_dict = {
                    'past_latents': past_latents,
                }
                
                # Forward
                predictions = self.model(noisy_latents, t, condition_dict, return_x0=True)
                
                # Targets
                targets = {
                    'noise': noise,
                    'track': target_track,
                    'intensity': target_intensity,
                    'frames': future_latents
                }
                
                # Loss
                loss, loss_dict = self.criterion(predictions, targets)
                
                # Accumulate
                for key, value in loss_dict.items():
                    if key not in metrics_accum:
                        metrics_accum[key] = 0
                    metrics_accum[key] += value
        
        # Average
        for key in metrics_accum:
            metrics_accum[key] /= len(self.val_loader)
        
        return metrics_accum
    
    def update_ema(self):
        """Update EMA model"""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
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
            'ema_model_state_dict': self.ema_model.state_dict(),
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
        self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint['epoch']

