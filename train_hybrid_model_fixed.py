"""
修复后的混合模型训练脚本
正确使用ImagenTrainer进行Diffusion训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
import json
from tqdm import tqdm

from hybrid_typhoon_predictor_v3 import HybridTyphoonPredictor_V3

# 导入ImagenTrainer
try:
    from imagen_pytorch import ImagenTrainer
    print("[OK] ImagenTrainer imported successfully!")
except ImportError as e:
    print(f"[ERROR] Failed to import ImagenTrainer: {e}")
    raise


# ============================================================================
# 数据集（与之前相同）
# ============================================================================

class TyphoonDataset(Dataset):
    """台风数据集"""
    def __init__(self, data_dir, max_samples=None):
        self.data_dir = Path(data_dir)
        self.samples = sorted(list(self.data_dir.glob("*.npz")))
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        
        print(f"[INFO] Found {len(self.samples)} samples in {data_dir}")
        
        if len(self.samples) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        
        try:
            data = np.load(sample_path)
            
            past_frames = torch.from_numpy(data['past_frames']).float()
            future_frames = torch.from_numpy(data['future_frames']).float()
            track_past = torch.from_numpy(data['track_past']).float()
            track_future = torch.from_numpy(data['track_future']).float()
            
            return {
                'past_frames': past_frames,
                'future_frames': future_frames,
                'track_past': track_past,
                'track_future': track_future,
                'sample_name': sample_path.stem
            }
        
        except Exception as e:
            print(f"[ERROR] Failed to load {sample_path}: {e}")
            raise


# ============================================================================
# 修复后的训练函数
# ============================================================================

def train_hybrid_model_fixed(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    lr=3e-4,
    device='cuda',
    save_dir='checkpoints_fixed',
    log_interval=1
):
    """
    修复后的训练函数 - 正确使用ImagenTrainer
    """
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'='*80}")
    print(f"[INFO] Starting training with FIXED Diffusion integration...")
    print(f"{'='*80}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Device: {device}")
    print(f"  - Save directory: {save_dir}")
    print(f"  - Train samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"  - Val samples: {len(val_loader.dataset)}")
    
    # 1. 创建优化器
    # Encoder和条件投影优化器
    encoder_params = list(model.convlstm_encoder.parameters()) + list(model.cond_proj.parameters())
    encoder_optimizer = torch.optim.Adam(encoder_params, lr=lr)
    
    # Track predictor优化器
    track_optimizer = torch.optim.Adam(model.track_predictor.parameters(), lr=lr)
    
    # 2. 创建ImagenTrainer（独立管理diffusion训练）
    print(f"\n[INFO] Creating ImagenTrainer for Diffusion...")
    try:
        diffusion_trainer = ImagenTrainer(
            model.imagen,
            lr=lr,
            verbose=False
        ).to(device)
        print(f"[OK] ImagenTrainer created successfully!")
        use_diffusion = True
    except Exception as e:
        print(f"[ERROR] Failed to create ImagenTrainer: {e}")
        print(f"[WARNING] Will skip diffusion training")
        diffusion_trainer = None
        use_diffusion = False
    
    # 训练历史
    history = {
        'train_track_loss': [],
        'train_diffusion_loss': [],
        'train_total_loss': [],
        'val_track_loss': [],
        'val_diffusion_loss': [],
        'val_total_loss': []
    }
    
    best_val_loss = float('inf')
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*80}")
        
        # ========== 训练阶段 ==========
        model.train()
        train_track_losses = []
        train_diffusion_losses = []
        train_total_losses = []
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            # 移动到设备
            past_frames = batch['past_frames'].to(device)
            future_frames = batch['future_frames'].to(device)
            track_past = batch['track_past'].to(device)
            track_future = batch['track_future'].to(device)
            
            # Step 1: 编码并准备条件
            convlstm_features, cond_video, predicted_track = model.encode_and_prepare_condition(past_frames)
            
            # Step 2: Track损失和更新
            track_loss = F.mse_loss(predicted_track, track_future)
            
            encoder_optimizer.zero_grad()
            track_optimizer.zero_grad()
            track_loss.backward(retain_graph=True)  # 保留计算图用于diffusion
            encoder_optimizer.step()
            track_optimizer.step()
            
            # Step 3: Diffusion训练（使用ImagenTrainer）
            diffusion_loss = torch.tensor(0.0, device=device)
            if use_diffusion and diffusion_trainer is not None:
                # 转换future_frames格式: (B, T, C, H, W) -> (B, C, T, H, W)
                future_frames_perm = future_frames.permute(0, 2, 1, 3, 4)
                
                # 重新计算条件（因为之前backward了）
                with torch.no_grad():
                    _, cond_video, _ = model.encode_and_prepare_condition(past_frames)
                
                # ImagenTrainer的正确调用方式（移除try-except查看真实错误）
                diffusion_loss = diffusion_trainer(
                    future_frames_perm,           # 目标视频 (B, C, T, H, W)
                    cond_video_frames=cond_video, # 条件视频 (B, C, T, H, W)
                    unet_number=1                 # 使用第1个unet
                )
                
                # 更新diffusion参数
                diffusion_trainer.update(unet_number=1)
            
            # 记录损失
            total_loss = track_loss + diffusion_loss
            train_track_losses.append(track_loss.item())
            train_diffusion_losses.append(diffusion_loss.item() if isinstance(diffusion_loss, torch.Tensor) else diffusion_loss)
            train_total_losses.append(total_loss.item())
            
            # 更新进度条
            pbar.set_postfix({
                'track': f'{track_loss.item():.2f}',
                'diff': f'{diffusion_loss.item() if isinstance(diffusion_loss, torch.Tensor) else diffusion_loss:.2f}',
                'total': f'{total_loss.item():.2f}'
            })
        
        # 计算平均训练损失
        avg_train_track = np.mean(train_track_losses)
        avg_train_diff = np.mean(train_diffusion_losses)
        avg_train_total = np.mean(train_total_losses)
        
        history['train_track_loss'].append(avg_train_track)
        history['train_diffusion_loss'].append(avg_train_diff)
        history['train_total_loss'].append(avg_train_total)
        
        print(f"\n[TRAIN] Epoch {epoch+1}:")
        print(f"  - Track Loss: {avg_train_track:.4f}")
        print(f"  - Diffusion Loss: {avg_train_diff:.4f} {'[OK]' if avg_train_diff > 0 else '[NOT TRAINING]'}")
        print(f"  - Total Loss: {avg_train_total:.4f}")
        
        # ========== 验证阶段 ==========
        if val_loader is not None:
            model.eval()
            val_track_losses = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                    past_frames = batch['past_frames'].to(device)
                    track_future = batch['track_future'].to(device)
                    
                    # Forward pass
                    _, cond_video, predicted_track = model.encode_and_prepare_condition(past_frames)
                    track_loss = F.mse_loss(predicted_track, track_future)
                    
                    val_track_losses.append(track_loss.item())
            
            avg_val_track = np.mean(val_track_losses)
            avg_val_total = avg_val_track  # 验证时只计算track loss
            
            history['val_track_loss'].append(avg_val_track)
            history['val_diffusion_loss'].append(0.0)
            history['val_total_loss'].append(avg_val_total)
            
            print(f"[VAL] Epoch {epoch+1}:")
            print(f"  - Track Loss: {avg_val_track:.4f}")
            print(f"  - Total Loss: {avg_val_total:.4f}")
            
            # 保存最佳模型
            if avg_val_total < best_val_loss:
                best_val_loss = avg_val_total
                best_model_path = os.path.join(save_dir, 'best_model.pt')
                
                # 保存完整状态
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                    'track_optimizer_state_dict': track_optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'history': history
                }, best_model_path)
                
                # 保存ImagenTrainer状态
                if diffusion_trainer is not None:
                    diffusion_trainer.save(os.path.join(save_dir, 'best_diffusion_trainer.pt'))
                
                print(f"  [OK] Best model saved! (val_loss: {best_val_loss:.4f})")
        
        # 每个epoch保存checkpoint
        if (epoch + 1) % log_interval == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                'track_optimizer_state_dict': track_optimizer.state_dict(),
                'history': history
            }, checkpoint_path)
            
            if diffusion_trainer is not None:
                diffusion_trainer.save(os.path.join(save_dir, f'diffusion_trainer_epoch_{epoch+1}.pt'))
            
            print(f"  [OK] Checkpoint saved: {checkpoint_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'final_model.pt')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
        'track_optimizer_state_dict': track_optimizer.state_dict(),
        'history': history
    }, final_model_path)
    
    if diffusion_trainer is not None:
        diffusion_trainer.save(os.path.join(save_dir, 'final_diffusion_trainer.pt'))
    
    print(f"\n[OK] Final model saved: {final_model_path}")
    
    # 保存训练历史
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"[OK] Training history saved: {history_path}")
    
    return history


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("="*80)
    print("FIXED Hybrid Model Training - Small Dataset Test")
    print("="*80)
    
    # 配置
    DATA_DIR = "D:/typhoon_data_2018_2021_full/train/cases"
    MAX_TRAIN_SAMPLES = 3
    MAX_VAL_SAMPLES = 1
    BATCH_SIZE = 1
    NUM_EPOCHS = 5
    LR = 3e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_DIR = 'checkpoints_fixed'
    
    print(f"\n[CONFIG]")
    print(f"  - Data directory: {DATA_DIR}")
    print(f"  - Max train samples: {MAX_TRAIN_SAMPLES}")
    print(f"  - Max val samples: {MAX_VAL_SAMPLES}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Learning rate: {LR}")
    print(f"  - Device: {DEVICE}")
    
    # 创建数据集
    print(f"\n{'='*80}")
    print("Loading data...")
    print(f"{'='*80}")
    
    try:
        train_dataset = TyphoonDataset(DATA_DIR, max_samples=MAX_TRAIN_SAMPLES)
        val_dataset = TyphoonDataset(DATA_DIR, max_samples=MAX_VAL_SAMPLES + MAX_TRAIN_SAMPLES)
        val_dataset.samples = val_dataset.samples[MAX_TRAIN_SAMPLES:]
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        print(f"[OK] Data loaded successfully!")
        
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建模型（使用V3）
    print(f"\n{'='*80}")
    print("Creating model...")
    print(f"{'='*80}")
    
    try:
        model = HybridTyphoonPredictor_V3(
            input_channels=24,
            hidden_channels=64,
            output_channels=24,
            past_timesteps=8,
            future_timesteps=12,
            image_size=(64, 64),
            unet_dim=32,
            unet_dim_mults=(1, 2, 4, 8),
            unet_num_resnet_blocks=3,
            diffusion_timesteps=250,
        ).to(DEVICE)
        
        print(f"\n[OK] Model created!")
        
    except Exception as e:
        print(f"[ERROR] Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 开始训练
    try:
        history = train_hybrid_model_fixed(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=NUM_EPOCHS,
            lr=LR,
            device=DEVICE,
            save_dir=SAVE_DIR,
            log_interval=1
        )
        
        print(f"\n{'='*80}")
        print("[OK] Training completed successfully!")
        print(f"{'='*80}")
        
        # 打印最终结果
        print(f"\n[FINAL RESULTS]")
        print(f"  Train Track Loss: {history['train_track_loss'][-1]:.4f}")
        print(f"  Train Diffusion Loss: {history['train_diffusion_loss'][-1]:.4f} {'[SUCCESS!]' if history['train_diffusion_loss'][-1] > 0 else '[FAILED]'}")
        print(f"  Train Total Loss: {history['train_total_loss'][-1]:.4f}")
        if history['val_track_loss']:
            print(f"  Val Track Loss: {history['val_track_loss'][-1]:.4f}")
            print(f"  Val Total Loss: {history['val_total_loss'][-1]:.4f}")
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

