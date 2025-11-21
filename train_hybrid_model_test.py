"""
混合模型训练脚本 - 小数据集测试版本
先用少量样本验证训练流程是否正确
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

from hybrid_typhoon_predictor_v2 import HybridTyphoonPredictor_V2

# 尝试导入imagen_pytorch的ImagenTrainer
try:
    from imagen_pytorch import ImagenTrainer
    print("[OK] ImagenTrainer imported successfully!")
except ImportError as e:
    print(f"[ERROR] Failed to import ImagenTrainer: {e}")
    raise


# ============================================================================
# 数据集
# ============================================================================

class TyphoonDataset(Dataset):
    """台风数据集"""
    def __init__(self, data_dir, max_samples=None):
        self.data_dir = Path(data_dir)
        
        # 查找所有.npz文件
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
            
            # 加载数据
            past_frames = torch.from_numpy(data['past_frames']).float()  # (T_past, C, H, W)
            future_frames = torch.from_numpy(data['future_frames']).float()  # (T_future, C, H, W)
            track_past = torch.from_numpy(data['track_past']).float()  # (T_past, 2)
            track_future = torch.from_numpy(data['track_future']).float()  # (T_future, 2)
            
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
# 训练函数
# ============================================================================

def train_hybrid_model(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    lr=3e-4,
    device='cuda',
    save_dir='checkpoints_test',
    log_interval=1
):
    """训练混合模型"""
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n[INFO] Starting training...")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Device: {device}")
    print(f"  - Save directory: {save_dir}")
    print(f"  - Train samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"  - Val samples: {len(val_loader.dataset)}")
    
    # 初始化优化器
    # 注意：对于diffusion部分，我们需要使用ImagenTrainer
    # 对于track predictor，我们使用单独的优化器
    
    # Track predictor优化器
    track_optimizer = torch.optim.Adam(model.track_predictor.parameters(), lr=lr)
    
    # ConvLSTM和条件投影优化器
    encoder_params = list(model.convlstm_encoder.parameters()) + list(model.cond_proj.parameters())
    encoder_optimizer = torch.optim.Adam(encoder_params, lr=lr)
    
    # ImagenTrainer for diffusion
    print("\n[INFO] Initializing ImagenTrainer for diffusion part...")
    try:
        # 创建ImagenTrainer
        diffusion_trainer = ImagenTrainer(
            model.imagen,
            lr=lr,
            verbose=False
        ).to(device)
        print("[OK] ImagenTrainer initialized!")
    except Exception as e:
        print(f"[ERROR] Failed to initialize ImagenTrainer: {e}")
        print("[WARNING] Will skip diffusion training and only train track predictor")
        diffusion_trainer = None
    
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
            past_frames = batch['past_frames'].to(device)  # (B, T_past, C, H, W)
            future_frames = batch['future_frames'].to(device)  # (B, T_future, C, H, W)
            track_past = batch['track_past'].to(device)
            track_future = batch['track_future'].to(device)
            
            # 1. Forward pass through ConvLSTM and get condition
            _, last_state = model.convlstm_encoder(past_frames)
            convlstm_features = last_state[-1][0]  # (B, C_hidden, H, W)
            
            # 2. Condition projection
            cond_features_2d = model.cond_proj(convlstm_features)
            cond_video = cond_features_2d.unsqueeze(2).repeat(1, 1, model.future_timesteps, 1, 1)
            
            # 3. Track prediction loss
            track_input = convlstm_features.reshape(past_frames.shape[0], -1)
            predicted_track = model.track_predictor(track_input)
            track_loss = F.mse_loss(predicted_track, track_future)
            
            # 4. Diffusion training
            diffusion_loss = torch.tensor(0.0, device=device)
            if diffusion_trainer is not None:
                try:
                    # 转换格式: (B, T, C, H, W) -> (B, C, T, H, W)
                    future_frames_perm = future_frames.permute(0, 2, 1, 3, 4)
                    
                    # ImagenTrainer的forward会计算diffusion loss
                    diffusion_loss = diffusion_trainer(
                        future_frames_perm,
                        cond_video_frames=cond_video,
                        unet_number=1  # 第一个unet
                    )
                    
                    # 更新diffusion部分
                    diffusion_trainer.update(unet_number=1)
                    
                except Exception as e:
                    print(f"\n[WARNING] Diffusion training error: {e}")
                    diffusion_loss = torch.tensor(0.0, device=device)
            
            # 5. 更新encoder和track predictor
            encoder_optimizer.zero_grad()
            track_optimizer.zero_grad()
            
            # 只对track loss反向传播（diffusion已经由ImagenTrainer处理）
            track_loss.backward()
            
            encoder_optimizer.step()
            track_optimizer.step()
            
            # 记录损失
            total_loss = diffusion_loss + track_loss
            train_track_losses.append(track_loss.item())
            train_diffusion_losses.append(diffusion_loss.item())
            train_total_losses.append(total_loss.item())
            
            # 更新进度条
            pbar.set_postfix({
                'track_loss': f'{track_loss.item():.4f}',
                'diff_loss': f'{diffusion_loss.item():.4f}',
                'total_loss': f'{total_loss.item():.4f}'
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
        print(f"  - Diffusion Loss: {avg_train_diff:.4f}")
        print(f"  - Total Loss: {avg_train_total:.4f}")
        
        # ========== 验证阶段 ==========
        if val_loader is not None:
            model.eval()
            val_track_losses = []
            val_diffusion_losses = []
            val_total_losses = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                    past_frames = batch['past_frames'].to(device)
                    future_frames = batch['future_frames'].to(device)
                    track_past = batch['track_past'].to(device)
                    track_future = batch['track_future'].to(device)
                    
                    # Forward pass
                    _, last_state = model.convlstm_encoder(past_frames)
                    convlstm_features = last_state[-1][0]
                    
                    # Track prediction
                    track_input = convlstm_features.reshape(past_frames.shape[0], -1)
                    predicted_track = model.track_predictor(track_input)
                    track_loss = F.mse_loss(predicted_track, track_future)
                    
                    # 记录验证损失（diffusion在验证时不计算）
                    val_track_losses.append(track_loss.item())
                    val_diffusion_losses.append(0.0)  # 占位符
                    val_total_losses.append(track_loss.item())
            
            avg_val_track = np.mean(val_track_losses)
            avg_val_diff = np.mean(val_diffusion_losses)
            avg_val_total = np.mean(val_total_losses)
            
            history['val_track_loss'].append(avg_val_track)
            history['val_diffusion_loss'].append(avg_val_diff)
            history['val_total_loss'].append(avg_val_total)
            
            print(f"[VAL] Epoch {epoch+1}:")
            print(f"  - Track Loss: {avg_val_track:.4f}")
            print(f"  - Total Loss: {avg_val_total:.4f}")
            
            # 保存最佳模型
            if avg_val_total < best_val_loss:
                best_val_loss = avg_val_total
                best_model_path = os.path.join(save_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                    'track_optimizer_state_dict': track_optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'history': history
                }, best_model_path)
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
    print("Hybrid Model Training - Small Dataset Test")
    print("="*80)
    
    # 配置
    DATA_DIR = "D:/typhoon_data_2018_2021_full/train/cases"  # 训练数据目录（.npz文件在cases子目录）
    MAX_TRAIN_SAMPLES = 3  # 只用3个样本测试
    MAX_VAL_SAMPLES = 1    # 只用1个样本验证
    BATCH_SIZE = 1
    NUM_EPOCHS = 5         # 少量epoch测试
    LR = 3e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_DIR = 'checkpoints_test'
    
    print(f"\n[CONFIG]")
    print(f"  - Data directory: {DATA_DIR}")
    print(f"  - Max train samples: {MAX_TRAIN_SAMPLES}")
    print(f"  - Max val samples: {MAX_VAL_SAMPLES}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Learning rate: {LR}")
    print(f"  - Device: {DEVICE}")
    
    # 检查数据目录
    if not os.path.exists(DATA_DIR):
        print(f"\n[ERROR] Data directory not found: {DATA_DIR}")
        print(f"[INFO] Please make sure you have generated data in this directory.")
        return
    
    # 创建数据集
    print(f"\n{'='*80}")
    print("Loading data...")
    print(f"{'='*80}")
    
    try:
        train_dataset = TyphoonDataset(DATA_DIR, max_samples=MAX_TRAIN_SAMPLES)
        # 从train中分出val（简化处理）
        val_dataset = TyphoonDataset(DATA_DIR, max_samples=MAX_VAL_SAMPLES + MAX_TRAIN_SAMPLES)
        val_dataset.samples = val_dataset.samples[MAX_TRAIN_SAMPLES:]
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0  # Windows建议设为0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        print(f"[OK] Data loaded successfully!")
        
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建模型
    print(f"\n{'='*80}")
    print("Creating model...")
    print(f"{'='*80}")
    
    try:
        model = HybridTyphoonPredictor_V2(
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
    print(f"\n{'='*80}")
    print("Starting training...")
    print(f"{'='*80}")
    
    try:
        history = train_hybrid_model(
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

