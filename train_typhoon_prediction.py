"""
Simple Typhoon Trajectory Prediction Model
Train on existing 72 samples to verify the pipeline works
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import json

print("=" * 80)
print("TYPHOON TRAJECTORY PREDICTION - TRAINING")
print("=" * 80)

# Configuration
DATA_DIR = Path("D:/typhoon_data_2018_2021_full")
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\nDevice: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")

# Dataset
class TyphoonDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir) / split / 'cases'
        self.samples = sorted(list(self.data_dir.glob('*.npz')))
        print(f"  Loaded {len(self.samples)} {split} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data = np.load(self.samples[idx])
        
        # Extract data
        past_frames = torch.FloatTensor(data['past_frames'])  # (8, 24, 64, 64)
        future_frames = torch.FloatTensor(data['future_frames'])  # (12, 24, 64, 64)
        track_past = torch.FloatTensor(data['track_past'])  # (8, 2)
        track_future = torch.FloatTensor(data['track_future'])  # (12, 2)
        
        return {
            'past_frames': past_frames,
            'future_frames': future_frames,
            'track_past': track_past,
            'track_future': track_future,
        }

# Simple ConvLSTM Cell
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )
    
    def forward(self, x, hidden_state):
        h, c = hidden_state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

# Simple Encoder-Decoder Model
class TyphoonPredictor(nn.Module):
    def __init__(self, input_channels=24, hidden_channels=64, output_channels=24):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, hidden_channels, 3, padding=1),
            nn.ReLU(),
        )
        
        self.encoder_lstm = ConvLSTMCell(hidden_channels, hidden_channels)
        
        # Decoder
        self.decoder_lstm = ConvLSTMCell(hidden_channels, hidden_channels)
        
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, output_channels, 3, padding=1),
        )
        
        # Track predictor (simple MLP)
        self.track_encoder = nn.Sequential(
            nn.Linear(8 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        
        self.track_decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 12 * 2),
        )
    
    def forward(self, past_frames, track_past):
        B, T_in, C, H, W = past_frames.shape
        T_out = 12
        
        # Encode past frames
        h = torch.zeros(B, self.hidden_channels, H, W, device=past_frames.device)
        c = torch.zeros(B, self.hidden_channels, H, W, device=past_frames.device)
        
        for t in range(T_in):
            x_t = self.encoder_conv(past_frames[:, t])
            h, c = self.encoder_lstm(x_t, (h, c))
        
        # Decode future frames
        future_frames = []
        x_t = h  # Start with last encoded state
        
        for t in range(T_out):
            h, c = self.decoder_lstm(x_t, (h, c))
            frame_t = self.decoder_conv(h)
            future_frames.append(frame_t)
            x_t = h
        
        future_frames = torch.stack(future_frames, dim=1)  # (B, T_out, C, H, W)
        
        # Predict track
        track_past_flat = track_past.reshape(B, -1)
        track_features = self.track_encoder(track_past_flat)
        track_future_flat = self.track_decoder(track_features)
        track_future = track_future_flat.reshape(B, T_out, 2)
        
        return future_frames, track_future

# Training function
def train_epoch(model, dataloader, optimizer, criterion_frame, criterion_track):
    model.train()
    total_loss = 0
    total_frame_loss = 0
    total_track_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        past_frames = batch['past_frames'].to(DEVICE)
        future_frames_gt = batch['future_frames'].to(DEVICE)
        track_past = batch['track_past'].to(DEVICE)
        track_future_gt = batch['track_future'].to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward
        future_frames_pred, track_future_pred = model(past_frames, track_past)
        
        # Loss
        frame_loss = criterion_frame(future_frames_pred, future_frames_gt)
        track_loss = criterion_track(track_future_pred, track_future_gt)
        loss = frame_loss + 10.0 * track_loss  # Weight track loss higher
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_frame_loss += frame_loss.item()
        total_track_loss += track_loss.item()
    
    n = len(dataloader)
    return total_loss / n, total_frame_loss / n, total_track_loss / n

# Validation function
def validate(model, dataloader, criterion_frame, criterion_track):
    model.eval()
    total_loss = 0
    total_frame_loss = 0
    total_track_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            past_frames = batch['past_frames'].to(DEVICE)
            future_frames_gt = batch['future_frames'].to(DEVICE)
            track_past = batch['track_past'].to(DEVICE)
            track_future_gt = batch['track_future'].to(DEVICE)
            
            # Forward
            future_frames_pred, track_future_pred = model(past_frames, track_past)
            
            # Loss
            frame_loss = criterion_frame(future_frames_pred, future_frames_gt)
            track_loss = criterion_track(track_future_pred, track_future_gt)
            loss = frame_loss + 10.0 * track_loss
            
            total_loss += loss.item()
            total_frame_loss += frame_loss.item()
            total_track_loss += track_loss.item()
    
    n = len(dataloader)
    return total_loss / n, total_frame_loss / n, total_track_loss / n

# Main
def main():
    # Load datasets
    print("\n[1] Loading datasets...")
    train_dataset = TyphoonDataset(DATA_DIR, 'train')
    val_dataset = TyphoonDataset(DATA_DIR, 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    print("\n[2] Creating model...")
    model = TyphoonPredictor(input_channels=24, hidden_channels=64).to(DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_frame = nn.MSELoss()
    criterion_track = nn.MSELoss()
    
    # Training loop
    print("\n[3] Training...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'train_frame_loss': [], 'train_track_loss': [],
        'val_loss': [], 'val_frame_loss': [], 'val_track_loss': []
    }
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train
        train_loss, train_frame, train_track = train_epoch(
            model, train_loader, optimizer, criterion_frame, criterion_track
        )
        
        # Validate
        val_loss, val_frame, val_track = validate(
            model, val_loader, criterion_frame, criterion_track
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_frame_loss'].append(train_frame)
        history['train_track_loss'].append(train_track)
        history['val_loss'].append(val_loss)
        history['val_frame_loss'].append(val_frame)
        history['val_track_loss'].append(val_track)
        
        # Print
        print(f"  Train Loss: {train_loss:.4f} (Frame: {train_frame:.4f}, Track: {train_track:.4f})")
        print(f"  Val Loss:   {val_loss:.4f} (Frame: {val_frame:.4f}, Track: {val_track:.4f})")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_typhoon_model.pt')
            print(f"  [BEST] Saved model (val_loss: {val_loss:.4f})")
    
    # Save history
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: best_typhoon_model.pt")
    print(f"History saved to: training_history.json")

if __name__ == '__main__':
    main()

