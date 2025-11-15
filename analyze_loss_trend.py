import re
from pathlib import Path

log_file = Path('training_autoencoder.log')
if not log_file.exists():
    print("Log file not found!")
    exit(1)

content = log_file.read_text()

# Extract epoch losses - they're on separate lines
epochs = {}
lines = content.split('\n')
current_epoch = None

for i, line in enumerate(lines):
    # Find epoch markers
    epoch_match = re.search(r'Epoch (\d+)/50', line)
    if epoch_match:
        current_epoch = int(epoch_match.group(1))
        epochs[current_epoch] = {}
    
    # Find train loss
    train_match = re.search(r'Train Loss: ([\d.]+)', line)
    if train_match and current_epoch:
        epochs[current_epoch]['train'] = float(train_match.group(1))
    
    # Find val loss
    val_match = re.search(r'Val Loss: ([\d.]+)', line)
    if val_match and current_epoch:
        epochs[current_epoch]['val'] = float(val_match.group(1))

if not epochs:
    print("No epoch data found!")
    exit(1)

matches = [(str(ep), str(epochs[ep]['train']), str(epochs[ep]['val'])) 
           for ep in sorted(epochs.keys()) if 'train' in epochs[ep] and 'val' in epochs[ep]]

print("=" * 70)
print("Loss Trend by Epoch")
print("=" * 70)
print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Status':<15}")
print("-" * 70)

best_val_loss = float('inf')
for epoch, train_loss, val_loss in matches:
    epoch_num = int(epoch)
    train = float(train_loss)
    val = float(val_loss)
    
    status = ""
    if val < best_val_loss:
        status = "NEW BEST ✓"
        best_val_loss = val
    else:
        status = ""
    
    print(f"{epoch_num:<8} {train:<15.2f} {val:<15.2f} {status:<15}")

print("=" * 70)
print(f"\nBest Validation Loss: {best_val_loss:.2f}")
print(f"Total Epochs Completed: {len(matches)}")

# Calculate improvement
if len(matches) > 1:
    first_train = float(matches[0][1])
    last_train = float(matches[-1][1])
    first_val = float(matches[0][2])
    last_val = float(matches[-1][2])
    
    train_improvement = ((first_train - last_train) / first_train) * 100
    val_improvement = ((first_val - last_val) / first_val) * 100 if first_val > 0 else 0
    
    print(f"\nImprovement:")
    print(f"  Train Loss: {first_train:.2f} → {last_train:.2f} ({train_improvement:+.1f}%)")
    print(f"  Val Loss:   {first_val:.2f} → {last_val:.2f} ({val_improvement:+.1f}%)")

