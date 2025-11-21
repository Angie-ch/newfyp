# Monitor Manual DDPM Training

Write-Host "=== MANUAL DDPM TRAINING MONITOR ===" -ForegroundColor Cyan
Write-Host ""

$maxAttempts = 60
$attempt = 0

while ($attempt -lt $maxAttempts) {
    Clear-Host
    Write-Host "=== MANUAL DDPM TRAINING MONITOR ===" -ForegroundColor Cyan
    Write-Host "Attempt: $($attempt + 1)/$maxAttempts" -ForegroundColor Gray
    Write-Host ""
    
    # Check for training history
    if (Test-Path "checkpoints_manual_ddpm\training_history.json") {
        Write-Host "TRAINING PROGRESS:" -ForegroundColor Green
        Write-Host ""
        
        $history = Get-Content "checkpoints_manual_ddpm\training_history.json" | ConvertFrom-Json
        
        $epochs = $history.train_diffusion_loss.Count
        
        if ($epochs -gt 0) {
            Write-Host "Completed Epochs: $epochs" -ForegroundColor Yellow
            Write-Host ""
            
            # Show last epoch
            $lastIdx = $epochs - 1
            Write-Host "Latest Epoch Results:" -ForegroundColor Cyan
            Write-Host "  Train Diffusion Loss: $($history.train_diffusion_loss[$lastIdx])" -ForegroundColor White
            Write-Host "  Train Track Loss:     $($history.train_track_loss[$lastIdx])" -ForegroundColor White
            Write-Host "  Train Total Loss:     $($history.train_total_loss[$lastIdx])" -ForegroundColor White
            Write-Host ""
            Write-Host "  Val Diffusion Loss:   $($history.val_diffusion_loss[$lastIdx])" -ForegroundColor White
            Write-Host "  Val Track Loss:       $($history.val_track_loss[$lastIdx])" -ForegroundColor White
            Write-Host "  Val Total Loss:       $($history.val_total_loss[$lastIdx])" -ForegroundColor White
            Write-Host ""
            
            # Check if diffusion loss > 0
            if ($history.train_diffusion_loss[$lastIdx] -gt 0) {
                Write-Host "[SUCCESS] Diffusion Loss > 0! Manual DDPM is working!" -ForegroundColor Green
            } else {
                Write-Host "[WARNING] Diffusion Loss = 0, still debugging..." -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "Training initializing..." -ForegroundColor Yellow
        
        # Check for Python processes
        $pythonProc = Get-Process python -ErrorAction SilentlyContinue | Where-Object {$_.MainWindowTitle -eq ""}
        if ($pythonProc) {
            Write-Host "Python training process is running (PID: $($pythonProc.Id))" -ForegroundColor Green
        } else {
            Write-Host "No Python process found - checking if training finished..." -ForegroundColor Red
        }
    }
    
    Write-Host ""
    Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor Gray
    
    Start-Sleep -Seconds 10
    $attempt++
}

Write-Host ""
Write-Host "Monitoring finished." -ForegroundColor Cyan

