# Real-time training log viewer
Write-Host "="*80
Write-Host "AUTOENCODER TRAINING - REAL-TIME LOGS"
Write-Host "="*80
Write-Host "Monitoring training output..."
Write-Host "Press Ctrl+C to stop"
Write-Host ""

# Start training in background and capture output
$logFile = "training_live.log"
$job = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    python train_autoencoder.py --config configs/autoencoder_config.yaml 2>&1 | Tee-Object -FilePath $using:logFile
}

Write-Host "Training started (Job ID: $($job.Id))"
Write-Host "Showing real-time output:"
Write-Host ""

# Monitor log file in real-time
try {
    if (Test-Path $logFile) {
        Get-Content $logFile -Wait -Tail 30
    } else {
        # Wait for log file to be created
        $timeout = 30
        $elapsed = 0
        while (-not (Test-Path $logFile) -and $elapsed -lt $timeout) {
            Start-Sleep -Seconds 1
            $elapsed++
        }
        if (Test-Path $logFile) {
            Get-Content $logFile -Wait -Tail 30
        } else {
            Write-Host "Log file not created. Checking job status..."
            Receive-Job $job
        }
    }
} finally {
    Stop-Job $job -ErrorAction SilentlyContinue
    Remove-Job $job -ErrorAction SilentlyContinue
}











