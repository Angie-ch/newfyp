# View training logs in real-time
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "AUTOENCODER TRAINING - REAL-TIME LOG VIEWER" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

# Check if training is running
$trainingProcess = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*train_autoencoder*" -or 
    (Get-WmiObject Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine -like "*train_autoencoder*"
}

if ($trainingProcess) {
    Write-Host "✓ Training process detected (PID: $($trainingProcess.Id))" -ForegroundColor Green
} else {
    Write-Host "⚠ Training process not detected. It may have finished or not started yet." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Monitoring log files..." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

# Monitor multiple log sources
$logFiles = @(
    "training_output.log",
    "autoencoder_training.log",
    "training_autoencoder.log"
)

$foundLog = $false
foreach ($logFile in $logFiles) {
    if (Test-Path $logFile) {
        Write-Host "Found log file: $logFile" -ForegroundColor Green
        Write-Host "Showing last 50 lines, then following in real-time..." -ForegroundColor Green
        Write-Host ""
        Get-Content $logFile -Tail 50
        Write-Host ""
        Write-Host "="*80 -ForegroundColor Cyan
        Write-Host "Following new log entries (real-time)..." -ForegroundColor Cyan
        Write-Host "="*80 -ForegroundColor Cyan
        Get-Content $logFile -Wait -Tail 10
        $foundLog = $true
        break
    }
}

if (-not $foundLog) {
    Write-Host "No log files found. Checking TensorBoard logs..." -ForegroundColor Yellow
    
    # Check TensorBoard logs
    $tbLogDir = "logs/autoencoder"
    if (Test-Path $tbLogDir) {
        $latestLog = Get-ChildItem $tbLogDir -Filter "events.out.tfevents.*" | 
            Sort-Object LastWriteTime -Descending | 
            Select-Object -First 1
        
        if ($latestLog) {
            Write-Host "Latest TensorBoard log: $($latestLog.Name)" -ForegroundColor Green
            Write-Host "Last modified: $($latestLog.LastWriteTime)" -ForegroundColor Green
            Write-Host ""
            Write-Host "To view TensorBoard logs, run:" -ForegroundColor Yellow
            Write-Host "  tensorboard --logdir logs/autoencoder" -ForegroundColor Cyan
        }
    }
    
    Write-Host ""
    Write-Host "Starting training with output capture..." -ForegroundColor Yellow
    Write-Host "Run this command to see real-time output:" -ForegroundColor Yellow
    $cmd = ".\pytorch_gpu\Scripts\Activate.ps1; python train_autoencoder.py --config configs/autoencoder_config.yaml"
    Write-Host "  $cmd" -ForegroundColor Cyan
}

