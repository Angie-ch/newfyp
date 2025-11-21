# Restart regeneration with optimized memory usage
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "regeneration_log_$timestamp.txt"

Write-Host "========================================"
Write-Host "RESTARTING REGENERATION"
Write-Host "========================================"
Write-Host "Log file: $logFile"
Write-Host ""

# Stop any existing processes
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Clean output directory
if (Test-Path "data/processed_temporal_split") {
    Remove-Item "data/processed_temporal_split" -Recurse -Force -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Path "data/processed_temporal_split" -Force | Out-Null
Write-Host "[OK] Created fresh output directory"
Write-Host ""

# Activate virtual environment and run
. .\pytorch_gpu\Scripts\Activate.ps1
$env:PYTHONUNBUFFERED = "1"

Write-Host "Starting regeneration..."
Write-Host ""

python -u data/generate_data_by_year.py 2>&1 | Tee-Object -FilePath $logFile

Write-Host ""
Write-Host "Regeneration completed. Check log: $logFile"

