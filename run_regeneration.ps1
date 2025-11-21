# Run regeneration with unbuffered output
$env:PYTHONUNBUFFERED = "1"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "regeneration_log_$timestamp.txt"

Write-Host "Starting regeneration..."
Write-Host "Log file: $logFile"
Write-Host ""

. .\pytorch_gpu\Scripts\Activate.ps1
python -u data/generate_data_by_year.py 2>&1 | Tee-Object -FilePath $logFile

