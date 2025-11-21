# Train joint autoencoder with real-time log file
cd $PSScriptRoot

$logFile = "training_joint_autoencoder_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Write-Host "Starting training with logs saved to: $logFile"
Write-Host "To view real-time logs, run in another terminal:"
Write-Host "  Get-Content $logFile -Wait -Tail 50"
Write-Host ""

.\pytorch_gpu\Scripts\python.exe train_joint_pipeline.py `
    --stage autoencoder `
    --config configs/joint_autoencoder.yaml `
    --device cuda `
    2>&1 | Tee-Object -FilePath $logFile

