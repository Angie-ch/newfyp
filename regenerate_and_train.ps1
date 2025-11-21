# Script to regenerate dataset with real ERA5 and train autoencoder
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "REGENERATING DATASET WITH REAL ERA5 DATA" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\pytorch_gpu\Scripts\Activate.ps1"

# Step 1: Regenerate dataset
Write-Host ""
Write-Host "Step 1: Regenerating dataset with real ERA5 data..." -ForegroundColor Green
Write-Host "This may take 30-60 minutes depending on the number of storms..." -ForegroundColor Yellow
Write-Host ""

python data/generate_data_by_year.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Dataset regeneration failed!" -ForegroundColor Red
    exit 1
}

# Step 2: Verify real ERA5 was used
Write-Host ""
Write-Host "Step 2: Verifying real ERA5 data was used..." -ForegroundColor Green
$datasetInfo = Get-Content "data/processed_temporal_split/dataset_info.json" | ConvertFrom-Json

if ($datasetInfo.meteorological_data -eq "ERA5") {
    Write-Host "✓ SUCCESS: Dataset uses REAL ERA5 data!" -ForegroundColor Green
    Write-Host "  Total samples: $($datasetInfo.total_samples)" -ForegroundColor Green
} else {
    Write-Host "⚠ WARNING: Dataset still shows: $($datasetInfo.meteorological_data)" -ForegroundColor Yellow
    Write-Host "  This may indicate an issue. Continuing anyway..." -ForegroundColor Yellow
}

# Step 3: Start training
Write-Host ""
Write-Host "Step 3: Starting autoencoder training..." -ForegroundColor Green
Write-Host "Training will use the newly generated real ERA5 data" -ForegroundColor Yellow
Write-Host ""

python train_autoencoder.py --config configs/autoencoder_config.yaml

Write-Host ""
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "TRAINING COMPLETE!" -ForegroundColor Green
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Best model saved to: checkpoints/autoencoder/best.pth" -ForegroundColor Green
Write-Host "View logs: Get-Content autoencoder_training.log -Tail 50" -ForegroundColor Yellow











