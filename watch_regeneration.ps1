# Watch regeneration log in real-time
$logFile = Get-ChildItem regeneration_log_*.txt | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($logFile) {
    Write-Host "Watching log file: $($logFile.Name)" -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop watching" -ForegroundColor Yellow
    Write-Host "="*80
    Write-Host ""
    
    # Tail the file with updates
    Get-Content $logFile.FullName -Wait -Tail 30
} else {
    Write-Host "No regeneration log file found!" -ForegroundColor Red
}





