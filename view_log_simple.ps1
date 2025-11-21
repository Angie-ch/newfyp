# Simple script to view log file
$logFile = Get-ChildItem "regeneration_log_*.txt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($logFile) {
    Write-Host "Opening log file: $($logFile.Name)" -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop`n" -ForegroundColor Gray
    
    # Show last 30 lines and then watch for updates
    Get-Content $logFile.FullName -Wait -Tail 30
} else {
    Write-Host "No log file found!" -ForegroundColor Red
}

