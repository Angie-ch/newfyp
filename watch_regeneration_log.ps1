# Watch regeneration log in real-time
param(
    [int]$Tail = 20  # Number of lines to show initially
)

$logFile = Get-ChildItem "regeneration_log_*.txt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if (-not $logFile) {
    Write-Host "No log file found!" -ForegroundColor Red
    exit
}

Write-Host "=== WATCHING REGENERATION LOG ===" -ForegroundColor Cyan
Write-Host "File: $($logFile.Name)" -ForegroundColor White
Write-Host "Press Ctrl+C to stop watching`n" -ForegroundColor Gray
Write-Host ("=" * 80) -ForegroundColor Gray

# Show last N lines first
$lines = Get-Content $logFile.FullName -Tail $Tail -ErrorAction SilentlyContinue
if ($lines) {
    $lines | ForEach-Object { Write-Host $_ }
}

Write-Host ("`n" + ("=" * 80)) -ForegroundColor Gray
Write-Host "Waiting for new log entries...`n" -ForegroundColor Yellow

# Watch for new lines
Get-Content $logFile.FullName -Wait -Tail 0

