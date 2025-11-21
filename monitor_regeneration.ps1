# Monitor regeneration process and verify no synthetic data
param(
    [int]$CheckInterval = 30  # Check every 30 seconds
)

Write-Host "=== REGENERATION MONITOR ===" -ForegroundColor Cyan
Write-Host "Monitoring for synthetic data usage..." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop monitoring`n" -ForegroundColor Gray

$logFile = Get-ChildItem "regeneration_log_*.txt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if (-not $logFile) {
    Write-Host "No log file found. Waiting for regeneration to start..." -ForegroundColor Yellow
    while (-not $logFile) {
        Start-Sleep -Seconds 5
        $logFile = Get-ChildItem "regeneration_log_*.txt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    }
    Write-Host "Log file found: $($logFile.Name)" -ForegroundColor Green
}

$lastCheck = 0
$syntheticFound = $false

while ($true) {
    $proc = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.WorkingSet64 -gt 10MB } | Sort-Object StartTime -Descending | Select-Object -First 1
    
    if ($proc) {
        $logFile = Get-ChildItem "regeneration_log_*.txt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($logFile) {
            $content = Get-Content $logFile.FullName -Raw -ErrorAction SilentlyContinue
            if ($content) {
                $lines = ($content -split "`n").Count
                $syntheticCount = ($content | Select-String -Pattern "synthetic|Synthetic|SYNTHETIC" -AllMatches).Matches.Count
                
                if ($syntheticCount -gt 0 -and -not $syntheticFound) {
                    $syntheticFound = $true
                    Write-Host "`n[WARNING] SYNTHETIC DATA DETECTED!" -ForegroundColor Red -BackgroundColor Black
                    $content | Select-String -Pattern "synthetic|Synthetic" -Context 2 | Select-Object -First 5
                }
                
                if (-not $syntheticFound) {
                    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Process running | Log: $lines lines | Synthetic refs: $syntheticCount | Memory: $([math]::Round($proc.WorkingSet64/1GB,2)) GB" -ForegroundColor Green
                }
            }
        }
    } else {
        Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] Process completed or stopped" -ForegroundColor Yellow
        break
    }
    
    Start-Sleep -Seconds $CheckInterval
}

if (-not $syntheticFound) {
    Write-Host "`n=== FINAL CHECK ===" -ForegroundColor Cyan
    Write-Host "No synthetic data references found in log!" -ForegroundColor Green
} else {
    Write-Host "`n=== WARNING ===" -ForegroundColor Red
    Write-Host "Synthetic data was detected in the log!" -ForegroundColor Red
}

