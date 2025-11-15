# Fix SSH Port Configuration
# Run as Administrator

Write-Host "Fixing SSH configuration..." -ForegroundColor Yellow

# Restart SSH service to apply config changes
Write-Host "Restarting SSH service..." -ForegroundColor Yellow
Restart-Service sshd -Force
Start-Sleep -Seconds 3

# Check if port 22 is now listening
$port22 = netstat -an | findstr ":22 " | findstr "LISTENING"
if ($port22) {
    Write-Host "SUCCESS: SSH is now listening on port 22!" -ForegroundColor Green
    netstat -an | findstr ":22 " | findstr "LISTENING"
} else {
    Write-Host "WARNING: Port 22 still not listening. Checking configuration..." -ForegroundColor Yellow
    Get-Content "C:\ProgramData\ssh\sshd_config" | Select-String -Pattern "^Port"
}

Write-Host "`nConnection command:" -ForegroundColor Cyan
Write-Host "  ssh fyp@10.119.178.85" -ForegroundColor White

