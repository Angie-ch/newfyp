# Run this script as Administrator to set SSH to port 2222

Write-Host "Setting SSH to port 2222..." -ForegroundColor Green

$configPath = "C:\ProgramData\ssh\sshd_config"

# Read current config
$content = Get-Content $configPath -Raw

# Check if Port is already set
if ($content -match "(?m)^Port\s+(\d+)") {
    $currentPort = $matches[1]
    Write-Host "Current port is: $currentPort" -ForegroundColor Yellow
    if ($currentPort -eq "2222") {
        Write-Host "Port is already 2222!" -ForegroundColor Green
    } else {
        # Replace existing Port line
        $content = $content -replace "(?m)^Port\s+\d+", "Port 2222"
        Write-Host "Changed port from $currentPort to 2222" -ForegroundColor Green
    }
} else {
    # Add Port 2222 (uncomment and set)
    $content = $content -replace "(?m)^#Port\s+22", "Port 2222"
    Write-Host "Set port to 2222" -ForegroundColor Green
}

# Also ensure PubkeyAuthentication is enabled
$content = $content -replace "(?m)^#PubkeyAuthentication\s+yes", "PubkeyAuthentication yes"
$content = $content -replace "(?m)^#PubkeyAuthentication\s+no", "PubkeyAuthentication yes"

# Write back
Set-Content -Path $configPath -Value $content -NoNewline

Write-Host "`nAdding firewall rule for port 2222..." -ForegroundColor Green

# Add firewall rule for port 2222
New-NetFirewallRule -Name sshd-2222 -DisplayName 'OpenSSH Server (port 2222)' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 2222 -ErrorAction SilentlyContinue

Write-Host "Restarting SSH service..." -ForegroundColor Green

# Restart SSH service
Restart-Service sshd

Write-Host "`n=== DONE! ===" -ForegroundColor Green
Write-Host "SSH is now running on port 2222" -ForegroundColor Cyan
Write-Host "`nConnect from your Mac with:" -ForegroundColor Yellow
Write-Host "  ssh -p 2222 fyp@10.119.178.85" -ForegroundColor Cyan
Write-Host "`nOr in Cursor/VS Code Remote-SSH:" -ForegroundColor Yellow
Write-Host "  ssh -p 2222 fyp@10.119.178.85" -ForegroundColor Cyan

