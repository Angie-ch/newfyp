# Run this script as Administrator to enable SSH Server on Windows

Write-Host "Installing OpenSSH Server..." -ForegroundColor Green

# Install OpenSSH Server
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0

Write-Host "Starting SSH service..." -ForegroundColor Green

# Start the SSH service
Start-Service sshd

# Set SSH service to start automatically
Set-Service -Name sshd -StartupType 'Automatic'

Write-Host "Configuring Windows Firewall..." -ForegroundColor Green

# Allow SSH through Windows Firewall
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server (sshd)' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 -ErrorAction SilentlyContinue

Write-Host "`nSSH Server is now enabled!" -ForegroundColor Green
Write-Host "Your connection details:" -ForegroundColor Yellow
Write-Host "  Username: fyp" -ForegroundColor Cyan
Write-Host "  IP Address: 10.119.178.85" -ForegroundColor Cyan
Write-Host "  Connection: ssh fyp@10.119.178.85" -ForegroundColor Cyan

