# Setup SSH Server on Windows for Remote Access
# Run this script as Administrator

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SSH Server Setup for Remote Access" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

Write-Host "Step 1: Installing OpenSSH Server..." -ForegroundColor Yellow
# Install OpenSSH Server
$sshServer = Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'
if ($sshServer.State -ne 'Installed') {
    Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
    Write-Host "OpenSSH Server installed successfully" -ForegroundColor Green
} else {
    Write-Host "OpenSSH Server is already installed" -ForegroundColor Green
}

Write-Host ""
Write-Host "Step 2: Starting SSH Service..." -ForegroundColor Yellow
# Start and configure SSH service
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'
Write-Host "SSH Service started and set to auto-start" -ForegroundColor Green

Write-Host ""
Write-Host "Step 3: Configuring Windows Firewall..." -ForegroundColor Yellow
# Configure firewall rule for SSH
$firewallRule = Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue
if (-not $firewallRule) {
    New-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -DisplayName "OpenSSH SSH Server (sshd)" -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
    Write-Host "Firewall rule created for port 22" -ForegroundColor Green
} else {
    Write-Host "Firewall rule already exists" -ForegroundColor Green
    Enable-NetFirewallRule -Name "OpenSSH-Server-In-TCP"
}

Write-Host ""
Write-Host "Step 4: Getting connection information..." -ForegroundColor Yellow
$computerName = $env:COMPUTERNAME
$username = $env:USERNAME
$ipAddress = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*"}).IPAddress | Select-Object -First 1

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "SSH Server Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Connection Information:" -ForegroundColor Cyan
Write-Host "  Computer Name: $computerName" -ForegroundColor White
Write-Host "  Username: $username" -ForegroundColor White
Write-Host "  IP Address: $ipAddress" -ForegroundColor White
Write-Host "  Port: 22" -ForegroundColor White
Write-Host ""
Write-Host "To connect from Cursor (Mac):" -ForegroundColor Yellow
Write-Host "  ssh $username@$ipAddress" -ForegroundColor White
Write-Host "  OR" -ForegroundColor White
Write-Host "  ssh $username@$computerName" -ForegroundColor White
Write-Host ""
Write-Host "Note: You may need to configure port forwarding if behind a router" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Green

