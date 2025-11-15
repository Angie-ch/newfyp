# Fix SSH Configuration - MUST RUN AS ADMINISTRATOR
# This script will:
# 1. Change SSH port from 2222 to 22
# 2. Enable password authentication
# 3. Restart SSH service
# 4. Verify the changes

Write-Host "=== SSH Configuration Fix ===" -ForegroundColor Cyan
Write-Host ""

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

$configPath = "C:\ProgramData\ssh\sshd_config"

# Backup original config
$backupPath = "$configPath.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
Copy-Item $configPath $backupPath -Force
Write-Host "Backed up config to: $backupPath" -ForegroundColor Green

# Read current config
$config = Get-Content $configPath

# Fix port (change 2222 to 22)
Write-Host "Changing SSH port from 2222 to 22..." -ForegroundColor Yellow
$config = $config -replace '^Port 2222', 'Port 22'
$config = $config -replace '^Port\s+2222', 'Port 22'

# Enable password authentication
Write-Host "Enabling password authentication..." -ForegroundColor Yellow
$config = $config -replace '^#PasswordAuthentication yes', 'PasswordAuthentication yes'
if ($config -notmatch '^PasswordAuthentication') {
    # Add if not present
    $config = $config -replace '^#PasswordAuthentication yes', "PasswordAuthentication yes`n#PasswordAuthentication yes"
}

# Save config
$config | Set-Content $configPath -Force
Write-Host "Configuration updated!" -ForegroundColor Green

# Verify changes
Write-Host "`nVerifying configuration..." -ForegroundColor Yellow
$portLine = Get-Content $configPath | Select-String -Pattern '^Port'
$passwordLine = Get-Content $configPath | Select-String -Pattern '^PasswordAuthentication'
Write-Host "  Port setting: $portLine" -ForegroundColor Cyan
Write-Host "  Password auth: $passwordLine" -ForegroundColor Cyan

# Restart SSH service
Write-Host "`nRestarting SSH service..." -ForegroundColor Yellow
Restart-Service sshd -Force
Start-Sleep -Seconds 3

# Check service status
$service = Get-Service sshd
Write-Host "SSH Service Status: $($service.Status)" -ForegroundColor $(if ($service.Status -eq 'Running') {'Green'} else {'Red'})

# Check if port 22 is listening
Write-Host "`nChecking ports..." -ForegroundColor Yellow
Start-Sleep -Seconds 2
$port22 = netstat -an | findstr ":22 " | findstr "LISTENING"
if ($port22) {
    Write-Host "SUCCESS: SSH is listening on port 22!" -ForegroundColor Green
    netstat -an | findstr ":22 " | findstr "LISTENING"
} else {
    Write-Host "WARNING: Port 22 not yet listening. May need a moment..." -ForegroundColor Yellow
    Write-Host "Current listening ports:" -ForegroundColor Yellow
    netstat -an | findstr "LISTENING" | findstr ":22"
}

# Get IP address
$ipAddress = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.InterfaceAlias -like "Ethernet*" -or $_.InterfaceAlias -like "Wi-Fi*"}).IPAddress | Select-Object -First 1

Write-Host "`n=== Connection Information ===" -ForegroundColor Cyan
Write-Host "IP Address: $ipAddress" -ForegroundColor Green
Write-Host "Username: $env:USERNAME" -ForegroundColor Green
Write-Host "Port: 22" -ForegroundColor Green
Write-Host ""
Write-Host "Connect from another machine:" -ForegroundColor Yellow
Write-Host "  ssh $env:USERNAME@$ipAddress" -ForegroundColor White
Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green

