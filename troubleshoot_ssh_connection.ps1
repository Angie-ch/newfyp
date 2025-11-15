# Troubleshoot SSH Connection Issues
# Run as Administrator for full diagnostics

Write-Host "=== SSH Connection Troubleshooting ===" -ForegroundColor Cyan
Write-Host ""

# Check if running as admin
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "WARNING: Some checks require administrator privileges" -ForegroundColor Yellow
    Write-Host ""
}

# 1. Check SSH Service
Write-Host "1. SSH Service Status:" -ForegroundColor Yellow
$sshService = Get-Service sshd -ErrorAction SilentlyContinue
if ($sshService) {
    Write-Host "   Status: $($sshService.Status)" -ForegroundColor $(if ($sshService.Status -eq 'Running') {'Green'} else {'Red'})
    Write-Host "   StartType: $($sshService.StartType)" -ForegroundColor Cyan
} else {
    Write-Host "   ERROR: SSH service not found!" -ForegroundColor Red
}
Write-Host ""

# 2. Check Port Listening
Write-Host "2. Port 22 Listening Status:" -ForegroundColor Yellow
$port22 = netstat -an | findstr ":22 " | findstr "LISTENING"
if ($port22) {
    Write-Host "   Port 22 is LISTENING:" -ForegroundColor Green
    $port22 | ForEach-Object { Write-Host "     $_" -ForegroundColor White }
} else {
    Write-Host "   ERROR: Port 22 is NOT listening!" -ForegroundColor Red
}
Write-Host ""

# 3. Check Firewall Rules
Write-Host "3. Firewall Rules for Port 22:" -ForegroundColor Yellow
$firewallRules = Get-NetFirewallRule | Where-Object {
    ($_.DisplayName -like "*SSH*" -or $_.DisplayName -like "*22*") -or
    (Get-NetFirewallPortFilter -AssociatedNetFirewallRule $_ | Where-Object {$_.LocalPort -eq 22})
}
if ($firewallRules) {
    $firewallRules | ForEach-Object {
        $portFilter = Get-NetFirewallPortFilter -AssociatedNetFirewallRule $_
        $port = if ($portFilter.LocalPort) { $portFilter.LocalPort } else { "Any" }
        $status = if ($_.Enabled) { "ENABLED" } else { "DISABLED" }
        $color = if ($_.Enabled) { "Green" } else { "Red" }
        Write-Host "   $($_.DisplayName): $status (Port: $port, Profile: $($_.Profile))" -ForegroundColor $color
    }
} else {
    Write-Host "   WARNING: No firewall rules found for SSH!" -ForegroundColor Yellow
}
Write-Host ""

# 4. Check Network Interfaces
Write-Host "4. Network Interfaces:" -ForegroundColor Yellow
$interfaces = Get-NetIPAddress -AddressFamily IPv4 | Where-Object {
    $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*"
}
if ($interfaces) {
    $interfaces | ForEach-Object {
        Write-Host "   $($_.IPAddress) - $($_.InterfaceAlias)" -ForegroundColor Cyan
    }
} else {
    Write-Host "   No network interfaces found!" -ForegroundColor Red
}
Write-Host ""

# 5. Test Local Connection
Write-Host "5. Testing Local Connection:" -ForegroundColor Yellow
$localIP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object {
    $_.InterfaceAlias -like "Ethernet*" -or $_.InterfaceAlias -like "Wi-Fi*"
}).IPAddress | Select-Object -First 1

if ($localIP) {
    Write-Host "   Testing connection to $localIP:22..." -ForegroundColor Cyan
    $test = Test-NetConnection -ComputerName $localIP -Port 22 -InformationLevel Quiet -WarningAction SilentlyContinue
    if ($test) {
        Write-Host "   SUCCESS: Local connection works!" -ForegroundColor Green
    } else {
        Write-Host "   FAILED: Cannot connect locally!" -ForegroundColor Red
    }
} else {
    Write-Host "   Could not determine local IP address" -ForegroundColor Yellow
}
Write-Host ""

# 6. Check SSH Configuration
Write-Host "6. SSH Configuration:" -ForegroundColor Yellow
$configPath = "C:\ProgramData\ssh\sshd_config"
if (Test-Path $configPath) {
    $portLine = Get-Content $configPath | Select-String -Pattern "^Port"
    $passwordLine = Get-Content $configPath | Select-String -Pattern "^PasswordAuthentication"
    $listenLine = Get-Content $configPath | Select-String -Pattern "^ListenAddress"
    
    Write-Host "   Config file: $configPath" -ForegroundColor Cyan
    if ($portLine) { Write-Host "   $portLine" -ForegroundColor White }
    if ($passwordLine) { Write-Host "   $passwordLine" -ForegroundColor White }
    if ($listenLine) {
        Write-Host "   $listenLine" -ForegroundColor White
    } else {
        Write-Host "   ListenAddress: Not set (defaults to all interfaces)" -ForegroundColor Cyan
    }
} else {
    Write-Host "   ERROR: SSH config file not found!" -ForegroundColor Red
}
Write-Host ""

# 7. Recommendations
Write-Host "=== Recommendations ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "If connection still fails from remote machine:" -ForegroundColor Yellow
Write-Host "1. Check if both machines are on the same network" -ForegroundColor White
Write-Host "2. Check router firewall settings (may block port 22)" -ForegroundColor White
Write-Host "3. Try using Tailscale or ngrok for remote access:" -ForegroundColor White
Write-Host "   - Tailscale: VPN solution (recommended)" -ForegroundColor Cyan
Write-Host "   - ngrok: TCP tunnel (temporary solution)" -ForegroundColor Cyan
Write-Host "4. Verify Windows Firewall is not blocking:" -ForegroundColor White
Write-Host "   Get-NetFirewallRule | Where-Object {$_.DisplayName -like '*SSH*'}" -ForegroundColor Gray
Write-Host ""

