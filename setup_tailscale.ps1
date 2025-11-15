# Setup Tailscale for SSH Remote Access
# This creates a permanent VPN solution

Write-Host "=== Tailscale Setup for SSH Remote Access ===" -ForegroundColor Cyan
Write-Host ""

# Check if Tailscale is installed
$tailscaleInstalled = Get-Command tailscale -ErrorAction SilentlyContinue

if (-not $tailscaleInstalled) {
    Write-Host "Tailscale not found. Installing..." -ForegroundColor Yellow
    
    # Try winget first
    $wingetInstalled = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetInstalled) {
        Write-Host "Installing Tailscale using winget..." -ForegroundColor Yellow
        winget install Tailscale.Tailscale
    } else {
        Write-Host "winget not found. Please install Tailscale manually:" -ForegroundColor Red
        Write-Host "  1. Go to: https://tailscale.com/download" -ForegroundColor Yellow
        Write-Host "  2. Download Windows version" -ForegroundColor Yellow
        Write-Host "  3. Install and run" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "Tailscale is already installed!" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Next Steps ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Open Tailscale app (search 'Tailscale' in Start menu)" -ForegroundColor Yellow
Write-Host "2. Sign up or login with your account" -ForegroundColor Yellow
Write-Host "3. Note your Tailscale IP address (shown in the app, e.g., 100.x.x.x)" -ForegroundColor Yellow
Write-Host ""
Write-Host "4. On your Mac:" -ForegroundColor Cyan
Write-Host "   - Install Tailscale: brew install tailscale" -ForegroundColor White
Write-Host "   - Or download from: https://tailscale.com/download" -ForegroundColor White
Write-Host "   - Sign in with the SAME account" -ForegroundColor White
Write-Host ""
Write-Host "5. Connect from Mac using Tailscale IP:" -ForegroundColor Cyan
Write-Host "   ssh fyp@<tailscale-ip>" -ForegroundColor White
Write-Host ""
Write-Host "=== Benefits ===" -ForegroundColor Green
Write-Host "  - Permanent IP address (doesn't change)" -ForegroundColor White
Write-Host "  - Works from anywhere (internet, different networks)" -ForegroundColor White
Write-Host "  - Secure VPN connection" -ForegroundColor White
Write-Host "  - Free for personal use" -ForegroundColor White
Write-Host ""

