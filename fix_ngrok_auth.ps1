# Fix ngrok Authentication
# This script helps you add your ngrok authtoken

Write-Host "=== ngrok Authentication Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check if authtoken is already configured
Write-Host "Checking current ngrok configuration..." -ForegroundColor Yellow
$configCheck = ngrok config check 2>&1

if ($configCheck -match "authtoken") {
    Write-Host "✅ Authtoken appears to be configured" -ForegroundColor Green
} else {
    Write-Host "❌ Authtoken not found or invalid" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== How to Get Your Authtoken ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Go to: https://dashboard.ngrok.com/get-started/your-authtoken" -ForegroundColor Yellow
Write-Host "2. Sign up or login to your ngrok account" -ForegroundColor Yellow
Write-Host "3. Copy your authtoken (long string)" -ForegroundColor Yellow
Write-Host ""
Write-Host "=== Add Authtoken ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Once you have your authtoken, run:" -ForegroundColor Yellow
Write-Host "  ngrok config add-authtoken YOUR_TOKEN_HERE" -ForegroundColor White
Write-Host ""
Write-Host "Example:" -ForegroundColor Cyan
Write-Host "  ngrok config add-authtoken 2abc123def456ghi789jkl012mno345pqr678stu901vwx234yz" -ForegroundColor Gray
Write-Host ""
Write-Host "=== After Adding Authtoken ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Verify: ngrok config check" -ForegroundColor Yellow
Write-Host "2. Start tunnel: ngrok tcp 22" -ForegroundColor Yellow
Write-Host ""
Write-Host "=== Alternative: Use Tailscale Instead ===" -ForegroundColor Green
Write-Host ""
Write-Host "Tailscale doesn't require authentication tokens and is more reliable:" -ForegroundColor Yellow
Write-Host "  winget install Tailscale.Tailscale" -ForegroundColor White
Write-Host ""
Write-Host "Then sign in on both Windows and Mac with the same account." -ForegroundColor Cyan
Write-Host ""

