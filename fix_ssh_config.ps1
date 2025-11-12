# Run this as Administrator to fix SSH config

$configPath = "C:\ProgramData\ssh\sshd_config"

# Read current config
$content = Get-Content $configPath -Raw

# Uncomment PubkeyAuthentication
$content = $content -replace "#PubkeyAuthentication yes", "PubkeyAuthentication yes"
$content = $content -replace "#PubkeyAuthentication no", "PubkeyAuthentication yes"

# Ensure AuthorizedKeysFile is set correctly
if ($content -notmatch "AuthorizedKeysFile\s+\.ssh/authorized_keys") {
    $content = $content -replace "(?m)^#?\s*AuthorizedKeysFile.*", "AuthorizedKeysFile .ssh/authorized_keys"
}

# Write back
Set-Content -Path $configPath -Value $content -NoNewline

Write-Host "SSH config updated!" -ForegroundColor Green

# Restart SSH service
Restart-Service sshd

Write-Host "SSH service restarted!" -ForegroundColor Green
Write-Host "`nTry connecting again: ssh fyp@10.119.178.85" -ForegroundColor Cyan

