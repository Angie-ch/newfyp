# Fix ngrok config file

Write-Host "Fixing ngrok configuration..." -ForegroundColor Green

# Check both possible locations
$configPath1 = "$env:APPDATA\ngrok\ngrok.yml"  # Roaming
$configPath2 = "$env:LOCALAPPDATA\ngrok\ngrok.yml"  # Local

# Find which one exists
$configPath = $null
if (Test-Path $configPath2) {
    $configPath = $configPath2
    Write-Host "Found config in Local folder" -ForegroundColor Yellow
} elseif (Test-Path $configPath1) {
    $configPath = $configPath1
    Write-Host "Found config in Roaming folder" -ForegroundColor Yellow
} else {
    # Default to Local (where ngrok saves by default)
    $configPath = $configPath2
    Write-Host "Config not found, will create in Local folder" -ForegroundColor Yellow
}

if (Test-Path $configPath) {
    Write-Host "Reading config file..." -ForegroundColor Yellow
    $content = Get-Content $configPath -Raw
    
    # Fix invalid update_channel
    if ($content -match "update_channel:\s*['\`"]?['\`"]") {
        Write-Host "Fixing invalid update_channel..." -ForegroundColor Yellow
        $content = $content -replace "update_channel:\s*['\`"]?['\`"]", "update_channel: stable"
    } elseif ($content -notmatch "update_channel:") {
        # Add update_channel if missing
        $content = "version: `"2`"`nupdate_channel: stable`n" + $content
    }
    
    # Ensure it's set to stable (handle empty, quotes, or any value)
    $content = $content -replace "update_channel:\s*['\`"]?['\`"]?\s*$", "update_channel: stable"
    $content = $content -replace "update_channel:\s*['\`"]\s*$", "update_channel: stable"
    $content = $content -replace "update_channel:\s*$", "update_channel: stable"
    
    # If update_channel line doesn't exist, add it
    if ($content -notmatch "update_channel:") {
        $content = $content.TrimEnd() + "`nupdate_channel: stable`n"
    }
    
    Set-Content -Path $configPath -Value $content -NoNewline -Force
    Write-Host "Config file fixed at: $configPath" -ForegroundColor Green
} else {
    Write-Host "Config file not found. Creating new one..." -ForegroundColor Yellow
    $configDir = Split-Path $configPath
    if (-not (Test-Path $configDir)) {
        New-Item -ItemType Directory -Path $configDir -Force | Out-Null
    }
    $defaultConfig = "version: `"2`"`nupdate_channel: stable`n"
    Set-Content -Path $configPath -Value $defaultConfig
    Write-Host "Created new config file!" -ForegroundColor Green
}

Write-Host "`nVerifying config..." -ForegroundColor Cyan
& ngrok config check

Write-Host "`nDone! Now try: ngrok tcp 2222" -ForegroundColor Green

