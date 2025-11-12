# Script to push code to GitHub
# Run this after installing Git for Windows

Write-Host "=== Pushing to GitHub ===" -ForegroundColor Green

# Check if git is available
try {
    $gitVersion = git --version
    Write-Host "Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Git is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Git for Windows from: https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "Or use GitHub Desktop: https://desktop.github.com/" -ForegroundColor Yellow
    exit 1
}

# Navigate to project directory
$projectDir = "C:\Users\fyp\Desktop\fyp\typhoon_prediction"
Set-Location $projectDir

Write-Host "`nCurrent directory: $(Get-Location)" -ForegroundColor Cyan

# Check if already a git repository
if (-not (Test-Path ".git")) {
    Write-Host "`nInitializing git repository..." -ForegroundColor Yellow
    git init
}

# Add remote if not exists
$remoteUrl = "https://github.com/Angie-ch/newfyp.git"
$existingRemote = git remote get-url origin 2>$null

if ($LASTEXITCODE -ne 0 -or $existingRemote -ne $remoteUrl) {
    Write-Host "`nSetting up remote repository..." -ForegroundColor Yellow
    if ($existingRemote) {
        git remote set-url origin $remoteUrl
    } else {
        git remote add origin $remoteUrl
    }
}

# Add all files (respecting .gitignore)
Write-Host "`nAdding files to git..." -ForegroundColor Yellow
git add .

# Check if there are changes to commit
$status = git status --porcelain
if ($status) {
    Write-Host "`nCommitting changes..." -ForegroundColor Yellow
    git commit -m "Initial commit: Typhoon prediction pipeline with physics-informed diffusion model"
    
    Write-Host "`nPushing to GitHub..." -ForegroundColor Yellow
    git branch -M main
    git push -u origin main
    
    Write-Host "`n=== SUCCESS! Code pushed to GitHub ===" -ForegroundColor Green
    Write-Host "Repository: $remoteUrl" -ForegroundColor Cyan
} else {
    Write-Host "`nNo changes to commit. Everything is up to date." -ForegroundColor Yellow
}

