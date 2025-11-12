# Instructions to Push Code to GitHub

## Option 1: Install Git and Use PowerShell Script (Recommended)

### Step 1: Install Git for Windows
1. Download from: https://git-scm.com/download/win
2. Run the installer with default settings
3. Restart your terminal/PowerShell

### Step 2: Run the Push Script
```powershell
cd C:\Users\fyp\Desktop\fyp\typhoon_prediction
.\push_to_github.ps1
```

---

## Option 2: Use GitHub Desktop (Easier, GUI-based)

### Step 1: Install GitHub Desktop
1. Download from: https://desktop.github.com/
2. Sign in with your GitHub account

### Step 2: Add Repository
1. Click "File" → "Add Local Repository"
2. Browse to: `C:\Users\fyp\Desktop\fyp\typhoon_prediction`
3. Click "Add Repository"

### Step 3: Push to GitHub
1. Review the changes in the left panel
2. Enter commit message: "Initial commit: Typhoon prediction pipeline"
3. Click "Commit to main"
4. Click "Publish repository" (or "Push origin" if already published)
5. Select the repository: `Angie-ch/newfyp`

---

## Option 3: Manual Git Commands

If Git is already installed, run these commands:

```powershell
cd C:\Users\fyp\Desktop\fyp\typhoon_prediction

# Initialize git (if not already done)
git init

# Add remote repository
git remote add origin https://github.com/Angie-ch/newfyp.git
# OR if remote exists:
git remote set-url origin https://github.com/Angie-ch/newfyp.git

# Add all files (respects .gitignore)
git add .

# Commit
git commit -m "Initial commit: Typhoon prediction pipeline with physics-informed diffusion model"

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## What Gets Excluded (via .gitignore)

The following are **NOT** pushed to GitHub (too large or sensitive):
- ✅ Virtual environment (`pytorch_gpu/`)
- ✅ Checkpoints (`checkpoints/`, `*.pth`, `*.pt`)
- ✅ Data files (`data/processed_temporal_split/`, `*.npz`)
- ✅ Log files (`*.log`)
- ✅ Results (`results/`)
- ✅ Python cache (`__pycache__/`)

---

## Important Notes

1. **Large Files**: Checkpoints and data files are excluded. If you need to share them, use:
   - Git LFS (Large File Storage)
   - Cloud storage (Google Drive, Dropbox)
   - Or create a separate repository for data

2. **Sensitive Information**: Make sure no API keys, passwords, or personal data are in the code

3. **First Push**: If the repository is empty, the first push will upload all code files

4. **Authentication**: You may need to authenticate with GitHub:
   - Personal Access Token (recommended)
   - Or use GitHub Desktop which handles authentication

---

## Troubleshooting

### "Git is not recognized"
- Install Git for Windows (see Option 1, Step 1)
- Or use GitHub Desktop (Option 2)

### "Authentication failed"
- Generate a Personal Access Token: https://github.com/settings/tokens
- Use token as password when pushing

### "Repository not found"
- Make sure the repository exists at: https://github.com/Angie-ch/newfyp
- Check you have write access to the repository

### "Large file error"
- Some files might be too large (>100MB)
- Use Git LFS or exclude them in `.gitignore`

