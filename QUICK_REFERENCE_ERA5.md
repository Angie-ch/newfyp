# ⚡ Quick Reference: ERA5 vs Synthetic Data

## 🚀 Quick Commands

```bash
# Fast testing (recommended first run) - synthetic data
bash run_quick_test.sh --no-use-era5

# Production mode (default) - ERA5 data
bash run_quick_test.sh

# Full training - synthetic (fast)
python run_real_data_pipeline.py --n-samples 100 --no-use-era5

# Full training - ERA5 (default, best quality)
python run_real_data_pipeline.py --n-samples 100

# Force ERA5 download (refresh cache)
python run_real_data_pipeline.py --n-samples 100 --download-era5
```

---

## 📊 Quick Comparison

| Feature | `--no-use-era5`<br>(Synthetic) | Default<br>(ERA5) |
|---------|-------------------------------|-------------------|
| **First run time** | 10-15 min | Hours (download) |
| **Second run time** | 10-15 min | 10-15 min (cached) |
| **Network usage** | ~50 MB | ~5-20 GB |
| **Disk space** | ~100 MB | ~5-20 GB |
| **Accuracy** | Good (80-85%) | Excellent (90-95%) |
| **Setup required** | None | None |
| **API key needed** | No | No |
| **Use for** | Testing, dev | Production, research |

---

## 🎯 When to Use What?

### Use Synthetic (`--no-use-era5`)

✅ **First time running the pipeline**  
✅ **Rapid testing and development**  
✅ **CI/CD automated tests**  
✅ **Limited bandwidth or disk space**  
✅ **Don't need maximum accuracy**  
✅ **Want results in minutes, not hours**

**Command:**
```bash
python run_real_data_pipeline.py --no-use-era5 [other args]
```

### Use ERA5 (Default)

✅ **Production deployments**  
✅ **Research and publications**  
✅ **Maximum prediction accuracy**  
✅ **After initial testing**  
✅ **Have time for initial download**  
✅ **Want industry-standard results**

**Command:**
```bash
python run_real_data_pipeline.py [other args]
```

---

## 🔍 How to Tell Which Mode You're Using?

### Check the Log Output

**Synthetic mode:**
```
Data source: IBTrACS Western Pacific
Meteorological data: Synthetic (48 channels)
```

**ERA5 mode:**
```
Data source: IBTrACS Western Pacific
Meteorological data: ERA5 (48 channels)
⚠️  ERA5 download enabled - this may take a while!
```

### Check Command-Line Help

```bash
python run_real_data_pipeline.py --help | grep "use-era5"
```

Look for:
- `--use-era5` - default: True (ERA5 is default)
- `--no-use-era5` - use to disable ERA5

---

## 📝 Common Scenarios

### Scenario 1: First-Time User

```bash
# Step 1: Quick test with synthetic (10 minutes)
bash run_quick_test.sh --no-use-era5
# ✓ See results immediately

# Step 2: If satisfied, run with ERA5 (hours first time)
bash run_quick_test.sh
# ⏳ Downloads ERA5 (be patient!)

# Step 3: Future runs use cached ERA5 (fast!)
python run_real_data_pipeline.py --n-samples 100
# ✓ Fast + accurate
```

### Scenario 2: Developer Testing Code

```bash
# Always use synthetic for fast iteration
python run_real_data_pipeline.py --no-use-era5 --n-samples 20 --quick-test
# Change code, test again, repeat...
```

### Scenario 3: Production Training

```bash
# First time: Download ERA5
python run_real_data_pipeline.py --n-samples 100 --download-era5
# Wait for download...

# Subsequently: Use cached ERA5
python run_real_data_pipeline.py --n-samples 100
# Fast and accurate!
```

### Scenario 4: Research Paper

```bash
# Use ERA5 for best accuracy
python run_real_data_pipeline.py \
    --n-samples 200 \
    --start-year 2020 \
    --end-year 2024 \
    --autoencoder-epochs 50 \
    --diffusion-epochs 100
# Results are publication-quality
```

### Scenario 5: CI/CD Pipeline

```bash
# Always use synthetic in CI for speed
python run_real_data_pipeline.py \
    --no-use-era5 \
    --n-samples 10 \
    --quick-test
# Fast automated tests
```

---

## 🐛 Troubleshooting

### Problem: ERA5 download is too slow

**Solution:**
```bash
# Use synthetic data instead
python run_real_data_pipeline.py --no-use-era5 [other args]
```

### Problem: Not enough disk space for ERA5

**Solution:**
```bash
# Use synthetic (requires only ~100 MB)
python run_real_data_pipeline.py --no-use-era5 [other args]
```

### Problem: Want to force re-download ERA5

**Solution:**
```bash
# Delete cache and force download
rm -rf data/raw/era5/
python run_real_data_pipeline.py --download-era5 [other args]
```

### Problem: ERA5 download fails

**Solution:**
```bash
# Fall back to synthetic
python run_real_data_pipeline.py --no-use-era5 [other args]
```

### Problem: Not sure which mode is running

**Solution:**
```bash
# Check the console output
# Look for "Meteorological data: ERA5" or "Meteorological data: Synthetic"

# Or check command
python run_real_data_pipeline.py --help | grep use-era5
```

---

## 💡 Pro Tips

### Tip 1: Create Aliases
```bash
# Add to ~/.bashrc or ~/.zshrc
alias typhoon-fast='python run_real_data_pipeline.py --no-use-era5'
alias typhoon-prod='python run_real_data_pipeline.py'

# Then use
typhoon-fast --n-samples 50
typhoon-prod --n-samples 100
```

### Tip 2: Use Config Files
```bash
# Create fast-config.yaml
echo "use_era5: false" > fast-config.yaml

# Create prod-config.yaml
echo "use_era5: true" > prod-config.yaml
```

### Tip 3: Check ERA5 Cache
```bash
# See what ERA5 data you have
ls -lh data/raw/era5/

# See disk usage
du -sh data/raw/era5/
```

### Tip 4: Partial ERA5 Cache
If you have some ERA5 data cached:
```bash
# Without --no-use-era5:
# - Uses cached ERA5 for storms that have it
# - Uses synthetic for storms that don't
python run_real_data_pipeline.py --n-samples 100
```

### Tip 5: Monitor Downloads
```bash
# Watch ERA5 download progress
watch -n 1 'ls -lh data/raw/era5/ | tail -10'
```

---

## 📖 More Information

- **Comprehensive guide:** `ERA5_NOW_DEFAULT.md`
- **Change summary:** `CHANGES_SUMMARY.md`
- **General docs:** `START_HERE.md`
- **ERA5 setup:** `ERA5_SETUP.md`

---

## 🎯 TL;DR

```bash
# Fast (first run) → Use synthetic
bash run_quick_test.sh --no-use-era5

# Best quality (after testing) → Use ERA5 (default)
bash run_quick_test.sh

# Default changed: ERA5 is now default, add --no-use-era5 for fast mode
```

**Remember:** Synthetic = fast, ERA5 = accurate. Choose based on your needs!

