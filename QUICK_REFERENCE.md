# Quick Reference Card

## ⚡ TL;DR - Just Want to Run It?

```bash
cd /Volumes/data/fyp/typhoon_prediction
bash run_quick_test.sh
open results_real_data/prediction_report.html
```

**Done!** You've trained and evaluated a model on real typhoon data.

---

## 🎯 What It Uses (Default)

✅ **Real typhoon tracks** from IBTrACS WP (79 storms, 2021-2024)  
✅ **Synthetic atmospheric frames** (48 channels, no download)  
❌ **NO ERA5 download** (unless you request it)  
❌ **NO API setup required**  

---

## 📝 Common Commands

### Quick Test (~15 minutes)
```bash
bash run_quick_test.sh
```

### Full Training (~1-4 hours)
```bash
python run_real_data_pipeline.py --n-samples 100
```

### With More Data
```bash
python run_real_data_pipeline.py --n-samples 200
```

### Evaluation Only
```bash
python run_real_data_pipeline.py --eval-only --eval-samples 20
```

### Monitor Training
```bash
tensorboard --logdir logs/
```

### View Results
```bash
open results_real_data/prediction_report.html
```

---

## 🔧 Useful Options

| Option | What It Does | Example |
|--------|--------------|---------|
| `--n-samples N` | Number of training samples | `--n-samples 200` |
| `--quick-test` | Fast test (3 epochs) | `--quick-test` |
| `--batch-size N` | Reduce if out of memory | `--batch-size 2` |
| `--eval-samples N` | How many to evaluate | `--eval-samples 20` |
| `--start-year YEAR` | Start year for data | `--start-year 2018` |
| `--end-year YEAR` | End year for data | `--end-year 2023` |
| `--autoencoder-epochs N` | Autoencoder training | `--autoencoder-epochs 30` |
| `--diffusion-epochs N` | Diffusion training | `--diffusion-epochs 50` |
| `--skip-data-generation` | Use existing samples | `--skip-data-generation` |
| `--skip-autoencoder` | Use existing model | `--skip-autoencoder` |
| `--eval-only` | Only evaluate | `--eval-only` |

---

## 📊 Expected Performance

| Mode | Time (CPU/GPU) | Track Error | Intensity MAE |
|------|----------------|-------------|---------------|
| Quick test | 15 min / 5 min | 70-90 km | 2-3 m/s |
| Full (100 samples) | 4 hrs / 1 hr | 50-70 km | 1.5-2.5 m/s |
| With ERA5 | Same / Same | 45-60 km | 1.3-2.0 m/s |

---

## 🐛 Quick Troubleshooting

### "No storms found"
```bash
python run_real_data_pipeline.py --start-year 2018 --end-year 2024
```

### Out of memory
```bash
python run_real_data_pipeline.py --batch-size 2 --n-samples 50
```

### Training too slow
```bash
python run_real_data_pipeline.py --quick-test
```

### Want to see progress
```bash
tensorboard --logdir logs/
```

---

## 📚 Documentation

| File | When to Read |
|------|--------------|
| `START_HERE.md` | First time orientation |
| `DEFAULT_MODE_EXPLAINED.md` | Understanding synthetic frames |
| `QUICK_REFERENCE.md` | This file - quick lookup |
| `README_REAL_DATA_PIPELINE.md` | Complete guide |
| `ERA5_SETUP.md` | Setting up real ERA5 data |

---

## 🎓 Workflow

```
1. Test integration
   python test_era5_integration.py

2. Quick test
   bash run_quick_test.sh

3. View results
   open results_real_data/prediction_report.html

4. If satisfied, full training
   python run_real_data_pipeline.py --n-samples 100

5. Monitor
   tensorboard --logdir logs/
```

---

## ✅ Output Files

```
results_real_data/
├── prediction_report.html       ⭐ START HERE
├── trajectory_sample_*.png
├── intensity_sample_*.png
├── error_statistics.png
└── evaluation_results.json

checkpoints/
├── autoencoder/best.pth
└── diffusion/best.pth

data/processed/
├── cases/*.npz
├── metadata.csv
└── statistics.json
```

---

## 💡 Pro Tips

1. **Start small** - Use `--quick-test` first
2. **Monitor progress** - TensorBoard shows real-time loss
3. **Save time** - Use `--skip-*` flags to resume work
4. **Check results** - Always review HTML report
5. **Iterate** - Adjust hyperparameters based on results

---

## 🚀 Ready to Go!

**No setup, no downloads, just run:**

```bash
cd /Volumes/data/fyp/typhoon_prediction
bash run_quick_test.sh
```

**That's it!** 🌪️

