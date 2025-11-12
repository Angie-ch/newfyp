#!/bin/bash
###############################################################################
# Quick Test - Real Data Pipeline
###############################################################################
#
# This script runs a quick test of the complete pipeline with real data:
# - Generates 20 training samples from IBTrACS WP (2021-2024)
# - Trains autoencoder for 3 epochs
# - Trains diffusion model for 3 epochs
# - Evaluates on 3 test samples
# - Creates visualizations
#
# Expected runtime: ~10-15 minutes (CPU) or ~5-8 minutes (GPU)
#
# Usage:
#   bash run_quick_test.sh                  # Default: Uses ERA5 (or downloads)
#   bash run_quick_test.sh --no-use-era5    # Use synthetic frames instead
#   bash run_quick_test.sh --download-era5  # Force ERA5 download
#
###############################################################################

echo "================================================================================"
echo "QUICK TEST - Real Data Pipeline"
echo "================================================================================"
echo ""
echo "This will:"
echo "  1. Load real typhoon data from IBTrACS WP (2021-2024)"
echo "  2. Generate 20 training samples"
echo "  3. Train models for 3 epochs each (quick test)"
echo "  4. Evaluate and visualize results"
echo ""
echo "Expected runtime: ~10-15 minutes"
echo ""
echo "================================================================================"
echo ""

# Run pipeline with quick test flag
python run_real_data_pipeline.py \
    --n-samples 20 \
    --start-year 2021 \
    --end-year 2024 \
    --quick-test \
    --eval-samples 3 \
    "$@"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✓ QUICK TEST COMPLETE!"
    echo "================================================================================"
    echo ""
    echo "View results:"
    echo "  open results_real_data/prediction_report.html"
    echo ""
    echo "Next steps:"
    echo "  1. Review the visualizations and metrics"
    echo "  2. If satisfied, run full training:"
    echo "     python run_real_data_pipeline.py --n-samples 100"
    echo "  3. (Optional) Use synthetic data for faster testing:"
    echo "     python run_real_data_pipeline.py --n-samples 100 --no-use-era5"
    echo ""
    echo "================================================================================"
else
    echo ""
    echo "✗ Quick test failed. Check error messages above."
    exit 1
fi

