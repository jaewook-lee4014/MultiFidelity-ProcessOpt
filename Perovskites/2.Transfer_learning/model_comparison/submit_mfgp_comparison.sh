#!/bin/bash
#SBATCH --job-name=mfgp_compare
#SBATCH --reservation=rental_9427
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_mfgp_compare_%j.out
#SBATCH --error=slurm_mfgp_compare_%j.err

echo "============================================================"
echo "MFGP Model Comparison: MFGP vs DNGO vs BNN"
echo "============================================================"
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo ""

nvidia-smi

echo ""
echo "Checking BoTorch installation..."
python -c "import botorch; print(f'BoTorch version: {botorch.__version__}')" 2>/dev/null || {
    echo "Installing BoTorch..."
    pip install botorch --quiet
}

echo ""
echo "Starting comparison..."
echo ""

export PYTHONUNBUFFERED=1
stdbuf -oL -eL python -u compare_all_models.py \
    --n-runs 10 \
    --n-lofi 72 \
    --n-hifi 9 \
    --bo-trials 20 \
    --save-dir results_mfgp \
    2>&1 | stdbuf -oL tee mfgp_comparison.log

echo ""
echo "============================================================"
echo "Job finished at $(date)"
echo "============================================================"
