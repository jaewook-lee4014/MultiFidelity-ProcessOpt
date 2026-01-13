#!/bin/bash
#SBATCH --job-name=ol_compare
#SBATCH --reservation=rental_9427
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_ol_compare_%j.out
#SBATCH --error=slurm_ol_compare_%j.err

echo "============================================================"
echo "OL Model Comparison: DNGO-OL vs BNN-OL (with HP-BO)"
echo "============================================================"
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo ""

nvidia-smi

echo ""
echo "Starting comparison..."
echo ""

export PYTHONUNBUFFERED=1
stdbuf -oL -eL python -u run_ol_comparison.py \
    --n-runs 10 \
    --n-lofi 72 \
    --n-hifi 9 \
    --bo-trials 20 \
    --results-dir results \
    2>&1 | stdbuf -oL tee ol_comparison.log

echo ""
echo "============================================================"
echo "Job finished at $(date)"
echo "============================================================"
