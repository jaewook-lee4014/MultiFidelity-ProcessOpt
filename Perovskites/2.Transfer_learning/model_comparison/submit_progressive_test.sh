#!/bin/bash
#SBATCH --job-name=prog_unfreeze
#SBATCH --reservation=rental_9427
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=slurm_prog_unfreeze_%j.out
#SBATCH --error=slurm_prog_unfreeze_%j.err

echo "============================================================"
echo "Progressive Unfreezing Test"
echo "============================================================"
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo ""

nvidia-smi

echo ""
echo "Starting test..."
echo ""

export PYTHONUNBUFFERED=1
stdbuf -oL -eL python -u test_progressive_unfreezing.py \
    --n-runs 10 \
    --n-lofi 72 \
    --n-hifi 9 \
    --bo-trials 15 \
    2>&1 | stdbuf -oL tee progressive_unfreezing_test.log

echo ""
echo "============================================================"
echo "Job finished at $(date)"
echo "============================================================"
