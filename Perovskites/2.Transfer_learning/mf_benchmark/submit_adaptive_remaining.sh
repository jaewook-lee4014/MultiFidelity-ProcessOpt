#!/bin/bash
#SBATCH --job-name=adapt_fix
#SBATCH --partition=interruptible_gpu
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm_adaptive_fix_%j.out
#SBATCH --error=slurm_adaptive_fix_%j.err

echo "========================================"
echo "Adaptive Benchmark - Branin-Fav remaining 2 models"
echo "========================================"
echo "Start: $(date)"
echo "Node: $(hostname)"
nvidia-smi 2>/dev/null || echo "No GPU info available"

export PYTHONUNBUFFERED=1
export PATH="/scratch/users/k23070952/vllm_env310/bin:$PATH"

cd /scratch/prj/eng_waste_to_protein/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/mf_benchmark

python -u benchmark_adaptive.py \
    --n-seeds 20 \
    --n-workers 48 \
    --phase 2 \
    --benchmarks "Branin-Fav" \
    --models "Sparse MFGP,DKL Multi-Fidelity" \
    --output-dir "benchmark_adaptive_phase2"

echo "End: $(date)"
echo "========================================"
