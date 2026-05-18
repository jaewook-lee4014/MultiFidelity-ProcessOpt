#!/bin/bash
#SBATCH --job-name=rand_baselines
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_random_baselines_%j.out
#SBATCH --error=slurm_random_baselines_%j.err

echo "========================================"
echo "Non-Learning Baselines: HF-Only Random + LF-Screening"
echo "7 Benchmarks × 2 Baselines × 20 Seeds"
echo "========================================"
echo "Start: $(date)"
echo "Node: $(hostname)"

export PYTHONUNBUFFERED=1
export PATH="/scratch/users/k23070952/vllm_env310/bin:$PATH"

cd /scratch/prj/eng_waste_to_protein/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/mf_benchmark

python -u benchmark_random_baselines.py --n-seeds 20 --n-workers 8

echo ""
echo "End: $(date)"
echo "========================================"
