#!/bin/bash
#SBATCH --job-name=smoke_baselines
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=slurm_smoke_%j.out
#SBATCH --error=slurm_smoke_%j.err

echo "========================================"
echo "Smoke Test: 4 New Baselines on Branin-Fav"
echo "========================================"
echo "Start: $(date)"
echo "Node: $(hostname)"

export PYTHONUNBUFFERED=1
export PATH="/scratch/users/k23070952/vllm_env310/bin:$PATH"

cd /scratch/prj/eng_waste_to_protein/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/mf_benchmark

bash run_smoke_test.sh

echo "End: $(date)"
echo "========================================"
