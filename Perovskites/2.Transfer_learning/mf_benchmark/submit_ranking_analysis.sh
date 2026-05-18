#!/bin/bash
#SBATCH --job-name=rank_analysis
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_ranking_analysis_%j.out
#SBATCH --error=slurm_ranking_analysis_%j.err

echo "========================================"
echo "Fidelity Ranking Analysis"
echo "========================================"
echo "Start: $(date)"
echo "Node: $(hostname)"

export PYTHONUNBUFFERED=1
export PATH="/scratch/users/k23070952/vllm_env310/bin:$PATH"

cd /scratch/prj/eng_waste_to_protein/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/mf_benchmark

python -u ranking_analysis.py

echo ""
echo "End: $(date)"
echo "========================================"
