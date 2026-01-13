#!/bin/bash
#SBATCH --job-name=mf_bo_hp
#SBATCH --reservation=rental_9427
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --output=slurm_mf_bo_%j.out
#SBATCH --error=slurm_mf_bo_%j.err

echo "========================================"
echo "MF BO Benchmark (Branin, 5-fold CV HP)"
echo "GPU: 1"
echo "CPUs: 10 (rental_9427)"
echo "Seeds: 10"
echo "========================================"
echo "Start Time: $(date)"
echo "Node: $(hostname)"
echo ""

# Load environment
source ~/.bashrc
conda activate pytorch_env

# Show GPU
nvidia-smi

# Navigate to directory
cd /scratch/prj/eng_waste_to_protein/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/synthetic_benchmark

echo "Starting Python script..."
echo "which python: $(which python)"
echo "PWD: $(pwd)"

# Run MF benchmark
# - Branin only (Park excluded)
# - MF models: GP_MFGP (MFGP baseline), DNGO_Joint (best TL from model_comparison)
# - Two scenarios: favorable (ρ=0.1, R²>0.9), unfavorable (ρ=0.5, R²<0.75)
# - Optuna Bayesian HP optimization every 20 HF data points
# - seeds=10, workers=10 for parallel (12 CPUs rental node)
python -u mf_bo_benchmark.py \
    --seeds 10 \
    --budget 50 \
    --workers 10 \
    --scenario both \
    --hp-interval 20

echo ""
echo "========================================"
echo "End Time: $(date)"
echo "========================================"
