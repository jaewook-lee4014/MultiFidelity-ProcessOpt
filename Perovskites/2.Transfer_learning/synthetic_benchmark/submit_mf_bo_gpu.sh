#!/bin/bash
#SBATCH --job-name=mf_bo_hp50
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_mf_bo_%j.out
#SBATCH --error=slurm_mf_bo_%j.err

echo "========================================"
echo "MF BO Benchmark with HP Optimization"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time: $(date)"
echo "========================================"

cd /cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/synthetic_benchmark

# Activate conda environment if needed
# source activate your_env

# Run MF BO benchmark
# - seeds=5: 5 random seeds for statistical significance
# - budget=50: total cost budget
# - hp-interval=20: HP optimization every 20 HF data points
# - Initial HP optimization added (runs once before BO starts)
# - N_OPTUNA_TRIALS=50: 50 trials per HP optimization

python mf_bo_benchmark.py \
    --seeds 5 \
    --budget 50 \
    --workers 8 \
    --scenario both \
    --hp-interval 20

echo "========================================"
echo "End time: $(date)"
echo "========================================"
