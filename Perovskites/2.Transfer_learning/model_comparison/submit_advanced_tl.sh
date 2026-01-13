#!/bin/bash
#SBATCH --job-name=advanced_tl
#SBATCH --output=slurm_advanced_tl_%j.out
#SBATCH --error=slurm_advanced_tl_%j.err
#SBATCH --reservation=rental_9427
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

echo "========================================"
echo "Advanced Transfer Learning Comparison"
echo "Start Time: $(date)"
echo "========================================"

# Conda environment
source ~/.bashrc
conda activate pytorch_env

# GPU check
nvidia-smi

cd /cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/model_comparison

python -u run_advanced_tl_comparison.py

echo "========================================"
echo "End Time: $(date)"
echo "========================================"
