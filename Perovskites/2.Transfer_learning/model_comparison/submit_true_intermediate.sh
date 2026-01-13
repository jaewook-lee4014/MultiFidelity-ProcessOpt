#!/bin/bash
#SBATCH --job-name=true_int
#SBATCH --reservation=rental_9427
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_true_int_%j.out
#SBATCH --error=slurm_true_int_%j.err

# Load conda environment
source ~/.bashrc
conda activate mfbo

# Change to model_comparison directory
cd /cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/model_comparison

# Run True Intermediate comparison with unbuffered output
python -u run_true_intermediate.py
