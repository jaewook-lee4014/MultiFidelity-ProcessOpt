#!/bin/bash
#SBATCH --job-name=debug_alpha
#SBATCH --reservation=rental_9427
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_debug_alpha_%j.out
#SBATCH --error=slurm_debug_alpha_%j.err

# Load conda environment
source ~/.bashrc
conda activate mfbo

# Change to model_comparison directory
cd /cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/model_comparison

# Run alpha=0.99 test
python -u debug_alpha_fold1.py
