#!/bin/bash
#SBATCH --job-name=prog_unfreeze
#SBATCH --reservation=rental_9427
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=slurm_prog_unfreeze_%j.out
#SBATCH --error=slurm_prog_unfreeze_%j.err

# Load conda environment
source ~/.bashrc
conda activate mfbo

# Change to model_comparison directory
cd /cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/model_comparison

# Run Progressive Unfreezing visualization
python visualize_progressive_unfreezing.py
