#!/bin/bash
#SBATCH --job-name=full_compare
#SBATCH --reservation=rental_9427
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=slurm_full_compare_%j.out
#SBATCH --error=slurm_full_compare_%j.err

source ~/.bashrc
conda activate mfbo

cd /cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/model_comparison

echo "Starting full comparison with BO trials = 100"
echo "Start time: $(date)"

python run_full_comparison.py

echo "End time: $(date)"
