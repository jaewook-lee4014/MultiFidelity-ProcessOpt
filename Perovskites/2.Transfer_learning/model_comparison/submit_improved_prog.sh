#!/bin/bash
#SBATCH --job-name=improved_prog
#SBATCH --reservation=rental_9427
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=slurm_improved_prog_%j.out
#SBATCH --error=slurm_improved_prog_%j.err

source ~/.bashrc
conda activate mfbo

cd /cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/model_comparison

echo "Testing improved Progressive Unfreezing"
echo "lr_boost=2.0, lr_decay=0.7 (was: 1.0, 0.5)"
echo "Start: $(date)"

python test_improved_prog_unfreeze.py

echo "End: $(date)"
