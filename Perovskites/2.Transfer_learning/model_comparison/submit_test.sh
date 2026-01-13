#!/bin/bash
#SBATCH --job-name=test_layout
#SBATCH --reservation=rental_9427
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=0:10:00
#SBATCH --output=slurm_test_%j.out
#SBATCH --error=slurm_test_%j.err

source ~/.bashrc
conda activate mfbo

cd /cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/model_comparison

python test_single_fold.py
