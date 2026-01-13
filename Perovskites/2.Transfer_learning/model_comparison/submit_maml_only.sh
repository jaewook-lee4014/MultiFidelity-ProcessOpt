#!/bin/bash
#SBATCH --job-name=maml_hp_tune
#SBATCH --output=slurm_maml_hp_%j.out
#SBATCH --error=slurm_maml_hp_%j.err
#SBATCH --reservation=rental_9427
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

echo "========================================"
echo "MAML Hyperparameter Tuning (Stability Fix)"
echo "Start Time: $(date)"
echo "========================================"

# Conda environment
source ~/.bashrc
conda activate pytorch_env

# GPU check
nvidia-smi

cd /cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/model_comparison

python -u hp_tuning_maml_only.py

echo "========================================"
echo "End Time: $(date)"
echo "========================================"
