#!/bin/bash
#SBATCH --job-name=base_uq_hp
#SBATCH --output=slurm_base_uq_hp_%j.out
#SBATCH --error=slurm_base_uq_hp_%j.err
#SBATCH --reservation=rental_9427
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

echo "========================================"
echo "Base UQ Models - HP Tuning (200 trials)"
echo "Models: DNGO, BNN, MC-Dropout, Deep Ensemble, SNGP"
echo "Data Split: 6:2:2 (Train:Val:Test)"
echo "Start Time: $(date)"
echo "========================================"

# Conda environment
source ~/.bashrc
conda activate pytorch_env

# GPU check
nvidia-smi

cd /cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/model_comparison

python -u hp_tuning_base_uq_models.py

echo "========================================"
echo "End Time: $(date)"
echo "========================================"
