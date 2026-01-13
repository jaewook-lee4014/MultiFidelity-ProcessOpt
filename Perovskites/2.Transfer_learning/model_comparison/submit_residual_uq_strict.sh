#!/bin/bash
#SBATCH --job-name=ruq_strict
#SBATCH --output=slurm_ruq_strict_%j.out
#SBATCH --error=slurm_ruq_strict_%j.err
#SBATCH --reservation=rental_9427
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

echo "========================================"
echo "Start Time: $(date)"
echo "========================================"

source ~/.bashrc
conda activate pytorch_env

nvidia-smi

cd /cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/model_comparison

python -u visualize_6methods_residual_uq_strict.py

echo "========================================"
echo "End Time: $(date)"
echo "========================================"
