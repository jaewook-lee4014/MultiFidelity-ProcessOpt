#!/bin/bash
#SBATCH --job-name=gs_vs_mfgp_viz
#SBATCH --output=slurm_viz_%j.out
#SBATCH --error=slurm_viz_%j.err
#SBATCH --reservation=rental_9427
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

echo "========================================"
echo "Start Time: $(date)"
echo "========================================"

# Conda 환경 활성화
source ~/.bashrc
conda activate pytorch_env

# GPU 확인
nvidia-smi

cd /cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/Pure_TL_BO/experiments

python -u visualize_gradient_scaling_vs_mfgp.py

echo "========================================"
echo "End Time: $(date)"
echo "========================================"
