#!/bin/bash
#SBATCH --job-name=tanh_comp
#SBATCH --reservation=rental_9427
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --output=slurm_tanh_comp_%j.out
#SBATCH --error=slurm_tanh_comp_%j.err

source ~/.bashrc
conda activate mfbo

cd /cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/model_comparison

echo "DNGO tanh vs ReLU Comparison"
echo "5 Models: MFGP, DNGO-ReLU, DNGO-tanh, Pretrain-ReLU, Pretrain-tanh"
echo "BO Trials: 50, 10 Folds"
echo "Start: $(date)"

python -u run_tanh_comparison.py

echo "End: $(date)"
