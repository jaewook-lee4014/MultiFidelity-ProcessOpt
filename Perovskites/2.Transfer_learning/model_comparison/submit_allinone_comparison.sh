#!/bin/bash
#SBATCH --job-name=dngo_aio
#SBATCH --reservation=rental_9427
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_aio_comp_%j.out
#SBATCH --error=slurm_aio_comp_%j.err

source ~/.bashrc
conda activate mfbo

cd /cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/model_comparison

echo "DNGO-AllInOne vs Baseline Comparison"
echo "5 Models: MFGP, DNGO-Base, DNGO-AllInOne, Pretrain-Base, Pretrain-AllInOne"
echo "Optuna HP tuning: 300 trials (alpha 0-1)"
echo "3 Folds"
echo "Start: $(date)"

export PYTHONUNBUFFERED=1
python -u run_allinone_comparison.py

echo "End: $(date)"
