#!/bin/bash
#SBATCH --job-name=model_compare
#SBATCH --reservation=rental_9427
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=slurm_model_compare_%j.out
#SBATCH --error=slurm_model_compare_%j.err

# Model Comparison Experiment
# Compares DNGO and BNN transfer learning performance

echo "=============================================="
echo "Model Comparison Experiment"
echo "Timestamp: $(date)"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "=============================================="

# Run the experiment
python -u run_experiment.py \
    --n-sets 5 \
    --hidden-layers 2 \
    --hidden-dim 64 \
    --pretrain-epochs 200 \
    --finetune-epochs 100 \
    --verbose

echo ""
echo "=============================================="
echo "Experiment completed!"
echo "=============================================="
