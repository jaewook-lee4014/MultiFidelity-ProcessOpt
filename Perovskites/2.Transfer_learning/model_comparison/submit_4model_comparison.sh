#!/bin/bash
#SBATCH --job-name=4model_cmp
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_4model_cmp_%j.out
#SBATCH --error=slurm_4model_cmp_%j.err

echo "============================================================"
echo "4-Model Comparison: DNGO vs BNN with Hyperparameter BO"
echo "============================================================"
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo ""
echo "GPU info:"
nvidia-smi
echo ""

# Change to working directory
cd /cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/model_comparison

# Run 4-model comparison
# - 10 runs for statistical significance
# - 72 LOFI, 9 HIFI samples (cost_15 config)
# - 20 BO trials for hyperparameter optimization
python run_4model_comparison.py \
    --n-runs 10 \
    --n-lofi 72 \
    --n-hifi 9 \
    --bo-trials 20

echo ""
echo "============================================================"
echo "Job finished at $(date)"
echo "============================================================"
