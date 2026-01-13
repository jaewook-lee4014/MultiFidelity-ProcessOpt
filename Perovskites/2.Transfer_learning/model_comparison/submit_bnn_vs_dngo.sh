#!/bin/bash
#SBATCH --job-name=bnn_vs_dngo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_bnn_vs_dngo_%j.out
#SBATCH --error=slurm_bnn_vs_dngo_%j.err

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

# Change to working directory
cd /cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/model_comparison

# Run comparison
echo ""
echo "Running BNN vs DNGO comparison..."
echo ""

python run_bnn_vs_dngo.py --n-runs 10 --n-lofi 72 --n-hifi 9

echo ""
echo "Job finished at $(date)"
