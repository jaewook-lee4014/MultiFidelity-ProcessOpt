#!/bin/bash
#SBATCH --job-name=mf_viz
#SBATCH --reservation=rental_9427
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_viz_%j.out
#SBATCH --error=slurm_viz_%j.err

echo "========================================"
echo "MF BO Step Visualization"
echo "========================================"
echo "Start Time: $(date)"
echo "Node: $(hostname)"
echo ""

# Load environment
source ~/.bashrc
conda activate pytorch_env

cd /scratch/prj/eng_waste_to_protein/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/synthetic_benchmark

echo "which python: $(which python)"
echo "PWD: $(pwd)"

# Run visualization
python -u visualize_mf_bo_steps.py --seed 42 --budget 50 --save-interval 5

echo ""
echo "========================================"
echo "End Time: $(date)"
echo "========================================"
