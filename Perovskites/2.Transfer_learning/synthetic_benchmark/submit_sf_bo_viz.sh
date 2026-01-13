#!/bin/bash
#SBATCH --job-name=sf_bo_viz
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=slurm_sf_bo_viz_%j.out
#SBATCH --error=slurm_sf_bo_viz_%j.err

echo "========================================"
echo "Single-Fidelity BO with EI Visualization"
echo "6 UQ Models x 2 Test Functions"
echo "5 seeds, seed 0 with ALL-step EI plots"
echo "========================================"
echo "Start Time: $(date)"
echo "Node: $(hostname)"
echo ""

# Load environment
source /scratch/prj/eng_waste_to_protein/conda_envs/mfbo/bin/activate

# Show GPU
nvidia-smi

# Navigate to directory
cd /scratch/prj/eng_waste_to_protein/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/synthetic_benchmark

# Run benchmark with visualization
# 5 seeds x 50 iterations
# Seed 0 gets full EI visualization (50 images per model for Branin-2D)
python run_sf_bo_with_visualization.py --seeds 5 --iterations 50

echo ""
echo "========================================"
echo "End Time: $(date)"
echo "========================================"
