#!/bin/bash
#SBATCH --job-name=sf_bo_bench
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_sf_bo_%j.out
#SBATCH --error=slurm_sf_bo_%j.err

echo "========================================"
echo "Single-Fidelity BO Benchmark"
echo "6 UQ Models x 2 Test Functions"
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

# Run benchmark
# 20 seeds x 50 iterations
python run_sf_bo_benchmark.py --seeds 20 --iterations 50

echo ""
echo "========================================"
echo "End Time: $(date)"
echo "========================================"
