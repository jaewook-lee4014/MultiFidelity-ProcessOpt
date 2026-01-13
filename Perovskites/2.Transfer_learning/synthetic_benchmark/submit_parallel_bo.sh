#!/bin/bash
#SBATCH --job-name=bo_parallel
#SBATCH --reservation=rental_9427
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_parallel_%j.out
#SBATCH --error=slurm_parallel_%j.err

echo "========================================"
echo "Parallel SF BO Benchmark"
echo "GPU: 1 (batch processing)"
echo "CPUs: 32 (parallel seeds)"
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

echo "Starting Python script..."
echo "which python: $(which python)"
echo "PWD: $(pwd)"

# Run parallel benchmark
python -u bo_parallel_gpu.py --seeds 5 --iterations 50 --workers 5

echo ""
echo "========================================"
echo "End Time: $(date)"
echo "========================================"
