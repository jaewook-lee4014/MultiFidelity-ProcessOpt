#!/bin/bash
#SBATCH --job-name=mf_benchmark
#SBATCH --output=benchmark_%j.log
#SBATCH --error=benchmark_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# ============================================
# Multi-Fidelity Benchmark Run Script
# ============================================

# Load modules (adjust for your HPC system)
# module load python/3.10
# module load cuda/11.8

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Print environment info
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"
echo "============================================"

# Check GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo "============================================"
echo "Starting benchmark..."
echo "============================================"

# Run benchmark
python benchmark.py --n-seeds 5 --base-seed 42

echo "============================================"
echo "Benchmark completed!"
echo "Date: $(date)"
echo "============================================"
