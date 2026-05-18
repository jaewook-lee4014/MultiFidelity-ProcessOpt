#!/bin/bash
#SBATCH --job-name=plot_lf_blr
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=slurm_plot_lf_blr_%j.out
#SBATCH --error=slurm_plot_lf_blr_%j.err

echo "========================================"
echo "LF-BLR Benchmark Plots (1000 DPI)"
echo "========================================"
echo "Start Time: $(date)"
echo "Node: $(hostname)"
echo "Memory: $(free -h | head -2)"
echo ""

source ~/.bashrc
conda activate pytorch_env

cd /scratch/prj/eng_waste_to_protein/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/mf_benchmark

echo "which python: $(which python)"
echo "PWD: $(pwd)"

python -u plot_lf_blr_hires.py

echo ""
echo "========================================"
echo "End Time: $(date)"
echo "========================================"
