#!/bin/bash
#SBATCH --job-name=plot_cumeval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=slurm_plot_cumeval_%j.out
#SBATCH --error=slurm_plot_cumeval_%j.err

echo "========================================"
echo "Cumulative Evaluations Plot (1000 DPI)"
echo "========================================"
echo "Start Time: $(date)"
echo "Node: $(hostname)"

source ~/.bashrc
conda activate pytorch_env

cd /scratch/prj/eng_waste_to_protein/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/mf_benchmark

python -u plot_cumeval_hires.py

echo "End Time: $(date)"
echo "========================================"
