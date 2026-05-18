#!/bin/bash
#SBATCH --job-name=plot_ct_v7
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=slurm_plot_ct_v7_%j.out
#SBATCH --error=slurm_plot_ct_v7_%j.err

echo "Start: $(date)"
source ~/.bashrc
conda activate pytorch_env
cd /scratch/prj/eng_waste_to_protein/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/mf_benchmark
python -u plot_computing_time_v7.py
echo "End: $(date)"
