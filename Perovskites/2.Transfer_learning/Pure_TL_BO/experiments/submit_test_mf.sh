#!/bin/bash
#SBATCH --job-name=test_mf
#SBATCH --output=slurm_test_mf_%j.out
#SBATCH --error=slurm_test_mf_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

echo "Testing Multi-Fidelity Training"
echo "Start Time: $(date)"

# Conda 환경 활성화
source ~/.bashrc
conda activate pytorch_env

cd /cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/Pure_TL_BO

python experiments/test_mf_approaches.py

echo "End Time: $(date)"
