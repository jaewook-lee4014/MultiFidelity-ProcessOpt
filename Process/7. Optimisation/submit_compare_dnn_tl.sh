#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --job-name=dnn-vs-tl
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/k23070952/dnn_vs_tl_output_%j.txt
#SBATCH --error=/scratch/users/k23070952/dnn_vs_tl_error_%j.txt

source ~/.bashrc

echo "[$(date)] Job started on $(hostname)"
echo "Python: $(which python)"
python --version
nvidia-smi

WORKDIR="/cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Process/7. Optimisation"
cd "$WORKDIR"

export PYTHONUNBUFFERED=1

echo "[$(date)] Running DNN vs TL comparison..."
python "$WORKDIR/compare_dnn_vs_tl.py"
echo "[$(date)] Done."
