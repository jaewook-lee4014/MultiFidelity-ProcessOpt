#!/bin/bash
#SBATCH --job-name=bo_online_hp
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_online_hp_%j.out
#SBATCH --error=slurm_online_hp_%j.err

echo "========================================"
echo "SF BO with Online HP Tuning (LOOCV)"
echo "6 UQ Models x 2 Test Functions"
echo "HP tuning at start of each run"
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

# Run benchmark with online HP tuning
# 5 seeds x 50 iterations
# HP tuning every 50 iterations (= only at start for 50-iter runs)
# 15 Optuna trials for HP search
python bo_with_online_hp_tuning.py --seeds 5 --iterations 50 --hp-interval 50 --hp-trials 15

echo ""
echo "========================================"
echo "End Time: $(date)"
echo "========================================"
