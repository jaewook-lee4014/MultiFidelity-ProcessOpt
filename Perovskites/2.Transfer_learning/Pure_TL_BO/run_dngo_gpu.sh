#!/bin/bash
#SBATCH --job-name=dngo_100runs
#SBATCH --partition=nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=dngo_100runs_%j.log
#SBATCH --error=dngo_100runs_%j.err

cd /cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/Pure_TL_BO

echo "Starting DNGO 100 runs with GPU..."
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

yes | python main.py \
  --model-type dngo \
  --num-runs 100 \
  --cost-budget 50 \
  --use-hyperparameter-bo \
  --pretrain-bo-trials 50 \
  --finetune-bo-trials 50 \
  --use-loocv \
  --use-uncertainty-loss \
  --uncertainty-weight 0.3 \
  --save-results \
  --results-filename dngo_loocv_unc_100runs.csv \
  --plot-results

echo "Done!"
