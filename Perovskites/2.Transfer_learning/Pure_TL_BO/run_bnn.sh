#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --job-name=bnn
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --signal=USR2
#SBATCH --output=bnn_output.txt
#SBATCH --error=bnn_error.txt

source ~/.bashrc

echo "[$(date)] which python: $(which python)"
python --version

WORKDIR=$(pwd)
cd "$WORKDIR"

export PYTHONUNBUFFERED=1

TS=$(date +%Y%m%d_%H%M%S)

echo "[$(date)] Starting bnn..."

python main.py \
    --model-type bnn \
    --num-runs 100 \
    --cost-budget 50 \
    --use-hyperparameter-bo \
    --pretrain-bo-trials 50 \
    --finetune-bo-trials 50 \
    --use-loocv \
    --use-uncertainty-loss \
    --uncertainty-weight 0.3 \
    --save-results \
    --results-filename bnn_loocv_unc_${TS}_100runs.csv \
    --plot-results \
    2>&1 | tee bnn_${TS}_100runs.log

echo "[$(date)] bnn finished."
