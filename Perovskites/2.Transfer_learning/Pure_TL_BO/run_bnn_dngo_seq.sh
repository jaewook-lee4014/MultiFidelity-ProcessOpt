#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --job-name=bnn-dngo-seq
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --signal=USR2
#SBATCH --output=bnn_dngo_seq_output.txt
#SBATCH --error=bnn_dngo_seq_error.txt

source ~/.bashrc

echo "[$(date)] which python: $(which python)"
python --version

WORKDIR=$(pwd)
cd "$WORKDIR"

TS=$(date +%Y%m%d_%H%M%S)

echo "[$(date)] Starting bnn..."

# 1) bnn - 순차 실행
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

echo "[$(date)] bnn finished. Starting dngo..."

# 2) dngo - 순차 실행
python main.py \
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
    --results-filename dngo_loocv_unc_${TS}_100runs.csv \
    --plot-results \
    2>&1 | tee dngo_${TS}_100runs.log

echo "[$(date)] All runs finished."
