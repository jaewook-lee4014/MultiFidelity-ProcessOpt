#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --job-name=bnn-dngo
#SBATCH --partition=cpu
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --signal=USR2
#SBATCH --output=bnn_dngo_output.txt
#SBATCH --error=bnn_dngo_error.txt

source ~/.bashrc

echo "[$(date)] which python: $(which python)"
python --version

WORKDIR=$(pwd)
cd "$WORKDIR"

# Python 버퍼링 비활성화
export PYTHONUNBUFFERED=1

TS=$(date +%Y%m%d_%H%M%S)

echo "[$(date)] Starting bnn and dngo jobs..."

# srun 없이 직접 백그라운드 실행 (ntasks=2이므로 CPU 충분)
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
    > bnn_${TS}_100runs.log 2>&1 &

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
    > dngo_${TS}_100runs.log 2>&1 &

wait

echo "[$(date)] All runs finished."
