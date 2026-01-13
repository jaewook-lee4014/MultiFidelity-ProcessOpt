#!/bin/bash
#SBATCH --job-name=dngo_loocv
#SBATCH --reservation=rental_9427
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_dngo_loocv_%j.out
#SBATCH --error=slurm_dngo_loocv_%j.err

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs_dngo_loocv_${TIMESTAMP}"
mkdir -p $LOG_DIR

echo "=============================================="
echo "DNGO with LOOCV - 100 Runs"
echo "Timestamp: ${TIMESTAMP}"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "=============================================="
echo ""
echo "Parameters:"
echo "  --model-type: dngo"
echo "  --num-runs: 100"
echo "  --cost-budget: 50"
echo "  --use-hyperparameter-bo: True"
echo "  --pretrain-bo-trials: 100"
echo "  --finetune-bo-trials: 50"
echo "  --data-size: large"
echo "  --use-loocv: True  <-- LOOCV enabled (Finetune only)"
echo ""

# 파라미터 기록
cat > ${LOG_DIR}/params.txt << PARAMS
Model: DNGO with LOOCV
Timestamp: ${TIMESTAMP}
SLURM Job ID: ${SLURM_JOB_ID}

Parameters:
  --model-type: dngo
  --num-runs: 100
  --cost-budget: 50
  --use-hyperparameter-bo: True
  --pretrain-bo-trials: 100
  --finetune-bo-trials: 50
  --data-size: large
  --use-loocv: True
  --save-results: True

Note: LOOCV is applied to Finetune (HIFI) stage only
PARAMS

# DNGO with LOOCV 실행
python main.py --model-type dngo \
    --num-runs 100 \
    --cost-budget 50 \
    --use-hyperparameter-bo \
    --pretrain-bo-trials 100 \
    --finetune-bo-trials 50 \
    --data-size large \
    --use-loocv \
    --save-results \
    --results-filename ${LOG_DIR}/dngo_loocv_100runs.csv \
    > ${LOG_DIR}/dngo_loocv_100runs.log 2>&1

echo ""
echo "=============================================="
echo "Experiment completed!"
echo "Results: ${LOG_DIR}/dngo_loocv_100runs.csv"
echo "=============================================="
