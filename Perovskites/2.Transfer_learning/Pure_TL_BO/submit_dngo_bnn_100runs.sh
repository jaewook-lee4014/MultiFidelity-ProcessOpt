#!/bin/bash
#SBATCH --job-name=dngo_bnn_100
#SBATCH --reservation=rental_9427
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=slurm_dngo_bnn_%j.out
#SBATCH --error=slurm_dngo_bnn_%j.err

# DNGO와 BNN 100회 실행 (통합 하이퍼파라미터 탐색 공간)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs_100runs_${TIMESTAMP}"
mkdir -p $LOG_DIR

# 공통 파라미터 - 100 runs
COMMON_PARAMS="--num-runs 100 --cost-budget 50 --use-hyperparameter-bo --pretrain-bo-trials 100 --finetune-bo-trials 50 --data-size large --save-results"

echo "=============================================="
echo "DNGO & BNN 100 Runs Experiment"
echo "Timestamp: ${TIMESTAMP}"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "=============================================="

# 요약 파일 생성
cat > ${LOG_DIR}/experiment_summary.txt << SUMMARY
============================================
Experiment Summary
============================================
Timestamp: ${TIMESTAMP}
SLURM Job ID: ${SLURM_JOB_ID}
Total Models: 2 (DNGO, BNN)
Runs per Model: 100
Total Experiments: 200

Common Parameters:
  --num-runs: 100
  --cost-budget: 50
  --use-hyperparameter-bo: True
  --pretrain-bo-trials: 100
  --finetune-bo-trials: 50
  --data-size: large

Unified Search Space:
  Pretrain:
    - hidden_layers: 1-4
    - hidden_dim: [32, 64, 128, 256]
    - learning_rate: 1e-5 ~ 1e-2
    - epochs: 50-300
  Finetune:
    - learning_rate: 1e-6 ~ 1e-3
    - epochs: 50-300

HP Optimization Interval: 50 data points
============================================
SUMMARY

# 1. DNGO (unbuffered 모드로 실행)
echo "[1/2] Starting DNGO..."
python -u main.py --model-type dngo $COMMON_PARAMS \
    --results-filename ${LOG_DIR}/dngo_100runs.csv \
    2>&1 | tee ${LOG_DIR}/dngo_100runs.log &
DNGO_PID=$!

# DNGO 파라미터 기록
cat > ${LOG_DIR}/dngo_params.txt << PARAMS
Model: DNGO
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
PID: $DNGO_PID
PARAMS

# 2. BNN
echo "[2/2] Starting BNN..."
python -u main.py --model-type bnn $COMMON_PARAMS \
    --results-filename ${LOG_DIR}/bnn_100runs.csv \
    2>&1 | tee ${LOG_DIR}/bnn_100runs.log &
BNN_PID=$!

# BNN 파라미터 기록
cat > ${LOG_DIR}/bnn_params.txt << PARAMS
Model: BNN
Timestamp: ${TIMESTAMP}
SLURM Job ID: ${SLURM_JOB_ID}
Parameters:
  --model-type: bnn
  --num-runs: 100
  --cost-budget: 50
  --use-hyperparameter-bo: True
  --pretrain-bo-trials: 100
  --finetune-bo-trials: 50
  --data-size: large
PID: $BNN_PID
PARAMS

echo ""
echo "Both models started!"
echo "PIDs: DNGO=$DNGO_PID, BNN=$BNN_PID"
echo "Log directory: ${LOG_DIR}"
echo ""
echo "Results will be saved in real-time after each run completion."
echo ""

# 모든 프로세스가 완료될 때까지 대기
wait

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
