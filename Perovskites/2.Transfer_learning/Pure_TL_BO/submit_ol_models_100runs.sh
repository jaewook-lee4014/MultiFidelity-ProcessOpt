#!/bin/bash
#SBATCH --job-name=ol_models_100
#SBATCH --reservation=rental_9427
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=slurm_ol_models_%j.out
#SBATCH --error=slurm_ol_models_%j.err

# DNGO-OL과 BNN-OL 100회 실행 (확장된 하이퍼파라미터 탐색 공간)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs_ol_100runs_${TIMESTAMP}"
mkdir -p $LOG_DIR

# 공통 파라미터 - 100 runs
COMMON_PARAMS="--num-runs 100 --cost-budget 50 --use-hyperparameter-bo --pretrain-bo-trials 100 --finetune-bo-trials 50 --data-size large --save-results"

echo "=============================================="
echo "DNGO-OL & BNN-OL 100 Runs Experiment"
echo "Timestamp: ${TIMESTAMP}"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "=============================================="

# 요약 파일 생성
cat > ${LOG_DIR}/experiment_summary.txt << SUMMARY
============================================
Experiment Summary - Online Learning Models
============================================
Timestamp: ${TIMESTAMP}
SLURM Job ID: ${SLURM_JOB_ID}
Total Models: 2 (DNGO-OL, BNN-OL)
Runs per Model: 100
Total Experiments: 200

Common Parameters:
  --num-runs: 100
  --cost-budget: 50
  --use-hyperparameter-bo: True
  --pretrain-bo-trials: 100
  --finetune-bo-trials: 50
  --data-size: large

Expanded Search Space (Updated):
  Pretrain:
    - hidden_layers: 1-4
    - hidden_dim: [32, 64, 128, 256]
    - learning_rate: 1e-5 ~ 1e-2
    - epochs: 20-500 (expanded from 50-300)
    - batch_size: [8, 16, 32, 64, 128, 256, 512] (new)
  Finetune:
    - learning_rate: 1e-6 ~ 1e-3
    - epochs: 20-500 (expanded from 50-300)
    - batch_size: [8, 16, 32, 64, 128, 256, 512] (new)

HP Optimization Interval: 50 data points
============================================
SUMMARY

# 1. DNGO-OL (unbuffered 모드로 실행)
echo "[1/2] Starting DNGO-OL..."
python -u main.py --model-type dngo-ol $COMMON_PARAMS \
    --results-filename ${LOG_DIR}/dngo_ol_100runs.csv \
    2>&1 | tee ${LOG_DIR}/dngo_ol_100runs.log &
DNGO_OL_PID=$!

# DNGO-OL 파라미터 기록
cat > ${LOG_DIR}/dngo_ol_params.txt << PARAMS
Model: DNGO-OL (Online Learning)
Timestamp: ${TIMESTAMP}
SLURM Job ID: ${SLURM_JOB_ID}
Parameters:
  --model-type: dngo-ol
  --num-runs: 100
  --cost-budget: 50
  --use-hyperparameter-bo: True
  --pretrain-bo-trials: 100
  --finetune-bo-trials: 50
  --data-size: large
PID: $DNGO_OL_PID

Notes:
  - Uses expanded epochs range: 20-500
  - Uses expanded batch_size: [8, 16, 32, 64, 128, 256, 512]
PARAMS

# 2. BNN-OL
echo "[2/2] Starting BNN-OL..."
python -u main.py --model-type bnn-ol $COMMON_PARAMS \
    --results-filename ${LOG_DIR}/bnn_ol_100runs.csv \
    2>&1 | tee ${LOG_DIR}/bnn_ol_100runs.log &
BNN_OL_PID=$!

# BNN-OL 파라미터 기록
cat > ${LOG_DIR}/bnn_ol_params.txt << PARAMS
Model: BNN-OL (Online Learning)
Timestamp: ${TIMESTAMP}
SLURM Job ID: ${SLURM_JOB_ID}
Parameters:
  --model-type: bnn-ol
  --num-runs: 100
  --cost-budget: 50
  --use-hyperparameter-bo: True
  --pretrain-bo-trials: 100
  --finetune-bo-trials: 50
  --data-size: large
PID: $BNN_OL_PID

Notes:
  - Uses expanded epochs range: 20-500
  - Uses expanded batch_size: [8, 16, 32, 64, 128, 256, 512]
PARAMS

echo ""
echo "Both OL models started!"
echo "PIDs: DNGO-OL=$DNGO_OL_PID, BNN-OL=$BNN_OL_PID"
echo "Log directory: ${LOG_DIR}"
echo ""
echo "Results will be saved in real-time after each run completion."
echo ""

# 모든 프로세스가 완료될 때까지 대기
wait

echo ""
echo "=============================================="
echo "All OL experiments completed!"
echo "=============================================="
