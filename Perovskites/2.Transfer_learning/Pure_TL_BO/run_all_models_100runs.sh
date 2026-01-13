#!/bin/bash
# 4개 모델 100런씩 병렬 실행 스크립트
# 생성일: 2024-12-03

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs_${TIMESTAMP}"
mkdir -p $LOG_DIR

# 공통 파라미터
COMMON_PARAMS="--num-runs 100 --cost-budget 50 --use-hyperparameter-bo --pretrain-bo-trials 100 --finetune-bo-trials 50 --data-size large --save-results"

echo "=============================================="
echo "4 Models x 100 Runs Parallel Execution"
echo "Timestamp: ${TIMESTAMP}"
echo "=============================================="
echo ""
echo "Common Parameters:"
echo "  --num-runs: 100"
echo "  --cost-budget: 50"
echo "  --use-hyperparameter-bo: True"
echo "  --pretrain-bo-trials: 100"
echo "  --finetune-bo-trials: 50"
echo "  --data-size: large"
echo ""

# 1. DNGO
echo "[1/4] Starting DNGO..."
nohup python main.py --model-type dngo $COMMON_PARAMS \
    --results-filename ${LOG_DIR}/dngo_100runs.csv \
    > ${LOG_DIR}/dngo_100runs.log 2>&1 &
DNGO_PID=$!
echo "  PID: $DNGO_PID"
echo "  Log: ${LOG_DIR}/dngo_100runs.log"

# 파라미터 기록
cat > ${LOG_DIR}/dngo_params.txt << PARAMS
Model: DNGO
Timestamp: ${TIMESTAMP}
Parameters:
  --model-type: dngo
  --num-runs: 100
  --cost-budget: 50
  --use-hyperparameter-bo: True
  --pretrain-bo-trials: 100
  --finetune-bo-trials: 50
  --data-size: large
  --save-results: True
  --results-filename: ${LOG_DIR}/dngo_100runs.csv
PID: $DNGO_PID
PARAMS

# 2. BNN
echo "[2/4] Starting BNN..."
nohup python main.py --model-type bnn $COMMON_PARAMS \
    --results-filename ${LOG_DIR}/bnn_100runs.csv \
    > ${LOG_DIR}/bnn_100runs.log 2>&1 &
BNN_PID=$!
echo "  PID: $BNN_PID"
echo "  Log: ${LOG_DIR}/bnn_100runs.log"

cat > ${LOG_DIR}/bnn_params.txt << PARAMS
Model: BNN
Timestamp: ${TIMESTAMP}
Parameters:
  --model-type: bnn
  --num-runs: 100
  --cost-budget: 50
  --use-hyperparameter-bo: True
  --pretrain-bo-trials: 100
  --finetune-bo-trials: 50
  --data-size: large
  --save-results: True
  --results-filename: ${LOG_DIR}/bnn_100runs.csv
PID: $BNN_PID
PARAMS

# 3. DNGO-OL
echo "[3/4] Starting DNGO-OL..."
nohup python main.py --model-type dngo-ol $COMMON_PARAMS \
    --results-filename ${LOG_DIR}/dngo_ol_100runs.csv \
    > ${LOG_DIR}/dngo_ol_100runs.log 2>&1 &
DNGO_OL_PID=$!
echo "  PID: $DNGO_OL_PID"
echo "  Log: ${LOG_DIR}/dngo_ol_100runs.log"

cat > ${LOG_DIR}/dngo_ol_params.txt << PARAMS
Model: DNGO-OL
Timestamp: ${TIMESTAMP}
Parameters:
  --model-type: dngo-ol
  --num-runs: 100
  --cost-budget: 50
  --use-hyperparameter-bo: True
  --pretrain-bo-trials: 100
  --finetune-bo-trials: 50
  --data-size: large
  --save-results: True
  --results-filename: ${LOG_DIR}/dngo_ol_100runs.csv
PID: $DNGO_OL_PID
PARAMS

# 4. BNN-OL
echo "[4/4] Starting BNN-OL..."
nohup python main.py --model-type bnn-ol $COMMON_PARAMS \
    --results-filename ${LOG_DIR}/bnn_ol_100runs.csv \
    > ${LOG_DIR}/bnn_ol_100runs.log 2>&1 &
BNN_OL_PID=$!
echo "  PID: $BNN_OL_PID"
echo "  Log: ${LOG_DIR}/bnn_ol_100runs.log"

cat > ${LOG_DIR}/bnn_ol_params.txt << PARAMS
Model: BNN-OL
Timestamp: ${TIMESTAMP}
Parameters:
  --model-type: bnn-ol
  --num-runs: 100
  --cost-budget: 50
  --use-hyperparameter-bo: True
  --pretrain-bo-trials: 100
  --finetune-bo-trials: 50
  --data-size: large
  --save-results: True
  --results-filename: ${LOG_DIR}/bnn_ol_100runs.csv
PID: $BNN_OL_PID
PARAMS

# 요약 파일 생성
cat > ${LOG_DIR}/experiment_summary.txt << SUMMARY
============================================
Experiment Summary
============================================
Timestamp: ${TIMESTAMP}
Total Models: 4
Runs per Model: 100
Total Experiments: 400

Common Parameters:
  --cost-budget: 50
  --use-hyperparameter-bo: True
  --pretrain-bo-trials: 100
  --finetune-bo-trials: 50
  --data-size: large

Models:
  1. DNGO      (PID: $DNGO_PID)
  2. BNN       (PID: $BNN_PID)
  3. DNGO-OL   (PID: $DNGO_OL_PID)
  4. BNN-OL    (PID: $BNN_OL_PID)

Output Files:
  - ${LOG_DIR}/dngo_100runs.csv
  - ${LOG_DIR}/bnn_100runs.csv
  - ${LOG_DIR}/dngo_ol_100runs.csv
  - ${LOG_DIR}/bnn_ol_100runs.csv

Log Files:
  - ${LOG_DIR}/dngo_100runs.log
  - ${LOG_DIR}/bnn_100runs.log
  - ${LOG_DIR}/dngo_ol_100runs.log
  - ${LOG_DIR}/bnn_ol_100runs.log

Monitor Progress:
  tail -f ${LOG_DIR}/*.log
  ps aux | grep "python main.py"
============================================
SUMMARY

echo ""
echo "=============================================="
echo "All 4 models started in background!"
echo "Log directory: ${LOG_DIR}"
echo ""
echo "Monitor progress with:"
echo "  tail -f ${LOG_DIR}/*.log"
echo "  ps aux | grep 'python main.py'"
echo "=============================================="
