#!/bin/bash
#SBATCH --job-name=mfbo_4models_3runs
#SBATCH --reservation=rental_9427
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_4models_3runs_%j.out
#SBATCH --error=slurm_4models_3runs_%j.err

# 4개 모델 3런씩 병렬 실행 (테스트용)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs_3runs_${TIMESTAMP}"
mkdir -p $LOG_DIR

# 공통 파라미터 - 3 runs로 변경
COMMON_PARAMS="--num-runs 3 --cost-budget 50 --use-hyperparameter-bo --pretrain-bo-trials 100 --finetune-bo-trials 50 --data-size large --save-results"

echo "=============================================="
echo "4 Models x 3 Runs Parallel Execution (Test)"
echo "Timestamp: ${TIMESTAMP}"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "=============================================="

# 요약 파일 먼저 생성
cat > ${LOG_DIR}/experiment_summary.txt << SUMMARY
============================================
Experiment Summary
============================================
Timestamp: ${TIMESTAMP}
SLURM Job ID: ${SLURM_JOB_ID}
Total Models: 4
Runs per Model: 3
Total Experiments: 12

Common Parameters:
  --cost-budget: 50
  --use-hyperparameter-bo: True
  --pretrain-bo-trials: 100
  --finetune-bo-trials: 50
  --data-size: large

Models: DNGO, BNN, DNGO-OL, BNN-OL
============================================
SUMMARY

# 1. DNGO (unbuffered 모드로 실행)
echo "[1/4] Starting DNGO..."
python -u main.py --model-type dngo $COMMON_PARAMS \
    --results-filename ${LOG_DIR}/dngo_3runs.csv \
    2>&1 | tee ${LOG_DIR}/dngo_3runs.log &
DNGO_PID=$!

# 파라미터 기록
cat > ${LOG_DIR}/dngo_params.txt << PARAMS
Model: DNGO
Timestamp: ${TIMESTAMP}
SLURM Job ID: ${SLURM_JOB_ID}
Parameters:
  --model-type: dngo
  --num-runs: 3
  --cost-budget: 50
  --use-hyperparameter-bo: True
  --pretrain-bo-trials: 100
  --finetune-bo-trials: 50
  --data-size: large
PID: $DNGO_PID
PARAMS

# 2. BNN
echo "[2/4] Starting BNN..."
python -u main.py --model-type bnn $COMMON_PARAMS \
    --results-filename ${LOG_DIR}/bnn_3runs.csv \
    2>&1 | tee ${LOG_DIR}/bnn_3runs.log &
BNN_PID=$!

cat > ${LOG_DIR}/bnn_params.txt << PARAMS
Model: BNN
Timestamp: ${TIMESTAMP}
SLURM Job ID: ${SLURM_JOB_ID}
Parameters:
  --model-type: bnn
  --num-runs: 3
  --cost-budget: 50
  --use-hyperparameter-bo: True
  --pretrain-bo-trials: 100
  --finetune-bo-trials: 50
  --data-size: large
PID: $BNN_PID
PARAMS

# 3. DNGO-OL
echo "[3/4] Starting DNGO-OL..."
python -u main.py --model-type dngo-ol $COMMON_PARAMS \
    --results-filename ${LOG_DIR}/dngo_ol_3runs.csv \
    2>&1 | tee ${LOG_DIR}/dngo_ol_3runs.log &
DNGO_OL_PID=$!

cat > ${LOG_DIR}/dngo_ol_params.txt << PARAMS
Model: DNGO-OL
Timestamp: ${TIMESTAMP}
SLURM Job ID: ${SLURM_JOB_ID}
Parameters:
  --model-type: dngo-ol
  --num-runs: 3
  --cost-budget: 50
  --use-hyperparameter-bo: True
  --pretrain-bo-trials: 100
  --finetune-bo-trials: 50
  --data-size: large
PID: $DNGO_OL_PID
PARAMS

# 4. BNN-OL
echo "[4/4] Starting BNN-OL..."
python -u main.py --model-type bnn-ol $COMMON_PARAMS \
    --results-filename ${LOG_DIR}/bnn_ol_3runs.csv \
    2>&1 | tee ${LOG_DIR}/bnn_ol_3runs.log &
BNN_OL_PID=$!

cat > ${LOG_DIR}/bnn_ol_params.txt << PARAMS
Model: BNN-OL
Timestamp: ${TIMESTAMP}
SLURM Job ID: ${SLURM_JOB_ID}
Parameters:
  --model-type: bnn-ol
  --num-runs: 3
  --cost-budget: 50
  --use-hyperparameter-bo: True
  --pretrain-bo-trials: 100
  --finetune-bo-trials: 50
  --data-size: large
PID: $BNN_OL_PID
PARAMS

echo ""
echo "All 4 models started!"
echo "PIDs: DNGO=$DNGO_PID, BNN=$BNN_PID, DNGO-OL=$DNGO_OL_PID, BNN-OL=$BNN_OL_PID"
echo "Log directory: ${LOG_DIR}"
echo ""

# 모든 프로세스가 완료될 때까지 대기
wait

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
