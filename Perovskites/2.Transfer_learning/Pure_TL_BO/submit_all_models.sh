#!/bin/bash
#SBATCH --job-name=mfbo_4models
#SBATCH --reservation=rental_9427
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=slurm_4models_%j.out
#SBATCH --error=slurm_4models_%j.err

# 4개 모델 100런씩 병렬 실행
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs_${TIMESTAMP}"
mkdir -p $LOG_DIR

# 공통 파라미터
COMMON_PARAMS="--num-runs 100 --cost-budget 50 --use-hyperparameter-bo --pretrain-bo-trials 100 --finetune-bo-trials 50 --data-size large --save-results"

echo "=============================================="
echo "4 Models x 100 Runs Parallel Execution"
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
Runs per Model: 100
Total Experiments: 400

Common Parameters:
  --cost-budget: 50
  --use-hyperparameter-bo: True
  --pretrain-bo-trials: 100
  --finetune-bo-trials: 50
  --data-size: large

Models: DNGO, BNN, DNGO-OL, BNN-OL
============================================
SUMMARY

# 1. DNGO
echo "[1/4] Starting DNGO..."
python main.py --model-type dngo $COMMON_PARAMS \
    --results-filename ${LOG_DIR}/dngo_100runs.csv \
    > ${LOG_DIR}/dngo_100runs.log 2>&1 &
DNGO_PID=$!

# 파라미터 기록
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
echo "[2/4] Starting BNN..."
python main.py --model-type bnn $COMMON_PARAMS \
    --results-filename ${LOG_DIR}/bnn_100runs.csv \
    > ${LOG_DIR}/bnn_100runs.log 2>&1 &
BNN_PID=$!

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

# 3. DNGO-OL
echo "[3/4] Starting DNGO-OL..."
python main.py --model-type dngo-ol $COMMON_PARAMS \
    --results-filename ${LOG_DIR}/dngo_ol_100runs.csv \
    > ${LOG_DIR}/dngo_ol_100runs.log 2>&1 &
DNGO_OL_PID=$!

cat > ${LOG_DIR}/dngo_ol_params.txt << PARAMS
Model: DNGO-OL
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
PARAMS

# 4. BNN-OL
echo "[4/4] Starting BNN-OL..."
python main.py --model-type bnn-ol $COMMON_PARAMS \
    --results-filename ${LOG_DIR}/bnn_ol_100runs.csv \
    > ${LOG_DIR}/bnn_ol_100runs.log 2>&1 &
BNN_OL_PID=$!

cat > ${LOG_DIR}/bnn_ol_params.txt << PARAMS
Model: BNN-OL
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
