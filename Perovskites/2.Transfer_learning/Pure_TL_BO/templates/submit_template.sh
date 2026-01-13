#!/bin/bash
# ============================================================================
# SLURM Job Submission Template
# ============================================================================
# 사용법:
#   1. 이 파일을 복사: cp templates/submit_template.sh submit_my_job.sh
#   2. 아래 설정 수정
#   3. 제출: sbatch submit_my_job.sh
# ============================================================================

#SBATCH --job-name=JOB_NAME_HERE        # Job 이름 (수정 필요)
#SBATCH --reservation=rental_9427       # Rental 노드 예약
#SBATCH --cpus-per-task=10              # CPU 코어 수
#SBATCH --mem=80G                       # 메모리
#SBATCH --gres=gpu:1                    # GPU 수
#SBATCH --time=24:00:00                 # 최대 실행 시간
#SBATCH --output=slurm_%x_%j.out        # stdout (%x=job name, %j=job id)
#SBATCH --error=slurm_%x_%j.err         # stderr

# ============================================================================
# 설정 변수 (수정 필요)
# ============================================================================
MODEL_TYPE="bnn"                        # 모델 타입: dngo, bnn, dngo-ol, bnn-ol
NUM_RUNS=100                            # 실행 횟수
COST_BUDGET=50                          # 비용 예산
USE_HP_BO=true                          # 하이퍼파라미터 BO 사용 여부
PRETRAIN_BO_TRIALS=100                  # Pretrain BO 시행 횟수
FINETUNE_BO_TRIALS=50                   # Finetune BO 시행 횟수
DATA_SIZE="large"                       # 데이터 크기: small, medium, large

# ============================================================================
# 로그 디렉토리 설정
# ============================================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs_${MODEL_TYPE}_${NUM_RUNS}runs_${TIMESTAMP}"
mkdir -p $LOG_DIR

echo "=============================================="
echo "Experiment: ${MODEL_TYPE} ${NUM_RUNS} runs"
echo "Timestamp: ${TIMESTAMP}"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "=============================================="

# ============================================================================
# 실험 설정 저장
# ============================================================================
cat > ${LOG_DIR}/experiment_config.txt << CONFIG
============================================
Experiment Configuration
============================================
Timestamp: ${TIMESTAMP}
SLURM Job ID: ${SLURM_JOB_ID}

Model: ${MODEL_TYPE}
Runs: ${NUM_RUNS}
Cost Budget: ${COST_BUDGET}
Data Size: ${DATA_SIZE}

HP-BO Configuration:
  - use_hyperparameter_bo: ${USE_HP_BO}
  - pretrain_bo_trials: ${PRETRAIN_BO_TRIALS}
  - finetune_bo_trials: ${FINETUNE_BO_TRIALS}

Model-specific defaults (from code):
  - hidden_dims: [64, 64, 64]
  - kl_weight: 0.5
  - Scale Mixture Prior: pi=0.5, sigma1=1.0, sigma2=0.002
  - transfer_mode: consistent_bnn
============================================
CONFIG

echo "Configuration saved to: ${LOG_DIR}/experiment_config.txt"

# ============================================================================
# 실험 실행 (실시간 로그 출력)
# ============================================================================
echo "[Starting] ${MODEL_TYPE} ${NUM_RUNS} runs..."

# 실시간 로그를 위한 환경 설정
export PYTHONUNBUFFERED=1

# HP-BO 옵션 설정
if [ "$USE_HP_BO" = true ]; then
    HP_BO_OPTS="--use-hyperparameter-bo --pretrain-bo-trials ${PRETRAIN_BO_TRIALS} --finetune-bo-trials ${FINETUNE_BO_TRIALS}"
else
    HP_BO_OPTS=""
fi

# 실행
stdbuf -oL -eL python -u main.py \
    --model-type ${MODEL_TYPE} \
    --num-runs ${NUM_RUNS} \
    --cost-budget ${COST_BUDGET} \
    --data-size ${DATA_SIZE} \
    --save-results \
    --results-filename ${LOG_DIR}/${MODEL_TYPE}_${NUM_RUNS}runs.csv \
    ${HP_BO_OPTS} \
    2>&1 | stdbuf -oL tee ${LOG_DIR}/${MODEL_TYPE}_${NUM_RUNS}runs.log

# ============================================================================
# 완료 메시지
# ============================================================================
echo ""
echo "=============================================="
echo "Experiment completed!"
echo "Results: ${LOG_DIR}/${MODEL_TYPE}_${NUM_RUNS}runs.csv"
echo "Log: ${LOG_DIR}/${MODEL_TYPE}_${NUM_RUNS}runs.log"
echo "=============================================="
