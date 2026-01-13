#!/bin/bash
#SBATCH --job-name=bnn_updated_100
#SBATCH --reservation=rental_9427
#SBATCH --oversubscribe
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_bnn_updated_%j.out
#SBATCH --error=slurm_bnn_updated_%j.err

# ============================================================================
# 업데이트된 BNN 100회 최적화 실험
# ============================================================================
# 변경 사항:
# - Scale Mixture Prior 적용 (prior_pi=0.5, prior_sigma1=1.0, prior_sigma2=0.002)
# - hidden_dims: [64, 64, 64] (3층 구조)
# - kl_weight: 0.5 (최적화된 기본값)
# - transfer_mode: consistent_bnn
# - HP-BO kl_weight 탐색 범위: 0.2~2.0
# ============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs_bnn_updated_100runs_${TIMESTAMP}"
mkdir -p $LOG_DIR

echo "=============================================="
echo "Updated BNN 100 Runs Experiment"
echo "Timestamp: ${TIMESTAMP}"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "=============================================="

# 실험 설정 요약 파일 생성
cat > ${LOG_DIR}/experiment_config.txt << CONFIG
============================================
Updated BNN Experiment Configuration
============================================
Timestamp: ${TIMESTAMP}
SLURM Job ID: ${SLURM_JOB_ID}

Model: BNN (Bayesian Neural Network)
Runs: 100
Cost Budget: 50

Key Updates (from 4-model comparison results):
  - Scale Mixture Prior (Blundell et al. 2015):
    * prior_pi: 0.5
    * prior_sigma1: 1.0
    * prior_sigma2: 0.002
  - hidden_dims: [64, 64, 64] (3-layer structure)
  - kl_weight: 0.5 (optimized default, was 1.0)
  - transfer_mode: consistent_bnn

HP-BO Configuration:
  - use_hyperparameter_bo: True
  - pretrain_bo_trials: 100
  - finetune_bo_trials: 50
  - kl_weight search range: 0.2~2.0 (focused range)
  - epochs search range: 20~500
  - batch_size options: [8, 16, 32, 64, 128, 256, 512]

Expected Improvements (vs old BNN):
  - RMSE: ~1.9 -> ~0.68 (65% reduction)
  - R²: ~-2.0 -> ~0.62 (significant improvement)
  - Spearman: ~0.19 -> ~0.83 (4x improvement)
============================================
CONFIG

echo ""
echo "Configuration saved to: ${LOG_DIR}/experiment_config.txt"
echo ""

# BNN 100회 실행 (실시간 로그 출력)
echo "[Starting] Updated BNN 100 runs..."
export PYTHONUNBUFFERED=1
stdbuf -oL -eL python -u main.py \
    --model-type bnn \
    --num-runs 100 \
    --cost-budget 50 \
    --use-hyperparameter-bo \
    --pretrain-bo-trials 100 \
    --finetune-bo-trials 50 \
    --data-size large \
    --save-results \
    --results-filename ${LOG_DIR}/bnn_updated_100runs.csv \
    2>&1 | stdbuf -oL tee ${LOG_DIR}/bnn_updated_100runs.log

echo ""
echo "=============================================="
echo "Experiment completed!"
echo "Results saved to: ${LOG_DIR}/bnn_updated_100runs.csv"
echo "Log saved to: ${LOG_DIR}/bnn_updated_100runs.log"
echo "=============================================="
