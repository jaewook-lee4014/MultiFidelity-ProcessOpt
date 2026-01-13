#!/bin/bash
#SBATCH --job-name=mf_bo_loocv
#SBATCH --reservation=rental_9427
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=slurm_bo_loocv_%j.out
#SBATCH --error=slurm_bo_loocv_%j.err

echo "========================================"
echo "Bayesian Optimization with LOO-CV (No Data Leakage)"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start Time: $(date)"
echo "========================================"

# Conda 환경 활성화
source ~/.bashrc
conda activate pytorch_env

# GPU 확인
nvidia-smi

# 작업 디렉토리
cd /cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/Pure_TL_BO

# 결과 디렉토리 생성
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SAVE_DIR="results/bo_loocv_${TIMESTAMP}"
mkdir -p ${SAVE_DIR}

# Device 설정 (GPU 사용 가능하면 cuda, 아니면 cpu)
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    DEVICE="cuda"
else
    DEVICE="cpu"
fi

echo ""
echo "========================================"
echo "데이터 유출 방지 LOO-CV 기반 BO"
echo "========================================"
echo ""
echo "Validation Method: Leave-One-Out CV on HF 9 samples"
echo "  - Each fold: 8 HF train, 1 HF val"
echo "  - 9 folds per data split"
echo "  - 10 outer folds (different data splits)"
echo ""
echo "Configuration:"
echo "  N_TRIALS: 300"
echo "  N_OUTER_FOLDS: 10"
echo "  LOO-CV: 9 HF samples"
echo "  LF samples: 72"
echo "  DEVICE: ${DEVICE}"
echo "  SAVE_DIR: ${SAVE_DIR}"
echo ""
echo "이전 BO와의 차이점:"
echo "  - 이전: 전체 192개 테스트셋으로 R² 계산 (데이터 유출!)"
echo "  - 현재: HF 9개에 대한 LOO-CV R² 계산 (유출 없음)"
echo ""
echo "========================================"

# 실험 실행 (unbuffered 모드)
python -u experiments/large_scale_bo_loocv.py \
    --n-trials 300 \
    --n-outer-folds 10 \
    --device ${DEVICE} \
    --save-dir ${SAVE_DIR}

echo ""
echo "========================================"
echo "End Time: $(date)"
echo "========================================"
