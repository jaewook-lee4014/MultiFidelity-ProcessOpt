#!/bin/bash
#SBATCH --job-name=mf_large_bo
#SBATCH --reservation=rental_9427
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=slurm_large_bo_%j.out
#SBATCH --error=slurm_large_bo_%j.err

echo "========================================"
echo "Large-Scale Bayesian Optimization Experiment"
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
SAVE_DIR="results/large_scale_bo_v2_${TIMESTAMP}"
mkdir -p ${SAVE_DIR}

# Device 설정 (GPU 사용 가능하면 cuda, 아니면 cpu)
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    DEVICE="cuda"
else
    DEVICE="cpu"
fi

echo ""
echo "Configuration:"
echo "  N_TRIALS: 500"
echo "  N_FOLDS: 10 (with fixed seeds: 42,123,456,789,1011,1213,1415,1617,1819,2021)"
echo "  N_LOFI: 72"
echo "  N_HIFI: 9"
echo "  DEVICE: ${DEVICE}"
echo "  N_JOBS: 10"
echo "  SAVE_DIR: ${SAVE_DIR}"
echo ""

# 실험 실행 (unbuffered 모드)
python -u experiments/large_scale_bo_experiment.py \
    --n-trials 500 \
    --n-folds 10 \
    --device ${DEVICE} \
    --n-jobs 10 \
    --save-dir ${SAVE_DIR}

echo ""
echo "========================================"
echo "End Time: $(date)"
echo "========================================"
