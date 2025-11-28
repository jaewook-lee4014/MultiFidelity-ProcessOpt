#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --job-name=multi-process
#SBATCH --partition=cpu
#SBATCH --ntasks=3                 # 동시에 돌릴 작업 개수 = 3
#SBATCH --cpus-per-task=4          # 각 작업당 CPU 4개
#SBATCH --mem=16G
#SBATCH --signal=USR2
#SBATCH --output=multi_cpu_output.txt
#SBATCH --error=multi_cpu_error.txt

# 1) conda 초기화 (보통 .bashrc 에 있음)
#    cluster 환경에 따라 .bash_profile일 수도 있지만, 대부분 .bashrc
source ~/.bashrc

# 2) 로그인 노드에서 쓰는 conda 환경 활성화
#    프롬프트가 (base) 라고 되어 있었으니까 일단 base로 가정
#    별도 환경 쓰면 여기 이름만 바꾸면 됨
conda activate base

# 디버깅용: 실제로 어떤 파이썬을 쓰는지 로그에 기록
echo "[$(date)] which python: $(which python)"
python --version
python -m pip --version

# 3) 현재 디렉토리 (sbatch 제출한 위치 = main.py, requirements.txt 있는 곳)
WORKDIR=$(pwd)
cd "$WORKDIR"

# 4) requirements.txt 설치 (이미 설치돼 있으면 그냥 '이미 satisfied'만 뜸)
echo "[$(date)] Installing Python dependencies from requirements.txt..."
python -m pip install -r requirements.txt

# 공통 타임스탬프
TS=$(date +%Y%m%d_%H%M%S)

echo "[$(date)] Finished pip install. Starting jobs..."

# 1) dngo-ol
srun --exclusive -N1 -n1 python main.py \
    --model-type dngo-ol \
    --num-runs 100 \
    --cost-budget 50 \
    --use-hyperparameter-bo \
    --pretrain-bo-trials 50 \
    --finetune-bo-trials 50 \
    --use-loocv \
    --use-uncertainty-loss \
    --uncertainty-weight 0.3 \
    --save-results \
    --results-filename dngo_ol_loocv_unc_${TS}_100runs.csv \
    --plot-results \
    > dngo_ol_${TS}_100runs.log 2>&1 &

# 2) bnn
srun --exclusive -N1 -n1 python main.py \
    --model-type bnn \
    --num-runs 100 \
    --cost-budget 50 \
    --use-hyperparameter-bo \
    --pretrain-bo-trials 50 \
    --finetune-bo-trials 50 \
    --use-loocv \
    --use-uncertainty-loss \
    --uncertainty-weight 0.3 \
    --save-results \
    --results-filename bnn_loocv_unc_${TS}_100runs.csv \
    --plot-results \
    > bnn_${TS}_100runs.log 2>&1 &

# 3) dngo
srun --exclusive -N1 -n1 python main.py \
    --model-type dngo \
    --num-runs 100 \
    --cost-budget 50 \
    --use-hyperparameter-bo \
    --pretrain-bo-trials 50 \
    --finetune-bo-trials 50 \
    --use-loocv \
    --use-uncertainty-loss \
    --uncertainty-weight 0.3 \
    --save-results \
    --results-filename dngo_loocv_unc_${TS}_100runs.csv \
    --plot-results \
    > dngo_${TS}_100runs.log 2>&1 &

wait

echo "[$(date)] All runs finished."

