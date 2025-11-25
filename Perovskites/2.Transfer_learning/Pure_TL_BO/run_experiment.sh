#!/bin/bash

# Transfer Learning Bayesian Optimization with Stage-wise Hyperparameter Optimization
# Pretrain: 모든 파라미터 탐색 (모델 구조 + 학습 설정)
# Finetune: 학습률과 epochs만 탐색 (구조는 pretrain에서 고정)

python main.py \
    --model-type dngo \
    --num-runs 1 \
    --cost-budget 50 \
    --use-hyperparameter-bo \
    --pretrain-bo-trials 50 \
    --finetune-bo-trials 50 \
    --data-size small \
    --verbose \
    --save-images \
    --images-dir results/stage_wise_hp_bo
