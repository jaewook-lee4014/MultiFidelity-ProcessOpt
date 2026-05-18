#!/usr/bin/env python
"""
DNGO_Joint 상세 분석 스크립트

전이학습 전/후 성능 비교:
1. LF 데이터만 학습 후 성능 (pretrain)
2. LF+HF 데이터로 joint 학습 후 성능 (finetune)
3. 각 단계에서 LF/HF 테스트셋 성능 측정
"""

import numpy as np
import torch
import json
from datetime import datetime
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Local imports
from synthetic_functions_mfbo import branin_hf, branin_lf, SCENARIOS
from mf_uq_models import MF_MODEL_REGISTRY

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

def generate_data(seed, n_lf, n_hf, n_test, alpha, dim=2):
    """데이터 생성"""
    np.random.seed(seed)

    # Training data
    X_lf = np.random.uniform(0, 1, (n_lf, dim))
    y_lf = branin_lf(X_lf, alpha).flatten()

    X_hf = np.random.uniform(0, 1, (n_hf, dim))
    y_hf = branin_hf(X_hf).flatten()

    # Test data (separate for LF and HF evaluation)
    np.random.seed(seed + 10000)
    X_test = np.random.uniform(0, 1, (n_test, dim))
    y_test_hf = branin_hf(X_test).flatten()  # HF ground truth
    y_test_lf = branin_lf(X_test, alpha).flatten()  # LF ground truth

    return {
        'X_lf': X_lf, 'y_lf': y_lf,
        'X_hf': X_hf, 'y_hf': y_hf,
        'X_test': X_test,
        'y_test_hf': y_test_hf,
        'y_test_lf': y_test_lf
    }


def compute_metrics(y_true, y_pred):
    """평가 지표 계산"""
    return {
        'r2': float(r2_score(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred)))
    }


def analyze_dngo_joint_detailed(seed=0, n_lf=50, n_hf=45, n_test=200, alpha=0.8):
    """DNGO_Joint 상세 분석"""

    print(f"\n{'='*70}")
    print(f"DNGO_Joint 상세 분석 (seed={seed})")
    print(f"{'='*70}")
    print(f"LF 데이터: {n_lf}개, HF 데이터: {n_hf}개, Test: {n_test}개")
    print(f"Alpha (LF-HF 유사도): {alpha}")

    # 데이터 생성
    data = generate_data(seed, n_lf, n_hf, n_test, alpha)

    results = {
        'seed': seed,
        'n_lf': n_lf,
        'n_hf': n_hf,
        'n_test': n_test,
        'alpha': alpha,
        'stages': {}
    }

    # DNGO_Joint 모델 생성
    model_class = MF_MODEL_REGISTRY['DNGO_Joint']

    # =========================================================================
    # Stage 1: LF 데이터만으로 학습 (Pretrain 시뮬레이션)
    # =========================================================================
    print(f"\n{'-'*70}")
    print("Stage 1: LF 데이터만 학습 (Pretrain)")
    print(f"{'-'*70}")

    # DNGO_Joint는 LF+HF를 같이 학습하므로, HF를 최소한으로 넣어서 pretrain 효과 확인
    model_lf_only = model_class(input_dim=2, hidden_dim=64, epochs=300)

    # LF만 학습 (HF는 1개만 넣어서 최소화)
    X_hf_min = data['X_hf'][:1]
    y_hf_min = data['y_hf'][:1]
    model_lf_only.fit(data['X_lf'], data['y_lf'], X_hf_min, y_hf_min)

    # LF 데이터 예측
    pred_lf_train, std_lf_train = model_lf_only.predict(data['X_lf'])
    pred_lf_test, std_lf_test = model_lf_only.predict(data['X_test'])

    # HF 데이터 예측 (Transfer 없이)
    pred_hf_test_stage1, std_hf_test_stage1 = model_lf_only.predict(data['X_test'])

    stage1_metrics = {
        'lf_train': compute_metrics(data['y_lf'], pred_lf_train.flatten()),
        'lf_test': compute_metrics(data['y_test_lf'], pred_lf_test.flatten()),
        'hf_test_before_transfer': compute_metrics(data['y_test_hf'], pred_hf_test_stage1.flatten()),
        'mean_uncertainty': float(np.mean(std_lf_test))
    }
    results['stages']['pretrain_lf_only'] = stage1_metrics

    print(f"  LF Train R²: {stage1_metrics['lf_train']['r2']:.4f}")
    print(f"  LF Test R²:  {stage1_metrics['lf_test']['r2']:.4f}")
    print(f"  HF Test R² (before transfer): {stage1_metrics['hf_test_before_transfer']['r2']:.4f}")
    print(f"  Mean Uncertainty: {stage1_metrics['mean_uncertainty']:.4f}")

    # =========================================================================
    # Stage 2: LF + HF 데이터로 Joint 학습 (Full Training)
    # =========================================================================
    print(f"\n{'-'*70}")
    print("Stage 2: LF + HF Joint 학습")
    print(f"{'-'*70}")

    model_joint = model_class(input_dim=2, hidden_dim=64, epochs=300)
    model_joint.fit(data['X_lf'], data['y_lf'], data['X_hf'], data['y_hf'])

    # 각 데이터셋에 대한 예측
    pred_lf_train_joint, _ = model_joint.predict(data['X_lf'])
    pred_hf_train_joint, _ = model_joint.predict(data['X_hf'])
    pred_test_joint, std_test_joint = model_joint.predict(data['X_test'])

    stage2_metrics = {
        'lf_train': compute_metrics(data['y_lf'], pred_lf_train_joint.flatten()),
        'hf_train': compute_metrics(data['y_hf'], pred_hf_train_joint.flatten()),
        'lf_test': compute_metrics(data['y_test_lf'], pred_test_joint.flatten()),
        'hf_test': compute_metrics(data['y_test_hf'], pred_test_joint.flatten()),
        'mean_uncertainty': float(np.mean(std_test_joint))
    }
    results['stages']['joint_training'] = stage2_metrics

    print(f"  LF Train R²: {stage2_metrics['lf_train']['r2']:.4f}")
    print(f"  HF Train R²: {stage2_metrics['hf_train']['r2']:.4f}")
    print(f"  LF Test R²:  {stage2_metrics['lf_test']['r2']:.4f}")
    print(f"  HF Test R²:  {stage2_metrics['hf_test']['r2']:.4f}")
    print(f"  Mean Uncertainty: {stage2_metrics['mean_uncertainty']:.4f}")

    # =========================================================================
    # Stage 3: HF 데이터 증가에 따른 성능 변화
    # =========================================================================
    print(f"\n{'-'*70}")
    print("Stage 3: HF 데이터 증가에 따른 성능 변화")
    print(f"{'-'*70}")

    hf_increments = [2, 5, 10, 20, 30, 45]
    hf_progression = []

    for n in hf_increments:
        if n > n_hf:
            continue

        model_inc = model_class(input_dim=2, hidden_dim=64, epochs=300)
        model_inc.fit(data['X_lf'], data['y_lf'], data['X_hf'][:n], data['y_hf'][:n])

        pred_test, std_test = model_inc.predict(data['X_test'])

        metrics = {
            'n_hf': n,
            'hf_train': compute_metrics(data['y_hf'][:n], model_inc.predict(data['X_hf'][:n])[0].flatten()),
            'hf_test': compute_metrics(data['y_test_hf'], pred_test.flatten()),
            'lf_test': compute_metrics(data['y_test_lf'], pred_test.flatten()),
            'mean_uncertainty': float(np.mean(std_test))
        }
        hf_progression.append(metrics)

        print(f"  n_hf={n:3d}: HF Train R²={metrics['hf_train']['r2']:.4f}, "
              f"HF Test R²={metrics['hf_test']['r2']:.4f}, "
              f"Uncertainty={metrics['mean_uncertainty']:.4f}")

    results['stages']['hf_progression'] = hf_progression

    # =========================================================================
    # Stage 4: GP_MFGP와 비교
    # =========================================================================
    print(f"\n{'-'*70}")
    print("Stage 4: GP_MFGP와 비교")
    print(f"{'-'*70}")

    gp_model_class = MF_MODEL_REGISTRY['GP_MFGP']
    gp_model = gp_model_class(input_dim=2)
    gp_model.fit(data['X_lf'], data['y_lf'], data['X_hf'], data['y_hf'])

    pred_gp_test, std_gp_test = gp_model.predict(data['X_test'])

    gp_metrics = {
        'hf_train': compute_metrics(data['y_hf'], gp_model.predict(data['X_hf'])[0].flatten()),
        'hf_test': compute_metrics(data['y_test_hf'], pred_gp_test.flatten()),
        'lf_test': compute_metrics(data['y_test_lf'], pred_gp_test.flatten()),
        'mean_uncertainty': float(np.mean(std_gp_test))
    }
    results['gp_mfgp'] = gp_metrics

    print(f"  GP_MFGP:")
    print(f"    HF Train R²: {gp_metrics['hf_train']['r2']:.4f}")
    print(f"    HF Test R²:  {gp_metrics['hf_test']['r2']:.4f}")
    print(f"    LF Test R²:  {gp_metrics['lf_test']['r2']:.4f}")
    print(f"    Uncertainty: {gp_metrics['mean_uncertainty']:.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Model':<20} {'HF Train R²':>12} {'HF Test R²':>12} {'Gap':>10}")
    print("-" * 54)

    dngo_train = stage2_metrics['hf_train']['r2']
    dngo_test = stage2_metrics['hf_test']['r2']
    gp_train = gp_metrics['hf_train']['r2']
    gp_test = gp_metrics['hf_test']['r2']

    print(f"{'DNGO_Joint':<20} {dngo_train:>12.4f} {dngo_test:>12.4f} {dngo_train-dngo_test:>10.4f}")
    print(f"{'GP_MFGP':<20} {gp_train:>12.4f} {gp_test:>12.4f} {gp_train-gp_test:>10.4f}")

    # 과적합 분석
    print(f"\n과적합 분석:")
    print(f"  DNGO_Joint: Train-Test Gap = {dngo_train - dngo_test:.4f}")
    print(f"  GP_MFGP: Train-Test Gap = {gp_train - gp_test:.4f}")

    if dngo_test < 0:
        print(f"\n  ⚠️ DNGO_Joint HF Test R² = {dngo_test:.4f} < 0")
        print(f"     → 모델 예측이 단순 평균보다 나쁨 (심한 과적합)")

    return results


def main():
    print("=" * 70)
    print("DNGO_Joint 상세 분석")
    print("=" * 70)
    print(f"시작 시간: {datetime.now()}")

    # 여러 seed로 분석
    all_results = []

    for seed in range(3):  # 3개 seed로 테스트
        results = analyze_dngo_joint_detailed(
            seed=seed,
            n_lf=50,
            n_hf=45,
            n_test=200,
            alpha=0.8  # Favorable scenario
        )
        all_results.append(results)

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"dngo_joint_analysis_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n결과 저장: {output_file}")

    # 평균 결과 출력
    print(f"\n{'='*70}")
    print("평균 결과 (3 seeds)")
    print(f"{'='*70}")

    dngo_test_r2s = [r['stages']['joint_training']['hf_test']['r2'] for r in all_results]
    gp_test_r2s = [r['gp_mfgp']['hf_test']['r2'] for r in all_results]

    print(f"DNGO_Joint HF Test R²: {np.mean(dngo_test_r2s):.4f} ± {np.std(dngo_test_r2s):.4f}")
    print(f"GP_MFGP HF Test R²:    {np.mean(gp_test_r2s):.4f} ± {np.std(gp_test_r2s):.4f}")


if __name__ == "__main__":
    main()
