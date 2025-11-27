"""
Optuna 기반 BNN 하이퍼파라미터 베이지안 최적화 모듈

Pretrain과 Finetune 단계에서 각각 Optuna를 통해
최적의 BNN 하이퍼파라미터를 찾습니다.

개선 사항:
- LOOCV (Leave-One-Out Cross Validation) 지원
- BO-Aware Objective: 불확실성 calibration (L_uncertainty) 지원
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from .bnn_models import TransferLearningBNN
from scipy.stats import spearmanr
import time
from tqdm import tqdm


def compute_uncertainty_loss(errors: np.ndarray, uncertainties: np.ndarray) -> float:
    """
    불확실성 calibration 손실 계산 (Error-Uncertainty Correlation)

    L_uncertainty = 1 - Spearman_Correlation(|errors|, uncertainties)

    Args:
        errors: 예측 오차 배열 |y_pred - y_true|
        uncertainties: 예측 불확실성 배열 (표준편차)

    Returns:
        L_uncertainty: [0, 2] 범위
        - 0: 완벽한 calibration
        - 1: 상관관계 없음
        - 2: 역상관 (최악)
    """
    if len(errors) < 3:
        return 0.0

    if np.std(errors) < 1e-8 or np.std(uncertainties) < 1e-8:
        return 1.0

    correlation, _ = spearmanr(errors, uncertainties)

    if np.isnan(correlation):
        return 1.0

    return 1.0 - correlation


def evaluate_bnn_with_loocv(X: np.ndarray, y: np.ndarray,
                            hidden_dims: List[int], device: str,
                            pretrain_epochs: int, pretrain_lr: float,
                            finetune_epochs: int = 0, finetune_lr: float = 1e-4,
                            kl_weight: float = 1.0,
                            use_uncertainty_loss: bool = False,
                            uncertainty_weight: float = 0.3,
                            stage: str = 'pretrain') -> float:
    """
    LOOCV를 사용한 BNN 모델 평가

    Args:
        X: 전체 입력 데이터
        y: 전체 출력 데이터
        hidden_dims: hidden layer dimensions
        device: 디바이스
        pretrain_epochs: pretrain epoch 수
        pretrain_lr: pretrain 학습률
        finetune_epochs: finetune epoch 수 (finetune stage에서만)
        finetune_lr: finetune 학습률
        kl_weight: KL divergence 가중치
        use_uncertainty_loss: 불확실성 손실 사용 여부
        uncertainty_weight: 불확실성 손실 가중치
        stage: 'pretrain' 또는 'finetune'

    Returns:
        평균 검증 손실
    """
    n = len(X)
    predictions = np.zeros(n)
    uncertainties = np.zeros(n)

    # LOOCV epoch 수 감소 (n번 반복하므로)
    reduced_pretrain_epochs = max(pretrain_epochs // 3, 30)
    reduced_finetune_epochs = max(finetune_epochs // 3, 20) if finetune_epochs > 0 else 0

    for i in range(n):
        # Leave-One-Out
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_val = X[i:i+1]

        # BNN 모델 생성
        bnn = TransferLearningBNN(
            input_dim=X.shape[1],
            hidden_dims=hidden_dims,
            device=device
        )

        # Pretrain
        bnn.pretrain(X_train, y_train, epochs=reduced_pretrain_epochs, lr=pretrain_lr, verbose=False)

        # Finetune (stage가 finetune일 때만)
        if stage == 'finetune' and reduced_finetune_epochs > 0:
            bnn.finetune(X_train, y_train, epochs=reduced_finetune_epochs, lr=finetune_lr,
                        kl_weight=kl_weight, verbose=False)

        # 예측 (불확실성 포함)
        pred_mean, pred_std = bnn.predict(X_val, n_samples=30)
        predictions[i] = pred_mean[0]
        uncertainties[i] = pred_std[0]

    # MSE 계산
    errors = np.abs(predictions - y)
    mse = np.mean(errors ** 2)

    # 불확실성 손실 계산 (선택적)
    if use_uncertainty_loss and n >= 5:
        l_uncertainty = compute_uncertainty_loss(errors, uncertainties)
        mse_normalized = mse / (np.var(y) + 1e-8)
        total_loss = (1 - uncertainty_weight) * mse_normalized + uncertainty_weight * l_uncertainty
        return total_loss

    return mse


def evaluate_bnn_with_train_val_split(X_train: np.ndarray, y_train: np.ndarray,
                                       X_val: np.ndarray, y_val: np.ndarray,
                                       hidden_dims: List[int], device: str,
                                       pretrain_epochs: int, pretrain_lr: float,
                                       finetune_epochs: int = 0, finetune_lr: float = 1e-4,
                                       kl_weight: float = 1.0,
                                       use_uncertainty_loss: bool = False,
                                       uncertainty_weight: float = 0.3,
                                       stage: str = 'pretrain') -> float:
    """
    기존 Train/Val 분할을 사용한 BNN 모델 평가

    Args:
        X_train, y_train: 훈련 데이터
        X_val, y_val: 검증 데이터
        hidden_dims: hidden layer dimensions
        device: 디바이스
        pretrain_epochs: pretrain epoch 수
        pretrain_lr: pretrain 학습률
        finetune_epochs: finetune epoch 수
        finetune_lr: finetune 학습률
        kl_weight: KL divergence 가중치
        use_uncertainty_loss: 불확실성 손실 사용 여부
        uncertainty_weight: 불확실성 손실 가중치
        stage: 'pretrain' 또는 'finetune'

    Returns:
        검증 손실
    """
    # BNN 모델 생성
    bnn = TransferLearningBNN(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        device=device
    )

    # Pretrain
    bnn.pretrain(X_train, y_train, epochs=pretrain_epochs, lr=pretrain_lr, verbose=False)

    # Finetune (stage가 finetune일 때만)
    if stage == 'finetune' and finetune_epochs > 0:
        bnn.finetune(X_train, y_train, epochs=finetune_epochs, lr=finetune_lr,
                    kl_weight=kl_weight, verbose=False)

    # 예측
    pred_mean, pred_std = bnn.predict(X_val, n_samples=30)
    y_val_np = y_val.flatten()

    # MSE 계산
    errors = np.abs(pred_mean - y_val_np)
    mse = np.mean(errors ** 2)

    # 불확실성 손실 계산 (선택적)
    if use_uncertainty_loss and len(X_val) >= 3:
        l_uncertainty = compute_uncertainty_loss(errors, pred_std)
        mse_normalized = mse / (np.var(y_val_np) + 1e-8)
        total_loss = (1 - uncertainty_weight) * mse_normalized + uncertainty_weight * l_uncertainty
        return total_loss

    return mse


def create_bnn_optuna_objective(X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                input_dim: int, device: str, data_size: str, stage: str,
                                fixed_structure: Dict = None,
                                optimize_incremental: bool = False,
                                use_loocv: bool = False,
                                use_uncertainty_loss: bool = False,
                                uncertainty_weight: float = 0.3):
    """
    BNN Optuna objective function 생성

    Args:
        X_train, y_train: 훈련 데이터
        X_val, y_val: 검증 데이터 (use_loocv=False일 때 사용)
        input_dim: 입력 차원
        device: 디바이스
        data_size: 데이터 크기
        stage: 'pretrain' 또는 'finetune'
        fixed_structure: finetune시 고정할 구조
        optimize_incremental: incremental learning 최적화 여부
        use_loocv: LOOCV 사용 여부
        use_uncertainty_loss: 불확실성 calibration 손실 사용 여부
        uncertainty_weight: 불확실성 손실 가중치
    """
    # LOOCV 사용 시 전체 데이터 병합
    if use_loocv:
        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])
    else:
        X_all = None
        y_all = None

    def objective(trial):
        try:
            if stage == 'pretrain':
                # Pretrain 단계 하이퍼파라미터
                if data_size == 'small':
                    hidden_layers = trial.suggest_int('hidden_layers', 1, 2)
                    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64])
                    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                    epochs = trial.suggest_int('epochs', 100, 250)
                elif data_size == 'medium':
                    hidden_layers = trial.suggest_int('hidden_layers', 1, 3)
                    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
                    learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
                    epochs = trial.suggest_int('epochs', 150, 300)
                else:  # large
                    hidden_layers = trial.suggest_int('hidden_layers', 2, 4)
                    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
                    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
                    epochs = trial.suggest_int('epochs', 200, 400)

                hidden_dims = [hidden_dim] * hidden_layers

                # 평가 방법 선택
                if use_loocv and X_all is not None:
                    val_loss = evaluate_bnn_with_loocv(
                        X_all, y_all, hidden_dims, device,
                        pretrain_epochs=epochs, pretrain_lr=learning_rate,
                        use_uncertainty_loss=use_uncertainty_loss,
                        uncertainty_weight=uncertainty_weight,
                        stage='pretrain'
                    )
                else:
                    val_loss = evaluate_bnn_with_train_val_split(
                        X_train, y_train, X_val, y_val,
                        hidden_dims, device,
                        pretrain_epochs=epochs, pretrain_lr=learning_rate,
                        use_uncertainty_loss=use_uncertainty_loss,
                        uncertainty_weight=uncertainty_weight,
                        stage='pretrain'
                    )

                return val_loss

            else:  # finetune
                # Finetune 단계: 학습률, epochs, kl_weight만 탐색, 구조는 고정
                if fixed_structure is not None:
                    hidden_layers = fixed_structure['hidden_layers']
                    hidden_dim = fixed_structure['hidden_dim']
                else:
                    hidden_layers = 2
                    hidden_dim = 64

                # Learning rate, epochs, kl_weight만 탐색
                if data_size == 'small':
                    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
                    epochs = trial.suggest_int('epochs', 50, 150)
                    kl_weight = trial.suggest_float('kl_weight', 0.1, 10.0, log=True)
                elif data_size == 'medium':
                    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
                    epochs = trial.suggest_int('epochs', 75, 200)
                    kl_weight = trial.suggest_float('kl_weight', 0.1, 10.0, log=True)
                else:  # large
                    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
                    epochs = trial.suggest_int('epochs', 100, 250)
                    kl_weight = trial.suggest_float('kl_weight', 0.1, 10.0, log=True)

                # Incremental learning 파라미터 (finetune에서만, 향후 사용)
                if optimize_incremental:
                    incremental_mode = trial.suggest_categorical('incremental_mode',
                                                                ['full', 'incremental', 'hybrid'])
                    lr_boost_factor = trial.suggest_float('lr_boost_factor', 1.0, 5.0)
                    incremental_epochs = trial.suggest_int('incremental_epochs', 5, 30)
                    replay_ratio = trial.suggest_float('replay_ratio', 0.0, 0.5)
                    weight_decay_factor = trial.suggest_float('weight_decay_factor', 0.8, 1.0)
                    full_retrain_interval = trial.suggest_int('full_retrain_interval', 3, 10)
                    kl_reg_weight = trial.suggest_float('kl_reg_weight', 0.01, 1.0, log=True)

                hidden_dims = [hidden_dim] * hidden_layers

                # 평가 방법 선택
                if use_loocv and X_all is not None:
                    val_loss = evaluate_bnn_with_loocv(
                        X_all, y_all, hidden_dims, device,
                        pretrain_epochs=100, pretrain_lr=1e-3,  # pretrain은 고정
                        finetune_epochs=epochs, finetune_lr=learning_rate,
                        kl_weight=kl_weight,
                        use_uncertainty_loss=use_uncertainty_loss,
                        uncertainty_weight=uncertainty_weight,
                        stage='finetune'
                    )
                else:
                    val_loss = evaluate_bnn_with_train_val_split(
                        X_train, y_train, X_val, y_val,
                        hidden_dims, device,
                        pretrain_epochs=100, pretrain_lr=1e-3,  # pretrain은 고정
                        finetune_epochs=epochs, finetune_lr=learning_rate,
                        kl_weight=kl_weight,
                        use_uncertainty_loss=use_uncertainty_loss,
                        uncertainty_weight=uncertainty_weight,
                        stage='finetune'
                    )

                return val_loss

        except Exception as e:
            return float('inf')

    return objective


def optimize_bnn_hyperparameters_optuna(X_train: np.ndarray, y_train: np.ndarray,
                                        X_val: np.ndarray, y_val: np.ndarray,
                                        input_dim: int, n_trials: int = 10,
                                        data_size: str = 'small', device: str = 'cpu',
                                        verbose: bool = True, stage: str = 'pretrain',
                                        fixed_structure: Dict = None,
                                        optimize_incremental: bool = False,
                                        use_loocv: bool = False,
                                        use_uncertainty_loss: bool = False,
                                        uncertainty_weight: float = 0.3) -> Tuple[Dict, float, List]:
    """
    Optuna를 사용한 BNN 하이퍼파라미터 베이지안 최적화

    Args:
        X_train: 훈련 데이터
        y_train: 훈련 라벨
        X_val: 검증 데이터
        y_val: 검증 라벨
        input_dim: 입력 차원
        n_trials: BO 시행 횟수
        data_size: 데이터 크기 ('small', 'medium', 'large')
        device: 디바이스
        verbose: 상세 출력
        stage: 'pretrain' 또는 'finetune'
        fixed_structure: finetune일 때 사용할 고정된 구조 {'hidden_layers': int, 'hidden_dim': int}
        optimize_incremental: incremental learning 최적화 여부
        use_loocv: LOOCV 사용 여부 (기본: False, 기존 Train/Val 분할)
        use_uncertainty_loss: 불확실성 calibration 손실 사용 여부 (기본: False, 기존 MSE만)
        uncertainty_weight: 불확실성 손실 가중치 (기본: 0.3)

    Returns:
        최적 하이퍼파라미터, 최적 성능, 전체 기록
    """

    # Optuna study 생성
    sampler = TPESampler(seed=42)
    # LOOCV 사용 시 pruning 비활성화
    if use_loocv:
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=10)

    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner
    )

    # Objective function 생성
    objective = create_bnn_optuna_objective(
        X_train, y_train, X_val, y_val,
        input_dim, device, data_size, stage, fixed_structure, optimize_incremental,
        use_loocv=use_loocv,
        use_uncertainty_loss=use_uncertainty_loss,
        uncertainty_weight=uncertainty_weight
    )

    # 최적화 실행 with progress bar
    stage_prefix = "Pretrain" if stage == 'pretrain' else "Finetune"

    # 평가 방법 문자열 생성
    eval_method = "LOOCV" if use_loocv else "Train/Val"
    loss_type = f"MSE+Unc(w={uncertainty_weight})" if use_uncertainty_loss else "MSE"

    if verbose:
        # Optuna의 verbosity 조절
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        with tqdm(total=n_trials, desc=f"      {stage_prefix} HP-BO ({eval_method}, {loss_type})",
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}') as pbar:

            def callback(study, trial):
                pbar.set_postfix({
                    'best_loss': f'{study.best_value:.4f}' if study.best_value != float('inf') else 'inf',
                    'current_loss': f'{trial.value:.4f}' if trial.value is not None else 'inf'
                })
                pbar.update(1)

            study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials)
    
    # 결과 정리
    best_params = study.best_params
    best_performance = study.best_value
    
    # hidden_dims 리스트로 변환
    if 'hidden_layers' in best_params and 'hidden_dim' in best_params:
        hidden_dims = [best_params['hidden_dim']] * best_params['hidden_layers']
        best_params['hidden_dims'] = hidden_dims
        # 개별 파라미터 제거
        del best_params['hidden_layers']
        del best_params['hidden_dim']
    
    # BNN 고정 파라미터 추가 (finetune에서 필요)
    if stage == 'finetune':
        best_params['prior_std'] = 1.0  # 고정값
        best_params['noise_type'] = 'homoscedastic'  # 고정값
        best_params['kl_warmup_epochs'] = 10  # 고정값
        best_params['finetune_epochs'] = best_params.get('epochs', 100)
        best_params['finetune_lr'] = best_params.get('learning_rate', 1e-4)
    elif stage == 'pretrain':
        best_params['pretrain_epochs'] = best_params.get('epochs', 200)
        best_params['pretrain_lr'] = best_params.get('learning_rate', 1e-3)
    
    # 모든 시행 기록
    trial_history = []
    for trial in study.trials:
        if trial.value is not None:
            record = trial.params.copy()
            record['validation_loss'] = trial.value
            # hidden_dims 변환
            if 'hidden_layers' in record and 'hidden_dim' in record:
                record['hidden_dims'] = [record['hidden_dim']] * record['hidden_layers']
                del record['hidden_layers']
                del record['hidden_dim']
            
            # 고정 파라미터 추가 (기록에도 포함)
            if stage == 'finetune':
                record['prior_std'] = 1.0
                record['noise_type'] = 'homoscedastic'
                record['kl_warmup_epochs'] = 10
                record['finetune_epochs'] = record.get('epochs', 100)
                record['finetune_lr'] = record.get('learning_rate', 1e-4)
            elif stage == 'pretrain':
                record['pretrain_epochs'] = record.get('epochs', 200)
                record['pretrain_lr'] = record.get('learning_rate', 1e-3)
                
            trial_history.append(record)
    
    if verbose:
        if stage == 'pretrain':
            hidden_dims_str = f"dims={best_params.get('hidden_dims', 'N/A')}"
            lr_str = f"lr={best_params.get('learning_rate', 0):.1e}"
            epochs_str = f"epochs={best_params.get('epochs', 0)}"
            param_str = f"{hidden_dims_str}, {lr_str}, {epochs_str}"
            print(f"      ✅ Best {stage_prefix} params: {param_str}")
        else:
            lr_str = f"lr={best_params.get('learning_rate', 0):.1e}"
            epochs_str = f"epochs={best_params.get('epochs', 0)}"
            kl_str = f"kl={best_params.get('kl_weight', 'N/A'):.1f}" if 'kl_weight' in best_params else ""
            param_str = f"{lr_str}, {epochs_str}"
            if kl_str:
                param_str += f", {kl_str}"
            print(f"      ✅ Best {stage_prefix} params: {param_str} (structure fixed)")

        print(f"      ✅ Best loss: {best_performance:.4f} (eval: {eval_method}, loss: {loss_type})")

    return best_params, best_performance, trial_history