"""
Optuna 기반 BNN 하이퍼파라미터 베이지안 최적화 모듈

Pretrain과 Finetune 단계에서 각각 Optuna를 통해
최적의 BNN 하이퍼파라미터를 찾습니다.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from .bnn_models import TransferLearningBNN
import time
from tqdm import tqdm


def create_bnn_optuna_objective(X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                input_dim: int, device: str, data_size: str, stage: str,
                                fixed_structure: Dict = None,
                                optimize_incremental: bool = False):
    """BNN Optuna objective function 생성"""

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
                
                # 동적으로 hidden_dims 생성
                hidden_dims = [hidden_dim] * hidden_layers
                
                # BNN 모델 생성
                bnn = TransferLearningBNN(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    device=device
                )
                
                # Pretrain
                bnn.pretrain(X_train, y_train, epochs=epochs, lr=learning_rate, verbose=False)
                
                # 검증 평가
                pred_mean, pred_std = bnn.predict(X_val, n_samples=30)
                mse = np.mean((pred_mean - y_val) ** 2)
                
                return mse
                
            else:  # finetune
                # Finetune 단계: 학습률, epochs, kl_weight만 탐색, 구조는 고정
                # 구조는 pretrain에서 결정된 것 사용
                if fixed_structure is not None:
                    hidden_layers = fixed_structure['hidden_layers']
                    hidden_dim = fixed_structure['hidden_dim']
                else:
                    # fallback: 기본 구조 사용
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
                
                # Incremental learning 파라미터 (finetune에서만)
                if optimize_incremental:
                    incremental_mode = trial.suggest_categorical('incremental_mode', 
                                                                ['full', 'incremental', 'hybrid'])
                    lr_boost_factor = trial.suggest_float('lr_boost_factor', 1.0, 5.0)
                    incremental_epochs = trial.suggest_int('incremental_epochs', 5, 30)
                    replay_ratio = trial.suggest_float('replay_ratio', 0.0, 0.5)
                    weight_decay_factor = trial.suggest_float('weight_decay_factor', 0.8, 1.0)
                    full_retrain_interval = trial.suggest_int('full_retrain_interval', 3, 10)
                    kl_reg_weight = trial.suggest_float('kl_reg_weight', 0.01, 1.0, log=True)
                
                # 동적으로 hidden_dims 생성
                hidden_dims = [hidden_dim] * hidden_layers
                
                # BNN 모델 생성 (pretrain은 간단하게)
                bnn = TransferLearningBNN(
                    input_dim=input_dim,
                    hidden_dims=[64],  # pretrain은 고정
                    device=device
                )
                
                # 간단한 pretrain (finetune 최적화에 집중)
                bnn.pretrain(X_train, y_train, epochs=100, lr=1e-3, verbose=False)
                
                # Finetune with optimized parameters
                bnn.finetune(X_train, y_train, 
                           epochs=epochs, lr=learning_rate, 
                           kl_weight=kl_weight, verbose=False)
                
                # 검증 평가 (MSE + 불확실성 품질)
                pred_mean, pred_std = bnn.predict(X_val, n_samples=30)
                mse = np.mean((pred_mean - y_val) ** 2)
                
                # 불확실성 품질 평가
                errors = np.abs(pred_mean - y_val)
                if len(errors) > 1 and np.std(pred_std) > 1e-8:
                    uncertainty_quality = np.corrcoef(errors, pred_std)[0, 1]
                    if np.isnan(uncertainty_quality):
                        uncertainty_quality = 0
                else:
                    uncertainty_quality = 0
                
                # 점수: MSE 중심, 좋은 불확실성은 보너스
                score = mse - 0.1 * max(0, uncertainty_quality)
                
                return score
                
        except Exception as e:
            return float('inf')
    
    return objective


def optimize_bnn_hyperparameters_optuna(X_train: np.ndarray, y_train: np.ndarray,
                                        X_val: np.ndarray, y_val: np.ndarray,
                                        input_dim: int, n_trials: int = 10,
                                        data_size: str = 'small', device: str = 'cpu',
                                        verbose: bool = True, stage: str = 'pretrain',
                                        fixed_structure: Dict = None,
                                        optimize_incremental: bool = False) -> Tuple[Dict, float, List]:
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

    Returns:
        최적 하이퍼파라미터, 최적 성능, 전체 기록
    """

    # Optuna study 생성
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=10)

    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner
    )

    # Objective function 생성
    objective = create_bnn_optuna_objective(X_train, y_train, X_val, y_val,
                                          input_dim, device, data_size, stage, fixed_structure, optimize_incremental)
    
    # 최적화 실행 with progress bar
    stage_prefix = "Pretrain" if stage == 'pretrain' else "Finetune"
    if verbose:
        # Optuna의 verbosity 조절
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        with tqdm(total=n_trials, desc=f"      {stage_prefix} HP-BO",
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

        print(f"      ✅ Best loss: {best_performance:.4f}")
    
    return best_params, best_performance, trial_history