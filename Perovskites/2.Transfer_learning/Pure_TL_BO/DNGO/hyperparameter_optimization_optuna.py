"""
Optuna 기반 DNN 하이퍼파라미터 베이지안 최적화 모듈

Pretrain과 Finetune 단계에서 각각 Optuna를 통해
최적의 하이퍼파라미터를 찾습니다.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from .models import TransferLearningDNN
import time
from tqdm import tqdm


class OptunaDNN(nn.Module):
    """Optuna에서 사용할 동적 DNN"""
    
    def __init__(self, input_dim: int, hidden_layers: int, hidden_dim: int, device: str = 'cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.device = device
        
        # 동적으로 네트워크 구성
        layers = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(hidden_layers - 1):
            layers.extend([
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ])
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers).to(device).float()
    
    def forward(self, x):
        return self.network(x)


def create_optuna_objective(X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           input_dim: int, device: str, data_size: str, 
                           optimize_incremental: bool = False):
    """Optuna objective function 생성 (incremental learning 포함)"""
    
    def objective(trial):
        # 기본 하이퍼파라미터 제안
        if data_size == 'small':
            hidden_layers = trial.suggest_int('hidden_layers', 1, 3)
            hidden_dim = trial.suggest_categorical('hidden_dim', [16, 32, 64, 128])
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            epochs = trial.suggest_int('epochs', 50, 200)
        elif data_size == 'medium':
            hidden_layers = trial.suggest_int('hidden_layers', 1, 4)
            hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256])
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-3, log=True)
            epochs = trial.suggest_int('epochs', 100, 500)
        else:  # large
            hidden_layers = trial.suggest_int('hidden_layers', 2, 5)
            hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
            learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
            epochs = trial.suggest_int('epochs', 100, 1000)
        
        # Incremental learning 하이퍼파라미터 제안 (선택적)
        incremental_params = None
        if optimize_incremental:
            incremental_mode = trial.suggest_categorical('incremental_mode', 
                                                        ['full', 'incremental', 'hybrid'])
            lr_boost_factor = trial.suggest_float('lr_boost_factor', 1.0, 5.0)
            incremental_epochs = trial.suggest_int('incremental_epochs', 5, 30)
            replay_ratio = trial.suggest_float('replay_ratio', 0.0, 0.5)
            weight_decay_factor = trial.suggest_float('weight_decay_factor', 0.8, 1.0)
            full_retrain_interval = trial.suggest_int('full_retrain_interval', 3, 10)
            
            incremental_params = {
                'mode': incremental_mode,
                'lr_boost_factor': lr_boost_factor,
                'incremental_epochs': incremental_epochs,
                'replay_ratio': replay_ratio,
                'weight_decay_factor': weight_decay_factor,
                'full_retrain_interval': full_retrain_interval
            }
        
        try:
            # 모델 생성
            model = OptunaDNN(input_dim, hidden_layers, hidden_dim, device)
            
            # 데이터 준비
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
            
            # 옵티마이저 및 손실함수
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            # Incremental learning 시뮬레이션 또는 기본 학습
            if optimize_incremental and incremental_params:
                # Incremental learning 시뮬레이션
                val_loss = _simulate_incremental_learning(
                    model, X_train, y_train, X_val, y_val, 
                    learning_rate, epochs, incremental_params, device, trial
                )
            else:
                # 기본 전체 학습
                val_loss = _standard_training(
                    model, X_train, y_train, X_val, y_val,
                    learning_rate, epochs, device, trial
                )
            
            return val_loss
            
        except Exception as e:
            # 오류 시 매우 큰 값 반환
            return float('inf')
    
    return objective


def _standard_training(model, X_train, y_train, X_val, y_val, learning_rate, epochs, device, trial):
    """표준 전체 학습"""
    # 데이터 준비
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    
    # 옵티마이저 및 손실함수
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # 학습
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train_tensor)
        loss = criterion(pred, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Pruning을 위한 중간 검증 (10 epoch마다)
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_tensor)
                val_loss = criterion(val_pred, y_val_tensor).item()
            model.train()
            
            # Optuna pruning
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    # 최종 검증 성능 평가
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_tensor)
        val_loss = criterion(val_pred, y_val_tensor).item()
    
    return val_loss


def _simulate_incremental_learning(model, X_train, y_train, X_val, y_val, 
                                  learning_rate, epochs, incremental_params, device, trial):
    """Incremental learning 시뮬레이션"""
    # 데이터를 청크로 나누어 점진적 학습 시뮬레이션
    n_chunks = 5
    chunk_size = len(X_train) // n_chunks
    total_val_loss = 0.0
    
    # 첫 번째 청크로 초기 학습
    X_chunk = X_train[:chunk_size]
    y_chunk = y_train[:chunk_size]
    
    # 초기 학습
    X_tensor = torch.tensor(X_chunk, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_chunk, dtype=torch.float32).view(-1, 1).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # 초기 학습
    model.train()
    for epoch in range(epochs // 2):  # 초기는 짧게
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()
    
    # 점진적 업데이트 시뮬레이션
    for chunk_idx in range(1, n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(X_train))
        X_new = X_train[start_idx:end_idx]
        y_new = y_train[start_idx:end_idx]
        
        if incremental_params['mode'] == 'full':
            # 전체 재학습
            X_all = X_train[:end_idx]
            y_all = y_train[:end_idx]
            X_tensor = torch.tensor(X_all, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y_all, dtype=torch.float32).view(-1, 1).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            for epoch in range(epochs // n_chunks):
                optimizer.zero_grad()
                pred = model(X_tensor)
                loss = criterion(pred, y_tensor)
                loss.backward()
                optimizer.step()
                
        elif incremental_params['mode'] == 'incremental':
            # 점진적 업데이트 시뮬레이션
            X_new_tensor = torch.tensor(X_new, dtype=torch.float32).to(device)
            y_new_tensor = torch.tensor(y_new, dtype=torch.float32).view(-1, 1).to(device)
            
            # 부스트된 학습률 사용
            boosted_lr = learning_rate * incremental_params['lr_boost_factor']
            inc_optimizer = optim.Adam(model.parameters(), lr=boosted_lr)
            
            # 짧은 epoch로 점진적 학습
            for epoch in range(incremental_params['incremental_epochs']):
                inc_optimizer.zero_grad()
                pred = model(X_new_tensor)
                loss = criterion(pred, y_new_tensor)
                loss.backward()
                inc_optimizer.step()
                
        else:  # hybrid
            # 주기적 전체 재학습
            if chunk_idx % incremental_params['full_retrain_interval'] == 0:
                # 전체 재학습
                X_all = X_train[:end_idx]
                y_all = y_train[:end_idx]
                X_tensor = torch.tensor(X_all, dtype=torch.float32).to(device)
                y_tensor = torch.tensor(y_all, dtype=torch.float32).view(-1, 1).to(device)
                
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                for epoch in range(epochs // n_chunks):
                    optimizer.zero_grad()
                    pred = model(X_tensor)
                    loss = criterion(pred, y_tensor)
                    loss.backward()
                    optimizer.step()
            else:
                # 점진적 업데이트
                X_new_tensor = torch.tensor(X_new, dtype=torch.float32).to(device)
                y_new_tensor = torch.tensor(y_new, dtype=torch.float32).view(-1, 1).to(device)
                
                boosted_lr = learning_rate * incremental_params['lr_boost_factor']
                inc_optimizer = optim.Adam(model.parameters(), lr=boosted_lr)
                
                for epoch in range(incremental_params['incremental_epochs']):
                    inc_optimizer.zero_grad()
                    pred = model(X_new_tensor)
                    loss = criterion(pred, y_new_tensor)
                    loss.backward()
                    inc_optimizer.step()
        
        # 중간 검증
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor).item()
        model.train()
        
        total_val_loss += val_loss
        
        # Pruning 체크
        trial.report(val_loss, chunk_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return total_val_loss / n_chunks


def optimize_dnn_hyperparameters_optuna(X_train: np.ndarray, y_train: np.ndarray,
                                        X_val: np.ndarray, y_val: np.ndarray,
                                        input_dim: int, n_trials: int = 10,
                                        data_size: str = 'small', device: str = 'cpu',
                                        verbose: bool = True, optimize_incremental: bool = False) -> Tuple[Dict, float, List]:
    """
    Optuna를 사용한 DNN 하이퍼파라미터 베이지안 최적화
    
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
        
    Returns:
        최적 하이퍼파라미터, 최적 성능, 전체 기록
    """
    
    # Optuna study 생성
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=20)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner
    )
    
    # Objective function 생성 (incremental learning 포함)
    objective = create_optuna_objective(X_train, y_train, X_val, y_val, input_dim, device, data_size, optimize_incremental)
    
    # 최적화 실행 with progress bar
    if verbose:
        # Optuna의 verbosity 조절
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        with tqdm(total=n_trials, desc="      HP-BO Progress", 
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
    
    # 모든 시행 기록
    trial_history = []
    for trial in study.trials:
        if trial.value is not None:
            record = trial.params.copy()
            record['validation_loss'] = trial.value
            trial_history.append(record)
    
    if verbose:
        print(f"      ✅ Best params: layers={best_params['hidden_layers']}, dim={best_params['hidden_dim']}, lr={best_params['learning_rate']:.1e}, epochs={best_params['epochs']}")
        print(f"      ✅ Best loss: {best_performance:.4f}")
    
    return best_params, best_performance, trial_history