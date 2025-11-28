"""
Optuna 기반 DNN 하이퍼파라미터 베이지안 최적화 모듈

Pretrain과 Finetune 단계에서 각각 Optuna를 통해
최적의 하이퍼파라미터를 찾습니다.

개선 사항:
- LOOCV (Leave-One-Out Cross Validation) 지원
- BO-Aware Objective: 불확실성 calibration (L_uncertainty) 지원
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from .models import TransferLearningDNN, BayesianLinearRegression
from scipy.stats import spearmanr
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


def compute_uncertainty_loss(errors: np.ndarray, uncertainties: np.ndarray) -> float:
    """
    불확실성 calibration 손실 계산 (Error-Uncertainty Correlation)

    L_uncertainty = 1 - Spearman_Correlation(|errors|, uncertainties)

    Args:
        errors: 예측 오차 배열 |y_pred - y_true|
        uncertainties: 예측 불확실성 배열 (표준편차)

    Returns:
        L_uncertainty: [0, 2] 범위
        - 0: 완벽한 calibration (오차와 불확실성이 완전 상관)
        - 1: 상관관계 없음
        - 2: 역상관 (최악)
    """
    if len(errors) < 3:
        # 데이터가 너무 적으면 계산 불가
        return 0.0

    # 분산이 0인 경우 처리
    if np.std(errors) < 1e-8 or np.std(uncertainties) < 1e-8:
        return 1.0  # 상관관계 계산 불가 → 중립값 반환

    correlation, _ = spearmanr(errors, uncertainties)

    if np.isnan(correlation):
        return 1.0

    return 1.0 - correlation


def evaluate_with_loocv(X: np.ndarray, y: np.ndarray,
                        hidden_layers: int, hidden_dim: int,
                        learning_rate: float, epochs: int,
                        device: str, use_uncertainty_loss: bool = False,
                        uncertainty_weight: float = 0.3) -> float:
    """
    LOOCV (Leave-One-Out Cross Validation)를 사용한 모델 평가

    Args:
        X: 전체 입력 데이터
        y: 전체 출력 데이터
        hidden_layers: hidden layer 수
        hidden_dim: hidden dimension
        learning_rate: 학습률
        epochs: epoch 수
        device: 디바이스
        use_uncertainty_loss: 불확실성 손실 사용 여부
        uncertainty_weight: 불확실성 손실 가중치 (0~1)

    Returns:
        평균 검증 손실 (MSE 또는 MSE + L_uncertainty)
    """
    n = len(X)
    predictions = np.zeros(n)
    uncertainties = np.zeros(n)

    for i in range(n):
        # Leave-One-Out: i번째 샘플 제외
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_val = X[i:i+1]
        y_val = y[i:i+1]

        # 모델 생성 및 학습
        model = OptunaDNN(X.shape[1], hidden_layers, hidden_dim, device)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # 학습 (짧은 epoch - LOOCV는 n번 반복하므로)
        reduced_epochs = max(epochs // 3, 20)  # epoch 수 감소
        model.train()
        for _ in range(reduced_epochs):
            optimizer.zero_grad()
            pred = model(X_train_tensor)
            loss = criterion(pred, y_train_tensor)
            loss.backward()
            optimizer.step()

        # 예측
        model.eval()
        with torch.no_grad():
            pred = model(X_val_tensor).cpu().numpy().flatten()[0]
            predictions[i] = pred

        # 불확실성 계산 (BLR 사용)
        if use_uncertainty_loss:
            # Feature 추출을 위한 forward pass
            features_train = []
            features_val = []

            # 마지막 hidden layer 출력을 feature로 사용
            def get_features(module, input, output):
                features_train.append(output.detach().cpu().numpy())

            # Hook 등록 (마지막 ReLU 이전 layer)
            # 간단하게: 전체 네트워크의 출력 직전 hidden layer 사용
            with torch.no_grad():
                # 마지막 layer 제외한 feature 추출
                feature_extractor = nn.Sequential(*list(model.network.children())[:-1])
                feat_train = feature_extractor(X_train_tensor).cpu().numpy()
                feat_val = feature_extractor(X_val_tensor).cpu().numpy()

            # BLR로 불확실성 추정
            blr = BayesianLinearRegression(alpha=1.0, beta=25.0)
            blr.fit(feat_train, y_train)
            _, var = blr.predict(feat_val[0])
            uncertainties[i] = np.sqrt(max(var, 1e-8))

    # MSE 계산
    errors = np.abs(predictions - y)
    mse = np.mean(errors ** 2)

    # 불확실성 손실 계산 (선택적)
    if use_uncertainty_loss and n >= 5:
        l_uncertainty = compute_uncertainty_loss(errors, uncertainties)
        # 결합: (1-w)*MSE + w*L_uncertainty
        # MSE와 L_uncertainty 스케일 정규화
        mse_normalized = mse / (np.var(y) + 1e-8)  # 정규화
        total_loss = (1 - uncertainty_weight) * mse_normalized + uncertainty_weight * l_uncertainty
        return total_loss

    return mse


def evaluate_with_train_val_split(X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray,
                                   hidden_layers: int, hidden_dim: int,
                                   learning_rate: float, epochs: int,
                                   device: str, trial: optuna.Trial,
                                   use_uncertainty_loss: bool = False,
                                   uncertainty_weight: float = 0.3) -> float:
    """
    기존 Train/Val 분할을 사용한 모델 평가 (기본 방식)

    Args:
        X_train, y_train: 훈련 데이터
        X_val, y_val: 검증 데이터
        hidden_layers: hidden layer 수
        hidden_dim: hidden dimension
        learning_rate: 학습률
        epochs: epoch 수
        device: 디바이스
        trial: Optuna trial (pruning용)
        use_uncertainty_loss: 불확실성 손실 사용 여부
        uncertainty_weight: 불확실성 손실 가중치

    Returns:
        검증 손실
    """
    # 모델 생성
    model = OptunaDNN(X_train.shape[1], hidden_layers, hidden_dim, device)

    # 데이터 준비
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

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

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # 최종 평가
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_tensor).cpu().numpy().flatten()
        y_val_np = y_val.flatten()

    # MSE 계산
    errors = np.abs(val_pred - y_val_np)
    mse = np.mean(errors ** 2)

    # 불확실성 손실 계산 (선택적)
    if use_uncertainty_loss and len(X_val) >= 3:
        # Feature 추출
        with torch.no_grad():
            feature_extractor = nn.Sequential(*list(model.network.children())[:-1])
            feat_train = feature_extractor(X_train_tensor).cpu().numpy()
            feat_val = feature_extractor(X_val_tensor).cpu().numpy()

        # BLR로 불확실성 추정
        blr = BayesianLinearRegression(alpha=1.0, beta=25.0)
        blr.fit(feat_train, y_train.flatten())

        uncertainties = np.zeros(len(X_val))
        for i in range(len(X_val)):
            _, var = blr.predict(feat_val[i])
            uncertainties[i] = np.sqrt(max(var, 1e-8))

        l_uncertainty = compute_uncertainty_loss(errors, uncertainties)

        # 결합
        mse_normalized = mse / (np.var(y_val_np) + 1e-8)
        total_loss = (1 - uncertainty_weight) * mse_normalized + uncertainty_weight * l_uncertainty
        return total_loss

    return mse


def evaluate_with_freeze(model_for_freeze, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         unfreeze_ratio: float, learning_rate: float, epochs: int,
                         device: str, trial: optuna.Trial) -> float:
    """
    Freeze 모드에서 unfreeze_ratio를 적용하여 모델 평가

    pretrained 모델의 복사본을 만들어 unfreeze_ratio에 따라 레이어를 동결/해동하고
    finetune하여 검증 손실을 반환합니다.

    Args:
        model_for_freeze: pretrained TransferLearningDNN 모델
        X_train, y_train: 훈련 데이터
        X_val, y_val: 검증 데이터
        unfreeze_ratio: 해동할 레이어 비율 (0.0 ~ 1.0)
        learning_rate: 학습률
        epochs: epoch 수
        device: 디바이스
        trial: Optuna trial (pruning용)

    Returns:
        검증 손실 (MSE)
    """
    import copy

    # 모델 복사 (원본 보존)
    model_copy = copy.deepcopy(model_for_freeze)

    # unfreeze_ratio 적용
    model_copy._apply_unfreeze_ratio(unfreeze_ratio)

    # 학습 가능한 파라미터만 추출
    trainable_params = model_copy._get_trainable_parameters()

    if len(trainable_params) == 0:
        # 모든 파라미터가 동결된 경우 (unfreeze_ratio=0이고 out_layer도 없는 경우)
        return float('inf')

    # 데이터 준비
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    criterion = nn.MSELoss()

    # 학습
    model_copy.model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        features = model_copy.feature_net(X_train_tensor)
        pred = model_copy.out_layer(features)
        loss = criterion(pred, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Pruning을 위한 중간 검증 (10 epoch마다)
        if epoch % 10 == 0:
            model_copy.model.eval()
            with torch.no_grad():
                val_features = model_copy.feature_net(X_val_tensor)
                val_pred = model_copy.out_layer(val_features)
                val_loss = criterion(val_pred, y_val_tensor).item()
            model_copy.model.train()

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # 최종 평가
    model_copy.model.eval()
    with torch.no_grad():
        val_features = model_copy.feature_net(X_val_tensor)
        val_pred = model_copy.out_layer(val_features)
        val_loss = criterion(val_pred, y_val_tensor).item()

    return val_loss


def create_optuna_objective(X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           input_dim: int, device: str, data_size: str,
                           stage: str = 'pretrain',
                           fixed_structure: Dict = None,
                           optimize_incremental: bool = False,
                           use_loocv: bool = False,
                           use_uncertainty_loss: bool = False,
                           uncertainty_weight: float = 0.3,
                           use_freeze: bool = False,
                           model_for_freeze = None):
    """
    Optuna objective function 생성

    Args:
        X_train, y_train: 훈련 데이터
        X_val, y_val: 검증 데이터 (use_loocv=False일 때 사용)
        input_dim: 입력 차원
        device: 디바이스
        data_size: 데이터 크기
        stage: 'pretrain' 또는 'finetune'
        fixed_structure: finetune시 고정할 구조
        optimize_incremental: incremental learning 최적화 여부
        use_loocv: LOOCV 사용 여부 (True면 X_train+X_val 전체로 LOOCV)
        use_uncertainty_loss: 불확실성 calibration 손실 사용 여부
        uncertainty_weight: 불확실성 손실 가중치 (0~1)
        use_freeze: Freeze 기법 사용 여부 (finetune 단계에서만 적용)
        model_for_freeze: Freeze 평가를 위한 pretrained 모델 (use_freeze=True일 때 필요)
    """
    # LOOCV 사용 시 전체 데이터 병합
    if use_loocv:
        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])
    else:
        X_all = None
        y_all = None

    def objective(trial):
        # Stage별 하이퍼파라미터 제안
        if stage == 'pretrain':
            # Pretrain: 모든 파라미터 탐색
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
        else:
            # Finetune: learning_rate와 epochs만 탐색, 구조는 고정
            hidden_layers = fixed_structure['hidden_layers']
            hidden_dim = fixed_structure['hidden_dim']

            if data_size == 'small':
                learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
                epochs = trial.suggest_int('epochs', 50, 200)
            elif data_size == 'medium':
                learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-3, log=True)
                epochs = trial.suggest_int('epochs', 100, 500)
            else:  # large
                learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
                epochs = trial.suggest_int('epochs', 100, 1000)

            # Freeze 모드: unfreeze_ratio 최적화 (0.0 ~ 1.0)
            if use_freeze:
                unfreeze_ratio = trial.suggest_float('unfreeze_ratio', 0.0, 1.0)
        
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
            # 평가 방법 선택
            if use_freeze and stage == 'finetune' and model_for_freeze is not None:
                # Freeze 모드: pretrained 모델을 사용하여 unfreeze_ratio 평가
                val_loss = evaluate_with_freeze(
                    model_for_freeze, X_train, y_train, X_val, y_val,
                    unfreeze_ratio, learning_rate, epochs, device, trial
                )
            elif use_loocv and X_all is not None:
                # LOOCV 사용
                val_loss = evaluate_with_loocv(
                    X_all, y_all,
                    hidden_layers, hidden_dim,
                    learning_rate, epochs,
                    device,
                    use_uncertainty_loss=use_uncertainty_loss,
                    uncertainty_weight=uncertainty_weight
                )
            elif optimize_incremental and incremental_params:
                # Incremental learning 시뮬레이션
                model = OptunaDNN(input_dim, hidden_layers, hidden_dim, device)
                val_loss = _simulate_incremental_learning(
                    model, X_train, y_train, X_val, y_val,
                    learning_rate, epochs, incremental_params, device, trial
                )
            else:
                # 기본 Train/Val 분할 방식
                val_loss = evaluate_with_train_val_split(
                    X_train, y_train, X_val, y_val,
                    hidden_layers, hidden_dim,
                    learning_rate, epochs,
                    device, trial,
                    use_uncertainty_loss=use_uncertainty_loss,
                    uncertainty_weight=uncertainty_weight
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
                                        stage: str = 'pretrain',
                                        fixed_structure: Dict = None,
                                        verbose: bool = True, optimize_incremental: bool = False,
                                        use_loocv: bool = False,
                                        use_uncertainty_loss: bool = False,
                                        uncertainty_weight: float = 0.3,
                                        use_freeze: bool = False,
                                        model_for_freeze = None) -> Tuple[Dict, float, List]:
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
        stage: 'pretrain' 또는 'finetune' - 최적화 단계
        fixed_structure: finetune일 때 사용할 고정된 구조 {'hidden_layers': int, 'hidden_dim': int}
        verbose: 상세 출력
        optimize_incremental: incremental learning 최적화 여부
        use_loocv: LOOCV 사용 여부 (기본: False, 기존 Train/Val 분할)
        use_uncertainty_loss: 불확실성 calibration 손실 사용 여부 (기본: False, 기존 MSE만)
        uncertainty_weight: 불확실성 손실 가중치 (기본: 0.3, L_total = 0.7*MSE + 0.3*L_uncertainty)
        use_freeze: Freeze 기법 사용 여부 (finetune 단계에서 unfreeze_ratio 최적화)
        model_for_freeze: Freeze 평가를 위한 pretrained 모델

    Returns:
        최적 하이퍼파라미터, 최적 성능, 전체 기록
    """

    # Optuna study 생성
    sampler = TPESampler(seed=42)
    # LOOCV 사용 시 pruning 비활성화 (LOOCV는 중간 결과를 report하지 않음)
    if use_loocv:
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=20)

    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner
    )

    # Objective function 생성
    objective = create_optuna_objective(
        X_train, y_train, X_val, y_val, input_dim, device, data_size,
        stage, fixed_structure, optimize_incremental,
        use_loocv=use_loocv,
        use_uncertainty_loss=use_uncertainty_loss,
        uncertainty_weight=uncertainty_weight,
        use_freeze=use_freeze,
        model_for_freeze=model_for_freeze
    )
    
    # 최적화 실행 with progress bar
    stage_prefix = "Pretrain" if stage == 'pretrain' else "Finetune"

    # 평가 방법 문자열 생성
    eval_method = "LOOCV" if use_loocv else "Train/Val"
    if use_freeze and stage == 'finetune':
        eval_method = "Freeze"
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

    # 모든 시행 기록
    trial_history = []
    for trial in study.trials:
        if trial.value is not None:
            record = trial.params.copy()
            record['validation_loss'] = trial.value
            trial_history.append(record)

    if verbose:
        if stage == 'pretrain':
            print(f"      ✅ Best {stage_prefix} params: layers={best_params['hidden_layers']}, dim={best_params['hidden_dim']}, lr={best_params['learning_rate']:.1e}, epochs={best_params['epochs']}")
        else:
            if use_freeze and 'unfreeze_ratio' in best_params:
                print(f"      ✅ Best {stage_prefix} params: lr={best_params['learning_rate']:.1e}, epochs={best_params['epochs']}, unfreeze_ratio={best_params['unfreeze_ratio']:.2f}")
            else:
                print(f"      ✅ Best {stage_prefix} params: lr={best_params['learning_rate']:.1e}, epochs={best_params['epochs']} (structure fixed)")
        print(f"      ✅ Best loss: {best_performance:.4f} (eval: {eval_method}, loss: {loss_type})")

    return best_params, best_performance, trial_history