"""
DNN 하이퍼파라미터 베이지안 최적화 모듈

Pretrain과 Finetune 단계에서 각각 베이지안 최적화를 통해
최적의 하이퍼파라미터를 찾습니다.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from models import TransferLearningDNN
import time


class HyperparameterSpace:
    """하이퍼파라미터 탐색 공간 정의"""
    
    def __init__(self, data_size: int = 'small'):
        """
        Args:
            data_size: 'small', 'medium', 'large' - 데이터 크기에 따른 탐색 공간 조정
        """
        if data_size == 'small':
            # 작은 데이터셋용 (적은 파라미터)
            self.hidden_layers = [1, 2, 3]  # 레이어 수
            self.hidden_dims = [16, 32, 64, 128]  # 뉴런 수
            self.learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]  # 학습률
            self.epochs_range = (20, 200)  # epoch 범위
        elif data_size == 'medium':
            self.hidden_layers = [1, 2, 3, 4]
            self.hidden_dims = [32, 64, 128, 256]
            self.learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
            self.epochs_range = (50, 500)
        else:  # large
            self.hidden_layers = [2, 3, 4, 5]
            self.hidden_dims = [64, 128, 256, 512]
            self.learning_rates = [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
            self.epochs_range = (100, 1000)
    
    def sample_random(self) -> Dict:
        """랜덤 하이퍼파라미터 샘플링"""
        return {
            'hidden_layers': np.random.choice(self.hidden_layers),
            'hidden_dim': np.random.choice(self.hidden_dims),
            'learning_rate': np.random.choice(self.learning_rates),
            'epochs': np.random.randint(self.epochs_range[0], self.epochs_range[1] + 1)
        }
    
    def normalize_params(self, params: Dict) -> np.ndarray:
        """하이퍼파라미터를 [0,1] 범위로 정규화"""
        normalized = np.zeros(4)
        
        # hidden_layers 정규화
        normalized[0] = (params['hidden_layers'] - min(self.hidden_layers)) / \
                       (max(self.hidden_layers) - min(self.hidden_layers))
        
        # hidden_dim 정규화 (log scale)
        log_dims = np.log2(self.hidden_dims)
        log_param = np.log2(params['hidden_dim'])
        normalized[1] = (log_param - min(log_dims)) / (max(log_dims) - min(log_dims))
        
        # learning_rate 정규화 (log scale)
        log_lrs = np.log10(self.learning_rates)
        log_lr = np.log10(params['learning_rate'])
        normalized[2] = (log_lr - min(log_lrs)) / (max(log_lrs) - min(log_lrs))
        
        # epochs 정규화
        normalized[3] = (params['epochs'] - self.epochs_range[0]) / \
                       (self.epochs_range[1] - self.epochs_range[0])
        
        return normalized
    
    def denormalize_params(self, normalized: np.ndarray) -> Dict:
        """정규화된 값을 실제 하이퍼파라미터로 변환"""
        # hidden_layers
        layers_range = max(self.hidden_layers) - min(self.hidden_layers)
        layers = int(min(self.hidden_layers) + normalized[0] * layers_range)
        layers = max(min(self.hidden_layers), min(max(self.hidden_layers), layers))
        
        # hidden_dim (가장 가까운 값 선택)
        log_dims = np.log2(self.hidden_dims)
        log_range = max(log_dims) - min(log_dims)
        target_log = min(log_dims) + normalized[1] * log_range
        dim_idx = np.argmin(np.abs(log_dims - target_log))
        hidden_dim = self.hidden_dims[dim_idx]
        
        # learning_rate (가장 가까운 값 선택)
        log_lrs = np.log10(self.learning_rates)
        log_range = max(log_lrs) - min(log_lrs)
        target_log = min(log_lrs) + normalized[2] * log_range
        lr_idx = np.argmin(np.abs(log_lrs - target_log))
        learning_rate = self.learning_rates[lr_idx]
        
        # epochs
        epochs = int(self.epochs_range[0] + normalized[3] * 
                    (self.epochs_range[1] - self.epochs_range[0]))
        epochs = max(self.epochs_range[0], min(self.epochs_range[1], epochs))
        
        return {
            'hidden_layers': layers,
            'hidden_dim': hidden_dim,
            'learning_rate': learning_rate,
            'epochs': epochs
        }


class DynamicDNN(nn.Module):
    """동적으로 구조를 변경할 수 있는 DNN"""
    
    def __init__(self, input_dim: int, hidden_layers: int, hidden_dim: int, device: str = 'cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.device = device
        
        # 동적으로 레이어 생성
        layers = []
        
        # 첫 번째 레이어
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # 중간 레이어들
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.feature_net = nn.Sequential(*layers).to(device)
        self.out_layer = nn.Linear(hidden_dim, 1, bias=False).to(device)
        
        # float32로 설정
        self.feature_net = self.feature_net.float()
        self.out_layer = self.out_layer.float()
    
    def forward(self, x):
        features = self.feature_net(x)
        return self.out_layer(features)
    
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Feature 추출"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            features = self.feature_net(X_tensor)
            return features.cpu().numpy()


class HyperparameterBO:
    """하이퍼파라미터 베이지안 최적화"""
    
    def __init__(self, param_space: HyperparameterSpace, n_initial: int = 5):
        self.param_space = param_space
        self.n_initial = n_initial
        self.X_observed = []  # 정규화된 하이퍼파라미터
        self.y_observed = []  # 성능 (음수로 저장, 최소화 문제로 변환)
        self.param_history = []  # 실제 하이퍼파라미터 기록
        
        # GP 모델
        kernel = Matern(length_scale=0.5, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )
    
    def _evaluate_hyperparameters(self, params: Dict, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray, 
                                 input_dim: int, device: str, verbose: bool = False) -> float:
        """하이퍼파라미터 조합의 성능 평가"""
        try:
            # 동적 모델 생성
            model = DynamicDNN(
                input_dim=input_dim,
                hidden_layers=params['hidden_layers'],
                hidden_dim=params['hidden_dim'],
                device=device
            )
            
            # 데이터 준비
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
            
            # 옵티마이저 및 손실함수
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
            loss_fn = nn.MSELoss()
            
            # 학습
            model.train()
            for epoch in range(params['epochs']):
                optimizer.zero_grad()
                pred = model(X_train_tensor)
                loss = loss_fn(pred, y_train_tensor)
                loss.backward()
                optimizer.step()
                
                if verbose and (epoch + 1) % max(1, params['epochs'] // 5) == 0:
                    print(f"  Epoch {epoch+1}/{params['epochs']}: Loss {loss.item():.4f}")
            
            # 검증 성능 평가
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_tensor)
                val_loss = loss_fn(val_pred, y_val_tensor).item()
            
            return val_loss
            
        except Exception as e:
            if verbose:
                print(f"  Error evaluating params {params}: {e}")
            return float('inf')  # 오류 시 매우 큰 값 반환
    
    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """Expected Improvement 계산"""
        if len(self.y_observed) == 0:
            return np.ones(X.shape[0])
        
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-8)
        
        # 현재까지의 최적값
        f_best = np.min(self.y_observed)
        
        # Expected Improvement
        xi = 0.01  # exploration parameter
        z = (f_best - mu - xi) / sigma
        ei = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        
        return ei
    
    def _optimize_acquisition(self) -> np.ndarray:
        """Acquisition function 최적화"""
        best_ei = -np.inf
        best_x = None
        
        # 여러 시작점에서 최적화
        for _ in range(20):
            x0 = np.random.uniform(0, 1, 4)
            
            def neg_acquisition(x):
                return -self._acquisition_function(x.reshape(1, -1))[0]
            
            # 제약 조건: [0, 1] 범위
            bounds = [(0, 1) for _ in range(4)]
            
            try:
                result = minimize(neg_acquisition, x0, bounds=bounds, method='L-BFGS-B')
                if result.success and -result.fun > best_ei:
                    best_ei = -result.fun
                    best_x = result.x
            except:
                continue
        
        if best_x is None:
            # 최적화 실패 시 랜덤 샘플링
            best_x = np.random.uniform(0, 1, 4)
        
        return best_x
    
    def suggest_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               input_dim: int, device: str = 'cpu', 
                               verbose: bool = False) -> Dict:
        """다음 하이퍼파라미터 조합 제안"""
        
        if len(self.X_observed) < self.n_initial:
            # 초기 랜덤 샘플링
            params = self.param_space.sample_random()
            if verbose:
                print(f"Initial random sampling: {params}")
        else:
            # GP 기반 제안
            if verbose:
                print("GP-based suggestion...")
            
            # GP 학습
            X_array = np.array(self.X_observed)
            y_array = np.array(self.y_observed)
            self.gp.fit(X_array, y_array)
            
            # 다음 포인트 제안
            x_next = self._optimize_acquisition()
            params = self.param_space.denormalize_params(x_next)
            
            if verbose:
                print(f"GP-suggested params: {params}")
        
        # 성능 평가
        if verbose:
            print("Evaluating hyperparameters...")
        
        performance = self._evaluate_hyperparameters(
            params, X_train, y_train, X_val, y_val, input_dim, device, verbose
        )
        
        # 결과 기록
        normalized_params = self.param_space.normalize_params(params)
        self.X_observed.append(normalized_params)
        self.y_observed.append(performance)  # 최소화 문제
        self.param_history.append(params.copy())
        
        if verbose:
            print(f"Performance (validation loss): {performance:.4f}")
            print(f"Best so far: {np.min(self.y_observed):.4f}")
        
        return params, performance
    
    def get_best_hyperparameters(self) -> Tuple[Dict, float]:
        """최적 하이퍼파라미터 반환"""
        if not self.y_observed:
            return self.param_space.sample_random(), float('inf')
        
        best_idx = np.argmin(self.y_observed)
        return self.param_history[best_idx], self.y_observed[best_idx]


def optimize_dnn_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                input_dim: int, n_trials: int = 10,
                                data_size: str = 'small', device: str = 'cpu',
                                verbose: bool = True) -> Tuple[Dict, float, List]:
    """
    DNN 하이퍼파라미터 베이지안 최적화
    
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
    print(f"Starting hyperparameter optimization with {n_trials} trials...")
    
    # 하이퍼파라미터 공간 및 BO 초기화
    param_space = HyperparameterSpace(data_size)
    bo = HyperparameterBO(param_space, n_initial=min(3, n_trials))
    
    # 베이지안 최적화 실행
    for trial in range(n_trials):
        print(f"\n--- Trial {trial + 1}/{n_trials} ---")
        start_time = time.time()
        
        params, performance = bo.suggest_hyperparameters(
            X_train, y_train, X_val, y_val, input_dim, device, verbose
        )
        
        elapsed = time.time() - start_time
        print(f"Trial {trial + 1} completed in {elapsed:.2f}s")
        print(f"Params: {params}")
        print(f"Performance: {performance:.4f}")
    
    # 최적 결과 반환
    best_params, best_performance = bo.get_best_hyperparameters()
    
    print(f"\n=== Optimization Complete ===")
    print(f"Best hyperparameters: {best_params}")
    print(f"Best performance: {best_performance:.4f}")
    
    return best_params, best_performance, bo.param_history 