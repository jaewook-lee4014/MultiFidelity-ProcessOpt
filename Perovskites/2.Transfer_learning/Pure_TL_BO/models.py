import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.model_selection import train_test_split


class TransferLearningDNN:
    """
    Transfer Learning을 위한 Deep Neural Network 클래스
    - Pretrain: low-fidelity 데이터로 feature extractor 학습
    - Finetune: high-fidelity 데이터로 전체 네트워크 미세조정
    - 하이퍼파라미터 베이지안 최적화 지원
    """
    
    def __init__(self, input_dim, hidden_dim=64, device='cpu', use_hyperparameter_bo=False):
        self.input_dim = input_dim
        self.device = device
        self.hidden_dim = hidden_dim
        self.use_hyperparameter_bo = use_hyperparameter_bo
        self.pretrain_losses = []
        self.finetune_losses = []
        
        # BO 관련 변수
        self.pretrain_best_params = None
        self.finetune_best_params = None
        self.pretrain_bo_history = []
        self.finetune_bo_history = []

        # 기본 모델 구조 (BO 사용하지 않을 때)
        if not use_hyperparameter_bo:
            self._build_default_model(hidden_dim)
    
    def _build_default_model(self, hidden_dim):
        """기본 모델 구조 생성"""
        # feature extractor (hidden layers)
        self.feature_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ).to(self.device)
        
        # 출력층
        self.out_layer = nn.Linear(hidden_dim, 1, bias=False).to(self.device)
        
        # 전체 모델
        self.model = nn.Sequential(self.feature_net, self.out_layer)
        
        # float32로 설정
        self.feature_net = self.feature_net.float()
        self.out_layer = self.out_layer.float()
        self.model = self.model.float()
    
    def _build_dynamic_model(self, params: Dict):
        """동적 모델 구조 생성 (BO 결과 기반)"""
        layers = []
        
        # 첫 번째 레이어
        layers.append(nn.Linear(self.input_dim, params['hidden_dim']))
        layers.append(nn.ReLU())
        
        # 중간 레이어들
        for _ in range(params['hidden_layers'] - 1):
            layers.append(nn.Linear(params['hidden_dim'], params['hidden_dim']))
            layers.append(nn.ReLU())
        
        self.feature_net = nn.Sequential(*layers).to(self.device)
        self.out_layer = nn.Linear(params['hidden_dim'], 1, bias=False).to(self.device)
        self.model = nn.Sequential(self.feature_net, self.out_layer)
        
        # float32로 설정
        self.feature_net = self.feature_net.float()
        self.out_layer = self.out_layer.float()
        self.model = self.model.float()
    
    def _split_validation_data(self, X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2) -> Tuple:
        """검증 데이터 분할"""
        if len(X) < 3:  # 데이터가 너무 적으면 분할하지 않음
            return X, y, X, y
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_ratio, random_state=42
        )
        return X_train, y_train, X_val, y_val

    def pretrain(self, X_low, y_low, epochs=50, lr=1e-3, verbose=False, 
                 bo_trials=None, data_size='small'):
        """
        Low-fidelity 데이터로 pretrain
        
        Args:
            X_low: low-fidelity 입력 데이터
            y_low: low-fidelity 출력 데이터
            epochs: 기본 epoch 수 (BO 사용 시 무시됨)
            lr: 기본 학습률 (BO 사용 시 무시됨)
            verbose: 상세 출력
            bo_trials: BO 시행 횟수 (None이면 BO 사용 안함)
            data_size: 데이터 크기 ('small', 'medium', 'large')
        """
        self.pretrain_losses = []
        X_low = np.asarray(X_low, dtype=np.float32)
        y_low = np.asarray(y_low, dtype=np.float32).flatten()
        
        if verbose:
            print(f"Pretrain with {len(X_low)} low-fidelity samples")
        
        if self.use_hyperparameter_bo and bo_trials is not None and bo_trials > 0:
            # 베이지안 최적화로 하이퍼파라미터 찾기
            if verbose:
                print(f"🔍 Optimizing pretrain hyperparameters with {bo_trials} trials...")
            
            # 검증 데이터 분할
            X_train, y_train, X_val, y_val = self._split_validation_data(X_low, y_low)
            
            # BO 실행
            from hyperparameter_optimization import optimize_dnn_hyperparameters
            best_params, best_performance, history = optimize_dnn_hyperparameters(
                X_train, y_train, X_val, y_val, 
                input_dim=self.input_dim,
                n_trials=bo_trials,
                data_size=data_size,
                device=self.device,
                verbose=verbose
            )
            
            self.pretrain_best_params = best_params
            self.pretrain_bo_history = history
            
            if verbose:
                print(f"✅ Best pretrain params: {best_params}")
                print(f"✅ Best validation loss: {best_performance:.4f}")
            
            # 최적 하이퍼파라미터로 모델 구성
            self._build_dynamic_model(best_params)
            epochs = best_params['epochs']
            lr = best_params['learning_rate']
        else:
            # 기본 하이퍼파라미터 사용
            if not hasattr(self, 'model'):
                self._build_default_model(self.hidden_dim)
        
        # 전체 데이터로 최종 학습
        X_tensor = torch.tensor(X_low, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_low, dtype=torch.float32).view(-1, 1).to(self.device)
        optimizer = optim.Adam(list(self.feature_net.parameters()) + list(self.out_layer.parameters()), lr=lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            features = self.feature_net(X_tensor)
            pred = self.out_layer(features)
            loss = loss_fn(pred, y_tensor)
            loss.backward()
            optimizer.step()
            self.pretrain_losses.append(loss.item())
            
            if verbose and (epoch+1) % max(1, epochs//10) == 0:
                print(f'[Pretrain] Epoch {epoch+1}/{epochs}: Loss {loss.item():.4f}')

    def finetune(self, X_high, y_high, epochs=50, lr=1e-4, verbose=False,
                 bo_trials=None, data_size='small'):
        """
        High-fidelity 데이터로 finetune
        
        Args:
            X_high: high-fidelity 입력 데이터
            y_high: high-fidelity 출력 데이터
            epochs: 기본 epoch 수 (BO 사용 시 무시됨)
            lr: 기본 학습률 (BO 사용 시 무시됨)
            verbose: 상세 출력
            bo_trials: BO 시행 횟수 (None이면 BO 사용 안함)
            data_size: 데이터 크기 ('small', 'medium', 'large')
        """
        self.finetune_losses = []
        X_high = np.asarray(X_high, dtype=np.float32)
        y_high = np.asarray(y_high, dtype=np.float32).flatten()
        
        if verbose:
            print(f"Finetune with {len(X_high)} high-fidelity samples")
        
        if self.use_hyperparameter_bo and bo_trials is not None and bo_trials > 0:
            # 베이지안 최적화로 하이퍼파라미터 찾기
            if verbose:
                print(f"🔍 Optimizing finetune hyperparameters with {bo_trials} trials...")
            
            # 검증 데이터 분할
            X_train, y_train, X_val, y_val = self._split_validation_data(X_high, y_high)
            
            # 현재 feature extractor의 출력 차원 확인
            with torch.no_grad():
                sample_input = torch.tensor(X_train[:1], dtype=torch.float32).to(self.device)
                feature_dim = self.feature_net(sample_input).shape[1]
            
            # BO 실행 (feature를 입력으로 사용)
            X_train_features = self.extract_features(X_train)
            X_val_features = self.extract_features(X_val)
            
            from hyperparameter_optimization import optimize_dnn_hyperparameters
            best_params, best_performance, history = optimize_dnn_hyperparameters(
                X_train_features, y_train, X_val_features, y_val,
                input_dim=feature_dim,
                n_trials=bo_trials,
                data_size=data_size,
                device=self.device,
                verbose=verbose
            )
            
            self.finetune_best_params = best_params
            self.finetune_bo_history = history
            
            if verbose:
                print(f"✅ Best finetune params: {best_params}")
                print(f"✅ Best validation loss: {best_performance:.4f}")
            
            # 최적 하이퍼파라미터로 출력층만 재구성 (feature extractor는 유지)
            self.out_layer = nn.Linear(feature_dim, 1, bias=False).to(self.device).float()
            self.model = nn.Sequential(self.feature_net, self.out_layer)
            
            epochs = best_params['epochs']
            lr = best_params['learning_rate']
        
        # 전체 데이터로 최종 학습
        X_tensor = torch.tensor(X_high, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_high, dtype=torch.float32).view(-1, 1).to(self.device)
        optimizer = optim.Adam(list(self.feature_net.parameters()) + list(self.out_layer.parameters()), lr=lr)
        loss_fn = nn.MSELoss()
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            features = self.feature_net(X_tensor)
            pred = self.out_layer(features)
            loss = loss_fn(pred, y_tensor)
            loss.backward()
            optimizer.step()
            self.finetune_losses.append(loss.item())
            
            if verbose and (epoch+1) % max(1, epochs//10) == 0:
                print(f'[Finetune] Epoch {epoch+1}/{epochs}: Loss {loss.item():.4f}')

    def predict(self, X):
        """예측"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            pred = self.model(X_tensor)
            return pred.cpu().numpy().flatten()

    def extract_features(self, X):
        """Feature 추출"""
        self.feature_net.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            features = self.feature_net(X_tensor)
            return features.cpu().numpy()
    
    def get_hyperparameter_summary(self) -> Dict:
        """하이퍼파라미터 최적화 결과 요약"""
        summary = {
            'use_hyperparameter_bo': self.use_hyperparameter_bo,
            'pretrain_best_params': self.pretrain_best_params,
            'finetune_best_params': self.finetune_best_params,
            'pretrain_trials': len(self.pretrain_bo_history),
            'finetune_trials': len(self.finetune_bo_history)
        }
        return summary


class BayesianLinearRegression:
    """
    베이지안 선형 회귀 모델
    불확실성을 포함한 예측을 제공합니다.
    """
    
    def __init__(self, alpha=1.0, beta=25.0):
        """
        Args:
            alpha: 가중치의 정밀도 (precision) 파라미터
            beta: 노이즈의 정밀도 파라미터
        """
        self.alpha = alpha
        self.beta = beta
        self.mean = None
        self.cov = None
        self.fitted = False
    
    def fit(self, X, y):
        """
        베이지안 선형 회귀 학습
        
        Args:
            X: 입력 특성 (N x D)
            y: 타겟 값 (N,)
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).flatten()
        
        # 편향 항 추가
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        
        # 사전 분포: w ~ N(0, α^(-1)I)
        S0_inv = self.alpha * np.eye(X_with_bias.shape[1])
        
        # 사후 분포 계산
        S_N_inv = S0_inv + self.beta * X_with_bias.T @ X_with_bias
        self.cov = np.linalg.inv(S_N_inv)
        self.mean = self.beta * self.cov @ X_with_bias.T @ y
        
        self.fitted = True
    
    def predict(self, x):
        """
        단일 점에 대한 예측
        
        Args:
            x: 입력 특성 벡터 (D,)
            
        Returns:
            평균, 분산
        """
        if not self.fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        x = np.asarray(x, dtype=np.float32).flatten()
        x_with_bias = np.concatenate([[1], x])
        
        # 예측 평균
        mu = x_with_bias @ self.mean
        
        # 예측 분산
        var = (1/self.beta) + x_with_bias @ self.cov @ x_with_bias
        
        return mu, var
    
    def predict_batch(self, X):
        """
        배치 예측
        
        Args:
            X: 입력 특성 행렬 (N x D)
            
        Returns:
            평균들, 분산들
        """
        X = np.asarray(X, dtype=np.float32)
        means = []
        variances = []
        
        for x in X:
            mu, var = self.predict(x)
            means.append(mu)
            variances.append(var)
        
        return np.array(means), np.array(variances) 