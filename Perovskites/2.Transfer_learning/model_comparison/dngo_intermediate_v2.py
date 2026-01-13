"""
DNGO-Intermediate 모델 (논문 기반 True Intermediate 구조)

핵심 차이점 (vs GP-mimic):
- GP-mimic: X → Shared → [LF head, HF head] (병렬)
- Intermediate: X → LF Network → y_LF, [X, y_LF] → HF Network → y_HF (직렬)

HF 예측 시 LF 예측값을 입력으로 사용하는 것이 핵심.
이는 ρ(x)·f_L(x) + δ(x) 형태의 AR(1) 모델을 신경망으로 구현한 것.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple


class LFNetwork(nn.Module):
    """Low-Fidelity 예측 네트워크"""

    def __init__(self, input_dim: int, hidden_dims: List[int], activation: str = 'tanh', dropout: float = 0.0):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)
        self.feature_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LF 예측값 반환"""
        return self.network(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """마지막 hidden layer의 feature 반환 (BLR용)"""
        # 출력층 직전까지만 실행
        for layer in list(self.network)[:-1]:
            x = layer(x)
        return x


class HFNetwork(nn.Module):
    """
    High-Fidelity 예측 네트워크 (Intermediate 구조)

    입력: [X, y_LF] (원본 입력 + LF 예측값)
    출력: y_HF

    이는 y_HF = ρ(x)·y_LF + δ(x) 형태를 학습하는 것과 유사
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], activation: str = 'tanh', dropout: float = 0.0):
        """
        Args:
            input_dim: 원본 입력 차원 (X의 차원)
            hidden_dims: hidden layer 차원들
            activation: 활성화 함수
            dropout: dropout 비율
        """
        super().__init__()

        # 입력: [X, y_LF] → input_dim + 1
        augmented_input_dim = input_dim + 1

        layers = []
        prev_dim = augmented_input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)
        self.feature_dim = hidden_dims[-1] if hidden_dims else augmented_input_dim

    def forward(self, x: torch.Tensor, y_lf: torch.Tensor) -> torch.Tensor:
        """
        HF 예측값 반환

        Args:
            x: 원본 입력 [batch, input_dim]
            y_lf: LF 예측값 [batch, 1] 또는 [batch]
        """
        if y_lf.dim() == 1:
            y_lf = y_lf.unsqueeze(-1)

        # [X, y_LF] 결합
        augmented_input = torch.cat([x, y_lf], dim=-1)
        return self.network(augmented_input)

    def extract_features(self, x: torch.Tensor, y_lf: torch.Tensor) -> torch.Tensor:
        """마지막 hidden layer의 feature 반환 (BLR용)"""
        if y_lf.dim() == 1:
            y_lf = y_lf.unsqueeze(-1)

        augmented_input = torch.cat([x, y_lf], dim=-1)

        # 출력층 직전까지만 실행
        for layer in list(self.network)[:-1]:
            augmented_input = layer(augmented_input)
        return augmented_input


class IntermediateMFNN_v2(nn.Module):
    """
    True Intermediate Multi-Fidelity Neural Network

    구조:
        Stage 1: X → LF Network → y_LF
        Stage 2: [X, y_LF] → HF Network → y_HF

    핵심: HF 예측 시 LF 예측값을 명시적으로 입력에 포함
    """

    def __init__(
        self,
        input_dim: int,
        lf_hidden_dims: List[int] = [64, 64],
        hf_hidden_dims: List[int] = [64, 32],
        activation: str = 'tanh',
        device: str = 'cpu',
        lf_dropout: float = 0.0,
        hf_dropout: float = 0.0
    ):
        """
        Args:
            input_dim: 입력 차원
            lf_hidden_dims: LF 네트워크 hidden layer 차원들
            hf_hidden_dims: HF 네트워크 hidden layer 차원들
            activation: 활성화 함수 ('tanh' or 'relu')
            device: 디바이스
            lf_dropout: LF 네트워크 dropout 비율
            hf_dropout: HF 네트워크 dropout 비율
        """
        super().__init__()

        self.input_dim = input_dim
        self.device = device
        self.activation = activation

        # LF Network: X → y_LF
        self.lf_network = LFNetwork(input_dim, lf_hidden_dims, activation, dropout=lf_dropout)

        # HF Network: [X, y_LF] → y_HF
        self.hf_network = HFNetwork(input_dim, hf_hidden_dims, activation, dropout=hf_dropout)

        self.to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        전체 forward pass

        Returns:
            y_lf: LF 예측값
            y_hf: HF 예측값
        """
        # Stage 1: LF 예측
        y_lf = self.lf_network(x)

        # Stage 2: HF 예측 (LF 예측값을 입력에 포함)
        y_hf = self.hf_network(x, y_lf)

        return y_lf, y_hf

    def predict_lf(self, x: torch.Tensor) -> torch.Tensor:
        """LF만 예측"""
        return self.lf_network(x)

    def predict_hf(self, x: torch.Tensor, y_lf: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        HF 예측

        Args:
            x: 입력
            y_lf: LF 예측값 (None이면 자동으로 계산)
        """
        if y_lf is None:
            y_lf = self.lf_network(x)
        return self.hf_network(x, y_lf)

    def extract_lf_features(self, x: torch.Tensor) -> torch.Tensor:
        """LF 네트워크의 feature 추출 (BLR용)"""
        return self.lf_network.extract_features(x)

    def extract_hf_features(self, x: torch.Tensor, y_lf: Optional[torch.Tensor] = None) -> torch.Tensor:
        """HF 네트워크의 feature 추출 (BLR용)"""
        if y_lf is None:
            y_lf = self.lf_network(x)
        return self.hf_network.extract_features(x, y_lf)


class TrueIntermediateDNGO:
    """
    True Intermediate DNGO (논문 구조 충실 구현)

    학습 전략:
        Option A (Sequential): LF 먼저 학습 → LF 고정 후 HF 학습
        Option B (Joint): LF와 HF 동시 학습 (weighted loss)
    """

    def __init__(
        self,
        input_dim: int,
        lf_hidden_dims: List[int] = [64, 64],
        hf_hidden_dims: List[int] = [64, 32],
        activation: str = 'tanh',
        device: str = 'cpu',
        alpha: float = 0.5,  # HF loss weight (joint training)
        lambda_reg: float = 1e-4,
        lf_dropout: float = 0.0,
        hf_dropout: float = 0.0
    ):
        """
        Args:
            input_dim: 입력 차원
            lf_hidden_dims: LF 네트워크 hidden dims
            hf_hidden_dims: HF 네트워크 hidden dims
            activation: 활성화 함수
            device: 디바이스
            alpha: HF loss 가중치 (0=LF만, 1=HF만)
            lambda_reg: L2 정규화 계수
            lf_dropout: LF 네트워크 dropout
            hf_dropout: HF 네트워크 dropout
        """
        self.input_dim = input_dim
        self.device = device
        self.alpha = alpha
        self.lambda_reg = lambda_reg

        self.model = IntermediateMFNN_v2(
            input_dim=input_dim,
            lf_hidden_dims=lf_hidden_dims,
            hf_hidden_dims=hf_hidden_dims,
            activation=activation,
            device=device,
            lf_dropout=lf_dropout,
            hf_dropout=hf_dropout
        )

        self.train_history = []

    def train_sequential(
        self,
        X_lf: np.ndarray,
        y_lf: np.ndarray,
        X_hf: np.ndarray,
        y_hf: np.ndarray,
        lf_epochs: int = 300,
        hf_epochs: int = 200,
        lf_lr: float = 1e-3,
        hf_lr: float = 1e-4,
        verbose: bool = False
    ) -> Dict:
        """
        Sequential 학습: LF 먼저 → HF 나중에

        Stage 1: LF 네트워크만 학습
        Stage 2: LF 고정, HF 네트워크만 학습
        """
        X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(self.device)
        y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(self.device)
        X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(self.device)
        y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(self.device)

        loss_fn = nn.MSELoss()
        history = {'lf_losses': [], 'hf_losses': []}

        # ========== Stage 1: LF 학습 ==========
        if verbose:
            print("  Stage 1: Training LF network...")

        # LF 파라미터만 학습
        lf_optimizer = optim.Adam(self.model.lf_network.parameters(), lr=lf_lr,
                                   weight_decay=self.lambda_reg)

        self.model.train()
        for epoch in range(lf_epochs):
            lf_optimizer.zero_grad()
            y_lf_pred = self.model.predict_lf(X_lf_t)
            lf_loss = loss_fn(y_lf_pred, y_lf_t)
            lf_loss.backward()
            lf_optimizer.step()
            history['lf_losses'].append(lf_loss.item())

        if verbose:
            print(f"    LF final loss: {history['lf_losses'][-1]:.6f}")

        # ========== Stage 2: HF 학습 (LF 고정) ==========
        if verbose:
            print("  Stage 2: Training HF network (LF frozen)...")

        # LF 네트워크 고정
        for param in self.model.lf_network.parameters():
            param.requires_grad = False

        # HF 파라미터만 학습
        hf_optimizer = optim.Adam(self.model.hf_network.parameters(), lr=hf_lr,
                                   weight_decay=self.lambda_reg)

        for epoch in range(hf_epochs):
            hf_optimizer.zero_grad()

            # LF 예측값 계산 (no grad)
            with torch.no_grad():
                y_lf_pred_for_hf = self.model.predict_lf(X_hf_t)

            # HF 예측
            y_hf_pred = self.model.hf_network(X_hf_t, y_lf_pred_for_hf)
            hf_loss = loss_fn(y_hf_pred, y_hf_t)
            hf_loss.backward()
            hf_optimizer.step()
            history['hf_losses'].append(hf_loss.item())

        # LF 네트워크 해동
        for param in self.model.lf_network.parameters():
            param.requires_grad = True

        if verbose:
            print(f"    HF final loss: {history['hf_losses'][-1]:.6f}")

        self.train_history = history
        return history

    def train_joint(
        self,
        X_lf: np.ndarray,
        y_lf: np.ndarray,
        X_hf: np.ndarray,
        y_hf: np.ndarray,
        epochs: int = 300,
        lr: float = 1e-3,
        hf_lr_ratio: float = 0.1,  # HF learning rate = lr * hf_lr_ratio
        hf_weight_decay: float = 1e-3,  # HF에 더 강한 L2 정규화
        verbose: bool = False
    ) -> Dict:
        """
        Joint 학습: LF와 HF 동시 학습 (HF에 별도 learning rate 적용)

        Loss = (1-α)·MSE_LF + α·MSE_HF

        Args:
            hf_lr_ratio: HF learning rate = lr * hf_lr_ratio (기본 0.1 = 10배 느림)
            hf_weight_decay: HF 네트워크의 L2 정규화 강도
        """
        X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(self.device)
        y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(self.device)
        X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(self.device)
        y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(self.device)

        loss_fn = nn.MSELoss()

        # LF와 HF에 별도의 learning rate 및 weight_decay 적용
        optimizer = optim.Adam([
            {'params': self.model.lf_network.parameters(), 'lr': lr, 'weight_decay': self.lambda_reg},
            {'params': self.model.hf_network.parameters(), 'lr': lr * hf_lr_ratio, 'weight_decay': hf_weight_decay}
        ])

        history = {'total_losses': [], 'lf_losses': [], 'hf_losses': [], 'best_val_loss': float('inf')}

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # LF 예측 (LF 데이터)
            y_lf_pred = self.model.predict_lf(X_lf_t)
            lf_loss = loss_fn(y_lf_pred, y_lf_t)

            # HF 예측 (HF 데이터) - LF 예측값을 입력에 포함
            y_lf_pred_for_hf = self.model.predict_lf(X_hf_t)
            y_hf_pred = self.model.hf_network(X_hf_t, y_lf_pred_for_hf)
            hf_loss = loss_fn(y_hf_pred, y_hf_t)

            # Weighted loss
            total_loss = (1 - self.alpha) * lf_loss + self.alpha * hf_loss

            total_loss.backward()
            optimizer.step()

            history['total_losses'].append(total_loss.item())
            history['lf_losses'].append(lf_loss.item())
            history['hf_losses'].append(hf_loss.item())

            if total_loss.item() < history['best_val_loss']:
                history['best_val_loss'] = total_loss.item()

        if verbose:
            print(f"  Final - LF loss: {history['lf_losses'][-1]:.6f}, "
                  f"HF loss: {history['hf_losses'][-1]:.6f}")

        self.train_history = history
        return history

    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, ...]:
        """
        HF 예측

        Args:
            X: 입력 데이터
            return_std: True면 불확실성도 반환 (현재는 더미값)
        """
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            _, y_hf = self.model(X_t)
            pred = y_hf.cpu().numpy().flatten()

        if return_std:
            # 더미 std (BLR 사용 시 대체됨)
            return pred, np.zeros_like(pred)
        return pred

    def predict_lf(self, X: np.ndarray) -> np.ndarray:
        """LF 예측"""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            y_lf = self.model.predict_lf(X_t)
            return y_lf.cpu().numpy().flatten()

    def extract_features(self, X: np.ndarray, for_fidelity: str = 'high') -> np.ndarray:
        """
        BLR용 feature 추출

        Args:
            X: 입력 데이터
            for_fidelity: 'low' or 'high'
        """
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)

            if for_fidelity == 'low':
                features = self.model.extract_lf_features(X_t)
            else:
                features = self.model.extract_hf_features(X_t)

            return features.cpu().numpy()


class BayesianLinearRegression:
    """베이지안 선형 회귀 (불확실성 추정용)"""

    def __init__(self, alpha: float = 1.0, beta: float = 25.0):
        self.alpha = alpha
        self.beta = beta
        self.mean = None
        self.cov = None
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).flatten()

        # Add bias
        X_bias = np.column_stack([np.ones(len(X)), X])

        # Posterior
        S0_inv = self.alpha * np.eye(X_bias.shape[1])
        S_N_inv = S0_inv + self.beta * X_bias.T @ X_bias
        self.cov = np.linalg.inv(S_N_inv)
        self.mean = self.beta * self.cov @ X_bias.T @ y
        self.fitted = True

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        x = np.asarray(x, dtype=np.float32).flatten()
        x_bias = np.concatenate([[1], x])

        mu = x_bias @ self.mean
        var = (1/self.beta) + x_bias @ self.cov @ x_bias

        return mu, var

    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float32)
        means, variances = [], []

        for x in X:
            mu, var = self.predict(x)
            means.append(mu)
            variances.append(var)

        return np.array(means), np.array(variances)


# ============== 편의 함수 ==============

def create_intermediate_dngo(
    input_dim: int,
    lf_width: int = 64,
    lf_layers: int = 2,
    hf_width: int = 32,
    hf_layers: int = 2,
    activation: str = 'tanh',
    device: str = 'cpu',
    alpha: float = 0.5,
    lambda_reg: float = 1e-4
) -> TrueIntermediateDNGO:
    """
    편의 함수: Intermediate DNGO 생성

    Args:
        input_dim: 입력 차원
        lf_width: LF 네트워크 width
        lf_layers: LF 네트워크 layer 수
        hf_width: HF 네트워크 width
        hf_layers: HF 네트워크 layer 수
        activation: 활성화 함수
        device: 디바이스
        alpha: HF loss 가중치
        lambda_reg: L2 정규화
    """
    lf_hidden_dims = [lf_width] * lf_layers
    hf_hidden_dims = [hf_width] * hf_layers

    return TrueIntermediateDNGO(
        input_dim=input_dim,
        lf_hidden_dims=lf_hidden_dims,
        hf_hidden_dims=hf_hidden_dims,
        activation=activation,
        device=device,
        alpha=alpha,
        lambda_reg=lambda_reg
    )
