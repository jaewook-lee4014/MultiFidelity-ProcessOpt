"""
DNGO-Intermediate Multi-Fidelity Neural Network

Based on "All-in-one NNMFR" from the MF-DNN architecture paper.
The key innovation is that LF output is situated in an intermediate hidden layer,
and HF output uses this LF prediction as additional input.

Architecture:
    input x → (Shared layers) → y_LF (3rd hidden layer) → (HF-specific layers) → y_HF

Loss function:
    L = α * MSE_HF + (1-α) * MSE_LF + λ * ||W||²

Reference:
    Multi-fidelity deep neural network (MF-DNN) architecture paper, Section 3.1
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple, List
from sklearn.model_selection import train_test_split


class IntermediateMFNN(nn.Module):
    """
    Intermediate Multi-Fidelity Neural Network

    논문의 All-in-one NNMFR (Fig. 1a) 구조 구현:
    - LF 출력은 3번째 hidden layer에 위치
    - HF 출력은 LF 출력을 입력으로 받아 최종 레이어에서 예측
    - 모든 hidden layer는 tanh 활성화 함수 사용
    - 출력층은 linear (활성화 없음)

    Args:
        input_dim: 입력 차원
        shared_width: Shared layers의 뉴런 수 (기본값 64, 논문 권장)
        num_shared_layers: Shared layers 수 (기본값 3, 논문에서 LF는 3번째 layer)
        hf_hidden_dims: HF-specific layers의 뉴런 수 리스트 (HPO로 결정 가능)
        dropout_rate: Dropout 비율 (기본값 0.0)
    """

    def __init__(self, input_dim: int, shared_width: int = 64,
                 num_shared_layers: int = 3, hf_hidden_dims: List[int] = None,
                 dropout_rate: float = 0.0):
        super().__init__()

        self.input_dim = input_dim
        self.shared_width = shared_width
        self.num_shared_layers = num_shared_layers
        self.hf_hidden_dims = hf_hidden_dims if hf_hidden_dims is not None else [32]
        self.dropout_rate = dropout_rate

        # 1. Shared layers (입력 → LF 출력 위치까지)
        shared_layers = []
        prev_dim = input_dim
        for i in range(num_shared_layers):
            shared_layers.append(nn.Linear(prev_dim, shared_width))
            shared_layers.append(nn.Tanh())  # 논문: tanh activation
            if dropout_rate > 0:
                shared_layers.append(nn.Dropout(dropout_rate))
            prev_dim = shared_width

        self.shared_net = nn.Sequential(*shared_layers)

        # 2. LF output head (3번째 hidden layer에서 LF 예측)
        # Linear output (no activation for regression)
        self.lf_head = nn.Linear(shared_width, 1)

        # 3. HF-specific layers
        # LF 예측값을 입력으로 포함: shared_width + 1 (y_lf)
        hf_layers = []
        prev_dim = shared_width + 1  # concatenate z and y_lf

        for width in self.hf_hidden_dims:
            hf_layers.append(nn.Linear(prev_dim, width))
            hf_layers.append(nn.Tanh())
            if dropout_rate > 0:
                hf_layers.append(nn.Dropout(dropout_rate))
            prev_dim = width

        self.hf_body = nn.Sequential(*hf_layers) if hf_layers else nn.Identity()

        # 4. HF output head
        # Linear output (no activation for regression)
        final_dim = prev_dim if self.hf_hidden_dims else shared_width + 1
        self.hf_head = nn.Linear(final_dim, 1)

        # Weight initialization (Xavier/Glorot)
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: 입력 텐서 (batch_size, input_dim)

        Returns:
            y_lf: LF 예측값 (batch_size, 1)
            y_hf: HF 예측값 (batch_size, 1)
        """
        # Shared layers
        z = self.shared_net(x)

        # LF prediction (intermediate layer)
        y_lf = self.lf_head(z)

        # HF input: concatenate shared features with LF prediction
        hf_input = torch.cat([z, y_lf], dim=1)

        # HF-specific layers
        h = self.hf_body(hf_input)

        # HF prediction
        y_hf = self.hf_head(h)

        return y_lf, y_hf

    def predict_lf(self, x: torch.Tensor) -> torch.Tensor:
        """LF 예측만 수행"""
        z = self.shared_net(x)
        return self.lf_head(z)

    def predict_hf(self, x: torch.Tensor) -> torch.Tensor:
        """HF 예측만 수행"""
        _, y_hf = self.forward(x)
        return y_hf

    def extract_shared_features(self, x: torch.Tensor) -> torch.Tensor:
        """Shared layer의 feature 추출 (BLR용)"""
        return self.shared_net(x)

    def extract_hf_features(self, x: torch.Tensor) -> torch.Tensor:
        """HF layer의 feature 추출 (BLR용)"""
        z = self.shared_net(x)
        y_lf = self.lf_head(z)
        hf_input = torch.cat([z, y_lf], dim=1)

        if len(self.hf_hidden_dims) > 0:
            # HF body를 통과한 feature 반환
            return self.hf_body(hf_input)
        else:
            # HF body가 없으면 hf_input 반환
            return hf_input


class IntermediateDNGO:
    """
    DNGO-Intermediate: Intermediate MFNN + Bayesian Linear Regression

    논문 기반 Multi-Fidelity 학습 파이프라인:
    1. LF/HF 데이터를 동시에 사용하여 IntermediateMFNN 학습
    2. Weighted loss: L = α * MSE_HF + (1-α) * MSE_LF + λ * ||W||²
    3. 학습된 feature로 BLR 적합

    핵심 특징:
    - LF → HF 정보 전달이 네트워크 구조에 내장됨
    - 동시 학습으로 LF/HF 상관관계 활용
    - α 파라미터로 fidelity 간 trade-off 조절
    """

    def __init__(self, input_dim: int, shared_width: int = 64,
                 num_shared_layers: int = 3, hf_hidden_dims: List[int] = None,
                 device: str = 'cpu', alpha: float = 0.5,
                 lambda_reg: float = 1e-4, dropout_rate: float = 0.0):
        """
        Args:
            input_dim: 입력 차원
            shared_width: Shared layers 뉴런 수
            num_shared_layers: Shared layers 수
            hf_hidden_dims: HF-specific layers 뉴런 수 리스트
            device: 디바이스 ('cpu' or 'cuda')
            alpha: HF loss 가중치 (0: LF만, 1: HF만, 0.5: 균형)
            lambda_reg: L2 정규화 가중치
            dropout_rate: Dropout 비율
        """
        self.input_dim = input_dim
        self.shared_width = shared_width
        self.num_shared_layers = num_shared_layers
        self.hf_hidden_dims = hf_hidden_dims if hf_hidden_dims is not None else [32]
        self.device = device
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.dropout_rate = dropout_rate

        # 모델 생성
        self.model = IntermediateMFNN(
            input_dim=input_dim,
            shared_width=shared_width,
            num_shared_layers=num_shared_layers,
            hf_hidden_dims=self.hf_hidden_dims,
            dropout_rate=dropout_rate
        ).to(device).float()

        # 학습 기록
        self.train_losses = []
        self.lf_losses = []
        self.hf_losses = []
        self.val_losses = []

        # BLR 관련
        self.blr_lf = None
        self.blr_hf = None

        # 하이퍼파라미터 최적화 관련
        self.best_params = None
        self.bo_history = []

    def _compute_loss(self, y_lf_pred: torch.Tensor, y_hf_pred: torch.Tensor,
                      y_lf_true: torch.Tensor, y_hf_true: torch.Tensor,
                      has_lf: bool, has_hf: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-fidelity weighted loss 계산

        논문 수식: L = α * MSE_HF + (1-α) * MSE_LF + λ * ||W||²

        Args:
            y_lf_pred: LF 예측값
            y_hf_pred: HF 예측값
            y_lf_true: LF 실제값 (LF 데이터에 대해서만)
            y_hf_true: HF 실제값 (HF 데이터에 대해서만)
            has_lf: LF 데이터 존재 여부
            has_hf: HF 데이터 존재 여부

        Returns:
            total_loss, lf_loss, hf_loss
        """
        mse_loss = nn.MSELoss()

        # LF loss
        if has_lf and y_lf_true is not None and len(y_lf_true) > 0:
            lf_loss = mse_loss(y_lf_pred, y_lf_true)
        else:
            lf_loss = torch.tensor(0.0, device=self.device)

        # HF loss
        if has_hf and y_hf_true is not None and len(y_hf_true) > 0:
            hf_loss = mse_loss(y_hf_pred, y_hf_true)
        else:
            hf_loss = torch.tensor(0.0, device=self.device)

        # L2 regularization
        l2_reg = torch.tensor(0.0, device=self.device)
        for param in self.model.parameters():
            l2_reg += torch.norm(param, 2) ** 2

        # Weighted loss
        # 한쪽 fidelity만 있는 경우 해당 loss만 사용
        if has_lf and has_hf:
            total_loss = self.alpha * hf_loss + (1 - self.alpha) * lf_loss + self.lambda_reg * l2_reg
        elif has_hf:
            total_loss = hf_loss + self.lambda_reg * l2_reg
        else:
            total_loss = lf_loss + self.lambda_reg * l2_reg

        return total_loss, lf_loss, hf_loss

    def train_joint(self, X_low: np.ndarray, y_low: np.ndarray,
                    X_high: np.ndarray, y_high: np.ndarray,
                    epochs: int = 300, lr: float = 1e-3,
                    batch_size: int = 32, val_ratio: float = 0.1,
                    early_stopping_patience: int = 50,
                    verbose: bool = False) -> Dict:
        """
        LF/HF 데이터를 동시에 사용하여 학습

        논문의 핵심 학습 방식:
        - LF 데이터: y_LF loss 계산에 사용
        - HF 데이터: y_HF loss 계산에 사용 (y_LF는 latent variable 역할)
        - 두 loss를 α 가중치로 결합

        Args:
            X_low: LF 입력 데이터
            y_low: LF 타겟 데이터
            X_high: HF 입력 데이터
            y_high: HF 타겟 데이터
            epochs: 학습 epoch 수
            lr: 학습률
            batch_size: 배치 크기
            val_ratio: 검증 데이터 비율
            early_stopping_patience: Early stopping patience
            verbose: 상세 출력

        Returns:
            학습 기록 딕셔너리
        """
        # 데이터 준비
        X_low = np.asarray(X_low, dtype=np.float32)
        y_low = np.asarray(y_low, dtype=np.float32).reshape(-1, 1)
        X_high = np.asarray(X_high, dtype=np.float32)
        y_high = np.asarray(y_high, dtype=np.float32).reshape(-1, 1)

        has_lf = len(X_low) > 0
        has_hf = len(X_high) > 0

        if not has_lf and not has_hf:
            raise ValueError("At least one of LF or HF data must be provided")

        # 검증 데이터 분할
        if has_lf and len(X_low) >= 3:
            X_low_train, X_low_val, y_low_train, y_low_val = train_test_split(
                X_low, y_low, test_size=val_ratio, random_state=42
            )
        else:
            X_low_train, X_low_val = X_low, X_low
            y_low_train, y_low_val = y_low, y_low

        if has_hf and len(X_high) >= 3:
            X_high_train, X_high_val, y_high_train, y_high_val = train_test_split(
                X_high, y_high, test_size=val_ratio, random_state=42
            )
        else:
            X_high_train, X_high_val = X_high, X_high
            y_high_train, y_high_val = y_high, y_high

        # 텐서 변환
        X_low_train_t = torch.tensor(X_low_train, dtype=torch.float32).to(self.device)
        y_low_train_t = torch.tensor(y_low_train, dtype=torch.float32).to(self.device)
        X_high_train_t = torch.tensor(X_high_train, dtype=torch.float32).to(self.device)
        y_high_train_t = torch.tensor(y_high_train, dtype=torch.float32).to(self.device)

        X_low_val_t = torch.tensor(X_low_val, dtype=torch.float32).to(self.device)
        y_low_val_t = torch.tensor(y_low_val, dtype=torch.float32).to(self.device)
        X_high_val_t = torch.tensor(X_high_val, dtype=torch.float32).to(self.device)
        y_high_val_t = torch.tensor(y_high_val, dtype=torch.float32).to(self.device)

        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0)  # L2 reg in loss

        # 학습 기록 초기화
        self.train_losses = []
        self.lf_losses = []
        self.hf_losses = []
        self.val_losses = []

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        self.model.train()

        for epoch in range(epochs):
            # === Training ===
            optimizer.zero_grad()

            # LF 데이터 forward
            if has_lf:
                y_lf_pred_lf, _ = self.model(X_low_train_t)
            else:
                y_lf_pred_lf = None

            # HF 데이터 forward
            if has_hf:
                y_lf_pred_hf, y_hf_pred = self.model(X_high_train_t)
            else:
                y_hf_pred = None

            # Loss 계산
            # LF loss: LF 데이터의 y_lf_pred vs y_low
            # HF loss: HF 데이터의 y_hf_pred vs y_high
            total_loss, lf_loss, hf_loss = self._compute_loss(
                y_lf_pred=y_lf_pred_lf if has_lf else torch.zeros(1, device=self.device),
                y_hf_pred=y_hf_pred if has_hf else torch.zeros(1, device=self.device),
                y_lf_true=y_low_train_t if has_lf else None,
                y_hf_true=y_high_train_t if has_hf else None,
                has_lf=has_lf,
                has_hf=has_hf
            )

            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            optimizer.step()

            # 기록
            self.train_losses.append(total_loss.item())
            self.lf_losses.append(lf_loss.item() if has_lf else 0.0)
            self.hf_losses.append(hf_loss.item() if has_hf else 0.0)

            # === Validation ===
            self.model.eval()
            with torch.no_grad():
                val_total_loss = torch.tensor(0.0, device=self.device)

                if has_lf and len(X_low_val) > 0:
                    y_lf_pred_val, _ = self.model(X_low_val_t)
                    val_lf_loss = nn.MSELoss()(y_lf_pred_val, y_low_val_t)
                else:
                    val_lf_loss = torch.tensor(0.0, device=self.device)

                if has_hf and len(X_high_val) > 0:
                    _, y_hf_pred_val = self.model(X_high_val_t)
                    val_hf_loss = nn.MSELoss()(y_hf_pred_val, y_high_val_t)
                else:
                    val_hf_loss = torch.tensor(0.0, device=self.device)

                if has_lf and has_hf:
                    val_total_loss = self.alpha * val_hf_loss + (1 - self.alpha) * val_lf_loss
                elif has_hf:
                    val_total_loss = val_hf_loss
                else:
                    val_total_loss = val_lf_loss

                self.val_losses.append(val_total_loss.item())

            self.model.train()

            # Early stopping
            if val_total_loss.item() < best_val_loss:
                best_val_loss = val_total_loss.item()
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch+1}")
                break

            if verbose and (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}/{epochs}: "
                      f"Loss={total_loss.item():.4f} "
                      f"(LF={lf_loss.item():.4f}, HF={hf_loss.item():.4f}), "
                      f"Val={val_total_loss.item():.4f}")

        # 최적 모델 복원
        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        return {
            'train_losses': self.train_losses,
            'lf_losses': self.lf_losses,
            'hf_losses': self.hf_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(self.train_losses)
        }

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        예측 수행

        Args:
            X: 입력 데이터

        Returns:
            y_lf_pred: LF 예측값
            y_hf_pred: HF 예측값
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            y_lf, y_hf = self.model(X_tensor)
            return y_lf.cpu().numpy().flatten(), y_hf.cpu().numpy().flatten()

    def extract_features(self, X: np.ndarray, for_fidelity: str = 'high') -> np.ndarray:
        """
        BLR용 feature 추출

        Args:
            X: 입력 데이터
            for_fidelity: 'low' (shared features) 또는 'high' (HF features)

        Returns:
            추출된 features
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

            if for_fidelity == 'low':
                features = self.model.extract_shared_features(X_tensor)
            else:
                features = self.model.extract_hf_features(X_tensor)

            return features.cpu().numpy()

    def get_hyperparameter_summary(self) -> Dict:
        """하이퍼파라미터 요약 (기존 DNGO와 호환)"""
        return {
            'alpha': self.alpha,
            'shared_width': self.shared_width,
            'num_shared_layers': self.num_shared_layers,
            'hf_hidden_dims': self.hf_hidden_dims,
            'lambda_reg': self.lambda_reg,
            'best_params': self.best_params,
            'pretrain_best_params': None,  # 호환성을 위해
            'finetune_best_params': self.best_params,
            'pretrain_bo_history': [],
            'finetune_bo_history': self.bo_history
        }


class BayesianLinearRegression:
    """
    베이지안 선형 회귀 (기존 코드와 동일)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 25.0):
        self.alpha = alpha
        self.beta = beta
        self.mean = None
        self.cov = None
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).flatten()

        X_with_bias = np.column_stack([np.ones(len(X)), X])
        S0_inv = self.alpha * np.eye(X_with_bias.shape[1])
        S_N_inv = S0_inv + self.beta * X_with_bias.T @ X_with_bias
        self.cov = np.linalg.inv(S_N_inv)
        self.mean = self.beta * self.cov @ X_with_bias.T @ y
        self.fitted = True

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        if not self.fitted:
            raise ValueError("Model not fitted")

        x = np.asarray(x, dtype=np.float32).flatten()
        x_with_bias = np.concatenate([[1], x])
        mu = x_with_bias @ self.mean
        var = (1/self.beta) + x_with_bias @ self.cov @ x_with_bias
        return mu, var

    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float32)
        means, variances = [], []
        for x in X:
            mu, var = self.predict(x)
            means.append(mu)
            variances.append(var)
        return np.array(means), np.array(variances)
