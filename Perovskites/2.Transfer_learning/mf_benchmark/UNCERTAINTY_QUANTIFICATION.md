# Uncertainty Quantification Methods

이 문서는 `benchmark_parallel.py`의 12개 모델에 이전에 적용했던 **불확실성 추정(UQ) 방법**을 설명합니다.
현재 `benchmark_parallel.py`에서는 UQ가 제외되어 있고, `std=0.1` 고정값을 사용합니다.

---

## 개요

### 이전 구현 (`compare_lf_blr.py`)

| 모델 | LF 예측 | HF 예측 | Acquisition |
|------|---------|---------|-------------|
| MFGP | GP posterior | GP posterior | EI (GP uncertainty) |
| Sequential-Vanilla | argmin(mean) | argmin(mean) | Exploitation only |
| **Sequential-LF-BLR** | **BLR uncertainty** | argmin(mean) | **LF: EI, HF: argmin** |
| Joint-Vanilla | argmin(mean) | argmin(mean) | Exploitation only |

### 현재 구현 (`benchmark_parallel.py`)

모든 DNN 모델: `std=0.1` 고정, `argmin(mean)` 사용

---

## LF-BLR 방식 (Sequential_LF_BLR)

**핵심 아이디어**: LF 평가 시에만 BLR로 불확실성 계산 → EI로 탐색, HF 평가 시에는 exploitation만

### 수학적 배경

**Prior**:
```
p(w) = N(w | 0, α⁻¹I)
```

**Likelihood**:
```
p(y | X, w) = N(y | Φw, β⁻¹I)
```
- Φ: feature matrix (LF DNN output)
- α: prior precision (weight regularization)
- β: noise precision (data fit)

**Posterior**:
```
p(w | X, y) = N(w | m, S)
S = (αI + βΦᵀΦ)⁻¹
m = βS Φᵀy
```

**Predictive Distribution**:
```
p(y* | x*, X, y) = N(y* | μ*, σ*²)
μ* = φ(x*)ᵀm
σ*² = 1/β + φ(x*)ᵀS φ(x*)
```

### 구현 (`compare_lf_blr.py:325-518`)

```python
class Sequential_LF_BLR:
    """
    Sequential Training with BLR ONLY on LF

    - LF network: x -> features -> BLR -> (mean, std) for LF
    - HF network: (x, y_lf) -> delta (no BLR)

    Selection Strategy:
    - LF evaluation: Use LF BLR uncertainty → EI → argmax(EI)
    - HF evaluation: Use HF mean → argmin(mean)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 feature_dim: int = 50,  # BLR feature dimension
                 lf_lr: float = 1e-3, hf_lr: float = 1e-3,
                 lf_epochs: int = 300, hf_epochs: int = 200,
                 num_layers: int = 2, l2_lambda: float = 1e-3,
                 alpha_blr: float = 1.0, beta_blr: float = 25.0):
        self.alpha_blr = alpha_blr
        self.beta_blr = beta_blr
        self.feature_dim = feature_dim
        # ...

    def _build_lf_feature_network(self):
        """LF network outputs features for BLR"""
        layers = []
        in_dim = self.input_dim
        for _ in range(self.num_layers):
            layers.extend([
                nn.Linear(in_dim, self.hidden_dim),
                nn.Tanh()
            ])
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, self.feature_dim))  # Feature output
        return nn.Sequential(*layers).to(device)

    def _build_hf_network(self):
        """HF network: (x, y_lf) -> delta (direct output, no BLR)"""
        layers = []
        in_dim = self.input_dim + 1  # x + y_lf
        for _ in range(self.num_layers):
            layers.extend([
                nn.Linear(in_dim, self.hidden_dim),
                nn.Tanh()
            ])
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, 1))  # Direct output
        return nn.Sequential(*layers).to(device)

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        # ============ Stage 1: Train LF feature network ============
        self.lf_feature_network = self._build_lf_feature_network()
        lf_optimizer = torch.optim.Adam(
            self.lf_feature_network.parameters(),
            lr=self.lf_lr, weight_decay=self.l2_lambda
        )

        for epoch in range(self.lf_epochs):
            lf_optimizer.zero_grad()
            features = self.lf_feature_network(X_lf_t)
            pred = features.mean(dim=1, keepdim=True)  # Simple mean of features
            loss = F.mse_loss(pred, y_lf_t)
            loss.backward()
            lf_optimizer.step()

        # ============ Fit BLR on learned LF features ============
        self.lf_feature_network.eval()
        with torch.no_grad():
            Phi_lf = self.lf_feature_network(X_lf_t).cpu().numpy()

        # BLR closed form
        A = self.alpha_blr * np.eye(self.feature_dim) + self.beta_blr * Phi_lf.T @ Phi_lf
        self.lf_A_inv = np.linalg.inv(A)
        self.lf_m = self.beta_blr * self.lf_A_inv @ Phi_lf.T @ y_lf_scaled

        # ============ Stage 2: Train HF network (no BLR) ============
        self.hf_network = self._build_hf_network()
        # ... standard MSE training ...

    def predict_lf(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict LF with BLR uncertainty
        Returns: (mean, std) in original scale
        """
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(device)

        self.lf_feature_network.eval()
        with torch.no_grad():
            Phi = self.lf_feature_network(X_t).cpu().numpy()

        # BLR prediction
        mean_scaled = Phi @ self.lf_m

        # BLR variance: σ² = 1/β + φᵀ S φ
        var_scaled = 1.0 / self.beta_blr + np.sum((Phi @ self.lf_A_inv) * Phi, axis=1)
        std_scaled = np.sqrt(np.maximum(var_scaled, 1e-10))

        # Inverse transform
        mean = self.scaler_y.inverse_transform(mean_scaled.reshape(-1, 1)).flatten()
        std = std_scaled * self.scaler_y.scale_[0]

        return mean, np.maximum(std, 1e-6)

    def predict_hf(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict HF (no BLR, just mean)
        Returns: (mean, dummy_std)
        """
        # ... HF network forward pass ...
        mean = self.scaler_y.inverse_transform(mean_scaled.reshape(-1, 1)).flatten()
        std = np.ones_like(mean) * 0.1  # Dummy for HF (no BLR)

        return mean, std
```

### LF-BLR 하이퍼파라미터

| Parameter | Value | 설명 |
|-----------|-------|------|
| `alpha_blr` | 1.0 | Prior precision (weight regularization) |
| `beta_blr` | 25.0 | Noise precision (1/noise_variance) |
| `feature_dim` | 50 | LF BLR feature dimension |
| `hidden_dim` | 64 | Network hidden size |
| `num_layers` | 2 | Number of hidden layers |
| `l2_lambda` | 1e-3 | Weight decay |
| `lf_epochs` | 300 | LF training epochs |
| `hf_epochs` | 200 | HF training epochs |

### Acquisition Strategy

```python
def run_bo_iteration(model, X_grid, y_best, eval_hf: bool):
    if eval_hf:
        # HF evaluation: exploitation only
        mean, _ = model.predict_hf(X_grid)
        next_idx = np.argmin(mean)
    else:
        # LF evaluation: exploration with EI
        mean, std = model.predict_lf(X_grid)
        ei = expected_improvement(mean, std, y_best)
        next_idx = np.argmax(ei)

    return next_idx
```

### 장점/단점

**장점**:
- LF 평가 시 탐색 가능 (EI 기반)
- HF 평가 시 exploitation 집중 (비용 효율)
- Closed-form posterior → 빠른 계산
- 구현 간단

**단점**:
- HF 예측에 불확실성 없음
- LF BLR이 HF 불확실성을 반영하지 못함
- α, β 하이퍼파라미터 민감

---

## 현재 코드와의 비교

### 이전 코드 (`compare_lf_blr.py`)
```python
def predict_lf(self, X):
    Phi = self.lf_feature_network(X_t).cpu().numpy()

    # BLR prediction with uncertainty
    mean_scaled = Phi @ self.lf_m
    var_scaled = 1.0 / self.beta_blr + np.sum((Phi @ self.lf_A_inv) * Phi, axis=1)
    std_scaled = np.sqrt(var_scaled)

    return mean, std  # Real uncertainty for LF

def predict_hf(self, X):
    # ... HF network ...
    return mean, np.ones_like(mean) * 0.1  # Dummy for HF
```

### 현재 코드 (`benchmark_parallel.py`)
```python
def predict(self, X):
    # ... network forward pass ...
    mean = ...
    return mean, np.ones_like(mean) * 0.1  # All models: dummy uncertainty
```

---

## benchmark_parallel.py에 LF-BLR 적용 방법

### 1. Sequential 모델에 LF-BLR 추가

```python
class Sequential(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, feature_dim=50,
                 alpha_blr=1.0, beta_blr=25.0, ...):
        self.feature_dim = feature_dim
        self.alpha_blr = alpha_blr
        self.beta_blr = beta_blr

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        # ... existing LF training ...

        # Add BLR on LF features
        self.lf_net.eval()
        with torch.no_grad():
            # Extract features from last hidden layer
            Phi_lf = self._extract_features(X_lf_t).cpu().numpy()

        # BLR closed form
        A = self.alpha_blr * np.eye(self.feature_dim) + self.beta_blr * Phi_lf.T @ Phi_lf
        self.lf_A_inv = np.linalg.inv(A + 1e-6 * np.eye(A.shape[0]))
        self.lf_m = self.beta_blr * self.lf_A_inv @ Phi_lf.T @ y_lf_s

        # ... existing HF training ...

    def predict_lf(self, X):
        """LF prediction with BLR uncertainty"""
        Phi = self._extract_features(X_t).cpu().numpy()

        mean_s = Phi @ self.lf_m
        var_s = 1.0 / self.beta_blr + np.sum((Phi @ self.lf_A_inv) * Phi, axis=1)
        std_s = np.sqrt(np.maximum(var_s, 1e-10))

        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        std = std_s * self.scaler_y.scale_[0]

        return mean, np.maximum(std, 1e-6)

    def predict(self, X):
        """HF prediction (no uncertainty)"""
        # ... existing code ...
        return mean, np.ones_like(mean) * 0.1
```

### 2. Acquisition 수정

```python
def select_next_point(model, X_grid, y_best, eval_hf, sampled_indices):
    mask = np.ones(len(X_grid), dtype=bool)
    mask[sampled_indices] = False
    X_candidates = X_grid[mask]

    if eval_hf:
        # HF: exploitation
        mean, _ = model.predict(X_candidates)
        local_idx = np.argmin(mean)
    else:
        # LF: exploration with EI
        mean, std = model.predict_lf(X_candidates)
        ei = expected_improvement(mean, std, y_best)
        local_idx = np.argmax(ei)

    return np.where(mask)[0][local_idx]
```

---

## 요약

| 방식 | 적용 위치 | UQ 방법 | Acquisition |
|------|-----------|---------|-------------|
| 현재 | 모든 모델 | std=0.1 고정 | argmin(mean) |
| LF-BLR | LF만 | BLR (α=1, β=25) | LF: EI, HF: argmin |
| Full BLR | LF+HF | BLR 양쪽 | EI 사용 |

### 권장 사항

1. **공정한 비교를 위해**: 현재처럼 모든 모델에 동일한 std=0.1 사용
2. **성능 개선을 위해**: LF-BLR 방식 적용 고려
3. **최대 성능을 위해**: Full BLR 또는 다른 UQ 방법 (MC-Dropout, Ensemble) 적용

---

## 참고 자료

- **DNGO**: Snoek et al., "Scalable Bayesian Optimization Using Deep Neural Networks"
- **BLR**: Bishop, "Pattern Recognition and Machine Learning", Chapter 3
- **compare_lf_blr.py**: 현재 레포지토리의 이전 구현
