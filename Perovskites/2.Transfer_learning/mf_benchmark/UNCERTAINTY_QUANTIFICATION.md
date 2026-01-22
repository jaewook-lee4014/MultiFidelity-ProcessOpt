# Uncertainty Quantification Methods (Previous Implementation)

이 문서는 이전 코드(`mf_uq_models.py`)에서 구현했던 **6가지 불확실성 추정(UQ) 방법**을 설명합니다.
현재 `benchmark_parallel.py`에서는 이 방법들이 제외되어 있고, `std=0.1` 고정값을 사용합니다.

---

## 개요

이전 구현에서는 6가지 UQ 방법 × 2가지 Transfer Learning 방식 = 12개 모델 조합을 사용했습니다.

### UQ Methods
1. **GP** - Gaussian Process (posterior variance)
2. **DNGO** - Deep Networks + Bayesian Linear Regression
3. **BNN** - Bayesian Neural Network (variational inference)
4. **MC-Dropout** - Monte Carlo Dropout
5. **Deep Ensemble** - Ensemble of networks
6. **SNGP** - Spectral-Normalized Gaussian Process

### Transfer Learning Approaches
1. **MFGP** - Multi-Fidelity with fidelity encoding
2. **TL** - Pretrain on LF, Fine-tune on HF

---

## 1. Bayesian Linear Regression (BLR)

가장 핵심적인 UQ 방법. DNN feature extractor + BLR output layer 조합.

### 수학적 배경

**Prior**:
```
p(w) = N(w | 0, α⁻¹I)
```

**Likelihood**:
```
p(y | X, w) = N(y | Φw, β⁻¹I)
```
- Φ: feature matrix (DNN output)
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

### 구현 (mf_uq_models.py:179-216)

```python
class DNGO_MFGP(BaseMFModel):
    def fit(self, X_lf, y_lf, X_hf, y_hf):
        # 1. DNN feature extractor 학습
        self.network = self._build_network()
        for _ in range(self.epochs):
            features = self.network(X_t)
            pred = features.mean(dim=1)
            loss = MSE(pred, y_t)
            loss.backward()

        # 2. BLR on HF features
        with torch.no_grad():
            Phi = self.network(X_hf_t).cpu().numpy()

        # BLR parameters
        alpha, beta = 1.0, 25.0
        A = alpha * np.eye(Phi.shape[1]) + beta * Phi.T @ Phi
        self.A_inv = np.linalg.inv(A + 1e-6 * np.eye(A.shape[0]))
        self.m = beta * self.A_inv @ Phi.T @ y_hf_scaled
        self.beta = beta

    def predict(self, X):
        Phi = self.network(X_t).cpu().numpy()

        # Mean
        mean = Phi @ self.m

        # Variance (predictive uncertainty)
        var = 1/self.beta + np.sum(Phi @ self.A_inv * Phi, axis=1)
        std = np.sqrt(var)

        return mean, std
```

### BLR 하이퍼파라미터

| Parameter | Value | 설명 |
|-----------|-------|------|
| `alpha` | 1.0 | Prior precision (weight regularization) |
| `beta` | 25.0 | Noise precision (1/noise_variance) |
| `feature_dim` | 50 | BLR input feature dimension |
| `l2_lambda` | 1e-3 | DNN weight decay (추가 regularization) |

### BLR 장점/단점

**장점**:
- Closed-form posterior → 빠른 계산
- 이론적으로 잘 정립된 uncertainty
- DNN feature extraction + Bayesian inference 결합

**단점**:
- Linear assumption on features
- α, β 하이퍼파라미터 민감
- Feature dimension이 크면 matrix inversion 비용

---

## 2. MC-Dropout

Dropout을 inference 시에도 활성화하여 Monte Carlo sampling.

### 수학적 배경

```
p(y* | x*, D) ≈ (1/T) Σ f(x*; W_t)
```
- T: Monte Carlo samples
- W_t: Dropout이 적용된 t번째 weight sample

**Predictive mean & variance**:
```
μ* = (1/T) Σ f_t(x*)
σ*² = (1/T) Σ (f_t(x*) - μ*)²
```

### 구현 (mf_uq_models.py:889-968)

```python
class MCDropout_MFGP(BaseMFModel):
    def __init__(self, ..., dropout=0.15, n_samples=50):
        self.dropout = dropout
        self.n_samples = n_samples

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),  # Dropout layer
            nn.Linear(hidden_dim, 1)
        )

    def predict(self, X):
        self.network.train()  # Keep dropout ACTIVE

        preds = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.network(X_t)
                preds.append(pred)

        preds = np.stack(preds)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)  # Epistemic uncertainty

        return mean, std
```

### MC-Dropout 하이퍼파라미터

| Parameter | Value | 설명 |
|-----------|-------|------|
| `dropout` | 0.15 | Dropout probability |
| `n_samples` | 50 | Monte Carlo samples for prediction |
| `hidden_dim` | 256 | Larger network for dropout |
| `activation` | ReLU | (Tanh보다 Dropout과 잘 작동) |

### MC-Dropout 장점/단점

**장점**:
- 구현 간단 (Dropout만 추가)
- 기존 DNN에 쉽게 적용
- Scalable

**단점**:
- Dropout rate 선택이 어려움
- 많은 forward pass 필요 (느림)
- Approximation quality가 dropout rate에 의존

---

## 3. Bayesian Neural Network (BNN)

Weight에 distribution을 가정하고 variational inference로 학습.

### 수학적 배경

**Weight prior**:
```
p(W) = N(W | 0, I)
```

**Variational posterior**:
```
q(W) = N(W | μ, σ²)
```

**ELBO Loss**:
```
L = E_q[log p(y|X,W)] - KL(q(W) || p(W))
```

### 구현 (mf_uq_models.py:651-764)

```python
class BNN_MFGP(BaseMFModel):
    def __init__(self, ..., kl_weight=0.2, n_samples=20):
        self.kl_weight = kl_weight
        self.n_samples = n_samples

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        # Variational parameters
        self.mu = nn.ParameterList()      # Mean
        self.log_sigma = nn.ParameterList()  # Log std

        for layer in layers:
            self.mu.append(nn.Parameter(torch.randn(...) * 0.1))
            self.log_sigma.append(nn.Parameter(torch.ones(...) * -3))

        for epoch in range(epochs):
            kl = 0
            preds = []

            # Sample weights
            for _ in range(5):
                W = []
                for mu, log_sigma in zip(self.mu, self.log_sigma):
                    sigma = torch.exp(log_sigma)
                    eps = torch.randn_like(mu)  # Reparameterization
                    w = mu + sigma * eps
                    W.append(w)
                    kl += 0.5 * sum(sigma² + mu² - 2*log_sigma - 1)

                # Forward pass
                pred = forward_with_weights(X, W)
                preds.append(pred)

            # ELBO loss
            nll = MSE(mean(preds), y)
            loss = nll + kl_weight * kl / n_data

    def predict(self, X):
        preds = []
        for _ in range(self.n_samples):
            W = sample_weights()
            pred = forward_with_weights(X, W)
            preds.append(pred)

        mean = preds.mean()
        std = preds.std()
        return mean, std
```

### BNN 하이퍼파라미터

| Parameter | Value | 설명 |
|-----------|-------|------|
| `kl_weight` | 0.2 | KL divergence weight (β in β-VAE) |
| `n_samples` | 20 | Prediction samples |
| `init_log_sigma` | -3 | Initial log std (exp(-3) ≈ 0.05) |
| `num_layers` | 2 | Number of Bayesian layers |

---

## 4. Deep Ensemble

여러 독립적인 네트워크를 학습하고 예측 결합.

### 수학적 배경

각 ensemble member가 mean과 variance를 출력:
```
f_m(x) = (μ_m(x), σ_m²(x))
```

**Ensemble predictive**:
```
μ*(x) = (1/M) Σ μ_m(x)
σ*²(x) = (1/M) Σ [σ_m²(x) + μ_m²(x)] - μ*²(x)
       = E[σ²] + Var[μ]  (aleatoric + epistemic)
```

### 구현 (mf_uq_models.py:1062-1150)

```python
class DeepEnsemble_MFGP(BaseMFModel):
    def __init__(self, ..., n_ensemble=3):
        self.n_ensemble = n_ensemble

    def _build_member(self):
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # mean + log_var
        )

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        self.networks = []
        for i in range(self.n_ensemble):
            torch.manual_seed(i * 1000)  # Different init
            net = self._build_member()

            for epoch in range(epochs):
                out = net(X_t)
                mean = out[:, 0]
                log_var = out[:, 1]
                var = softplus(log_var) + 1e-6

                # Negative log-likelihood with learned variance
                loss = 0.5 * mean(log(var) + (y - mean)² / var)

            self.networks.append(net)

    def predict(self, X):
        means, vars = [], []
        for net in self.networks:
            out = net(X)
            mean = out[:, 0]
            var = exp(out[:, 1]) + 1e-6
            means.append(mean)
            vars.append(var)

        # Combine predictions
        ensemble_mean = mean(means)
        ensemble_var = mean(vars) + var(means)  # aleatoric + epistemic

        return ensemble_mean, sqrt(ensemble_var)
```

### Deep Ensemble 하이퍼파라미터

| Parameter | Value | 설명 |
|-----------|-------|------|
| `n_ensemble` | 3 | Number of ensemble members |
| `output_dim` | 2 | mean + log_variance |
| `loss` | NLL | Negative log-likelihood with heteroscedastic variance |

---

## 5. SNGP (Spectral-Normalized Gaussian Process)

Spectral normalization + Random Fourier Features로 GP approximation.

### 수학적 배경

**Spectral Normalization**: Weight matrix의 spectral norm을 1로 제한
```
W_SN = W / ||W||_2
```
→ Lipschitz continuity 보장 → distance-aware uncertainty

**Random Fourier Features (RFF)**:
```
φ(x) = √(2/D) cos(Wx + b)
```
→ RBF kernel approximation

### 구현 (mf_uq_models.py:1256-1342)

```python
class SNGP_MFGP(BaseMFModel):
    def __init__(self, ..., num_inducing=512):
        self.num_inducing = num_inducing

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        # Backbone with spectral normalization
        self.backbone = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(in_dim, hidden_dim)),
            nn.ReLU()
        )

        # Random Fourier Features
        self.W = torch.randn(hidden_dim, num_inducing)
        self.b = torch.rand(num_inducing) * 2 * pi
        self.beta = nn.Parameter(torch.zeros(num_inducing, 1))

        for epoch in range(epochs):
            h = self.backbone(X_t)
            rff = cos(h @ W + b) * sqrt(2/num_inducing)
            mean = rff @ self.beta
            loss = MSE(mean, y)

        # Compute precision for GP uncertainty
        Phi = rff.numpy()
        self.precision = 0.1 * I + Φᵀ Φ
        self.cov = inv(self.precision)

    def predict(self, X):
        h = self.backbone(X_t)
        rff = cos(h @ W + b) * sqrt(2/num_inducing)

        mean = rff @ self.beta
        var = sum(rff @ self.cov * rff, axis=1)  # GP variance

        return mean, sqrt(var)
```

### SNGP 하이퍼파라미터

| Parameter | Value | 설명 |
|-----------|-------|------|
| `num_inducing` | 512 | RFF dimension |
| `hidden_dim` | 64 | Backbone hidden size |
| `spectral_norm` | True | Apply to all linear layers |

---

## 현재 코드와의 비교

### 이전 코드 (mf_uq_models.py)
```python
def predict(self, X):
    # BLR uncertainty
    mean = Phi @ self.m
    var = 1/self.beta + np.sum(Phi @ self.A_inv * Phi, axis=1)
    std = np.sqrt(var)
    return mean, std
```

### 현재 코드 (benchmark_parallel.py)
```python
def predict(self, X):
    mean = ...  # 실제 계산
    return mean, np.ones_like(mean) * 0.1  # 고정값
```

---

## UQ 방법 추가 시 고려사항

### 1. BLR 추가하려면
```python
# fit() 마지막에 추가
Phi = self.hf_net.extract_features(X_hf_t)
alpha, beta = 1.0, 25.0
A = alpha * I + beta * Φᵀ Φ
self.A_inv = inv(A)
self.m = beta * A_inv @ Φᵀ @ y_hf

# predict()에서
var = 1/beta + sum(Phi @ A_inv * Phi, axis=1)
std = sqrt(var)
```

### 2. MC-Dropout 추가하려면
```python
# 네트워크에 Dropout 추가
nn.Dropout(0.15)

# predict()에서
self.network.train()  # dropout 활성화
preds = [self.network(X) for _ in range(50)]
std = np.std(preds, axis=0)
```

### 3. Acquisition 수정
```python
# use_ei = True for all models (not just MFGP)
use_ei = True  # proper uncertainty 있으면

# Or Thompson Sampling
sampled_y = mean + std * np.random.randn(len(mean))
next_idx = np.argmin(sampled_y)
```

---

## 참고 자료

- **DNGO**: Snoek et al., "Scalable Bayesian Optimization Using Deep Neural Networks"
- **MC-Dropout**: Gal & Ghahramani, "Dropout as a Bayesian Approximation"
- **Deep Ensemble**: Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty"
- **SNGP**: Liu et al., "Simple and Principled Uncertainty Estimation"
