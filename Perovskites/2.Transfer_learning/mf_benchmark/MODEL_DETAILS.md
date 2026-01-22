# Model Implementation Details

12개 Multi-Fidelity Transfer Learning 모델의 상세 구현 설명.

---

## Network Architecture

모든 DNN 기반 모델은 두 가지 기본 네트워크를 사용:

### LFNetwork (Low-Fidelity Network)
```
Input(dim) → Linear(hidden_dim) → Tanh → Linear(hidden_dim) → Tanh → Linear(1) → Output
```
- Feature extractor: 2-layer MLP with Tanh activation
- Output layer: Linear regression head

### HFNetwork (High-Fidelity Network)
```
[Input(dim), y_lf] → Linear(hidden_dim) → Tanh → Linear(hidden_dim) → Tanh → Linear(1) → δ
Output = y_lf + δ (residual learning)
```
- Input: 원본 features + LF 예측값 concatenate
- Residual connection: LF 예측에 delta를 더해 HF 예측

### AdapterLayer (Adapter 모델용)
```
Input → Linear(bottleneck_dim) → ReLU → Linear(input_dim) → Output
Output = Input + Adapter(Input)  (residual)
```

---

## 1. MFGP (Multi-Fidelity Gaussian Process)

**카테고리**: GP-based (Baseline)

**구현**: BoTorch `SingleTaskMultiFidelityGP`

**학습 방식**:
```python
# Fidelity indicator 추가 (LF=0, HF=1)
X_lf_fid = [X_lf, 0]  # Low-fidelity
X_hf_fid = [X_hf, 1]  # High-fidelity

# 모든 데이터 합쳐서 GP fitting
model = SingleTaskMultiFidelityGP(X_all, y_all, data_fidelities=[input_dim])
fit_gpytorch_mll(mll)
```

**예측**:
- GP posterior에서 mean과 variance 추출
- **Proper uncertainty quantification** 제공 (유일하게)

**특징**:
- Acquisition: Expected Improvement (EI) 사용
- 다른 모델과 달리 실제 불확실성 제공
- 학습 시간이 데이터 증가에 따라 O(n³)

---

## 2. Sequential

**카테고리**: Sequential Transfer

**학습 방식**:
```
Stage 1: LF 데이터로 LFNetwork 학습 (200 epochs)
         → LFNetwork 가중치 freeze
Stage 2: HF 데이터로 HFNetwork 학습 (100 epochs)
         → LFNetwork 출력을 입력으로 사용
```

**핵심 코드**:
```python
# Stage 1: LF 학습
for _ in range(lf_epochs):
    loss = MSE(lf_net(X_lf), y_lf)

# Stage 2: HF 학습 (LF frozen)
for p in lf_net.parameters():
    p.requires_grad = False

for _ in range(hf_epochs):
    y_lf_pred = lf_net(X_hf)  # no grad
    loss = MSE(hf_net(X_hf, y_lf_pred), y_hf)
```

**특징**:
- 가장 기본적인 transfer learning 방식
- LF → HF 단방향 knowledge transfer
- LF 네트워크 완전 고정

---

## 3. Progressive

**카테고리**: Sequential Transfer

**학습 방식**:
```
Stage 1: LF 학습 (100 epochs)
Stage 2: HF output layer만 학습 (50 epochs) - feature net frozen
Stage 3: HF 마지막 2개 layer unfreeze + 학습 (50 epochs, lr=1e-4)
```

**핵심 코드**:
```python
# Stage 2: Output layer만
for p in hf_net.feature_net.parameters():
    p.requires_grad = False
# hf_net.out_layer만 학습

# Stage 3: 점진적 unfreeze
for p in list(hf_net.feature_net.parameters())[-2:]:
    p.requires_grad = True
# 낮은 learning rate로 fine-tuning
```

**특징**:
- Gradual unfreezing으로 catastrophic forgetting 방지
- 깊은 layer부터 점진적으로 adaptation
- Fine-tuning에 낮은 lr 사용 (1e-4)

---

## 4. Curriculum

**카테고리**: Sequential Transfer

**학습 방식**:
```
Stage 1: LF 학습 (200 epochs)
Stage 2: HF 학습 with curriculum
         - LF-HF residual이 작은 샘플부터 시작
         - epoch 진행에 따라 점점 어려운 샘플 추가
```

**핵심 코드**:
```python
# Residual 기준 정렬
residuals = |y_hf - lf_net(X_hf)|
sorted_idx = argsort(residuals)  # 쉬운 것부터

for epoch in range(epochs):
    # 점진적으로 샘플 추가
    n_use = int((epoch + 1) / epochs * n_hf)
    idx = sorted_idx[:n_use]
    loss = MSE(hf_net(X_hf[idx]), y_hf[idx])
```

**특징**:
- Curriculum learning: easy → hard
- LF와 유사한 샘플부터 학습 (residual 작은 것)
- 점진적으로 전체 데이터로 확장

---

## 5. Two-Stage Joint

**카테고리**: Joint Training

**학습 방식**:
```
Stage 1: LF만 학습 (100 epochs)
Stage 2: LF + HF 동시 학습 (100 epochs)
         Loss = 0.3 * LF_loss + 0.7 * HF_loss
```

**핵심 코드**:
```python
# Stage 2: Joint training
opt = Adam([lf_params, hf_params], lr=1e-4)
for _ in range(stage2_epochs):
    lf_loss = MSE(lf_net(X_lf), y_lf)
    hf_loss = MSE(hf_net(X_hf, lf_net(X_hf)), y_hf)
    loss = 0.3 * lf_loss + 0.7 * hf_loss
```

**특징**:
- Two-stage: pretrain → joint
- HF에 더 높은 가중치 (0.7)
- LF와 HF 네트워크가 서로 영향

---

## 6. DNGO-Joint

**카테고리**: DNGO Variants

**학습 방식**:
```
LF + HF 동시 학습 (300 epochs)
Loss = 0.5 * LF_loss + 0.5 * HF_loss
LF gradient는 HF로 전파 안됨 (detach)
```

**핵심 코드**:
```python
for _ in range(epochs):
    lf_loss = MSE(lf_net(X_lf), y_lf)

    with torch.no_grad():
        y_lf_pred = lf_net(X_hf)  # detach
    hf_loss = MSE(hf_net(X_hf, y_lf_pred), y_hf)

    loss = 0.5 * lf_loss + 0.5 * hf_loss
```

**특징**:
- Joint training이지만 LF→HF gradient 차단
- LF와 HF가 독립적으로 최적화
- alpha=0.5로 균등 가중치

---

## 7. DNGO-Gradient

**카테고리**: DNGO Variants

**학습 방식**:
```
LF + HF 동시 학습 (300 epochs)
LF → HF gradient 전파 허용
서로 다른 learning rate 사용
```

**핵심 코드**:
```python
opt = Adam([
    {'params': lf_net.parameters(), 'lr': 1e-3},
    {'params': hf_net.parameters(), 'lr': 5e-4}  # 낮은 lr
])

for _ in range(epochs):
    lf_loss = MSE(lf_net(X_lf), y_lf)
    y_lf_pred = lf_net(X_hf)  # gradient 전파됨
    hf_loss = MSE(hf_net(X_hf, y_lf_pred), y_hf)
    loss = lf_loss + hf_loss
```

**특징**:
- End-to-end gradient flow
- HF loss가 LF 네트워크도 업데이트
- HF에 낮은 lr로 안정적 학습

---

## 8. Knowledge Distillation

**카테고리**: Transfer Learning

**학습 방식**:
```
Stage 1: LF (teacher) 학습
Stage 2: HF (student) 학습
         Loss = (1-α) * hard_loss + α * soft_loss
         - hard_loss: MSE(pred, y_hf)
         - soft_loss: MSE(pred/T, teacher_pred/T) * T²
```

**핵심 코드**:
```python
teacher_pred = lf_net(X_hf)  # teacher output
student_pred = hf_net(X_hf, teacher_pred)

hard_loss = MSE(student_pred, y_hf)
soft_loss = MSE(student_pred/T, teacher_pred/T) * T²

loss = (1 - alpha) * hard_loss + alpha * soft_loss
```

**파라미터**:
- `alpha_kd = 0.3`: distillation 가중치
- `temperature = 3.0`: soft target temperature

**특징**:
- LF를 teacher로, HF를 student로 사용
- Soft targets으로 knowledge transfer
- Temperature scaling으로 distribution smoothing

---

## 9. Domain Adaptation (MMD)

**카테고리**: Transfer Learning

**학습 방식**:
```
Stage 1: LF 학습
Stage 2: HF 학습 + MMD loss
         Loss = task_loss + λ * MMD(LF_features, HF_features)
```

**MMD (Maximum Mean Discrepancy)**:
```python
def mmd_loss(source, target):
    # RBF kernel
    k_ss = rbf_kernel(source, source)
    k_tt = rbf_kernel(target, target)
    k_st = rbf_kernel(source, target)
    return k_ss.mean() + k_tt.mean() - 2 * k_st.mean()
```

**핵심 코드**:
```python
lf_features = lf_net.extract_features(X_lf)
hf_features = hf_net.extract_features(X_hf, y_lf_pred)

task_loss = MSE(hf_pred, y_hf)
mmd = mmd_loss(lf_features, hf_features)

loss = task_loss + lambda_mmd * mmd
```

**파라미터**:
- `lambda_mmd = 0.1`: MMD 가중치

**특징**:
- Feature space에서 LF-HF distribution alignment
- Domain adaptation 기법 적용
- RBF kernel 기반 MMD

---

## 10. Soft Parameter Sharing

**카테고리**: Transfer Learning

**학습 방식**:
```
LF + HF 동시 학습
Loss = 0.5*LF_loss + 0.5*HF_loss + λ*||W_lf - W_hf||²
```

**핵심 코드**:
```python
lf_loss = MSE(lf_net(X_lf), y_lf)
hf_loss = MSE(hf_net(X_hf, y_lf_pred), y_hf)

# Parameter difference regularization
lf_w = lf_net.feature_net[0].weight
hf_w = hf_net.feature_net[0].weight
param_diff = sum((lf_w - hf_w) ** 2)

loss = 0.5*lf_loss + 0.5*hf_loss + lambda_soft*param_diff
```

**파라미터**:
- `lambda_soft = 0.01`: regularization 강도

**특징**:
- LF와 HF 네트워크 파라미터를 비슷하게 유지
- L2 regularization으로 soft sharing
- Hard sharing과 달리 유연한 adaptation 허용

---

## 11. Pseudo-Labeling

**카테고리**: Transfer Learning

**학습 방식**:
```
Stage 1: LF 학습
Stage 2: LF로 pseudo-label 생성
         pseudo_label = lf_pred + offset
         (offset = mean(y_hf - lf_pred_on_hf))
Stage 3: HF 학습
         Loss = real_loss + weight * pseudo_loss
```

**핵심 코드**:
```python
# Pseudo-label 생성
lf_pred_on_hf = lf_net(X_hf)
offset = (y_hf - lf_pred_on_hf).mean()
pseudo_labels = lf_net(X_lf) + offset

# HF 학습
real_loss = MSE(hf_net(X_hf), y_hf)
pseudo_loss = MSE(hf_net(X_lf), pseudo_labels)
loss = real_loss + pseudo_weight * pseudo_loss
```

**파라미터**:
- `pseudo_weight = 0.5`: pseudo-label 가중치

**특징**:
- LF 데이터를 pseudo HF 데이터로 활용
- Offset correction으로 bias 보정
- Data augmentation 효과

---

## 12. Adapter

**카테고리**: Transfer Learning

**학습 방식**:
```
Stage 1: Backbone + LF output layer 학습
Stage 2: Backbone freeze, Adapter + HF output만 학습
```

**Architecture**:
```
Input → [Backbone Layer 1] → Tanh → [Adapter 1] →
        [Backbone Layer 2] → Tanh → [Adapter 2] → [HF Output]
```

**핵심 코드**:
```python
# Stage 1: LF 학습
h = backbone(X_lf)
loss = MSE(out_layer(h), y_lf)

# Stage 2: Adapter만 학습 (backbone frozen)
for p in backbone.parameters():
    p.requires_grad = False

for _ in range(adapter_epochs):
    h = X_hf
    for i, module in enumerate(backbone):
        h = module(h)
        if isinstance(module, Tanh):
            h = adapters[adapter_idx](h)  # adapter 적용
    loss = MSE(hf_out(h), y_hf)
```

**파라미터**:
- `bottleneck_dim = 16`: adapter bottleneck 크기

**특징**:
- Parameter-efficient fine-tuning
- Backbone 고정, adapter만 학습
- Residual connection으로 원본 정보 보존

---

## 공통 하이퍼파라미터

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_dim` | 64 | Hidden layer size |
| `num_layers` | 2 | Number of hidden layers |
| `learning_rate` | 1e-3 (LF), 1e-4 (HF) | Adam optimizer |
| `weight_decay` | 1e-4 | L2 regularization |
| `lf_epochs` | 200 | LF training epochs |
| `hf_epochs` | 100 | HF training epochs |

---

## Acquisition Function

| Model | Acquisition | Uncertainty |
|-------|-------------|-------------|
| MFGP | Expected Improvement | GP posterior variance |
| Others (11개) | argmin(mean) | constant 0.1 (dummy) |

**Note**: DNN 모델들은 proper uncertainty estimation이 없어서 pure exploitation (greedy) 전략 사용.
