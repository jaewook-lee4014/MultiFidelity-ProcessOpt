# Bayesian Neural Network (BNN) 가이드

이 가이드는 DNGO 대신 Bayesian Neural Network를 사용하여 페로브스카이트 최적화를 수행하는 방법을 설명합니다.

## 🧠 BNN 구현 특징

### 1. Variational Inference
- **Bayes by Backprop**: 가중치를 확률분포로 모델링
- **Local Reparameterization Trick**: 효율적인 미니배치 학습
- **Mean-field Gaussian**: 각 가중치를 독립적인 가우시안으로 근사

### 2. 불확실성 정량화
- **Epistemic Uncertainty**: 모델 파라미터의 불확실성 (데이터가 많아지면 감소)
- **Aleatoric Uncertainty**: 관측 노이즈의 불확실성 (고정)
- **Total Uncertainty**: Epistemic + Aleatoric (Expected Improvement 계산용)

### 3. 노이즈 모델링
- **Homoscedastic**: 전역 노이즈 분산 (기본)
- **Heteroscedastic**: 입력에 따라 달라지는 노이즈 분산

### 4. ELBO 손실함수
```
Loss = NLL + λ_KL × KL_divergence
```
- **NLL**: Negative Log Likelihood (데이터 적합도)
- **KL**: Prior와 Posterior 간의 KL divergence (정규화)
- **KL Warm-up**: 초기 epoch에서 λ_KL을 점진적으로 증가

## 📋 사용법

### 기본 BNN 실행
```bash
# 자동 디바이스 감지 (권장)
python main.py --model-type bnn --device auto --mode single --cost-budget 50 --verbose

# 특정 디바이스 지정
python main.py --model-type bnn --device mps --mode single --cost-budget 50 --verbose   # Apple Silicon
python main.py --model-type bnn --device cuda --mode single --cost-budget 50 --verbose  # NVIDIA GPU
python main.py --model-type bnn --device cpu --mode single --cost-budget 50 --verbose   # CPU

# 다중 실행
python main.py --model-type bnn --mode multiple --num-runs 10 --cost-budget 50
```

### BNN 하이퍼파라미터 조정 (수동)
```bash
python main.py --model-type bnn \
  --bnn-hidden-dims 64 64 \
  --kl-weight 1.0 \
  --kl-warmup-epochs 10 \
  --prior-std 1.0 \
  --noise-type homoscedastic \
  --n-samples 100 \
  --mode single --verbose
```

### BNN 하이퍼파라미터 베이지안 최적화 (자동)
```bash
# BNN 자동 하이퍼파라미터 튜닝
python main.py --model-type bnn \
  --use-hyperparameter-bo \
  --finetune-bo-trials 10 \
  --data-size small \
  --mode single --verbose

# 더 정밀한 튜닝 (대용량 데이터)
python main.py --model-type bnn \
  --use-hyperparameter-bo \
  --finetune-bo-trials 20 \
  --data-size medium \
  --mode single --verbose
```

### DNGO vs BNN 비교
```bash
# DNGO 실행
python main.py --model-type dngo --mode single --cost-budget 50 --verbose

# BNN 실행  
python main.py --model-type bnn --mode single --cost-budget 50 --verbose
```

## ⚙️ BNN 설정 매개변수

| 매개변수 | 기본값 | 설명 |
|---------|--------|------|
| `--bnn-hidden-dims` | `[64, 64]` | BNN hidden layer 차원들 |
| `--kl-weight` | `1.0` | KL divergence 가중치 |
| `--kl-warmup-epochs` | `10` | KL warm-up epochs |
| `--prior-std` | `1.0` | 사전분포 표준편차 |
| `--noise-type` | `homoscedastic` | 노이즈 모델링 타입 |
| `--n-samples` | `100` | 예측 시 몬테카를로 샘플 수 |
| `--use-hyperparameter-bo` | `False` | BNN 하이퍼파라미터 BO 사용 |
| `--finetune-bo-trials` | `0` | BNN BO 시행 횟수 (5-15 권장) |
| `--data-size` | `small` | BO 탐색 공간 크기 (small/medium/large) |

## 🧪 테스트

### BNN 기능 테스트
```bash
# BNN 기본 기능 테스트
python test_bnn.py
```

이 테스트는 다음을 확인합니다:
- BNN 기본 학습/예측 기능
- 동차분산 vs 이차분산 모델링
- 불확실성 정량화 (근거리 vs 원거리)

### MPS (Apple Silicon) 테스트
```bash
# MPS 지원 및 성능 테스트
python test_mps.py
```

이 테스트는 다음을 확인합니다:
- MPS 사용 가능 여부 자동 감지
- MPS vs CPU 성능 비교
- MPS 메모리 관리 최적화

## 📊 예상 결과

### DNGO 대비 BNN 장점
1. **더 강건한 불확실성**: Epistemic + Aleatoric 분리
2. **전이학습 지원**: 사전분포를 통한 pretrained weights 활용
3. **이론적 근거**: 베이지안 추론의 엄밀한 수학적 기반
4. **자동 하이퍼파라미터 튜닝**: BNN 특화 베이지안 최적화 지원

### 하이퍼파라미터 자동 최적화 범위 (축소 버전)
**최적화 대상 (4개 핵심 파라미터):**
- **모델 구조**: hidden_dims ([32], [64], [32,32], [64,32] 등)
- **학습 설정**: finetune_epochs (50-150), finetune_lr (1e-4 ~ 1e-3)  
- **베이지안 설정**: kl_weight (0.1, 1.0, 10.0)

**고정 파라미터 (최적화하지 않음):**
- pretrain_epochs=200, pretrain_lr=1e-3
- kl_warmup_epochs=10, prior_std=1.0  
- noise_type='homoscedastic'

### 계산 비용 (축소 후)
- BNN (수동): DNGO 대비 약 2-3배 느림
- BNN (BO): DNGO 대비 약 3-5배 느림 (핵심 4개 파라미터만 탐색)
- 탐색 공간 축소로 빠른 최적화 + 성능 향상 기대

## 🔧 고급 설정

### 1. Apple Silicon (MPS) 최적화
```bash
# 자동 감지 (권장)
python main.py --model-type bnn --device auto --mode single --verbose

# MPS 강제 사용
python main.py --model-type bnn --device mps --mode single --verbose

# 큰 모델 + MPS
python main.py --model-type bnn --device mps \
  --bnn-hidden-dims 128 64 32 \
  --n-samples 200 \
  --mode single --verbose
```

### 2. 이차분산 노이즈 모델링
```bash
python main.py --model-type bnn --noise-type heteroscedastic --device auto
```

### 3. 깊은 네트워크
```bash
python main.py --model-type bnn --bnn-hidden-dims 128 64 32 --device auto
```

### 4. 높은 정밀도 예측
```bash
python main.py --model-type bnn --n-samples 200 --device auto
```

## 📁 파일 구조

```
├── bnn_models.py           # BNN 구현 (VariationalLinear, BayesianNeuralNetwork)
├── optimization_bnn.py     # BNN 최적화 루프
├── device_utils.py         # 자동 디바이스 감지 및 MPS 최적화
├── test_bnn.py            # BNN 기능 테스트 스크립트
├── test_mps.py            # MPS 지원 및 성능 테스트
├── main.py                # 통합 실행 스크립트 (DNGO/BNN 선택 가능)
└── README_BNN.md          # BNN 사용 가이드 (이 파일)
```

## 🔬 이론적 배경

### Variational Inference
가중치 W의 사후분포 p(W|D)를 변분분포 q(W|θ)로 근사:
```
q(W|θ) = ∏ᵢ N(μᵢ, σᵢ²)
```

### ELBO 최적화
```
ELBO = E_q[log p(y|x,W)] - KL(q(W)||p(W))
```

### 예측 분포
```
p(y*|x*,D) ≈ ∫ p(y*|x*,W) q(W|θ) dW
```
몬테카를로 샘플링으로 근사

## 💡 팁

1. **디바이스 자동 감지**: `--device auto` 사용으로 최적 디바이스 자동 선택
2. **Apple Silicon 최적화**: M1/M2/M3 Mac에서는 MPS로 2-3배 가속 가능
3. **KL weight 조정**: 작은 데이터셋에서는 kl-weight를 줄여보세요
4. **Warm-up 기간**: 복잡한 데이터에서는 kl-warmup-epochs를 늘려보세요  
5. **Prior std**: 전이학습 시 prior-std를 줄이면 사전 지식을 더 활용
6. **샘플 수**: 정확한 불확실성이 필요하면 n-samples를 늘려보세요
7. **MPS 메모리**: 큰 모델 사용 시 메모리 부족하면 hidden-dims를 줄여보세요

## 🚀 다음 단계

1. 실제 데이터에서 DNGO vs BNN 성능 비교
2. 하이퍼파라미터 자동 튜닝 (Optuna 등)
3. GPU 가속화 (`--device cuda`)
4. 앙상블 방법과의 조합