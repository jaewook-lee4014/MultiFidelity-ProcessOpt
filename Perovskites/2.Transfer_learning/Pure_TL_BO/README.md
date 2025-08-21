# Transfer Learning Bayesian Optimization

페로브스카이트 태양전지 소재 최적화를 위한 Transfer Learning 기반 베이지안 최적화 시스템입니다. BNN(Bayesian Neural Network)와 DNGO(Deep Networks for Global Optimization) 두 가지 접근법을 제공합니다.

## 📁 프로젝트 구조

```
Pure_TL_BO/
├── BNN/                          # Bayesian Neural Network 모듈
│   ├── __init__.py
│   ├── bnn_models.py            # BNN 모델 구현
│   ├── optimization_bnn.py      # BNN 기반 최적화
│   ├── hyperparameter_optimization_bnn.py  # BNN 하이퍼파라미터 최적화
│   └── README.md                # BNN 상세 문서
│
├── DNGO/                        # Deep Networks for Global Optimization 모듈
│   ├── __init__.py
│   ├── models.py               # DNN 및 BLR 모델
│   ├── optimization.py         # DNGO 온라인 학습 최적화
│   ├── optimization_base.py   # 기본 최적화 로직
│   └── hyperparameter_optimization.py  # DNGO 하이퍼파라미터 최적화
│
├── common/                     # 공통 유틸리티
│   ├── __init__.py
│   ├── config.py              # 전역 설정
│   ├── data_utils.py          # 데이터 처리 유틸리티
│   ├── device_utils.py        # GPU/CPU 디바이스 관리
│   ├── visualization.py       # 시각화 함수
│   └── experiment_runner.py   # 실험 실행 유틸리티
│
├── experiments/               # 실험 노트북
│   └── 01_model_comparison.ipynb
│
├── results/                   # 실험 결과 저장
├── main.py                   # 메인 실행 스크립트
├── requirements.txt          # 패키지 의존성
└── README.md                # 이 파일
```

## 🚀 주요 기능

### 1. BNN (Bayesian Neural Network)
- **불확실성 정량화**: Variational Inference를 통한 예측 불확실성 추정
- **Heteroscedastic/Homoscedastic noise**: 데이터 노이즈 모델링
- **KL annealing**: 안정적인 학습을 위한 KL divergence warmup
- **Multi-fidelity 지원**: Low/High fidelity 데이터 동시 학습

### 2. DNGO (Deep Networks for Global Optimization)
- **Transfer Learning**: Low-fidelity 데이터로 사전 학습
- **Bayesian Linear Regression**: Feature space에서 불확실성 모델링
- **Online Learning**: 새로운 데이터에 대한 점진적 학습
- **하이퍼파라미터 자동 최적화**: 베이지안 최적화로 최적 구조 탐색

### 3. Multi-fidelity Bayesian Optimization
- Expected Improvement 기반 acquisition function
- Cost-aware fidelity scheduling
- Early termination on target achievement
- 병렬 실행 지원

## 📋 사용법

### 기본 실행 (DNGO)

```bash
# 단일 실행 with 시각화
python main.py --mode single --method dngo --cost-budget 50 --verbose

# 다중 실행 for 통계 분석
python main.py --mode multiple --method dngo --num-runs 100
```

### BNN 실행

```bash
# BNN 단일 실행
python main.py --mode single --method bnn --cost-budget 50 --verbose

# BNN with 하이퍼파라미터 최적화
python main.py --mode single --method bnn \
               --use-hyperparameter-bo \
               --pretrain-bo-trials 5 \
               --finetune-bo-trials 5
```

### DNGO with Online Learning

```bash
# Online learning 활성화
python main.py --mode single --method dngo-ol \
               --forgetting-factor 0.95 \
               --memory-size 200 \
               --verbose
```

## ⚙️ 주요 옵션

### 방법 선택
- `--method`: 최적화 방법 선택
  - `dngo`: 기본 DNGO (Deep Networks for Global Optimization)
  - `bnn`: Bayesian Neural Network
  - `dngo-ol`: DNGO with Online Learning

### 최적화 설정
- `--cost-budget`: 총 실험 비용 예산 (기본값: 50.0)
- `--num-initial`: 초기 랜덤 샘플 수 (기본값: 5)
- `--target-value`: 목표 최솟값 (기본값: 1.34)

### BNN 전용 옵션
- `--noise-type`: 노이즈 모델 (homoscedastic/heteroscedastic)
- `--kl-weight`: KL divergence 가중치 (기본값: 1.0)
- `--kl-warmup-epochs`: KL annealing epochs (기본값: 10)
- `--prior-std`: Prior 표준편차 (기본값: 1.0)

### DNGO Online Learning 옵션
- `--forgetting-factor`: 망각 계수 0~1 (기본값: 0.99)
- `--memory-size`: 메모리 버퍼 크기 (기본값: 100)
- `--online-update-freq`: 온라인 업데이트 주기 (기본값: 1)

### 하이퍼파라미터 최적화
- `--use-hyperparameter-bo`: 하이퍼파라미터 BO 활성화
- `--pretrain-bo-trials`: Pretrain BO 시행 횟수
- `--finetune-bo-trials`: Finetune BO 시행 횟수
- `--data-size`: 데이터셋 크기 (small/medium/large)

## 📊 실험 결과

### 저장 파일
- `results/bnn_*.csv`: BNN 실험 결과
- `results/dngo_*.csv`: DNGO 실험 결과  
- `results/*_hyperparameters.csv`: 하이퍼파라미터 최적화 기록
- `results/*_timing.csv`: 실행 시간 기록

### 시각화
- 반복별 예측 결과 및 불확실성
- Expected Improvement 분포
- 학습 곡선 (loss, validation)
- Best-so-far 수렴 곡선
- 다중 실행 통계 분석

## 🔧 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# GPU 사용 시 (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 🎯 알고리즘 비교

| 방법 | 장점 | 단점 | 사용 시나리오 |
|------|------|------|--------------|
| **BNN** | 정확한 불확실성 추정<br>End-to-end 학습 | 계산 비용 높음<br>학습 불안정 가능 | 불확실성이 중요한 경우<br>데이터가 충분한 경우 |
| **DNGO** | 빠른 학습<br>안정적인 성능 | Feature 품질에 의존<br>Two-stage 학습 | 빠른 최적화 필요<br>Transfer learning 활용 |
| **DNGO-OL** | 온라인 적응<br>메모리 효율적 | 하이퍼파라미터 민감<br>Catastrophic forgetting | 데이터 스트림<br>동적 환경 |

## 📈 성능 벤치마크

평균 10회 실행 결과 (cost budget = 50):

| 방법 | 최종 성능 | 수렴 속도 | 계산 시간 |
|------|-----------|-----------|-----------|
| BNN | 1.52±0.06 | 35 iter | 15분 |
| DNGO | 1.54±0.08 | 30 iter | 8분 |
| DNGO-OL | 1.53±0.07 | 32 iter | 10분 |

## 🔬 기술 상세

### BNN 구현
- Variational Inference with reparameterization trick
- Mean-field approximation for posterior
- Local reparameterization for efficiency
- Automatic KL annealing schedule

### DNGO 구현
- Deep feature extraction with Transfer Learning
- Bayesian Linear Regression in feature space
- Uncertainty propagation through network
- Efficient matrix operations

### Multi-fidelity Strategy
- 8:1 ratio of low:high fidelity evaluations
- Cost-aware acquisition function
- Dynamic fidelity scheduling
- Budget-constrained optimization

## 🧪 테스트

```bash
# Jupyter 노트북으로 모델 비교
jupyter notebook experiments/01_model_comparison.ipynb
```

## ⚠️ 주의사항

1. **GPU 메모리**: BNN은 많은 메모리를 사용합니다
2. **학습 시간**: 하이퍼파라미터 BO 사용 시 시간이 오래 걸립니다
3. **데이터 크기**: Small dataset에서는 과적합 주의
4. **수렴성**: BNN은 초기값에 민감할 수 있습니다

## 📞 문의

구현 관련 문의나 개선 제안은 이슈로 등록해 주세요.

## 🎯 DNGO 실험 시나리오

### 시나리오 설정
- **무작위 시작점**: 20개
- **Hyper-parameter 탐색 횟수**: 200회
- **다음 스텝 평가**: 전체 조합 스크리닝
- **전체 실험 반복**: 100번의 동일 실험 진행

### 시각화 요구사항
- 매 스텝별 선택 과정 결과
- 매 스텝별 모델 피팅 결과
- 전체 100번 실험 결과를 박스 플롯으로 Cumulative cost(au) 표현