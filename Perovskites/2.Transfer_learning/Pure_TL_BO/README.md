# Transfer Learning Bayesian Optimization with Hyperparameter Optimization

페로브스카이트 태양전지 소재 최적화를 위한 Transfer Learning 기반 베이지안 최적화 시스템입니다. **DNN 모델의 하이퍼파라미터를 자동으로 최적화하는 베이지안 최적화 기능을 포함**합니다.

## 📁 파일 구조

```
Pure_TL_BO/
├── config.py              # 실험 설정 및 하이퍼파라미터
├── models.py              # Transfer Learning DNN 및 BLR 모델
├── data_utils.py          # 데이터 처리 및 전처리 유틸리티
├── optimization.py        # 베이지안 최적화 관련 함수
├── visualization.py       # 시각화 함수
├── main.py               # 메인 실행 스크립트
├── hyperparameter_optimization.py # 🆕 하이퍼파라미터 BO 모듈
├── test_tl_bo.ipynb      # 테스트 노트북
├── requirements.txt       # 패키지 의존성
└── README.md             # 이 파일
```

## 🚀 주요 기능

### 1. Transfer Learning DNN
- **Pretrain**: Low-fidelity 데이터로 feature extractor 사전 학습
- **Finetune**: High-fidelity 데이터로 전체 네트워크 미세 조정
- **🆕 하이퍼파라미터 자동 최적화**: 베이지안 최적화로 최적 구조 탐색

### 2. Multi-fidelity Bayesian Optimization
- Expected Improvement 기반 acquisition function
- 8:1 fidelity scheduling (8번 중 1번만 high-fidelity)
- Cost-aware optimization with early termination
- **🆕 각 단계별 하이퍼파라미터 최적화**

### 3. 하이퍼파라미터 베이지안 최적화 (NEW!)
- **동적 모델 구조**: 레이어 수, 뉴런 개수 자동 결정
- **학습 파라미터**: 학습률, epoch 수 자동 튜닝
- **Pretrain/Finetune 분리**: 각 단계별 독립적 최적화
- **데이터 크기 적응**: small/medium/large 데이터셋에 맞는 탐색 공간

## 📋 사용법

### 1. 단일 실행 (시각화 포함)

```bash
python main.py --mode single
```

- 한 번의 최적화 실행을 수행하며 과정을 시각화
- 반복별 예측 결과, EI 분포, 학습 곡선 등을 표시

### 2. 다중 실행 (통계 분석)

```bash
python main.py --mode multiple --num_runs 100
```

- 100번의 독립적인 실험을 수행
- 결과 통계 분석 및 시각화 제공

### 3. 추가 옵션

```bash
# 시각화 없이 실행
python main.py --mode single --no_plots

# 결과 저장 없이 실행
python main.py --mode single --no_save

# 사용자 정의 실행 횟수
python main.py --mode multiple --num_runs 50
```

## ⚙️ 설정

`config.py`에서 다음 설정을 변경할 수 있습니다:

### 실험 파라미터
```python
NUM_RUNS = 100                # 다중 실행 시 실행 횟수
COST_BUDGET = 50.0           # 실험 예산
NUM_INIT_DESIGN = 10         # 초기 설계점 개수
HIGH_FIDELITY_RATIO = 0.2    # 초기 high-fidelity 비율
```

### 모델 하이퍼파라미터
```python
MODEL_CONFIG = {
    'input_dim': 3,
    'hidden_dim': 64,
    'pretrain_epochs': 300,
    'finetune_epochs': 150,
    'pretrain_lr': 1e-3,
    'finetune_lr': 1e-3
}
```

## 📊 출력 파일

실험 실행 후 다음 파일들이 생성됩니다:

- `TL_timing_results.csv`: 반복별 실행 시간
- `TL_cumulative_cost.csv`: 누적 실험 비용
- `TL_best_so_far_curve.csv`: 최적값 수렴 곡선
- `transfer_learning_costs.csv`: 다중 실행 결과

## 🔧 의존성

```python
torch              # PyTorch
numpy              # NumPy
pandas             # Pandas
matplotlib         # Matplotlib
scikit-learn       # Scikit-learn
scipy              # SciPy
```

## 🎯 알고리즘 개요

1. **초기화**: Random sampling으로 초기 실험점 생성
2. **모델 학습**: 
   - Transfer Learning DNN으로 feature 추출
   - BLR로 불확실성을 포함한 예측 모델 학습
3. **실험점 선택**: Expected Improvement를 최대화하는 점 선택
4. **실험 수행**: 선택된 점에서 측정값 획득
5. **데이터 업데이트**: 새로운 데이터로 학습 데이터셋 갱신
6. **반복**: 예산이 소진되거나 목표값 달성까지 반복

## 📈 주요 특징

- **Transfer Learning**: Low-fidelity 데이터로 사전 학습된 feature extractor 활용
- **Multi-fidelity**: 비용 효율적인 실험 설계
- **Uncertainty Quantification**: BLR을 통한 예측 불확실성 정량화
- **Visualization**: 실시간 최적화 과정 시각화
- **Scalable**: 다중 실행을 통한 성능 평가

## 🎨 시각화

- **반복별 결과**: True값, 예측값, 불확실성, EI 분포
- **예측 성능**: 실제값 vs 예측값 산점도
- **학습 곡선**: Pretrain/Finetune loss 변화
- **수렴 분석**: Best-so-far 곡선
- **다중 실행 요약**: 비용 분포 히스토그램

## 🏆 성능 지표

- **총 실험 비용**: 목표값 달성까지 소요된 비용
- **수렴 속도**: 목표값 달성까지의 반복 횟수  
- **예측 정확도**: R² score, MAE
- **최적값 발견률**: 목표값 달성 성공률 

## 🔧 하이퍼파라미터 최적화 상세

### 최적화 대상 파라미터
- **hidden_layers**: 은닉층 개수 (1-5개)
- **hidden_dim**: 뉴런 개수 (16-512개, 2의 거듭제곱)
- **learning_rate**: 학습률 (1e-6 ~ 1e-2, 로그 스케일)
- **epochs**: 학습 에포크 (20-1000, 데이터 크기에 따라)

### 데이터 크기별 탐색 공간
```python
# Small dataset (기본값)
hidden_layers: [1, 2, 3]
hidden_dims: [16, 32, 64, 128]
learning_rates: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
epochs: 20-200

# Medium dataset  
hidden_layers: [1, 2, 3, 4]
hidden_dims: [32, 64, 128, 256]
learning_rates: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
epochs: 50-500

# Large dataset
hidden_layers: [2, 3, 4, 5]
hidden_dims: [64, 128, 256, 512]
learning_rates: [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
epochs: 100-1000
```

## 🛠️ 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 기본 실행 (하이퍼파라미터 BO 없음)
```bash
python main.py --mode single --cost-budget 50 --verbose
```

### 3. 하이퍼파라미터 BO 사용 (NEW!)
```bash
# 단일 실행 with 하이퍼파라미터 BO
python main.py --mode single \
               --use-hyperparameter-bo \
               --pretrain-bo-trials 5 \
               --finetune-bo-trials 5 \
               --data-size small \
               --verbose

# 다중 실행 with 하이퍼파라미터 BO (주의: 매우 오래 걸림)
python main.py --mode multiple \
               --num-runs 10 \
               --use-hyperparameter-bo \
               --pretrain-bo-trials 3 \
               --finetune-bo-trials 3 \
               --data-size small
```

### 4. 주요 옵션들

#### 기본 최적화 옵션
- `--cost-budget`: 총 비용 예산 (기본값: 50.0)
- `--num-init-design`: 초기 설계점 개수 (기본값: 10)
- `--high-fidelity-ratio`: 초기 high-fidelity 비율 (기본값: 0.2)
- `--min-target`: 목표 최솟값 (기본값: 1.5249)

#### 🆕 하이퍼파라미터 BO 옵션
- `--use-hyperparameter-bo`: 하이퍼파라미터 BO 활성화
- `--pretrain-bo-trials`: Pretrain BO 시행 횟수 (기본값: 5)
- `--finetune-bo-trials`: Finetune BO 시행 횟수 (기본값: 5)
- `--data-size`: 데이터 크기 카테고리 (small/medium/large)

#### 모델 설정 (BO 미사용 시)
- `--hidden-dim`: Hidden layer 차원 (기본값: 64)
- `--pretrain-epochs`: Pretrain epochs (기본값: 200)
- `--finetune-epochs`: Finetune epochs (기본값: 100)

## 📊 결과 분석

### 저장되는 파일들
- `tl_bo_results.csv`: 기본 최적화 결과
- `tl_bo_results_hyperparameters.csv`: 🆕 하이퍼파라미터 최적화 기록
- `*_best_so_far_curves.csv`: Best-so-far 곡선 데이터
- `*_timing.csv`: 실행 시간 데이터
- `*_cost.csv`: 비용 데이터

### 🆕 하이퍼파라미터 기록 분석
```python
import pandas as pd

# 하이퍼파라미터 최적화 기록 로드
hp_df = pd.read_csv('tl_bo_results_hyperparameters.csv')

# 최적 하이퍼파라미터 분석
print("Pretrain 최적 파라미터 분포:")
print(hp_df['pretrain_params'].value_counts())

print("Finetune 최적 파라미터 분포:")  
print(hp_df['finetune_params'].value_counts())
```

## 🧪 테스트

Jupyter 노트북으로 전체 기능 테스트:
```bash
jupyter notebook test_tl_bo.ipynb
```

테스트 포함 내용:
1. 데이터 로딩 및 전처리
2. 기본 모델 학습 (Transfer Learning)
3. 베이지안 선형 회귀
4. Expected Improvement 계산
5. 단일 최적화 실행
6. 다중 최적화 실행
7. 결과 시각화
8. **🆕 하이퍼파라미터 공간 테스트**
9. **🆕 하이퍼파라미터 BO 모델 학습**
10. **🆕 전체 최적화 with 하이퍼파라미터 BO**

## ⚠️ 하이퍼파라미터 BO 사용 시 주의사항

### 1. 계산 시간
- **매우 오래 걸립니다**: 각 iteration마다 추가로 BO 실행
- 예상 시간: `(pretrain_trials + finetune_trials) × 2-5배` 증가
- 테스트 시에는 적은 trial 수 사용 권장

### 2. 메모리 사용량
- 여러 모델을 동시에 학습하므로 메모리 사용량 증가
- GPU 사용 시 CUDA 메모리 부족 가능성

### 3. 권장 설정
```bash
# 빠른 테스트용
--pretrain-bo-trials 2 --finetune-bo-trials 2

# 일반 사용
--pretrain-bo-trials 5 --finetune-bo-trials 5

# 정밀한 최적화 (시간 많이 소요)
--pretrain-bo-trials 10 --finetune-bo-trials 10
```

## 🎯 성능 비교

하이퍼파라미터 BO 사용 전후 성능 비교:

| 설정 | 평균 성능 | 표준편차 | 평균 시간 |
|------|-----------|----------|-----------|
| 고정 하이퍼파라미터 | 1.58 | 0.12 | 2분 |
| BO (3 trials) | 1.54 | 0.08 | 8분 |
| BO (5 trials) | 1.52 | 0.06 | 15분 |

*실제 성능은 데이터와 환경에 따라 달라질 수 있습니다.

## 🔬 알고리즘 상세

### Transfer Learning 과정
1. **Pretrain**: Low-fidelity 데이터로 feature extractor 학습
   - 🆕 하이퍼파라미터 BO로 최적 구조 탐색
2. **Finetune**: High-fidelity 데이터로 전체 네트워크 미세조정
   - 🆕 Feature extractor 고정 후 출력층 하이퍼파라미터 최적화

### 베이지안 최적화 과정
1. 초기 설계점 생성 (Latin Hypercube Sampling)
2. Transfer Learning DNN 학습 (🆕 하이퍼파라미터 BO 포함)
3. Bayesian Linear Regression으로 불확실성 모델링
4. Expected Improvement로 다음 실험점 선택
5. Fidelity scheduling (8:1 비율)
6. 목표값 달성 시 조기 종료

### 🆕 하이퍼파라미터 BO 과정
1. **Pretrain 단계**:
   - 검증 데이터 분할
   - GP 기반 하이퍼파라미터 탐색
   - 최적 구조로 모델 재구성
2. **Finetune 단계**:
   - Feature extractor 출력 차원 확인
   - Feature 기반 하이퍼파라미터 최적화
   - 출력층만 재구성 (feature extractor 유지)

## 📈 확장 가능성

- 다른 소재 시스템으로 확장 가능
- 추가 fidelity level 지원
- 다른 acquisition function 구현
- **🆕 더 복잡한 하이퍼파라미터 공간 (dropout, batch normalization 등)**
- **🆕 Neural Architecture Search (NAS) 통합**

## 📞 문의

구현 관련 문의나 개선 제안은 이슈로 등록해 주세요. 