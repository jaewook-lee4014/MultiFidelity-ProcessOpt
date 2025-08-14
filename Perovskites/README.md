# Perovskite Multi-Fidelity Bayesian Optimization

페로브스카이트 태양전지 물질 최적화를 위한 Multi-Fidelity Bayesian Optimization 연구 코드베이스

## 📁 프로젝트 구조

```
Perovskites/
├── 0.Data/                         # 원시 데이터
│   ├── lookup_table.pkl           # Perovskite 속성 데이터베이스
│   ├── organics.json              # 유기 양이온 descriptors
│   ├── cations.json               # 금속 양이온 descriptors
│   └── anions.json                # 음이온 descriptors
│
├── 1.Atlas_cod/                    # Atlas 기반 MFBO
│   ├── MFBO.ipynb                 # Multi-fidelity BO 구현
│   └── visualization.ipynb        # 결과 시각화
│
├── 2.Transfer_learning/           
│   ├── Pure_TL/                   # Transfer Learning 실험
│   │   └── *.ipynb                # 다양한 TL 실험 노트북
│   │
│   └── Pure_TL_BO/                # Transfer Learning + BO (메인)
│       ├── main.py                # CLI 실행 스크립트
│       ├── config.py              # 설정 관리
│       ├── models.py              # 모델 구현
│       ├── data_utils.py          # 데이터 처리
│       ├── optimization.py        # BO 알고리즘
│       ├── visualization.py       # 시각화
│       ├── experiment_runner.py   # 실험 관리 도구
│       ├── test_tl_bo.ipynb       # 테스트 노트북
│       ├── experiments/           # 실험 노트북
│       │   └── 01_model_comparison.ipynb
│       └── results/               # 실험 결과 저장
│
├── results/                        # 전체 프로젝트 결과
│   ├── atlas_mfbo/                # Atlas 실험 결과
│   ├── transfer_learning/         # TL 실험 결과
│   └── data_exports/              # 데이터 내보내기
│
└── CLAUDE.md                      # Claude AI 가이드
```

## 🚀 Quick Start

### 1. 환경 설정
```bash
cd 2.Transfer_learning/Pure_TL_BO
pip install -r requirements.txt
```

### 2. 기본 실행
```bash
# 단일 최적화 실행
python main.py --mode single --cost-budget 50 --verbose

# 다중 실행 (통계 분석)
python main.py --mode multiple --num-runs 100

# Hyperparameter BO 포함
python main.py --mode single --use-hyperparameter-bo
```

### 3. 노트북 실험
```python
# Jupyter notebook 실행
jupyter notebook test_tl_bo.ipynb

# 또는 모델 비교 실험
jupyter notebook experiments/01_model_comparison.ipynb
```

## 🔬 연구 개요

### 목표
- **계산 비용 최소화**: Multi-fidelity 접근으로 8-10배 비용 절감
- **성능 유지**: High-fidelity만 사용한 것과 유사한 최적화 성능
- **Transfer Learning**: Low-fidelity 데이터로 사전학습, High-fidelity로 미세조정

### 핵심 기술
1. **Multi-Fidelity BO**: 저비용/고비용 시뮬레이션 균형
2. **Transfer Learning DNN**: 특징 추출 및 불확실성 정량화
3. **Bayesian Optimization**: Expected Improvement 획득 함수

## 📊 실험 재현성

### 시드 고정
```python
from experiment_runner import ExperimentRunner
runner = ExperimentRunner()
runner.set_seed(42)  # 완전한 재현성 보장
```

### 실험 추적
- 모든 실험 자동 기록
- 고유 ID로 각 실험 식별
- 설정과 결과 자동 저장

## 🔄 모델 확장

### 새 모델 추가 방법
```python
# 1. models.py에 모델 클래스 추가
class MyModel(BaseModel):
    def fit(self, X_low, y_low, X_high, y_high):
        # 구현
    
    def predict(self, X, return_std=False):
        # 구현

# 2. experiment_runner.py에 등록
from experiment_runner import model_registry
model_registry.register('my_model', MyModel)

# 3. 실험에서 사용
model = model_registry.create('my_model')
```

## 📈 주요 결과

- **비용 효율성**: 동일 성능 달성에 8-10배 적은 계산 비용
- **수렴 속도**: 50 비용 단위 내 목표값 도달
- **최적 조성**: 목표 bandgap (1.34 eV)에 근접한 페로브스카이트 발견

## 📝 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `cost_budget` | 50 | 총 계산 비용 예산 |
| `target_value` | 1.34 | 목표 bandgap (eV) |
| `num_initial` | 5 | 초기 랜덤 샘플 수 |
| `high_fidelity_ratio` | 0.125 | High-fidelity 비율 (1:8) |

## 🛠️ 문제 해결

### Import 에러
```bash
# Python path 설정
export PYTHONPATH=$PYTHONPATH:/path/to/Perovskites
```

### GPU 사용
```python
# config.py에서 설정
DEVICE_CONFIG = {'use_gpu': True}
```

## 📚 참고문헌

- Multi-fidelity Bayesian Optimization 논문
- Transfer Learning for Materials Discovery
- Perovskite Solar Cell 최적화 연구

## 🤝 기여

연구 관련 문의나 개선사항은 이슈로 등록해주세요.