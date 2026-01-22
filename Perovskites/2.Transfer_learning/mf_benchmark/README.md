# Multi-Fidelity Bayesian Optimization Benchmark

12개의 Multi-Fidelity Transfer Learning 모델을 7개의 벤치마크에서 비교하는 실험 프레임워크.

## Experimental Settings

### Benchmarks (7개)

| Benchmark | Type | Dim | Cost Ratio (ρ) | R² | Budget | Objective |
|-----------|------|-----|----------------|-----|--------|-----------|
| **Branin-Fav** | Synthetic | 2D | 0.1 | high (α=0.8) | 50 | minimize |
| **Branin-Unfav** | Synthetic | 2D | 0.5 | low (α=0.1) | 50 | minimize |
| **Park-Fav** | Synthetic | 4D | 0.1 | high (α=0.6) | 50 | minimize |
| **Park-Unfav** | Synthetic | 4D | 0.5 | low (α=0.0) | 50 | minimize |
| **COFs** | Chemistry | 14D | 0.065 | 0.98 | 30 | maximize → negate |
| **FreeSolv** | Chemistry | 10D | 0.1 | 0.88 | 50 | minimize |
| **Polarizability** | Chemistry | 10D | 0.167 | 0.99 | 30 | maximize → negate |

### Models (12개)

| Model | Category | Description |
|-------|----------|-------------|
| **MFGP** | GP-based | BoTorch SingleTaskMultiFidelityGP (baseline) |
| **Sequential** | Sequential Transfer | LF pretrain → HF finetune (freeze LF) |
| **Progressive** | Sequential Transfer | Gradual unfreezing during HF training |
| **Curriculum** | Sequential Transfer | Curriculum learning (LF→mixed→HF) |
| **Two-Stage Joint** | Joint Training | Joint training with fidelity embedding |
| **DNGO-Joint** | DNGO Variants | Deep Network + GP Output layer (joint) |
| **DNGO-Gradient** | DNGO Variants | DNGO with gradient-based transfer |
| **Knowledge Distillation** | Transfer Learning | LF→HF knowledge transfer via soft targets |
| **Domain Adaptation (MMD)** | Transfer Learning | MMD loss for domain alignment |
| **Soft Parameter Sharing** | Transfer Learning | L2 regularization between LF/HF networks |
| **Pseudo-Labeling** | Transfer Learning | Use LF predictions as pseudo-labels |
| **Adapter** | Transfer Learning | Adapter layers for transfer |

### Feature Representation (논문 기준)

| Benchmark | Method | Dimension |
|-----------|--------|-----------|
| Synthetic (Branin, Park) | Grid coordinates normalized [0,1]^d | 2D / 4D |
| COFs | Composition + crystal structure (직접 사용) | 14D |
| FreeSolv | RDKit 2D descriptors → PCA | 10D |
| Polarizability | RDKit 2D descriptors → PCA | 10D |

### Initial Sampling (논문 기준)

- **Budget 할당**: 전체 budget의 10%
- **HF/LF 비율**: 50% HF, 50% LF
- **Synthetic benchmarks**: Latin Hypercube Sampling (LHS)
- **Chemistry benchmarks**: Furthest Point Sampling (FPS)

### Acquisition Function

| Model | Acquisition | Uncertainty |
|-------|-------------|-------------|
| MFGP | Expected Improvement (EI) | GP posterior variance |
| Others (11개) | argmin(mean) - pure exploitation | 없음 (constant 0.1) |

### Fidelity Selection

- **Method**: Fixed round-robin schedule
- **Ratio**: `lf_per_hf = int(1/ρ)` LF evaluations per 1 HF evaluation
- **Note**: Cost-scaled acquisition (EI/ρ) 미사용 - 모든 모델에 동일 스케줄 적용

### Evaluation Metric

- **Simple Regret**: `r_t = best_found - f*` (minimize 문제 기준)
- **Maximize 문제**: 값을 negate하여 minimize로 변환
- **Seeds**: 20개 independent runs
- **보고**: mean ± std of final regret

---

## Known Issues & Limitations

### 1. Hardcoded Uncertainty (std = 0.1)

**위치**: `benchmark_parallel.py` lines 210, 270, 325, 372, 415, 459, 514, 579, 629, 688, 756

```python
return mean, np.ones_like(mean) * 0.1  # 11개 모델 전부
```

**문제점**:
- MFGP를 제외한 11개 모델이 **고정된 불확실성(0.1)**을 반환
- 실제 모델 불확실성을 반영하지 않음
- Bayesian Linear Regression 또는 Monte Carlo Dropout 등으로 대체 필요

**영향**:
- EI 계산 시 exploration 항이 의미 없음 (std가 constant)
- 현재 이 모델들은 `use_ei=False`로 설정되어 있어 실제 영향은 없음

### 2. Acquisition Asymmetry

**위치**: `benchmark_parallel.py` line 1046

```python
use_ei = (model_class == MFGP)  # MFGP만 EI 사용
```

**문제점**:
- MFGP: Expected Improvement (exploration + exploitation)
- 나머지 11개 모델: argmin(mean) (pure exploitation)
- 공정한 비교가 아닐 수 있음

**배경**:
- DNN 기반 모델들은 proper uncertainty를 제공하지 않아 불가피한 설계
- GP 기반 vs DNN 기반 모델의 본질적 차이

### 3. No Cost-Scaled Acquisition

**현재 구현**:
- 고정 스케줄로 LF/HF 선택 (round-robin)
- Acquisition을 cost로 나누지 않음

**논문 MFBO 방식**:
```
score_LF = EI(x, LF) / ρ
score_HF = EI(x, HF) / 1
→ 높은 score의 fidelity 선택
```

**영향**:
- 모든 모델이 동일한 fidelity 스케줄을 사용하므로 **모델 간 비교는 공정**
- 단, true MFBO 효과를 측정하려면 cost-scaled acquisition 필요

### 4. Exception Handling with Random Fallback

**위치**: `benchmark_parallel.py` line 1089

```python
except Exception as e:
    next_idx = np.random.choice(list(available))  # 랜덤 선택
```

**문제점**:
- 모델 학습/예측 실패 시 랜덤으로 다음 점 선택
- 어떤 조건에서 실패하는지 로깅되지 않음
- 결과의 재현성에 영향 줄 수 있음

### 5. Fixed Network Architecture

**현재 구현**:
- 모든 DNN 모델: `hidden_dim=64`, `num_layers=2`
- Learning rate: `1e-3`, Weight decay: `1e-4`
- Epochs: `lf_epochs=200`, `hf_epochs=100`

**문제점**:
- Hyperparameter tuning 없음
- 벤치마크/데이터셋에 따라 최적 설정이 다를 수 있음

---

## Usage

```bash
# Full benchmark (7 benchmarks × 12 models × 20 seeds = 1,680 runs)
python benchmark_parallel.py --n-seeds 20 --n-workers 48

# Quick test
python benchmark_parallel.py --n-seeds 3 --n-workers 8
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-seeds` | 20 | Number of independent seeds |
| `--n-workers` | 48 | Number of parallel workers |
| `--base-seed` | 42 | Starting seed value |
| `--output-dir` | auto | Output directory (auto-generated with timestamp) |

## Output Files

```
benchmark_YYYYMMDD_HHMMSS/
├── results_summary.csv      # Final regret per (benchmark, model, seed)
├── results_trajectory.csv   # Budget vs regret trajectory for plotting
├── config.json              # Experiment configuration
└── {bench}_{model}.csv      # Per-combination results (real-time saved)
```

## Hardware Requirements

Optimized for **NVIDIA GH200** (H100 96GB + 72 ARM Cortex-A cores):
- GPU: CUDA 12.4+
- CPU: 48+ cores recommended
- Memory: ~32GB RAM

### Installation (ARM64/GH200)

```bash
conda create -n mfbo python=3.11 -y
conda activate mfbo
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## References

- Cost ratios (ρ) and R² values: MFGP reference paper
- Synthetic functions: Branin, Park with favorable/unfavorable scenarios
- Chemistry benchmarks: COFs, FreeSolv, Polarizability datasets
