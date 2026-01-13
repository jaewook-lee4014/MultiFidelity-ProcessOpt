# DNGO/MFGP 시각화 가이드

## 2가지 시각화 유형

각 Fold마다 **2개의 Figure**를 생성:
1. **Composition 정렬**: Organic -> Cation -> Anion 순서
2. **Value 정렬**: 실제 bandgap 값 기준 오름차순

---

## Figure 1: Composition 정렬

### 기본 구조
- **레이아웃**: N개 패널 세로 배치 (`plt.subplots(N, 1, figsize=(24, 6*N))`)
- **X축**: Composition 정렬 (Organic -> Cation -> Anion 순서)
- **Y축**: Bandgap (eV)

### 각 패널 구성요소

```python
# 1. 불확실성 밴드 (±2σ)
ax.fill_between(x_axis,
                pred_sorted - 2*std_sorted,
                pred_sorted + 2*std_sorted,
                alpha=0.3, color='blue', label='±2σ')

# 2. 예측값 라인
ax.plot(x_axis, pred_sorted, 'b-', linewidth=0.8, alpha=0.7, label='Predicted')

# 3. 테스트 포인트 (검은색)
ax.scatter(x_axis[~train_mask], y_sorted[~train_mask],
           c='black', s=15, zorder=5, label='Test', alpha=0.6)

# 4. 훈련 포인트 (빨간 별)
ax.scatter(x_axis[train_mask], y_sorted[train_mask],
           c='red', s=100, marker='*', zorder=6, label='Train', edgecolors='darkred')

# 5. Organic 그룹 구분선
for i in range(1, len(param_space['organic'])):
    ax.axvline(x=i*group_size - 0.5, color='gray', linestyle='--', alpha=0.5)
```

### Composition 정렬 함수

```python
def get_composition_sort_idx(composition_info):
    """Organic -> Cation -> Anion 순서로 정렬"""
    sort_keys = [(c['org_idx'], c['cat_idx'], c['ani_idx']) for c in composition_info]
    return np.lexsort((
        [k[2] for k in sort_keys],  # Anion (3rd priority)
        [k[1] for k in sort_keys],  # Cation (2nd priority)
        [k[0] for k in sort_keys]   # Organic (1st priority)
    ))
```

### 색상 규칙
| 모델 | 색상 |
|------|------|
| MFGP | blue |
| DNGO | green |
| DNGO-ProgUnfreeze | purple |
| Pretrain Only | orange |
| DNGO-tanh | purple |
| Pretrain-tanh | red |

### 마커 규칙
| 데이터 | 마커 | 크기 |
|--------|------|------|
| Test 포인트 | 원 (기본) | s=15 |
| Train HF (9개) | 별 (*) | s=100 |
| Train LF (72개) | 원 (o) | s=40 |

### 제목 형식
```python
ax.set_title(f'{model_name} (vs {HF/LF}): RMSE={rmse:.3f}, R²={r2:.3f}', fontsize=14)
```

---

## Figure 2: Value 정렬 (Bandgap 값 기준)

### Value 정렬 함수

```python
def get_value_sort_idx(y_values):
    """Bandgap 값 기준 오름차순 정렬"""
    return np.argsort(y_values)
```

### 사용 예시

```python
# Value 기준 정렬 (HF 모델용)
value_sort_idx = get_value_sort_idx(data['y_all'])  # HF bandgap 기준
y_sorted_by_value = data['y_all'][value_sort_idx]
pred_sorted_by_value = pred_all[value_sort_idx]
std_sorted_by_value = std_all[value_sort_idx]

# Train/Test mask도 재정렬
train_mask_by_value = np.isin(value_sort_idx, data['hifi_idx'])
```

### 시각화 코드 (Value 정렬)

```python
fig, axes = plt.subplots(N, 1, figsize=(24, 6*N))
x_axis = np.arange(len(y_sorted_by_value))

ax = axes[0]
# 불확실성 밴드
ax.fill_between(x_axis,
                pred_sorted_by_value - 2*std_sorted_by_value,
                pred_sorted_by_value + 2*std_sorted_by_value,
                alpha=0.3, color='blue', label='±2σ')

# 예측값
ax.plot(x_axis, pred_sorted_by_value, 'b-', linewidth=0.8, alpha=0.7)

# 실제값 (정렬되어 있으므로 monotonic)
ax.scatter(x_axis[~train_mask_by_value], y_sorted_by_value[~train_mask_by_value],
           c='black', s=15, zorder=5, alpha=0.6)
ax.scatter(x_axis[train_mask_by_value], y_sorted_by_value[train_mask_by_value],
           c='red', s=100, marker='*', zorder=6)

ax.set_xlabel('Sample Index (sorted by bandgap value)', fontsize=12)
ax.set_ylabel('Bandgap (eV)', fontsize=12)
ax.set_title(f'{model_name} (vs HF): RMSE={rmse:.3f}, R²={r2:.3f} [Value Sorted]')
```

### Value 정렬의 장점
- 예측 오차가 어느 bandgap 영역에서 큰지 시각적으로 확인 가능
- 불확실성 밴드가 실제값을 얼마나 커버하는지 직관적
- Train 포인트가 어느 값 범위에 분포하는지 파악

---

## 파일 저장 규칙

```python
# Composition 정렬
plt.savefig(fold_dir / 'predictions_by_composition.png', dpi=150, bbox_inches='tight')

# Value 정렬
plt.savefig(fold_dir / 'predictions_by_value.png', dpi=150, bbox_inches='tight')
```

---

## 참고 코드
- **Composition 정렬 예시**: `run_full_comparison.py:295-377`
- **5패널 버전**: `run_tanh_comparison.py:196-308`

## 저장 설정
```python
plt.savefig(output_path, dpi=150, bbox_inches='tight')
```
