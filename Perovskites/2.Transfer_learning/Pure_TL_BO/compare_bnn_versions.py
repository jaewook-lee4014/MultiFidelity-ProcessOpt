#!/usr/bin/env python3
"""
BNN 버전 비교: 이전 BNN vs 현재 BNN (개선됨) vs DNGO
모델 예측 성능 비교 시각화
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 4-model comparison 결과 (slurm_4model_cmp_30299523.out)
# 10회 실험 평균
results_4model = {
    'DNGO (baseline)': {
        'rmse': [0.9547, 0.9681, 0.8198, 0.9144, 0.9524, 1.1584, 0.6834, 0.9051, 1.3700, 0.7790],
        'r2': [0.2942, 0.2742, 0.4796, 0.3525, 0.2976, -0.0392, 0.6383, 0.3657, -0.4534, 0.5301],
        'spearman': [0.5985, 0.7281, 0.6597, 0.5588, 0.4969, 0.5529, 0.7727, 0.6912, 0.7143, 0.7309]
    },
    'DNGO + HP-BO': {
        'rmse': [0.7874, 0.5093, 0.5501, 0.6411, 0.5801, 0.7997, 0.6762, 0.5966, 0.6336, 0.7153],
        'r2': [0.5199, 0.7992, 0.7657, 0.6817, 0.7394, 0.5048, 0.6459, 0.7244, 0.6891, 0.6038],
        'spearman': [0.7225, 0.8440, 0.8286, 0.8409, 0.8806, 0.7230, 0.8084, 0.7982, 0.7872, 0.7937]
    },
    'BNN (baseline)\n이전 버전': {
        'rmse': [0.7203, 0.8168, 0.6524, 0.6883, 0.8461, 0.7352, 0.7600, 0.7772, 0.7915, 0.8092],
        'r2': [0.5982, 0.4833, 0.6704, 0.6331, 0.4457, 0.5814, 0.5528, 0.5322, 0.5149, 0.4929],
        'spearman': [0.7187, 0.6552, 0.7601, 0.7414, 0.6111, 0.7276, 0.7108, 0.7291, 0.6799, 0.6881]
    },
    'BNN + HP-BO\n현재 버전 (개선됨)': {
        'rmse': [0.5883, 0.5692, 0.5631, 0.9406, 0.7987, 0.6119, 0.9973, 0.6151, 0.5765, 0.5730],
        'r2': [0.7320, 0.7491, 0.7545, 0.3149, 0.5060, 0.7101, 0.2297, 0.7070, 0.7426, 0.7458],
        'spearman': [0.7992, 0.7934, 0.8367, 0.8624, 0.8818, 0.7975, 0.8518, 0.8154, 0.7798, 0.8472]
    }
}

# metrics_summary.csv에서의 이전 BNN (최적화 없이 돌린 것) - 매우 낮은 성능
old_bnn_metrics = {
    'rmse': [1.2664, 2.7482, 1.5417, 1.6453, 2.3476, 1.8628, 2.2838, 1.7351, 1.5557, 2.3862],
    'r2': [-0.2419, -4.8485, -0.8406, -1.0964, -3.2678, -1.6873, -3.0391, -1.3314, -0.8741, -3.4094],
    'spearman': [0.4433, -0.0581, 0.0401, 0.0318, -0.0005, 0.4254, 0.4794, 0.3230, 0.0040, 0.1852]
}

# 통계 계산
def calc_stats(data):
    return np.mean(data), np.std(data)

# 결과 정리
models = ['DNGO\n(baseline)', 'DNGO\n+ HP-BO', 'BNN (기존)\nbaseline', 'BNN (개선)\n+ HP-BO', 'BNN (구버전)\nNo HP-BO']
metrics = ['RMSE ↓', 'R² ↑', 'Spearman ρ ↑']

# 데이터 준비
rmse_means = []
rmse_stds = []
r2_means = []
r2_stds = []
spearman_means = []
spearman_stds = []

for key in ['DNGO (baseline)', 'DNGO + HP-BO', 'BNN (baseline)\n이전 버전', 'BNN + HP-BO\n현재 버전 (개선됨)']:
    m, s = calc_stats(results_4model[key]['rmse'])
    rmse_means.append(m)
    rmse_stds.append(s)
    m, s = calc_stats(results_4model[key]['r2'])
    r2_means.append(m)
    r2_stds.append(s)
    m, s = calc_stats(results_4model[key]['spearman'])
    spearman_means.append(m)
    spearman_stds.append(s)

# 구버전 BNN 추가
m, s = calc_stats(old_bnn_metrics['rmse'])
rmse_means.append(m)
rmse_stds.append(s)
m, s = calc_stats(old_bnn_metrics['r2'])
r2_means.append(m)
r2_stds.append(s)
m, s = calc_stats(old_bnn_metrics['spearman'])
spearman_means.append(m)
spearman_stds.append(s)

# Figure 생성
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

colors = ['#2ecc71', '#27ae60', '#e74c3c', '#c0392b', '#95a5a6']
labels = ['DNGO\n(baseline)', 'DNGO\n+ HP-BO', 'BNN 기존\n(baseline)', 'BNN 개선\n+ HP-BO', 'BNN 구버전\n(No HP-BO)']

x = np.arange(len(labels))
width = 0.6

# RMSE (낮을수록 좋음)
ax1 = axes[0]
bars1 = ax1.bar(x, rmse_means, width, yerr=rmse_stds, color=colors, capsize=5, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('RMSE (낮을수록 좋음)', fontsize=12, fontweight='bold')
ax1.set_title('RMSE 비교', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10)
ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Threshold')
ax1.set_ylim(0, 2.5)
for i, (bar, mean) in enumerate(zip(bars1, rmse_means)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + rmse_stds[i] + 0.05,
             f'{mean:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# R² (높을수록 좋음)
ax2 = axes[1]
bars2 = ax2.bar(x, r2_means, width, yerr=r2_stds, color=colors, capsize=5, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('R² (높을수록 좋음)', fontsize=12, fontweight='bold')
ax2.set_title('R² 비교', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=10)
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax2.set_ylim(-3, 1)
for i, (bar, mean) in enumerate(zip(bars2, r2_means)):
    ypos = max(bar.get_height(), 0) + 0.05
    if mean < 0:
        ypos = 0.05
    ax2.text(bar.get_x() + bar.get_width()/2, ypos,
             f'{mean:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Spearman (높을수록 좋음)
ax3 = axes[2]
bars3 = ax3.bar(x, spearman_means, width, yerr=spearman_stds, color=colors, capsize=5, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Spearman ρ (높을수록 좋음)', fontsize=12, fontweight='bold')
ax3.set_title('Spearman Correlation 비교', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=10)
ax3.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good threshold')
ax3.set_ylim(-0.2, 1.0)
for i, (bar, mean) in enumerate(zip(bars3, spearman_means)):
    ypos = max(bar.get_height(), 0) + spearman_stds[i] + 0.02
    if mean < 0:
        ypos = 0.02
    ax3.text(bar.get_x() + bar.get_width()/2, ypos,
             f'{mean:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('BNN 버전별 성능 비교: 구버전 vs 개선된 버전 vs DNGO\n(10회 실험 평균 ± 표준편차)',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('bnn_version_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: bnn_version_comparison.png")

# 개선률 계산
print("\n" + "="*70)
print("성능 개선 분석")
print("="*70)

print("\n1. BNN 개선 효과 (기존 baseline → HP-BO 적용):")
rmse_improve = (rmse_means[2] - rmse_means[3]) / rmse_means[2] * 100
r2_improve = (r2_means[3] - r2_means[2]) / abs(r2_means[2]) * 100
spearman_improve = (spearman_means[3] - spearman_means[2]) / spearman_means[2] * 100
print(f"   RMSE: {rmse_means[2]:.4f} → {rmse_means[3]:.4f} ({rmse_improve:+.1f}% 감소)")
print(f"   R²:   {r2_means[2]:.4f} → {r2_means[3]:.4f} ({r2_improve:+.1f}% 증가)")
print(f"   ρ:    {spearman_means[2]:.4f} → {spearman_means[3]:.4f} ({spearman_improve:+.1f}% 증가)")

print("\n2. 구버전 BNN vs 개선된 BNN (HP-BO 적용):")
rmse_improve2 = (rmse_means[4] - rmse_means[3]) / rmse_means[4] * 100
print(f"   RMSE: {rmse_means[4]:.4f} → {rmse_means[3]:.4f} ({rmse_improve2:+.1f}% 감소)")
print(f"   R²:   {r2_means[4]:.4f} → {r2_means[3]:.4f} (개선)")
print(f"   ρ:    {spearman_means[4]:.4f} → {spearman_means[3]:.4f} (대폭 개선)")

print("\n3. DNGO + HP-BO vs BNN + HP-BO (최종 비교):")
print(f"   DNGO+BO RMSE: {rmse_means[1]:.4f}, BNN+BO RMSE: {rmse_means[3]:.4f}")
print(f"   DNGO+BO R²:   {r2_means[1]:.4f}, BNN+BO R²:   {r2_means[3]:.4f}")
print(f"   DNGO+BO ρ:    {spearman_means[1]:.4f}, BNN+BO ρ:    {spearman_means[3]:.4f}")

if rmse_means[1] < rmse_means[3]:
    print(f"\n   → DNGO + HP-BO가 RMSE 기준으로 더 좋음 ({rmse_means[1]:.4f} < {rmse_means[3]:.4f})")
else:
    print(f"\n   → BNN + HP-BO가 RMSE 기준으로 더 좋음")

if spearman_means[3] > spearman_means[1]:
    print(f"   → BNN + HP-BO가 Spearman 기준으로 더 좋음 ({spearman_means[3]:.4f} > {spearman_means[1]:.4f})")

print("\n" + "="*70)
print("결론")
print("="*70)
print("""
1. 구버전 BNN (No HP-BO): 매우 낮은 성능 (RMSE ~1.9, R² < 0)
   - 100회 BO 실험에서 20% 성공률의 원인

2. 개선된 BNN + HP-BO: 성능 대폭 향상
   - RMSE: 1.93 → 0.68 (65% 감소)
   - R²: -2.0 → 0.62 (크게 개선)
   - Spearman: 0.19 → 0.83 (4배 이상 향상)

3. DNGO + HP-BO vs BNN + HP-BO:
   - RMSE: DNGO가 약간 우수 (0.65 vs 0.68)
   - R²: DNGO가 약간 우수 (0.67 vs 0.62)
   - Spearman: BNN이 우수 (0.83 vs 0.80)
   - 두 모델 모두 양호한 성능

4. 현재 OL 모델 실험이 확장된 HP 탐색 공간으로 진행 중
   - epochs: 20-500, batch_size: [8~512]
   - DNGO-OL과 BNN-OL이 어떤 성능을 보일지 확인 예정
""")

# 추가: 박스플롯으로 분포 비교
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 6))

# 데이터 준비 (박스플롯용)
all_rmse = [
    results_4model['DNGO (baseline)']['rmse'],
    results_4model['DNGO + HP-BO']['rmse'],
    results_4model['BNN (baseline)\n이전 버전']['rmse'],
    results_4model['BNN + HP-BO\n현재 버전 (개선됨)']['rmse'],
    old_bnn_metrics['rmse']
]
all_r2 = [
    results_4model['DNGO (baseline)']['r2'],
    results_4model['DNGO + HP-BO']['r2'],
    results_4model['BNN (baseline)\n이전 버전']['r2'],
    results_4model['BNN + HP-BO\n현재 버전 (개선됨)']['r2'],
    old_bnn_metrics['r2']
]
all_spearman = [
    results_4model['DNGO (baseline)']['spearman'],
    results_4model['DNGO + HP-BO']['spearman'],
    results_4model['BNN (baseline)\n이전 버전']['spearman'],
    results_4model['BNN + HP-BO\n현재 버전 (개선됨)']['spearman'],
    old_bnn_metrics['spearman']
]

box_labels = ['DNGO\nbaseline', 'DNGO\n+HP-BO', 'BNN 기존\nbaseline', 'BNN 개선\n+HP-BO', 'BNN 구버전\nNo HP-BO']

# RMSE 박스플롯
bp1 = axes2[0].boxplot(all_rmse, labels=box_labels, patch_artist=True)
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes2[0].set_ylabel('RMSE', fontsize=12, fontweight='bold')
axes2[0].set_title('RMSE 분포 (낮을수록 좋음)', fontsize=14, fontweight='bold')
axes2[0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)

# R² 박스플롯
bp2 = axes2[1].boxplot(all_r2, labels=box_labels, patch_artist=True)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes2[1].set_ylabel('R²', fontsize=12, fontweight='bold')
axes2[1].set_title('R² 분포 (높을수록 좋음)', fontsize=14, fontweight='bold')
axes2[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)

# Spearman 박스플롯
bp3 = axes2[2].boxplot(all_spearman, labels=box_labels, patch_artist=True)
for patch, color in zip(bp3['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes2[2].set_ylabel('Spearman ρ', fontsize=12, fontweight='bold')
axes2[2].set_title('Spearman 분포 (높을수록 좋음)', fontsize=14, fontweight='bold')
axes2[2].axhline(y=0.7, color='green', linestyle='--', alpha=0.5)

plt.suptitle('BNN 버전별 성능 분포 비교 (박스플롯)\n(10회 실험)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('bnn_version_comparison_boxplot.png', dpi=150, bbox_inches='tight', facecolor='white')
print("\nSaved: bnn_version_comparison_boxplot.png")
