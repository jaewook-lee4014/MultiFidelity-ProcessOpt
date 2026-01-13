#!/usr/bin/env python
"""
Unfreeze ratio 실험 스크립트
- 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 비율 테스트
- 순차 실행 (메모리 안정성)
- 결과 시각화 및 저장
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 결과 저장 디렉토리
RESULTS_DIR = './unfreeze_ratio_experiment'
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_single_experiment(unfreeze_ratio, random_state, cost_budget, lookup, label_maps, PARAM_SPACE):
    """단일 unfreeze_ratio 실험 실행"""
    from DNGO.optimization import single_optimization_run_dngo_ol

    # use_freeze는 unfreeze_ratio < 1.0일 때만 True
    use_freeze = (unfreeze_ratio < 1.0)

    try:
        result = single_optimization_run_dngo_ol(
            param_space=PARAM_SPACE,
            label_maps=label_maps,
            lookup=lookup,
            cost_budget=cost_budget,
            num_init_design=5,
            verbose=False,
            random_state=random_state,
            use_freeze=use_freeze,
            unfreeze_ratio=unfreeze_ratio
        )

        # 결과 추출
        th = result.get('training_history', {})
        blr_L = th.get('blr_L_history', {})
        blr_H = th.get('blr_H_history', {})

        blr_l_errors = blr_L.get('prediction_errors', [])
        blr_h_errors = blr_H.get('prediction_errors', [])

        return {
            'unfreeze_ratio': unfreeze_ratio,
            'random_state': random_state,
            'best_so_far': result['best_so_far'],
            'total_cost': result['total_cost'],
            'blr_l_mean': float(np.mean(np.abs(blr_l_errors))) if blr_l_errors else None,
            'blr_l_std': float(np.std(blr_l_errors)) if blr_l_errors else None,
            'blr_h_mean': float(np.mean(np.abs(blr_h_errors))) if blr_h_errors else None,
            'blr_h_std': float(np.std(blr_h_errors)) if blr_h_errors else None,
            'n_iterations': len(result.get('costs', [])),
            'success': True,
            'error': None
        }
    except Exception as e:
        import traceback
        return {
            'unfreeze_ratio': unfreeze_ratio,
            'random_state': random_state,
            'success': False,
            'error': str(e) + "\n" + traceback.format_exc()
        }


def run_experiments_sequential(unfreeze_ratios, num_runs, cost_budget, lookup, label_maps, PARAM_SPACE):
    """순차적으로 실험 실행"""
    total_experiments = len(unfreeze_ratios) * num_runs
    print(f"총 {total_experiments}개 실험 실행 (ratios: {unfreeze_ratios}, runs: {num_runs})")
    print("=" * 60)

    results = []
    exp_idx = 0

    for ratio in unfreeze_ratios:
        for run_idx in range(num_runs):
            random_state = 42 + run_idx
            exp_idx += 1

            print(f"\n[{exp_idx}/{total_experiments}] Running: ratio={ratio:.1f}, seed={random_state}")

            result = run_single_experiment(
                ratio, random_state, cost_budget, lookup, label_maps, PARAM_SPACE
            )
            results.append(result)

            if result['success']:
                print(f"  ✓ Best={result['best_so_far']:.4f}, "
                      f"BLR-L={result['blr_l_mean']:.4f}, "
                      f"BLR-H={result['blr_h_mean']:.4f}")
            else:
                print(f"  ✗ FAILED: {result['error'][:100]}")

    return results


def aggregate_results(results):
    """결과를 unfreeze_ratio별로 집계"""
    from collections import defaultdict

    aggregated = defaultdict(list)

    for r in results:
        if r['success']:
            ratio = r['unfreeze_ratio']
            aggregated[ratio].append(r)

    summary = {}
    for ratio in sorted(aggregated.keys()):
        runs = aggregated[ratio]
        if len(runs) == 0:
            continue
        summary[ratio] = {
            'n_runs': len(runs),
            'best_so_far': {
                'mean': np.mean([r['best_so_far'] for r in runs]),
                'std': np.std([r['best_so_far'] for r in runs]),
                'min': np.min([r['best_so_far'] for r in runs]),
                'values': [r['best_so_far'] for r in runs]
            },
            'blr_l_mean': {
                'mean': np.mean([r['blr_l_mean'] for r in runs if r['blr_l_mean']]),
                'std': np.std([r['blr_l_mean'] for r in runs if r['blr_l_mean']]),
                'values': [r['blr_l_mean'] for r in runs if r['blr_l_mean']]
            },
            'blr_h_mean': {
                'mean': np.mean([r['blr_h_mean'] for r in runs if r['blr_h_mean']]),
                'std': np.std([r['blr_h_mean'] for r in runs if r['blr_h_mean']]),
                'values': [r['blr_h_mean'] for r in runs if r['blr_h_mean']]
            }
        }

    return summary


def plot_results(summary, save_path):
    """결과 시각화"""
    if len(summary) == 0:
        print("No data to plot")
        return

    ratios = sorted(summary.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Best Value (목표: 1.34에 가까울수록 좋음)
    ax1 = axes[0]
    means = [summary[r]['best_so_far']['mean'] for r in ratios]
    stds = [summary[r]['best_so_far']['std'] for r in ratios]
    mins = [summary[r]['best_so_far']['min'] for r in ratios]

    ax1.errorbar(ratios, means, yerr=stds, fmt='o-', capsize=5, label='Mean ± Std', markersize=8)
    ax1.scatter(ratios, mins, marker='*', s=150, c='red', label='Min (Best)', zorder=5)
    ax1.axhline(y=1.34, color='green', linestyle='--', linewidth=2, label='Target (1.34)')
    ax1.set_xlabel('Unfreeze Ratio', fontsize=12)
    ax1.set_ylabel('Best Value', fontsize=12)
    ax1.set_title('Best Value vs Unfreeze Ratio', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(ratios)

    # 2. BLR-L Prediction Error
    ax2 = axes[1]
    means = [summary[r]['blr_l_mean']['mean'] for r in ratios]
    stds = [summary[r]['blr_l_mean']['std'] for r in ratios]

    ax2.errorbar(ratios, means, yerr=stds, fmt='s-', capsize=5, color='blue', markersize=8)
    ax2.set_xlabel('Unfreeze Ratio', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error', fontsize=12)
    ax2.set_title('BLR-L (Low-Fidelity) Prediction Error', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(ratios)

    # 3. BLR-H Prediction Error
    ax3 = axes[2]
    means = [summary[r]['blr_h_mean']['mean'] for r in ratios]
    stds = [summary[r]['blr_h_mean']['std'] for r in ratios]

    ax3.errorbar(ratios, means, yerr=stds, fmt='^-', capsize=5, color='orange', markersize=8)
    ax3.set_xlabel('Unfreeze Ratio', fontsize=12)
    ax3.set_ylabel('Mean Absolute Error', fontsize=12)
    ax3.set_title('BLR-H (High-Fidelity) Prediction Error', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(ratios)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_detailed_results(summary, save_path):
    """상세 결과 시각화 (박스플롯)"""
    if len(summary) == 0:
        print("No data to plot")
        return

    ratios = sorted(summary.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Best Value 박스플롯
    ax1 = axes[0]
    data = [summary[r]['best_so_far']['values'] for r in ratios]
    bp1 = ax1.boxplot(data, positions=range(len(ratios)), widths=0.6, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    ax1.set_xticks(range(len(ratios)))
    ax1.set_xticklabels([f'{r:.1f}' for r in ratios])
    ax1.axhline(y=1.34, color='green', linestyle='--', linewidth=2, label='Target (1.34)')
    ax1.set_xlabel('Unfreeze Ratio', fontsize=12)
    ax1.set_ylabel('Best Value', fontsize=12)
    ax1.set_title('Best Value Distribution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. BLR-L Error 박스플롯
    ax2 = axes[1]
    data = [summary[r]['blr_l_mean']['values'] for r in ratios]
    bp2 = ax2.boxplot(data, positions=range(len(ratios)), widths=0.6, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightgreen')
    ax2.set_xticks(range(len(ratios)))
    ax2.set_xticklabels([f'{r:.1f}' for r in ratios])
    ax2.set_xlabel('Unfreeze Ratio', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error', fontsize=12)
    ax2.set_title('BLR-L Error Distribution', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 3. BLR-H Error 박스플롯
    ax3 = axes[2]
    data = [summary[r]['blr_h_mean']['values'] for r in ratios]
    bp3 = ax3.boxplot(data, positions=range(len(ratios)), widths=0.6, patch_artist=True)
    for patch in bp3['boxes']:
        patch.set_facecolor('lightsalmon')
    ax3.set_xticks(range(len(ratios)))
    ax3.set_xticklabels([f'{r:.1f}' for r in ratios])
    ax3.set_xlabel('Unfreeze Ratio', fontsize=12)
    ax3.set_ylabel('Mean Absolute Error', fontsize=12)
    ax3.set_title('BLR-H Error Distribution', fontsize=14)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")


def print_summary_table(summary):
    """결과 요약 테이블 출력"""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Unfreeze':<10} {'Best Value':<25} {'BLR-L Error':<20} {'BLR-H Error':<20}")
    print(f"{'Ratio':<10} {'Mean ± Std (Min)':<25} {'Mean ± Std':<20} {'Mean ± Std':<20}")
    print("-" * 80)

    for ratio in sorted(summary.keys()):
        s = summary[ratio]
        best_str = f"{s['best_so_far']['mean']:.3f}±{s['best_so_far']['std']:.3f} ({s['best_so_far']['min']:.3f})"
        blr_l_str = f"{s['blr_l_mean']['mean']:.3f}±{s['blr_l_mean']['std']:.3f}"
        blr_h_str = f"{s['blr_h_mean']['mean']:.3f}±{s['blr_h_mean']['std']:.3f}"
        print(f"{ratio:<10.1f} {best_str:<25} {blr_l_str:<20} {blr_h_str:<20}")

    print("=" * 80)
    print(f"Target bandgap: 1.34 eV (lower best_so_far = closer to target = better)")
    print("=" * 80)


def main():
    # 실험 설정
    UNFREEZE_RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    NUM_RUNS = 3  # 각 ratio당 실행 횟수
    COST_BUDGET = 20

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("Unfreeze Ratio Experiment")
    print("=" * 60)
    print(f"Ratios: {UNFREEZE_RATIOS}")
    print(f"Runs per ratio: {NUM_RUNS}")
    print(f"Cost budget: {COST_BUDGET}")
    print("=" * 60)

    # 데이터 로드 (한번만)
    print("\nLoading data...")
    with open('../../0.Data/lookup_table.pkl', 'rb') as f:
        lookup = pickle.load(f)

    from common.config import PARAM_SPACE
    from common.data_utils import create_label_maps
    label_maps = create_label_maps(PARAM_SPACE)
    print("Data loaded!")

    # 실험 실행 (순차)
    results = run_experiments_sequential(
        UNFREEZE_RATIOS,
        num_runs=NUM_RUNS,
        cost_budget=COST_BUDGET,
        lookup=lookup,
        label_maps=label_maps,
        PARAM_SPACE=PARAM_SPACE
    )

    # 결과 집계
    summary = aggregate_results(results)

    # 결과 출력
    print_summary_table(summary)

    # 결과 저장
    results_file = os.path.join(RESULTS_DIR, f'results_{timestamp}.json')
    with open(results_file, 'w') as f:
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump({
            'config': {
                'unfreeze_ratios': UNFREEZE_RATIOS,
                'num_runs': NUM_RUNS,
                'cost_budget': COST_BUDGET
            },
            'raw_results': results,
            'summary': {str(k): v for k, v in summary.items()}
        }, f, indent=2, default=convert)
    print(f"\nSaved results: {results_file}")

    # 시각화
    if len(summary) > 0:
        plot_file = os.path.join(RESULTS_DIR, f'comparison_{timestamp}.png')
        plot_results(summary, plot_file)

        boxplot_file = os.path.join(RESULTS_DIR, f'boxplot_{timestamp}.png')
        plot_detailed_results(summary, boxplot_file)

    print("\n✓ Experiment completed!")
    print(f"Results directory: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
