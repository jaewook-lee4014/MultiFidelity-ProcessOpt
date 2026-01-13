#!/usr/bin/env python
"""
대규모 Unfreeze Ratio 실험 스크립트
- 더 세밀한 unfreeze_ratio (0.1 간격)
- 하이퍼파라미터 BO 적용
- 각 설정당 10회 실행으로 안정적인 트렌드 확인
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


def run_single_experiment(unfreeze_ratio, random_state, cost_budget, use_hyperparameter_bo,
                          lookup, label_maps, PARAM_SPACE):
    """단일 실험 실행"""
    from DNGO.optimization import single_optimization_run_dngo_ol

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
            unfreeze_ratio=unfreeze_ratio,
            use_hyperparameter_bo=use_hyperparameter_bo,
            pretrain_bo_trials=5 if use_hyperparameter_bo else 0,
            finetune_bo_trials=5 if use_hyperparameter_bo else 0
        )

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
            'blr_h_mean': float(np.mean(np.abs(blr_h_errors))) if blr_h_errors else None,
            'success': True,
            'error': None
        }
    except Exception as e:
        import traceback
        return {
            'unfreeze_ratio': unfreeze_ratio,
            'random_state': random_state,
            'success': False,
            'error': str(e)
        }


def run_experiments(unfreeze_ratios, num_runs, cost_budget, use_hyperparameter_bo,
                    lookup, label_maps, PARAM_SPACE):
    """모든 실험 실행"""
    results = []
    total = len(unfreeze_ratios) * num_runs
    idx = 0

    for ratio in unfreeze_ratios:
        for run_idx in range(num_runs):
            random_state = 42 + run_idx
            idx += 1

            print(f"[{idx}/{total}] ratio={ratio:.1f}, seed={random_state}", end=" ... ")

            result = run_single_experiment(
                ratio, random_state, cost_budget, use_hyperparameter_bo,
                lookup, label_maps, PARAM_SPACE
            )
            results.append(result)

            if result['success']:
                print(f"Best={result['best_so_far']:.3f}, BLR-H={result['blr_h_mean']:.3f}")
            else:
                print("FAILED")

    return results


def aggregate_by_ratio(results):
    """ratio별 집계"""
    from collections import defaultdict
    aggregated = defaultdict(list)

    for r in results:
        if r['success']:
            aggregated[r['unfreeze_ratio']].append(r)

    summary = {}
    for ratio in sorted(aggregated.keys()):
        runs = aggregated[ratio]
        if len(runs) == 0:
            continue

        best_values = [r['best_so_far'] for r in runs]
        blr_l_values = [r['blr_l_mean'] for r in runs if r['blr_l_mean'] is not None]
        blr_h_values = [r['blr_h_mean'] for r in runs if r['blr_h_mean'] is not None]

        summary[ratio] = {
            'n_runs': len(runs),
            'best_so_far': {
                'mean': np.mean(best_values),
                'std': np.std(best_values),
                'sem': np.std(best_values) / np.sqrt(len(best_values)),  # Standard Error
                'min': np.min(best_values),
                'max': np.max(best_values),
                'median': np.median(best_values),
                'values': best_values
            },
            'blr_l_mean': {
                'mean': np.mean(blr_l_values) if blr_l_values else 0,
                'std': np.std(blr_l_values) if blr_l_values else 0,
                'sem': np.std(blr_l_values) / np.sqrt(len(blr_l_values)) if blr_l_values else 0,
            },
            'blr_h_mean': {
                'mean': np.mean(blr_h_values) if blr_h_values else 0,
                'std': np.std(blr_h_values) if blr_h_values else 0,
                'sem': np.std(blr_h_values) / np.sqrt(len(blr_h_values)) if blr_h_values else 0,
            }
        }
    return summary


def plot_results_with_trend(summary, save_path, title_suffix=""):
    """트렌드 라인이 있는 결과 시각화"""
    from scipy import stats

    ratios = sorted(summary.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 데이터 준비
    means = np.array([summary[r]['best_so_far']['mean'] for r in ratios])
    sems = np.array([summary[r]['best_so_far']['sem'] for r in ratios])
    mins = np.array([summary[r]['best_so_far']['min'] for r in ratios])
    medians = np.array([summary[r]['best_so_far']['median'] for r in ratios])

    blr_l_means = np.array([summary[r]['blr_l_mean']['mean'] for r in ratios])
    blr_l_sems = np.array([summary[r]['blr_l_mean']['sem'] for r in ratios])

    blr_h_means = np.array([summary[r]['blr_h_mean']['mean'] for r in ratios])
    blr_h_sems = np.array([summary[r]['blr_h_mean']['sem'] for r in ratios])

    ratios_arr = np.array(ratios)

    # 1. Best Value with SEM (Standard Error of Mean)
    ax1 = axes[0, 0]
    ax1.errorbar(ratios, means, yerr=sems, fmt='o-', capsize=5,
                 label='Mean ± SEM', markersize=8, linewidth=2, color='blue')
    ax1.fill_between(ratios, means - sems, means + sems, alpha=0.2, color='blue')

    # 트렌드 라인 (다항식 피팅)
    z = np.polyfit(ratios_arr, means, 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(0, 1, 100)
    ax1.plot(x_smooth, p(x_smooth), '--', color='red', linewidth=2,
             label=f'Trend (quadratic)', alpha=0.7)

    ax1.axhline(y=1.34, color='green', linestyle='--', linewidth=2, label='Target (1.34)')
    ax1.set_xlabel('Unfreeze Ratio', fontsize=12)
    ax1.set_ylabel('Best Value', fontsize=12)
    ax1.set_title(f'Best Value vs Unfreeze Ratio {title_suffix}', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(ratios)

    # 2. Box Plot for Best Value
    ax2 = axes[0, 1]
    data = [summary[r]['best_so_far']['values'] for r in ratios]
    bp = ax2.boxplot(data, positions=range(len(ratios)), widths=0.6, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax2.set_xticks(range(len(ratios)))
    ax2.set_xticklabels([f'{r:.1f}' for r in ratios])
    ax2.axhline(y=1.34, color='green', linestyle='--', linewidth=2, label='Target (1.34)')
    ax2.set_xlabel('Unfreeze Ratio', fontsize=12)
    ax2.set_ylabel('Best Value', fontsize=12)
    ax2.set_title(f'Best Value Distribution {title_suffix}', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. BLR-L Error with trend
    ax3 = axes[1, 0]
    ax3.errorbar(ratios, blr_l_means, yerr=blr_l_sems, fmt='s-', capsize=5,
                 color='blue', markersize=8, linewidth=2, label='Mean ± SEM')
    ax3.fill_between(ratios, blr_l_means - blr_l_sems, blr_l_means + blr_l_sems,
                     alpha=0.2, color='blue')

    # 트렌드 라인
    z = np.polyfit(ratios_arr, blr_l_means, 2)
    p = np.poly1d(z)
    ax3.plot(x_smooth, p(x_smooth), '--', color='red', linewidth=2,
             label='Trend', alpha=0.7)

    ax3.set_xlabel('Unfreeze Ratio', fontsize=12)
    ax3.set_ylabel('Mean Absolute Error', fontsize=12)
    ax3.set_title(f'BLR-L (Low-Fidelity) Error {title_suffix}', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(ratios)

    # 4. BLR-H Error with trend
    ax4 = axes[1, 1]
    ax4.errorbar(ratios, blr_h_means, yerr=blr_h_sems, fmt='^-', capsize=5,
                 color='orange', markersize=8, linewidth=2, label='Mean ± SEM')
    ax4.fill_between(ratios, blr_h_means - blr_h_sems, blr_h_means + blr_h_sems,
                     alpha=0.2, color='orange')

    # 트렌드 라인
    z = np.polyfit(ratios_arr, blr_h_means, 2)
    p = np.poly1d(z)
    ax4.plot(x_smooth, p(x_smooth), '--', color='red', linewidth=2,
             label='Trend', alpha=0.7)

    ax4.set_xlabel('Unfreeze Ratio', fontsize=12)
    ax4.set_ylabel('Mean Absolute Error', fontsize=12)
    ax4.set_title(f'BLR-H (High-Fidelity) Error {title_suffix}', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(ratios)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")


def print_summary_table(summary, title=""):
    """결과 요약 테이블"""
    print(f"\n{'='*90}")
    print(f"RESULTS SUMMARY {title}")
    print(f"{'='*90}")
    print(f"{'Ratio':<8} {'Best (Mean±SEM)':<20} {'Median':<10} {'Min':<10} {'BLR-L':<15} {'BLR-H':<15} {'N':<5}")
    print(f"{'-'*90}")

    for ratio in sorted(summary.keys()):
        s = summary[ratio]
        best_str = f"{s['best_so_far']['mean']:.3f}±{s['best_so_far']['sem']:.3f}"
        median_str = f"{s['best_so_far']['median']:.3f}"
        min_str = f"{s['best_so_far']['min']:.3f}"
        blr_l_str = f"{s['blr_l_mean']['mean']:.3f}±{s['blr_l_mean']['sem']:.3f}"
        blr_h_str = f"{s['blr_h_mean']['mean']:.3f}±{s['blr_h_mean']['sem']:.3f}"
        print(f"{ratio:<8.1f} {best_str:<20} {median_str:<10} {min_str:<10} {blr_l_str:<15} {blr_h_str:<15} {s['n_runs']:<5}")

    print(f"{'='*90}")
    print("Target: 1.34 eV | SEM = Standard Error of Mean")
    print(f"{'='*90}")


def main():
    # 실험 설정
    UNFREEZE_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    NUM_RUNS = 10
    COST_BUDGET = 20
    USE_HYPERPARAMETER_BO = True  # 하이퍼파라미터 BO 사용

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    total_experiments = len(UNFREEZE_RATIOS) * NUM_RUNS

    print("=" * 80)
    print("Large-Scale Unfreeze Ratio Experiment")
    print("=" * 80)
    print(f"Unfreeze ratios: {UNFREEZE_RATIOS}")
    print(f"Runs per ratio: {NUM_RUNS}")
    print(f"Cost budget: {COST_BUDGET}")
    print(f"Hyperparameter BO: {USE_HYPERPARAMETER_BO}")
    print(f"Total experiments: {total_experiments}")
    print("=" * 80)

    # 데이터 로드
    print("\nLoading data...")
    with open('../../0.Data/lookup_table.pkl', 'rb') as f:
        lookup = pickle.load(f)

    from common.config import PARAM_SPACE
    from common.data_utils import create_label_maps
    label_maps = create_label_maps(PARAM_SPACE)
    print("Data loaded!\n")

    # 실험 실행
    results = run_experiments(
        unfreeze_ratios=UNFREEZE_RATIOS,
        num_runs=NUM_RUNS,
        cost_budget=COST_BUDGET,
        use_hyperparameter_bo=USE_HYPERPARAMETER_BO,
        lookup=lookup,
        label_maps=label_maps,
        PARAM_SPACE=PARAM_SPACE
    )

    # 결과 집계
    summary = aggregate_by_ratio(results)

    # 결과 출력
    hp_str = "(with HP-BO)" if USE_HYPERPARAMETER_BO else "(no HP-BO)"
    print_summary_table(summary, hp_str)

    # 결과 저장
    results_file = os.path.join(RESULTS_DIR, f'large_scale_results_{timestamp}.json')
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
                'cost_budget': COST_BUDGET,
                'use_hyperparameter_bo': USE_HYPERPARAMETER_BO
            },
            'raw_results': results,
            'summary': {str(k): v for k, v in summary.items()}
        }, f, indent=2, default=convert)
    print(f"\nSaved results: {results_file}")

    # 시각화
    plot_file = os.path.join(RESULTS_DIR, f'large_scale_trend_{timestamp}.png')
    plot_results_with_trend(summary, plot_file, hp_str)

    print("\n✓ Experiment completed!")
    print(f"Results directory: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
