#!/usr/bin/env python
"""
Multiple seed experiments for LF-BLR comparison

Runs compare_lf_blr.py with multiple seeds and aggregates results.
Visualizes regret distribution across seeds for both scenarios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import subprocess
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import sys

# Import from compare_lf_blr
sys.path.insert(0, str(Path(__file__).parent))
from compare_lf_blr import run_comparison


def run_single_experiment(args):
    """Run a single experiment with given seed and scenario"""
    seed, scenario, budget = args

    try:
        results = run_comparison(seed, budget, save_interval=100, scenario_name=scenario)

        # Extract final regrets
        comparison_data = results['comparison_data']
        model_keys = results['model_keys']
        f_star = results['f_star']

        # Get final regret from the last comparison data entry
        if len(comparison_data) > 0:
            final = comparison_data[-1]
            final_regrets = {k: final.get(f'{k}_regret', np.nan) for k in model_keys}
        else:
            final_regrets = {k: np.nan for k in model_keys}

        return {
            'seed': seed,
            'scenario': scenario,
            **final_regrets
        }

    except Exception as e:
        print(f"  Error: seed={seed}, scenario={scenario}: {e}")
        return {
            'seed': seed,
            'scenario': scenario,
            'mfgp': np.nan,
            'seq_vanilla': np.nan,
            'seq_lf_blr': np.nan,
            'joint_vanilla': np.nan
        }


def run_experiments(n_seeds: int = 10, budget: float = 50.0,
                    base_seed: int = 1004, parallel: bool = False):
    """Run experiments with multiple seeds for both scenarios"""

    seeds = [base_seed + i for i in range(n_seeds)]
    scenarios = ['favorable', 'unfavorable']

    all_results = []

    # Create task list
    tasks = [(seed, scenario, budget) for seed in seeds for scenario in scenarios]

    print(f"Running {len(tasks)} experiments ({n_seeds} seeds x 2 scenarios)")
    print(f"Seeds: {seeds}")
    print(f"Budget: {budget}")
    print("=" * 60)

    if parallel:
        # Parallel execution with spawn context for CUDA compatibility
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as executor:
            futures = {executor.submit(run_single_experiment, task): task for task in tasks}

            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                print(f"  Completed: seed={result['seed']}, scenario={result['scenario']}")
    else:
        # Sequential execution
        for i, task in enumerate(tasks):
            seed, scenario, _ = task
            print(f"\n[{i+1}/{len(tasks)}] seed={seed}, scenario={scenario}")
            result = run_single_experiment(task)
            all_results.append(result)

    return pd.DataFrame(all_results)


def visualize_results(df: pd.DataFrame, output_dir: Path):
    """Visualize regret distribution across seeds"""

    model_keys = ['mfgp', 'seq_vanilla', 'seq_lf_blr', 'joint_vanilla']
    model_names = ['MFGP', 'Seq-Vanilla', 'Seq-LF-BLR', 'Joint-Vanilla']
    colors = ['green', 'blue', 'red', 'purple']

    scenarios = ['favorable', 'unfavorable']

    # Figure 1: Box plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, scenario in enumerate(scenarios):
        ax = axes[ax_idx]
        df_scenario = df[df['scenario'] == scenario]

        data = [df_scenario[k].dropna().values for k in model_keys]

        bp = ax.boxplot(data, labels=model_names, patch_artist=True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel('Final Regret')
        ax.set_title(f'{scenario.upper()} Scenario')
        ax.grid(True, alpha=0.3)

        # Add individual points
        for i, (k, c) in enumerate(zip(model_keys, colors)):
            y = df_scenario[k].dropna().values
            x = np.random.normal(i + 1, 0.04, size=len(y))
            ax.scatter(x, y, c=c, alpha=0.6, s=30, edgecolors='black', linewidths=0.5)

    fig.suptitle('Final Regret Distribution (Multiple Seeds)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / 'regret_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: regret_boxplot.png")

    # Figure 2: Bar plot with error bars (mean ± std)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, scenario in enumerate(scenarios):
        ax = axes[ax_idx]
        df_scenario = df[df['scenario'] == scenario]

        means = [df_scenario[k].mean() for k in model_keys]
        stds = [df_scenario[k].std() for k in model_keys]

        x = np.arange(len(model_names))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylabel('Final Regret (mean ± std)')
        ax.set_title(f'{scenario.upper()} Scenario')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                    f'{mean:.4f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Mean Final Regret with Standard Deviation', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / 'regret_barplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: regret_barplot.png")

    # Figure 3: Violin plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, scenario in enumerate(scenarios):
        ax = axes[ax_idx]
        df_scenario = df[df['scenario'] == scenario]

        data = [df_scenario[k].dropna().values for k in model_keys]

        parts = ax.violinplot(data, positions=range(1, len(model_keys)+1),
                              showmeans=True, showmedians=True)

        for i, (pc, color) in enumerate(zip(parts['bodies'], colors)):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        ax.set_xticks(range(1, len(model_keys)+1))
        ax.set_xticklabels(model_names)
        ax.set_ylabel('Final Regret')
        ax.set_title(f'{scenario.upper()} Scenario')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Final Regret Distribution (Violin Plot)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / 'regret_violin.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: regret_violin.png")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    for scenario in scenarios:
        print(f"\n{scenario.upper()} Scenario:")
        df_scenario = df[df['scenario'] == scenario]

        for k, name in zip(model_keys, model_names):
            values = df_scenario[k].dropna()
            print(f"  {name:15s}: mean={values.mean():.6f}, std={values.std():.6f}, "
                  f"min={values.min():.6f}, max={values.max():.6f}")


def main():
    parser = argparse.ArgumentParser(description='Run multiple seed experiments')
    parser.add_argument('--n-seeds', type=int, default=10, help='Number of seeds')
    parser.add_argument('--base-seed', type=int, default=1004, help='Base seed')
    parser.add_argument('--budget', type=float, default=50, help='Total budget')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'multi_seed_results_{timestamp}')
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Multiple Seed Experiment")
    print("=" * 80)
    print(f"Number of seeds: {args.n_seeds}")
    print(f"Base seed: {args.base_seed}")
    print(f"Budget: {args.budget}")
    print(f"Parallel: {args.parallel}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    # Run experiments
    df = run_experiments(
        n_seeds=args.n_seeds,
        budget=args.budget,
        base_seed=args.base_seed,
        parallel=args.parallel
    )

    # Save raw results
    df.to_csv(output_dir / 'raw_results.csv', index=False)
    print(f"\nSaved: raw_results.csv")

    # Visualize
    print("\nGenerating visualizations...")
    visualize_results(df, output_dir)

    print("\n" + "=" * 80)
    print("Done!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
