#!/usr/bin/env python
"""
Visualize Adaptive vs Round-Robin fidelity selection results.

4 panels:
1. Delta-regret heatmap: (model x benchmark), color = regret(adaptive) - regret(round-robin)
2. Budget vs Regret trajectory: adaptive vs round-robin curves (median + IQR)
3. Fidelity allocation: adaptive LF:HF ratio vs round-robin fixed ratio
4. Paired Wilcoxon test: (model, benchmark) p-value + effect size

Usage:
    python plot_adaptive_vs_roundrobin.py \
        --adaptive-dir benchmark_adaptive_YYYYMMDD_HHMMSS \
        --roundrobin-dir benchmark_lf_blr_YYYYMMDD_HHMMSS \
        --output-dir plots_adaptive_comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from pathlib import Path
from scipy.stats import wilcoxon
import argparse
import warnings
warnings.filterwarnings('ignore')


def load_results(result_dir):
    """Load summary and trajectory CSVs from a benchmark output directory."""
    result_dir = Path(result_dir)
    summary_file = result_dir / 'results_summary.csv'
    trajectory_file = result_dir / 'results_trajectory.csv'

    df_summary = pd.read_csv(summary_file)
    df_trajectory = pd.read_csv(trajectory_file)

    return df_summary, df_trajectory


def plot_delta_regret_heatmap(df_adaptive, df_roundrobin, output_dir):
    """Panel 1: Heatmap of regret(adaptive) - regret(round-robin)"""
    # Compute mean regret per (benchmark, model)
    adapt_mean = df_adaptive.groupby(['benchmark', 'model'])['final_regret'].mean().reset_index()
    adapt_mean.columns = ['benchmark', 'model', 'regret_adaptive']

    rr_mean = df_roundrobin.groupby(['benchmark', 'model'])['final_regret'].mean().reset_index()
    rr_mean.columns = ['benchmark', 'model', 'regret_roundrobin']

    merged = pd.merge(adapt_mean, rr_mean, on=['benchmark', 'model'], how='inner')
    merged['delta'] = merged['regret_adaptive'] - merged['regret_roundrobin']

    benchmarks = sorted(merged['benchmark'].unique())
    models = sorted(merged['model'].unique())

    # Create pivot table
    pivot = merged.pivot(index='model', columns='benchmark', values='delta')
    pivot = pivot.reindex(index=models, columns=benchmarks)

    fig, ax = plt.subplots(figsize=(max(10, len(benchmarks) * 1.5), max(6, len(models) * 0.5)))

    # Symmetric colormap centered at 0
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    if vmax == 0:
        vmax = 0.1

    im = ax.imshow(pivot.values, cmap='RdBu_r', aspect='auto',
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(benchmarks)))
    ax.set_xticklabels(benchmarks, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(benchmarks)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Regret(Adaptive) - Regret(Round-Robin)', fontsize=10)

    ax.set_title('Delta Regret: Adaptive vs Round-Robin\n(Blue = Adaptive better, Red = Round-Robin better)',
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / 'delta_regret_heatmap.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'delta_regret_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved delta_regret_heatmap.pdf/png")


def plot_trajectory_comparison(df_traj_adaptive, df_traj_roundrobin, output_dir):
    """Panel 2: Budget vs Regret trajectory for key models"""
    benchmarks = sorted(set(df_traj_adaptive['benchmark'].unique()) &
                        set(df_traj_roundrobin['benchmark'].unique()))
    models = sorted(set(df_traj_adaptive['model'].unique()) &
                    set(df_traj_roundrobin['model'].unique()))

    # Select up to 5 key models for readability
    priority_models = ['MFGP', 'Sequential', 'Curriculum', 'DNGO-Joint', 'NARGP']
    plot_models = [m for m in priority_models if m in models]
    if len(plot_models) < len(models):
        for m in models:
            if m not in plot_models:
                plot_models.append(m)
                if len(plot_models) >= 5:
                    break

    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_models)))

    n_bench = len(benchmarks)
    n_cols = min(3, n_bench)
    n_rows = (n_bench + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows),
                             squeeze=False)

    for b_idx, bench in enumerate(benchmarks):
        ax = axes[b_idx // n_cols][b_idx % n_cols]

        for m_idx, model in enumerate(plot_models):
            color = colors[m_idx]

            for strategy, df_traj, ls in [('adaptive', df_traj_adaptive, '-'),
                                           ('round-robin', df_traj_roundrobin, '--')]:
                mask = (df_traj['benchmark'] == bench) & (df_traj['model'] == model)
                data = df_traj[mask]
                if data.empty:
                    continue

                # Group by seed, interpolate to common budget grid
                seeds = data['seed'].unique()
                budget_max = data['budget'].max()
                budget_grid = np.linspace(0, budget_max, 50)

                all_regrets = []
                for seed in seeds:
                    seed_data = data[data['seed'] == seed].sort_values('budget')
                    if len(seed_data) > 1:
                        regret_interp = np.interp(budget_grid, seed_data['budget'].values,
                                                  seed_data['regret'].values)
                        all_regrets.append(regret_interp)

                if not all_regrets:
                    continue

                all_regrets = np.array(all_regrets)
                median = np.median(all_regrets, axis=0)
                q25 = np.percentile(all_regrets, 25, axis=0)
                q75 = np.percentile(all_regrets, 75, axis=0)

                label = f'{model} ({strategy})' if strategy == 'adaptive' else None
                ax.plot(budget_grid, median, ls=ls, color=color, linewidth=1.5,
                        label=f'{model}' if strategy == 'adaptive' else None)
                ax.fill_between(budget_grid, q25, q75, alpha=0.1, color=color)

        ax.set_xlabel('Budget', fontsize=10)
        ax.set_ylabel('Regret', fontsize=10)
        ax.set_title(bench, fontsize=11)
        if b_idx == 0:
            ax.legend(fontsize=7, loc='upper right')

    # Turn off unused axes
    for idx in range(n_bench, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle('Regret Trajectories: Adaptive (solid) vs Round-Robin (dashed)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'trajectory_comparison.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'trajectory_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved trajectory_comparison.pdf/png")


def plot_fidelity_allocation(adaptive_dir, df_summary_adaptive, df_summary_roundrobin, output_dir):
    """Panel 3: Fidelity allocation comparison"""
    fidelity_file = Path(adaptive_dir) / 'fidelity_decisions.csv'
    if not fidelity_file.exists():
        print("  Skipping fidelity allocation (no fidelity_decisions.csv)")
        return

    df_fid = pd.read_csv(fidelity_file)

    benchmarks = sorted(df_fid['benchmark'].unique())
    models = sorted(df_fid['model'].unique())

    # Compute HF ratio for adaptive
    adapt_hf_ratio = df_fid.groupby(['benchmark', 'model'])['fidelity_chosen'].mean()

    # Compute HF ratio for round-robin from summary
    rr_summary = df_summary_roundrobin.copy()
    rr_summary['hf_ratio'] = rr_summary['n_hf'] / (rr_summary['n_hf'] + rr_summary['n_lf'])
    rr_hf_ratio = rr_summary.groupby(['benchmark', 'model'])['hf_ratio'].mean()

    n_bench = len(benchmarks)
    fig, axes = plt.subplots(1, n_bench, figsize=(4 * n_bench, 5), squeeze=False)

    for b_idx, bench in enumerate(benchmarks):
        ax = axes[0][b_idx]

        models_in_bench = sorted(set(
            df_fid[df_fid['benchmark'] == bench]['model'].unique()
        ) & set(
            rr_summary[rr_summary['benchmark'] == bench]['model'].unique()
        ))

        x = np.arange(len(models_in_bench))
        width = 0.35

        adapt_vals = []
        rr_vals = []
        for m in models_in_bench:
            key = (bench, m)
            adapt_vals.append(adapt_hf_ratio.get(key, 0.0))
            rr_vals.append(rr_hf_ratio.get(key, 0.0))

        ax.bar(x - width/2, adapt_vals, width, label='Adaptive', color='tab:blue', alpha=0.8)
        ax.bar(x + width/2, rr_vals, width, label='Round-Robin', color='tab:orange', alpha=0.8)

        ax.set_xlabel('Model', fontsize=9)
        ax.set_ylabel('HF Evaluation Ratio', fontsize=9)
        ax.set_title(bench, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models_in_bench, rotation=45, ha='right', fontsize=7)
        ax.set_ylim(0, 1)
        if b_idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle('Fidelity Allocation: HF Ratio (Adaptive vs Round-Robin)', fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'fidelity_allocation.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fidelity_allocation.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved fidelity_allocation.pdf/png")


def plot_wilcoxon_tests(df_adaptive, df_roundrobin, output_dir):
    """Panel 4: Paired Wilcoxon signed-rank test results"""
    results = []

    benchmarks = sorted(set(df_adaptive['benchmark'].unique()) &
                        set(df_roundrobin['benchmark'].unique()))
    models = sorted(set(df_adaptive['model'].unique()) &
                    set(df_roundrobin['model'].unique()))

    for bench in benchmarks:
        for model in models:
            adapt_data = df_adaptive[
                (df_adaptive['benchmark'] == bench) & (df_adaptive['model'] == model)
            ].sort_values('seed')
            rr_data = df_roundrobin[
                (df_roundrobin['benchmark'] == bench) & (df_roundrobin['model'] == model)
            ].sort_values('seed')

            # Match seeds
            common_seeds = sorted(set(adapt_data['seed']) & set(rr_data['seed']))
            if len(common_seeds) < 5:
                continue

            adapt_regrets = adapt_data[adapt_data['seed'].isin(common_seeds)].sort_values('seed')['final_regret'].values
            rr_regrets = rr_data[rr_data['seed'].isin(common_seeds)].sort_values('seed')['final_regret'].values

            # Remove NaN pairs
            valid = ~(np.isnan(adapt_regrets) | np.isnan(rr_regrets))
            adapt_regrets = adapt_regrets[valid]
            rr_regrets = rr_regrets[valid]

            if len(adapt_regrets) < 5:
                continue

            diff = adapt_regrets - rr_regrets
            if np.all(diff == 0):
                p_val = 1.0
                effect = 0.0
            else:
                try:
                    stat, p_val = wilcoxon(adapt_regrets, rr_regrets)
                    n = len(adapt_regrets)
                    effect = stat / (n * (n + 1) / 2)  # rank-biserial
                except Exception:
                    p_val = 1.0
                    effect = 0.0

            results.append({
                'benchmark': bench,
                'model': model,
                'p_value': p_val,
                'effect_size': effect,
                'mean_adapt': np.mean(adapt_regrets),
                'mean_rr': np.mean(rr_regrets),
                'n_seeds': len(adapt_regrets),
                'adapt_wins': np.sum(adapt_regrets < rr_regrets),
                'rr_wins': np.sum(rr_regrets < adapt_regrets),
            })

    if not results:
        print("  Skipping Wilcoxon test (insufficient overlapping data)")
        return

    df_results = pd.DataFrame(results)

    # Save table
    df_results.to_csv(output_dir / 'wilcoxon_results.csv', index=False)

    # Create heatmap of p-values
    pivot_p = df_results.pivot(index='model', columns='benchmark', values='p_value')
    pivot_dir = df_results.copy()
    pivot_dir['direction'] = np.where(
        pivot_dir['mean_adapt'] < pivot_dir['mean_rr'], -1,
        np.where(pivot_dir['mean_adapt'] > pivot_dir['mean_rr'], 1, 0)
    )
    pivot_direction = pivot_dir.pivot(index='model', columns='benchmark', values='direction')

    fig, ax = plt.subplots(figsize=(max(10, len(benchmarks) * 1.5),
                                     max(6, len(models) * 0.5)))

    # Color: green if adaptive wins (p<0.05), red if RR wins, white if n.s.
    data = np.full_like(pivot_p.values, fill_value=0.5, dtype=float)
    for i in range(pivot_p.shape[0]):
        for j in range(pivot_p.shape[1]):
            p = pivot_p.iloc[i, j]
            d = pivot_direction.iloc[i, j]
            if np.isnan(p):
                data[i, j] = 0.5
            elif p < 0.05:
                data[i, j] = 0.0 if d < 0 else 1.0  # green vs red
            else:
                data[i, j] = 0.5

    # Custom colormap: green (adaptive) - white (n.s.) - red (round-robin)
    cmap = mcolors.LinearSegmentedColormap.from_list('grn_wht_red',
                                                      ['#2ca02c', 'white', '#d62728'])
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot_p.columns)))
    ax.set_xticklabels(pivot_p.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(pivot_p.index)))
    ax.set_yticklabels(pivot_p.index, fontsize=9)

    # Annotate
    for i in range(pivot_p.shape[0]):
        for j in range(pivot_p.shape[1]):
            p = pivot_p.iloc[i, j]
            if np.isnan(p):
                ax.text(j, i, 'N/A', ha='center', va='center', fontsize=7, color='gray')
            else:
                sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
                ax.text(j, i, f'p={p:.3f}\n{sig}', ha='center', va='center', fontsize=7)

    ax.set_title('Paired Wilcoxon Test: Adaptive vs Round-Robin\n(Green = Adaptive better, Red = Round-Robin better)',
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / 'wilcoxon_heatmap.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'wilcoxon_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved wilcoxon_heatmap.pdf/png")
    print(f"  Saved wilcoxon_results.csv")


def main():
    parser = argparse.ArgumentParser(description='Plot Adaptive vs Round-Robin comparison')
    parser.add_argument('--adaptive-dir', type=str, required=True,
                        help='Directory with adaptive benchmark results')
    parser.add_argument('--roundrobin-dir', type=str, required=True,
                        help='Directory with round-robin (lf_blr) benchmark results')
    parser.add_argument('--output-dir', type=str, default='plots_adaptive_comparison',
                        help='Output directory for plots')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("Loading results...")
    df_adapt_summary, df_adapt_traj = load_results(args.adaptive_dir)
    df_rr_summary, df_rr_traj = load_results(args.roundrobin_dir)

    print(f"  Adaptive: {len(df_adapt_summary)} summary rows, {len(df_adapt_traj)} trajectory rows")
    print(f"  Round-Robin: {len(df_rr_summary)} summary rows, {len(df_rr_traj)} trajectory rows")

    print("\nGenerating plots...")

    print("\n1. Delta-regret heatmap")
    plot_delta_regret_heatmap(df_adapt_summary, df_rr_summary, output_dir)

    print("\n2. Trajectory comparison")
    plot_trajectory_comparison(df_adapt_traj, df_rr_traj, output_dir)

    print("\n3. Fidelity allocation")
    plot_fidelity_allocation(args.adaptive_dir, df_adapt_summary, df_rr_summary, output_dir)

    print("\n4. Wilcoxon test")
    plot_wilcoxon_tests(df_adapt_summary, df_rr_summary, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
