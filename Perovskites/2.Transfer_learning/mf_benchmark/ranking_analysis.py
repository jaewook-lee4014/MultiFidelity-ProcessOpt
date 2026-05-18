#!/usr/bin/env python
"""
Fidelity Ranking Analysis — ICML 2026 Reviewer Response

Quantifies the alignment between LF and HF rankings across all 7 benchmarks.
Answers the reviewer question: "If LF/HF top-k rankings are the same,
why do we need a surrogate model?"

Metrics computed:
  - Spearman ρ (rank correlation)
  - Kendall τ (pairwise concordance)
  - R² (coefficient of determination)
  - Top-k overlap at k = 1%, 5%, 10%, 20%
  - Top-k Kendall τ (within top-k union)
  - LF-screening regret: y_hf[argmin(y_lf)] - min(y_hf)
  - Top-1 hit: whether argmin(y_lf) == argmin(y_hf)

Outputs:
  - ranking_metrics.csv
  - rank_scatter.pdf (7-panel scatter plot)
  - topk_overlap.pdf (overlap curves)
  - ranking_table.tex (LaTeX table for paper)
  - topk_detail.csv (per-task top-10 candidates)

Usage:
    python ranking_analysis.py [--output-dir ranking_analysis_YYYYMMDD]
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import r2_score
from pathlib import Path
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

from benchmark_parallel import SyntheticBenchmark, ChemistryBenchmark
from synthetic_functions import branin_hf, branin_lf, park_hf, park_lf


def build_benchmarks():
    """Create all 7 benchmark objects."""
    data_dir = Path(__file__).parent / 'data'
    benchmarks = {}

    # Synthetic
    benchmarks['Branin-Fav'] = SyntheticBenchmark(
        'Branin-Fav', branin_hf, branin_lf, 2, 0.8, 0.1, 0.397887, 50)
    benchmarks['Branin-Unfav'] = SyntheticBenchmark(
        'Branin-Unfav', branin_hf, branin_lf, 2, 0.1, 0.5, 0.397887, 50)
    benchmarks['Park-Fav'] = SyntheticBenchmark(
        'Park-Fav', park_hf, park_lf, 4, 0.6, 0.1, 0.0, 10)
    benchmarks['Park-Unfav'] = SyntheticBenchmark(
        'Park-Unfav', park_hf, park_lf, 4, 0.0, 0.5, 0.0, 10)

    # Chemistry
    benchmarks['COFs'] = ChemistryBenchmark(
        'COFs', data_dir / 'cofs.csv', 0.065,
        use_smiles=False, minimize=True, negate=True)
    benchmarks['FreeSolv'] = ChemistryBenchmark(
        'FreeSolv', data_dir / 'freesolv.csv', 0.1,
        use_smiles=True, minimize=True, negate=False)
    benchmarks['Polarizability'] = ChemistryBenchmark(
        'Polarizability', data_dir / 'polarizability.csv', 0.167,
        use_smiles=True, minimize=True, negate=True)

    return benchmarks


def compute_metrics(y_hf, y_lf, name, cost_ratio):
    """Compute all ranking metrics for one benchmark."""
    n = len(y_hf)

    # Ranks (1-based, ascending: rank 1 = best = lowest value for minimization)
    rank_hf = np.argsort(np.argsort(y_hf)) + 1
    rank_lf = np.argsort(np.argsort(y_lf)) + 1

    # Correlation metrics
    sp_rho, sp_p = spearmanr(y_hf, y_lf)
    kt_tau, kt_p = kendalltau(y_hf, y_lf)
    r2 = r2_score(y_hf, y_lf) if np.std(y_hf) > 0 else 0.0

    # Top-k overlap at various fractions (percentage-based)
    k_fractions = [0.01, 0.05, 0.10, 0.20]
    overlaps = {}
    topk_taus = {}
    for frac in k_fractions:
        k = max(1, int(frac * n))
        top_hf = set(np.argsort(y_hf)[:k])
        top_lf = set(np.argsort(y_lf)[:k])
        overlap = len(top_hf & top_lf) / k
        overlaps[f'top{int(frac*100)}%_overlap'] = overlap

        # Kendall tau within top-k union
        union = sorted(top_hf | top_lf)
        if len(union) > 1:
            tau_k, _ = kendalltau(y_hf[union], y_lf[union])
            topk_taus[f'top{int(frac*100)}%_tau'] = tau_k
        else:
            topk_taus[f'top{int(frac*100)}%_tau'] = np.nan

    # Top-k overlap at absolute counts (k = 5, 10, 20, 50)
    k_absolutes = [5, 10, 20, 50]
    abs_overlaps = {}
    for k in k_absolutes:
        if k > n:
            abs_overlaps[f'topk{k}_overlap'] = np.nan
            continue
        top_hf = set(np.argsort(y_hf)[:k])
        top_lf = set(np.argsort(y_lf)[:k])
        abs_overlaps[f'topk{k}_overlap'] = len(top_hf & top_lf) / k

    # LF-screening regret
    best_lf_idx = np.argmin(y_lf)
    best_hf_idx = np.argmin(y_hf)
    screening_regret = y_hf[best_lf_idx] - y_hf[best_hf_idx]
    top1_hit = int(best_lf_idx == best_hf_idx)

    return {
        'benchmark': name,
        'n_candidates': n,
        'cost_ratio': cost_ratio,
        'spearman_rho': sp_rho,
        'spearman_p': sp_p,
        'kendall_tau': kt_tau,
        'kendall_p': kt_p,
        'R2': r2,
        **overlaps,
        **abs_overlaps,
        **topk_taus,
        'screening_regret': screening_regret,
        'top1_hit': top1_hit,
        'hf_best_idx': int(best_hf_idx),
        'lf_best_idx': int(best_lf_idx),
        'hf_best_val': y_hf[best_hf_idx],
        'lf_rank_of_hf_best': int(rank_lf[best_hf_idx]),
    }


def plot_rank_scatter(benchmarks, metrics_df, output_dir):
    """7-panel scatter plot: LF rank vs HF rank, with top-5% region highlighted."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    bench_order = ['Branin-Fav', 'Branin-Unfav', 'Park-Fav', 'Park-Unfav',
                   'COFs', 'FreeSolv', 'Polarizability']

    for i, name in enumerate(bench_order):
        ax = axes[i]
        bench = benchmarks[name]
        y_hf, y_lf = bench.y_hf, bench.y_lf
        n = len(y_hf)

        rank_hf = np.argsort(np.argsort(y_hf)) + 1
        rank_lf = np.argsort(np.argsort(y_lf)) + 1

        # Top 5% threshold
        k5 = max(1, int(0.05 * n))

        # Color: blue for top-5% HF, gray otherwise
        colors = np.where(rank_hf <= k5, '#e74c3c', '#bdc3c7')
        sizes = np.where(rank_hf <= k5, 20, 5)

        ax.scatter(rank_lf, rank_hf, c=colors, s=sizes, alpha=0.6, edgecolors='none')

        # Highlight top-5% region
        ax.add_patch(Rectangle((0.5, 0.5), k5, k5, linewidth=1.5,
                               edgecolor='#e74c3c', facecolor='#e74c3c', alpha=0.08))

        # Diagonal
        ax.plot([1, n], [1, n], 'k--', alpha=0.3, linewidth=0.8)

        # Metrics annotation
        row = metrics_df[metrics_df['benchmark'] == name].iloc[0]
        topk10_str = f"{row['topk10_overlap']:.0%}" if not pd.isna(row['topk10_overlap']) else '--'
        ax.text(0.97, 0.03,
                f"ρ={row['spearman_rho']:.2f}\n"
                f"τ={row['kendall_tau']:.2f}\n"
                f"R²={row['R2']:.2f}\n"
                f"top10={topk10_str}",
                transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_title(f'{name} (n={n})', fontsize=11, fontweight='bold')
        ax.set_xlabel('LF Rank')
        ax.set_ylabel('HF Rank')
        ax.set_xlim(0.5, n + 0.5)
        ax.set_ylim(0.5, n + 0.5)

    # Hide unused subplot
    axes[-1].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'rank_scatter.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'rank_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved rank_scatter.pdf/png")


def plot_topk_overlap(benchmarks, output_dir):
    """Top-k overlap curves for all 7 benchmarks."""
    fig, ax = plt.subplots(figsize=(8, 5))

    bench_order = ['Branin-Fav', 'Branin-Unfav', 'Park-Fav', 'Park-Unfav',
                   'COFs', 'FreeSolv', 'Polarizability']
    colors = plt.cm.tab10(np.linspace(0, 1, len(bench_order)))

    k_fractions = np.linspace(0.01, 0.50, 50)

    for idx, name in enumerate(bench_order):
        bench = benchmarks[name]
        y_hf, y_lf = bench.y_hf, bench.y_lf
        n = len(y_hf)

        overlaps = []
        for frac in k_fractions:
            k = max(1, int(frac * n))
            top_hf = set(np.argsort(y_hf)[:k])
            top_lf = set(np.argsort(y_lf)[:k])
            overlaps.append(len(top_hf & top_lf) / k)

        ax.plot(k_fractions * 100, overlaps, '-', color=colors[idx],
                label=name, linewidth=2, alpha=0.8)

    # Random baseline (expected overlap = k/n ≈ fraction)
    ax.plot(k_fractions * 100, k_fractions, 'k--', alpha=0.4, label='Random', linewidth=1)

    ax.set_xlabel('Top-k Fraction (%)', fontsize=12)
    ax.set_ylabel('Overlap (LF top-k ∩ HF top-k) / k', fontsize=12)
    ax.set_title('Fidelity Ranking Alignment: Top-k Overlap', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'topk_overlap.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'topk_overlap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved topk_overlap.pdf/png")


def generate_latex_table(metrics_df, output_dir):
    """Generate LaTeX table for paper insertion (absolute top-k focus)."""
    df = metrics_df.copy()

    tex_lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Fidelity ranking alignment across benchmarks. '
        r'Top-$k$ overlap measures the fraction of the $k$ best HF candidates '
        r'that also appear in the $k$ best LF candidates. '
        r'Screening regret is $f_\mathrm{HF}[\arg\min f_\mathrm{LF}] - f^*_\mathrm{HF}$.}',
        r'\label{tab:ranking}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{l r r r r r r r r r r r c}',
        r'\toprule',
        r'Benchmark & $n$ & $\rho_\mathrm{cost}$ & $R^2$ & $\rho_s$ & $\tau$ '
        r'& Top-5 & Top-10 & Top-20 & Top-50 & Top-5\% & Screen.\ Regret & Top-1 Hit \\',
        r'\midrule',
    ]

    for _, row in df.iterrows():
        top1_str = r'\checkmark' if row['top1_hit'] else r'$\times$'
        # Handle NaN for small pools (Park n=256: top-50 is valid but be safe)
        def fmt_overlap(val):
            if pd.isna(val):
                return '--'
            return f'{val:.0%}'

        tex_lines.append(
            f"  {row['benchmark']} & {int(row['n_candidates'])} & {row['cost_ratio']:.3f} "
            f"& {row['R2']:.2f} & {row['spearman_rho']:.2f} & {row['kendall_tau']:.2f} "
            f"& {fmt_overlap(row['topk5_overlap'])} & {fmt_overlap(row['topk10_overlap'])} "
            f"& {fmt_overlap(row['topk20_overlap'])} & {fmt_overlap(row['topk50_overlap'])} "
            f"& {fmt_overlap(row['top5%_overlap'])} "
            f"& {row['screening_regret']:.4f} "
            f"& {top1_str} \\\\"
        )

    tex_lines.extend([
        r'\bottomrule',
        r'\end{tabular}}',
        r'\end{table}',
    ])

    tex_content = '\n'.join(tex_lines)
    (output_dir / 'ranking_table.tex').write_text(tex_content)
    print(f"  Saved ranking_table.tex")


def generate_topk_detail(benchmarks, output_dir):
    """Per-task top-10 candidate details (LF rank, HF rank, values)."""
    rows = []
    for name, bench in benchmarks.items():
        y_hf, y_lf = bench.y_hf, bench.y_lf
        rank_hf = np.argsort(np.argsort(y_hf)) + 1
        rank_lf = np.argsort(np.argsort(y_lf)) + 1

        # Top 10 by HF
        top10_hf = np.argsort(y_hf)[:10]
        for idx in top10_hf:
            rows.append({
                'benchmark': name,
                'candidate_idx': int(idx),
                'hf_rank': int(rank_hf[idx]),
                'lf_rank': int(rank_lf[idx]),
                'hf_value': y_hf[idx],
                'lf_value': y_lf[idx],
                'rank_gap': abs(int(rank_hf[idx]) - int(rank_lf[idx])),
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'topk_detail.csv', index=False)
    print(f"  Saved topk_detail.csv")


def main():
    parser = argparse.ArgumentParser(description='Fidelity Ranking Analysis')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f'ranking_analysis_{timestamp}')
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("Fidelity Ranking Analysis")
    print("=" * 70)
    print(f"Output: {output_dir}")

    # Build benchmarks
    print("\nBuilding benchmarks...")
    benchmarks = build_benchmarks()

    # Compute metrics
    print("Computing ranking metrics...")
    metrics = []
    for name, bench in benchmarks.items():
        m = compute_metrics(bench.y_hf, bench.y_lf, name, bench.cost_ratio)
        metrics.append(m)
        print(f"  {name}: ρ_s={m['spearman_rho']:.3f}, τ={m['kendall_tau']:.3f}, "
              f"R²={m['R2']:.3f}, top10={m['topk10_overlap']:.1%}, "
              f"top5%={m['top5%_overlap']:.1%}, "
              f"screen_regret={m['screening_regret']:.4f}")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_dir / 'ranking_metrics.csv', index=False)
    print(f"\n  Saved ranking_metrics.csv")

    # Plots
    print("\nGenerating plots...")
    plot_rank_scatter(benchmarks, metrics_df, output_dir)
    plot_topk_overlap(benchmarks, output_dir)

    # LaTeX table
    print("\nGenerating LaTeX table...")
    generate_latex_table(metrics_df, output_dir)

    # Top-k detail
    print("\nGenerating top-k detail...")
    generate_topk_detail(benchmarks, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary_cols = ['benchmark', 'n_candidates', 'spearman_rho', 'kendall_tau', 'R2',
                     'topk5_overlap', 'topk10_overlap', 'topk20_overlap', 'topk50_overlap',
                     'top5%_overlap', 'screening_regret', 'top1_hit']
    print(metrics_df[summary_cols].to_string(index=False))
    print("=" * 70)

    # Key insight based on absolute top-k
    low_top10 = metrics_df[metrics_df['topk10_overlap'] < 0.5]
    high_top10 = metrics_df[metrics_df['topk10_overlap'] >= 0.8]
    print(f"\nTop-10 overlap >= 80%: {', '.join(high_top10['benchmark'].tolist()) or 'None'}")
    print(f"Top-10 overlap <  50%: {', '.join(low_top10['benchmark'].tolist()) or 'None'}")
    print("\nKey finding: Even when global ρ_s is high, the top-k overlap at small k")
    print("can be poor — justifying surrogate models that learn local ranking structure")
    print("beyond what simple LF screening provides.")


if __name__ == '__main__':
    main()
