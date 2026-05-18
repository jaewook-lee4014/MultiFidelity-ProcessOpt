#!/usr/bin/env python
"""
Visualize MFGP Variants Benchmark Results

Creates trajectory plots showing regret vs budget for all 7 benchmarks
with 6 MFGP acquisition function variants.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# Results directory
results_dir = Path('mfgp_variants_20260204_210020')

# Load trajectory data
df = pd.read_csv(results_dir / 'results_trajectory.csv')

print(f"Loaded {len(df)} trajectory rows")
print(f"Benchmarks: {df['benchmark'].unique()}")
print(f"Models: {df['model'].unique()}")

benchmarks = ['Branin-Fav', 'Branin-Unfav', 'Park-Fav', 'Park-Unfav', 'COFs', 'FreeSolv', 'Polarizability']

# Model display names and colors
model_colors = {
    'MFGP-EI': '#1f77b4',        # Blue
    'MFGP-UCB': '#ff7f0e',       # Orange
    'MFGP-PI': '#2ca02c',        # Green
    'MFGP-EI-Cool': '#d62728',   # Red
    'MFGP-TS': '#9467bd',        # Purple
    'MFGP-UCB-Decay': '#8c564b', # Brown
}

models = list(model_colors.keys())

# Create figure: 2x4 grid (7 benchmarks + 1 legend)
fig, axes = plt.subplots(2, 4, figsize=(24, 11))
TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12
LEGEND_SIZE = 13

for idx, benchmark in enumerate(benchmarks):
    ax = axes[idx // 4, idx % 4]
    bench_data = df[df['benchmark'] == benchmark]

    for model in models:
        model_data = bench_data[bench_data['model'] == model]
        if len(model_data) == 0:
            continue

        seeds = model_data['seed'].unique()
        budget_grid = np.linspace(model_data['budget'].min(), model_data['budget'].max(), 100)

        interpolated = []
        for seed in seeds:
            sd = model_data[model_data['seed'] == seed].sort_values('budget')
            if len(sd) < 2:
                continue
            interpolated.append(np.interp(budget_grid, sd['budget'].values, sd['regret'].values))

        if not interpolated:
            continue

        interpolated = np.array(interpolated)
        mean_r = np.mean(interpolated, axis=0)
        se_r = np.std(interpolated, axis=0) / np.sqrt(len(interpolated))

        color = model_colors[model]
        ax.plot(budget_grid, mean_r, color=color, linewidth=1.5, alpha=0.9)
        ax.fill_between(budget_grid, np.maximum(mean_r - se_r, 1e-10), mean_r + se_r, color=color, alpha=0.15)

    ax.set_yscale('log')
    ax.set_title(benchmark, fontsize=TITLE_SIZE, fontweight='normal')
    ax.set_xlabel('Budget (Cost)', fontsize=LABEL_SIZE, fontweight='normal')
    ax.set_ylabel('Regret (Mean ± SE)', fontsize=LABEL_SIZE, fontweight='normal')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3, linewidth=0.5)

# Legend in the last subplot
ax_legend = axes[1, 3]
ax_legend.axis('off')

legend_handles = [Line2D([0], [0], color=model_colors[m], linewidth=2.5, label=m) for m in models]
ax_legend.legend(
    handles=legend_handles,
    loc='center',
    fontsize=LEGEND_SIZE,
    frameon=True,
    fancybox=False,
    edgecolor='gray',
    ncol=1,
    handlelength=2.5,
    labelspacing=1.0
)

plt.tight_layout(w_pad=3.0, h_pad=3.0)

# Save figures
out_png = results_dir / 'mfgp_variants_trajectory.png'
out_pdf = results_dir / 'mfgp_variants_trajectory.pdf'

fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_png} (300 DPI)")

fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_pdf} (vector)")

plt.close()

# =============================================================================
# Summary Statistics
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: Mean Final Regret by Model and Benchmark")
print("=" * 80)

df_summary = pd.read_csv(results_dir / 'results_summary.csv')

for benchmark in benchmarks:
    print(f"\n{benchmark}:")
    bench_data = df_summary[df_summary['benchmark'] == benchmark]
    summary = bench_data.groupby('model')['final_regret'].agg(['mean', 'std']).sort_values('mean')
    for model, row in summary.iterrows():
        print(f"  {model:<20}: {row['mean']:.4f} ± {row['std']:.4f}")

# Overall ranking
print("\n" + "=" * 80)
print("OVERALL RANKING (Mean regret across all benchmarks)")
print("=" * 80)
overall = df_summary.groupby('model')['final_regret'].mean().sort_values()
for i, (model, regret) in enumerate(overall.items(), 1):
    print(f"  {i}. {model:<20}: {regret:.4f}")

print("\nDone!")
