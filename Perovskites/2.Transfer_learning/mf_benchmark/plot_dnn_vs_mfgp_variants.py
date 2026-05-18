#!/usr/bin/env python
"""
Combined Visualization: DNN Models vs MFGP Variants

Plots all DNN models (one color) and all MFGP variants (another color)
to compare the two families of approaches.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# Load DNN benchmark data (includes one MFGP)
dnn_dir = Path('benchmark_parallel_20260122_154131')
df_dnn = pd.read_csv(dnn_dir / 'results_trajectory.csv')

# Load MFGP variants data
mfgp_dir = Path('mfgp_variants_20260204_210020')
df_mfgp = pd.read_csv(mfgp_dir / 'results_trajectory.csv')

# Remove the single MFGP from DNN data (we have variants now)
df_dnn = df_dnn[df_dnn['model'] != 'MFGP']

# Combine datasets
df = pd.concat([df_dnn, df_mfgp], ignore_index=True)

print(f"DNN models: {df_dnn['model'].unique()}")
print(f"MFGP variants: {df_mfgp['model'].unique()}")
print(f"Total trajectory rows: {len(df)}")

benchmarks = ['Branin-Fav', 'Branin-Unfav', 'Park-Fav', 'Park-Unfav', 'COFs', 'FreeSolv', 'Polarizability']

# Define model categories
dnn_models = [
    'Sequential', 'Progressive', 'Curriculum', 'Two-Stage Joint',
    'DNGO-Joint', 'DNGO-Gradient', 'Knowledge Distillation',
    'Domain Adaptation (MMD)', 'Soft Parameter Sharing', 'Pseudo-Labeling', 'Adapter'
]

mfgp_models = ['MFGP-EI', 'MFGP-UCB', 'MFGP-PI', 'MFGP-EI-Cool', 'MFGP-TS', 'MFGP-UCB-Decay']

# Colors: DNN = blue tones, MFGP = orange/red tones
DNN_COLOR = '#1f77b4'   # Blue
MFGP_COLOR = '#d62728'  # Red

# Create figure: 2x4 grid (7 benchmarks + 1 legend)
fig, axes = plt.subplots(2, 4, figsize=(24, 11))
TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12
LEGEND_SIZE = 13

for idx, benchmark in enumerate(benchmarks):
    ax = axes[idx // 4, idx % 4]
    bench_data = df[df['benchmark'] == benchmark]

    # Plot DNN models
    for model in dnn_models:
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

        ax.plot(budget_grid, mean_r, color=DNN_COLOR, linewidth=1.0, alpha=0.5)

    # Plot MFGP variants
    for model in mfgp_models:
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

        ax.plot(budget_grid, mean_r, color=MFGP_COLOR, linewidth=1.0, alpha=0.5)

    ax.set_yscale('log')
    ax.set_title(benchmark, fontsize=TITLE_SIZE, fontweight='normal')
    ax.set_xlabel('Budget (Cost)', fontsize=LABEL_SIZE, fontweight='normal')
    ax.set_ylabel('Regret (Mean)', fontsize=LABEL_SIZE, fontweight='normal')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3, linewidth=0.5)

# Legend in the last subplot
ax_legend = axes[1, 3]
ax_legend.axis('off')

legend_handles = [
    Line2D([0], [0], color=DNN_COLOR, linewidth=3, alpha=0.7, label=f'DNN Models ({len(dnn_models)} variants)'),
    Line2D([0], [0], color=MFGP_COLOR, linewidth=3, alpha=0.7, label=f'MFGP Variants ({len(mfgp_models)} variants)'),
]
ax_legend.legend(
    handles=legend_handles,
    loc='center',
    fontsize=LEGEND_SIZE + 2,
    frameon=True,
    fancybox=False,
    edgecolor='gray',
    ncol=1,
    handlelength=3,
    labelspacing=1.5
)

plt.tight_layout(w_pad=3.0, h_pad=3.0)

# Save figures
out_png = 'dnn_vs_mfgp_variants.png'
out_pdf = 'dnn_vs_mfgp_variants.pdf'

fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_png} (300 DPI)")

fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_pdf} (vector)")

plt.close()

# =============================================================================
# Summary Statistics: Compare DNN vs MFGP families
# =============================================================================
print("\n" + "=" * 80)
print("COMPARISON: DNN Models vs MFGP Variants")
print("=" * 80)

# Load summary data
df_dnn_summary = pd.read_csv(dnn_dir / 'results_summary.csv')
df_dnn_summary = df_dnn_summary[df_dnn_summary['model'] != 'MFGP']

df_mfgp_summary = pd.read_csv(mfgp_dir / 'results_summary.csv')

for benchmark in benchmarks:
    print(f"\n{benchmark}:")

    # DNN stats
    dnn_bench = df_dnn_summary[df_dnn_summary['benchmark'] == benchmark]
    dnn_mean = dnn_bench['final_regret'].mean()
    dnn_std = dnn_bench['final_regret'].std()
    dnn_best = dnn_bench.groupby('model')['final_regret'].mean().min()
    dnn_best_model = dnn_bench.groupby('model')['final_regret'].mean().idxmin()

    # MFGP stats
    mfgp_bench = df_mfgp_summary[df_mfgp_summary['benchmark'] == benchmark]
    mfgp_mean = mfgp_bench['final_regret'].mean()
    mfgp_std = mfgp_bench['final_regret'].std()
    mfgp_best = mfgp_bench.groupby('model')['final_regret'].mean().min()
    mfgp_best_model = mfgp_bench.groupby('model')['final_regret'].mean().idxmin()

    print(f"  DNN  (avg): {dnn_mean:.4f} ± {dnn_std:.4f} | Best: {dnn_best_model} ({dnn_best:.4f})")
    print(f"  MFGP (avg): {mfgp_mean:.4f} ± {mfgp_std:.4f} | Best: {mfgp_best_model} ({mfgp_best:.4f})")

    if mfgp_mean < dnn_mean:
        print(f"  → MFGP wins by {((dnn_mean - mfgp_mean) / dnn_mean * 100):.1f}%")
    else:
        print(f"  → DNN wins by {((mfgp_mean - dnn_mean) / mfgp_mean * 100):.1f}%")

# Overall comparison
print("\n" + "=" * 80)
print("OVERALL (across all benchmarks)")
print("=" * 80)

dnn_overall = df_dnn_summary['final_regret'].mean()
mfgp_overall = df_mfgp_summary['final_regret'].mean()

print(f"  DNN  average regret: {dnn_overall:.4f}")
print(f"  MFGP average regret: {mfgp_overall:.4f}")

if mfgp_overall < dnn_overall:
    print(f"\n  → MFGP variants are {((dnn_overall - mfgp_overall) / dnn_overall * 100):.1f}% better overall")
else:
    print(f"\n  → DNN models are {((mfgp_overall - dnn_overall) / mfgp_overall * 100):.1f}% better overall")

print("\nDone!")
