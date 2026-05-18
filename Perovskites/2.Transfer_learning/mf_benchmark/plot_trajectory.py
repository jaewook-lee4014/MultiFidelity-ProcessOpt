import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

results_dir = 'benchmark_parallel_20260122_154131'
df = pd.read_csv(f'{results_dir}/results_trajectory.csv')

benchmarks = ['Branin-Fav', 'Branin-Unfav', 'Park-Fav', 'Park-Unfav', 'COFs', 'FreeSolv', 'Polarizability']

# Rename models for display
rename_map = {
    'DNGO-Joint': 'Stop-Gradient Joint Training',
    'DNGO-Gradient': 'End-to-End Joint Training',
    'Two-Stage Joint': 'Pretrain-then-Joint Training',
}
df['model'] = df['model'].replace(rename_map)

models = sorted(df['model'].unique())

# Assign colors: highest variance -> blue (#4e95d9), 2nd highest -> orange (#f2aa84), rest -> default palette
var_per_model = df.groupby('model')['regret'].var().sort_values(ascending=False)
top1_model = var_per_model.index[0]
top2_model = var_per_model.index[1]

other_colors = [
    "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#4d4d4d", "#bdbdbd",
]
model_colors = {}
other_idx = 0
for m in models:
    if m == top1_model:
        model_colors[m] = "#4e95d9"
    elif m == top2_model:
        model_colors[m] = "#f2aa84"
    else:
        model_colors[m] = other_colors[other_idx % len(other_colors)]
        other_idx += 1
models_by_length = sorted(models, key=lambda x: len(x))

fig, axes = plt.subplots(2, 4, figsize=(24, 11))
TITLE_SIZE = 16; LABEL_SIZE = 14; TICK_SIZE = 12; LEGEND_SIZE = 13

for idx, benchmark in enumerate(benchmarks):
    ax = axes[idx // 4, idx % 4]
    bench_data = df[df['benchmark'] == benchmark]
    for model in models:
        model_data = bench_data[bench_data['model'] == model]
        if len(model_data) == 0: continue
        seeds = model_data['seed'].unique()
        budget_grid = np.linspace(model_data['budget'].min(), model_data['budget'].max(), 100)
        interpolated = []
        for seed in seeds:
            sd = model_data[model_data['seed'] == seed].sort_values('budget')
            if len(sd) < 2: continue
            interpolated.append(np.interp(budget_grid, sd['budget'].values, sd['regret'].values))
        if not interpolated: continue
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

ax_legend = axes[1, 3]
ax_legend.axis('off')
legend_handles = [Line2D([0], [0], color=model_colors[m], linewidth=2.5, label=m) for m in models_by_length]
ax_legend.legend(handles=legend_handles, loc='center', fontsize=LEGEND_SIZE,
                frameon=True, fancybox=False, edgecolor='gray', ncol=1, handlelength=2.5, labelspacing=1.0)
plt.tight_layout(w_pad=3.0, h_pad=3.0)

out_png = f'{results_dir}/all_benchmarks_trajectory_v6.png'
out_pdf = f'{results_dir}/all_benchmarks_trajectory_v6.pdf'
fig.savefig(out_png, dpi=1000, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_png} (1000 DPI)")
fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_pdf} (vector)")
plt.close()
print("Done!")
