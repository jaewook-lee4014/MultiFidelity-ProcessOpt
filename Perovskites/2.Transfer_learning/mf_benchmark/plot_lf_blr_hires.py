"""Generate high-res (1000 DPI) trajectory + barplot for benchmark_lf_blr_20260123_045253."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

results_dir = 'benchmark_lf_blr_20260123_045253'
df = pd.read_csv(f'{results_dir}/results_trajectory.csv')
df_summary = pd.read_csv(f'{results_dir}/results_summary.csv')

benchmarks = ['Branin-Fav', 'Branin-Unfav', 'Park-Fav', 'Park-Unfav', 'COFs', 'FreeSolv', 'Polarizability']

rename_map = {
    'DNGO-Joint': 'Stop-Gradient Joint Training',
    'DNGO-Gradient': 'End-to-End Joint Training',
    'Two-Stage Joint': 'Pretrain-then-Joint Training',
}
df['model'] = df['model'].replace(rename_map)
df_summary['model'] = df_summary['model'].replace(rename_map)
models = sorted(df['model'].unique())

# =====================================================================
# 1. TRAJECTORY PLOT
# =====================================================================
var_per_model = df.groupby('model')['regret'].var().sort_values(ascending=False)
top1_model = var_per_model.index[0]
top2_model = var_per_model.index[1]

other_colors = [
    "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#4d4d4d", "#bdbdbd",
]
model_colors_traj = {}
other_idx = 0
for m in models:
    if m == top1_model:
        model_colors_traj[m] = "#4e95d9"
    elif m == top2_model:
        model_colors_traj[m] = "#f2aa84"
    else:
        model_colors_traj[m] = other_colors[other_idx % len(other_colors)]
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
        color = model_colors_traj[model]
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
legend_handles = [Line2D([0], [0], color=model_colors_traj[m], linewidth=2.5, label=m) for m in models_by_length]
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

# =====================================================================
# 2. BARPLOT
# =====================================================================
model_colors_bar = {m: ("#f2aa84" if m == "MFGP" else "#4e95d9") for m in models}

early_budget_cutoff = 10
bench_total_budget = {
    'Branin-Fav': 50, 'Branin-Unfav': 50,
    'Park-Fav': 50, 'Park-Unfav': 50,
    'COFs': 30, 'FreeSolv': 50, 'Polarizability': 30,
}

def get_regret_at_budget(df_traj, benchmark, budget_cutoff):
    bench_data = df_traj[df_traj['benchmark'] == benchmark]
    results = []
    for model in models:
        model_data = bench_data[bench_data['model'] == model]
        for seed in model_data['seed'].unique():
            seed_data = model_data[model_data['seed'] == seed].sort_values('budget')
            valid = seed_data[seed_data['budget'] <= budget_cutoff]
            if len(valid) > 0:
                regret = valid.iloc[-1]['regret']
            else:
                regret = seed_data.iloc[0]['regret'] if len(seed_data) > 0 else np.nan
            results.append({'benchmark': benchmark, 'model': model, 'seed': seed, 'final_regret': regret})
    return pd.DataFrame(results)

early_cutoff_benchmarks = ['Park-Fav', 'Park-Unfav']
regret_data = {}
for bench in benchmarks:
    if bench in early_cutoff_benchmarks:
        df_bench = get_regret_at_budget(df, bench, early_budget_cutoff)
    else:
        df_bench = df_summary[df_summary['benchmark'] == bench][['benchmark', 'model', 'seed', 'final_regret']]
    stats = df_bench.groupby('model')['final_regret'].agg(['mean', 'std', 'count']).reset_index()
    stats['se'] = stats['std'] / np.sqrt(stats['count'])
    stats = stats.sort_values('mean', ascending=True)
    regret_data[bench] = stats

fig, axes = plt.subplots(2, 4, figsize=(24, 11))

for idx, bench in enumerate(benchmarks):
    ax = axes[idx // 4, idx % 4]
    stats = regret_data[bench]
    y_pos = np.arange(len(stats))
    ax.barh(y_pos, stats['mean'], xerr=stats['se'],
            color=[model_colors_bar[m] for m in stats['model']],
            capsize=3, alpha=0.85, edgecolor='none', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stats['model'], fontsize=TICK_SIZE)
    if bench in early_cutoff_benchmarks:
        title = f'{bench} (Budget = {early_budget_cutoff})'
    else:
        title = f'{bench} (Budget = {bench_total_budget[bench]})'
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='normal')
    ax.set_xlabel('Final Regret (Mean ± SE)', fontsize=LABEL_SIZE, fontweight='normal')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(axis='x', alpha=0.3, linewidth=0.5)

# Average Rank
rank_data = []
for bench in benchmarks:
    stats = regret_data[bench].copy()
    stats['rank'] = stats['mean'].rank(method='average')
    for _, row in stats.iterrows():
        rank_data.append({'model': row['model'], 'rank': row['rank']})

df_rank = pd.DataFrame(rank_data)
avg_rank = df_rank.groupby('model')['rank'].agg(['mean', 'std']).reset_index()
avg_rank['se'] = avg_rank['std'] / np.sqrt(len(benchmarks))
avg_rank = avg_rank.sort_values('mean', ascending=True)

ax_avg = axes[1, 3]
y_pos = np.arange(len(avg_rank))
ax_avg.barh(y_pos, avg_rank['mean'], xerr=avg_rank['se'],
            color=[model_colors_bar[m] for m in avg_rank['model']],
            capsize=3, alpha=0.85, edgecolor='none', height=0.7)
ax_avg.set_yticks(y_pos)
ax_avg.set_yticklabels(avg_rank['model'], fontsize=TICK_SIZE)
ax_avg.set_title('Average Rank', fontsize=TITLE_SIZE, fontweight='normal')
ax_avg.set_xlabel('Average Rank (Mean ± SE)', fontsize=LABEL_SIZE, fontweight='normal')
ax_avg.tick_params(axis='both', labelsize=TICK_SIZE)
ax_avg.grid(axis='x', alpha=0.3, linewidth=0.5)

plt.tight_layout(w_pad=3.0, h_pad=3.0)

out_png = f'{results_dir}/all_benchmarks_barplot_v6.png'
out_pdf = f'{results_dir}/all_benchmarks_barplot_v6.pdf'
fig.savefig(out_png, dpi=1000, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_png} (1000 DPI)")
fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_pdf} (vector)")
plt.close()

print("All done!")
