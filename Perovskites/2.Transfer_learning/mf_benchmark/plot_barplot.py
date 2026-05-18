import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results_dir = 'benchmark_parallel_20260122_154131'
df_summary = pd.read_csv(f'{results_dir}/results_summary.csv')
df_traj = pd.read_csv(f'{results_dir}/results_trajectory.csv')

benchmarks = ['Branin-Fav', 'Branin-Unfav', 'Park-Fav', 'Park-Unfav', 'COFs', 'FreeSolv', 'Polarizability']

# Rename models for display
rename_map = {
    'DNGO-Joint': 'Stop-Gradient Joint Training',
    'DNGO-Gradient': 'End-to-End Joint Training',
    'Two-Stage Joint': 'Pretrain-then-Joint Training',
}
df_summary['model'] = df_summary['model'].replace(rename_map)
df_traj['model'] = df_traj['model'].replace(rename_map)

models = sorted(df_summary['model'].unique())

# MFGP = orange (baseline), all DNN models = blue
model_colors = {m: ("#f2aa84" if m == "MFGP" else "#4e95d9") for m in models}

# Budget cutoff for early-budget evaluation (Branin & Park use cutoff=10)
early_budget_cutoff = 10

# Actual total budget per benchmark
bench_total_budget = {
    'Branin-Fav': 50, 'Branin-Unfav': 50,
    'Park-Fav': 50, 'Park-Unfav': 50,
    'COFs': 30, 'FreeSolv': 50, 'Polarizability': 30,
}

def get_regret_at_budget(df_traj, benchmark, budget_cutoff):
    """Get regret at specific budget for each model and seed."""
    bench_data = df_traj[df_traj['benchmark'] == benchmark]
    results = []
    for model in models:
        model_data = bench_data[bench_data['model'] == model]
        for seed in model_data['seed'].unique():
            seed_data = model_data[model_data['seed'] == seed].sort_values('budget')
            # Get regret at or just before budget_cutoff
            valid = seed_data[seed_data['budget'] <= budget_cutoff]
            if len(valid) > 0:
                regret = valid.iloc[-1]['regret']
            else:
                regret = seed_data.iloc[0]['regret'] if len(seed_data) > 0 else np.nan
            results.append({'benchmark': benchmark, 'model': model, 'seed': seed, 'final_regret': regret})
    return pd.DataFrame(results)

# Build final regret data
# Branin & Park: use early budget cutoff (10), others: use full budget (final regret)
early_cutoff_benchmarks = ['Park-Fav', 'Park-Unfav']
regret_data = {}
for bench in benchmarks:
    if bench in early_cutoff_benchmarks:
        df_bench = get_regret_at_budget(df_traj, bench, early_budget_cutoff)
    else:
        df_bench = df_summary[df_summary['benchmark'] == bench][['benchmark', 'model', 'seed', 'final_regret']]

    stats = df_bench.groupby('model')['final_regret'].agg(['mean', 'std', 'count']).reset_index()
    stats['se'] = stats['std'] / np.sqrt(stats['count'])
    stats = stats.sort_values('mean', ascending=True)  # best (lowest) at bottom
    regret_data[bench] = stats

# Figure: 2 rows x 4 cols
fig, axes = plt.subplots(2, 4, figsize=(24, 11))
TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12
TEXT_SIZE = 11

for idx, bench in enumerate(benchmarks):
    ax = axes[idx // 4, idx % 4]
    stats = regret_data[bench]

    y_pos = np.arange(len(stats))
    bars = ax.barh(y_pos, stats['mean'], xerr=stats['se'],
                   color=[model_colors[m] for m in stats['model']],
                   capsize=3, alpha=0.85, edgecolor='none', height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(stats['model'], fontsize=TICK_SIZE)

    # Title: show budget for all benchmarks
    if bench in early_cutoff_benchmarks:
        title = f'{bench} (Budget = {early_budget_cutoff})'
    else:
        title = f'{bench} (Budget = {bench_total_budget[bench]})'
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='normal')
    ax.set_xlabel('Final Regret (Mean ± SE)', fontsize=LABEL_SIZE, fontweight='normal')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(axis='x', alpha=0.3, linewidth=0.5)

# Last cell: average rank across all benchmarks
# Rank each model per benchmark (1=best, 12=worst), then average ranks
rank_data = []
for bench in benchmarks:
    stats = regret_data[bench].copy()
    stats['rank'] = stats['mean'].rank(method='average')  # lower regret = lower rank = better
    for _, row in stats.iterrows():
        rank_data.append({'model': row['model'], 'rank': row['rank']})

df_rank = pd.DataFrame(rank_data)
avg_rank = df_rank.groupby('model')['rank'].agg(['mean', 'std']).reset_index()
avg_rank['se'] = avg_rank['std'] / np.sqrt(len(benchmarks))
avg_rank = avg_rank.sort_values('mean', ascending=True)

ax_avg = axes[1, 3]
y_pos = np.arange(len(avg_rank))
ax_avg.barh(y_pos, avg_rank['mean'], xerr=avg_rank['se'],
            color=[model_colors[m] for m in avg_rank['model']],
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
print("Done!")
