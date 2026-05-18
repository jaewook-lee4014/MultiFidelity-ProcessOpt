"""Generate high-res (1000 DPI) cumulative evaluations plot."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results_dir = 'benchmark_lf_blr_20260123_045253'
df = pd.read_csv(f'{results_dir}/results_trajectory.csv')

benchmarks = ['Branin-Fav', 'Branin-Unfav', 'Park-Fav', 'Park-Unfav', 'COFs', 'FreeSolv', 'Polarizability']
cost_ratios = {
    'Branin-Fav': 0.1, 'Branin-Unfav': 0.5,
    'Park-Fav': 0.1, 'Park-Unfav': 0.5,
    'COFs': 0.065, 'FreeSolv': 0.1, 'Polarizability': 0.167,
}
total_budgets = {
    'Branin-Fav': 50, 'Branin-Unfav': 50,
    'Park-Fav': 50, 'Park-Unfav': 50,
    'COFs': 30, 'FreeSolv': 50, 'Polarizability': 30,
}
labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']

model0 = df['model'].unique()[0]
seed0 = df['seed'].unique()[0]

fig, axes = plt.subplots(2, 4, figsize=(24, 11))
TITLE_SIZE = 16; LABEL_SIZE = 14; TICK_SIZE = 12

for idx, bench in enumerate(benchmarks):
    ax = axes[idx // 4, idx % 4]
    rho = cost_ratios[bench]
    threshold = (1.0 + rho) / 2

    sub = df[(df['benchmark'] == bench) & (df['model'] == model0) & (df['seed'] == seed0)].sort_values('budget')
    budgets_arr = sub['budget'].values
    diffs = np.diff(budgets_arr)

    total_budget = total_budgets[bench]
    init_budget = 0.1 * total_budget
    n_init_hf = max(2, int(init_budget * 0.5 / 1.0))
    n_init_lf = max(2, int(init_budget * 0.5 / rho))

    cum_hf = [n_init_hf]
    cum_lf = [n_init_lf]
    cum_total = [n_init_hf + n_init_lf]
    budget_points = [budgets_arr[0]]

    for i, d in enumerate(diffs):
        if d > threshold:
            cum_hf.append(cum_hf[-1] + 1)
            cum_lf.append(cum_lf[-1])
        else:
            cum_hf.append(cum_hf[-1])
            cum_lf.append(cum_lf[-1] + 1)
        cum_total.append(cum_hf[-1] + cum_lf[-1])
        budget_points.append(budgets_arr[i + 1])

    ax.plot(budget_points, cum_hf, color='#d62728', linewidth=2.0, label=f'HF (cost=1.0)')
    ax.plot(budget_points, cum_lf, color='#4e95d9', linewidth=2.0, label=f'LF (cost={rho})')
    ax.plot(budget_points, cum_total, color='#2ca02c', linewidth=2.0, linestyle='--', label='Total')

    ax.fill_between(budget_points, 0, cum_hf, color='#d62728', alpha=0.1)
    ax.fill_between(budget_points, cum_hf, cum_total, color='#4e95d9', alpha=0.1)

    ax.annotate(f'HF={cum_hf[-1]}', xy=(budget_points[-1], cum_hf[-1]),
                fontsize=11, fontweight='bold', color='#d62728', ha='right', va='bottom')
    ax.annotate(f'LF={cum_lf[-1]}', xy=(budget_points[-1], cum_lf[-1]),
                fontsize=11, fontweight='bold', color='#4e95d9', ha='right', va='bottom')
    ax.annotate(f'Total={cum_total[-1]}', xy=(budget_points[-1], cum_total[-1]),
                fontsize=11, fontweight='bold', color='#2ca02c', ha='right', va='bottom')

    ax.set_title(f'{labels[idx]} {bench}  (ρ = {rho})', fontsize=TITLE_SIZE, fontweight='normal')
    ax.set_xlabel('Budget (Cost)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Cumulative Evaluations', fontsize=LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)

axes[1, 3].axis('off')
plt.tight_layout(w_pad=3.0, h_pad=3.0)

out_png = f'{results_dir}/cumulative_evaluations_per_benchmark.png'
out_pdf = f'{results_dir}/cumulative_evaluations_per_benchmark.pdf'
fig.savefig(out_png, dpi=1000, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_png} (1000 DPI)")
fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_pdf} (vector)")
plt.close()
print("Done!")
