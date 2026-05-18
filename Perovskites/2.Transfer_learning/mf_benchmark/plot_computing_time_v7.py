"""Computing time per model - v7 style (matching barplot colors, alphabet labels, no suptitle)."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results_dir = 'benchmark_parallel_20260122_154131'
df = pd.read_csv(f'{results_dir}/results_summary.csv')

# Rename models to match barplot
rename_map = {
    'DNGO-Joint': 'Stop-Gradient Joint Training',
    'DNGO-Gradient': 'End-to-End Joint Training',
    'Two-Stage Joint': 'Pretrain-then-Joint Training',
}
df['model'] = df['model'].replace(rename_map)

benchmarks = ['Branin-Fav', 'Branin-Unfav', 'Park-Fav', 'Park-Unfav', 'COFs', 'FreeSolv', 'Polarizability']
labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']

# Colors: MFGP = orange, rest = blue (matching barplot_v7)
models = sorted(df['model'].unique())
model_colors = {m: ("#f2aa84" if m == "MFGP" else "#4e95d9") for m in models}

TITLE_SIZE = 16; LABEL_SIZE = 14; TICK_SIZE = 12

fig, axes = plt.subplots(2, 4, figsize=(24, 11))

for idx, bench in enumerate(benchmarks):
    ax = axes[idx // 4, idx % 4]
    bench_data = df[df['benchmark'] == bench]

    stats = bench_data.groupby('model')['elapsed_sec'].agg(['mean', 'std', 'count']).reset_index()
    stats['se'] = stats['std'] / np.sqrt(stats['count'])
    stats = stats.sort_values('mean', ascending=True)  # smallest at bottom, largest at top

    y_pos = np.arange(len(stats))
    bars = ax.barh(y_pos, stats['mean'].values, xerr=stats['se'].values,
                   color=[model_colors[m] for m in stats['model']],
                   capsize=3, alpha=0.85, edgecolor='none', height=0.7)

    # Add time labels
    x_max = stats['mean'].max() + stats['se'].max()
    for i, (_, row) in enumerate(stats.iterrows()):
        label_text = f"{int(round(row['mean']))}s"
        if row['model'] == 'MFGP':
            # MFGP: place below the error bar tip to avoid clipping
            x_pos = row['mean'] + row['se']
            ax.text(x_pos, i - 0.35, label_text, va='top', ha='right',
                    fontsize=9, color='#333333')
        else:
            # Others: place right of the error bar
            ax.text(row['mean'] + row['se'] + x_max * 0.01, i,
                    label_text, va='center', fontsize=10, color='#333333')

    # Extend x-axis for non-MFGP labels
    ax.set_xlim(0, x_max * 1.15)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stats['model'].values, fontsize=TICK_SIZE, fontweight='normal')
    ax.set_title(f'{labels[idx]} {bench}', fontsize=TITLE_SIZE, fontweight='normal')
    ax.set_xlabel('Elapsed Time (sec)', fontsize=LABEL_SIZE, fontweight='normal')
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(axis='x', alpha=0.3, linewidth=0.5)

# Hide last subplot
axes[1, 3].axis('off')

plt.tight_layout(w_pad=3.0, h_pad=3.0)

for ext, dpi in [('png', 1000), ('pdf', None)]:
    out = f'{results_dir}/computing_time_per_model_v7.{ext}'
    if dpi:
        fig.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white')
    else:
        fig.savefig(out, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out}")
plt.close()
print("Done!")
