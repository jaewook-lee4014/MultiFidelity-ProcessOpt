"""
100 Runs Results Visualization for DNGO, BNN, and DNGO-OL models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
try:
    plt.style.use('seaborn-whitegrid')
except:
    plt.style.use('ggplot')
sns.set_palette("husl")

# File paths
BASE_DIR = Path("/cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/Pure_TL_BO")

# Load data
dngo_df = pd.read_csv(BASE_DIR / "logs_100runs_20251203_154548/dngo_100runs.csv")
bnn_df = pd.read_csv(BASE_DIR / "logs_100runs_20251203_154548/bnn_100runs.csv")
dngo_ol_df = pd.read_csv(BASE_DIR / "logs_ol_100runs_20251204_160036/dngo_ol_100runs.csv")

# Add model labels
dngo_df['model'] = 'DNGO'
bnn_df['model'] = 'BNN'
dngo_ol_df['model'] = 'DNGO-OL'

# Combine all data
all_data = pd.concat([dngo_df, bnn_df, dngo_ol_df], ignore_index=True)

# Target value
TARGET = 1.5249

# Calculate success rate (finding target)
def calculate_success_rate(df, target=TARGET):
    return (df['best_so_far'] == target).sum() / len(df) * 100

# Calculate statistics
def get_stats(df):
    return {
        'mean_cost': df['total_cost'].mean(),
        'std_cost': df['total_cost'].std(),
        'mean_best': df['best_so_far'].mean(),
        'std_best': df['best_so_far'].std(),
        'success_rate': calculate_success_rate(df),
        'median_cost': df['total_cost'].median(),
        'min_cost': df['total_cost'].min(),
        'max_cost': df['total_cost'].max()
    }

dngo_stats = get_stats(dngo_df)
bnn_stats = get_stats(bnn_df)
dngo_ol_stats = get_stats(dngo_ol_df)

print("="*70)
print("100 Runs Results Summary")
print("="*70)
print(f"\n{'Model':<12} {'Success Rate':<15} {'Mean Cost':<15} {'Std Cost':<12} {'Mean Best':<12}")
print("-"*70)
print(f"{'DNGO':<12} {dngo_stats['success_rate']:.1f}%{'':<10} {dngo_stats['mean_cost']:.2f}{'':<8} {dngo_stats['std_cost']:.2f}{'':<5} {dngo_stats['mean_best']:.4f}")
print(f"{'BNN':<12} {bnn_stats['success_rate']:.1f}%{'':<10} {bnn_stats['mean_cost']:.2f}{'':<8} {bnn_stats['std_cost']:.2f}{'':<5} {bnn_stats['mean_best']:.4f}")
print(f"{'DNGO-OL':<12} {dngo_ol_stats['success_rate']:.1f}%{'':<10} {dngo_ol_stats['mean_cost']:.2f}{'':<8} {dngo_ol_stats['std_cost']:.2f}{'':<5} {dngo_ol_stats['mean_best']:.4f}")
print("="*70)

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Color palette
colors = {'DNGO': '#2ecc71', 'BNN': '#e74c3c', 'DNGO-OL': '#3498db'}

# 1. Success Rate Bar Plot
ax1 = axes[0, 0]
models = ['DNGO', 'BNN', 'DNGO-OL']
success_rates = [dngo_stats['success_rate'], bnn_stats['success_rate'], dngo_ol_stats['success_rate']]
bars = ax1.bar(models, success_rates, color=[colors[m] for m in models], edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Success Rate (%)', fontsize=12)
ax1.set_title('Success Rate (Finding Target 1.5249)', fontsize=13, fontweight='bold')
ax1.set_ylim(0, 100)
for bar, rate in zip(bars, success_rates):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 2. Mean Cost Bar Plot with Error Bars
ax2 = axes[0, 1]
mean_costs = [dngo_stats['mean_cost'], bnn_stats['mean_cost'], dngo_ol_stats['mean_cost']]
std_costs = [dngo_stats['std_cost'], bnn_stats['std_cost'], dngo_ol_stats['std_cost']]
bars = ax2.bar(models, mean_costs, yerr=std_costs, capsize=5,
               color=[colors[m] for m in models], edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Cost', fontsize=12)
ax2.set_title('Mean Cost (with Std Dev)', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 60)
for bar, cost in zip(bars, mean_costs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             f'{cost:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 3. Best Value Distribution (Box Plot)
ax3 = axes[0, 2]
box_data = [dngo_df['best_so_far'].values, bnn_df['best_so_far'].values, dngo_ol_df['best_so_far'].values]
bp = ax3.boxplot(box_data, labels=models, patch_artist=True)
for patch, model in zip(bp['boxes'], models):
    patch.set_facecolor(colors[model])
    patch.set_alpha(0.7)
ax3.axhline(y=TARGET, color='red', linestyle='--', linewidth=2, label=f'Target ({TARGET})')
ax3.set_ylabel('Best Value Found', fontsize=12)
ax3.set_title('Best Value Distribution', fontsize=13, fontweight='bold')
ax3.legend(loc='upper right')

# 4. Cost Distribution (Histogram)
ax4 = axes[1, 0]
for model, df, color in [('DNGO', dngo_df, colors['DNGO']),
                          ('BNN', bnn_df, colors['BNN']),
                          ('DNGO-OL', dngo_ol_df, colors['DNGO-OL'])]:
    ax4.hist(df['total_cost'], bins=15, alpha=0.6, label=model, color=color, edgecolor='black')
ax4.set_xlabel('Total Cost', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.set_title('Cost Distribution', fontsize=13, fontweight='bold')
ax4.legend()
ax4.axvline(x=50, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Budget Limit')

# 5. Cost vs Best Value Scatter Plot
ax5 = axes[1, 1]
for model, df, color in [('DNGO', dngo_df, colors['DNGO']),
                          ('BNN', bnn_df, colors['BNN']),
                          ('DNGO-OL', dngo_ol_df, colors['DNGO-OL'])]:
    ax5.scatter(df['total_cost'], df['best_so_far'], alpha=0.6, label=model,
                color=color, s=50, edgecolor='black', linewidth=0.5)
ax5.axhline(y=TARGET, color='red', linestyle='--', linewidth=2, alpha=0.8)
ax5.set_xlabel('Total Cost', fontsize=12)
ax5.set_ylabel('Best Value Found', fontsize=12)
ax5.set_title('Cost vs Best Value', fontsize=13, fontweight='bold')
ax5.legend()

# 6. Summary Statistics Table
ax6 = axes[1, 2]
ax6.axis('off')
table_data = [
    ['Metric', 'DNGO', 'BNN', 'DNGO-OL'],
    ['Success Rate', f'{dngo_stats["success_rate"]:.1f}%', f'{bnn_stats["success_rate"]:.1f}%', f'{dngo_ol_stats["success_rate"]:.1f}%'],
    ['Mean Cost', f'{dngo_stats["mean_cost"]:.2f}', f'{bnn_stats["mean_cost"]:.2f}', f'{dngo_ol_stats["mean_cost"]:.2f}'],
    ['Std Cost', f'{dngo_stats["std_cost"]:.2f}', f'{bnn_stats["std_cost"]:.2f}', f'{dngo_ol_stats["std_cost"]:.2f}'],
    ['Median Cost', f'{dngo_stats["median_cost"]:.2f}', f'{bnn_stats["median_cost"]:.2f}', f'{dngo_ol_stats["median_cost"]:.2f}'],
    ['Min Cost', f'{dngo_stats["min_cost"]:.2f}', f'{bnn_stats["min_cost"]:.2f}', f'{dngo_ol_stats["min_cost"]:.2f}'],
    ['Max Cost', f'{dngo_stats["max_cost"]:.2f}', f'{bnn_stats["max_cost"]:.2f}', f'{dngo_ol_stats["max_cost"]:.2f}'],
    ['Mean Best', f'{dngo_stats["mean_best"]:.4f}', f'{bnn_stats["mean_best"]:.4f}', f'{dngo_ol_stats["mean_best"]:.4f}'],
]

table = ax6.table(cellText=table_data, loc='center', cellLoc='center',
                  colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# Header styling
for j in range(4):
    table[(0, j)].set_facecolor('#34495e')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Column colors
col_colors = ['#ecf0f1', colors['DNGO'], colors['BNN'], colors['DNGO-OL']]
for i in range(1, len(table_data)):
    for j in range(4):
        if j > 0:
            table[(i, j)].set_facecolor(col_colors[j])
            table[(i, j)].set_alpha(0.3)

ax6.set_title('Summary Statistics', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(BASE_DIR / 'results_100runs_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(BASE_DIR / 'results_100runs_comparison.pdf', bbox_inches='tight')
print(f"\nFigure saved to: {BASE_DIR / 'results_100runs_comparison.png'}")

# Additional: Violin Plot for Cost Distribution
fig2, ax = plt.subplots(figsize=(10, 6))
violin_data = []
for model, df in [('DNGO', dngo_df), ('BNN', bnn_df), ('DNGO-OL', dngo_ol_df)]:
    for cost in df['total_cost']:
        violin_data.append({'Model': model, 'Cost': cost})

violin_df = pd.DataFrame(violin_data)
sns.violinplot(x='Model', y='Cost', data=violin_df, palette=colors, ax=ax)
ax.set_title('Cost Distribution by Model (Violin Plot)', fontsize=14, fontweight='bold')
ax.set_ylabel('Total Cost', fontsize=12)
ax.set_xlabel('Model', fontsize=12)
ax.axhline(y=50, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Budget Limit (50)')
ax.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / 'results_100runs_violin.png', dpi=300, bbox_inches='tight')
print(f"Violin plot saved to: {BASE_DIR / 'results_100runs_violin.png'}")

# Cumulative Success Plot
fig3, ax = plt.subplots(figsize=(10, 6))

for model, df, color in [('DNGO', dngo_df, colors['DNGO']),
                          ('BNN', bnn_df, colors['BNN']),
                          ('DNGO-OL', dngo_ol_df, colors['DNGO-OL'])]:
    # Sort by cost
    sorted_df = df.sort_values('total_cost')
    cumulative_success = (sorted_df['best_so_far'] == TARGET).cumsum()
    ax.plot(sorted_df['total_cost'], cumulative_success, label=model,
            color=color, linewidth=2.5)

ax.set_xlabel('Cost', fontsize=12)
ax.set_ylabel('Cumulative Success Count', fontsize=12)
ax.set_title('Cumulative Success vs Cost', fontsize=14, fontweight='bold')
ax.legend()
ax.axvline(x=50, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Budget Limit')
plt.tight_layout()
plt.savefig(BASE_DIR / 'results_100runs_cumulative.png', dpi=300, bbox_inches='tight')
print(f"Cumulative plot saved to: {BASE_DIR / 'results_100runs_cumulative.png'}")

plt.show()
