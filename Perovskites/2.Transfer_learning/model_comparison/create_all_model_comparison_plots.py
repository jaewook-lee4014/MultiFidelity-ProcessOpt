#!/usr/bin/env python3
"""
Create comparison plots for all models (R² and RMSE)

Models included:
- Baselines: MFGP, DNGO-Joint, DNGO-Gradient, Sequential, Progressive, Curriculum, Two-Stage Joint
- Advanced TL (HP tuned): Pseudo-Labeling, Domain Adaptation (MMD), Knowledge Distillation,
                          Soft Parameter Sharing, Adapter
- MAML (if available)

Author: Claude Code
Date: 2025-12-17
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ============================================================================
# Data Collection
# ============================================================================

# Directory paths
BASE_DIR = Path(__file__).parent
VIZ_DIR = BASE_DIR / 'visualizations'

# Load baseline results from 20251211_163454_all_6methods
baseline_results_path = VIZ_DIR / '20251211_163454_all_6methods' / 'results_summary.csv'

# Prepare data structure
all_results = {}

# 1. Load baseline results
if baseline_results_path.exists():
    df_baseline = pd.read_csv(baseline_results_path)

    # MFGP
    all_results['MFGP'] = {
        'r2_mean': df_baseline['mfgp_r2'].mean(),
        'r2_std': df_baseline['mfgp_r2'].std(),
        'rmse_mean': df_baseline['mfgp_rmse'].mean(),
        'rmse_std': df_baseline['mfgp_rmse'].std(),
    }

    # DNGO-Joint
    all_results['DNGO-Joint'] = {
        'r2_mean': df_baseline['joint_r2'].mean(),
        'r2_std': df_baseline['joint_r2'].std(),
        'rmse_mean': df_baseline['joint_rmse'].mean(),
        'rmse_std': df_baseline['joint_rmse'].std(),
    }

    # DNGO-Gradient
    all_results['DNGO-Gradient'] = {
        'r2_mean': df_baseline['gradient_scaling_r2'].mean(),
        'r2_std': df_baseline['gradient_scaling_r2'].std(),
        'rmse_mean': df_baseline['gradient_scaling_rmse'].mean(),
        'rmse_std': df_baseline['gradient_scaling_rmse'].std(),
    }

    # Sequential
    all_results['Sequential'] = {
        'r2_mean': df_baseline['sequential_r2'].mean(),
        'r2_std': df_baseline['sequential_r2'].std(),
        'rmse_mean': df_baseline['sequential_rmse'].mean(),
        'rmse_std': df_baseline['sequential_rmse'].std(),
    }

    # Progressive
    all_results['Progressive'] = {
        'r2_mean': df_baseline['progressive_r2'].mean(),
        'r2_std': df_baseline['progressive_r2'].std(),
        'rmse_mean': df_baseline['progressive_rmse'].mean(),
        'rmse_std': df_baseline['progressive_rmse'].std(),
    }

    # Curriculum
    all_results['Curriculum'] = {
        'r2_mean': df_baseline['curriculum_r2'].mean(),
        'r2_std': df_baseline['curriculum_r2'].std(),
        'rmse_mean': df_baseline['curriculum_rmse'].mean(),
        'rmse_std': df_baseline['curriculum_rmse'].std(),
    }

    # Two-Stage Joint
    all_results['Two-Stage Joint'] = {
        'r2_mean': df_baseline['two_stage_joint_r2'].mean(),
        'r2_std': df_baseline['two_stage_joint_r2'].std(),
        'rmse_mean': df_baseline['two_stage_joint_rmse'].mean(),
        'rmse_std': df_baseline['two_stage_joint_rmse'].std(),
    }
else:
    print(f"Warning: Baseline results not found at {baseline_results_path}")

# 2. Add HP-tuned Advanced TL results (from slurm output)
# These are results from HP tuning with 200 trials each
advanced_tl_results = {
    'Pseudo-Labeling': {
        'r2_mean': 0.7939,
        'r2_std': 0.0375,
        'rmse_mean': 0.5152,  # average from fold results
        'rmse_std': 0.0469,
    },
    'Domain Adaptation (MMD)': {
        'r2_mean': 0.7690,
        'r2_std': 0.0341,
        'rmse_mean': 0.5465,
        'rmse_std': 0.0424,
    },
    'Knowledge Distillation': {
        'r2_mean': 0.7431,
        'r2_std': 0.0384,
        'rmse_mean': 0.5760,
        'rmse_std': 0.0459,
    },
    'Soft Parameter Sharing': {
        'r2_mean': 0.7414,
        'r2_std': 0.0458,
        'rmse_mean': 0.5775,
        'rmse_std': 0.0536,
    },
    'Adapter': {
        'r2_mean': 0.6616,
        'r2_std': 0.1482,
        'rmse_mean': 0.6497,
        'rmse_std': 0.1368,
    },
}
all_results.update(advanced_tl_results)

# 3. Check for MAML results (if available)
maml_results_dirs = list(VIZ_DIR.glob('*_maml_hp_tuning'))
for maml_dir in sorted(maml_results_dirs):
    summary_file = maml_dir / 'summary.json'
    if summary_file.exists():
        with open(summary_file) as f:
            maml_summary = json.load(f)
        if not np.isnan(maml_summary.get('mean_r2', np.nan)):
            all_results['MAML'] = {
                'r2_mean': maml_summary['mean_r2'],
                'r2_std': maml_summary.get('std_r2', 0),
                'rmse_mean': maml_summary.get('mean_rmse', np.nan),
                'rmse_std': 0,
            }
            print(f"MAML results loaded from {maml_dir}")

# ============================================================================
# Create Plots
# ============================================================================

# Sort by R² performance
sorted_models = sorted(all_results.items(), key=lambda x: x[1]['r2_mean'], reverse=True)
model_names = [m[0] for m in sorted_models]
r2_means = [m[1]['r2_mean'] for m in sorted_models]
r2_stds = [m[1]['r2_std'] for m in sorted_models]
rmse_means = [m[1]['rmse_mean'] for m in sorted_models]
rmse_stds = [m[1]['rmse_std'] for m in sorted_models]

# Color mapping by category
def get_color(name):
    if name == 'MFGP':
        return '#FF6B6B'  # Red - baseline
    elif 'DNGO' in name:
        return '#4ECDC4'  # Teal - DNGO variants
    elif name in ['Sequential', 'Progressive', 'Curriculum', 'Two-Stage Joint']:
        return '#45B7D1'  # Blue - training strategies
    elif name in ['Pseudo-Labeling', 'Domain Adaptation (MMD)', 'Knowledge Distillation',
                  'Soft Parameter Sharing', 'Adapter', 'MAML']:
        return '#96CEB4'  # Green - Advanced TL
    return '#95A5A6'  # Gray - other

colors = [get_color(name) for name in model_names]

# Figure 1: R² Comparison
fig1, ax1 = plt.subplots(figsize=(14, 8))

x = np.arange(len(model_names))
bars1 = ax1.bar(x, r2_means, yerr=r2_stds, capsize=5, color=colors, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for i, (bar, val, std) in enumerate(zip(bars1, r2_means, r2_stds)):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.02,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax1.set_title('Model Comparison: R² Score (Higher is Better)\n72 LF + 9 HF Samples, 10-Fold CV', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
ax1.set_ylim(0, 1.0)
ax1.axhline(y=0.78, color='red', linestyle='--', alpha=0.7, label='DNGO-Joint baseline (0.78)')
ax1.legend(loc='lower right')
ax1.grid(axis='y', alpha=0.3)

# Add category legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF6B6B', edgecolor='black', label='Baseline (MFGP)'),
    Patch(facecolor='#4ECDC4', edgecolor='black', label='DNGO Variants'),
    Patch(facecolor='#45B7D1', edgecolor='black', label='Training Strategies'),
    Patch(facecolor='#96CEB4', edgecolor='black', label='Advanced TL (HP tuned)'),
]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()
fig1.savefig(BASE_DIR / 'all_models_r2_comparison.png', dpi=150, bbox_inches='tight')
fig1.savefig(BASE_DIR / 'all_models_r2_comparison.pdf', bbox_inches='tight')
print(f"Saved: all_models_r2_comparison.png/pdf")

# Figure 2: RMSE Comparison (sorted by RMSE, lower is better)
sorted_by_rmse = sorted(all_results.items(), key=lambda x: x[1]['rmse_mean'])
model_names_rmse = [m[0] for m in sorted_by_rmse]
rmse_means_sorted = [m[1]['rmse_mean'] for m in sorted_by_rmse]
rmse_stds_sorted = [m[1]['rmse_std'] for m in sorted_by_rmse]
colors_rmse = [get_color(name) for name in model_names_rmse]

fig2, ax2 = plt.subplots(figsize=(14, 8))

x = np.arange(len(model_names_rmse))
bars2 = ax2.bar(x, rmse_means_sorted, yerr=rmse_stds_sorted, capsize=5, color=colors_rmse, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for i, (bar, val, std) in enumerate(zip(bars2, rmse_means_sorted, rmse_stds_sorted)):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.02,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
ax2.set_ylabel('RMSE', fontsize=12, fontweight='bold')
ax2.set_title('Model Comparison: RMSE (Lower is Better)\n72 LF + 9 HF Samples, 10-Fold CV', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(model_names_rmse, rotation=45, ha='right', fontsize=10)
ax2.set_ylim(0, max(rmse_means_sorted) * 1.3)
ax2.axhline(y=0.53, color='green', linestyle='--', alpha=0.7, label='DNGO-Joint baseline (~0.53)')
ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig2.savefig(BASE_DIR / 'all_models_rmse_comparison.png', dpi=150, bbox_inches='tight')
fig2.savefig(BASE_DIR / 'all_models_rmse_comparison.pdf', bbox_inches='tight')
print(f"Saved: all_models_rmse_comparison.png/pdf")

# Figure 3: Combined side-by-side comparison
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(18, 8))

# R² subplot
x = np.arange(len(model_names))
bars3a = ax3a.barh(x, r2_means, xerr=r2_stds, capsize=3, color=colors, edgecolor='black', linewidth=1)
ax3a.set_yticks(x)
ax3a.set_yticklabels(model_names, fontsize=10)
ax3a.set_xlabel('R² Score', fontsize=12, fontweight='bold')
ax3a.set_title('R² Score (Higher is Better)', fontsize=12, fontweight='bold')
ax3a.set_xlim(0, 1.0)
ax3a.axvline(x=0.78, color='red', linestyle='--', alpha=0.7, linewidth=2)
ax3a.grid(axis='x', alpha=0.3)

# Add values
for i, (val, std) in enumerate(zip(r2_means, r2_stds)):
    ax3a.text(val + std + 0.02, i, f'{val:.3f}', va='center', fontsize=9)

# RMSE subplot
bars3b = ax3b.barh(x, rmse_means, xerr=rmse_stds, capsize=3, color=colors, edgecolor='black', linewidth=1)
ax3b.set_yticks(x)
ax3b.set_yticklabels(model_names, fontsize=10)
ax3b.set_xlabel('RMSE', fontsize=12, fontweight='bold')
ax3b.set_title('RMSE (Lower is Better)', fontsize=12, fontweight='bold')
ax3b.set_xlim(0, max(rmse_means) * 1.3)
ax3b.grid(axis='x', alpha=0.3)

# Add values
for i, (val, std) in enumerate(zip(rmse_means, rmse_stds)):
    ax3b.text(val + std + 0.02, i, f'{val:.3f}', va='center', fontsize=9)

fig3.suptitle('All Models Performance Comparison\n72 LF + 9 HF Samples, 10-Fold Cross-Validation',
              fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig3.savefig(BASE_DIR / 'all_models_combined_comparison.png', dpi=150, bbox_inches='tight')
fig3.savefig(BASE_DIR / 'all_models_combined_comparison.pdf', bbox_inches='tight')
print(f"Saved: all_models_combined_comparison.png/pdf")

# ============================================================================
# Print Summary Table
# ============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY (Sorted by R²)")
print("="*80)
print(f"{'Model':<30} {'R² Mean':>12} {'R² Std':>10} {'RMSE Mean':>12} {'RMSE Std':>10}")
print("-"*80)
for model, metrics in sorted_models:
    print(f"{model:<30} {metrics['r2_mean']:>12.4f} {metrics['r2_std']:>10.4f} {metrics['rmse_mean']:>12.4f} {metrics['rmse_std']:>10.4f}")
print("="*80)

# Save summary to CSV
summary_df = pd.DataFrame([
    {
        'Model': model,
        'R2_Mean': metrics['r2_mean'],
        'R2_Std': metrics['r2_std'],
        'RMSE_Mean': metrics['rmse_mean'],
        'RMSE_Std': metrics['rmse_std'],
    }
    for model, metrics in sorted_models
])
summary_df.to_csv(BASE_DIR / 'all_models_comparison_summary.csv', index=False)
print(f"\nSaved: all_models_comparison_summary.csv")

plt.show()
print("\nDone!")
