#!/usr/bin/env python
"""
Visualize SF BO benchmark results
- Budget (iterations) vs Regret with std shading
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results_dir = Path("results_parallel_20251218_205819")
with open(results_dir / "results.json", "r") as f:
    results = json.load(f)

# Color palette for models
colors = {
    'GP': '#1f77b4',
    'DNGO': '#ff7f0e',
    'BNN': '#2ca02c',
    'MC-Dropout': '#d62728',
    'Deep Ensemble': '#9467bd',
    'SNGP': '#8c564b'
}

# Create figure with 2 subplots (one per function)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (func_name, func_data) in enumerate(results.items()):
    ax = axes[idx]

    for model_name, model_data in func_data.items():
        mean = np.array(model_data['regrets_mean'])
        std = np.array(model_data['regrets_std'])
        iterations = np.arange(1, len(mean) + 1)

        color = colors.get(model_name, '#333333')

        # Plot mean line
        ax.plot(iterations, mean, label=model_name, color=color, linewidth=2)

        # Plot std shading (mean ± std)
        ax.fill_between(iterations,
                        np.maximum(mean - std, 1e-6),  # Avoid negative
                        mean + std,
                        color=color, alpha=0.2)

    ax.set_xlabel('Budget (Iterations)', fontsize=12)
    ax.set_ylabel('Simple Regret', fontsize=12)
    ax.set_title(func_name, fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, 51])

plt.tight_layout()
plt.savefig(results_dir / "regret_curves.png", dpi=150, bbox_inches='tight')
plt.savefig(results_dir / "regret_curves.pdf", dpi=150, bbox_inches='tight')
print(f"Saved: {results_dir / 'regret_curves.png'}")
print(f"Saved: {results_dir / 'regret_curves.pdf'}")

# Also create a bar chart for final regret
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

for idx, (func_name, func_data) in enumerate(results.items()):
    ax = axes2[idx]

    models = list(func_data.keys())
    final_means = [func_data[m]['regrets_mean'][-1] for m in models]
    final_stds = [func_data[m]['regrets_std'][-1] for m in models]

    # Sort by performance
    sorted_idx = np.argsort(final_means)
    models = [models[i] for i in sorted_idx]
    final_means = [final_means[i] for i in sorted_idx]
    final_stds = [final_stds[i] for i in sorted_idx]
    bar_colors = [colors.get(m, '#333333') for m in models]

    bars = ax.bar(models, final_means, yerr=final_stds, capsize=5,
                  color=bar_colors, edgecolor='black', linewidth=1)

    ax.set_ylabel('Final Simple Regret', fontsize=12)
    ax.set_title(f'{func_name} - Final Performance', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(results_dir / "final_regret_bars.png", dpi=150, bbox_inches='tight')
plt.savefig(results_dir / "final_regret_bars.pdf", dpi=150, bbox_inches='tight')
print(f"Saved: {results_dir / 'final_regret_bars.png'}")
print(f"Saved: {results_dir / 'final_regret_bars.pdf'}")

print("\nDone!")
