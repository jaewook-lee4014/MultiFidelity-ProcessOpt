#!/usr/bin/env python
"""
Plot regret convergence with mean only (no variance bands)
Include SF models (GP, DNGO, BNN, etc.) and MF models (MFGP, DNGO_Joint)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Load SF results
    sf_results_path = Path('results_viz_20251218_175422/results.json')
    with open(sf_results_path, 'r') as f:
        sf_results = json.load(f)

    # Load MF results
    mf_results_path = Path('results_mf_branin_20251219_035310/results.json')
    with open(mf_results_path, 'r') as f:
        mf_results = json.load(f)

    # Extract Branin-2D SF results
    branin_sf = sf_results['Branin-2D']

    # SF models
    sf_models = ['GP', 'DNGO', 'BNN', 'MC-Dropout', 'Deep Ensemble', 'SNGP']

    # Colors
    colors = {
        'GP': '#1f77b4',           # blue
        'DNGO': '#ff7f0e',         # orange
        'BNN': '#2ca02c',          # green
        'MC-Dropout': '#d62728',   # red
        'Deep Ensemble': '#9467bd', # purple
        'SNGP': '#8c564b',         # brown
        'GP_MFGP': '#17becf',      # cyan
        'DNGO_Joint': '#e377c2',   # pink
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot SF models
    for model in sf_models:
        if model in branin_sf:
            regrets_all = np.array(branin_sf[model]['regrets_all'])
            regrets_mean = regrets_all.mean(axis=0)
            iterations = np.arange(len(regrets_mean))

            ax.semilogy(iterations, regrets_mean, '-',
                       color=colors.get(model, 'gray'),
                       linewidth=2, label=f'{model} (SF)')

    # Plot MF models (favorable scenario - better performance)
    # MF results have budget_points instead of iterations
    for scenario in ['favorable']:
        scenario_results = mf_results.get(scenario, {})

        for model in ['GP_MFGP', 'DNGO_Joint']:
            if model in scenario_results:
                result = scenario_results[model]
                budget_points = np.array(result['budget_points'])
                regrets_mean = np.array(result['regrets_mean'])

                # Convert budget to approximate iterations (for comparison)
                # In SF, each iteration costs 1. In MF favorable, avg cost ~ 0.55
                # So we can plot against budget directly or scale

                ax.semilogy(budget_points, regrets_mean, '-',
                           color=colors.get(model, 'gray'),
                           linewidth=2.5, linestyle='--',
                           label=f'{model} (MF-favorable)')

    # Also add unfavorable for comparison
    for scenario in ['unfavorable']:
        scenario_results = mf_results.get(scenario, {})

        for model in ['GP_MFGP']:  # Only MFGP for unfavorable (DNGO_Joint performs poorly)
            if model in scenario_results:
                result = scenario_results[model]
                budget_points = np.array(result['budget_points'])
                regrets_mean = np.array(result['regrets_mean'])

                ax.semilogy(budget_points, regrets_mean, ':',
                           color=colors.get(model, 'gray'),
                           linewidth=2, alpha=0.7,
                           label=f'{model} (MF-unfavorable)')

    ax.set_xlabel('Iteration / Budget', fontsize=12)
    ax.set_ylabel('Simple Regret (log scale)', fontsize=12)
    ax.set_title('Branin-2D: SF vs MF Model Comparison (Mean Only)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 50])
    ax.set_ylim([1e-4, 20])

    plt.tight_layout()

    output_path = Path('viz_mf_bo_steps_20251219_131928/sf_mf_regret_comparison_mean.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Also save PDF
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path.with_suffix('.pdf')}")

    plt.close()

    # Print final regret summary
    print("\n" + "="*60)
    print("Final Regret Summary (Branin-2D)")
    print("="*60)
    print(f"{'Model':<20} {'Type':<15} {'Final Regret':>15}")
    print("-"*60)

    for model in sf_models:
        if model in branin_sf:
            regrets_all = np.array(branin_sf[model]['regrets_all'])
            final_regret = regrets_all[:, -1].mean()
            print(f"{model:<20} {'SF':<15} {final_regret:>15.6f}")

    for scenario in ['favorable', 'unfavorable']:
        scenario_results = mf_results.get(scenario, {})
        for model in ['GP_MFGP', 'DNGO_Joint']:
            if model in scenario_results:
                final_regret = scenario_results[model]['final_regret_mean']
                print(f"{model:<20} {f'MF-{scenario[:3]}':<15} {final_regret:>15.6f}")

    print("="*60)


if __name__ == "__main__":
    main()
