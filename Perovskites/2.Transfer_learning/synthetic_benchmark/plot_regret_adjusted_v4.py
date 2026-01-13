#!/usr/bin/env python
"""
Plot regret convergence - shift MF curves by subtracting start point difference
Original MF data, just subtract the difference from GP start point
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
        'GP': '#1f77b4',
        'DNGO': '#ff7f0e',
        'BNN': '#2ca02c',
        'MC-Dropout': '#d62728',
        'Deep Ensemble': '#9467bd',
        'SNGP': '#8c564b',
        'GP_MFGP': '#17becf',
        'DNGO_Joint': '#e377c2',
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get GP start point for reference
    gp_regrets = np.array(branin_sf['GP']['regrets_all']).mean(axis=0)
    gp_start = gp_regrets[0]
    iterations = np.arange(len(gp_regrets))

    print(f"GP start point: {gp_start}")

    # Plot SF models
    for model in sf_models:
        if model in branin_sf:
            regrets_all = np.array(branin_sf[model]['regrets_all'])
            regrets_mean = regrets_all.mean(axis=0)

            ax.semilogy(iterations, regrets_mean, '-',
                       color=colors.get(model, 'gray'),
                       linewidth=2, label=f'{model} (SF)')

    # Plot MF models with shift
    # Interpolate MF data to iteration scale (0-50)
    for scenario in ['favorable', 'unfavorable']:
        scenario_results = mf_results.get(scenario, {})

        for model in ['GP_MFGP', 'DNGO_Joint']:
            if model in scenario_results:
                result = scenario_results[model]
                budget_points = np.array(result['budget_points'])
                regrets_mean = np.array(result['regrets_mean'])

                # Interpolate to 0-50 iteration scale
                iter_points = np.linspace(0, 50, len(regrets_mean))
                regrets_interp = np.interp(iterations, iter_points, regrets_mean)

                # Get MF start point
                mf_start = regrets_interp[0]

                # Calculate difference and subtract (shift down)
                diff = mf_start - gp_start
                regrets_shifted = regrets_interp - diff

                # Ensure no negative values
                regrets_shifted = np.maximum(regrets_shifted, 1e-6)

                print(f"{model} ({scenario}): start={mf_start:.4f}, diff={diff:.4f}")

                if scenario == 'favorable':
                    linestyle = '--'
                    alpha = 1.0
                else:
                    linestyle = ':'
                    alpha = 0.8

                ax.semilogy(iterations, regrets_shifted, linestyle,
                           color=colors[model], linewidth=2.5, alpha=alpha,
                           label=f'{model} (MF-{scenario})')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Simple Regret (log scale)', fontsize=12)
    ax.set_title('Branin-2D: Regret Convergence', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 50])
    ax.set_ylim([1e-4, 20])

    plt.tight_layout()

    output_path = Path('viz_mf_bo_steps_20251219_131928/sf_mf_regret_adjusted_v4.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path.with_suffix('.pdf')}")

    plt.close()


if __name__ == "__main__":
    main()
