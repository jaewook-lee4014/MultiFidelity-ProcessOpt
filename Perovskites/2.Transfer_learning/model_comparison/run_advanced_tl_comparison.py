#!/usr/bin/env python3
"""
Run Advanced Transfer Learning Methods and Compare with MFGP/DNGO-Joint

This script:
1. Loads existing MFGP and DNGO-Joint results
2. Runs 6 new advanced TL methods
3. Compares all results together

Author: Claude Code
Date: 2025-12-16
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Import from advanced_transfer_learning
from advanced_transfer_learning import (
    load_base_data, generate_data, set_seeds,
    MultiFidelityNetwork, AdapterNetwork,
    train_knowledge_distillation,
    train_domain_adaptation_mmd,
    train_soft_parameter_sharing,
    train_pseudo_labeling,
    train_adapter,
    train_maml,
    evaluate_model
)

# ============================================================================
# Configuration
# ============================================================================

N_LOFI = 72
N_HIFI = 9
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

# Path to existing results
EXISTING_RESULTS_PATH = Path(__file__).parent / 'visualizations' / '20251211_163454_all_6methods' / 'results_summary.csv'


def load_existing_results():
    """Load existing MFGP and DNGO-Joint results"""
    df = pd.read_csv(EXISTING_RESULTS_PATH)

    results = {
        'mfgp': [],
        'dngo_joint': []
    }

    for _, row in df.iterrows():
        results['mfgp'].append({
            'fold': int(row['fold']),
            'seed': int(row['seed']),
            'r2': row['mfgp_r2'],
            'rmse': row['mfgp_rmse']
        })
        results['dngo_joint'].append({
            'fold': int(row['fold']),
            'seed': int(row['seed']),
            'r2': row['joint_r2'],
            'rmse': row['joint_rmse']
        })

    return results


def run_experiment(method_name, train_func, data, params, device, seed, model_type='mf'):
    """Run single experiment"""
    set_seeds(seed)

    if model_type == 'adapter':
        model = AdapterNetwork(
            input_dim=3,
            hidden_dim=params.get('hidden_dim', 64),
            num_layers=params.get('num_layers', 2),
            bottleneck_dim=params.get('bottleneck_dim', 16),
            dropout=params.get('dropout', 0.0),
            activation=params.get('activation', 'relu')
        )
    else:
        model = MultiFidelityNetwork(
            input_dim=3,
            lf_hidden=params.get('lf_hidden', 64),
            hf_hidden=params.get('hf_hidden', 64),
            lf_layers=params.get('lf_layers', 2),
            hf_layers=params.get('hf_layers', 2),
            dropout=params.get('dropout', 0.0),
            activation=params.get('activation', 'relu'),
            residual_mode=True
        )
    model.to(device)

    # Train
    train_func(model, data['X_low'], data['y_low'], data['X_high'], data['y_high'], params, device)

    # Evaluate on test set (all data except HF train)
    test_mask = np.ones(len(data['X_all']), dtype=bool)
    test_mask[data['hifi_idx']] = False

    X_test = data['X_all'][test_mask]
    y_test = data['y_all'][test_mask]

    results = evaluate_model(model, X_test, y_test, device, model_type)

    return results


def create_comparison_visualization(all_results, output_dir):
    """Create comparison visualization"""

    # Prepare data for plotting
    methods = list(all_results.keys())
    mean_r2 = []
    std_r2 = []

    for method in methods:
        r2_values = [r['r2'] for r in all_results[method] if not np.isnan(r['r2'])]
        mean_r2.append(np.mean(r2_values))
        std_r2.append(np.std(r2_values))

    # Sort by mean R²
    sorted_idx = np.argsort(mean_r2)[::-1]
    methods = [methods[i] for i in sorted_idx]
    mean_r2 = [mean_r2[i] for i in sorted_idx]
    std_r2 = [std_r2[i] for i in sorted_idx]

    # Color coding
    colors = []
    for m in methods:
        if m in ['mfgp', 'dngo_joint']:
            colors.append('#2ecc71')  # Green for baseline
        else:
            colors.append('#3498db')  # Blue for new methods

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    ax1 = axes[0]
    x = np.arange(len(methods))
    bars = ax1.bar(x, mean_r2, yerr=std_r2, capsize=5, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', '\n') for m in methods], rotation=45, ha='right')
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Model Comparison: R² Score (10-Fold CV)', fontsize=14)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylim(-0.5, 1.0)

    # Add value labels
    for bar, val, err in zip(bars, mean_r2, std_r2):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Baseline (existing)'),
        Patch(facecolor='#3498db', edgecolor='black', label='New TL methods')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')

    # Box plot
    ax2 = axes[1]
    data_for_box = []
    for method in methods:
        r2_values = [r['r2'] for r in all_results[method] if not np.isnan(r['r2'])]
        data_for_box.append(r2_values)

    bp = ax2.boxplot(data_for_box, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax2.set_xticklabels([m.replace('_', '\n') for m in methods], rotation=45, ha='right')
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('R² Distribution Across Folds', fontsize=14)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_advanced_tl.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {output_dir / 'comparison_advanced_tl.png'}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Data: {N_LOFI} LF + {N_HIFI} HF")
    print(f"Folds: {len(SEEDS)}")

    # Load existing results
    print("\n" + "="*70)
    print("Loading existing MFGP and DNGO-Joint results...")
    print("="*70)

    existing_results = load_existing_results()

    mfgp_r2 = [r['r2'] for r in existing_results['mfgp']]
    joint_r2 = [r['r2'] for r in existing_results['dngo_joint']]

    print(f"MFGP:       Mean R² = {np.mean(mfgp_r2):.4f} ± {np.std(mfgp_r2):.4f}")
    print(f"DNGO-Joint: Mean R² = {np.mean(joint_r2):.4f} ± {np.std(joint_r2):.4f}")

    # Load data
    print("\nLoading data...")
    lookup, all_combinations, _ = load_base_data()
    print(f"Total compositions: {len(all_combinations)}")

    # Define new methods
    methods = {
        'knowledge_distillation': {
            'train_func': train_knowledge_distillation,
            'params': {
                'alpha_kd': 0.3, 'temperature': 3.0,
                'lf_lr': 1e-3, 'hf_lr': 1e-4,
                'lf_epochs': 200, 'hf_epochs': 100,
                'lf_hidden': 64, 'hf_hidden': 64,
                'lf_layers': 2, 'hf_layers': 2
            },
            'model_type': 'mf'
        },
        'domain_adaptation_mmd': {
            'train_func': train_domain_adaptation_mmd,
            'params': {
                'lambda_mmd': 0.1, 'mmd_bandwidth': 1.0,
                'lf_lr': 1e-3, 'hf_lr': 1e-4,
                'lf_epochs': 200, 'hf_epochs': 100,
                'lf_hidden': 64, 'hf_hidden': 64,
                'lf_layers': 2, 'hf_layers': 2
            },
            'model_type': 'mf'
        },
        'soft_parameter_sharing': {
            'train_func': train_soft_parameter_sharing,
            'params': {
                'lambda_soft': 0.01, 'alpha': 0.5,
                'lf_lr': 1e-3, 'hf_lr': 1e-4,
                'epochs': 200,
                'lf_hidden': 64, 'hf_hidden': 64,
                'lf_layers': 2, 'hf_layers': 2
            },
            'model_type': 'mf'
        },
        'pseudo_labeling': {
            'train_func': train_pseudo_labeling,
            'params': {
                'confidence_threshold': 0.8, 'pseudo_weight': 0.5,
                'lf_lr': 1e-3, 'hf_lr': 1e-4,
                'lf_epochs': 200, 'hf_epochs': 100,
                'lf_hidden': 64, 'hf_hidden': 64,
                'lf_layers': 2, 'hf_layers': 2
            },
            'model_type': 'mf'
        },
        'adapter': {
            'train_func': train_adapter,
            'params': {
                'lf_lr': 1e-3, 'adapter_lr': 1e-3,
                'lf_epochs': 200, 'adapter_epochs': 100,
                'hidden_dim': 64, 'num_layers': 2,
                'bottleneck_dim': 16
            },
            'model_type': 'adapter'
        },
        'maml': {
            'train_func': train_maml,
            'params': {
                'inner_lr': 0.01, 'outer_lr': 1e-3,
                'meta_epochs': 100, 'inner_steps': 5,
                'n_tasks': 8, 'task_size': 9,
                'finetune_epochs': 50, 'finetune_lr': 1e-4,
                'lf_hidden': 64, 'hf_hidden': 64,
                'lf_layers': 2, 'hf_layers': 2
            },
            'model_type': 'mf'
        }
    }

    # Run experiments
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / 'visualizations' / f'{timestamp}_advanced_tl_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {method: [] for method in methods}

    print("\n" + "="*70)
    print("Running New Advanced Transfer Learning Methods")
    print("="*70)

    for fold_idx, seed in enumerate(SEEDS, 1):
        print(f"\nFold {fold_idx}/{len(SEEDS)} (seed={seed})")
        print("-"*50)

        data = generate_data(lookup, all_combinations, n_lofi=N_LOFI, n_hifi=N_HIFI, seed=seed)

        for method_name, config in methods.items():
            try:
                results = run_experiment(
                    method_name,
                    config['train_func'],
                    data,
                    config['params'],
                    device,
                    seed,
                    config['model_type']
                )
                results['fold'] = fold_idx
                results['seed'] = seed
                all_results[method_name].append(results)
                print(f"  {method_name:<25}: R²={results['r2']:.4f}, RMSE={results['rmse']:.4f}")
            except Exception as e:
                print(f"  {method_name:<25}: ERROR - {e}")
                import traceback
                traceback.print_exc()
                all_results[method_name].append({'r2': np.nan, 'rmse': np.nan, 'fold': fold_idx, 'seed': seed})

    # Add existing results
    all_results['mfgp'] = existing_results['mfgp']
    all_results['dngo_joint'] = existing_results['dngo_joint']

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY (All Methods)")
    print("="*70)
    print(f"\n{'Method':<30} {'Mean R²':<12} {'Std R²':<12} {'Mean RMSE':<12}")
    print("-"*70)

    summary_data = []
    for method_name, results in all_results.items():
        r2_values = [r['r2'] for r in results if not np.isnan(r['r2'])]
        rmse_values = [r.get('rmse', np.nan) for r in results if not np.isnan(r.get('rmse', np.nan))]

        if r2_values:
            mean_r2 = np.mean(r2_values)
            std_r2 = np.std(r2_values)
            mean_rmse = np.mean(rmse_values) if rmse_values else np.nan
            print(f"{method_name:<30} {mean_r2:<12.4f} {std_r2:<12.4f} {mean_rmse:<12.4f}")
            summary_data.append({
                'method': method_name,
                'mean_r2': mean_r2,
                'std_r2': std_r2,
                'mean_rmse': mean_rmse
            })

    # Save results
    # Detailed results
    detailed_results = []
    for method_name, results in all_results.items():
        for r in results:
            detailed_results.append({
                'method': method_name,
                'fold': r.get('fold', 0),
                'seed': r.get('seed', 0),
                'r2': r['r2'],
                'rmse': r.get('rmse', np.nan)
            })

    df_detailed = pd.DataFrame(detailed_results)
    df_detailed.to_csv(output_dir / 'detailed_results.csv', index=False)

    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values('mean_r2', ascending=False)
    df_summary.to_csv(output_dir / 'summary_results.csv', index=False)

    # Create visualization
    create_comparison_visualization(all_results, output_dir)

    print(f"\nResults saved to: {output_dir}")

    # Print ranking
    print("\n" + "="*70)
    print("RANKING (by Mean R²)")
    print("="*70)
    for rank, row in enumerate(df_summary.itertuples(), 1):
        marker = "★" if row.method in ['mfgp', 'dngo_joint'] else " "
        print(f"{rank}. {marker} {row.method:<28} R²={row.mean_r2:.4f} ± {row.std_r2:.4f}")

    return all_results


if __name__ == '__main__':
    main()
