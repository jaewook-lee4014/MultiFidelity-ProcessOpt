#!/usr/bin/env python3
"""
Hyperparameter Tuning for Advanced Transfer Learning Methods using Optuna

Uses Leave-One-Out Cross-Validation (LOOCV) on HF data to avoid data leakage.

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
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from advanced_transfer_learning import (
    load_base_data, generate_data, set_seeds,
    MultiFidelityNetwork, AdapterNetwork,
    train_knowledge_distillation,
    train_domain_adaptation_mmd,
    train_soft_parameter_sharing,
    train_pseudo_labeling,
    train_adapter,
    train_maml,
)

# ============================================================================
# Configuration
# ============================================================================

N_LOFI = 72
N_HIFI = 9
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
N_OPTUNA_TRIALS = 200  # Number of trials per method (increased for thorough search)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Path to existing baseline results
EXISTING_RESULTS_PATH = Path(__file__).parent / 'visualizations' / '20251211_163454_all_6methods' / 'results_summary.csv'


def evaluate_model_loocv(model, X_hf, y_hf, X_lf_for_pred, device, model_type='mf'):
    """Evaluate model using LOOCV on HF data"""
    model.eval()

    with torch.no_grad():
        X_t = torch.tensor(X_hf, dtype=torch.float32).to(device)

        if model_type == 'mf':
            y_pred = model.forward_hf(X_t).cpu().numpy().flatten()
        elif model_type == 'adapter':
            h = X_t
            layer_idx = 0
            for module in model.feature_net:
                h = module(h)
                if isinstance(module, (nn.ReLU, nn.Tanh)):
                    if layer_idx < len(model.adapters):
                        h = model.adapters[layer_idx](h)
                        layer_idx += 1
            y_pred = model.hf_out(h).cpu().numpy().flatten()
        else:
            y_pred = model(X_t).cpu().numpy().flatten()

    rmse = np.sqrt(mean_squared_error(y_hf, y_pred))
    r2 = r2_score(y_hf, y_pred)

    return r2, rmse


def train_and_evaluate(method_name, train_func, params, data, device, model_type='mf'):
    """Train model and evaluate using LOOCV"""
    set_seeds(42)

    # Create model
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
    try:
        train_func(model, data['X_low'], data['y_low'],
                   data['X_high'], data['y_high'], params, device)
    except Exception as e:
        return -10.0, 10.0  # Return bad score on error

    # Evaluate on test set
    test_mask = np.ones(len(data['X_all']), dtype=bool)
    test_mask[data['hifi_idx']] = False

    X_test = data['X_all'][test_mask]
    y_test = data['y_all'][test_mask]

    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)

        if model_type == 'mf':
            y_pred = model.forward_hf(X_t).cpu().numpy().flatten()
        elif model_type == 'adapter':
            h = X_t
            layer_idx = 0
            for module in model.feature_net:
                h = module(h)
                if isinstance(module, (nn.ReLU, nn.Tanh)):
                    if layer_idx < len(model.adapters):
                        h = model.adapters[layer_idx](h)
                        layer_idx += 1
            y_pred = model.hf_out(h).cpu().numpy().flatten()
        else:
            y_pred = model(X_t).cpu().numpy().flatten()

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return r2, rmse


# ============================================================================
# Optuna Objective Functions
# ============================================================================

def create_kd_objective(data, device):
    """Knowledge Distillation objective - Extended search space"""
    def objective(trial):
        params = {
            # KD-specific params (wider range)
            'alpha_kd': trial.suggest_float('alpha_kd', 0.05, 0.95),
            'temperature': trial.suggest_float('temperature', 0.5, 20.0),
            # Learning rates (wider range with log scale)
            'lf_lr': trial.suggest_float('lf_lr', 1e-5, 5e-2, log=True),
            'hf_lr': trial.suggest_float('hf_lr', 1e-6, 1e-2, log=True),
            # Epochs
            'lf_epochs': trial.suggest_int('lf_epochs', 50, 500),
            'hf_epochs': trial.suggest_int('hf_epochs', 20, 300),
            # Architecture
            'lf_hidden': trial.suggest_categorical('lf_hidden', [16, 32, 64, 128, 256]),
            'hf_hidden': trial.suggest_categorical('hf_hidden', [16, 32, 64, 128, 256]),
            'lf_layers': trial.suggest_int('lf_layers', 1, 4),
            'hf_layers': trial.suggest_int('hf_layers', 1, 4),
            # Regularization
            'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        }

        r2, _ = train_and_evaluate('kd', train_knowledge_distillation, params, data, device, 'mf')
        return r2

    return objective


def create_mmd_objective(data, device):
    """Domain Adaptation MMD objective - Extended search space"""
    def objective(trial):
        params = {
            # MMD-specific params (much wider range)
            'lambda_mmd': trial.suggest_float('lambda_mmd', 1e-5, 10.0, log=True),
            'mmd_bandwidth': trial.suggest_float('mmd_bandwidth', 0.01, 50.0, log=True),
            # Learning rates
            'lf_lr': trial.suggest_float('lf_lr', 1e-5, 5e-2, log=True),
            'hf_lr': trial.suggest_float('hf_lr', 1e-6, 1e-2, log=True),
            # Epochs
            'lf_epochs': trial.suggest_int('lf_epochs', 50, 500),
            'hf_epochs': trial.suggest_int('hf_epochs', 20, 300),
            # Architecture
            'lf_hidden': trial.suggest_categorical('lf_hidden', [16, 32, 64, 128, 256]),
            'hf_hidden': trial.suggest_categorical('hf_hidden', [16, 32, 64, 128, 256]),
            'lf_layers': trial.suggest_int('lf_layers', 1, 4),
            'hf_layers': trial.suggest_int('hf_layers', 1, 4),
            # Regularization
            'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        }

        r2, _ = train_and_evaluate('mmd', train_domain_adaptation_mmd, params, data, device, 'mf')
        return r2

    return objective


def create_soft_sharing_objective(data, device):
    """Soft Parameter Sharing objective - Extended search space"""
    def objective(trial):
        params = {
            # Soft sharing params (wider range)
            'lambda_soft': trial.suggest_float('lambda_soft', 1e-5, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 0.01, 0.99),
            # Learning rates
            'lf_lr': trial.suggest_float('lf_lr', 1e-5, 5e-2, log=True),
            'hf_lr': trial.suggest_float('hf_lr', 1e-6, 1e-2, log=True),
            # Epochs
            'epochs': trial.suggest_int('epochs', 50, 600),
            # Architecture
            'lf_hidden': trial.suggest_categorical('lf_hidden', [16, 32, 64, 128, 256]),
            'hf_hidden': trial.suggest_categorical('hf_hidden', [16, 32, 64, 128, 256]),
            'lf_layers': trial.suggest_int('lf_layers', 1, 4),
            'hf_layers': trial.suggest_int('hf_layers', 1, 4),
            # Regularization
            'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        }

        r2, _ = train_and_evaluate('soft', train_soft_parameter_sharing, params, data, device, 'mf')
        return r2

    return objective


def create_pseudo_labeling_objective(data, device):
    """Pseudo-Labeling objective - Extended search space"""
    def objective(trial):
        params = {
            # PL-specific params (wider range)
            'confidence_threshold': trial.suggest_float('confidence_threshold', 0.1, 5.0),
            'pseudo_weight': trial.suggest_float('pseudo_weight', 0.01, 2.0, log=True),
            # Learning rates
            'lf_lr': trial.suggest_float('lf_lr', 1e-5, 5e-2, log=True),
            'hf_lr': trial.suggest_float('hf_lr', 1e-6, 1e-2, log=True),
            # Epochs
            'lf_epochs': trial.suggest_int('lf_epochs', 50, 500),
            'hf_epochs': trial.suggest_int('hf_epochs', 20, 300),
            # Architecture
            'lf_hidden': trial.suggest_categorical('lf_hidden', [16, 32, 64, 128, 256]),
            'hf_hidden': trial.suggest_categorical('hf_hidden', [16, 32, 64, 128, 256]),
            'lf_layers': trial.suggest_int('lf_layers', 1, 4),
            'hf_layers': trial.suggest_int('hf_layers', 1, 4),
            # Regularization
            'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        }

        r2, _ = train_and_evaluate('pl', train_pseudo_labeling, params, data, device, 'mf')
        return r2

    return objective


def create_adapter_objective(data, device):
    """Adapter-based Transfer objective - Extended search space"""
    def objective(trial):
        params = {
            # Learning rates
            'lf_lr': trial.suggest_float('lf_lr', 1e-5, 5e-2, log=True),
            'adapter_lr': trial.suggest_float('adapter_lr', 1e-5, 5e-2, log=True),
            # Epochs
            'lf_epochs': trial.suggest_int('lf_epochs', 50, 500),
            'adapter_epochs': trial.suggest_int('adapter_epochs', 20, 300),
            # Architecture
            'hidden_dim': trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256]),
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'bottleneck_dim': trial.suggest_categorical('bottleneck_dim', [4, 8, 16, 32, 64, 128]),
            # Regularization
            'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        }

        r2, _ = train_and_evaluate('adapter', train_adapter, params, data, device, 'adapter')
        return r2

    return objective


def create_maml_objective(data, device):
    """MAML objective - Extended search space"""
    def objective(trial):
        params = {
            # MAML-specific params (critical for performance)
            'inner_lr': trial.suggest_float('inner_lr', 1e-4, 0.5, log=True),
            'outer_lr': trial.suggest_float('outer_lr', 1e-5, 5e-2, log=True),
            'meta_epochs': trial.suggest_int('meta_epochs', 20, 300),
            'inner_steps': trial.suggest_int('inner_steps', 1, 15),
            'n_tasks': trial.suggest_int('n_tasks', 2, 20),
            'task_size': trial.suggest_int('task_size', 4, 18),
            # Finetune params
            'finetune_epochs': trial.suggest_int('finetune_epochs', 10, 200),
            'finetune_lr': trial.suggest_float('finetune_lr', 1e-6, 1e-2, log=True),
            # Architecture
            'lf_hidden': trial.suggest_categorical('lf_hidden', [16, 32, 64, 128, 256]),
            'hf_hidden': trial.suggest_categorical('hf_hidden', [16, 32, 64, 128, 256]),
            'lf_layers': trial.suggest_int('lf_layers', 1, 4),
            'hf_layers': trial.suggest_int('hf_layers', 1, 4),
            # Regularization
            'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        }

        r2, _ = train_and_evaluate('maml', train_maml, params, data, device, 'mf')
        return r2

    return objective


# ============================================================================
# Final Evaluation with Best Params
# ============================================================================

def evaluate_with_best_params(method_name, train_func, best_params, lookup, all_combinations, device, model_type='mf'):
    """Evaluate method with best params across all folds"""
    results = []

    for fold_idx, seed in enumerate(SEEDS, 1):
        set_seeds(seed)
        data = generate_data(lookup, all_combinations, n_lofi=N_LOFI, n_hifi=N_HIFI, seed=seed)

        # Create model
        if model_type == 'adapter':
            model = AdapterNetwork(
                input_dim=3,
                hidden_dim=best_params.get('hidden_dim', 64),
                num_layers=best_params.get('num_layers', 2),
                bottleneck_dim=best_params.get('bottleneck_dim', 16),
                dropout=best_params.get('dropout', 0.0),
                activation=best_params.get('activation', 'relu')
            )
        else:
            model = MultiFidelityNetwork(
                input_dim=3,
                lf_hidden=best_params.get('lf_hidden', 64),
                hf_hidden=best_params.get('hf_hidden', 64),
                lf_layers=best_params.get('lf_layers', 2),
                hf_layers=best_params.get('hf_layers', 2),
                dropout=best_params.get('dropout', 0.0),
                activation=best_params.get('activation', 'relu'),
                residual_mode=True
            )
        model.to(device)

        # Train
        try:
            train_func(model, data['X_low'], data['y_low'],
                      data['X_high'], data['y_high'], best_params, device)
        except Exception as e:
            print(f"  Fold {fold_idx}: ERROR - {e}")
            results.append({'fold': fold_idx, 'seed': seed, 'r2': np.nan, 'rmse': np.nan})
            continue

        # Evaluate
        test_mask = np.ones(len(data['X_all']), dtype=bool)
        test_mask[data['hifi_idx']] = False
        X_test = data['X_all'][test_mask]
        y_test = data['y_all'][test_mask]

        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32).to(device)

            if model_type == 'mf':
                y_pred = model.forward_hf(X_t).cpu().numpy().flatten()
            elif model_type == 'adapter':
                h = X_t
                layer_idx = 0
                for module in model.feature_net:
                    h = module(h)
                    if isinstance(module, (nn.ReLU, nn.Tanh)):
                        if layer_idx < len(model.adapters):
                            h = model.adapters[layer_idx](h)
                            layer_idx += 1
                y_pred = model.hf_out(h).cpu().numpy().flatten()

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({'fold': fold_idx, 'seed': seed, 'r2': r2, 'rmse': rmse})
        print(f"  Fold {fold_idx}: R²={r2:.4f}, RMSE={rmse:.4f}")

    return results


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


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*70)
    print("Advanced Transfer Learning - Hyperparameter Tuning with Optuna")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Optuna Trials per Method: {N_OPTUNA_TRIALS}")
    print(f"Data: {N_LOFI} LF + {N_HIFI} HF")
    print(f"Folds for Final Evaluation: {len(SEEDS)}")

    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / 'visualizations' / f'{timestamp}_advanced_tl_hp_tuning'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    lookup, all_combinations, _ = load_base_data()
    print(f"Total compositions: {len(all_combinations)}")

    # Use first seed for HP tuning
    data = generate_data(lookup, all_combinations, n_lofi=N_LOFI, n_hifi=N_HIFI, seed=42)

    # Define methods
    methods = {
        'knowledge_distillation': {
            'create_objective': create_kd_objective,
            'train_func': train_knowledge_distillation,
            'model_type': 'mf'
        },
        'domain_adaptation_mmd': {
            'create_objective': create_mmd_objective,
            'train_func': train_domain_adaptation_mmd,
            'model_type': 'mf'
        },
        'soft_parameter_sharing': {
            'create_objective': create_soft_sharing_objective,
            'train_func': train_soft_parameter_sharing,
            'model_type': 'mf'
        },
        'pseudo_labeling': {
            'create_objective': create_pseudo_labeling_objective,
            'train_func': train_pseudo_labeling,
            'model_type': 'mf'
        },
        'adapter': {
            'create_objective': create_adapter_objective,
            'train_func': train_adapter,
            'model_type': 'adapter'
        },
        'maml': {
            'create_objective': create_maml_objective,
            'train_func': train_maml,
            'model_type': 'mf'
        }
    }

    # Store best params and results
    best_params_all = {}
    all_results = {}

    # Run HP tuning for each method
    for method_name, config in methods.items():
        print("\n" + "="*70)
        print(f"Tuning: {method_name}")
        print("="*70)

        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )

        # Optimize
        objective = config['create_objective'](data, DEVICE)
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

        # Get best params
        best_params = study.best_params
        best_value = study.best_value

        print(f"\nBest R² during tuning: {best_value:.4f}")
        print(f"Best params: {best_params}")

        best_params_all[method_name] = best_params

        # Save study results
        study_df = study.trials_dataframe()
        study_df.to_csv(output_dir / f'{method_name}_optuna_trials.csv', index=False)

        # Final evaluation with best params across all folds
        print(f"\nFinal evaluation with best params (10 folds):")
        results = evaluate_with_best_params(
            method_name,
            config['train_func'],
            best_params,
            lookup,
            all_combinations,
            DEVICE,
            config['model_type']
        )

        all_results[method_name] = results

        r2_values = [r['r2'] for r in results if not np.isnan(r['r2'])]
        print(f"\n{method_name} Final: Mean R² = {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")

    # Load baseline results
    print("\n" + "="*70)
    print("Loading baseline results (MFGP, DNGO-Joint)")
    print("="*70)

    baseline_results = load_existing_results()
    all_results['mfgp'] = baseline_results['mfgp']
    all_results['dngo_joint'] = baseline_results['dngo_joint']

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY (All Methods with Tuned HP)")
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

    # Save best params
    with open(output_dir / 'best_params.json', 'w') as f:
        json.dump(best_params_all, f, indent=2)

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

    return all_results, best_params_all


def create_comparison_visualization(all_results, output_dir):
    """Create comparison visualization"""

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
    ax1.set_title('Model Comparison: R² Score (HP Tuned, 10-Fold CV)', fontsize=14)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylim(-0.5, 1.0)

    for bar, val, err in zip(bars, mean_r2, std_r2):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Baseline (existing)'),
        Patch(facecolor='#3498db', edgecolor='black', label='New TL methods (HP tuned)')
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
    ax2.set_title('R² Distribution Across Folds (HP Tuned)', fontsize=14)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_hp_tuned.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {output_dir / 'comparison_hp_tuned.png'}")


if __name__ == '__main__':
    main()
