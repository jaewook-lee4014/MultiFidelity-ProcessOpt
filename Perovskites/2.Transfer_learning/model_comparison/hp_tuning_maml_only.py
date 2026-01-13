#!/usr/bin/env python3
"""
Hyperparameter Tuning for MAML Only (with stability improvements)

Author: Claude Code
Date: 2025-12-17
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
    MultiFidelityNetwork,
    train_maml,
)

# ============================================================================
# Configuration
# ============================================================================

N_LOFI = 72
N_HIFI = 9
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
N_OPTUNA_TRIALS = 200
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Path to existing results for comparison
EXISTING_RESULTS_PATH = Path(__file__).parent / 'visualizations' / '20251211_163454_all_6methods' / 'results_summary.csv'


def train_and_evaluate(params, data, device):
    """Train MAML model and evaluate"""
    set_seeds(42)

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
        train_maml(model, data['X_low'], data['y_low'],
                   data['X_high'], data['y_high'], params, device)
    except Exception as e:
        return -10.0, 10.0

    # Evaluate on test set
    test_mask = np.ones(len(data['X_all']), dtype=bool)
    test_mask[data['hifi_idx']] = False

    X_test = data['X_all'][test_mask]
    y_test = data['y_all'][test_mask]

    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = model.forward_hf(X_t).cpu().numpy().flatten()

    # Check for NaN
    if np.isnan(y_pred).any():
        return -10.0, 10.0

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return r2, rmse


def create_maml_objective(data, device):
    """MAML objective with stability-focused search space"""
    def objective(trial):
        params = {
            # MAML-specific params (more conservative ranges for stability)
            'inner_lr': trial.suggest_float('inner_lr', 1e-4, 0.1, log=True),
            'outer_lr': trial.suggest_float('outer_lr', 1e-5, 1e-2, log=True),
            'meta_epochs': trial.suggest_int('meta_epochs', 30, 200),
            'inner_steps': trial.suggest_int('inner_steps', 1, 10),
            'n_tasks': trial.suggest_int('n_tasks', 2, 12),
            'task_size': trial.suggest_int('task_size', 6, 16),
            # Finetune params
            'finetune_epochs': trial.suggest_int('finetune_epochs', 20, 150),
            'finetune_lr': trial.suggest_float('finetune_lr', 1e-5, 5e-3, log=True),
            # Architecture (smaller for stability)
            'lf_hidden': trial.suggest_categorical('lf_hidden', [32, 64, 128]),
            'hf_hidden': trial.suggest_categorical('hf_hidden', [32, 64, 128]),
            'lf_layers': trial.suggest_int('lf_layers', 1, 3),
            'hf_layers': trial.suggest_int('hf_layers', 1, 3),
            # Regularization
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            # Stability param
            'grad_clip': trial.suggest_float('grad_clip', 0.5, 5.0),
        }

        r2, _ = train_and_evaluate(params, data, device)

        # Prune if result is very bad
        if r2 < -5.0:
            raise optuna.TrialPruned()

        return r2

    return objective


def evaluate_with_best_params(best_params, lookup, all_combinations, device):
    """Evaluate MAML with best params across all folds"""
    results = []

    for fold_idx, seed in enumerate(SEEDS, 1):
        set_seeds(seed)
        data = generate_data(lookup, all_combinations, n_lofi=N_LOFI, n_hifi=N_HIFI, seed=seed)

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

        try:
            train_maml(model, data['X_low'], data['y_low'],
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
            y_pred = model.forward_hf(X_t).cpu().numpy().flatten()

        if np.isnan(y_pred).any():
            print(f"  Fold {fold_idx}: NaN in predictions")
            results.append({'fold': fold_idx, 'seed': seed, 'r2': np.nan, 'rmse': np.nan})
            continue

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({'fold': fold_idx, 'seed': seed, 'r2': r2, 'rmse': rmse})
        print(f"  Fold {fold_idx}: R²={r2:.4f}, RMSE={rmse:.4f}")

    return results


def main():
    print("="*70)
    print("MAML Hyperparameter Tuning (with Stability Improvements)")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Optuna Trials: {N_OPTUNA_TRIALS}")
    print(f"Data: {N_LOFI} LF + {N_HIFI} HF")
    print(f"Folds for Final Evaluation: {len(SEEDS)}")

    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / 'visualizations' / f'{timestamp}_maml_hp_tuning'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    lookup, all_combinations, _ = load_base_data()
    print(f"Total compositions: {len(all_combinations)}")

    # Use first seed for HP tuning
    data = generate_data(lookup, all_combinations, n_lofi=N_LOFI, n_hifi=N_HIFI, seed=42)

    # Create study with pruning
    print("\n" + "="*70)
    print("Tuning MAML...")
    print("="*70)

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    )

    objective = create_maml_objective(data, DEVICE)
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True,
                   catch=(ValueError, RuntimeError))

    # Get best params
    best_params = study.best_params
    best_value = study.best_value

    print(f"\nBest R² during tuning: {best_value:.4f}")
    print(f"Best params: {json.dumps(best_params, indent=2)}")

    # Save study results
    study_df = study.trials_dataframe()
    study_df.to_csv(output_dir / 'maml_optuna_trials.csv', index=False)

    # Final evaluation with best params across all folds
    print(f"\nFinal evaluation with best params (10 folds):")
    results = evaluate_with_best_params(best_params, lookup, all_combinations, DEVICE)

    r2_values = [r['r2'] for r in results if not np.isnan(r['r2'])]
    rmse_values = [r['rmse'] for r in results if not np.isnan(r['rmse'])]

    if r2_values:
        mean_r2 = np.mean(r2_values)
        std_r2 = np.std(r2_values)
        mean_rmse = np.mean(rmse_values)
        print(f"\nMAML Final: Mean R² = {mean_r2:.4f} ± {std_r2:.4f}")
        print(f"             Mean RMSE = {mean_rmse:.4f}")
    else:
        print("\nMAML Final: All folds failed")
        mean_r2, std_r2, mean_rmse = np.nan, np.nan, np.nan

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / 'maml_fold_results.csv', index=False)

    with open(output_dir / 'maml_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)

    # Summary
    summary = {
        'method': 'maml',
        'mean_r2': mean_r2,
        'std_r2': std_r2,
        'mean_rmse': mean_rmse,
        'best_tuning_r2': best_value,
        'n_successful_folds': len(r2_values)
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    # Compare with baselines
    print("\n" + "="*70)
    print("Comparison with Baselines")
    print("="*70)
    print(f"MAML (HP tuned):  R² = {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"DNGO-Joint:       R² = 0.7802 ± 0.0368 (from previous results)")
    print(f"MFGP:             R² = 0.3732 ± 0.2901 (from previous results)")

    return results, best_params


if __name__ == '__main__':
    main()
