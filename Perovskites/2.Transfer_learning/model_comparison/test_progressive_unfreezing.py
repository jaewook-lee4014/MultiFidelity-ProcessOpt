#!/usr/bin/env python
"""
Test Progressive Unfreezing vs Static Unfreezing vs No Freeze (baseline)
"""
import sys
import os
import numpy as np
import torch
import pickle
import json
from pathlib import Path
from scipy.stats import spearmanr
import time
import warnings
warnings.filterwarnings('ignore')

# Add parent path
sys.path.insert(0, str(Path(__file__).parent.parent / 'Pure_TL_BO'))

from DNGO.models import TransferLearningDNN


def load_data():
    """Load perovskite data"""
    data_path = Path(__file__).parent.parent.parent / '0.Data'

    with open(data_path / 'lookup_table.pkl', 'rb') as f:
        lookup = pickle.load(f)

    with open(data_path / 'organics.json', 'r') as f:
        organics_map = json.load(f)
    with open(data_path / 'cations.json', 'r') as f:
        cations_map = json.load(f)
    with open(data_path / 'anions.json', 'r') as f:
        anions_map = json.load(f)

    param_space = {
        'organic': list(organics_map.keys()),
        'cation': list(cations_map.keys()),
        'anion': list(anions_map.keys())
    }

    return lookup, param_space


def generate_dataset(lookup, param_space, n_lofi: int, n_hifi: int,
                     random_state: int = None):
    """Generate random train dataset"""
    rng = np.random.default_rng(random_state)

    organics = param_space['organic']
    cations = param_space['cation']
    anions = param_space['anion']

    all_combinations = []
    for i, org in enumerate(organics, 1):
        for j, cat in enumerate(cations, 1):
            for k, ani in enumerate(anions, 1):
                all_combinations.append({
                    'labels': [i, j, k],
                    'names': [org, cat, ani]
                })

    n_total = len(all_combinations)
    lofi_indices = rng.choice(n_total, size=min(n_lofi, n_total), replace=False)
    hifi_indices = rng.choice(n_total, size=min(n_hifi, n_total), replace=False)

    X_low, y_low = [], []
    for idx in lofi_indices:
        combo = all_combinations[idx]
        X_low.append(combo['labels'])
        org, cat, ani = combo['names']
        bandgap = np.amin(lookup[org.capitalize()][cat][ani]['bandgap_gga'])
        y_low.append(bandgap)

    X_high, y_high = [], []
    for idx in hifi_indices:
        combo = all_combinations[idx]
        X_high.append(combo['labels'])
        org, cat, ani = combo['names']
        bandgap = np.amin(lookup[org.capitalize()][cat][ani]['bandgap_hse06'])
        y_high.append(bandgap)

    return {
        'X_low': np.array(X_low, dtype=np.float32),
        'y_low': np.array(y_low, dtype=np.float32),
        'X_high': np.array(X_high, dtype=np.float32),
        'y_high': np.array(y_high, dtype=np.float32)
    }


def generate_test_data(lookup, param_space):
    """Generate test data for all combinations"""
    organics = param_space['organic']
    cations = param_space['cation']
    anions = param_space['anion']

    all_X, all_y_hifi = [], []

    for i, org in enumerate(organics, 1):
        for j, cat in enumerate(cations, 1):
            for k, ani in enumerate(anions, 1):
                all_X.append([i, j, k])
                all_y_hifi.append(np.amin(lookup[org.capitalize()][cat][ani]['bandgap_hse06']))

    return np.array(all_X, dtype=np.float32), np.array(all_y_hifi, dtype=np.float32)


def evaluate_model(model, X_test, y_test):
    """Calculate RMSE, R2, and Spearman correlation"""
    predictions = model.predict(X_test)

    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
    ss_res = np.sum((y_test - predictions) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    sp, _ = spearmanr(predictions, y_test)

    return rmse, r2, sp if not np.isnan(sp) else 0.0


def run_comparison(n_runs=5, n_lofi=72, n_hifi=9, bo_trials=10):
    """Compare Progressive Unfreezing vs Static Unfreezing vs Baseline"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Runs: {n_runs}")
    print(f"Data: {n_lofi} LOFI, {n_hifi} HIFI samples")
    print(f"BO trials: {bo_trials}")
    print("=" * 60)

    # Load data
    lookup, param_space = load_data()
    X_test, y_test = generate_test_data(lookup, param_space)
    print(f"Test set: {len(X_test)} samples")

    # Results storage
    results = {
        'baseline': {'rmse': [], 'r2': [], 'spearman': [], 'time': []},
        'static_freeze': {'rmse': [], 'r2': [], 'spearman': [], 'time': []},
        'progressive': {'rmse': [], 'r2': [], 'spearman': [], 'time': []}
    }

    for run in range(n_runs):
        print(f"\n{'='*60}")
        print(f"RUN {run+1}/{n_runs}")
        print(f"{'='*60}")

        # Generate random training data
        dataset = generate_dataset(lookup, param_space, n_lofi, n_hifi, random_state=run*42)
        X_lofi, y_lofi = dataset['X_low'], dataset['y_low']
        X_hifi, y_hifi = dataset['X_high'], dataset['y_high']
        print(f"Dataset: {len(X_lofi)} LOFI, {len(X_hifi)} HIFI")

        # 1. Baseline (No Freeze)
        print("\n1. Baseline (No Freeze)...")
        start = time.time()
        model_baseline = TransferLearningDNN(
            input_dim=X_lofi.shape[1],
            use_hyperparameter_bo=True,
            device=device
        )
        model_baseline.pretrain(X_lofi, y_lofi, bo_trials=bo_trials, verbose=False)
        model_baseline.finetune(X_hifi, y_hifi, bo_trials=bo_trials, verbose=False,
                               use_freeze=False, use_progressive_unfreezing=False)
        elapsed = time.time() - start
        rmse, r2, sp = evaluate_model(model_baseline, X_test, y_test)
        results['baseline']['rmse'].append(rmse)
        results['baseline']['r2'].append(r2)
        results['baseline']['spearman'].append(sp)
        results['baseline']['time'].append(elapsed)
        print(f"   RMSE: {rmse:.4f}, R²: {r2:.4f}, ρ: {sp:.4f}, Time: {elapsed:.1f}s")

        # 2. Static Freeze (use_freeze=True)
        print("\n2. Static Freeze...")
        start = time.time()
        model_static = TransferLearningDNN(
            input_dim=X_lofi.shape[1],
            use_hyperparameter_bo=True,
            device=device
        )
        model_static.pretrain(X_lofi, y_lofi, bo_trials=bo_trials, verbose=False)
        model_static.finetune(X_hifi, y_hifi, bo_trials=bo_trials, verbose=False,
                             use_freeze=True, use_progressive_unfreezing=False)
        elapsed = time.time() - start
        rmse, r2, sp = evaluate_model(model_static, X_test, y_test)
        results['static_freeze']['rmse'].append(rmse)
        results['static_freeze']['r2'].append(r2)
        results['static_freeze']['spearman'].append(sp)
        results['static_freeze']['time'].append(elapsed)
        print(f"   RMSE: {rmse:.4f}, R²: {r2:.4f}, ρ: {sp:.4f}, Time: {elapsed:.1f}s")

        # 3. Progressive Unfreezing
        print("\n3. Progressive Unfreezing...")
        start = time.time()
        model_progressive = TransferLearningDNN(
            input_dim=X_lofi.shape[1],
            use_hyperparameter_bo=True,
            device=device
        )
        model_progressive.pretrain(X_lofi, y_lofi, bo_trials=bo_trials, verbose=False)
        model_progressive.finetune(X_hifi, y_hifi, bo_trials=bo_trials, verbose=False,
                                   use_freeze=False, use_progressive_unfreezing=True)
        elapsed = time.time() - start
        rmse, r2, sp = evaluate_model(model_progressive, X_test, y_test)
        results['progressive']['rmse'].append(rmse)
        results['progressive']['r2'].append(r2)
        results['progressive']['spearman'].append(sp)
        results['progressive']['time'].append(elapsed)
        print(f"   RMSE: {rmse:.4f}, R²: {r2:.4f}, ρ: {sp:.4f}, Time: {elapsed:.1f}s")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'RMSE':>12} {'R²':>12} {'Spearman':>12} {'Time':>10}")
    print("-" * 70)

    for method, data in results.items():
        rmse_mean = np.mean(data['rmse'])
        rmse_std = np.std(data['rmse'])
        r2_mean = np.mean(data['r2'])
        r2_std = np.std(data['r2'])
        sp_mean = np.mean(data['spearman'])
        sp_std = np.std(data['spearman'])
        time_mean = np.mean(data['time'])

        print(f"{method:<25} {rmse_mean:>5.4f}±{rmse_std:.4f} {r2_mean:>5.4f}±{r2_std:.4f} "
              f"{sp_mean:>5.4f}±{sp_std:.4f} {time_mean:>8.1f}s")

    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-runs', type=int, default=5)
    parser.add_argument('--n-lofi', type=int, default=72)
    parser.add_argument('--n-hifi', type=int, default=9)
    parser.add_argument('--bo-trials', type=int, default=10)
    args = parser.parse_args()

    results = run_comparison(
        n_runs=args.n_runs,
        n_lofi=args.n_lofi,
        n_hifi=args.n_hifi,
        bo_trials=args.bo_trials
    )
