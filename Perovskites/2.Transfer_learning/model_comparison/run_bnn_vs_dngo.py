#!/usr/bin/env python
"""
Compare improved BNN (with Scale Mixture Prior + consistent training) vs DNGO

This script compares:
1. DNGO: Deterministic DNN + Bayesian Linear Regression
2. BNN (consistent_bnn): Full BNN with consistent Bayesian training
3. BNN (dngo_style): Deterministic features + BNN head (old approach)
"""

import sys
import os
import numpy as np
import torch
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import time

# Add parent path
sys.path.insert(0, str(Path(__file__).parent.parent / 'Pure_TL_BO'))

from DNGO.models import TransferLearningDNN, BayesianLinearRegression
from BNN.bnn_models import TransferLearningBNN


@dataclass
class EvaluationMetrics:
    rmse: float
    mae: float
    r2: float
    spearman_corr: float
    best_prediction: float
    best_actual: float


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> EvaluationMetrics:
    """Calculate evaluation metrics"""
    from scipy.stats import spearmanr

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # RMSE
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

    # MAE
    mae = np.mean(np.abs(y_pred - y_true))

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Spearman correlation
    spearman_corr, _ = spearmanr(y_true, y_pred)
    if np.isnan(spearman_corr):
        spearman_corr = 0.0

    # Best predictions
    best_pred_idx = np.argmin(y_pred)
    best_prediction = y_pred[best_pred_idx]
    best_actual = y_true[best_pred_idx]

    return EvaluationMetrics(
        rmse=rmse,
        mae=mae,
        r2=r2,
        spearman_corr=spearman_corr,
        best_prediction=best_prediction,
        best_actual=best_actual
    )


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

    label_maps = {
        'organic': {name: i+1 for i, name in enumerate(param_space['organic'])},
        'cation': {name: i+1 for i, name in enumerate(param_space['cation'])},
        'anion': {name: i+1 for i, name in enumerate(param_space['anion'])}
    }

    return lookup, param_space, label_maps


def generate_dataset(lookup, param_space, label_maps, n_lofi: int, n_hifi: int,
                     random_state: int = None) -> Dict:
    """Generate random train dataset"""
    rng = np.random.default_rng(random_state)

    organics = param_space['organic']
    cations = param_space['cation']
    anions = param_space['anion']

    # All combinations
    all_combinations = []
    for i, org in enumerate(organics, 1):
        for j, cat in enumerate(cations, 1):
            for k, ani in enumerate(anions, 1):
                all_combinations.append({
                    'labels': [i, j, k],
                    'names': [org, cat, ani]
                })

    n_total = len(all_combinations)

    # Random sampling
    lofi_indices = rng.choice(n_total, size=min(n_lofi, n_total), replace=False)
    hifi_indices = rng.choice(n_total, size=min(n_hifi, n_total), replace=False)

    # Extract data
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
        'y_high': np.array(y_high, dtype=np.float32),
        'n_lofi': len(X_low),
        'n_hifi': len(X_high)
    }


def generate_test_data(lookup, param_space) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate test data for all combinations"""
    organics = param_space['organic']
    cations = param_space['cation']
    anions = param_space['anion']

    all_X = []
    all_y_hifi = []
    all_y_lofi = []

    for i, org in enumerate(organics, 1):
        for j, cat in enumerate(cations, 1):
            for k, ani in enumerate(anions, 1):
                all_X.append([i, j, k])
                all_y_hifi.append(np.amin(lookup[org.capitalize()][cat][ani]['bandgap_hse06']))
                all_y_lofi.append(np.amin(lookup[org.capitalize()][cat][ani]['bandgap_gga']))

    return (np.array(all_X, dtype=np.float32),
            np.array(all_y_hifi, dtype=np.float32),
            np.array(all_y_lofi, dtype=np.float32))


def run_dngo(dataset: Dict, all_X: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run DNGO model"""
    input_dim = dataset['X_low'].shape[1]

    model = TransferLearningDNN(
        input_dim=input_dim,
        hidden_dim=64,
        device=device
    )

    start_time = time.time()

    # Pretrain
    model.pretrain(
        dataset['X_low'], dataset['y_low'],
        epochs=200, lr=1e-3, verbose=False
    )

    # Finetune
    model.finetune(
        dataset['X_high'], dataset['y_high'],
        epochs=100, lr=1e-4, verbose=False
    )

    # BLR fitting
    features = model.extract_features(dataset['X_high'])
    blr = BayesianLinearRegression(alpha=1.0, beta=25.0)
    blr.fit(features, dataset['y_high'])

    elapsed = time.time() - start_time

    # Predict
    test_features = model.extract_features(all_X)
    y_pred, y_var = blr.predict_batch(test_features)
    y_std = np.sqrt(y_var)

    return y_pred, y_std, elapsed


def run_bnn(dataset: Dict, all_X: np.ndarray, device: str,
            transfer_mode: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run BNN model"""
    input_dim = dataset['X_low'].shape[1]

    model = TransferLearningBNN(
        input_dim=input_dim,
        hidden_dims=[64, 64],
        device=device,
        prior_pi=0.5,
        prior_sigma1=1.0,
        prior_sigma2=0.002,
        kl_weight=1.0,
        transfer_mode=transfer_mode
    )

    start_time = time.time()

    # Pretrain
    model.pretrain(
        dataset['X_low'], dataset['y_low'],
        epochs=200, lr=1e-3, verbose=False
    )

    # Finetune
    model.finetune(
        dataset['X_high'], dataset['y_high'],
        epochs=100, lr=1e-4, verbose=False
    )

    elapsed = time.time() - start_time

    # Predict
    y_pred, y_std = model.predict(all_X, n_samples=50)

    return y_pred, y_std, elapsed


def run_comparison(n_runs: int = 5, n_lofi: int = 72, n_hifi: int = 9, verbose: bool = True):
    """Run comparison experiment"""

    # Device setup
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Using device: {device}")
    print(f"Running {n_runs} experiments with {n_lofi} LOFI, {n_hifi} HIFI samples\n")

    # Load data
    lookup, param_space, label_maps = load_data()
    all_X, all_y_hifi, all_y_lofi = generate_test_data(lookup, param_space)

    print(f"Test set: {len(all_X)} combinations")
    print(f"HIFI bandgap range: [{all_y_hifi.min():.3f}, {all_y_hifi.max():.3f}]")
    print(f"LOFI bandgap range: [{all_y_lofi.min():.3f}, {all_y_lofi.max():.3f}]\n")

    # Results storage
    results = {
        'dngo': {'hifi': [], 'lofi': [], 'time': []},
        'bnn_consistent': {'hifi': [], 'lofi': [], 'time': []},
        'bnn_dngo_style': {'hifi': [], 'lofi': [], 'time': []}
    }

    for run in range(n_runs):
        print(f"{'='*60}")
        print(f"Run {run + 1}/{n_runs}")
        print(f"{'='*60}")

        # Generate dataset
        dataset = generate_dataset(lookup, param_space, label_maps,
                                   n_lofi=n_lofi, n_hifi=n_hifi,
                                   random_state=run * 1000)

        print(f"Dataset: {dataset['n_lofi']} LOFI, {dataset['n_hifi']} HIFI samples\n")

        # 1. DNGO
        print("Running DNGO...")
        try:
            y_pred, y_std, elapsed = run_dngo(dataset, all_X, device)
            metrics_hifi = calculate_metrics(all_y_hifi, y_pred)
            metrics_lofi = calculate_metrics(all_y_lofi, y_pred)
            results['dngo']['hifi'].append(metrics_hifi)
            results['dngo']['lofi'].append(metrics_lofi)
            results['dngo']['time'].append(elapsed)
            print(f"  [HIFI] RMSE: {metrics_hifi.rmse:.4f}, R²: {metrics_hifi.r2:.4f}, ρ: {metrics_hifi.spearman_corr:.4f}")
            print(f"  [LOFI] RMSE: {metrics_lofi.rmse:.4f}, R²: {metrics_lofi.r2:.4f}, ρ: {metrics_lofi.spearman_corr:.4f}")
            print(f"  Time: {elapsed:.2f}s\n")
        except Exception as e:
            print(f"  Error: {e}\n")

        # 2. BNN (consistent_bnn)
        print("Running BNN (consistent_bnn mode)...")
        try:
            y_pred, y_std, elapsed = run_bnn(dataset, all_X, device, 'consistent_bnn')
            metrics_hifi = calculate_metrics(all_y_hifi, y_pred)
            metrics_lofi = calculate_metrics(all_y_lofi, y_pred)
            results['bnn_consistent']['hifi'].append(metrics_hifi)
            results['bnn_consistent']['lofi'].append(metrics_lofi)
            results['bnn_consistent']['time'].append(elapsed)
            print(f"  [HIFI] RMSE: {metrics_hifi.rmse:.4f}, R²: {metrics_hifi.r2:.4f}, ρ: {metrics_hifi.spearman_corr:.4f}")
            print(f"  [LOFI] RMSE: {metrics_lofi.rmse:.4f}, R²: {metrics_lofi.r2:.4f}, ρ: {metrics_lofi.spearman_corr:.4f}")
            print(f"  Time: {elapsed:.2f}s\n")
        except Exception as e:
            print(f"  Error: {e}\n")
            import traceback
            traceback.print_exc()

        # 3. BNN (dngo_style) - old approach
        print("Running BNN (dngo_style mode)...")
        try:
            y_pred, y_std, elapsed = run_bnn(dataset, all_X, device, 'dngo_style')
            metrics_hifi = calculate_metrics(all_y_hifi, y_pred)
            metrics_lofi = calculate_metrics(all_y_lofi, y_pred)
            results['bnn_dngo_style']['hifi'].append(metrics_hifi)
            results['bnn_dngo_style']['lofi'].append(metrics_lofi)
            results['bnn_dngo_style']['time'].append(elapsed)
            print(f"  [HIFI] RMSE: {metrics_hifi.rmse:.4f}, R²: {metrics_hifi.r2:.4f}, ρ: {metrics_hifi.spearman_corr:.4f}")
            print(f"  [LOFI] RMSE: {metrics_lofi.rmse:.4f}, R²: {metrics_lofi.r2:.4f}, ρ: {metrics_lofi.spearman_corr:.4f}")
            print(f"  Time: {elapsed:.2f}s\n")
        except Exception as e:
            print(f"  Error: {e}\n")
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY (Mean ± Std)")
    print("="*80)

    for model_name, model_results in results.items():
        if not model_results['hifi']:
            continue

        print(f"\n{model_name.upper()}:")

        # HIFI metrics
        hifi_rmse = [m.rmse for m in model_results['hifi']]
        hifi_r2 = [m.r2 for m in model_results['hifi']]
        hifi_spearman = [m.spearman_corr for m in model_results['hifi']]

        print(f"  [HIFI Target] (Finetune quality)")
        print(f"    RMSE:     {np.mean(hifi_rmse):.4f} ± {np.std(hifi_rmse):.4f}")
        print(f"    R²:       {np.mean(hifi_r2):.4f} ± {np.std(hifi_r2):.4f}")
        print(f"    Spearman: {np.mean(hifi_spearman):.4f} ± {np.std(hifi_spearman):.4f}")

        # LOFI metrics
        lofi_rmse = [m.rmse for m in model_results['lofi']]
        lofi_r2 = [m.r2 for m in model_results['lofi']]
        lofi_spearman = [m.spearman_corr for m in model_results['lofi']]

        print(f"  [LOFI Target] (Pretrain quality)")
        print(f"    RMSE:     {np.mean(lofi_rmse):.4f} ± {np.std(lofi_rmse):.4f}")
        print(f"    R²:       {np.mean(lofi_r2):.4f} ± {np.std(lofi_r2):.4f}")
        print(f"    Spearman: {np.mean(lofi_spearman):.4f} ± {np.std(lofi_spearman):.4f}")

        # Time
        times = model_results['time']
        print(f"  Time: {np.mean(times):.2f} ± {np.std(times):.2f}s")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-runs', type=int, default=5, help='Number of runs')
    parser.add_argument('--n-lofi', type=int, default=72, help='Number of LOFI samples')
    parser.add_argument('--n-hifi', type=int, default=9, help='Number of HIFI samples')
    args = parser.parse_args()

    run_comparison(n_runs=args.n_runs, n_lofi=args.n_lofi, n_hifi=args.n_hifi)
