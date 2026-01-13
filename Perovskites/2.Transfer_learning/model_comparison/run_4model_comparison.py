#!/usr/bin/env python
"""
4-Model Comparison with Hyperparameter Bayesian Optimization

Compares:
1. DNGO (baseline)
2. DNGO + BO (with hyperparameter optimization)
3. BNN (consistent_bnn mode, baseline)
4. BNN + BO (consistent_bnn mode, with hyperparameter optimization)

All models use:
- Transfer learning: Pretrain on LOFI → Finetune on HIFI
- LOOCV for validation (when applicable)
- Same dataset splits for fair comparison
"""

import sys
import os
import numpy as np
import torch
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import time
import warnings
warnings.filterwarnings('ignore')

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

    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mae = np.mean(np.abs(y_pred - y_true))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    spearman_corr, _ = spearmanr(y_true, y_pred)
    if np.isnan(spearman_corr):
        spearman_corr = 0.0

    best_pred_idx = np.argmin(y_pred)
    best_prediction = y_pred[best_pred_idx]
    best_actual = y_true[best_pred_idx]

    return EvaluationMetrics(
        rmse=rmse, mae=mae, r2=r2, spearman_corr=spearman_corr,
        best_prediction=best_prediction, best_actual=best_actual
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

    return lookup, param_space


def generate_dataset(lookup, param_space, n_lofi: int, n_hifi: int,
                     random_state: int = None) -> Dict:
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
        'y_high': np.array(y_high, dtype=np.float32),
        'n_lofi': len(X_low),
        'n_hifi': len(X_high)
    }


def generate_test_data(lookup, param_space) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate test data for all combinations"""
    organics = param_space['organic']
    cations = param_space['cation']
    anions = param_space['anion']

    all_X, all_y_hifi, all_y_lofi = [], [], []

    for i, org in enumerate(organics, 1):
        for j, cat in enumerate(cations, 1):
            for k, ani in enumerate(anions, 1):
                all_X.append([i, j, k])
                all_y_hifi.append(np.amin(lookup[org.capitalize()][cat][ani]['bandgap_hse06']))
                all_y_lofi.append(np.amin(lookup[org.capitalize()][cat][ani]['bandgap_gga']))

    return (np.array(all_X, dtype=np.float32),
            np.array(all_y_hifi, dtype=np.float32),
            np.array(all_y_lofi, dtype=np.float32))


# ============================================================================
# Model Runners
# ============================================================================

def run_dngo(dataset: Dict, all_X: np.ndarray, device: str,
             use_bo: bool = False, bo_trials: int = 20) -> Dict:
    """Run DNGO model with optional hyperparameter BO"""
    input_dim = dataset['X_low'].shape[1]

    model = TransferLearningDNN(
        input_dim=input_dim,
        hidden_dim=64,
        device=device,
        use_hyperparameter_bo=use_bo
    )

    start_time = time.time()

    # Pretrain
    model.pretrain(
        dataset['X_low'], dataset['y_low'],
        epochs=200, lr=1e-3, verbose=False,
        bo_trials=bo_trials if use_bo else None,
        data_size='small',
        use_loocv=use_bo  # LOOCV only when using BO
    )

    # Finetune
    model.finetune(
        dataset['X_high'], dataset['y_high'],
        epochs=100, lr=1e-4, verbose=False,
        bo_trials=bo_trials if use_bo else None,
        data_size='small',
        use_loocv=use_bo
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

    # Get best params if BO was used
    best_params = None
    if use_bo:
        best_params = {
            'pretrain': model.pretrain_best_params,
            'finetune': model.finetune_best_params
        }

    return {
        'y_pred': y_pred,
        'y_std': y_std,
        'elapsed': elapsed,
        'best_params': best_params
    }


def run_bnn(dataset: Dict, all_X: np.ndarray, device: str,
            use_bo: bool = False, bo_trials: int = 20) -> Dict:
    """Run BNN (consistent_bnn) model with optional hyperparameter BO"""
    input_dim = dataset['X_low'].shape[1]

    model = TransferLearningBNN(
        input_dim=input_dim,
        hidden_dims=[64, 64],
        device=device,
        prior_pi=0.5,
        prior_sigma1=1.0,
        prior_sigma2=0.002,
        kl_weight=1.0,
        transfer_mode='consistent_bnn',
        use_hyperparameter_bo=use_bo
    )

    start_time = time.time()

    # Pretrain
    model.pretrain(
        dataset['X_low'], dataset['y_low'],
        epochs=200, lr=1e-3, verbose=False,
        bo_trials=bo_trials if use_bo else None,
        data_size='small',
        use_loocv=False  # No LOOCV for pretrain (too slow)
    )

    # Finetune
    model.finetune(
        dataset['X_high'], dataset['y_high'],
        epochs=100, lr=1e-4, verbose=False,
        bo_trials=bo_trials if use_bo else None,
        data_size='small',
        use_loocv=use_bo  # LOOCV for finetune
    )

    elapsed = time.time() - start_time

    # Predict
    y_pred, y_std = model.predict(all_X, n_samples=50)

    # Get best params if BO was used
    best_params = None
    if use_bo:
        best_params = {
            'pretrain': model.pretrain_best_params,
            'finetune': model.finetune_best_params
        }

    return {
        'y_pred': y_pred,
        'y_std': y_std,
        'elapsed': elapsed,
        'best_params': best_params
    }


# ============================================================================
# Main Comparison
# ============================================================================

def run_comparison(n_runs: int = 5, n_lofi: int = 72, n_hifi: int = 9,
                   bo_trials: int = 20, verbose: bool = True):
    """Run 4-model comparison"""

    # Device setup
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print("="*80)
    print("4-MODEL COMPARISON: DNGO vs BNN with Hyperparameter Optimization")
    print("="*80)
    print(f"Device: {device}")
    print(f"Runs: {n_runs}")
    print(f"Data: {n_lofi} LOFI, {n_hifi} HIFI samples")
    print(f"BO trials: {bo_trials}")
    print("="*80 + "\n")

    # Load data
    lookup, param_space = load_data()
    all_X, all_y_hifi, all_y_lofi = generate_test_data(lookup, param_space)

    print(f"Test set: {len(all_X)} combinations")
    print(f"HIFI bandgap range: [{all_y_hifi.min():.3f}, {all_y_hifi.max():.3f}]")
    print(f"LOFI bandgap range: [{all_y_lofi.min():.3f}, {all_y_lofi.max():.3f}]\n")

    # Results storage
    models = ['dngo', 'dngo_bo', 'bnn', 'bnn_bo']
    results = {m: {'hifi': [], 'lofi': [], 'time': [], 'best_params': []} for m in models}

    for run in range(n_runs):
        print(f"\n{'='*60}")
        print(f"RUN {run + 1}/{n_runs}")
        print(f"{'='*60}")

        # Generate dataset (same for all models in this run)
        dataset = generate_dataset(lookup, param_space,
                                   n_lofi=n_lofi, n_hifi=n_hifi,
                                   random_state=run * 1000)

        print(f"Dataset: {dataset['n_lofi']} LOFI, {dataset['n_hifi']} HIFI\n")

        # 1. DNGO (baseline)
        print("1. DNGO (baseline)...")
        try:
            result = run_dngo(dataset, all_X, device, use_bo=False)
            metrics_hifi = calculate_metrics(all_y_hifi, result['y_pred'])
            metrics_lofi = calculate_metrics(all_y_lofi, result['y_pred'])
            results['dngo']['hifi'].append(metrics_hifi)
            results['dngo']['lofi'].append(metrics_lofi)
            results['dngo']['time'].append(result['elapsed'])
            print(f"   [HIFI] RMSE: {metrics_hifi.rmse:.4f}, R²: {metrics_hifi.r2:.4f}, ρ: {metrics_hifi.spearman_corr:.4f}")
            print(f"   Time: {result['elapsed']:.2f}s")
        except Exception as e:
            print(f"   Error: {e}")

        # 2. DNGO + BO
        print("\n2. DNGO + BO (hyperparameter optimization)...")
        try:
            result = run_dngo(dataset, all_X, device, use_bo=True, bo_trials=bo_trials)
            metrics_hifi = calculate_metrics(all_y_hifi, result['y_pred'])
            metrics_lofi = calculate_metrics(all_y_lofi, result['y_pred'])
            results['dngo_bo']['hifi'].append(metrics_hifi)
            results['dngo_bo']['lofi'].append(metrics_lofi)
            results['dngo_bo']['time'].append(result['elapsed'])
            results['dngo_bo']['best_params'].append(result['best_params'])
            print(f"   [HIFI] RMSE: {metrics_hifi.rmse:.4f}, R²: {metrics_hifi.r2:.4f}, ρ: {metrics_hifi.spearman_corr:.4f}")
            print(f"   Time: {result['elapsed']:.2f}s")
            if result['best_params']:
                print(f"   Best params: {result['best_params']}")
        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()

        # 3. BNN (baseline)
        print("\n3. BNN consistent_bnn (baseline)...")
        try:
            result = run_bnn(dataset, all_X, device, use_bo=False)
            metrics_hifi = calculate_metrics(all_y_hifi, result['y_pred'])
            metrics_lofi = calculate_metrics(all_y_lofi, result['y_pred'])
            results['bnn']['hifi'].append(metrics_hifi)
            results['bnn']['lofi'].append(metrics_lofi)
            results['bnn']['time'].append(result['elapsed'])
            print(f"   [HIFI] RMSE: {metrics_hifi.rmse:.4f}, R²: {metrics_hifi.r2:.4f}, ρ: {metrics_hifi.spearman_corr:.4f}")
            print(f"   Time: {result['elapsed']:.2f}s")
        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()

        # 4. BNN + BO
        print("\n4. BNN consistent_bnn + BO (hyperparameter optimization)...")
        try:
            result = run_bnn(dataset, all_X, device, use_bo=True, bo_trials=bo_trials)
            metrics_hifi = calculate_metrics(all_y_hifi, result['y_pred'])
            metrics_lofi = calculate_metrics(all_y_lofi, result['y_pred'])
            results['bnn_bo']['hifi'].append(metrics_hifi)
            results['bnn_bo']['lofi'].append(metrics_lofi)
            results['bnn_bo']['time'].append(result['elapsed'])
            results['bnn_bo']['best_params'].append(result['best_params'])
            print(f"   [HIFI] RMSE: {metrics_hifi.rmse:.4f}, R²: {metrics_hifi.r2:.4f}, ρ: {metrics_hifi.spearman_corr:.4f}")
            print(f"   Time: {result['elapsed']:.2f}s")
            if result['best_params']:
                print(f"   Best params: {result['best_params']}")
        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # Print Summary
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY (Mean ± Std)")
    print("="*80)

    model_names = {
        'dngo': 'DNGO (baseline)',
        'dngo_bo': 'DNGO + BO',
        'bnn': 'BNN (baseline)',
        'bnn_bo': 'BNN + BO'
    }

    summary_data = []

    for model_key in models:
        model_results = results[model_key]
        if not model_results['hifi']:
            continue

        hifi_rmse = [m.rmse for m in model_results['hifi']]
        hifi_r2 = [m.r2 for m in model_results['hifi']]
        hifi_spearman = [m.spearman_corr for m in model_results['hifi']]
        times = model_results['time']

        print(f"\n{model_names[model_key]}:")
        print(f"  [HIFI Target]")
        print(f"    RMSE:     {np.mean(hifi_rmse):.4f} ± {np.std(hifi_rmse):.4f}")
        print(f"    R²:       {np.mean(hifi_r2):.4f} ± {np.std(hifi_r2):.4f}")
        print(f"    Spearman: {np.mean(hifi_spearman):.4f} ± {np.std(hifi_spearman):.4f}")
        print(f"  Time: {np.mean(times):.2f} ± {np.std(times):.2f}s")

        summary_data.append({
            'model': model_names[model_key],
            'rmse_mean': np.mean(hifi_rmse),
            'rmse_std': np.std(hifi_rmse),
            'r2_mean': np.mean(hifi_r2),
            'r2_std': np.std(hifi_r2),
            'spearman_mean': np.mean(hifi_spearman),
            'spearman_std': np.std(hifi_spearman),
            'time_mean': np.mean(times),
            'time_std': np.std(times)
        })

    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Model':<20} {'RMSE':<18} {'R²':<18} {'Spearman':<18} {'Time (s)':<12}")
    print("-"*80)

    for row in summary_data:
        print(f"{row['model']:<20} "
              f"{row['rmse_mean']:.4f} ± {row['rmse_std']:.4f}  "
              f"{row['r2_mean']:.4f} ± {row['r2_std']:.4f}  "
              f"{row['spearman_mean']:.4f} ± {row['spearman_std']:.4f}  "
              f"{row['time_mean']:.1f} ± {row['time_std']:.1f}")

    # Find best model
    if summary_data:
        best_rmse = min(summary_data, key=lambda x: x['rmse_mean'])
        best_r2 = max(summary_data, key=lambda x: x['r2_mean'])
        print("\n" + "-"*80)
        print(f"Best RMSE: {best_rmse['model']} ({best_rmse['rmse_mean']:.4f})")
        print(f"Best R²:   {best_r2['model']} ({best_r2['r2_mean']:.4f})")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-runs', type=int, default=5, help='Number of runs')
    parser.add_argument('--n-lofi', type=int, default=72, help='Number of LOFI samples')
    parser.add_argument('--n-hifi', type=int, default=9, help='Number of HIFI samples')
    parser.add_argument('--bo-trials', type=int, default=20, help='BO trials for hyperparameter optimization')
    args = parser.parse_args()

    run_comparison(
        n_runs=args.n_runs,
        n_lofi=args.n_lofi,
        n_hifi=args.n_hifi,
        bo_trials=args.bo_trials
    )
