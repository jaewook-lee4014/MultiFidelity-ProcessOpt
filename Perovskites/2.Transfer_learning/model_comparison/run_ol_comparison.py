#!/usr/bin/env python
"""
Online Learning Model Comparison: DNGO-OL vs BNN-OL (with HP-BO)

Compares:
1. DNGO-OL + BO (Online Learning with hyperparameter optimization)
2. BNN-OL + BO (Online Learning with hyperparameter optimization)

Results saved to CSV for comparison with existing 4-model results.
"""

import sys
import os
import numpy as np
import torch
import pickle
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Add parent path
sys.path.insert(0, str(Path(__file__).parent.parent / 'Pure_TL_BO'))

from DNGO.models import BayesianLinearRegression
from DNGO.optimization import OnlineTransferLearningDNN
from BNN.bnn_models import OnlineTransferLearningBNN


@dataclass
class EvaluationMetrics:
    rmse: float
    mae: float
    r2: float
    spearman_corr: float


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

    return EvaluationMetrics(rmse=rmse, mae=mae, r2=r2, spearman_corr=spearman_corr)


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


def run_dngo_ol(dataset: Dict, all_X: np.ndarray, device: str,
                bo_trials: int = 20) -> Dict:
    """Run DNGO-OL with HP-BO"""
    input_dim = dataset['X_low'].shape[1]

    model = OnlineTransferLearningDNN(
        input_dim=input_dim,
        hidden_dim=64,
        device=device,
        use_hyperparameter_bo=True,
        replay_buffer_size=100,
        online_batch_size=16,
        online_epochs=5
    )

    start_time = time.time()

    # Pretrain on LOFI data with HP-BO
    model.pretrain(
        dataset['X_low'], dataset['y_low'],
        epochs=200, lr=1e-3, verbose=False,
        bo_trials=bo_trials,
        data_size='small',
        use_loocv=True
    )

    # Finetune on HIFI data with HP-BO
    model.finetune(
        dataset['X_high'], dataset['y_high'],
        epochs=100, lr=1e-4, verbose=False,
        bo_trials=bo_trials,
        data_size='small',
        use_loocv=True
    )

    # Fit BLR for HIFI
    features_H = model.extract_features(dataset['X_high'])
    model.blr_H = BayesianLinearRegression(alpha=1.0, beta=25.0)
    model.blr_H.fit(features_H, dataset['y_high'])

    elapsed = time.time() - start_time

    # Predict
    test_features = model.extract_features(all_X)
    y_pred, y_var = model.blr_H.predict_batch(test_features)
    y_std = np.sqrt(y_var)

    best_params = {
        'pretrain': getattr(model, 'pretrain_best_params', None),
        'finetune': getattr(model, 'finetune_best_params', None)
    }

    return {
        'y_pred': y_pred,
        'y_std': y_std,
        'elapsed': elapsed,
        'best_params': best_params
    }


def run_bnn_ol(dataset: Dict, all_X: np.ndarray, device: str,
               bo_trials: int = 20) -> Dict:
    """Run BNN-OL with HP-BO"""
    input_dim = dataset['X_low'].shape[1]

    model = OnlineTransferLearningBNN(
        input_dim=input_dim,
        hidden_dims=[64, 64, 64],
        device=device,
        prior_pi=0.5,
        prior_sigma1=1.0,
        prior_sigma2=0.002,
        kl_weight=0.5,
        transfer_mode='consistent_bnn',
        use_hyperparameter_bo=True,
        replay_buffer_size=100,
        online_batch_size=16,
        online_epochs=5
    )

    start_time = time.time()

    # Pretrain on LOFI data with HP-BO
    model.pretrain(
        dataset['X_low'], dataset['y_low'],
        epochs=200, lr=1e-3, verbose=False,
        bo_trials=bo_trials,
        data_size='small',
        use_loocv=False
    )

    # Finetune on HIFI data with HP-BO
    model.finetune(
        dataset['X_high'], dataset['y_high'],
        epochs=100, lr=1e-4, verbose=False,
        bo_trials=bo_trials,
        data_size='small',
        use_loocv=True
    )

    elapsed = time.time() - start_time

    # Predict
    y_pred, y_std = model.predict(all_X, n_samples=50)

    best_params = {
        'pretrain': getattr(model, 'pretrain_best_params', None),
        'finetune': getattr(model, 'finetune_best_params', None)
    }

    return {
        'y_pred': y_pred,
        'y_std': y_std,
        'elapsed': elapsed,
        'best_params': best_params
    }


def run_comparison(n_runs: int = 10, n_lofi: int = 72, n_hifi: int = 9,
                   bo_trials: int = 20, results_dir: str = 'results'):
    """Run OL model comparison"""

    # Device setup
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print("="*80)
    print("OL MODEL COMPARISON: DNGO-OL vs BNN-OL (with HP-BO)")
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
    models = ['dngo_ol_bo', 'bnn_ol_bo']
    model_names = {
        'dngo_ol_bo': 'DNGO-OL + BO',
        'bnn_ol_bo': 'BNN-OL + BO'
    }
    results = {m: {'metrics': [], 'time': [], 'best_params': []} for m in models}
    all_run_results = []

    for run in range(n_runs):
        print(f"\n{'='*60}")
        print(f"RUN {run + 1}/{n_runs}")
        print(f"{'='*60}")

        dataset = generate_dataset(lookup, param_space,
                                   n_lofi=n_lofi, n_hifi=n_hifi,
                                   random_state=run * 1000)

        print(f"Dataset: {dataset['n_lofi']} LOFI, {dataset['n_hifi']} HIFI\n")

        run_results = {'run': run + 1}

        # 1. DNGO-OL + BO
        print("1. DNGO-OL + BO...")
        try:
            result = run_dngo_ol(dataset, all_X, device, bo_trials=bo_trials)
            metrics = calculate_metrics(all_y_hifi, result['y_pred'])
            results['dngo_ol_bo']['metrics'].append(metrics)
            results['dngo_ol_bo']['time'].append(result['elapsed'])
            results['dngo_ol_bo']['best_params'].append(result['best_params'])
            print(f"   RMSE: {metrics.rmse:.4f}, R²: {metrics.r2:.4f}, ρ: {metrics.spearman_corr:.4f}")
            print(f"   Time: {result['elapsed']:.2f}s")
            if result['best_params']:
                print(f"   Best params: {result['best_params']}")
            run_results['dngo_ol_bo_rmse'] = metrics.rmse
            run_results['dngo_ol_bo_r2'] = metrics.r2
            run_results['dngo_ol_bo_spearman'] = metrics.spearman_corr
            run_results['dngo_ol_bo_time'] = result['elapsed']
        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            run_results['dngo_ol_bo_rmse'] = np.nan

        # 2. BNN-OL + BO
        print("\n2. BNN-OL + BO...")
        try:
            result = run_bnn_ol(dataset, all_X, device, bo_trials=bo_trials)
            metrics = calculate_metrics(all_y_hifi, result['y_pred'])
            results['bnn_ol_bo']['metrics'].append(metrics)
            results['bnn_ol_bo']['time'].append(result['elapsed'])
            results['bnn_ol_bo']['best_params'].append(result['best_params'])
            print(f"   RMSE: {metrics.rmse:.4f}, R²: {metrics.r2:.4f}, ρ: {metrics.spearman_corr:.4f}")
            print(f"   Time: {result['elapsed']:.2f}s")
            if result['best_params']:
                print(f"   Best params: {result['best_params']}")
            run_results['bnn_ol_bo_rmse'] = metrics.rmse
            run_results['bnn_ol_bo_r2'] = metrics.r2
            run_results['bnn_ol_bo_spearman'] = metrics.spearman_corr
            run_results['bnn_ol_bo_time'] = result['elapsed']
        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            run_results['bnn_ol_bo_rmse'] = np.nan

        all_run_results.append(run_results)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY (Mean ± Std)")
    print("="*80)

    summary_data = []

    for model_key in models:
        model_results = results[model_key]
        if not model_results['metrics']:
            continue

        rmse = [m.rmse for m in model_results['metrics']]
        r2 = [m.r2 for m in model_results['metrics']]
        spearman = [m.spearman_corr for m in model_results['metrics']]
        times = model_results['time']

        print(f"\n{model_names[model_key]}:")
        print(f"  RMSE:     {np.mean(rmse):.4f} ± {np.std(rmse):.4f}")
        print(f"  R²:       {np.mean(r2):.4f} ± {np.std(r2):.4f}")
        print(f"  Spearman: {np.mean(spearman):.4f} ± {np.std(spearman):.4f}")
        print(f"  Time:     {np.mean(times):.2f} ± {np.std(times):.2f}s")

        summary_data.append({
            'model': model_names[model_key],
            'model_key': model_key,
            'rmse_mean': np.mean(rmse),
            'rmse_std': np.std(rmse),
            'r2_mean': np.mean(r2),
            'r2_std': np.std(r2),
            'spearman_mean': np.mean(spearman),
            'spearman_std': np.std(spearman),
            'time_mean': np.mean(times),
            'time_std': np.std(times)
        })

    # Comparison table
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

    # Save results
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Per-run results
    run_df = pd.DataFrame(all_run_results)
    run_csv = results_path / f'ol_comparison_runs_{timestamp}.csv'
    run_df.to_csv(run_csv, index=False)
    print(f"\nPer-run results saved to: {run_csv}")

    # Summary
    summary_df = pd.DataFrame(summary_data)
    summary_csv = results_path / f'ol_comparison_summary_{timestamp}.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary saved to: {summary_csv}")

    return results, summary_data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-runs', type=int, default=10, help='Number of runs')
    parser.add_argument('--n-lofi', type=int, default=72, help='Number of LOFI samples')
    parser.add_argument('--n-hifi', type=int, default=9, help='Number of HIFI samples')
    parser.add_argument('--bo-trials', type=int, default=20, help='BO trials')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    args = parser.parse_args()

    run_comparison(
        n_runs=args.n_runs,
        n_lofi=args.n_lofi,
        n_hifi=args.n_hifi,
        bo_trials=args.bo_trials,
        results_dir=args.results_dir
    )
