#!/usr/bin/env python
"""
New Model Comparison: MFGP, Standard GP, DNGO+Progressive

Only runs models that don't have results yet:
1. MFGP (Multi-Fidelity GP using BoTorch)
2. Standard GP (HF only, baseline)
3. DNGO + BO + Progressive Unfreezing

Existing results (from 4-model comparison):
- DNGO + BO: RMSE 0.6489 ± 0.09
- BNN + BO: RMSE 0.6834 ± 0.16

Uses fixed seeds for reproducibility and visualization.
"""
import sys
import os
import numpy as np
import torch
import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
import time
import warnings
warnings.filterwarnings('ignore')

# Add parent path
sys.path.insert(0, str(Path(__file__).parent.parent / 'Pure_TL_BO'))

from DNGO.models import TransferLearningDNN
from mfgp_model import MultiFidelityGP, StandardGP

# =============================================================================
# FIXED SEEDS FOR REPRODUCIBILITY
# =============================================================================
FIXED_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_data():
    """Load perovskite data."""
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


def generate_dataset_with_seed(lookup, param_space, n_lofi: int, n_hifi: int, seed: int):
    """Generate dataset with fixed seed for reproducibility."""
    set_all_seeds(seed)
    rng = np.random.default_rng(seed)

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
        'lofi_indices': lofi_indices,
        'hifi_indices': hifi_indices
    }


def generate_test_data(lookup, param_space):
    """Generate test data for all combinations."""
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


def calculate_metrics(y_true, y_pred):
    """Calculate RMSE, R2, and Spearman correlation."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    sp, _ = spearmanr(y_true, y_pred)

    return {
        'rmse': rmse,
        'r2': r2,
        'spearman': sp if not np.isnan(sp) else 0.0
    }


def run_mfgp(dataset, X_test, device):
    """Run Multi-Fidelity GP."""
    input_dim = dataset['X_low'].shape[1]
    model = MultiFidelityGP(input_dim=input_dim, device=device)

    start = time.time()
    model.fit(dataset['X_low'], dataset['y_low'],
              dataset['X_high'], dataset['y_high'], verbose=False)
    y_pred, y_std = model.predict(X_test)
    elapsed = time.time() - start

    return y_pred, y_std, elapsed


def run_standard_gp(dataset, X_test, device):
    """Run Standard GP (HF only)."""
    input_dim = dataset['X_high'].shape[1]
    model = StandardGP(input_dim=input_dim, device=device)

    start = time.time()
    model.fit(dataset['X_high'], dataset['y_high'], verbose=False)
    y_pred, y_std = model.predict(X_test)
    elapsed = time.time() - start

    return y_pred, y_std, elapsed


def run_dngo_progressive(dataset, X_test, device, bo_trials=20):
    """Run DNGO with Progressive Unfreezing."""
    input_dim = dataset['X_low'].shape[1]
    model = TransferLearningDNN(
        input_dim=input_dim,
        use_hyperparameter_bo=True,
        device=device
    )

    start = time.time()
    model.pretrain(dataset['X_low'], dataset['y_low'],
                   bo_trials=bo_trials, verbose=False)
    model.finetune(dataset['X_high'], dataset['y_high'],
                   bo_trials=bo_trials, verbose=False,
                   use_freeze=False, use_progressive_unfreezing=True)
    y_pred = model.predict(X_test)
    elapsed = time.time() - start

    return y_pred, None, elapsed


def plot_all_predictions(y_true, predictions_dict, run_idx, save_dir):
    """Plot predictions vs true values with uncertainty bands."""
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    if n_models == 1:
        axes = [axes]

    sort_idx = np.argsort(y_true)
    y_true_sorted = y_true[sort_idx]
    x_axis = np.arange(len(y_true))

    for idx, (model_name, (y_pred, y_std)) in enumerate(predictions_dict.items()):
        ax = axes[idx]
        y_pred_sorted = y_pred[sort_idx]

        # Plot true values
        ax.scatter(x_axis, y_true_sorted, c='black', s=8, alpha=0.6, label='True HF', zorder=3)

        # Plot predictions with uncertainty
        ax.plot(x_axis, y_pred_sorted, c='blue', lw=1, alpha=0.8, label='Predicted', zorder=2)

        if y_std is not None:
            y_std_sorted = y_std[sort_idx]
            ax.fill_between(x_axis,
                           y_pred_sorted - 2*y_std_sorted,
                           y_pred_sorted + 2*y_std_sorted,
                           alpha=0.25, color='blue', label='±2σ', zorder=1)

        metrics = calculate_metrics(y_true, y_pred)
        ax.set_xlabel('Sample Index (sorted by true value)')
        ax.set_ylabel('Bandgap (eV)')
        ax.set_title(f'{model_name}\nRMSE={metrics["rmse"]:.3f}, R²={metrics["r2"]:.3f}, ρ={metrics["spearman"]:.3f}')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f'predictions_run{run_idx+1}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_parity_plots(y_true, predictions_dict, run_idx, save_dir):
    """Plot parity plots for all models."""
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    if n_models == 1:
        axes = [axes]

    for idx, (model_name, (y_pred, y_std)) in enumerate(predictions_dict.items()):
        ax = axes[idx]

        min_val = min(y_true.min(), y_pred.min()) - 0.5
        max_val = max(y_true.max(), y_pred.max()) + 0.5
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1, alpha=0.5, label='y=x')

        if y_std is not None:
            ax.errorbar(y_true, y_pred, yerr=2*y_std, fmt='o', markersize=4,
                       alpha=0.5, elinewidth=0.5, capsize=0)
        else:
            ax.scatter(y_true, y_pred, s=15, alpha=0.6)

        metrics = calculate_metrics(y_true, y_pred)
        ax.set_xlabel('True Bandgap (eV)')
        ax.set_ylabel('Predicted Bandgap (eV)')
        ax.set_title(f'{model_name}\nRMSE={metrics["rmse"]:.3f}, R²={metrics["r2"]:.3f}, ρ={metrics["spearman"]:.3f}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_dir / f'parity_run{run_idx+1}.png', dpi=150, bbox_inches='tight')
    plt.close()


def run_comparison(n_runs=10, n_lofi=72, n_hifi=9, bo_trials=20, save_dir='results_new_models'):
    """Run comparison for new models only."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Runs: {n_runs}")
    print(f"Data: {n_lofi} LOFI, {n_hifi} HIFI samples")
    print(f"BO trials: {bo_trials}")
    print(f"Fixed seeds: {FIXED_SEEDS[:n_runs]}")
    print("=" * 70)
    print("Models to run: MFGP, Standard_GP, DNGO+BO+Progressive")
    print("Existing results: DNGO+BO (0.6489), BNN+BO (0.6834)")
    print("=" * 70)

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    lookup, param_space = load_data()
    X_test, y_test_hifi, y_test_lofi = generate_test_data(lookup, param_space)
    print(f"Test set: {len(X_test)} samples")
    print(f"HIFI bandgap range: [{y_test_hifi.min():.3f}, {y_test_hifi.max():.3f}]")

    models = ['MFGP', 'Standard_GP', 'DNGO+Prog']
    results = {model: {'rmse': [], 'r2': [], 'spearman': [], 'time': []} for model in models}

    for run in range(n_runs):
        seed = FIXED_SEEDS[run]
        print(f"\n{'='*70}")
        print(f"RUN {run+1}/{n_runs} (seed={seed})")
        print(f"{'='*70}")

        dataset = generate_dataset_with_seed(lookup, param_space, n_lofi, n_hifi, seed)
        print(f"Dataset: {len(dataset['X_low'])} LOFI, {len(dataset['X_high'])} HIFI")

        predictions = {}

        # 1. MFGP
        print("\n1. MFGP...")
        try:
            y_pred, y_std, elapsed = run_mfgp(dataset, X_test, device)
            metrics = calculate_metrics(y_test_hifi, y_pred)
            results['MFGP']['rmse'].append(metrics['rmse'])
            results['MFGP']['r2'].append(metrics['r2'])
            results['MFGP']['spearman'].append(metrics['spearman'])
            results['MFGP']['time'].append(elapsed)
            predictions['MFGP'] = (y_pred, y_std)
            print(f"   RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}, ρ: {metrics['spearman']:.4f}, Time: {elapsed:.1f}s")
        except Exception as e:
            print(f"   MFGP failed: {e}")
            results['MFGP']['rmse'].append(np.nan)
            results['MFGP']['r2'].append(np.nan)
            results['MFGP']['spearman'].append(np.nan)
            results['MFGP']['time'].append(np.nan)

        # 2. Standard GP (HF only)
        print("\n2. Standard GP (HF only)...")
        try:
            y_pred, y_std, elapsed = run_standard_gp(dataset, X_test, device)
            metrics = calculate_metrics(y_test_hifi, y_pred)
            results['Standard_GP']['rmse'].append(metrics['rmse'])
            results['Standard_GP']['r2'].append(metrics['r2'])
            results['Standard_GP']['spearman'].append(metrics['spearman'])
            results['Standard_GP']['time'].append(elapsed)
            predictions['Standard_GP'] = (y_pred, y_std)
            print(f"   RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}, ρ: {metrics['spearman']:.4f}, Time: {elapsed:.1f}s")
        except Exception as e:
            print(f"   Standard GP failed: {e}")
            results['Standard_GP']['rmse'].append(np.nan)
            results['Standard_GP']['r2'].append(np.nan)
            results['Standard_GP']['spearman'].append(np.nan)
            results['Standard_GP']['time'].append(np.nan)

        # 3. DNGO + Progressive Unfreezing
        print("\n3. DNGO + BO + Progressive...")
        try:
            set_all_seeds(seed)
            y_pred, y_std, elapsed = run_dngo_progressive(dataset, X_test, device, bo_trials=bo_trials)
            metrics = calculate_metrics(y_test_hifi, y_pred)
            results['DNGO+Prog']['rmse'].append(metrics['rmse'])
            results['DNGO+Prog']['r2'].append(metrics['r2'])
            results['DNGO+Prog']['spearman'].append(metrics['spearman'])
            results['DNGO+Prog']['time'].append(elapsed)
            predictions['DNGO+Prog'] = (y_pred, y_std)
            print(f"   RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}, ρ: {metrics['spearman']:.4f}, Time: {elapsed:.1f}s")
        except Exception as e:
            print(f"   DNGO+Prog failed: {e}")
            results['DNGO+Prog']['rmse'].append(np.nan)
            results['DNGO+Prog']['r2'].append(np.nan)
            results['DNGO+Prog']['spearman'].append(np.nan)
            results['DNGO+Prog']['time'].append(np.nan)

        # Generate plots
        if predictions:
            plot_all_predictions(y_test_hifi, predictions, run, save_path)
            plot_parity_plots(y_test_hifi, predictions, run, save_path)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY (New Models)")
    print("=" * 80)
    print(f"{'Model':<20} {'RMSE':>15} {'R²':>15} {'Spearman':>15} {'Time':>10}")
    print("-" * 80)

    summary_data = []
    for model in models:
        rmse_vals = [x for x in results[model]['rmse'] if not np.isnan(x)]
        r2_vals = [x for x in results[model]['r2'] if not np.isnan(x)]
        sp_vals = [x for x in results[model]['spearman'] if not np.isnan(x)]
        time_vals = [x for x in results[model]['time'] if not np.isnan(x)]

        if rmse_vals:
            rmse_str = f"{np.mean(rmse_vals):.4f}±{np.std(rmse_vals):.4f}"
            r2_str = f"{np.mean(r2_vals):.4f}±{np.std(r2_vals):.4f}"
            sp_str = f"{np.mean(sp_vals):.4f}±{np.std(sp_vals):.4f}"
            time_str = f"{np.mean(time_vals):.1f}s"
        else:
            rmse_str = r2_str = sp_str = time_str = "N/A"

        print(f"{model:<20} {rmse_str:>15} {r2_str:>15} {sp_str:>15} {time_str:>10}")

        summary_data.append({
            'Model': model,
            'RMSE_mean': np.mean(rmse_vals) if rmse_vals else np.nan,
            'RMSE_std': np.std(rmse_vals) if rmse_vals else np.nan,
            'R2_mean': np.mean(r2_vals) if r2_vals else np.nan,
            'R2_std': np.std(r2_vals) if r2_vals else np.nan,
            'Spearman_mean': np.mean(sp_vals) if sp_vals else np.nan,
            'Spearman_std': np.std(sp_vals) if sp_vals else np.nan,
            'Time_mean': np.mean(time_vals) if time_vals else np.nan
        })

    # Add existing results for comparison
    print("-" * 80)
    print("EXISTING RESULTS (from 4-model comparison, BO=20):")
    print(f"{'DNGO+BO':<20} {'0.6489±0.0917':>15} {'0.6674±0.0943':>15} {'0.8027±0.0480':>15} {'163.6s':>10}")
    print(f"{'BNN+BO':<20} {'0.6834±0.1575':>15} {'0.6192±0.1877':>15} {'0.8265±0.0324':>15} {'1714.6s':>10}")
    print("=" * 80)

    # Save results
    df = pd.DataFrame(summary_data)
    df.to_csv(save_path / 'summary_new_models.csv', index=False)

    detailed_df = pd.DataFrame({
        'run': list(range(1, n_runs + 1)) * len(models),
        'seed': FIXED_SEEDS[:n_runs] * len(models),
        'model': [m for m in models for _ in range(n_runs)],
        'rmse': [results[m]['rmse'][i] for m in models for i in range(n_runs)],
        'r2': [results[m]['r2'][i] for m in models for i in range(n_runs)],
        'spearman': [results[m]['spearman'][i] for m in models for i in range(n_runs)],
        'time': [results[m]['time'][i] for m in models for i in range(n_runs)]
    })
    detailed_df.to_csv(save_path / 'detailed_results_new_models.csv', index=False)

    print(f"\nResults saved to {save_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-runs', type=int, default=10)
    parser.add_argument('--n-lofi', type=int, default=72)
    parser.add_argument('--n-hifi', type=int, default=9)
    parser.add_argument('--bo-trials', type=int, default=20)
    parser.add_argument('--save-dir', type=str, default='results_new_models')
    args = parser.parse_args()

    results = run_comparison(
        n_runs=args.n_runs,
        n_lofi=args.n_lofi,
        n_hifi=args.n_hifi,
        bo_trials=args.bo_trials,
        save_dir=args.save_dir
    )
