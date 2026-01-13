"""
Visualize DNGO+BO vs MFGP for all 10 folds/seeds
Training data shown in RED, test data in other colors
"""
import sys
import os
import numpy as np
import torch
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'Pure_TL_BO'))
from DNGO.models import TransferLearningDNN, BayesianLinearRegression
from mfgp_model import MultiFidelityGP

# Seeds for 10 folds
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_base_data():
    """Load lookup table and parameter space"""
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

    # Generate all combinations
    all_combinations = []
    for i, org in enumerate(param_space['organic'], 1):
        for j, cat in enumerate(param_space['cation'], 1):
            for k, ani in enumerate(param_space['anion'], 1):
                all_combinations.append({'labels': [i,j,k], 'names': [org,cat,ani]})

    return lookup, all_combinations


def generate_data(lookup, all_combinations, n_lofi=72, n_hifi=9, seed=42):
    """Generate train/test data with given seed"""
    set_seeds(seed)
    rng = np.random.default_rng(seed)

    n_total = len(all_combinations)
    lofi_idx = rng.choice(n_total, size=n_lofi, replace=False)
    hifi_idx = rng.choice(n_total, size=n_hifi, replace=False)

    X_low, y_low = [], []
    for idx in lofi_idx:
        c = all_combinations[idx]
        X_low.append(c['labels'])
        y_low.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_gga']))

    X_high, y_high = [], []
    hifi_combinations = []
    for idx in hifi_idx:
        c = all_combinations[idx]
        X_high.append(c['labels'])
        y_high.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_hse06']))
        hifi_combinations.append(c)

    # All data for test (full HF dataset)
    X_all, y_all = [], []
    for c in all_combinations:
        X_all.append(c['labels'])
        y_all.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_hse06']))

    return {
        'X_low': np.array(X_low, dtype=np.float32),
        'y_low': np.array(y_low, dtype=np.float32),
        'X_high': np.array(X_high, dtype=np.float32),
        'y_high': np.array(y_high, dtype=np.float32),
        'X_all': np.array(X_all, dtype=np.float32),
        'y_all': np.array(y_all, dtype=np.float32),
        'hifi_idx': hifi_idx
    }


def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    sp, _ = spearmanr(y_true, y_pred)
    return rmse, r2, sp if not np.isnan(sp) else 0.0


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load base data once
    lookup, all_combinations = load_base_data()
    print(f"Total combinations: {len(all_combinations)}")

    # Create visualizations directory
    vis_dir = Path(__file__).parent / 'visualizations'
    vis_dir.mkdir(exist_ok=True)

    all_results = []

    for fold_idx, seed in enumerate(SEEDS, 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx} (seed={seed})")
        print('='*60)

        # Create fold directory
        fold_dir = vis_dir / f'fold{fold_idx}'
        fold_dir.mkdir(exist_ok=True)

        # Generate data with this seed
        data = generate_data(lookup, all_combinations, n_lofi=72, n_hifi=9, seed=seed)

        # Train/test indices
        train_idx = data['hifi_idx']  # indices in all_combinations that are used for training
        n_all = len(data['X_all'])
        test_mask = np.ones(n_all, dtype=bool)
        test_mask[train_idx] = False
        test_idx = np.where(test_mask)[0]

        print(f"Train (HF): {len(train_idx)}, Test: {len(test_idx)}")

        # ============== Train MFGP ==============
        set_seeds(seed)
        mfgp = MultiFidelityGP(input_dim=data['X_low'].shape[1], device=device)
        mfgp.fit(data['X_low'], data['y_low'], data['X_high'], data['y_high'])

        # Predict on ALL data
        mfgp_pred_all, mfgp_std_all = mfgp.predict(data['X_all'], return_std=True)

        # ============== Train DNGO ==============
        # Use fixed hyperparameters from previous BO results
        # pretrain: hidden_layers=4, hidden_dim=64, lr~0.006, epochs~200
        # finetune: lr~0.001, epochs~190
        set_seeds(seed)
        dngo = TransferLearningDNN(
            input_dim=data['X_low'].shape[1],
            hidden_dim=64,
            device=device
        )
        dngo.pretrain(data['X_low'], data['y_low'], epochs=200, lr=0.006, verbose=False)
        dngo.finetune(data['X_high'], data['y_high'], epochs=190, lr=0.001, verbose=False)

        # Extract features and fit BLR
        features_train = dngo.extract_features(data['X_high'])
        blr = BayesianLinearRegression(alpha=1.0, beta=25.0)
        blr.fit(features_train, data['y_high'])

        # Predict on ALL data
        features_all = dngo.extract_features(data['X_all'])
        dngo_pred_all, dngo_var_all = blr.predict_batch(features_all)
        dngo_std_all = np.sqrt(dngo_var_all)

        # Calculate metrics on TEST data only
        y_test = data['y_all'][test_idx]
        mfgp_pred_test = mfgp_pred_all[test_idx]
        dngo_pred_test = dngo_pred_all[test_idx]

        mfgp_rmse, mfgp_r2, mfgp_rho = calc_metrics(y_test, mfgp_pred_test)
        dngo_rmse, dngo_r2, dngo_rho = calc_metrics(y_test, dngo_pred_test)

        print(f"MFGP: RMSE={mfgp_rmse:.4f}, R²={mfgp_r2:.4f}, ρ={mfgp_rho:.4f}")
        print(f"DNGO: RMSE={dngo_rmse:.4f}, R²={dngo_r2:.4f}, ρ={dngo_rho:.4f}")

        all_results.append({
            'fold': fold_idx,
            'seed': seed,
            'mfgp_rmse': mfgp_rmse, 'mfgp_r2': mfgp_r2, 'mfgp_rho': mfgp_rho,
            'dngo_rmse': dngo_rmse, 'dngo_r2': dngo_r2, 'dngo_rho': dngo_rho
        })

        # ============== Plot 1: Predictions with uncertainty (sorted by true value) ==============
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Sort all data by true value
        sort_idx = np.argsort(data['y_all'])
        y_sorted = data['y_all'][sort_idx]

        # Create mask for train points in sorted order
        train_mask = np.isin(sort_idx, train_idx)

        x_axis = np.arange(len(y_sorted))

        # MFGP plot
        ax = axes[0]
        mfgp_pred_sorted = mfgp_pred_all[sort_idx]
        mfgp_std_sorted = mfgp_std_all[sort_idx]

        # Plot uncertainty band
        ax.fill_between(x_axis,
                        mfgp_pred_sorted - 2*mfgp_std_sorted,
                        mfgp_pred_sorted + 2*mfgp_std_sorted,
                        alpha=0.3, color='blue', label=f'±2σ (mean σ={mfgp_std_all.mean():.3f})')

        # Plot predictions
        ax.plot(x_axis, mfgp_pred_sorted, 'b-', linewidth=1, label='MFGP Predicted')

        # Plot true values - TEST in BLACK, TRAIN in RED
        ax.scatter(x_axis[~train_mask], y_sorted[~train_mask], c='black', s=20,
                   zorder=5, label='Test HF (true)', alpha=0.7)
        ax.scatter(x_axis[train_mask], y_sorted[train_mask], c='red', s=100,
                   marker='*', zorder=6, label='Train HF (true)', edgecolors='darkred')

        ax.set_ylabel('Bandgap (eV)', fontsize=12)
        ax.set_title(f'MFGP: RMSE={mfgp_rmse:.3f}, R²={mfgp_r2:.3f}, ρ={mfgp_rho:.3f}', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # DNGO plot
        ax = axes[1]
        dngo_pred_sorted = dngo_pred_all[sort_idx]
        dngo_std_sorted = dngo_std_all[sort_idx]

        # Plot uncertainty band
        ax.fill_between(x_axis,
                        dngo_pred_sorted - 2*dngo_std_sorted,
                        dngo_pred_sorted + 2*dngo_std_sorted,
                        alpha=0.3, color='green', label=f'±2σ (mean σ={dngo_std_all.mean():.3f})')

        # Plot predictions
        ax.plot(x_axis, dngo_pred_sorted, 'g-', linewidth=1, label='DNGO+BO Predicted')

        # Plot true values - TEST in BLACK, TRAIN in RED
        ax.scatter(x_axis[~train_mask], y_sorted[~train_mask], c='black', s=20,
                   zorder=5, label='Test HF (true)', alpha=0.7)
        ax.scatter(x_axis[train_mask], y_sorted[train_mask], c='red', s=100,
                   marker='*', zorder=6, label='Train HF (true)', edgecolors='darkred')

        ax.set_xlabel('Sample Index (sorted by true value)', fontsize=12)
        ax.set_ylabel('Bandgap (eV)', fontsize=12)
        ax.set_title(f'DNGO+BO: RMSE={dngo_rmse:.3f}, R²={dngo_r2:.3f}, ρ={dngo_rho:.3f}', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'Fold {fold_idx} (seed={seed})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(fold_dir / 'predictions_with_uncertainty.png', dpi=150, bbox_inches='tight')
        plt.close()

        # ============== Plot 2: Parity plots ==============
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        y_train = data['y_high']

        # MFGP parity
        ax = axes[0]
        mfgp_pred_train = mfgp_pred_all[train_idx]
        mfgp_std_train = mfgp_std_all[train_idx]
        mfgp_std_test = mfgp_std_all[test_idx]

        # Test points in blue
        ax.errorbar(y_test, mfgp_pred_test, yerr=2*mfgp_std_test,
                    fmt='o', color='blue', alpha=0.5, capsize=2, markersize=4, label='Test')
        # Train points in red
        ax.errorbar(y_train, mfgp_pred_train, yerr=2*mfgp_std_train,
                    fmt='*', color='red', markersize=15, alpha=0.9, capsize=3, label='Train')

        # y=x line
        all_vals = np.concatenate([data['y_all'], mfgp_pred_all])
        lims = [all_vals.min() - 0.5, all_vals.max() + 0.5]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('True Bandgap (eV)', fontsize=12)
        ax.set_ylabel('Predicted Bandgap (eV)', fontsize=12)
        ax.set_title(f'MFGP\nRMSE={mfgp_rmse:.3f}, R²={mfgp_r2:.3f}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # DNGO parity
        ax = axes[1]
        dngo_pred_train = dngo_pred_all[train_idx]
        dngo_std_train = dngo_std_all[train_idx]
        dngo_std_test = dngo_std_all[test_idx]

        # Test points in green
        ax.errorbar(y_test, dngo_pred_test, yerr=2*dngo_std_test,
                    fmt='o', color='green', alpha=0.5, capsize=2, markersize=4, label='Test')
        # Train points in red
        ax.errorbar(y_train, dngo_pred_train, yerr=2*dngo_std_train,
                    fmt='*', color='red', markersize=15, alpha=0.9, capsize=3, label='Train')

        all_vals = np.concatenate([data['y_all'], dngo_pred_all])
        lims = [all_vals.min() - 0.5, all_vals.max() + 0.5]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('True Bandgap (eV)', fontsize=12)
        ax.set_ylabel('Predicted Bandgap (eV)', fontsize=12)
        ax.set_title(f'DNGO+BO\nRMSE={dngo_rmse:.3f}, R²={dngo_r2:.3f}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.suptitle(f'Fold {fold_idx} (seed={seed}) - Parity Plots', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(fold_dir / 'parity_plots.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved to: {fold_dir}/")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL FOLDS")
    print('='*60)
    print(f"{'Fold':<6} {'MFGP RMSE':<12} {'MFGP R²':<10} {'DNGO RMSE':<12} {'DNGO R²':<10}")
    print('-'*60)
    for r in all_results:
        print(f"{r['fold']:<6} {r['mfgp_rmse']:<12.4f} {r['mfgp_r2']:<10.4f} {r['dngo_rmse']:<12.4f} {r['dngo_r2']:<10.4f}")
    print('-'*60)

    # Averages
    avg_mfgp_rmse = np.mean([r['mfgp_rmse'] for r in all_results])
    avg_mfgp_r2 = np.mean([r['mfgp_r2'] for r in all_results])
    avg_dngo_rmse = np.mean([r['dngo_rmse'] for r in all_results])
    avg_dngo_r2 = np.mean([r['dngo_r2'] for r in all_results])
    print(f"{'Avg':<6} {avg_mfgp_rmse:<12.4f} {avg_mfgp_r2:<10.4f} {avg_dngo_rmse:<12.4f} {avg_dngo_r2:<10.4f}")

    print(f"\nAll {len(SEEDS)} folds visualized!")
    print(f"Output directory: {vis_dir}")


if __name__ == '__main__':
    main()
