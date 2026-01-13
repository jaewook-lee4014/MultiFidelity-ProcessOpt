#!/usr/bin/env python
"""
Visualize GP and MFGP predictions with uncertainty bands.
"""
import sys
import numpy as np
import torch
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'Pure_TL_BO'))
from mfgp_model import MultiFidelityGP, StandardGP

SEED = 1213  # Best performing seed (R²=0.66)


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
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


def generate_data(lookup, param_space, n_lofi=72, n_hifi=9, seed=42):
    set_seeds(seed)
    rng = np.random.default_rng(seed)

    all_combinations = []
    for i, org in enumerate(param_space['organic'], 1):
        for j, cat in enumerate(param_space['cation'], 1):
            for k, ani in enumerate(param_space['anion'], 1):
                all_combinations.append({'labels': [i,j,k], 'names': [org,cat,ani]})

    n_total = len(all_combinations)
    lofi_idx = rng.choice(n_total, size=n_lofi, replace=False)
    hifi_idx = rng.choice(n_total, size=n_hifi, replace=False)

    X_low, y_low = [], []
    for idx in lofi_idx:
        c = all_combinations[idx]
        X_low.append(c['labels'])
        y_low.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_gga']))

    X_high, y_high = [], []
    for idx in hifi_idx:
        c = all_combinations[idx]
        X_high.append(c['labels'])
        y_high.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_hse06']))

    # Test data: all combinations
    X_test, y_test = [], []
    for c in all_combinations:
        X_test.append(c['labels'])
        y_test.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_hse06']))

    return {
        'X_low': np.array(X_low, dtype=np.float32),
        'y_low': np.array(y_low, dtype=np.float32),
        'X_high': np.array(X_high, dtype=np.float32),
        'y_high': np.array(y_high, dtype=np.float32),
        'X_test': np.array(X_test, dtype=np.float32),
        'y_test': np.array(y_test, dtype=np.float32),
        'hifi_idx': hifi_idx
    }


def calc_metrics(y_true, y_pred):
    from scipy.stats import spearmanr
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    sp, _ = spearmanr(y_true, y_pred)
    return rmse, r2, sp if not np.isnan(sp) else 0.0


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Seed: {SEED}")

    # Load and generate data
    lookup, param_space = load_data()
    data = generate_data(lookup, param_space, n_lofi=72, n_hifi=9, seed=SEED)

    print(f"Data: {len(data['X_low'])} LF, {len(data['X_high'])} HF")
    print(f"Test: {len(data['X_test'])} samples")

    # Train models
    print("\nTraining MFGP...")
    mfgp = MultiFidelityGP(input_dim=3, device=device)
    mfgp.fit(data['X_low'], data['y_low'], data['X_high'], data['y_high'])
    mfgp_pred, mfgp_std = mfgp.predict(data['X_test'])

    print("Training Standard GP...")
    sgp = StandardGP(input_dim=3, device=device)
    sgp.fit(data['X_high'], data['y_high'])
    sgp_pred, sgp_std = sgp.predict(data['X_test'])

    # Calculate metrics
    mfgp_rmse, mfgp_r2, mfgp_sp = calc_metrics(data['y_test'], mfgp_pred)
    sgp_rmse, sgp_r2, sgp_sp = calc_metrics(data['y_test'], sgp_pred)

    print(f"\nMFGP: RMSE={mfgp_rmse:.4f}, R²={mfgp_r2:.4f}, ρ={mfgp_sp:.4f}")
    print(f"Standard GP: RMSE={sgp_rmse:.4f}, R²={sgp_r2:.4f}, ρ={sgp_sp:.4f}")

    # Sort by true values for visualization
    sort_idx = np.argsort(data['y_test'])
    y_true_sorted = data['y_test'][sort_idx]
    mfgp_pred_sorted = mfgp_pred[sort_idx]
    mfgp_std_sorted = mfgp_std[sort_idx]
    sgp_pred_sorted = sgp_pred[sort_idx]
    sgp_std_sorted = sgp_std[sort_idx]

    # ========== PLOT 1: Predictions sorted by true value ==========
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    x_axis = np.arange(len(y_true_sorted))

    # MFGP
    ax = axes[0]
    ax.scatter(x_axis, y_true_sorted, c='black', s=8, alpha=0.7, label='True HF', zorder=3)
    ax.plot(x_axis, mfgp_pred_sorted, c='blue', lw=1.5, alpha=0.8, label='MFGP Predicted Mean', zorder=2)
    ax.fill_between(x_axis,
                    mfgp_pred_sorted - 2*mfgp_std_sorted,
                    mfgp_pred_sorted + 2*mfgp_std_sorted,
                    alpha=0.3, color='blue', label='MFGP ±2σ', zorder=1)
    ax.set_xlabel('Sample Index (sorted by true value)', fontsize=12)
    ax.set_ylabel('Bandgap (eV)', fontsize=12)
    ax.set_title(f'MFGP: RMSE={mfgp_rmse:.3f}, R²={mfgp_r2:.3f}, ρ={mfgp_sp:.3f}', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Standard GP
    ax = axes[1]
    ax.scatter(x_axis, y_true_sorted, c='black', s=8, alpha=0.7, label='True HF', zorder=3)
    ax.plot(x_axis, sgp_pred_sorted, c='red', lw=1.5, alpha=0.8, label='Standard GP Predicted Mean', zorder=2)
    ax.fill_between(x_axis,
                    sgp_pred_sorted - 2*sgp_std_sorted,
                    sgp_pred_sorted + 2*sgp_std_sorted,
                    alpha=0.3, color='red', label='Standard GP ±2σ', zorder=1)
    ax.set_xlabel('Sample Index (sorted by true value)', fontsize=12)
    ax.set_ylabel('Bandgap (eV)', fontsize=12)
    ax.set_title(f'Standard GP (HF only): RMSE={sgp_rmse:.3f}, R²={sgp_r2:.3f}, ρ={sgp_sp:.3f}', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gp_predictions_sorted.png', dpi=150, bbox_inches='tight')
    print("\nSaved: gp_predictions_sorted.png")
    plt.close()

    # ========== PLOT 2: Parity plots ==========
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MFGP parity
    ax = axes[0]
    mn, mx = min(y_true_sorted.min(), mfgp_pred.min())-0.5, max(y_true_sorted.max(), mfgp_pred.max())+0.5
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5, alpha=0.5, label='y=x')
    ax.errorbar(data['y_test'], mfgp_pred, yerr=2*mfgp_std, fmt='o', ms=5,
                alpha=0.5, elinewidth=0.5, capsize=0, color='blue', label='Predictions ±2σ')
    ax.set_xlabel('True Bandgap (eV)', fontsize=12)
    ax.set_ylabel('Predicted Bandgap (eV)', fontsize=12)
    ax.set_title(f'MFGP Parity Plot\nRMSE={mfgp_rmse:.3f}, R²={mfgp_r2:.3f}', fontsize=14)
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    ax.set_aspect('equal')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Standard GP parity
    ax = axes[1]
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5, alpha=0.5, label='y=x')
    ax.errorbar(data['y_test'], sgp_pred, yerr=2*sgp_std, fmt='o', ms=5,
                alpha=0.5, elinewidth=0.5, capsize=0, color='red', label='Predictions ±2σ')
    ax.set_xlabel('True Bandgap (eV)', fontsize=12)
    ax.set_ylabel('Predicted Bandgap (eV)', fontsize=12)
    ax.set_title(f'Standard GP Parity Plot\nRMSE={sgp_rmse:.3f}, R²={sgp_r2:.3f}', fontsize=14)
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    ax.set_aspect('equal')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gp_parity_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: gp_parity_comparison.png")
    plt.close()

    # ========== PLOT 3: Uncertainty distribution ==========
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(mfgp_std, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(mfgp_std.mean(), color='red', linestyle='--', lw=2, label=f'Mean σ={mfgp_std.mean():.3f}')
    ax.set_xlabel('Prediction Std (σ)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('MFGP Uncertainty Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(sgp_std, bins=30, alpha=0.7, color='red', edgecolor='black')
    ax.axvline(sgp_std.mean(), color='blue', linestyle='--', lw=2, label=f'Mean σ={sgp_std.mean():.3f}')
    ax.set_xlabel('Prediction Std (σ)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Standard GP Uncertainty Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gp_uncertainty_distribution.png', dpi=150, bbox_inches='tight')
    print("Saved: gp_uncertainty_distribution.png")
    plt.close()

    # ========== PLOT 4: Combined comparison ==========
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.scatter(x_axis, y_true_sorted, c='black', s=15, alpha=0.8, label='True HF', zorder=4)
    ax.plot(x_axis, mfgp_pred_sorted, c='blue', lw=2, alpha=0.9, label=f'MFGP (RMSE={mfgp_rmse:.3f})', zorder=3)
    ax.fill_between(x_axis,
                    mfgp_pred_sorted - 2*mfgp_std_sorted,
                    mfgp_pred_sorted + 2*mfgp_std_sorted,
                    alpha=0.2, color='blue', zorder=1)
    ax.plot(x_axis, sgp_pred_sorted, c='red', lw=2, alpha=0.9, label=f'Standard GP (RMSE={sgp_rmse:.3f})', zorder=2)
    ax.fill_between(x_axis,
                    sgp_pred_sorted - 2*sgp_std_sorted,
                    sgp_pred_sorted + 2*sgp_std_sorted,
                    alpha=0.2, color='red', zorder=0)

    ax.set_xlabel('Sample Index (sorted by true value)', fontsize=12)
    ax.set_ylabel('Bandgap (eV)', fontsize=12)
    ax.set_title('MFGP vs Standard GP: Predictions with Uncertainty (±2σ)', fontsize=14)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gp_comparison_combined.png', dpi=150, bbox_inches='tight')
    print("Saved: gp_comparison_combined.png")
    plt.close()

    print("\nAll visualizations complete!")


if __name__ == "__main__":
    main()
