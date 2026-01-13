#!/usr/bin/env python
"""
Compare DNGO+BO vs MFGP visualization
"""
import sys
import numpy as np
import torch
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'Pure_TL_BO'))
from mfgp_model import MultiFidelityGP
from DNGO.models import TransferLearningDNN

SEED = 1213  # Best performing seed


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


def generate_data(lookup, param_space, n_lofi=72, n_hifi=9, seed=1213):
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
        'y_test': np.array(y_test, dtype=np.float32)
    }


def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    sp, _ = spearmanr(y_true, y_pred)
    return rmse, r2, sp if not np.isnan(sp) else 0.0


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Seed: {SEED}")

    # Load data
    lookup, param_space = load_data()
    data = generate_data(lookup, param_space, seed=SEED)
    print(f"Data: {len(data['X_low'])} LF, {len(data['X_high'])} HF, Test: {len(data['X_test'])}")

    # 1. Train MFGP
    print("\nTraining MFGP...")
    t0 = time.time()
    mfgp = MultiFidelityGP(input_dim=3, device=device)
    mfgp.fit(data['X_low'], data['y_low'], data['X_high'], data['y_high'])
    mfgp_pred, mfgp_std = mfgp.predict(data['X_test'])
    mfgp_time = time.time() - t0
    mfgp_rmse, mfgp_r2, mfgp_sp = calc_metrics(data['y_test'], mfgp_pred)
    print(f"MFGP: RMSE={mfgp_rmse:.4f}, R²={mfgp_r2:.4f}, ρ={mfgp_sp:.4f}, Time={mfgp_time:.1f}s")

    # 2. Train DNGO+BO
    print("\nTraining DNGO+BO...")
    set_seeds(SEED)  # Reset seed
    t0 = time.time()
    dngo = TransferLearningDNN(input_dim=3, use_hyperparameter_bo=True, device=device)
    dngo.pretrain(data['X_low'], data['y_low'], bo_trials=20, verbose=False)
    dngo.finetune(data['X_high'], data['y_high'], bo_trials=20, verbose=False)
    dngo_pred = dngo.predict(data['X_test'])
    dngo_time = time.time() - t0
    dngo_rmse, dngo_r2, dngo_sp = calc_metrics(data['y_test'], dngo_pred)
    print(f"DNGO+BO: RMSE={dngo_rmse:.4f}, R²={dngo_r2:.4f}, ρ={dngo_sp:.4f}, Time={dngo_time:.1f}s")

    # Sort by true values
    sort_idx = np.argsort(data['y_test'])
    y_true_sorted = data['y_test'][sort_idx]
    mfgp_pred_sorted = mfgp_pred[sort_idx]
    mfgp_std_sorted = mfgp_std[sort_idx]
    dngo_pred_sorted = dngo_pred[sort_idx]
    x_axis = np.arange(len(y_true_sorted))

    # ========== PLOT 1: Stacked comparison ==========
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # MFGP (top)
    ax = axes[0]
    ax.scatter(x_axis, y_true_sorted, c='black', s=10, alpha=0.7, label='True HF', zorder=3)
    ax.plot(x_axis, mfgp_pred_sorted, c='blue', lw=1.5, alpha=0.9, label='MFGP Predicted', zorder=2)
    ax.fill_between(x_axis,
                    mfgp_pred_sorted - 2*mfgp_std_sorted,
                    mfgp_pred_sorted + 2*mfgp_std_sorted,
                    alpha=0.3, color='blue', label='±2σ', zorder=1)
    ax.set_ylabel('Bandgap (eV)', fontsize=12)
    ax.set_title(f'MFGP: RMSE={mfgp_rmse:.3f}, R²={mfgp_r2:.3f}, ρ={mfgp_sp:.3f} (Time: {mfgp_time:.1f}s)', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(x_axis))

    # DNGO+BO (bottom)
    ax = axes[1]
    ax.scatter(x_axis, y_true_sorted, c='black', s=10, alpha=0.7, label='True HF', zorder=3)
    ax.plot(x_axis, dngo_pred_sorted, c='green', lw=1.5, alpha=0.9, label='DNGO+BO Predicted', zorder=2)
    ax.set_xlabel('Sample Index (sorted by true value)', fontsize=12)
    ax.set_ylabel('Bandgap (eV)', fontsize=12)
    ax.set_title(f'DNGO+BO: RMSE={dngo_rmse:.3f}, R²={dngo_r2:.3f}, ρ={dngo_sp:.3f} (Time: {dngo_time:.1f}s)', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(x_axis))

    plt.tight_layout()
    plt.savefig('dngo_vs_mfgp_predictions.png', dpi=150, bbox_inches='tight')
    print("\nSaved: dngo_vs_mfgp_predictions.png")
    plt.close()

    # ========== PLOT 2: Parity plots side by side ==========
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    mn = min(data['y_test'].min(), mfgp_pred.min(), dngo_pred.min()) - 0.5
    mx = max(data['y_test'].max(), mfgp_pred.max(), dngo_pred.max()) + 0.5

    # MFGP parity
    ax = axes[0]
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5, alpha=0.5, label='y=x')
    ax.errorbar(data['y_test'], mfgp_pred, yerr=2*mfgp_std, fmt='o', ms=5,
                alpha=0.5, elinewidth=0.5, capsize=0, color='blue', label='MFGP ±2σ')
    ax.set_xlabel('True Bandgap (eV)', fontsize=12)
    ax.set_ylabel('Predicted Bandgap (eV)', fontsize=12)
    ax.set_title(f'MFGP\nRMSE={mfgp_rmse:.3f}, R²={mfgp_r2:.3f}', fontsize=14)
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    ax.set_aspect('equal')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # DNGO+BO parity
    ax = axes[1]
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5, alpha=0.5, label='y=x')
    ax.scatter(data['y_test'], dngo_pred, s=25, alpha=0.5, color='green', label='DNGO+BO')
    ax.set_xlabel('True Bandgap (eV)', fontsize=12)
    ax.set_ylabel('Predicted Bandgap (eV)', fontsize=12)
    ax.set_title(f'DNGO+BO\nRMSE={dngo_rmse:.3f}, R²={dngo_r2:.3f}', fontsize=14)
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    ax.set_aspect('equal')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dngo_vs_mfgp_parity.png', dpi=150, bbox_inches='tight')
    print("Saved: dngo_vs_mfgp_parity.png")
    plt.close()

    # ========== PLOT 3: Combined overlay ==========
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.scatter(x_axis, y_true_sorted, c='black', s=15, alpha=0.8, label='True HF', zorder=4)
    ax.plot(x_axis, mfgp_pred_sorted, c='blue', lw=2, alpha=0.9,
            label=f'MFGP (RMSE={mfgp_rmse:.3f}, {mfgp_time:.1f}s)', zorder=3)
    ax.fill_between(x_axis,
                    mfgp_pred_sorted - 2*mfgp_std_sorted,
                    mfgp_pred_sorted + 2*mfgp_std_sorted,
                    alpha=0.2, color='blue', zorder=1)
    ax.plot(x_axis, dngo_pred_sorted, c='green', lw=2, alpha=0.9,
            label=f'DNGO+BO (RMSE={dngo_rmse:.3f}, {dngo_time:.1f}s)', zorder=2)

    ax.set_xlabel('Sample Index (sorted by true value)', fontsize=12)
    ax.set_ylabel('Bandgap (eV)', fontsize=12)
    ax.set_title('MFGP vs DNGO+BO: Prediction Comparison', fontsize=14)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dngo_vs_mfgp_overlay.png', dpi=150, bbox_inches='tight')
    print("Saved: dngo_vs_mfgp_overlay.png")
    plt.close()

    print("\nComparison complete!")
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"{'Model':<15} {'RMSE':>10} {'R²':>10} {'ρ':>10} {'Time':>10}")
    print(f"{'-'*50}")
    print(f"{'MFGP':<15} {mfgp_rmse:>10.4f} {mfgp_r2:>10.4f} {mfgp_sp:>10.4f} {mfgp_time:>9.1f}s")
    print(f"{'DNGO+BO':<15} {dngo_rmse:>10.4f} {dngo_r2:>10.4f} {dngo_sp:>10.4f} {dngo_time:>9.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
