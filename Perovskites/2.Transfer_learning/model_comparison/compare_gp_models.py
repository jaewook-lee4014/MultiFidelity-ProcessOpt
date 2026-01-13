#!/usr/bin/env python
"""
GP Model Comparison: MFGP vs Standard GP

Only runs:
1. MFGP (Multi-Fidelity GP) - uses both LF and HF data
2. Standard GP - uses HF data only (baseline)

Fixed seeds for reproducibility.
"""
import sys
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

sys.path.insert(0, str(Path(__file__).parent.parent / 'Pure_TL_BO'))
from mfgp_model import MultiFidelityGP, StandardGP

FIXED_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]


def set_all_seeds(seed):
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


def generate_dataset(lookup, param_space, n_lofi, n_hifi, seed):
    set_all_seeds(seed)
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

    return {
        'X_low': np.array(X_low, dtype=np.float32),
        'y_low': np.array(y_low, dtype=np.float32),
        'X_high': np.array(X_high, dtype=np.float32),
        'y_high': np.array(y_high, dtype=np.float32)
    }


def generate_test_data(lookup, param_space):
    all_X, all_y = [], []
    for i, org in enumerate(param_space['organic'], 1):
        for j, cat in enumerate(param_space['cation'], 1):
            for k, ani in enumerate(param_space['anion'], 1):
                all_X.append([i,j,k])
                all_y.append(np.amin(lookup[org.capitalize()][cat][ani]['bandgap_hse06']))
    return np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.float32)


def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    sp, _ = spearmanr(y_true, y_pred)
    return rmse, r2, sp if not np.isnan(sp) else 0.0


def plot_results(y_true, preds_dict, run_idx, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (name, (y_pred, y_std)) in enumerate(preds_dict.items()):
        ax = axes[idx]
        rmse, r2, sp = calc_metrics(y_true, y_pred)

        # Parity plot
        mn, mx = min(y_true.min(), y_pred.min())-0.5, max(y_true.max(), y_pred.max())+0.5
        ax.plot([mn,mx], [mn,mx], 'k--', lw=1, alpha=0.5)

        if y_std is not None:
            ax.errorbar(y_true, y_pred, yerr=2*y_std, fmt='o', ms=4, alpha=0.5, elinewidth=0.5, capsize=0)
        else:
            ax.scatter(y_true, y_pred, s=15, alpha=0.6)

        ax.set_xlabel('True Bandgap (eV)')
        ax.set_ylabel('Predicted Bandgap (eV)')
        ax.set_title(f'{name}\nRMSE={rmse:.3f}, R²={r2:.3f}, ρ={sp:.3f}')
        ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f'gp_parity_run{run_idx+1}.png', dpi=150)
    plt.close()


def run_comparison(n_runs=10, n_lofi=72, n_hifi=9, save_dir='results_gp'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Runs: {n_runs}, Data: {n_lofi} LF + {n_hifi} HF")
    print(f"Seeds: {FIXED_SEEDS[:n_runs]}")
    print("="*60)

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    lookup, param_space = load_data()
    X_test, y_test = generate_test_data(lookup, param_space)
    print(f"Test: {len(X_test)} samples, HF range: [{y_test.min():.2f}, {y_test.max():.2f}]")

    results = {'MFGP': [], 'Standard_GP': []}

    for run in range(n_runs):
        seed = FIXED_SEEDS[run]
        print(f"\nRUN {run+1}/{n_runs} (seed={seed})")

        data = generate_dataset(lookup, param_space, n_lofi, n_hifi, seed)
        preds = {}

        # 1. MFGP
        print("  1. MFGP...", end=" ")
        try:
            model = MultiFidelityGP(input_dim=3, device=device)
            t0 = time.time()
            model.fit(data['X_low'], data['y_low'], data['X_high'], data['y_high'])
            y_pred, y_std = model.predict(X_test)
            elapsed = time.time() - t0
            rmse, r2, sp = calc_metrics(y_test, y_pred)
            results['MFGP'].append({'rmse': rmse, 'r2': r2, 'sp': sp, 'time': elapsed})
            preds['MFGP'] = (y_pred, y_std)
            print(f"RMSE={rmse:.4f}, R²={r2:.4f}, ρ={sp:.4f}, {elapsed:.1f}s")
        except Exception as e:
            print(f"Failed: {e}")
            results['MFGP'].append({'rmse': np.nan, 'r2': np.nan, 'sp': np.nan, 'time': np.nan})

        # 2. Standard GP
        print("  2. Standard GP...", end=" ")
        try:
            model = StandardGP(input_dim=3, device=device)
            t0 = time.time()
            model.fit(data['X_high'], data['y_high'])
            y_pred, y_std = model.predict(X_test)
            elapsed = time.time() - t0
            rmse, r2, sp = calc_metrics(y_test, y_pred)
            results['Standard_GP'].append({'rmse': rmse, 'r2': r2, 'sp': sp, 'time': elapsed})
            preds['Standard_GP'] = (y_pred, y_std)
            print(f"RMSE={rmse:.4f}, R²={r2:.4f}, ρ={sp:.4f}, {elapsed:.1f}s")
        except Exception as e:
            print(f"Failed: {e}")
            results['Standard_GP'].append({'rmse': np.nan, 'r2': np.nan, 'sp': np.nan, 'time': np.nan})

        if preds:
            plot_results(y_test, preds, run, save_path)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Model':<15} {'RMSE':>18} {'R²':>18} {'Spearman':>18}")
    print("-"*70)

    for name in ['MFGP', 'Standard_GP']:
        vals = [r for r in results[name] if not np.isnan(r['rmse'])]
        if vals:
            rmse = np.mean([v['rmse'] for v in vals])
            rmse_std = np.std([v['rmse'] for v in vals])
            r2 = np.mean([v['r2'] for v in vals])
            r2_std = np.std([v['r2'] for v in vals])
            sp = np.mean([v['sp'] for v in vals])
            sp_std = np.std([v['sp'] for v in vals])
            print(f"{name:<15} {rmse:.4f}±{rmse_std:.4f}    {r2:.4f}±{r2_std:.4f}    {sp:.4f}±{sp_std:.4f}")

    print("-"*70)
    print("EXISTING (for comparison):")
    print(f"{'DNGO+BO':<15} {'0.6489±0.0917':>18} {'0.6674±0.0943':>18} {'0.8027±0.0480':>18}")
    print(f"{'BNN+BO':<15} {'0.6834±0.1575':>18} {'0.6192±0.1877':>18} {'0.8265±0.0324':>18}")
    print("="*70)

    # Save
    df = pd.DataFrame([
        {'model': m, 'run': i+1, 'seed': FIXED_SEEDS[i], **r}
        for m in results for i, r in enumerate(results[m])
    ])
    df.to_csv(save_path / 'gp_results.csv', index=False)
    print(f"\nSaved to {save_path}")

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n-runs', type=int, default=10)
    p.add_argument('--n-lofi', type=int, default=72)
    p.add_argument('--n-hifi', type=int, default=9)
    p.add_argument('--save-dir', type=str, default='results_gp')
    args = p.parse_args()
    run_comparison(args.n_runs, args.n_lofi, args.n_hifi, args.save_dir)
