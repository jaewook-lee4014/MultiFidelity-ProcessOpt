"""
Visualize DNGO+BO with Progressive Unfreezing vs MFGP for all 10 folds/seeds
Sorted by composition (Organic -> Cation -> Anion) instead of true value
Training data shown in RED
Results saved in timestamped folder
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
from datetime import datetime
import time
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

    # Generate all combinations with names
    all_combinations = []
    for i, org in enumerate(param_space['organic'], 1):
        for j, cat in enumerate(param_space['cation'], 1):
            for k, ani in enumerate(param_space['anion'], 1):
                all_combinations.append({
                    'labels': [i, j, k],
                    'names': [org, cat, ani],
                    'org_idx': i, 'cat_idx': j, 'ani_idx': k
                })

    return lookup, all_combinations, param_space


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
    for idx in hifi_idx:
        c = all_combinations[idx]
        X_high.append(c['labels'])
        y_high.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_hse06']))

    # All data for test (both HF and LF)
    X_all, y_all_hf, y_all_lf = [], [], []
    composition_info = []
    for c in all_combinations:
        X_all.append(c['labels'])
        y_all_hf.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_hse06']))
        y_all_lf.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_gga']))
        composition_info.append({
            'org': c['names'][0],
            'cat': c['names'][1],
            'ani': c['names'][2],
            'org_idx': c['org_idx'],
            'cat_idx': c['cat_idx'],
            'ani_idx': c['ani_idx']
        })

    return {
        'X_low': np.array(X_low, dtype=np.float32),
        'y_low': np.array(y_low, dtype=np.float32),
        'X_high': np.array(X_high, dtype=np.float32),
        'y_high': np.array(y_high, dtype=np.float32),
        'X_all': np.array(X_all, dtype=np.float32),
        'y_all': np.array(y_all_hf, dtype=np.float32),  # HF for MFGP/DNGO+BO
        'y_all_lf': np.array(y_all_lf, dtype=np.float32),  # LF for Pretrain Only
        'hifi_idx': hifi_idx,
        'lofi_idx': lofi_idx,
        'composition_info': composition_info
    }


def get_composition_sort_idx(composition_info):
    """Sort by Organic -> Cation -> Anion"""
    # Create sort keys: (org_idx, cat_idx, ani_idx)
    sort_keys = [(c['org_idx'], c['cat_idx'], c['ani_idx']) for c in composition_info]
    return np.lexsort((
        [k[2] for k in sort_keys],  # Anion (tertiary)
        [k[1] for k in sort_keys],  # Cation (secondary)
        [k[0] for k in sort_keys]   # Organic (primary)
    ))


def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    sp, _ = spearmanr(y_true, y_pred)
    return rmse, r2, sp if not np.isnan(sp) else 0.0


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load base data once
    lookup, all_combinations, param_space = load_base_data()
    print(f"Total combinations: {len(all_combinations)}")
    print(f"Organics: {param_space['organic']}")
    print(f"Cations: {param_space['cation']}")
    print(f"Anions: {param_space['anion']}")

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    vis_dir = Path(__file__).parent / 'visualizations' / f'{timestamp}_progressive_unfreezing'
    vis_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {vis_dir}")

    # Store all results for summary
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

        # Train indices
        train_idx = data['hifi_idx']
        n_all = len(data['X_all'])
        test_mask = np.ones(n_all, dtype=bool)
        test_mask[train_idx] = False
        test_idx = np.where(test_mask)[0]

        print(f"Train (HF): {len(train_idx)}, Test: {len(test_idx)}")

        # ============== Train MFGP ==============
        t0 = time.time()
        set_seeds(seed)
        mfgp = MultiFidelityGP(input_dim=data['X_low'].shape[1], device=device)
        mfgp.fit(data['X_low'], data['y_low'], data['X_high'], data['y_high'])
        mfgp_pred_all, mfgp_std_all = mfgp.predict(data['X_all'], return_std=True)
        mfgp_time = time.time() - t0

        # ============== Train DNGO (Pretrain + Finetune with Progressive Unfreezing) ==============
        t0 = time.time()
        set_seeds(seed)
        dngo = TransferLearningDNN(
            input_dim=data['X_low'].shape[1],
            hidden_dim=64,
            device=device,
            use_hyperparameter_bo=True  # Enable BO for hyperparameters
        )
        # Pretrain with BO (3 trials)
        dngo.pretrain(data['X_low'], data['y_low'], bo_trials=3, verbose=False)
        # Finetune with Progressive Unfreezing (BO 3 trials)
        dngo.finetune(data['X_high'], data['y_high'], bo_trials=3,
                      use_progressive_unfreezing=True, verbose=False)

        features_train = dngo.extract_features(data['X_high'])
        blr = BayesianLinearRegression(alpha=1.0, beta=25.0)
        blr.fit(features_train, data['y_high'])

        features_all = dngo.extract_features(data['X_all'])
        dngo_pred_all, dngo_var_all = blr.predict_batch(features_all)
        dngo_std_all = np.sqrt(dngo_var_all)
        dngo_time = time.time() - t0

        # ============== Train DNGO (Pretrain Only) ==============
        t0 = time.time()
        set_seeds(seed)
        dngo_pre = TransferLearningDNN(
            input_dim=data['X_low'].shape[1],
            hidden_dim=64,
            device=device,
            use_hyperparameter_bo=True
        )
        dngo_pre.pretrain(data['X_low'], data['y_low'], bo_trials=3, verbose=False)
        # No finetune - use LF data for BLR

        features_lf = dngo_pre.extract_features(data['X_low'])
        blr_pre = BayesianLinearRegression(alpha=1.0, beta=25.0)
        blr_pre.fit(features_lf, data['y_low'])

        features_all_pre = dngo_pre.extract_features(data['X_all'])
        pre_pred_all, pre_var_all = blr_pre.predict_batch(features_all_pre)
        pre_std_all = np.sqrt(pre_var_all)
        pretrain_time = time.time() - t0

        # Calculate metrics (HF-based for MFGP/DNGO, LF-based for Pretrain)
        y_test_hf = data['y_all'][test_idx]
        y_test_lf = data['y_all_lf'][test_idx]

        mfgp_pred_test = mfgp_pred_all[test_idx]
        dngo_pred_test = dngo_pred_all[test_idx]
        pre_pred_test = pre_pred_all[test_idx]

        mfgp_rmse, mfgp_r2, mfgp_rho = calc_metrics(y_test_hf, mfgp_pred_test)
        dngo_rmse, dngo_r2, dngo_rho = calc_metrics(y_test_hf, dngo_pred_test)
        pre_rmse, pre_r2, pre_rho = calc_metrics(y_test_lf, pre_pred_test)  # LF comparison!

        print(f"MFGP: RMSE={mfgp_rmse:.4f}, R²={mfgp_r2:.4f} (vs HF) - Time: {mfgp_time:.2f}s")
        print(f"DNGO+ProgUnfreeze: RMSE={dngo_rmse:.4f}, R²={dngo_r2:.4f} (vs HF) - Time: {dngo_time:.2f}s")
        print(f"Pretrain Only: RMSE={pre_rmse:.4f}, R²={pre_r2:.4f} (vs LF) - Time: {pretrain_time:.2f}s")

        # Store results
        all_results.append({
            'fold': fold_idx,
            'seed': seed,
            'mfgp_rmse': mfgp_rmse, 'mfgp_r2': mfgp_r2, 'mfgp_time': mfgp_time,
            'dngo_rmse': dngo_rmse, 'dngo_r2': dngo_r2, 'dngo_time': dngo_time,
            'pre_rmse': pre_rmse, 'pre_r2': pre_r2, 'pre_time': pretrain_time
        })

        # ============== Sort by composition ==============
        sort_idx = get_composition_sort_idx(data['composition_info'])
        y_sorted_hf = data['y_all'][sort_idx]      # HF true values
        y_sorted_lf = data['y_all_lf'][sort_idx]   # LF true values
        mfgp_pred_sorted = mfgp_pred_all[sort_idx]
        mfgp_std_sorted = mfgp_std_all[sort_idx]
        dngo_pred_sorted = dngo_pred_all[sort_idx]
        dngo_std_sorted = dngo_std_all[sort_idx]
        pre_pred_sorted = pre_pred_all[sort_idx]
        pre_std_sorted = pre_std_all[sort_idx]

        # Train mask in sorted order (HF train data)
        train_mask = np.isin(sort_idx, train_idx)

        # LF train mask (for pretrain visualization)
        lofi_idx = data['lofi_idx']
        lf_train_mask = np.isin(sort_idx, lofi_idx)

        # Get composition labels for x-axis (full names)
        comp_labels = []
        for idx in sort_idx:
            c = data['composition_info'][idx]
            # Use shorter but recognizable names
            org_short = c['org'][:4].capitalize()
            cat_short = c['cat'][:2].capitalize()
            ani_short = c['ani'][:2].capitalize()
            comp_labels.append(f"{org_short}-{cat_short}-{ani_short}")

        # ============== Plot ==============
        fig, axes = plt.subplots(3, 1, figsize=(24, 18))
        x_axis = np.arange(len(y_sorted_hf))
        n_cat = len(param_space['cation'])
        n_ani = len(param_space['anion'])
        group_size = n_cat * n_ani

        # MFGP plot (vs HF true)
        ax = axes[0]
        ax.fill_between(x_axis,
                        mfgp_pred_sorted - 2*mfgp_std_sorted,
                        mfgp_pred_sorted + 2*mfgp_std_sorted,
                        alpha=0.3, color='blue', label=f'±2σ')
        ax.plot(x_axis, mfgp_pred_sorted, 'b-', linewidth=0.8, alpha=0.7, label='MFGP Predicted')
        ax.scatter(x_axis[~train_mask], y_sorted_hf[~train_mask], c='black', s=15,
                   zorder=5, label='Test HF (true)', alpha=0.6)
        ax.scatter(x_axis[train_mask], y_sorted_hf[train_mask], c='red', s=100,
                   marker='*', zorder=6, label='Train HF (9)', edgecolors='darkred')
        ax.set_ylabel('Bandgap (eV)', fontsize=12)
        ax.set_title(f'MFGP (vs HF): RMSE={mfgp_rmse:.3f}, R²={mfgp_r2:.3f}', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        for i in range(1, len(param_space['organic'])):
            ax.axvline(x=i*group_size - 0.5, color='gray', linestyle='--', alpha=0.5)

        # DNGO+BO+ProgUnfreeze plot (vs HF true)
        ax = axes[1]
        ax.fill_between(x_axis,
                        dngo_pred_sorted - 2*dngo_std_sorted,
                        dngo_pred_sorted + 2*dngo_std_sorted,
                        alpha=0.3, color='green', label=f'±2σ')
        ax.plot(x_axis, dngo_pred_sorted, 'g-', linewidth=0.8, alpha=0.7, label='DNGO+ProgUnfreeze Predicted')
        ax.scatter(x_axis[~train_mask], y_sorted_hf[~train_mask], c='black', s=15,
                   zorder=5, label='Test HF (true)', alpha=0.6)
        ax.scatter(x_axis[train_mask], y_sorted_hf[train_mask], c='red', s=100,
                   marker='*', zorder=6, label='Train HF (9)', edgecolors='darkred')
        ax.set_ylabel('Bandgap (eV)', fontsize=12)
        ax.set_title(f'DNGO + Progressive Unfreezing (vs HF): RMSE={dngo_rmse:.3f}, R²={dngo_r2:.3f}', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        for i in range(1, len(param_space['organic'])):
            ax.axvline(x=i*group_size - 0.5, color='gray', linestyle='--', alpha=0.5)

        # Pretrain Only plot (vs LF true!)
        ax = axes[2]
        ax.fill_between(x_axis,
                        pre_pred_sorted - 2*pre_std_sorted,
                        pre_pred_sorted + 2*pre_std_sorted,
                        alpha=0.3, color='orange', label=f'±2σ')
        ax.plot(x_axis, pre_pred_sorted, color='orange', linewidth=0.8, alpha=0.7, label='Pretrain Only Predicted')
        ax.scatter(x_axis[~lf_train_mask], y_sorted_lf[~lf_train_mask], c='black', s=15,
                   zorder=5, label='Test LF (true)', alpha=0.6)
        ax.scatter(x_axis[lf_train_mask], y_sorted_lf[lf_train_mask], c='red', s=40,
                   marker='o', zorder=6, label='Train LF (72)', edgecolors='darkred', alpha=0.7)
        ax.set_ylabel('Bandgap (eV)', fontsize=12)
        ax.set_title(f'DNGO Pretrain Only (vs LF): RMSE={pre_rmse:.3f}, R²={pre_r2:.3f}', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        for i in range(1, len(param_space['organic'])):
            ax.axvline(x=i*group_size - 0.5, color='gray', linestyle='--', alpha=0.5)

        # Set x-axis labels with composition names (only on bottom plot)
        ax.set_xticks(x_axis)
        ax.set_xticklabels(comp_labels, rotation=90, fontsize=6)
        ax.set_xlabel('Composition (Organic-Cation-Anion)', fontsize=12)

        plt.suptitle(f'Fold {fold_idx} (seed={seed}) - Progressive Unfreezing - Sorted by Composition', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space at top for suptitle

        # Add organic labels at the top (after tight_layout)
        org_names = param_space['organic']
        for i, org in enumerate(org_names):
            mid_x = i * group_size + group_size / 2
            # Use axes transform to place labels above the plot
            y_top = axes[0].get_ylim()[1]
            y_range = axes[0].get_ylim()[1] - axes[0].get_ylim()[0]
            axes[0].text(mid_x, y_top + y_range * 0.08, org.capitalize(),
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
        plt.savefig(fold_dir / 'predictions_by_composition.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved to: {fold_dir}/")

    # Save summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    mfgp_r2_mean = np.mean([r['mfgp_r2'] for r in all_results])
    dngo_r2_mean = np.mean([r['dngo_r2'] for r in all_results])
    pre_r2_mean = np.mean([r['pre_r2'] for r in all_results])

    mfgp_rmse_mean = np.mean([r['mfgp_rmse'] for r in all_results])
    dngo_rmse_mean = np.mean([r['dngo_rmse'] for r in all_results])
    pre_rmse_mean = np.mean([r['pre_rmse'] for r in all_results])

    mfgp_time_mean = np.mean([r['mfgp_time'] for r in all_results])
    dngo_time_mean = np.mean([r['dngo_time'] for r in all_results])
    pre_time_mean = np.mean([r['pre_time'] for r in all_results])

    print(f"MFGP:              Avg RMSE={mfgp_rmse_mean:.4f}, Avg R²={mfgp_r2_mean:.4f}, Avg Time={mfgp_time_mean:.2f}s")
    print(f"DNGO+ProgUnfreeze: Avg RMSE={dngo_rmse_mean:.4f}, Avg R²={dngo_r2_mean:.4f}, Avg Time={dngo_time_mean:.2f}s")
    print(f"Pretrain Only:     Avg RMSE={pre_rmse_mean:.4f}, Avg R²={pre_r2_mean:.4f}, Avg Time={pre_time_mean:.2f}s")

    # Save results to CSV
    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv(vis_dir / 'results_summary.csv', index=False)
    print(f"\nResults saved to: {vis_dir / 'results_summary.csv'}")

    print(f"\nAll {len(SEEDS)} folds visualized!")
    print(f"Output directory: {vis_dir}")


if __name__ == '__main__':
    main()
