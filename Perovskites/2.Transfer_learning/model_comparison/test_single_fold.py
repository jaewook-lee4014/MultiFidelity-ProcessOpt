"""Quick test for single fold visualization"""
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

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_base_data():
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

def generate_data(lookup, all_combinations, seed=42):
    set_seeds(seed)
    rng = np.random.default_rng(seed)
    n_total = len(all_combinations)
    lofi_idx = rng.choice(n_total, size=72, replace=False)
    hifi_idx = rng.choice(n_total, size=9, replace=False)
    
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
    
    X_all, y_all_hf, y_all_lf = [], [], []
    composition_info = []
    for c in all_combinations:
        X_all.append(c['labels'])
        y_all_hf.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_hse06']))
        y_all_lf.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_gga']))
        composition_info.append({
            'org': c['names'][0], 'cat': c['names'][1], 'ani': c['names'][2],
            'org_idx': c['org_idx'], 'cat_idx': c['cat_idx'], 'ani_idx': c['ani_idx']
        })
    
    return {
        'X_low': np.array(X_low, dtype=np.float32),
        'y_low': np.array(y_low, dtype=np.float32),
        'X_high': np.array(X_high, dtype=np.float32),
        'y_high': np.array(y_high, dtype=np.float32),
        'X_all': np.array(X_all, dtype=np.float32),
        'y_all': np.array(y_all_hf, dtype=np.float32),
        'y_all_lf': np.array(y_all_lf, dtype=np.float32),
        'hifi_idx': hifi_idx,
        'lofi_idx': lofi_idx,
        'composition_info': composition_info
    }

def get_composition_sort_idx(composition_info):
    sort_keys = [(c['org_idx'], c['cat_idx'], c['ani_idx']) for c in composition_info]
    return np.lexsort(([k[2] for k in sort_keys], [k[1] for k in sort_keys], [k[0] for k in sort_keys]))

def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    return rmse, r2

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    lookup, all_combinations, param_space = load_base_data()
    seed = 456  # Use fold 3 seed for good results
    
    data = generate_data(lookup, all_combinations, seed=seed)
    train_idx = data['hifi_idx']
    n_all = len(data['X_all'])
    test_mask = np.ones(n_all, dtype=bool)
    test_mask[train_idx] = False
    test_idx = np.where(test_mask)[0]
    
    # Train MFGP
    set_seeds(seed)
    mfgp = MultiFidelityGP(input_dim=data['X_low'].shape[1], device=device)
    mfgp.fit(data['X_low'], data['y_low'], data['X_high'], data['y_high'])
    mfgp_pred_all, mfgp_std_all = mfgp.predict(data['X_all'], return_std=True)
    
    # Train DNGO (no BO for speed)
    set_seeds(seed)
    dngo = TransferLearningDNN(input_dim=data['X_low'].shape[1], hidden_dim=64, device=device)
    dngo.pretrain(data['X_low'], data['y_low'], epochs=100, verbose=False)
    dngo.finetune(data['X_high'], data['y_high'], epochs=50, verbose=False)
    features_train = dngo.extract_features(data['X_high'])
    blr = BayesianLinearRegression(alpha=1.0, beta=25.0)
    blr.fit(features_train, data['y_high'])
    features_all = dngo.extract_features(data['X_all'])
    dngo_pred_all, dngo_var_all = blr.predict_batch(features_all)
    dngo_std_all = np.sqrt(dngo_var_all)
    
    # Metrics
    y_test_hf = data['y_all'][test_idx]
    mfgp_rmse, mfgp_r2 = calc_metrics(y_test_hf, mfgp_pred_all[test_idx])
    dngo_rmse, dngo_r2 = calc_metrics(y_test_hf, dngo_pred_all[test_idx])
    print(f"MFGP: RMSE={mfgp_rmse:.4f}, R²={mfgp_r2:.4f}")
    print(f"DNGO: RMSE={dngo_rmse:.4f}, R²={dngo_r2:.4f}")
    
    # Sort by composition
    sort_idx = get_composition_sort_idx(data['composition_info'])
    y_sorted_hf = data['y_all'][sort_idx]
    mfgp_pred_sorted = mfgp_pred_all[sort_idx]
    mfgp_std_sorted = mfgp_std_all[sort_idx]
    dngo_pred_sorted = dngo_pred_all[sort_idx]
    dngo_std_sorted = dngo_std_all[sort_idx]
    train_mask = np.isin(sort_idx, train_idx)
    
    # Plot (2 panels for quick test)
    fig, axes = plt.subplots(2, 1, figsize=(24, 12))
    x_axis = np.arange(len(y_sorted_hf))
    n_cat = len(param_space['cation'])
    n_ani = len(param_space['anion'])
    group_size = n_cat * n_ani
    
    # MFGP
    ax = axes[0]
    ax.fill_between(x_axis, mfgp_pred_sorted - 2*mfgp_std_sorted, mfgp_pred_sorted + 2*mfgp_std_sorted,
                    alpha=0.3, color='blue', label='±2σ')
    ax.plot(x_axis, mfgp_pred_sorted, 'b-', linewidth=0.8, alpha=0.7, label='MFGP Predicted')
    ax.scatter(x_axis[~train_mask], y_sorted_hf[~train_mask], c='black', s=15, zorder=5, label='Test HF', alpha=0.6)
    ax.scatter(x_axis[train_mask], y_sorted_hf[train_mask], c='red', s=100, marker='*', zorder=6, label='Train HF (9)')
    ax.set_ylabel('Bandgap (eV)', fontsize=12)
    ax.set_title(f'MFGP (vs HF): RMSE={mfgp_rmse:.3f}, R²={mfgp_r2:.3f}', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    for i in range(1, len(param_space['organic'])):
        ax.axvline(x=i*group_size - 0.5, color='gray', linestyle='--', alpha=0.5)
    
    # DNGO
    ax = axes[1]
    ax.fill_between(x_axis, dngo_pred_sorted - 2*dngo_std_sorted, dngo_pred_sorted + 2*dngo_std_sorted,
                    alpha=0.3, color='green', label='±2σ')
    ax.plot(x_axis, dngo_pred_sorted, 'g-', linewidth=0.8, alpha=0.7, label='DNGO Predicted')
    ax.scatter(x_axis[~train_mask], y_sorted_hf[~train_mask], c='black', s=15, zorder=5, label='Test HF', alpha=0.6)
    ax.scatter(x_axis[train_mask], y_sorted_hf[train_mask], c='red', s=100, marker='*', zorder=6, label='Train HF (9)')
    ax.set_ylabel('Bandgap (eV)', fontsize=12)
    ax.set_title(f'DNGO (vs HF): RMSE={dngo_rmse:.3f}, R²={dngo_r2:.3f}', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Composition Index', fontsize=12)
    for i in range(1, len(param_space['organic'])):
        ax.axvline(x=i*group_size - 0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Suptitle and organic labels with spacing fix
    plt.suptitle(f'Test Fold (seed={seed}) - Sorted by Composition', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Add organic labels
    org_names = param_space['organic']
    for i, org in enumerate(org_names):
        mid_x = i * group_size + group_size / 2
        y_top = axes[0].get_ylim()[1]
        y_range = axes[0].get_ylim()[1] - axes[0].get_ylim()[0]
        axes[0].text(mid_x, y_top + y_range * 0.08, org.capitalize(),
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.savefig('test_layout.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: test_layout.png")

if __name__ == '__main__':
    main()
