"""
DNGO tanh vs ReLU 비교 실험
5개 모델 비교: MFGP, DNGO-ReLU, DNGO-tanh, Pretrain-ReLU (vs LF), Pretrain-tanh (vs LF)
시각화: 기존과 동일한 형식 (분산 포함, composition 정렬)
"""
import sys
import numpy as np
import torch
import pickle
import json
import copy
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from datetime import datetime
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'Pure_TL_BO'))
from DNGO.models import TransferLearningDNN, BayesianLinearRegression

# MFGP import
try:
    from mfgp_model import MultiFidelityGP
    MFGP_AVAILABLE = True
except ImportError:
    MFGP_AVAILABLE = False
    print("Warning: MFGP not available")

# Configuration
BO_TRIALS = 50
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]


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


def generate_data(lookup, all_combinations, n_lofi=72, n_hifi=9, seed=42):
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
        'y_all': np.array(y_all_hf, dtype=np.float32),
        'y_all_lf': np.array(y_all_lf, dtype=np.float32),
        'hifi_idx': hifi_idx,
        'lofi_idx': lofi_idx,
        'composition_info': composition_info
    }


def get_composition_sort_idx(composition_info):
    """Sort by Organic -> Cation -> Anion"""
    sort_keys = [(c['org_idx'], c['cat_idx'], c['ani_idx']) for c in composition_info]
    return np.lexsort((
        [k[2] for k in sort_keys],
        [k[1] for k in sort_keys],
        [k[0] for k in sort_keys]
    ))


def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    sp, _ = spearmanr(y_true, y_pred)
    return rmse, r2, sp if not np.isnan(sp) else 0.0


def run_mfgp(data, device, seed):
    """MFGP 실행"""
    if not MFGP_AVAILABLE:
        return None, None

    set_seeds(seed)
    mfgp = MultiFidelityGP(input_dim=data['X_low'].shape[1], device=device)
    mfgp.fit(data['X_low'], data['y_low'], data['X_high'], data['y_high'])
    mfgp_pred, mfgp_std = mfgp.predict(data['X_all'], return_std=True)
    return mfgp_pred, mfgp_std


def run_pretrain(data, device, seed, activation='relu'):
    """Pretrain 모델 (Low-fidelity 예측용)"""
    set_seeds(seed)

    model = TransferLearningDNN(
        input_dim=data['X_low'].shape[1],
        hidden_dim=64,
        device=device,
        use_hyperparameter_bo=True,
        activation=activation
    )
    model.pretrain(data['X_low'], data['y_low'], bo_trials=BO_TRIALS, verbose=False)

    # BLR for uncertainty
    features = model.extract_features(data['X_low'])
    blr = BayesianLinearRegression(alpha=1.0, beta=25.0)
    blr.fit(features, data['y_low'])

    features_all = model.extract_features(data['X_all'])
    pred_all, var_all = blr.predict_batch(features_all)
    std_all = np.sqrt(var_all)

    return pred_all, std_all, model


def run_dngo(data, device, seed, pretrained_model, activation='relu'):
    """DNGO 실행 (pretrained model 제공)"""
    set_seeds(seed)

    model = copy.deepcopy(pretrained_model)
    model.finetune(data['X_high'], data['y_high'], bo_trials=BO_TRIALS, verbose=False)

    # BLR for uncertainty
    features = model.extract_features(data['X_high'])
    blr = BayesianLinearRegression(alpha=1.0, beta=25.0)
    blr.fit(features, data['y_high'])

    features_all = model.extract_features(data['X_all'])
    pred_all, var_all = blr.predict_batch(features_all)
    std_all = np.sqrt(var_all)

    return pred_all, std_all, model


def create_5panel_visualization(fold_idx, seed, data, results, param_space, output_dir):
    """5패널 시각화 생성 (기존 형식과 동일)"""

    # Sort by composition
    sort_idx = get_composition_sort_idx(data['composition_info'])
    y_sorted_hf = data['y_all'][sort_idx]
    y_sorted_lf = data['y_all_lf'][sort_idx]

    train_mask = np.isin(sort_idx, data['hifi_idx'])
    lf_train_mask = np.isin(sort_idx, data['lofi_idx'])

    # Composition labels
    comp_labels = []
    for idx in sort_idx:
        c = data['composition_info'][idx]
        org_short = c['org'][:4].capitalize()
        cat_short = c['cat'][:2].capitalize()
        ani_short = c['ani'][:2].capitalize()
        comp_labels.append(f"{org_short}-{cat_short}-{ani_short}")

    # Create figure
    fig, axes = plt.subplots(5, 1, figsize=(24, 30))
    x_axis = np.arange(len(y_sorted_hf))
    n_cat = len(param_space['cation'])
    n_ani = len(param_space['anion'])
    group_size = n_cat * n_ani

    # Test indices
    n_all = len(data['X_all'])
    test_mask_global = np.ones(n_all, dtype=bool)
    test_mask_global[data['hifi_idx']] = False
    test_idx = np.where(test_mask_global)[0]
    y_test_hf = data['y_all'][test_idx]
    y_test_lf = data['y_all_lf'][test_idx]

    panels_info = [
        ('MFGP', results.get('mfgp_pred'), results.get('mfgp_std'), 'blue', 'HF', y_sorted_hf, train_mask),
        ('DNGO-ReLU', results.get('dngo_relu_pred'), results.get('dngo_relu_std'), 'green', 'HF', y_sorted_hf, train_mask),
        ('DNGO-tanh', results.get('dngo_tanh_pred'), results.get('dngo_tanh_std'), 'purple', 'HF', y_sorted_hf, train_mask),
        ('Pretrain-ReLU', results.get('pretrain_relu_pred'), results.get('pretrain_relu_std'), 'orange', 'LF', y_sorted_lf, lf_train_mask),
        ('Pretrain-tanh', results.get('pretrain_tanh_pred'), results.get('pretrain_tanh_std'), 'red', 'LF', y_sorted_lf, lf_train_mask),
    ]

    for ax_idx, (name, pred, std, color, compare_type, y_sorted, train_mask_panel) in enumerate(panels_info):
        ax = axes[ax_idx]

        if pred is None:
            ax.text(0.5, 0.5, f'{name}\nNot Available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=16)
            ax.set_title(f'{name} (vs {compare_type})', fontsize=14)
            continue

        # Sort predictions
        pred_sorted = pred[sort_idx]
        std_sorted = std[sort_idx]

        # Calculate metrics
        if compare_type == 'HF':
            rmse, r2, _ = calc_metrics(y_test_hf, pred[test_idx])
        else:
            rmse, r2, _ = calc_metrics(y_test_lf, pred[test_idx])

        # Plot uncertainty band
        ax.fill_between(x_axis, pred_sorted - 2*std_sorted, pred_sorted + 2*std_sorted,
                        alpha=0.3, color=color, label='±2σ')

        # Plot prediction line
        ax.plot(x_axis, pred_sorted, color=color, linewidth=0.8, alpha=0.7, label=f'{name} Predicted')

        # Plot test points
        ax.scatter(x_axis[~train_mask_panel], y_sorted[~train_mask_panel],
                  c='black', s=15, zorder=5, label=f'Test {compare_type}', alpha=0.6)

        # Plot train points
        if compare_type == 'HF':
            ax.scatter(x_axis[train_mask_panel], y_sorted[train_mask_panel],
                      c='red', s=100, marker='*', zorder=6, label='Train HF (9)', edgecolors='darkred')
        else:
            ax.scatter(x_axis[train_mask_panel], y_sorted[train_mask_panel],
                      c='red', s=40, marker='o', zorder=6, label='Train LF (72)', edgecolors='darkred', alpha=0.7)

        ax.set_ylabel('Bandgap (eV)', fontsize=12)
        ax.set_title(f'{name} (vs {compare_type}): RMSE={rmse:.3f}, R²={r2:.3f}', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Organic group separators
        for i in range(1, len(param_space['organic'])):
            ax.axvline(x=i*group_size - 0.5, color='gray', linestyle='--', alpha=0.5)

    # X-axis labels on bottom plot
    axes[-1].set_xticks(x_axis)
    axes[-1].set_xticklabels(comp_labels, rotation=90, fontsize=6)
    axes[-1].set_xlabel('Composition (Organic-Cation-Anion)', fontsize=12)

    # Suptitle and organic labels
    plt.suptitle(f'Fold {fold_idx} (seed={seed}) - tanh vs ReLU Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Organic labels
    org_names = param_space['organic']
    for i, org in enumerate(org_names):
        mid_x = i * group_size + group_size / 2
        y_top = axes[0].get_ylim()[1]
        y_range = axes[0].get_ylim()[1] - axes[0].get_ylim()[0]
        axes[0].text(mid_x, y_top + y_range * 0.08, org.capitalize(),
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    output_path = output_dir / f'fold_{fold_idx}_5panel.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"BO Trials: {BO_TRIALS}")
    print(f"Testing: MFGP, DNGO-ReLU, DNGO-tanh, Pretrain-ReLU, Pretrain-tanh")

    lookup, all_combinations, param_space = load_base_data()

    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / 'visualizations' / f'{timestamp}_tanh_comparison_bo{BO_TRIALS}'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    results_list = []
    total_start = time.time()

    for fold_idx, seed in enumerate(SEEDS, 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx} (seed={seed})")
        print('='*60)

        data = generate_data(lookup, all_combinations, seed=seed)

        train_idx = data['hifi_idx']
        n_all = len(data['X_all'])
        test_mask = np.ones(n_all, dtype=bool)
        test_mask[train_idx] = False
        test_idx = np.where(test_mask)[0]
        y_test_hf = data['y_all'][test_idx]
        y_test_lf = data['y_all_lf'][test_idx]

        fold_results = {}

        # 1. MFGP
        t0 = time.time()
        mfgp_pred, mfgp_std = run_mfgp(data, device, seed)
        mfgp_time = time.time() - t0
        if mfgp_pred is not None:
            mfgp_rmse, mfgp_r2, _ = calc_metrics(y_test_hf, mfgp_pred[test_idx])
            print(f"  MFGP: RMSE={mfgp_rmse:.4f}, R²={mfgp_r2:.4f} (time: {mfgp_time:.2f}s)")
            fold_results['mfgp_pred'] = mfgp_pred
            fold_results['mfgp_std'] = mfgp_std
        else:
            mfgp_rmse, mfgp_r2 = None, None
            print(f"  MFGP: Not available")

        # 2. Pretrain ReLU
        t0 = time.time()
        pretrain_relu_pred, pretrain_relu_std, pretrain_relu_model = run_pretrain(data, device, seed, activation='relu')
        pretrain_relu_time = time.time() - t0
        pretrain_relu_rmse, pretrain_relu_r2, _ = calc_metrics(y_test_lf, pretrain_relu_pred[test_idx])
        print(f"  Pretrain-ReLU (vs LF): RMSE={pretrain_relu_rmse:.4f}, R²={pretrain_relu_r2:.4f} (time: {pretrain_relu_time:.2f}s)")
        fold_results['pretrain_relu_pred'] = pretrain_relu_pred
        fold_results['pretrain_relu_std'] = pretrain_relu_std

        # 3. DNGO-ReLU
        t0 = time.time()
        dngo_relu_pred, dngo_relu_std, _ = run_dngo(data, device, seed, pretrain_relu_model, activation='relu')
        dngo_relu_time = time.time() - t0
        dngo_relu_rmse, dngo_relu_r2, _ = calc_metrics(y_test_hf, dngo_relu_pred[test_idx])
        print(f"  DNGO-ReLU: RMSE={dngo_relu_rmse:.4f}, R²={dngo_relu_r2:.4f} (time: {dngo_relu_time:.2f}s)")
        fold_results['dngo_relu_pred'] = dngo_relu_pred
        fold_results['dngo_relu_std'] = dngo_relu_std

        # 4. Pretrain tanh
        t0 = time.time()
        pretrain_tanh_pred, pretrain_tanh_std, pretrain_tanh_model = run_pretrain(data, device, seed, activation='tanh')
        pretrain_tanh_time = time.time() - t0
        pretrain_tanh_rmse, pretrain_tanh_r2, _ = calc_metrics(y_test_lf, pretrain_tanh_pred[test_idx])
        print(f"  Pretrain-tanh (vs LF): RMSE={pretrain_tanh_rmse:.4f}, R²={pretrain_tanh_r2:.4f} (time: {pretrain_tanh_time:.2f}s)")
        fold_results['pretrain_tanh_pred'] = pretrain_tanh_pred
        fold_results['pretrain_tanh_std'] = pretrain_tanh_std

        # 5. DNGO-tanh
        t0 = time.time()
        dngo_tanh_pred, dngo_tanh_std, _ = run_dngo(data, device, seed, pretrain_tanh_model, activation='tanh')
        dngo_tanh_time = time.time() - t0
        dngo_tanh_rmse, dngo_tanh_r2, _ = calc_metrics(y_test_hf, dngo_tanh_pred[test_idx])
        print(f"  DNGO-tanh: RMSE={dngo_tanh_rmse:.4f}, R²={dngo_tanh_r2:.4f} (time: {dngo_tanh_time:.2f}s)")
        fold_results['dngo_tanh_pred'] = dngo_tanh_pred
        fold_results['dngo_tanh_std'] = dngo_tanh_std

        # Create 5-panel visualization
        viz_path = create_5panel_visualization(fold_idx, seed, data, fold_results, param_space, output_dir)
        print(f"  Visualization saved: {viz_path.name}")

        # Store results
        results_list.append({
            'fold': fold_idx,
            'seed': seed,
            'mfgp_rmse': mfgp_rmse,
            'mfgp_r2': mfgp_r2,
            'dngo_relu_rmse': dngo_relu_rmse,
            'dngo_relu_r2': dngo_relu_r2,
            'dngo_tanh_rmse': dngo_tanh_rmse,
            'dngo_tanh_r2': dngo_tanh_r2,
            'pretrain_relu_rmse': pretrain_relu_rmse,
            'pretrain_relu_r2': pretrain_relu_r2,
            'pretrain_tanh_rmse': pretrain_tanh_rmse,
            'pretrain_tanh_r2': pretrain_tanh_r2,
        })

    total_time = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    df = pd.DataFrame(results_list)

    print(f"\n{'Model':<20} {'Avg RMSE':<12} {'Avg R²':<12} {'Compare'}")
    print('-' * 60)

    if df['mfgp_r2'].notna().any():
        print(f"{'MFGP':<20} {df['mfgp_rmse'].mean():.4f}       {df['mfgp_r2'].mean():.4f}       vs HF")
    print(f"{'DNGO-ReLU':<20} {df['dngo_relu_rmse'].mean():.4f}       {df['dngo_relu_r2'].mean():.4f}       vs HF")
    print(f"{'DNGO-tanh':<20} {df['dngo_tanh_rmse'].mean():.4f}       {df['dngo_tanh_r2'].mean():.4f}       vs HF")
    print(f"{'Pretrain-ReLU':<20} {df['pretrain_relu_rmse'].mean():.4f}       {df['pretrain_relu_r2'].mean():.4f}       vs LF")
    print(f"{'Pretrain-tanh':<20} {df['pretrain_tanh_rmse'].mean():.4f}       {df['pretrain_tanh_r2'].mean():.4f}       vs LF")

    print(f"\n{'='*60}")
    print("tanh vs ReLU Comparison (HF prediction)")
    print('='*60)
    improvement = df['dngo_tanh_r2'].mean() - df['dngo_relu_r2'].mean()
    print(f"DNGO-tanh Avg R²: {df['dngo_tanh_r2'].mean():.4f}")
    print(f"DNGO-ReLU Avg R²: {df['dngo_relu_r2'].mean():.4f}")
    print(f"Improvement: {'+' if improvement > 0 else ''}{improvement:.4f}")

    tanh_wins = (df['dngo_tanh_r2'] > df['dngo_relu_r2']).sum()
    print(f"Folds where tanh > ReLU: {tanh_wins}/{len(df)}")

    print(f"\nTotal time: {total_time/60:.2f} minutes")

    # Save results
    df.to_csv(output_dir / 'results_summary.csv', index=False)
    print(f"Results saved to: {output_dir / 'results_summary.csv'}")


if __name__ == '__main__':
    main()
