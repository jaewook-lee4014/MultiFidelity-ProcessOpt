"""
HF Learning Rate sweep (alpha=0.75 고정)
hf_lr_ratio: 0.3 (현재), 0.5, 1.0, 2.0, 3.0
"""
import sys
import numpy as np
import torch
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'Pure_TL_BO'))

from dngo_intermediate_v2 import TrueIntermediateDNGO, BayesianLinearRegression

# Configuration
SEED = 42
N_LOFI = 72
N_HIFI = 9
ALPHA = 0.99
HF_LR_RATIOS = [0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0]

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
    for c in all_combinations:
        X_all.append(c['labels'])
        y_all_hf.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_hse06']))
        y_all_lf.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_gga']))

    all_idx = set(range(n_total))
    hifi_set = set(hifi_idx)
    remaining = list(all_idx - hifi_set)
    test_idx = np.array(remaining[:10])

    return {
        'X_low': np.array(X_low, dtype=np.float32),
        'y_low': np.array(y_low, dtype=np.float32),
        'X_high': np.array(X_high, dtype=np.float32),
        'y_high': np.array(y_high, dtype=np.float32),
        'X_all': np.array(X_all, dtype=np.float32),
        'y_all_hf': np.array(y_all_hf, dtype=np.float32),
        'y_all_lf': np.array(y_all_lf, dtype=np.float32),
        'lofi_idx': lofi_idx,
        'hifi_idx': hifi_idx,
        'test_idx': test_idx
    }

def train_model(data, device, params):
    set_seeds(SEED)

    import torch.nn as nn
    import torch.optim as optim

    model = TrueIntermediateDNGO(
        input_dim=data['X_low'].shape[1],
        lf_hidden_dims=[params['lf_width']] * params['lf_layers'],
        hf_hidden_dims=[params['hf_width']] * params['hf_layers'],
        device=device,
        alpha=params['alpha'],
        lf_dropout=params['lf_dropout'],
        hf_dropout=params['hf_dropout']
    )

    X_lf_t = torch.tensor(data['X_low'], dtype=torch.float32).to(device)
    y_lf_t = torch.tensor(data['y_low'], dtype=torch.float32).view(-1, 1).to(device)
    X_hf_t = torch.tensor(data['X_high'], dtype=torch.float32).to(device)
    y_hf_t = torch.tensor(data['y_high'], dtype=torch.float32).view(-1, 1).to(device)
    X_all_t = torch.tensor(data['X_all'], dtype=torch.float32).to(device)

    loss_fn = nn.MSELoss()

    lr = params['lr']
    hf_lr_ratio = params['hf_lr_ratio']
    hf_weight_decay = params['hf_weight_decay']

    optimizer = optim.Adam([
        {'params': model.model.lf_network.parameters(), 'lr': lr, 'weight_decay': 1e-4},
        {'params': model.model.hf_network.parameters(), 'lr': lr * hf_lr_ratio, 'weight_decay': hf_weight_decay}
    ])

    history = {'lf_loss': [], 'hf_loss': [], 'total_loss': [], 'test_r2': []}
    snapshot_epochs = [1, 10, 50, 100, 200, 300, params['epochs']]

    model.model.train()
    for epoch in range(params['epochs']):
        optimizer.zero_grad()

        y_lf_pred = model.model.predict_lf(X_lf_t)
        lf_loss = loss_fn(y_lf_pred, y_lf_t)

        y_lf_for_hf = model.model.predict_lf(X_hf_t)
        y_hf_pred = model.model.hf_network(X_hf_t, y_lf_for_hf)
        hf_loss = loss_fn(y_hf_pred, y_hf_t)

        alpha = params['alpha']
        total_loss = (1 - alpha) * lf_loss + alpha * hf_loss

        total_loss.backward()
        optimizer.step()

        history['lf_loss'].append(lf_loss.item())
        history['hf_loss'].append(hf_loss.item())
        history['total_loss'].append(total_loss.item())

        if epoch + 1 in snapshot_epochs:
            model.model.eval()
            with torch.no_grad():
                y_lf_for_all = model.model.predict_lf(X_all_t)
                hf_pred_all = model.model.hf_network(X_all_t, y_lf_for_all).cpu().numpy().flatten()

            test_pred = hf_pred_all[data['test_idx']]
            test_true = data['y_all_hf'][data['test_idx']]
            ss_res = np.sum((test_true - test_pred) ** 2)
            ss_tot = np.sum((test_true - np.mean(test_true)) ** 2)
            test_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            history['test_r2'].append((epoch + 1, test_r2))
            model.model.train()

    model.model.eval()
    with torch.no_grad():
        y_lf_for_all = model.model.predict_lf(X_all_t)
        hf_pred_all = model.model.hf_network(X_all_t, y_lf_for_all).cpu().numpy().flatten()
        lf_pred_all = model.model.predict_lf(X_all_t).cpu().numpy().flatten()

    return history, hf_pred_all, lf_pred_all

def calc_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return rmse, r2

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    lookup, all_combinations, param_space = load_base_data()
    print(f"Data loaded: {len(all_combinations)} samples")

    data = generate_data(lookup, all_combinations, n_lofi=N_LOFI, n_hifi=N_HIFI, seed=SEED)
    print(f"Fold 1: {len(data['X_low'])} LF, {len(data['X_high'])} HF")

    base_params = {
        'lf_width': 64,
        'lf_layers': 2,
        'lf_dropout': 0.07440368256156257,
        'hf_width': 16,
        'hf_layers': 3,
        'hf_dropout': 0.17435383544380412,
        'alpha': ALPHA,
        'lr': 0.0031616993585377342,
        'epochs': 426,
        'hf_weight_decay': 0.01159130898812208
    }

    print(f"\nFixed Parameters:")
    print(f"  alpha: {ALPHA}")
    print(f"  base lr: {base_params['lr']:.6f}")
    print(f"\nHF LR Ratio sweep: {HF_LR_RATIOS}")
    print(f"Actual HF LRs: {[base_params['lr'] * r for r in HF_LR_RATIOS]}")

    results = {}

    for hf_lr_ratio in HF_LR_RATIOS:
        actual_hf_lr = base_params['lr'] * hf_lr_ratio
        print(f"\n{'='*60}")
        print(f"Training with hf_lr_ratio = {hf_lr_ratio} (HF LR = {actual_hf_lr:.6f})")
        print(f"{'='*60}")

        params = base_params.copy()
        params['hf_lr_ratio'] = hf_lr_ratio

        history, hf_pred, lf_pred = train_model(data, device, params)

        test_rmse, test_r2 = calc_metrics(data['y_all_hf'][data['test_idx']], hf_pred[data['test_idx']])
        all_rmse, all_r2 = calc_metrics(data['y_all_hf'], hf_pred)
        lf_rmse, lf_r2 = calc_metrics(data['y_all_lf'], lf_pred)

        results[hf_lr_ratio] = {
            'history': history,
            'hf_pred': hf_pred,
            'lf_pred': lf_pred,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'all_r2': all_r2,
            'lf_r2': lf_r2,
            'actual_hf_lr': actual_hf_lr
        }

        print(f"  Final LF Loss: {history['lf_loss'][-1]:.4f}")
        print(f"  Final HF Loss: {history['hf_loss'][-1]:.4f}")
        print(f"  LF R² (all): {lf_r2:.4f}")
        print(f"  HF R² (test): {test_r2:.4f}")
        print(f"  HF R² (all): {all_r2:.4f}")

    # Visualization
    output_dir = Path(__file__).parent / 'visualizations' / 'hf_lr_sweep'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Loss curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(HF_LR_RATIOS)))

    for i, hf_lr_ratio in enumerate(HF_LR_RATIOS):
        history = results[hf_lr_ratio]['history']
        epochs = range(1, len(history['lf_loss']) + 1)
        actual_lr = results[hf_lr_ratio]['actual_hf_lr']

        axes[0].plot(epochs, history['lf_loss'], color=colors[i], label=f'HF_LR={actual_lr:.4f}')
        axes[1].plot(epochs, history['hf_loss'], color=colors[i], label=f'HF_LR={actual_lr:.4f}')
        axes[2].plot(epochs, history['total_loss'], color=colors[i], label=f'HF_LR={actual_lr:.4f}')

    for ax, title in zip(axes, ['LF Loss', 'HF Loss', 'Total Loss']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')
        ax.set_title(title)
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Loss Curves (alpha={ALPHA})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves_hf_lr.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_dir / 'loss_curves_hf_lr.png'}")

    # Figure 2: Test R² vs HF LR
    fig, ax = plt.subplots(figsize=(8, 5))
    actual_lrs = [results[r]['actual_hf_lr'] for r in HF_LR_RATIOS]
    test_r2_values = [results[r]['test_r2'] for r in HF_LR_RATIOS]

    ax.plot(actual_lrs, test_r2_values, 'ro-', markersize=10)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('HF Learning Rate')
    ax.set_ylabel('Test R²')
    ax.set_title(f'Test R² vs HF Learning Rate (alpha={ALPHA})')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    for lr, r2 in zip(actual_lrs, test_r2_values):
        ax.annotate(f'{r2:.3f}', (lr, r2), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.tight_layout()
    plt.savefig(output_dir / 'test_r2_vs_hf_lr.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'test_r2_vs_hf_lr.png'}")

    # Figure 3: Predictions
    fig, axes = plt.subplots(len(HF_LR_RATIOS), 2, figsize=(14, 4 * len(HF_LR_RATIOS)))
    x_axis = np.arange(len(data['X_all']))

    for i, hf_lr_ratio in enumerate(HF_LR_RATIOS):
        res = results[hf_lr_ratio]

        axes[i, 0].plot(x_axis, data['y_all_lf'], 'k-', alpha=0.5, linewidth=1, label='True LF')
        axes[i, 0].plot(x_axis, res['lf_pred'], 'b-', alpha=0.7, linewidth=1, label='Pred LF')
        axes[i, 0].scatter(data['lofi_idx'], data['y_low'], c='blue', s=15, zorder=5)
        axes[i, 0].set_title(f'HF_LR={res["actual_hf_lr"]:.4f}: LF (R²={res["lf_r2"]:.4f})')
        axes[i, 0].set_ylabel('Bandgap (eV)')
        axes[i, 0].legend(loc='upper right', fontsize=8)
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(x_axis, data['y_all_hf'], 'k-', alpha=0.5, linewidth=1, label='True HF')
        axes[i, 1].plot(x_axis, res['hf_pred'], 'r-', alpha=0.7, linewidth=1, label='Pred HF')
        axes[i, 1].scatter(data['hifi_idx'], data['y_high'], c='red', s=30, zorder=5)
        axes[i, 1].scatter(data['test_idx'], data['y_all_hf'][data['test_idx']], c='green', s=30, marker='x', zorder=5)
        axes[i, 1].set_title(f'HF_LR={res["actual_hf_lr"]:.4f}: HF (Test R²={res["test_r2"]:.4f})')
        axes[i, 1].set_ylabel('Bandgap (eV)')
        axes[i, 1].legend(loc='upper right', fontsize=8)
        axes[i, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Composition Index')
    axes[-1, 1].set_xlabel('Composition Index')

    plt.suptitle(f'HF LR Sweep Predictions (alpha={ALPHA})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'predictions_by_hf_lr.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'predictions_by_hf_lr.png'}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY (alpha={ALPHA})")
    print(f"{'='*60}")
    print(f"{'HF LR Ratio':<12} {'HF LR':<12} {'LF R²':<10} {'HF Test R²':<12} {'HF Loss':<10}")
    print("-" * 56)
    for hf_lr_ratio in HF_LR_RATIOS:
        res = results[hf_lr_ratio]
        print(f"{hf_lr_ratio:<12} {res['actual_hf_lr']:<12.6f} {res['lf_r2']:<10.4f} {res['test_r2']:<12.4f} {res['history']['hf_loss'][-1]:<10.4f}")

    best_ratio = max(HF_LR_RATIOS, key=lambda r: results[r]['test_r2'])
    print(f"\nBest HF LR Ratio: {best_ratio} (Test R²={results[best_ratio]['test_r2']:.4f})")

if __name__ == '__main__':
    main()
