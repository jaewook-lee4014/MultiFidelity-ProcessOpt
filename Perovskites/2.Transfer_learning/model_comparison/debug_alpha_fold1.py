"""
Fold 1에서 alpha=0.0001로 True-Intermediate 학습하고
학습 스텝별 예측 결과 시각화
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

    # Find test indices (not in hifi)
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

def train_with_snapshots(data, device, params, snapshot_epochs):
    """학습 중 특정 에포크에서 스냅샷 저장"""
    set_seeds(SEED)

    model = TrueIntermediateDNGO(
        input_dim=data['X_low'].shape[1],
        lf_hidden_dims=[params['lf_width']] * params['lf_layers'],
        hf_hidden_dims=[params['hf_width']] * params['hf_layers'],
        device=device,
        alpha=params['alpha'],
        lf_dropout=params['lf_dropout'],
        hf_dropout=params['hf_dropout']
    )

    # Manual training with snapshots
    import torch.nn as nn
    import torch.optim as optim

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

    snapshots = {}
    history = {'lf_loss': [], 'hf_loss': [], 'total_loss': []}

    model.model.train()
    for epoch in range(params['epochs']):
        optimizer.zero_grad()

        # LF prediction
        y_lf_pred = model.model.predict_lf(X_lf_t)
        lf_loss = loss_fn(y_lf_pred, y_lf_t)

        # HF prediction
        y_lf_for_hf = model.model.predict_lf(X_hf_t)
        y_hf_pred = model.model.hf_network(X_hf_t, y_lf_for_hf)
        hf_loss = loss_fn(y_hf_pred, y_hf_t)

        # Weighted loss
        alpha = params['alpha']
        total_loss = (1 - alpha) * lf_loss + alpha * hf_loss

        total_loss.backward()
        optimizer.step()

        history['lf_loss'].append(lf_loss.item())
        history['hf_loss'].append(hf_loss.item())
        history['total_loss'].append(total_loss.item())

        # Save snapshot
        if epoch + 1 in snapshot_epochs:
            model.model.eval()
            with torch.no_grad():
                # LF predictions
                lf_pred_all = model.model.predict_lf(X_all_t).cpu().numpy().flatten()

                # HF predictions
                y_lf_for_all = model.model.predict_lf(X_all_t)
                hf_pred_all = model.model.hf_network(X_all_t, y_lf_for_all).cpu().numpy().flatten()

            snapshots[epoch + 1] = {
                'lf_pred': lf_pred_all.copy(),
                'hf_pred': hf_pred_all.copy(),
                'lf_loss': lf_loss.item(),
                'hf_loss': hf_loss.item()
            }
            model.model.train()
            print(f"  Epoch {epoch+1}: LF loss={lf_loss.item():.6f}, HF loss={hf_loss.item():.6f}")

    return snapshots, history

def calc_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return rmse, r2

def visualize_snapshots(data, snapshots, history, params, output_dir):
    """학습 스텝별 예측 결과 시각화"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs_list = sorted(snapshots.keys())
    n_epochs = len(epochs_list)

    # Figure 1: 예측 결과 변화 (composition 순서)
    fig, axes = plt.subplots(n_epochs, 2, figsize=(16, 4 * n_epochs))

    x_axis = np.arange(len(data['X_all']))

    for i, epoch in enumerate(epochs_list):
        snap = snapshots[epoch]

        # LF prediction
        ax_lf = axes[i, 0]
        ax_lf.plot(x_axis, data['y_all_lf'], 'k-', alpha=0.5, label='True LF', linewidth=1)
        ax_lf.plot(x_axis, snap['lf_pred'], 'b-', alpha=0.7, label='Pred LF', linewidth=1)
        ax_lf.scatter(data['lofi_idx'], data['y_low'], c='blue', s=20, zorder=5, label='Train LF')

        rmse_lf, r2_lf = calc_metrics(data['y_all_lf'], snap['lf_pred'])
        ax_lf.set_title(f'Epoch {epoch}: LF Prediction (R²={r2_lf:.4f}, Loss={snap["lf_loss"]:.6f})')
        ax_lf.set_ylabel('Bandgap (eV)')
        ax_lf.legend(loc='upper right')
        ax_lf.grid(True, alpha=0.3)

        # HF prediction
        ax_hf = axes[i, 1]
        ax_hf.plot(x_axis, data['y_all_hf'], 'k-', alpha=0.5, label='True HF', linewidth=1)
        ax_hf.plot(x_axis, snap['hf_pred'], 'r-', alpha=0.7, label='Pred HF', linewidth=1)
        ax_hf.scatter(data['hifi_idx'], data['y_high'], c='red', s=40, zorder=5, label='Train HF')
        ax_hf.scatter(data['test_idx'], data['y_all_hf'][data['test_idx']], c='green', s=40, marker='x', zorder=5, label='Test HF')

        rmse_hf, r2_hf = calc_metrics(data['y_all_hf'], snap['hf_pred'])
        test_rmse, test_r2 = calc_metrics(data['y_all_hf'][data['test_idx']], snap['hf_pred'][data['test_idx']])
        ax_hf.set_title(f'Epoch {epoch}: HF Prediction (All R²={r2_hf:.4f}, Test R²={test_r2:.4f})')
        ax_hf.set_ylabel('Bandgap (eV)')
        ax_hf.legend(loc='upper right')
        ax_hf.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Composition Index')
    axes[-1, 1].set_xlabel('Composition Index')

    plt.suptitle(f'True-Intermediate Training Progress (alpha={params["alpha"]}, lr={params["lr"]:.4f})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_snapshots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'training_snapshots.png'}")

    # Figure 2: Loss curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs_range = range(1, len(history['lf_loss']) + 1)

    axes[0].plot(epochs_range, history['lf_loss'], 'b-', label='LF Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('LF Loss')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history['hf_loss'], 'r-', label='HF Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE Loss')
    axes[1].set_title('HF Loss')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs_range, history['total_loss'], 'g-', label='Total Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('MSE Loss')
    axes[2].set_title(f'Total Loss (alpha={params["alpha"]})')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'Training Loss Curves', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'loss_curves.png'}")

    # Figure 3: Test R² 변화
    test_r2_list = []
    for epoch in epochs_list:
        snap = snapshots[epoch]
        _, test_r2 = calc_metrics(data['y_all_hf'][data['test_idx']], snap['hf_pred'][data['test_idx']])
        test_r2_list.append(test_r2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_list, test_r2_list, 'ro-', markersize=8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test R²')
    ax.set_title(f'Test R² over Training (alpha={params["alpha"]})')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    for i, (ep, r2) in enumerate(zip(epochs_list, test_r2_list)):
        ax.annotate(f'{r2:.3f}', (ep, r2), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.tight_layout()
    plt.savefig(output_dir / 'test_r2_progress.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'test_r2_progress.png'}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load data
    lookup, all_combinations, param_space = load_base_data()
    print(f"Data loaded: {len(all_combinations)} samples")

    # Prepare Fold 1 data
    data = generate_data(lookup, all_combinations, n_lofi=N_LOFI, n_hifi=N_HIFI, seed=SEED)
    print(f"Fold 1: {len(data['X_low'])} LF, {len(data['X_high'])} HF")

    # Parameters from BO (but with alpha=0.99)
    params = {
        'lf_width': 64,
        'lf_layers': 2,
        'lf_dropout': 0.07440368256156257,
        'hf_width': 16,
        'hf_layers': 3,
        'hf_dropout': 0.17435383544380412,
        'alpha': 0.99,  # Changed to 0.99
        'lr': 0.0031616993585377342,
        'epochs': 426,
        'hf_lr_ratio': 0.30881879455521566,
        'hf_weight_decay': 0.01159130898812208
    }

    print(f"\nParameters:")
    print(f"  alpha: {params['alpha']} (original: 0.988)")
    print(f"  lr: {params['lr']:.6f}")
    print(f"  hf_lr_ratio: {params['hf_lr_ratio']:.4f}")
    print(f"  epochs: {params['epochs']}")

    # Snapshot epochs
    snapshot_epochs = [1, 10, 50, 100, 200, 300, 426]
    print(f"\nSnapshot epochs: {snapshot_epochs}")

    # Train with snapshots
    print("\nTraining...")
    snapshots, history = train_with_snapshots(data, device, params, snapshot_epochs)

    # Visualize
    output_dir = Path(__file__).parent / 'visualizations' / 'debug_alpha_0001'
    visualize_snapshots(data, snapshots, history, params, output_dir)

    # Final metrics
    final_snap = snapshots[426]
    test_rmse, test_r2 = calc_metrics(data['y_all_hf'][data['test_idx']], final_snap['hf_pred'][data['test_idx']])
    print(f"\nFinal Test Results:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R²: {test_r2:.4f}")

if __name__ == '__main__':
    main()
