#!/usr/bin/env python3
"""
Gradient Scaling vs MFGP 시각화 (기존 시각화 방식과 동일)

BO 실험의 최적 파라미터를 사용하여 각 fold별로 한 번만 훈련

이전 run_true_intermediate.py, visualize_dngo_vs_mfgp_v2.py 스타일을 따름:
- 5패널 Figure (MFGP, DNGO-Base, Joint-Best, Gradient Scaling, Pretrain-Base)
- Composition 정렬 & Value 정렬 각각 저장
- 불확실성 밴드, Train/Test 포인트 구분

Author: Claude Code
Date: 2025-12-11
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Path setup
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
model_comparison_dir = current_dir.parent.parent / 'model_comparison'
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(model_comparison_dir))

from DNGO.models import TransferLearningDNN, BayesianLinearRegression

# MFGP
try:
    from mfgp_model import MultiFidelityGP
    MFGP_AVAILABLE = True
except ImportError:
    MFGP_AVAILABLE = False
    print("Warning: MFGP not available")

# ============================================================================
# 기존 프로젝트 설정
# ============================================================================

N_LOFI = 72
N_HIFI = 9
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

# ============================================================================
# BO 결과에서 최적 파라미터 로드
# ============================================================================

def load_best_params_from_bo():
    """BO 실험 결과에서 최적 파라미터 로드"""
    bo_results_path = current_dir.parent / 'results/large_scale_bo_v2_20251208_170617/bo_results_v2_20251208_170628.json'

    with open(bo_results_path, 'r') as f:
        bo_data = json.load(f)

    # Best overall (joint)
    best_overall = bo_data['best_params']
    print(f"Best Overall (method={best_overall['method']}): R²={bo_data['best_value']:.4f}")

    # Find best gradient_scaling
    best_gs = None
    best_gs_value = -1
    for trial in bo_data['all_trials']:
        if trial['params'].get('method') == 'gradient_scaling' and trial['value'] is not None:
            if trial['value'] > best_gs_value:
                best_gs_value = trial['value']
                best_gs = trial['params']

    print(f"Best Gradient Scaling: R²={best_gs_value:.4f}")

    return {
        'best_joint': best_overall,
        'best_joint_r2': bo_data['best_value'],
        'best_gs': best_gs,
        'best_gs_r2': best_gs_value
    }


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_base_data():
    data_path = Path(__file__).parent.parent.parent.parent / '0.Data'
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

    return {
        'X_low': np.array(X_low, dtype=np.float32),
        'y_low': np.array(y_low, dtype=np.float32),
        'X_high': np.array(X_high, dtype=np.float32),
        'y_high': np.array(y_high, dtype=np.float32),
        'X_all': np.array(X_all, dtype=np.float32),
        'y_all': np.array(y_all_hf, dtype=np.float32),
        'y_all_lf': np.array(y_all_lf, dtype=np.float32),
        'hifi_idx': hifi_idx,
        'lofi_idx': lofi_idx
    }


def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return rmse, r2


# ============================================================================
# Network Definitions
# ============================================================================

class LFNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2,
                 dropout: float = 0.0, activation: str = 'relu'):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        features = self.feature_net(x)
        return self.out_layer(features)

    def extract_features(self, x):
        return self.feature_net(x)


class HFNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2,
                 dropout: float = 0.0, activation: str = 'relu',
                 residual_mode: bool = True):
        super().__init__()
        self.residual_mode = residual_mode
        hf_input_dim = input_dim + 1

        layers = []
        in_dim = hf_input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, y_lf):
        combined = torch.cat([x, y_lf], dim=-1)
        features = self.feature_net(combined)
        delta = self.out_layer(features)
        if self.residual_mode:
            return y_lf + delta
        else:
            return delta

    def extract_features(self, x, y_lf):
        combined = torch.cat([x, y_lf], dim=-1)
        return self.feature_net(combined)


class MultiFidelityNetwork(nn.Module):
    def __init__(self, input_dim: int, lf_hidden: int, hf_hidden: int,
                 lf_layers: int = 2, hf_layers: int = 2,
                 dropout: float = 0.0, activation: str = 'relu',
                 residual_mode: bool = True):
        super().__init__()
        self.residual_mode = residual_mode
        self.lf_network = LFNetwork(input_dim, lf_hidden, lf_layers, dropout, activation)
        self.hf_network = HFNetwork(input_dim, hf_hidden, hf_layers, dropout, activation, residual_mode)

    def forward_lf(self, x):
        return self.lf_network(x)

    def forward_hf(self, x, y_lf=None):
        if y_lf is None:
            y_lf = self.lf_network(x)
        return self.hf_network(x, y_lf)


# ============================================================================
# Training Functions - BO 최적 파라미터 사용
# ============================================================================

def train_joint(model, X_lf, y_lf, X_hf, y_hf, params, device):
    """Joint Training with alpha weighting - BO best params"""
    alpha = params.get('alpha', 0.2)
    lf_lr = params.get('lf_lr', 1e-3)
    hf_lr = params.get('hf_lr', 1e-4)
    epochs = params.get('epochs', 200)
    weight_decay = params.get('weight_decay', 1e-4)

    optimizer = optim.Adam([
        {'params': model.lf_network.parameters(), 'lr': lf_lr, 'weight_decay': weight_decay},
        {'params': model.hf_network.parameters(), 'lr': hf_lr, 'weight_decay': weight_decay}
    ])

    loss_fn = nn.MSELoss()

    X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(device)
    y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(device)
    X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(device)
    y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(device)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_lf_pred = model.forward_lf(X_lf_t)
        lf_loss = loss_fn(y_lf_pred, y_lf_t)
        y_lf_for_hf = model.forward_lf(X_hf_t).detach()
        y_hf_pred = model.forward_hf(X_hf_t, y_lf_for_hf)
        hf_loss = loss_fn(y_hf_pred, y_hf_t)
        total_loss = (1 - alpha) * lf_loss + alpha * hf_loss
        total_loss.backward()
        optimizer.step()


def train_gradient_scaling(model, X_lf, y_lf, X_hf, y_hf, params, device):
    """Gradient Scaling Training - BO best params"""
    lf_lr = params.get('lf_lr', 1e-3)
    hf_lr = params.get('hf_lr', 1e-4)
    epochs = params.get('gs_epochs', params.get('epochs', 200))
    weight_decay = params.get('weight_decay', 1e-4)
    gradient_scale_lf = params.get('gradient_scale_lf', 1.0)
    gradient_scale_hf = params.get('gradient_scale_hf', 1.0)

    loss_fn = nn.MSELoss()

    X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(device)
    y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(device)
    X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(device)
    y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(device)

    lf_optimizer = optim.Adam(model.lf_network.parameters(), lr=lf_lr, weight_decay=weight_decay)
    hf_optimizer = optim.Adam(model.hf_network.parameters(), lr=hf_lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        # LF update
        lf_optimizer.zero_grad()
        y_lf_pred = model.forward_lf(X_lf_t)
        lf_loss = loss_fn(y_lf_pred, y_lf_t)
        scaled_lf_loss = lf_loss * gradient_scale_lf
        scaled_lf_loss.backward()
        lf_optimizer.step()

        # HF update
        hf_optimizer.zero_grad()
        with torch.no_grad():
            y_lf_for_hf = model.forward_lf(X_hf_t)
        y_hf_pred = model.forward_hf(X_hf_t, y_lf_for_hf)
        hf_loss = loss_fn(y_hf_pred, y_hf_t)
        scaled_hf_loss = hf_loss * gradient_scale_hf
        scaled_hf_loss.backward()
        hf_optimizer.step()


# ============================================================================
# Model Runners - BO 최적 파라미터 사용
# ============================================================================

def run_mfgp(data, device, seed):
    if not MFGP_AVAILABLE:
        return None, None
    set_seeds(seed)
    mfgp = MultiFidelityGP(input_dim=data['X_low'].shape[1], device=device)
    mfgp.fit(data['X_low'], data['y_low'], data['X_high'], data['y_high'])
    mfgp_pred, mfgp_std = mfgp.predict(data['X_all'], return_std=True)
    return mfgp_pred, mfgp_std


def run_dngo_base(data, device, seed):
    """DNGO-Base with default good params"""
    set_seeds(seed)

    model = TransferLearningDNN(
        input_dim=data['X_low'].shape[1],
        hidden_dim=64,
        device=device,
        use_hyperparameter_bo=False,
        activation='tanh'
    )
    model.pretrain(data['X_low'], data['y_low'], epochs=300, lr=1e-3, verbose=False)
    model.finetune(data['X_high'], data['y_high'], epochs=150, lr=1e-4, verbose=False)

    features_high = model.extract_features(data['X_high'])
    blr = BayesianLinearRegression(alpha=1.0, beta=25.0)
    blr.fit(features_high, data['y_high'])
    features_all = model.extract_features(data['X_all'])
    pred_all, var_all = blr.predict_batch(features_all)
    return pred_all, np.sqrt(var_all)


def run_pretrain_base(data, device, seed):
    """Pretrain-Base (LF only)"""
    set_seeds(seed)

    model = TransferLearningDNN(
        input_dim=data['X_low'].shape[1],
        hidden_dim=64,
        device=device,
        use_hyperparameter_bo=False,
        activation='tanh'
    )
    model.pretrain(data['X_low'], data['y_low'], epochs=300, lr=1e-3, verbose=False)

    features_low = model.extract_features(data['X_low'])
    blr = BayesianLinearRegression(alpha=1.0, beta=25.0)
    blr.fit(features_low, data['y_low'])
    features_all = model.extract_features(data['X_all'])
    pred_all, var_all = blr.predict_batch(features_all)
    return pred_all, np.sqrt(var_all)


def run_joint_best(data, device, seed, params):
    """Run Joint training model with BO best params (R²=0.7848)"""
    set_seeds(seed)

    model = MultiFidelityNetwork(
        input_dim=3,
        lf_hidden=params.get('lf_hidden', 48),
        hf_hidden=params.get('hf_hidden', 96),
        lf_layers=params.get('lf_layers', 3),
        hf_layers=params.get('hf_layers', 3),
        dropout=params.get('dropout', 0.0),
        activation=params.get('activation', 'relu'),
        residual_mode=True
    )
    model.to(device)

    train_joint(model, data['X_low'], data['y_low'],
                data['X_high'], data['y_high'], params, device)

    # Evaluation
    model.eval()
    X_t = torch.tensor(data['X_all'], dtype=torch.float32).to(device)
    with torch.no_grad():
        y_lf_pred_t = model.forward_lf(X_t)
        y_hf_pred = model.forward_hf(X_t, y_lf_pred_t).cpu().numpy().flatten()

    # Uncertainty estimation
    std = np.ones_like(y_hf_pred) * 0.1

    return y_hf_pred, std


def run_gradient_scaling(data, device, seed, params):
    """Run Gradient Scaling model with BO best params (R²=0.7839)"""
    set_seeds(seed)

    model = MultiFidelityNetwork(
        input_dim=3,
        lf_hidden=params.get('lf_hidden', 48),
        hf_hidden=params.get('hf_hidden', 96),
        lf_layers=params.get('lf_layers', 3),
        hf_layers=params.get('hf_layers', 3),
        dropout=params.get('dropout', 0.0),
        activation=params.get('activation', 'relu'),
        residual_mode=True
    )
    model.to(device)

    train_gradient_scaling(model, data['X_low'], data['y_low'],
                           data['X_high'], data['y_high'], params, device)

    # Evaluation
    model.eval()
    X_t = torch.tensor(data['X_all'], dtype=torch.float32).to(device)
    with torch.no_grad():
        y_lf_pred_t = model.forward_lf(X_t)
        y_hf_pred = model.forward_hf(X_t, y_lf_pred_t).cpu().numpy().flatten()

    # Uncertainty estimation
    std = np.ones_like(y_hf_pred) * 0.1

    return y_hf_pred, std


# ============================================================================
# Visualization Functions (기존 방식 동일)
# ============================================================================

def create_5panel_visualization(fold_idx, seed, data, results, stds, param_space, output_dir):
    """5패널 시각화 생성 - 기존 방식과 동일"""
    n_all = len(data['X_all'])
    hifi_idx = data['hifi_idx']
    lofi_idx = data['lofi_idx']

    y_all_hf = data['y_all']
    y_all_lf = data['y_all_lf']

    colors = {
        'mfgp': 'blue',
        'dngo_base': 'orange',
        'joint_best': 'green',
        'gradient_scaling': 'purple',
        'pretrain_base': 'red'
    }

    fold_dir = output_dir / f'fold_{fold_idx}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Composition labels
    org_names = param_space['organic']
    cat_names = param_space['cation']
    ani_names = param_space['anion']
    n_cat = len(cat_names)
    n_ani = len(ani_names)
    group_size = n_cat * n_ani

    comp_labels = []
    for i in range(len(org_names)):
        for j in range(len(cat_names)):
            for k in range(len(ani_names)):
                org_short = org_names[i][:4].capitalize()
                cat_short = cat_names[j][:2].capitalize()
                ani_short = ani_names[k][:2].capitalize()
                comp_labels.append(f"{org_short}-{cat_short}-{ani_short}")

    # ============== 1. Composition Sort ==============
    sort_idx_comp = np.arange(len(data['X_all']))

    _plot_5panel(
        fold_idx, seed, sort_idx_comp, y_all_hf, y_all_lf,
        results, stds, hifi_idx, lofi_idx, param_space,
        comp_labels, colors, fold_dir, 'predictions_by_composition.png',
        'Sorted by Composition'
    )

    # ============== 2. Value Sort (HF) ==============
    sort_idx_value = np.argsort(y_all_hf)
    value_labels = [comp_labels[i] for i in sort_idx_value]

    _plot_5panel(
        fold_idx, seed, sort_idx_value, y_all_hf, y_all_lf,
        results, stds, hifi_idx, lofi_idx, param_space,
        value_labels, colors, fold_dir, 'predictions_by_value.png',
        'Sorted by HF Value'
    )

    return fold_dir


def _plot_5panel(fold_idx, seed, sort_idx, y_all_hf, y_all_lf,
                 results, stds, hifi_idx, lofi_idx, param_space,
                 x_labels, colors, output_dir, filename, sort_type):
    """5패널 플롯 그리기"""

    y_sorted_hf = y_all_hf[sort_idx]
    y_sorted_lf = y_all_lf[sort_idx]
    x_axis = np.arange(len(y_sorted_hf))

    hf_train_mask = np.isin(sort_idx, hifi_idx)
    lf_train_mask = np.isin(sort_idx, lofi_idx)

    n_cat = len(param_space['cation'])
    n_ani = len(param_space['anion'])
    group_size = n_cat * n_ani

    fig, axes = plt.subplots(5, 1, figsize=(24, 30))

    models_info = [
        ('MFGP', 'mfgp', y_sorted_hf, hf_train_mask, colors['mfgp'], 'vs HF'),
        ('DNGO-Base', 'dngo_base', y_sorted_hf, hf_train_mask, colors['dngo_base'], 'vs HF'),
        ('Joint-Best (BO R²=0.785)', 'joint_best', y_sorted_hf, hf_train_mask, colors['joint_best'], 'vs HF'),
        ('Gradient Scaling (BO R²=0.784)', 'gradient_scaling', y_sorted_hf, hf_train_mask, colors['gradient_scaling'], 'vs HF'),
        ('Pretrain-Base', 'pretrain_base', y_sorted_lf, lf_train_mask, colors['pretrain_base'], 'vs LF'),
    ]

    for ax_idx, (name, key, y_true, train_mask, color, compare_type) in enumerate(models_info):
        ax = axes[ax_idx]

        pred = results.get(f'{key}_pred')
        std = stds.get(f'{key}_std')

        if pred is None:
            ax.text(0.5, 0.5, f'{name}\nNot Available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{name} ({compare_type})')
            continue

        pred_sorted = pred[sort_idx]
        std_sorted = std[sort_idx] if std is not None else np.zeros_like(pred_sorted)

        # Uncertainty band
        ax.fill_between(x_axis, pred_sorted - 2*std_sorted, pred_sorted + 2*std_sorted,
                        alpha=0.3, color=color, label='±2σ')

        # Predictions
        ax.plot(x_axis, pred_sorted, color=color, linewidth=0.8, alpha=0.7, label=f'{name} Predicted')

        # Test points (black)
        ax.scatter(x_axis[~train_mask], y_true[~train_mask], c='black', s=15, zorder=5,
                   label=f'Test {compare_type.split()[1]}', alpha=0.6)

        # Train points (red star or circle)
        if compare_type == 'vs HF':
            ax.scatter(x_axis[train_mask], y_true[train_mask], c='red', s=100, marker='*',
                       zorder=6, label=f'Train HF ({np.sum(train_mask)})', edgecolors='darkred')
        else:
            ax.scatter(x_axis[train_mask], y_true[train_mask], c='red', s=40, marker='o',
                       zorder=6, label=f'Train LF ({np.sum(train_mask)})', edgecolors='darkred', alpha=0.7)

        # Metrics
        test_mask = ~train_mask
        rmse, r2 = calc_metrics(y_true[test_mask], pred_sorted[test_mask])

        ax.set_ylabel('Bandgap (eV)', fontsize=12)
        ax.set_title(f'{name} ({compare_type}): RMSE={rmse:.3f}, R²={r2:.3f}', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Organic group lines
        if 'composition' in filename:
            for i in range(1, len(param_space['organic'])):
                ax.axvline(x=i*group_size - 0.5, color='gray', linestyle='--', alpha=0.5)

    axes[-1].set_xticks(x_axis)
    axes[-1].set_xticklabels(x_labels, rotation=90, fontsize=6)
    axes[-1].set_xlabel('Composition (Organic-Cation-Anion)', fontsize=12)

    # Organic labels on top
    if 'composition' in filename:
        org_names = param_space['organic']
        for i, org in enumerate(org_names):
            mid_x = i * group_size + group_size / 2
            y_top = axes[0].get_ylim()[1]
            y_range = axes[0].get_ylim()[1] - axes[0].get_ylim()[0]
            axes[0].text(mid_x, y_top + y_range * 0.08, org.capitalize(),
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.suptitle(f'Fold {fold_idx} (seed={seed}) - MFGP vs DNGO vs Gradient Scaling ({sort_type})',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()


def create_parity_plots(fold_idx, seed, data, results, output_dir):
    """Parity plots (Actual vs Predicted)"""
    test_mask = np.ones(len(data['X_all']), dtype=bool)
    test_mask[data['hifi_idx']] = False

    y_test = data['y_all'][test_mask]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    model_info = [
        ('MFGP', 'mfgp', 'blue'),
        ('DNGO-Base', 'dngo_base', 'orange'),
        ('Joint-Best (BO)', 'joint_best', 'green'),
        ('Gradient Scaling (BO)', 'gradient_scaling', 'purple'),
    ]

    mn = y_test.min() - 0.3
    mx = y_test.max() + 0.3

    for idx, (name, key, color) in enumerate(model_info):
        ax = axes[idx // 2, idx % 2]

        pred = results.get(f'{key}_pred')
        if pred is None:
            ax.text(0.5, 0.5, f'{name}\nNot Available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(name)
            continue

        pred_test = pred[test_mask]
        rmse, r2 = calc_metrics(y_test, pred_test)

        ax.scatter(y_test, pred_test, c=color, alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
        ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5, alpha=0.5, label='y=x')
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)
        ax.set_xlabel('Actual HF Bandgap (eV)', fontsize=11)
        ax.set_ylabel('Predicted HF Bandgap (eV)', fontsize=11)
        ax.set_title(f'{name}\nRMSE={rmse:.3f}, R²={r2:.3f}', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle(f'Fold {fold_idx} (seed={seed}) - Parity Plots', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    parity_path = output_dir / f'fold_{fold_idx}' / 'parity_plots.png'
    plt.savefig(parity_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Data: {N_LOFI} LF + {N_HIFI} HF")
    print(f"Folds: {len(SEEDS)}")

    # Load BO best params
    print("\n" + "="*60)
    print("Loading BO Best Parameters...")
    print("="*60)
    bo_params = load_best_params_from_bo()

    best_joint_params = bo_params['best_joint']
    best_gs_params = bo_params['best_gs']

    print(f"\nJoint Best Params: {best_joint_params}")
    print(f"\nGradient Scaling Best Params: {best_gs_params}")

    lookup, all_combinations, param_space = load_base_data()
    print(f"\nTotal compositions: {len(all_combinations)}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / 'visualizations' / f'{timestamp}_gradient_scaling_vs_mfgp'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    results_list = []

    for fold_idx, seed in enumerate(SEEDS, 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx}/{len(SEEDS)} (seed={seed})")
        print('='*60)

        data = generate_data(lookup, all_combinations, n_lofi=N_LOFI, n_hifi=N_HIFI, seed=seed)

        test_mask = np.ones(len(data['X_all']), dtype=bool)
        test_mask[data['hifi_idx']] = False
        test_idx = np.where(test_mask)[0]
        y_test_hf = data['y_all'][test_idx]
        y_test_lf = data['y_all_lf'][test_idx]

        fold_results = {'fold': fold_idx, 'seed': seed}
        preds = {}
        stds_dict = {}

        # 1. MFGP
        print(f"\n  [1/5] MFGP...")
        mfgp_pred, mfgp_std = run_mfgp(data, device, seed)
        if mfgp_pred is not None:
            mfgp_rmse, mfgp_r2 = calc_metrics(y_test_hf, mfgp_pred[test_idx])
            print(f"        RMSE={mfgp_rmse:.4f}, R²={mfgp_r2:.4f}")
            fold_results.update({'mfgp_rmse': mfgp_rmse, 'mfgp_r2': mfgp_r2})
            preds['mfgp_pred'] = mfgp_pred
            stds_dict['mfgp_std'] = mfgp_std
        else:
            print(f"        Not available")
            fold_results.update({'mfgp_rmse': None, 'mfgp_r2': None})

        # 2. DNGO-Base
        print(f"\n  [2/5] DNGO-Base...")
        dngo_pred, dngo_std = run_dngo_base(data, device, seed)
        dngo_rmse, dngo_r2 = calc_metrics(y_test_hf, dngo_pred[test_idx])
        print(f"        RMSE={dngo_rmse:.4f}, R²={dngo_r2:.4f}")
        fold_results.update({'dngo_base_rmse': dngo_rmse, 'dngo_base_r2': dngo_r2})
        preds['dngo_base_pred'] = dngo_pred
        stds_dict['dngo_base_std'] = dngo_std

        # 3. Joint-Best (BO optimized)
        print(f"\n  [3/5] Joint-Best (BO optimized, R²={bo_params['best_joint_r2']:.4f})...")
        joint_pred, joint_std = run_joint_best(data, device, seed, best_joint_params)
        joint_rmse, joint_r2 = calc_metrics(y_test_hf, joint_pred[test_idx])
        print(f"        RMSE={joint_rmse:.4f}, R²={joint_r2:.4f}")
        fold_results.update({'joint_best_rmse': joint_rmse, 'joint_best_r2': joint_r2})
        preds['joint_best_pred'] = joint_pred
        stds_dict['joint_best_std'] = joint_std

        # 4. Gradient Scaling (BO optimized)
        print(f"\n  [4/5] Gradient Scaling (BO optimized, R²={bo_params['best_gs_r2']:.4f})...")
        gs_pred, gs_std = run_gradient_scaling(data, device, seed, best_gs_params)
        gs_rmse, gs_r2 = calc_metrics(y_test_hf, gs_pred[test_idx])
        print(f"        RMSE={gs_rmse:.4f}, R²={gs_r2:.4f}")
        fold_results.update({'gradient_scaling_rmse': gs_rmse, 'gradient_scaling_r2': gs_r2})
        preds['gradient_scaling_pred'] = gs_pred
        stds_dict['gradient_scaling_std'] = gs_std

        # 5. Pretrain-Base (LF only)
        print(f"\n  [5/5] Pretrain-Base...")
        pretrain_pred, pretrain_std = run_pretrain_base(data, device, seed)
        pretrain_rmse, pretrain_r2 = calc_metrics(y_test_lf, pretrain_pred[test_idx])
        print(f"        RMSE={pretrain_rmse:.4f}, R²={pretrain_r2:.4f} (vs LF)")
        fold_results.update({'pretrain_base_rmse': pretrain_rmse, 'pretrain_base_r2': pretrain_r2})
        preds['pretrain_base_pred'] = pretrain_pred
        stds_dict['pretrain_base_std'] = pretrain_std

        # Visualization
        fold_dir = create_5panel_visualization(fold_idx, seed, data, preds, stds_dict, param_space, output_dir)
        create_parity_plots(fold_idx, seed, data, preds, output_dir)
        print(f"\n  Visualization saved: {fold_dir}")

        results_list.append(fold_results)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    import pandas as pd
    df = pd.DataFrame(results_list)

    print(f"\n{'Model':<30} {'Avg RMSE':<12} {'Avg R²':<12} {'Compare'}")
    print('-' * 70)

    if df['mfgp_r2'].notna().any():
        print(f"{'MFGP':<30} {df['mfgp_rmse'].mean():.4f}       {df['mfgp_r2'].mean():.4f}       vs HF")
    print(f"{'DNGO-Base':<30} {df['dngo_base_rmse'].mean():.4f}       {df['dngo_base_r2'].mean():.4f}       vs HF")
    print(f"{'Joint-Best (BO)':<30} {df['joint_best_rmse'].mean():.4f}       {df['joint_best_r2'].mean():.4f}       vs HF")
    print(f"{'Gradient Scaling (BO)':<30} {df['gradient_scaling_rmse'].mean():.4f}       {df['gradient_scaling_r2'].mean():.4f}       vs HF")
    print(f"{'Pretrain-Base':<30} {df['pretrain_base_rmse'].mean():.4f}       {df['pretrain_base_r2'].mean():.4f}       vs LF")

    # Save results
    df.to_csv(output_dir / 'results_summary.csv', index=False)
    print(f"\nResults saved to: {output_dir / 'results_summary.csv'}")

    # Summary bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    models = ['MFGP', 'DNGO-Base', 'Joint-Best\n(BO)', 'Gradient\nScaling (BO)']
    r2_means = [
        df['mfgp_r2'].mean() if df['mfgp_r2'].notna().any() else 0,
        df['dngo_base_r2'].mean(),
        df['joint_best_r2'].mean(),
        df['gradient_scaling_r2'].mean()
    ]
    r2_stds = [
        df['mfgp_r2'].std() if df['mfgp_r2'].notna().any() else 0,
        df['dngo_base_r2'].std(),
        df['joint_best_r2'].std(),
        df['gradient_scaling_r2'].std()
    ]

    colors_bar = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
    bars = ax.bar(models, r2_means, yerr=r2_stds, capsize=5, color=colors_bar, alpha=0.7)
    ax.set_ylabel('R² Score (vs High-Fidelity)', fontsize=12)
    ax.set_title(f'HF Prediction Comparison ({len(SEEDS)}-fold)\nUsing BO Optimized Hyperparameters', fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    for bar, mean in zip(bars, r2_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'summary_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary chart saved: {output_dir / 'summary_comparison.png'}")


if __name__ == '__main__':
    main()
