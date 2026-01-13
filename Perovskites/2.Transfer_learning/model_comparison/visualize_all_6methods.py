#!/usr/bin/env python3
"""
6가지 Multi-Fidelity 학습 방법 비교 시각화

BO 실험에서 최적화된 6가지 방법 모두를 시각화:
1. Joint Training (R²=0.7848)
2. Gradient Scaling (R²=0.7839)
3. Sequential (R²=0.7793)
4. Progressive Unfreezing (R²=0.7549)
5. Curriculum Learning (R²=0.7489)
6. Two-Stage Joint (R²=0.7223)

+ MFGP (baseline)

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
parent_dir = current_dir.parent / 'Pure_TL_BO'
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(current_dir))

# MFGP
try:
    from mfgp_model import MultiFidelityGP
    MFGP_AVAILABLE = True
except ImportError:
    MFGP_AVAILABLE = False
    print("Warning: MFGP not available")

# ============================================================================
# Configuration
# ============================================================================

N_LOFI = 72
N_HIFI = 9
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

# ============================================================================
# Load BO Best Parameters for ALL 6 methods
# ============================================================================

def load_all_best_params():
    """Load best parameters for all 6 methods from BO results"""
    bo_results_path = parent_dir / 'results/large_scale_bo_v2_20251208_170617/bo_results_v2_20251208_170628.json'

    with open(bo_results_path, 'r') as f:
        bo_data = json.load(f)

    methods = ['joint', 'gradient_scaling', 'sequential', 'progressive', 'curriculum', 'two_stage_joint']
    best_params = {}

    for method in methods:
        best_value = -float('inf')
        best_p = None
        for trial in bo_data['all_trials']:
            if trial['params'].get('method') == method and trial['value'] is not None:
                if trial['value'] > best_value:
                    best_value = trial['value']
                    best_p = trial['params']

        if best_p:
            best_params[method] = {
                'params': best_p,
                'r2': best_value
            }
            print(f"  {method:<20}: R²={best_value:.4f}")

    return best_params


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
# Training Functions for ALL 6 Methods
# ============================================================================

def train_joint(model, X_lf, y_lf, X_hf, y_hf, params, device):
    """1. Joint Training with alpha weighting"""
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
    """2. Gradient Scaling Training"""
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
        lf_optimizer.zero_grad()
        y_lf_pred = model.forward_lf(X_lf_t)
        lf_loss = loss_fn(y_lf_pred, y_lf_t)
        (lf_loss * gradient_scale_lf).backward()
        lf_optimizer.step()

        hf_optimizer.zero_grad()
        with torch.no_grad():
            y_lf_for_hf = model.forward_lf(X_hf_t)
        y_hf_pred = model.forward_hf(X_hf_t, y_lf_for_hf)
        hf_loss = loss_fn(y_hf_pred, y_hf_t)
        (hf_loss * gradient_scale_hf).backward()
        hf_optimizer.step()


def train_sequential(model, X_lf, y_lf, X_hf, y_hf, params, device):
    """3. Sequential Training: LF first, then HF"""
    lf_lr = params.get('lf_lr', 1e-3)
    hf_lr = params.get('hf_lr', 1e-4)
    lf_epochs = params.get('lf_epochs', 200)
    hf_epochs = params.get('hf_epochs', 100)
    weight_decay = params.get('weight_decay', 1e-4)

    loss_fn = nn.MSELoss()

    X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(device)
    y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(device)
    X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(device)
    y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(device)

    # Stage 1: LF
    lf_optimizer = optim.Adam(model.lf_network.parameters(), lr=lf_lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(lf_epochs):
        lf_optimizer.zero_grad()
        y_lf_pred = model.forward_lf(X_lf_t)
        lf_loss = loss_fn(y_lf_pred, y_lf_t)
        lf_loss.backward()
        lf_optimizer.step()

    # Stage 2: Freeze LF, train HF
    for param in model.lf_network.parameters():
        param.requires_grad = False

    hf_optimizer = optim.Adam(model.hf_network.parameters(), lr=hf_lr, weight_decay=weight_decay)
    for epoch in range(hf_epochs):
        hf_optimizer.zero_grad()
        with torch.no_grad():
            y_lf_for_hf = model.forward_lf(X_hf_t)
        y_hf_pred = model.forward_hf(X_hf_t, y_lf_for_hf)
        hf_loss = loss_fn(y_hf_pred, y_hf_t)
        hf_loss.backward()
        hf_optimizer.step()

    for param in model.lf_network.parameters():
        param.requires_grad = True


def train_progressive(model, X_lf, y_lf, X_hf, y_hf, params, device):
    """4. Progressive Unfreezing"""
    lf_lr = params.get('lf_lr', 1e-3)
    hf_lr = params.get('hf_lr', 1e-4)
    lf_epochs = params.get('prog_lf_epochs', params.get('lf_epochs', 200))
    hf_epochs_per_phase = params.get('hf_epochs_per_phase', 50)
    weight_decay = params.get('weight_decay', 1e-4)
    lr_decay_per_phase = params.get('lr_decay_per_phase', 0.7)

    loss_fn = nn.MSELoss()

    X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(device)
    y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(device)
    X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(device)
    y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(device)

    # Stage 1: LF
    lf_optimizer = optim.Adam(model.lf_network.parameters(), lr=lf_lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(lf_epochs):
        lf_optimizer.zero_grad()
        y_lf_pred = model.forward_lf(X_lf_t)
        lf_loss = loss_fn(y_lf_pred, y_lf_t)
        lf_loss.backward()
        lf_optimizer.step()

    # Stage 2: Progressive unfreezing HF
    for param in model.lf_network.parameters():
        param.requires_grad = False
    for param in model.hf_network.parameters():
        param.requires_grad = False

    hf_layers = []
    for name, module in model.hf_network.named_children():
        if name == 'feature_net':
            for sub_module in module:
                if isinstance(sub_module, nn.Linear):
                    hf_layers.append(sub_module)
        elif name == 'out_layer':
            hf_layers.append(module)

    current_lr = hf_lr
    for phase, layer in enumerate(reversed(hf_layers)):
        for param in layer.parameters():
            param.requires_grad = True

        trainable_params = [p for p in model.hf_network.parameters() if p.requires_grad]
        hf_optimizer = optim.Adam(trainable_params, lr=current_lr, weight_decay=weight_decay)

        for epoch in range(hf_epochs_per_phase):
            hf_optimizer.zero_grad()
            with torch.no_grad():
                y_lf_for_hf = model.forward_lf(X_hf_t)
            y_hf_pred = model.forward_hf(X_hf_t, y_lf_for_hf)
            hf_loss = loss_fn(y_hf_pred, y_hf_t)
            hf_loss.backward()
            hf_optimizer.step()

        current_lr *= lr_decay_per_phase

    for param in model.parameters():
        param.requires_grad = True


def train_curriculum(model, X_lf, y_lf, X_hf, y_hf, params, device):
    """5. Curriculum Training: gradual alpha increase"""
    alpha_start = params.get('alpha_start', 0.1)
    alpha_end = params.get('alpha_end', 0.9)
    lf_lr = params.get('lf_lr', 1e-3)
    hf_lr = params.get('hf_lr', 1e-4)
    epochs = params.get('curriculum_epochs', params.get('epochs', 200))
    weight_decay = params.get('weight_decay', 1e-4)
    schedule = params.get('curriculum_schedule', 'linear')

    loss_fn = nn.MSELoss()

    X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(device)
    y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(device)
    X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(device)
    y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(device)

    optimizer = optim.Adam([
        {'params': model.lf_network.parameters(), 'lr': lf_lr, 'weight_decay': weight_decay},
        {'params': model.hf_network.parameters(), 'lr': hf_lr, 'weight_decay': weight_decay}
    ])

    model.train()
    for epoch in range(epochs):
        progress = epoch / max(epochs - 1, 1)
        if schedule == 'cosine':
            alpha = alpha_start + (alpha_end - alpha_start) * (1 - np.cos(progress * np.pi)) / 2
        elif schedule == 'step':
            if progress < 0.33:
                alpha = alpha_start
            elif progress < 0.66:
                alpha = (alpha_start + alpha_end) / 2
            else:
                alpha = alpha_end
        else:  # linear
            alpha = alpha_start + (alpha_end - alpha_start) * progress

        optimizer.zero_grad()
        y_lf_pred = model.forward_lf(X_lf_t)
        lf_loss = loss_fn(y_lf_pred, y_lf_t)
        y_lf_for_hf = model.forward_lf(X_hf_t)
        y_hf_pred = model.forward_hf(X_hf_t, y_lf_for_hf)
        hf_loss = loss_fn(y_hf_pred, y_hf_t)
        total_loss = (1 - alpha) * lf_loss + alpha * hf_loss
        total_loss.backward()
        optimizer.step()


def train_two_stage_joint(model, X_lf, y_lf, X_hf, y_hf, params, device):
    """6. Two-Stage Joint: LF warmup then joint"""
    alpha = params.get('alpha_twostage', params.get('alpha', 0.5))
    lf_lr = params.get('lf_lr', 1e-3)
    hf_lr = params.get('hf_lr', 1e-4)
    lf_warmup_epochs = params.get('lf_warmup_epochs', 100)
    joint_epochs = params.get('joint_epochs', 100)
    weight_decay = params.get('weight_decay', 1e-4)

    loss_fn = nn.MSELoss()

    X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(device)
    y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(device)
    X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(device)
    y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(device)

    # Stage 1: LF Warmup
    lf_optimizer = optim.Adam(model.lf_network.parameters(), lr=lf_lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(lf_warmup_epochs):
        lf_optimizer.zero_grad()
        y_lf_pred = model.forward_lf(X_lf_t)
        lf_loss = loss_fn(y_lf_pred, y_lf_t)
        lf_loss.backward()
        lf_optimizer.step()

    # Stage 2: Joint Training
    joint_lf_lr = lf_lr * 0.1
    joint_optimizer = optim.Adam([
        {'params': model.lf_network.parameters(), 'lr': joint_lf_lr, 'weight_decay': weight_decay},
        {'params': model.hf_network.parameters(), 'lr': hf_lr, 'weight_decay': weight_decay}
    ])

    for epoch in range(joint_epochs):
        joint_optimizer.zero_grad()
        y_lf_pred = model.forward_lf(X_lf_t)
        lf_loss = loss_fn(y_lf_pred, y_lf_t)
        y_lf_for_hf = model.forward_lf(X_hf_t)
        y_hf_pred = model.forward_hf(X_hf_t, y_lf_for_hf)
        hf_loss = loss_fn(y_hf_pred, y_hf_t)
        total_loss = (1 - alpha) * lf_loss + alpha * hf_loss
        total_loss.backward()
        joint_optimizer.step()


# ============================================================================
# Model Runners
# ============================================================================

def create_and_train_model(method, data, device, seed, params):
    """Create and train model with specified method"""
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

    # Select training function
    train_funcs = {
        'joint': train_joint,
        'gradient_scaling': train_gradient_scaling,
        'sequential': train_sequential,
        'progressive': train_progressive,
        'curriculum': train_curriculum,
        'two_stage_joint': train_two_stage_joint
    }

    train_func = train_funcs[method]
    train_func(model, data['X_low'], data['y_low'], data['X_high'], data['y_high'], params, device)

    # Evaluation
    model.eval()
    X_t = torch.tensor(data['X_all'], dtype=torch.float32).to(device)
    with torch.no_grad():
        y_lf_pred_t = model.forward_lf(X_t)
        y_hf_pred = model.forward_hf(X_t, y_lf_pred_t).cpu().numpy().flatten()

    std = np.ones_like(y_hf_pred) * 0.1
    return y_hf_pred, std


def run_mfgp(data, device, seed):
    """Run MFGP baseline"""
    if not MFGP_AVAILABLE:
        return None, None
    set_seeds(seed)
    mfgp = MultiFidelityGP(input_dim=data['X_low'].shape[1], device=device)
    mfgp.fit(data['X_low'], data['y_low'], data['X_high'], data['y_high'])
    mfgp_pred, mfgp_std = mfgp.predict(data['X_all'], return_std=True)
    return mfgp_pred, mfgp_std


# ============================================================================
# Visualization Functions
# ============================================================================

def create_7panel_visualization(fold_idx, seed, data, results, stds, param_space, output_dir, best_params):
    """7-panel visualization (MFGP + 6 methods)"""
    n_all = len(data['X_all'])
    hifi_idx = data['hifi_idx']
    lofi_idx = data['lofi_idx']
    y_all_hf = data['y_all']

    colors = {
        'mfgp': '#1f77b4',
        'joint': '#2ca02c',
        'gradient_scaling': '#9467bd',
        'sequential': '#ff7f0e',
        'progressive': '#d62728',
        'curriculum': '#8c564b',
        'two_stage_joint': '#17becf'
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

    # Value Sort
    sort_idx = np.argsort(y_all_hf)
    y_sorted = y_all_hf[sort_idx]
    x_axis = np.arange(len(y_sorted))
    hf_train_mask = np.isin(sort_idx, hifi_idx)
    value_labels = [comp_labels[i] for i in sort_idx]

    fig, axes = plt.subplots(7, 1, figsize=(24, 42))

    models_info = [
        ('MFGP (Baseline)', 'mfgp', colors['mfgp'], None),
        ('Joint (R²=0.785)', 'joint', colors['joint'], best_params.get('joint', {}).get('r2', 0)),
        ('Gradient Scaling (R²=0.784)', 'gradient_scaling', colors['gradient_scaling'], best_params.get('gradient_scaling', {}).get('r2', 0)),
        ('Sequential (R²=0.779)', 'sequential', colors['sequential'], best_params.get('sequential', {}).get('r2', 0)),
        ('Progressive (R²=0.755)', 'progressive', colors['progressive'], best_params.get('progressive', {}).get('r2', 0)),
        ('Curriculum (R²=0.749)', 'curriculum', colors['curriculum'], best_params.get('curriculum', {}).get('r2', 0)),
        ('Two-Stage Joint (R²=0.722)', 'two_stage_joint', colors['two_stage_joint'], best_params.get('two_stage_joint', {}).get('r2', 0)),
    ]

    for ax_idx, (name, key, color, bo_r2) in enumerate(models_info):
        ax = axes[ax_idx]

        pred = results.get(f'{key}_pred')
        std = stds.get(f'{key}_std')

        if pred is None:
            ax.text(0.5, 0.5, f'{name}\nNot Available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(name)
            continue

        pred_sorted = pred[sort_idx]
        std_sorted = std[sort_idx] if std is not None else np.zeros_like(pred_sorted)

        # Uncertainty band
        ax.fill_between(x_axis, pred_sorted - 2*std_sorted, pred_sorted + 2*std_sorted,
                        alpha=0.3, color=color, label='±2σ')

        # Predictions
        ax.plot(x_axis, pred_sorted, color=color, linewidth=0.8, alpha=0.7, label=f'{name.split(" (")[0]} Predicted')

        # Test points
        ax.scatter(x_axis[~hf_train_mask], y_sorted[~hf_train_mask], c='black', s=15, zorder=5,
                   label='Test HF', alpha=0.6)

        # Train points
        ax.scatter(x_axis[hf_train_mask], y_sorted[hf_train_mask], c='red', s=100, marker='*',
                   zorder=6, label=f'Train HF ({np.sum(hf_train_mask)})', edgecolors='darkred')

        # Metrics
        test_mask = ~hf_train_mask
        rmse, r2 = calc_metrics(y_sorted[test_mask], pred_sorted[test_mask])

        ax.set_ylabel('Bandgap (eV)', fontsize=12)
        ax.set_title(f'{name}: RMSE={rmse:.3f}, R²={r2:.3f}', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xticks(x_axis[::4])
    axes[-1].set_xticklabels([value_labels[i] for i in range(0, len(value_labels), 4)], rotation=90, fontsize=6)
    axes[-1].set_xlabel('Composition (sorted by HF value)', fontsize=12)

    plt.suptitle(f'Fold {fold_idx} (seed={seed}) - All 6 MF Methods vs MFGP Baseline',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(fold_dir / 'predictions_7panel.png', dpi=150, bbox_inches='tight')
    plt.close()

    return fold_dir


def create_parity_plots(fold_idx, seed, data, results, output_dir, best_params):
    """Parity plots for all 7 models (2x4 grid, last cell for summary)"""
    test_mask = np.ones(len(data['X_all']), dtype=bool)
    test_mask[data['hifi_idx']] = False
    y_test = data['y_all'][test_mask]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    colors = {
        'mfgp': '#1f77b4',
        'joint': '#2ca02c',
        'gradient_scaling': '#9467bd',
        'sequential': '#ff7f0e',
        'progressive': '#d62728',
        'curriculum': '#8c564b',
        'two_stage_joint': '#17becf'
    }

    model_info = [
        ('MFGP', 'mfgp'),
        ('Joint', 'joint'),
        ('Gradient Scaling', 'gradient_scaling'),
        ('Sequential', 'sequential'),
        ('Progressive', 'progressive'),
        ('Curriculum', 'curriculum'),
        ('Two-Stage Joint', 'two_stage_joint'),
    ]

    mn = y_test.min() - 0.3
    mx = y_test.max() + 0.3

    r2_values = {}

    for idx, (name, key) in enumerate(model_info):
        ax = axes[idx // 4, idx % 4]
        color = colors[key]

        pred = results.get(f'{key}_pred')
        if pred is None:
            ax.text(0.5, 0.5, f'{name}\nNot Available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(name)
            r2_values[key] = None
            continue

        pred_test = pred[test_mask]
        rmse, r2 = calc_metrics(y_test, pred_test)
        r2_values[key] = r2

        ax.scatter(y_test, pred_test, c=color, alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
        ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5, alpha=0.5, label='y=x')
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)
        ax.set_xlabel('Actual HF (eV)', fontsize=10)
        ax.set_ylabel('Predicted HF (eV)', fontsize=10)
        ax.set_title(f'{name}\nRMSE={rmse:.3f}, R²={r2:.3f}', fontsize=11)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Summary bar chart in last cell
    ax_summary = axes[1, 3]
    valid_models = [(name, key) for name, key in model_info if r2_values.get(key) is not None]
    names = [name for name, key in valid_models]
    r2s = [r2_values[key] for name, key in valid_models]
    bar_colors = [colors[key] for name, key in valid_models]

    bars = ax_summary.barh(names, r2s, color=bar_colors, alpha=0.7)
    ax_summary.set_xlabel('R² Score', fontsize=11)
    ax_summary.set_title('R² Comparison', fontsize=12)
    ax_summary.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax_summary.set_xlim(-0.5, 1.0)

    for bar, r2 in zip(bars, r2s):
        ax_summary.text(max(r2 + 0.02, 0.05), bar.get_y() + bar.get_height()/2,
                       f'{r2:.3f}', va='center', fontsize=9)

    plt.suptitle(f'Fold {fold_idx} (seed={seed}) - Parity Plots for All Methods', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    parity_path = output_dir / f'fold_{fold_idx}' / 'parity_plots.png'
    plt.savefig(parity_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_chart(results_df, output_dir, best_params):
    """Create summary comparison bar chart"""
    methods = ['mfgp', 'joint', 'gradient_scaling', 'sequential', 'progressive', 'curriculum', 'two_stage_joint']
    labels = ['MFGP\n(Baseline)', 'Joint\n(Best)', 'Gradient\nScaling', 'Sequential', 'Progressive', 'Curriculum', 'Two-Stage\nJoint']
    colors = ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#d62728', '#8c564b', '#17becf']

    r2_means = []
    r2_stds = []

    for method in methods:
        col = f'{method}_r2'
        if col in results_df.columns and results_df[col].notna().any():
            r2_means.append(results_df[col].mean())
            r2_stds.append(results_df[col].std())
        else:
            r2_means.append(0)
            r2_stds.append(0)

    fig, ax = plt.subplots(figsize=(14, 7))
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, r2_means, yerr=r2_stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('R² Score (vs High-Fidelity)', fontsize=12)
    ax.set_title(f'HF Prediction Comparison: All 6 Methods ({len(results_df)}-fold)\nUsing BO Optimized Hyperparameters', fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(-0.5, 1.0)

    for bar, mean, std in zip(bars, r2_means, r2_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add BO best R² annotations
    bo_r2s = [None, 0.7848, 0.7839, 0.7793, 0.7549, 0.7489, 0.7223]
    for i, bo_r2 in enumerate(bo_r2s):
        if bo_r2:
            ax.annotate(f'BO: {bo_r2:.3f}', xy=(i, -0.35), ha='center', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(output_dir / 'summary_all_methods.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Data: {N_LOFI} LF + {N_HIFI} HF")
    print(f"Folds: {len(SEEDS)}")

    # Load BO best params for all methods
    print("\n" + "="*60)
    print("Loading BO Best Parameters for All 6 Methods...")
    print("="*60)
    best_params = load_all_best_params()

    lookup, all_combinations, param_space = load_base_data()
    print(f"\nTotal compositions: {len(all_combinations)}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / 'visualizations' / f'{timestamp}_all_6methods'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    results_list = []
    methods = ['joint', 'gradient_scaling', 'sequential', 'progressive', 'curriculum', 'two_stage_joint']

    for fold_idx, seed in enumerate(SEEDS, 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx}/{len(SEEDS)} (seed={seed})")
        print('='*60)

        data = generate_data(lookup, all_combinations, n_lofi=N_LOFI, n_hifi=N_HIFI, seed=seed)

        test_mask = np.ones(len(data['X_all']), dtype=bool)
        test_mask[data['hifi_idx']] = False
        test_idx = np.where(test_mask)[0]
        y_test_hf = data['y_all'][test_idx]

        fold_results = {'fold': fold_idx, 'seed': seed}
        preds = {}
        stds_dict = {}

        # 1. MFGP
        print(f"\n  [1/7] MFGP...")
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

        # 2-7. All 6 methods
        for i, method in enumerate(methods, 2):
            method_info = best_params.get(method, {})
            params = method_info.get('params', {})
            bo_r2 = method_info.get('r2', 0)

            print(f"\n  [{i}/7] {method} (BO R²={bo_r2:.4f})...")

            try:
                pred, std = create_and_train_model(method, data, device, seed, params)
                rmse, r2 = calc_metrics(y_test_hf, pred[test_idx])
                print(f"        RMSE={rmse:.4f}, R²={r2:.4f}")
                fold_results.update({f'{method}_rmse': rmse, f'{method}_r2': r2})
                preds[f'{method}_pred'] = pred
                stds_dict[f'{method}_std'] = std
            except Exception as e:
                print(f"        Error: {e}")
                fold_results.update({f'{method}_rmse': None, f'{method}_r2': None})

        # Visualization
        fold_dir = create_7panel_visualization(fold_idx, seed, data, preds, stds_dict, param_space, output_dir, best_params)
        create_parity_plots(fold_idx, seed, data, preds, output_dir, best_params)
        print(f"\n  Visualization saved: {fold_dir}")

        results_list.append(fold_results)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    import pandas as pd
    df = pd.DataFrame(results_list)

    print(f"\n{'Model':<25} {'Avg RMSE':<12} {'Avg R²':<12} {'Std R²':<12}")
    print('-' * 70)

    all_methods = ['mfgp'] + methods
    for method in all_methods:
        rmse_col = f'{method}_rmse'
        r2_col = f'{method}_r2'
        if r2_col in df.columns and df[r2_col].notna().any():
            print(f"{method:<25} {df[rmse_col].mean():.4f}       {df[r2_col].mean():.4f}       {df[r2_col].std():.4f}")

    # Save results
    df.to_csv(output_dir / 'results_summary.csv', index=False)
    print(f"\nResults saved to: {output_dir / 'results_summary.csv'}")

    # Create summary chart
    create_summary_chart(df, output_dir, best_params)
    print(f"Summary chart saved: {output_dir / 'summary_all_methods.png'}")


if __name__ == '__main__':
    main()
