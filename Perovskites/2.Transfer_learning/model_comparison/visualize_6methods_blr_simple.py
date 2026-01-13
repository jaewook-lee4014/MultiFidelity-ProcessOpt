#!/usr/bin/env python3
"""
6가지 Multi-Fidelity 학습 방법 + Simple BLR (DNN 출력 기반)

핵심 아이디어:
- DNN이 이미 좋은 예측을 하므로, DNN 출력 자체를 BLR의 입력으로 사용
- BLR은 DNN 출력과 실제 값 사이의 선형 보정 + 불확실성 추정만 담당
- 파라미터: 2개 (scale + bias) → 9개 샘플로 충분히 학습 가능

접근법 1: DNN 출력 기반 BLR
- X_blr = [dnn_pred, 1]
- y_blr = y_hf
- BLR이 학습: y = w1 * dnn_pred + w0

접근법 2: 잔차 기반 간단 불확실성
- DNN 예측 후, 테스트 포인트와 학습 포인트의 거리 기반 불확실성

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
# Bayesian Linear Regression (DNN 출력 기반)
# ============================================================================

class SimpleBLR:
    """
    Simple Bayesian Linear Regression
    DNN 출력을 입력으로 받아 보정 + 불확실성 추정

    y = w1 * x + w0  (x = dnn_pred)
    """

    def __init__(self, alpha=1.0, beta=25.0):
        self.alpha = alpha
        self.beta = beta
        self.mean = None
        self.cov = None
        self.fitted = False

    def fit(self, dnn_preds, y_true):
        """
        Args:
            dnn_preds: DNN의 예측값 (N,)
            y_true: 실제 HF 값 (N,)
        """
        X = np.column_stack([np.ones(len(dnn_preds)), dnn_preds])  # [1, x]
        y = np.asarray(y_true, dtype=np.float64).flatten()

        # Prior: w ~ N(0, α^(-1)I)
        S0_inv = self.alpha * np.eye(2)

        # Posterior
        S_N_inv = S0_inv + self.beta * X.T @ X
        S_N_inv += 1e-6 * np.eye(2)  # numerical stability

        self.cov = np.linalg.inv(S_N_inv)
        self.mean = self.beta * self.cov @ X.T @ y

        self.fitted = True

    def predict(self, dnn_pred):
        """단일 점 예측"""
        x = np.array([1, dnn_pred])
        mu = x @ self.mean
        var = (1/self.beta) + x @ self.cov @ x
        return mu, max(var, 1e-8)

    def predict_batch(self, dnn_preds):
        """배치 예측"""
        means, variances = [], []
        for pred in dnn_preds:
            mu, var = self.predict(pred)
            means.append(mu)
            variances.append(var)
        return np.array(means), np.array(variances)


# ============================================================================
# Distance-Based Uncertainty (보조 방법)
# ============================================================================

class DistanceUncertainty:
    """
    학습 데이터와의 거리 기반 불확실성 추정
    - 학습 데이터에서 멀수록 불확실성 증가
    - 학습 데이터에서의 잔차 기반 베이스라인 불확실성
    """

    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale
        self.X_train = None
        self.residual_std = None

    def fit(self, X_train, y_train, y_pred_train):
        """
        Args:
            X_train: 학습 입력 (N, D)
            y_train: 학습 타겟 (N,)
            y_pred_train: 학습 데이터에 대한 DNN 예측 (N,)
        """
        self.X_train = np.asarray(X_train, dtype=np.float32)
        residuals = np.asarray(y_train) - np.asarray(y_pred_train)
        self.residual_std = np.std(residuals) if len(residuals) > 1 else 0.1

    def estimate_std(self, X_test):
        """
        테스트 포인트에 대한 불확실성 추정
        """
        X_test = np.asarray(X_test, dtype=np.float32)
        stds = []

        for x in X_test:
            # 가장 가까운 학습 포인트와의 거리
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            min_dist = np.min(distances)

            # 거리 기반 불확실성: 멀수록 증가
            dist_factor = 1 + min_dist / self.length_scale
            std = self.residual_std * dist_factor

            stds.append(std)

        return np.array(stds)


# ============================================================================
# Load BO Best Parameters
# ============================================================================

def load_all_best_params():
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
# Network Definitions (동일)
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
# Training Functions (동일)
# ============================================================================

def train_joint(model, X_lf, y_lf, X_hf, y_hf, params, device):
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

    lf_optimizer = optim.Adam(model.lf_network.parameters(), lr=lf_lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(lf_epochs):
        lf_optimizer.zero_grad()
        y_lf_pred = model.forward_lf(X_lf_t)
        lf_loss = loss_fn(y_lf_pred, y_lf_t)
        lf_loss.backward()
        lf_optimizer.step()

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

    lf_optimizer = optim.Adam(model.lf_network.parameters(), lr=lf_lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(lf_epochs):
        lf_optimizer.zero_grad()
        y_lf_pred = model.forward_lf(X_lf_t)
        lf_loss = loss_fn(y_lf_pred, y_lf_t)
        lf_loss.backward()
        lf_optimizer.step()

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
        else:
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

    lf_optimizer = optim.Adam(model.lf_network.parameters(), lr=lf_lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(lf_warmup_epochs):
        lf_optimizer.zero_grad()
        y_lf_pred = model.forward_lf(X_lf_t)
        lf_loss = loss_fn(y_lf_pred, y_lf_t)
        lf_loss.backward()
        lf_optimizer.step()

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
# Model Runner with Simple BLR
# ============================================================================

def create_and_train_model_with_simple_blr(method, data, device, seed, params):
    """
    DNN 훈련 후 Simple BLR로 보정 + 불확실성 추정

    핵심:
    1. DNN 훈련 (기존과 동일)
    2. HF 데이터에 대해 DNN 예측 수행
    3. Simple BLR 학습: y_hf = w1 * dnn_pred + w0
    4. 전체 데이터 예측 시 BLR 적용

    장점:
    - DNN 예측을 그대로 활용 (성능 유지)
    - BLR이 선형 보정만 담당 (2개 파라미터 → 9개 샘플로 충분)
    - 적절한 불확실성 추정 가능
    """
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

    # Step 1: DNN 훈련
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

    # Step 2: HF 데이터에 대한 DNN 예측
    model.eval()
    X_hf_t = torch.tensor(data['X_high'], dtype=torch.float32).to(device)

    with torch.no_grad():
        dnn_pred_hf = model.forward_hf(X_hf_t).cpu().numpy().flatten()

    # Step 3: Simple BLR 학습 (DNN 출력 → HF 실제 값)
    blr = SimpleBLR(alpha=1.0, beta=25.0)
    blr.fit(dnn_pred_hf, data['y_high'])

    # Step 4: 전체 데이터 예측
    X_all_t = torch.tensor(data['X_all'], dtype=torch.float32).to(device)

    with torch.no_grad():
        dnn_pred_all = model.forward_hf(X_all_t).cpu().numpy().flatten()

    # BLR 적용
    y_pred, y_var = blr.predict_batch(dnn_pred_all)
    std = np.sqrt(np.maximum(y_var, 1e-8))

    return y_pred, std, dnn_pred_all


def create_and_train_model_dnn_only(method, data, device, seed, params):
    """DNN만 사용 (비교용)"""
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

    model.eval()
    X_all_t = torch.tensor(data['X_all'], dtype=torch.float32).to(device)

    with torch.no_grad():
        y_hf_pred = model.forward_hf(X_all_t).cpu().numpy().flatten()

    std = np.full_like(y_hf_pred, 0.1)  # 고정 불확실성

    return y_hf_pred, std


def run_mfgp(data, device, seed):
    if not MFGP_AVAILABLE:
        return None, None
    set_seeds(seed)
    mfgp = MultiFidelityGP(input_dim=data['X_low'].shape[1], device=device)
    mfgp.fit(data['X_low'], data['y_low'], data['X_high'], data['y_high'])
    mfgp_pred, mfgp_std = mfgp.predict(data['X_all'], return_std=True)
    return mfgp_pred, mfgp_std


# ============================================================================
# Visualization
# ============================================================================

def create_summary_chart(results_df, output_dir):
    methods = ['joint', 'gradient_scaling', 'sequential', 'progressive', 'curriculum', 'two_stage_joint']
    labels = ['Joint', 'Grad\nScale', 'Seq', 'Prog', 'Curr', '2Stage']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    x_pos = np.arange(len(methods))
    width = 0.35

    # R² Comparison
    ax1 = axes[0]
    dnn_r2 = [results_df[f'{m}_dnn_r2'].mean() if f'{m}_dnn_r2' in results_df.columns else 0 for m in methods]
    blr_r2 = [results_df[f'{m}_blr_r2'].mean() if f'{m}_blr_r2' in results_df.columns else 0 for m in methods]
    dnn_r2_std = [results_df[f'{m}_dnn_r2'].std() if f'{m}_dnn_r2' in results_df.columns else 0 for m in methods]
    blr_r2_std = [results_df[f'{m}_blr_r2'].std() if f'{m}_blr_r2' in results_df.columns else 0 for m in methods]

    bars1 = ax1.bar(x_pos - width/2, dnn_r2, width, yerr=dnn_r2_std, label='DNN Only', color='steelblue', alpha=0.8, capsize=3)
    bars2 = ax1.bar(x_pos + width/2, blr_r2, width, yerr=blr_r2_std, label='DNN+SimpleBLR', color='coral', alpha=0.8, capsize=3)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('R² Score')
    ax1.set_title(f'R² Comparison ({len(results_df)}-fold CV)')
    ax1.legend()
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars1, dnn_r2):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, f'{val:.3f}',
                ha='center', fontsize=8)
    for bar, val in zip(bars2, blr_r2):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, f'{val:.3f}',
                ha='center', fontsize=8)

    # RMSE Comparison
    ax2 = axes[1]
    dnn_rmse = [results_df[f'{m}_dnn_rmse'].mean() if f'{m}_dnn_rmse' in results_df.columns else 0 for m in methods]
    blr_rmse = [results_df[f'{m}_blr_rmse'].mean() if f'{m}_blr_rmse' in results_df.columns else 0 for m in methods]

    bars1 = ax2.bar(x_pos - width/2, dnn_rmse, width, label='DNN Only', color='steelblue', alpha=0.8)
    bars2 = ax2.bar(x_pos + width/2, blr_rmse, width, label='DNN+SimpleBLR', color='coral', alpha=0.8)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('RMSE')
    ax2.set_title(f'RMSE Comparison ({len(results_df)}-fold CV)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Average Uncertainty
    ax3 = axes[2]
    mfgp_std = results_df['mfgp_avg_std'].mean() if 'mfgp_avg_std' in results_df.columns else 0
    blr_stds = [results_df[f'{m}_blr_avg_std'].mean() if f'{m}_blr_avg_std' in results_df.columns else 0 for m in methods]

    all_labels = ['MFGP'] + labels
    all_stds = [mfgp_std] + blr_stds
    colors = ['#1f77b4'] + ['coral'] * len(methods)

    bars = ax3.bar(all_labels, all_stds, color=colors, alpha=0.8)
    ax3.set_ylabel('Average Uncertainty (σ)')
    ax3.set_title('Uncertainty Estimation (Simple BLR)')
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, all_stds):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
                ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'summary_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Data: {N_LOFI} LF + {N_HIFI} HF")
    print(f"Folds: {len(SEEDS)}")

    print("\n" + "="*60)
    print("Loading BO Best Parameters...")
    print("="*60)
    best_params = load_all_best_params()

    lookup, all_combinations, param_space = load_base_data()
    print(f"\nTotal compositions: {len(all_combinations)}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / 'visualizations' / f'{timestamp}_simple_blr'
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

        # 1. MFGP
        print(f"\n  [1/{len(methods)*2 + 1}] MFGP...")
        mfgp_pred, mfgp_std = run_mfgp(data, device, seed)
        if mfgp_pred is not None:
            mfgp_rmse, mfgp_r2 = calc_metrics(y_test_hf, mfgp_pred[test_idx])
            mfgp_avg_std = np.mean(mfgp_std[test_idx])
            print(f"        RMSE={mfgp_rmse:.4f}, R²={mfgp_r2:.4f}, Avg σ={mfgp_avg_std:.4f}")
            fold_results.update({'mfgp_rmse': mfgp_rmse, 'mfgp_r2': mfgp_r2, 'mfgp_avg_std': mfgp_avg_std})

        # 2-7. All 6 methods
        for i, method in enumerate(methods):
            method_info = best_params.get(method, {})
            params = method_info.get('params', {})
            bo_r2 = method_info.get('r2', 0)

            # DNN Only
            print(f"\n  [{2 + i*2}/{len(methods)*2 + 1}] {method} (DNN only, BO R²={bo_r2:.4f})...")
            try:
                pred_dnn, _ = create_and_train_model_dnn_only(method, data, device, seed, params)
                rmse, r2 = calc_metrics(y_test_hf, pred_dnn[test_idx])
                print(f"        RMSE={rmse:.4f}, R²={r2:.4f}")
                fold_results.update({f'{method}_dnn_rmse': rmse, f'{method}_dnn_r2': r2})
            except Exception as e:
                print(f"        Error: {e}")

            # DNN + Simple BLR
            print(f"\n  [{3 + i*2}/{len(methods)*2 + 1}] {method} + Simple BLR...")
            try:
                pred_blr, std_blr, _ = create_and_train_model_with_simple_blr(method, data, device, seed, params)
                rmse, r2 = calc_metrics(y_test_hf, pred_blr[test_idx])
                avg_std = np.mean(std_blr[test_idx])
                print(f"        RMSE={rmse:.4f}, R²={r2:.4f}, Avg σ={avg_std:.4f}")
                fold_results.update({f'{method}_blr_rmse': rmse, f'{method}_blr_r2': r2, f'{method}_blr_avg_std': avg_std})
            except Exception as e:
                print(f"        Error: {e}")
                import traceback
                traceback.print_exc()

        results_list.append(fold_results)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    import pandas as pd
    df = pd.DataFrame(results_list)

    print(f"\n{'Model':<30} {'Avg RMSE':<12} {'Avg R²':<12} {'Avg σ':<12}")
    print('-' * 70)

    if 'mfgp_r2' in df.columns:
        print(f"{'MFGP':<30} {df['mfgp_rmse'].mean():.4f}       {df['mfgp_r2'].mean():.4f}       {df['mfgp_avg_std'].mean():.4f}")

    print()
    for method in methods:
        dnn_r2_col = f'{method}_dnn_r2'
        if dnn_r2_col in df.columns and df[dnn_r2_col].notna().any():
            print(f"{method + ' (DNN)':<30} {df[f'{method}_dnn_rmse'].mean():.4f}       {df[dnn_r2_col].mean():.4f}       {'N/A':<12}")

        blr_r2_col = f'{method}_blr_r2'
        if blr_r2_col in df.columns and df[blr_r2_col].notna().any():
            print(f"{method + ' + SimpleBLR':<30} {df[f'{method}_blr_rmse'].mean():.4f}       {df[blr_r2_col].mean():.4f}       {df[f'{method}_blr_avg_std'].mean():.4f}")

    df.to_csv(output_dir / 'results_simple_blr.csv', index=False)
    print(f"\nResults saved to: {output_dir / 'results_simple_blr.csv'}")

    create_summary_chart(df, output_dir)
    print(f"Summary chart saved: {output_dir / 'summary_comparison.png'}")


if __name__ == '__main__':
    main()
