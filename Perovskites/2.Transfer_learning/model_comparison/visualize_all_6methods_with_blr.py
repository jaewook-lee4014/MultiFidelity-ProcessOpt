#!/usr/bin/env python3
"""
6가지 Multi-Fidelity 학습 방법 + BLR 불확실성 추정 비교

기존 visualize_all_6methods.py와 동일한 훈련 방식이지만,
마지막에 BLR(Bayesian Linear Regression)을 추가하여 불확실성을 제대로 추정

핵심 변경점:
- DNN 훈련 후, HF 네트워크의 마지막 층 특징을 추출
- BLR을 HF 특징 + HF 타겟으로 학습
- 예측 시 BLR의 평균/분산 사용

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
# Bayesian Linear Regression (from DNGO)
# ============================================================================

class BayesianLinearRegression:
    """
    베이지안 선형 회귀 모델
    불확실성을 포함한 예측을 제공합니다.
    """

    def __init__(self, alpha=1.0, beta=25.0):
        """
        Args:
            alpha: 가중치의 정밀도 (precision) 파라미터
            beta: 노이즈의 정밀도 파라미터
        """
        self.alpha = alpha
        self.beta = beta
        self.mean = None
        self.cov = None
        self.fitted = False

    def fit(self, X, y):
        """
        베이지안 선형 회귀 학습

        Args:
            X: 입력 특성 (N x D)
            y: 타겟 값 (N,)
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).flatten()

        # 편향 항 추가
        X_with_bias = np.column_stack([np.ones(len(X)), X])

        # 사전 분포: w ~ N(0, α^(-1)I)
        S0_inv = self.alpha * np.eye(X_with_bias.shape[1])

        # 사후 분포 계산
        S_N_inv = S0_inv + self.beta * X_with_bias.T @ X_with_bias
        self.cov = np.linalg.inv(S_N_inv)
        self.mean = self.beta * self.cov @ X_with_bias.T @ y

        self.fitted = True

    def predict(self, x):
        """단일 점에 대한 예측"""
        if not self.fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        x = np.asarray(x, dtype=np.float32).flatten()
        x_with_bias = np.concatenate([[1], x])

        mu = x_with_bias @ self.mean
        var = (1/self.beta) + x_with_bias @ self.cov @ x_with_bias

        return mu, var

    def predict_batch(self, X):
        """배치 예측"""
        X = np.asarray(X, dtype=np.float32)
        means = []
        variances = []

        for x in X:
            mu, var = self.predict(x)
            means.append(mu)
            variances.append(var)

        return np.array(means), np.array(variances)


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
        self.hidden_dim = hidden_dim

    def forward(self, x, y_lf):
        combined = torch.cat([x, y_lf], dim=-1)
        features = self.feature_net(combined)
        delta = self.out_layer(features)
        if self.residual_mode:
            return y_lf + delta
        else:
            return delta

    def extract_features(self, x, y_lf):
        """BLR을 위한 특징 추출"""
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

    def extract_hf_features(self, x, y_lf=None):
        """HF 네트워크의 특징 추출 (BLR용)"""
        if y_lf is None:
            y_lf = self.lf_network(x)
        return self.hf_network.extract_features(x, y_lf)


# ============================================================================
# Training Functions for ALL 6 Methods (동일)
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
# Model Runners with BLR
# ============================================================================

def create_and_train_model_with_blr(method, data, device, seed, params):
    """
    모델 훈련 후 BLR을 붙여서 불확실성 추정

    핵심 변경:
    1. DNN 훈련 (기존과 동일)
    2. HF 네트워크의 마지막 특징층에서 특징 추출
    3. BLR을 HF 학습 데이터로 학습
    4. 예측 시 BLR 사용
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

    # Step 1: DNN 훈련 (기존과 동일)
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

    # Step 2: HF 특징 추출 (BLR 학습용)
    model.eval()
    X_hf_t = torch.tensor(data['X_high'], dtype=torch.float32).to(device)

    with torch.no_grad():
        y_lf_for_hf = model.forward_lf(X_hf_t)
        features_hf = model.extract_hf_features(X_hf_t, y_lf_for_hf).cpu().numpy()

    # Step 3: BLR 학습 (HF 데이터에서 추출한 특징 + HF 타겟)
    # Residual learning이므로 delta = y_hf - y_lf_pred 를 학습
    y_lf_pred_hf = y_lf_for_hf.cpu().numpy().flatten()
    y_hf_true = data['y_high']
    delta_hf = y_hf_true - y_lf_pred_hf  # HF 학습 데이터의 델타

    blr = BayesianLinearRegression(alpha=1.0, beta=25.0)
    blr.fit(features_hf, delta_hf)

    # Step 4: 전체 데이터에 대해 예측
    X_all_t = torch.tensor(data['X_all'], dtype=torch.float32).to(device)

    with torch.no_grad():
        y_lf_all = model.forward_lf(X_all_t)
        features_all = model.extract_hf_features(X_all_t, y_lf_all).cpu().numpy()
        y_lf_all_np = y_lf_all.cpu().numpy().flatten()

    # BLR로 delta 예측
    delta_pred, delta_var = blr.predict_batch(features_all)

    # 최종 예측: y_hf = y_lf + delta
    y_hf_pred = y_lf_all_np + delta_pred
    std = np.sqrt(delta_var)

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
    """7-panel visualization (MFGP + 6 methods with BLR)"""
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
        ('MFGP (Baseline)', 'mfgp', colors['mfgp']),
        ('Joint + BLR', 'joint', colors['joint']),
        ('Gradient Scaling + BLR', 'gradient_scaling', colors['gradient_scaling']),
        ('Sequential + BLR', 'sequential', colors['sequential']),
        ('Progressive + BLR', 'progressive', colors['progressive']),
        ('Curriculum + BLR', 'curriculum', colors['curriculum']),
        ('Two-Stage Joint + BLR', 'two_stage_joint', colors['two_stage_joint']),
    ]

    for ax_idx, (name, key, color) in enumerate(models_info):
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

        # Uncertainty band (BLR에서 나온 실제 불확실성)
        ax.fill_between(x_axis, pred_sorted - 2*std_sorted, pred_sorted + 2*std_sorted,
                        alpha=0.3, color=color, label='±2σ (BLR)')

        # Predictions
        ax.plot(x_axis, pred_sorted, color=color, linewidth=0.8, alpha=0.7, label=f'{name.split(" +")[0]} Predicted')

        # Test points
        ax.scatter(x_axis[~hf_train_mask], y_sorted[~hf_train_mask], c='black', s=15, zorder=5,
                   label='Test HF', alpha=0.6)

        # Train points
        ax.scatter(x_axis[hf_train_mask], y_sorted[hf_train_mask], c='red', s=100, marker='*',
                   zorder=6, label=f'Train HF ({np.sum(hf_train_mask)})', edgecolors='darkred')

        # Metrics
        test_mask = ~hf_train_mask
        rmse, r2 = calc_metrics(y_sorted[test_mask], pred_sorted[test_mask])

        # 평균 불확실성 계산
        avg_std = np.mean(std_sorted[test_mask])

        ax.set_ylabel('Bandgap (eV)', fontsize=12)
        ax.set_title(f'{name}: RMSE={rmse:.3f}, R²={r2:.3f}, Avg σ={avg_std:.3f}', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xticks(x_axis[::4])
    axes[-1].set_xticklabels([value_labels[i] for i in range(0, len(value_labels), 4)], rotation=90, fontsize=6)
    axes[-1].set_xlabel('Composition (sorted by HF value)', fontsize=12)

    plt.suptitle(f'Fold {fold_idx} (seed={seed}) - All 6 MF Methods + BLR vs MFGP',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(fold_dir / 'predictions_7panel_blr.png', dpi=150, bbox_inches='tight')
    plt.close()

    return fold_dir


def create_uncertainty_comparison(fold_idx, seed, data, results, stds, output_dir):
    """불확실성 비교 시각화"""
    test_mask = np.ones(len(data['X_all']), dtype=bool)
    test_mask[data['hifi_idx']] = False

    y_test = data['y_all'][test_mask]

    methods = ['mfgp', 'joint', 'gradient_scaling', 'sequential', 'progressive', 'curriculum', 'two_stage_joint']
    labels = ['MFGP', 'Joint+BLR', 'GradScale+BLR', 'Seq+BLR', 'Prog+BLR', 'Curr+BLR', '2Stage+BLR']
    colors = ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#d62728', '#8c564b', '#17becf']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 평균 불확실성 비교
    ax1 = axes[0, 0]
    avg_stds = []
    for method in methods:
        std = stds.get(f'{method}_std')
        if std is not None:
            avg_stds.append(np.mean(std[test_mask]))
        else:
            avg_stds.append(0)

    bars = ax1.bar(labels, avg_stds, color=colors, alpha=0.7)
    ax1.set_ylabel('Average Uncertainty (σ)', fontsize=11)
    ax1.set_title('Average Uncertainty per Method', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, avg_stds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=9)

    # 2. 예측 오차 vs 불확실성 (Calibration)
    ax2 = axes[0, 1]
    for method, label, color in zip(methods[:4], labels[:4], colors[:4]):
        pred = results.get(f'{method}_pred')
        std = stds.get(f'{method}_std')
        if pred is not None and std is not None:
            errors = np.abs(y_test - pred[test_mask])
            uncertainties = std[test_mask]
            ax2.scatter(uncertainties, errors, c=color, alpha=0.5, s=20, label=label)

    ax2.plot([0, ax2.get_xlim()[1]], [0, ax2.get_xlim()[1]], 'k--', alpha=0.5, label='Perfect calibration')
    ax2.set_xlabel('Predicted Uncertainty (σ)', fontsize=11)
    ax2.set_ylabel('Actual Error |y - ŷ|', fontsize=11)
    ax2.set_title('Calibration: Error vs Uncertainty', fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. R² vs 평균 불확실성
    ax3 = axes[1, 0]
    r2_values = []
    for method in methods:
        pred = results.get(f'{method}_pred')
        if pred is not None:
            _, r2 = calc_metrics(y_test, pred[test_mask])
            r2_values.append(r2)
        else:
            r2_values.append(0)

    for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
        if avg_stds[i] > 0:
            ax3.scatter(avg_stds[i], r2_values[i], c=color, s=150, label=label, edgecolors='black', zorder=5)
            ax3.annotate(label.split('+')[0], (avg_stds[i], r2_values[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax3.set_xlabel('Average Uncertainty (σ)', fontsize=11)
    ax3.set_ylabel('R² Score', fontsize=11)
    ax3.set_title('R² vs Average Uncertainty', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # 4. 불확실성 분포
    ax4 = axes[1, 1]
    for method, label, color in zip(methods[:4], labels[:4], colors[:4]):
        std = stds.get(f'{method}_std')
        if std is not None:
            ax4.hist(std[test_mask], bins=30, alpha=0.5, color=color, label=label, density=True)

    ax4.set_xlabel('Uncertainty (σ)', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
    ax4.set_title('Uncertainty Distribution', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Fold {fold_idx} - Uncertainty Analysis (with BLR)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(output_dir / f'fold_{fold_idx}' / 'uncertainty_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_chart(results_df, output_dir):
    """Create summary comparison bar chart"""
    methods = ['mfgp', 'joint', 'gradient_scaling', 'sequential', 'progressive', 'curriculum', 'two_stage_joint']
    labels = ['MFGP\n(Baseline)', 'Joint\n+BLR', 'Gradient\nScaling+BLR', 'Sequential\n+BLR', 'Progressive\n+BLR', 'Curriculum\n+BLR', 'Two-Stage\n+BLR']
    colors = ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#d62728', '#8c564b', '#17becf']

    r2_means = []
    r2_stds = []
    std_means = []

    for method in methods:
        r2_col = f'{method}_r2'
        std_col = f'{method}_avg_std'
        if r2_col in results_df.columns and results_df[r2_col].notna().any():
            r2_means.append(results_df[r2_col].mean())
            r2_stds.append(results_df[r2_col].std())
        else:
            r2_means.append(0)
            r2_stds.append(0)

        if std_col in results_df.columns and results_df[std_col].notna().any():
            std_means.append(results_df[std_col].mean())
        else:
            std_means.append(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # R² comparison
    x_pos = np.arange(len(methods))
    bars = ax1.bar(x_pos, r2_means, yerr=r2_stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title(f'HF Prediction R² ({len(results_df)}-fold)\nDNN Methods with BLR Uncertainty', fontsize=14)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylim(-0.5, 1.0)

    for bar, mean in zip(bars, r2_means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Average uncertainty comparison
    bars2 = ax2.bar(x_pos, std_means, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel('Average Uncertainty (σ)', fontsize=12)
    ax2.set_title(f'Average Predicted Uncertainty ({len(results_df)}-fold)', fontsize=14)

    for bar, mean in zip(bars2, std_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'summary_all_methods_blr.png', dpi=150, bbox_inches='tight')
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
    output_dir = Path(__file__).parent / 'visualizations' / f'{timestamp}_all_6methods_with_BLR'
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
            mfgp_avg_std = np.mean(mfgp_std[test_idx])
            print(f"        RMSE={mfgp_rmse:.4f}, R²={mfgp_r2:.4f}, Avg σ={mfgp_avg_std:.4f}")
            fold_results.update({'mfgp_rmse': mfgp_rmse, 'mfgp_r2': mfgp_r2, 'mfgp_avg_std': mfgp_avg_std})
            preds['mfgp_pred'] = mfgp_pred
            stds_dict['mfgp_std'] = mfgp_std
        else:
            print(f"        Not available")
            fold_results.update({'mfgp_rmse': None, 'mfgp_r2': None, 'mfgp_avg_std': None})

        # 2-7. All 6 methods with BLR
        for i, method in enumerate(methods, 2):
            method_info = best_params.get(method, {})
            params = method_info.get('params', {})
            bo_r2 = method_info.get('r2', 0)

            print(f"\n  [{i}/7] {method} + BLR (BO R²={bo_r2:.4f})...")

            try:
                pred, std = create_and_train_model_with_blr(method, data, device, seed, params)
                rmse, r2 = calc_metrics(y_test_hf, pred[test_idx])
                avg_std = np.mean(std[test_idx])
                print(f"        RMSE={rmse:.4f}, R²={r2:.4f}, Avg σ={avg_std:.4f}")
                fold_results.update({f'{method}_rmse': rmse, f'{method}_r2': r2, f'{method}_avg_std': avg_std})
                preds[f'{method}_pred'] = pred
                stds_dict[f'{method}_std'] = std
            except Exception as e:
                print(f"        Error: {e}")
                import traceback
                traceback.print_exc()
                fold_results.update({f'{method}_rmse': None, f'{method}_r2': None, f'{method}_avg_std': None})

        # Visualization
        fold_dir = create_7panel_visualization(fold_idx, seed, data, preds, stds_dict, param_space, output_dir, best_params)
        create_uncertainty_comparison(fold_idx, seed, data, preds, stds_dict, output_dir)
        print(f"\n  Visualization saved: {fold_dir}")

        results_list.append(fold_results)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY (with BLR)")
    print('='*60)

    import pandas as pd
    df = pd.DataFrame(results_list)

    print(f"\n{'Model':<25} {'Avg RMSE':<12} {'Avg R²':<12} {'Avg σ':<12}")
    print('-' * 70)

    all_methods = ['mfgp'] + methods
    for method in all_methods:
        rmse_col = f'{method}_rmse'
        r2_col = f'{method}_r2'
        std_col = f'{method}_avg_std'
        if r2_col in df.columns and df[r2_col].notna().any():
            r2_mean = df[r2_col].mean()
            rmse_mean = df[rmse_col].mean()
            std_mean = df[std_col].mean() if std_col in df.columns else 0
            print(f"{method:<25} {rmse_mean:.4f}       {r2_mean:.4f}       {std_mean:.4f}")

    # Save results
    df.to_csv(output_dir / 'results_summary_blr.csv', index=False)
    print(f"\nResults saved to: {output_dir / 'results_summary_blr.csv'}")

    # Create summary chart
    create_summary_chart(df, output_dir)
    print(f"Summary chart saved: {output_dir / 'summary_all_methods_blr.png'}")


if __name__ == '__main__':
    main()
