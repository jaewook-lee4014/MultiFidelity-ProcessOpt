#!/usr/bin/env python3
"""
6가지 Multi-Fidelity 학습 방법 + BLR with PCA Dimension Reduction

문제점:
- 기존 BLR: 96차원 특징 + 1 bias = 97 파라미터를 9개 샘플로 학습 → 과적합

해결책:
- PCA로 특징 차원을 N_HF-1 = 8 차원으로 축소
- 9개 샘플로 9개 파라미터(8차원 + bias) 학습 → 적절한 자유도

추가 개선:
- LF 데이터(72개)로 PCA 학습 → HF에 적용 (정보 전이)
- Regularization 강화 (alpha 증가)

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

# PCA 차원 설정: N_HF - 1 = 8 (9개 샘플로 9개 파라미터 학습)
PCA_DIM = 8

# ============================================================================
# Bayesian Linear Regression with Stronger Regularization
# ============================================================================

class BayesianLinearRegression:
    """
    베이지안 선형 회귀 모델 (강화된 정규화)

    작은 샘플에서 과적합을 방지하기 위해 alpha를 증가시킴
    """

    def __init__(self, alpha=10.0, beta=25.0):
        """
        Args:
            alpha: 가중치의 정밀도 (precision) - 높을수록 강한 정규화
            beta: 노이즈의 정밀도
        """
        self.alpha = alpha
        self.beta = beta
        self.mean = None
        self.cov = None
        self.fitted = False

    def fit(self, X, y):
        """베이지안 선형 회귀 학습"""
        X = np.asarray(X, dtype=np.float64)  # 수치 안정성을 위해 float64 사용
        y = np.asarray(y, dtype=np.float64).flatten()

        # 편향 항 추가
        X_with_bias = np.column_stack([np.ones(len(X)), X])

        # 사전 분포: w ~ N(0, α^(-1)I)
        n_features = X_with_bias.shape[1]
        S0_inv = self.alpha * np.eye(n_features)

        # 사후 분포 계산
        S_N_inv = S0_inv + self.beta * X_with_bias.T @ X_with_bias

        # 수치 안정성을 위한 정규화
        S_N_inv += 1e-6 * np.eye(n_features)

        self.cov = np.linalg.inv(S_N_inv)
        self.mean = self.beta * self.cov @ X_with_bias.T @ y

        self.fitted = True

    def predict(self, x):
        """단일 점에 대한 예측"""
        if not self.fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        x = np.asarray(x, dtype=np.float64).flatten()
        x_with_bias = np.concatenate([[1], x])

        mu = x_with_bias @ self.mean
        var = (1/self.beta) + x_with_bias @ self.cov @ x_with_bias

        return mu, max(var, 1e-8)  # 음수 분산 방지

    def predict_batch(self, X):
        """배치 예측"""
        X = np.asarray(X, dtype=np.float64)
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
# Model Runners with BLR + PCA
# ============================================================================

def create_and_train_model_with_blr_pca(method, data, device, seed, params, pca_dim=8):
    """
    모델 훈련 후 PCA + BLR을 붙여서 불확실성 추정

    핵심 개선:
    1. DNN 훈련 (기존과 동일)
    2. LF 데이터의 특징으로 PCA 학습 (72개 샘플 → 풍부한 정보)
    3. HF 특징에 PCA 적용하여 차원 축소 (96 → pca_dim)
    4. 축소된 특징으로 BLR 학습 (pca_dim + 1 파라미터를 9개 샘플로 학습)
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

    # Step 2: LF 데이터로 PCA 학습 (72개 샘플 활용)
    model.eval()
    X_lf_t = torch.tensor(data['X_low'], dtype=torch.float32).to(device)

    with torch.no_grad():
        y_lf_for_lf = model.forward_lf(X_lf_t)
        features_lf = model.extract_hf_features(X_lf_t, y_lf_for_lf).cpu().numpy()

    # StandardScaler + PCA 학습 (LF 데이터로)
    scaler = StandardScaler()
    features_lf_scaled = scaler.fit_transform(features_lf)

    pca = PCA(n_components=pca_dim)
    pca.fit(features_lf_scaled)

    # Step 3: HF 특징 추출 및 PCA 변환
    X_hf_t = torch.tensor(data['X_high'], dtype=torch.float32).to(device)

    with torch.no_grad():
        y_lf_for_hf = model.forward_lf(X_hf_t)
        features_hf = model.extract_hf_features(X_hf_t, y_lf_for_hf).cpu().numpy()

    features_hf_scaled = scaler.transform(features_hf)
    features_hf_pca = pca.transform(features_hf_scaled)  # 96 → pca_dim

    # Step 4: BLR 학습 (차원 축소된 특징 + HF 타겟)
    y_lf_pred_hf = y_lf_for_hf.cpu().numpy().flatten()
    y_hf_true = data['y_high']
    delta_hf = y_hf_true - y_lf_pred_hf  # HF 학습 데이터의 델타

    # 강화된 정규화로 BLR 학습
    # alpha를 높여서 prior를 강하게 하여 과적합 방지
    blr = BayesianLinearRegression(alpha=10.0, beta=25.0)
    blr.fit(features_hf_pca, delta_hf)

    # Step 5: 전체 데이터에 대해 예측
    X_all_t = torch.tensor(data['X_all'], dtype=torch.float32).to(device)

    with torch.no_grad():
        y_lf_all = model.forward_lf(X_all_t)
        features_all = model.extract_hf_features(X_all_t, y_lf_all).cpu().numpy()
        y_lf_all_np = y_lf_all.cpu().numpy().flatten()

    # PCA 변환
    features_all_scaled = scaler.transform(features_all)
    features_all_pca = pca.transform(features_all_scaled)

    # BLR로 delta 예측
    delta_pred, delta_var = blr.predict_batch(features_all_pca)

    # 최종 예측: y_hf = y_lf + delta
    y_hf_pred = y_lf_all_np + delta_pred
    std = np.sqrt(np.maximum(delta_var, 1e-8))

    return y_hf_pred, std


def create_and_train_model_dnn_only(method, data, device, seed, params):
    """
    DNN만 사용 (BLR 없이) - 원래 성능 비교용
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

    # DNN only: 고정 불확실성 (0.1)
    std = np.full_like(y_hf_pred, 0.1)

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

def create_comparison_visualization(fold_idx, seed, data, results, stds, param_space, output_dir):
    """DNN vs DNN+BLR+PCA 비교 시각화"""
    n_all = len(data['X_all'])
    hifi_idx = data['hifi_idx']
    y_all_hf = data['y_all']

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

    fold_dir = output_dir / f'fold_{fold_idx}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Main methods to compare
    methods = ['joint', 'gradient_scaling', 'sequential']
    colors = {'joint': '#2ca02c', 'gradient_scaling': '#9467bd', 'sequential': '#ff7f0e'}

    fig, axes = plt.subplots(len(methods), 2, figsize=(20, 5*len(methods)))

    test_mask = ~hf_train_mask
    y_test = y_sorted[test_mask]

    for row, method in enumerate(methods):
        # DNN only
        ax1 = axes[row, 0]
        pred_dnn = results.get(f'{method}_dnn_pred')
        if pred_dnn is not None:
            pred_sorted = pred_dnn[sort_idx]
            ax1.plot(x_axis, pred_sorted, color=colors[method], linewidth=0.8, alpha=0.7)
            ax1.scatter(x_axis[~hf_train_mask], y_sorted[~hf_train_mask], c='black', s=15, alpha=0.6)
            ax1.scatter(x_axis[hf_train_mask], y_sorted[hf_train_mask], c='red', s=100, marker='*', edgecolors='darkred')

            rmse, r2 = calc_metrics(y_test, pred_sorted[test_mask])
            ax1.set_title(f'{method.replace("_", " ").title()} - DNN Only\nRMSE={rmse:.4f}, R²={r2:.4f}', fontsize=12)
        ax1.set_ylabel('Bandgap (eV)')
        ax1.grid(True, alpha=0.3)

        # DNN + BLR + PCA
        ax2 = axes[row, 1]
        pred_blr = results.get(f'{method}_blr_pred')
        std_blr = stds.get(f'{method}_blr_std')
        if pred_blr is not None and std_blr is not None:
            pred_sorted = pred_blr[sort_idx]
            std_sorted = std_blr[sort_idx]

            ax2.fill_between(x_axis, pred_sorted - 2*std_sorted, pred_sorted + 2*std_sorted,
                            alpha=0.3, color=colors[method])
            ax2.plot(x_axis, pred_sorted, color=colors[method], linewidth=0.8, alpha=0.7)
            ax2.scatter(x_axis[~hf_train_mask], y_sorted[~hf_train_mask], c='black', s=15, alpha=0.6)
            ax2.scatter(x_axis[hf_train_mask], y_sorted[hf_train_mask], c='red', s=100, marker='*', edgecolors='darkred')

            rmse, r2 = calc_metrics(y_test, pred_sorted[test_mask])
            avg_std = np.mean(std_sorted[test_mask])
            ax2.set_title(f'{method.replace("_", " ").title()} + BLR + PCA\nRMSE={rmse:.4f}, R²={r2:.4f}, Avg σ={avg_std:.4f}', fontsize=12)
        ax2.set_ylabel('Bandgap (eV)')
        ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Fold {fold_idx} (seed={seed}) - DNN vs DNN+BLR+PCA(dim={PCA_DIM})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(fold_dir / 'comparison_dnn_vs_blr_pca.png', dpi=150, bbox_inches='tight')
    plt.close()

    return fold_dir


def create_summary_chart(results_df, output_dir):
    """Summary comparison chart"""
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
    bars2 = ax1.bar(x_pos + width/2, blr_r2, width, yerr=blr_r2_std, label='DNN+BLR+PCA', color='coral', alpha=0.8, capsize=3)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('R² Score')
    ax1.set_title(f'R² Comparison ({len(results_df)}-fold CV)')
    ax1.legend()
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
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
    bars2 = ax2.bar(x_pos + width/2, blr_rmse, width, label='DNN+BLR+PCA', color='coral', alpha=0.8)

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
    ax3.set_title('Uncertainty Estimation (BLR+PCA)')
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
    print(f"PCA dimension: {PCA_DIM}")

    # Load BO best params for all methods
    print("\n" + "="*60)
    print("Loading BO Best Parameters...")
    print("="*60)
    best_params = load_all_best_params()

    lookup, all_combinations, param_space = load_base_data()
    print(f"\nTotal compositions: {len(all_combinations)}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / 'visualizations' / f'{timestamp}_blr_pca'
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
        print(f"\n  [1/{len(methods)*2 + 1}] MFGP...")
        mfgp_pred, mfgp_std = run_mfgp(data, device, seed)
        if mfgp_pred is not None:
            mfgp_rmse, mfgp_r2 = calc_metrics(y_test_hf, mfgp_pred[test_idx])
            mfgp_avg_std = np.mean(mfgp_std[test_idx])
            print(f"        RMSE={mfgp_rmse:.4f}, R²={mfgp_r2:.4f}, Avg σ={mfgp_avg_std:.4f}")
            fold_results.update({'mfgp_rmse': mfgp_rmse, 'mfgp_r2': mfgp_r2, 'mfgp_avg_std': mfgp_avg_std})
            preds['mfgp_pred'] = mfgp_pred
            stds_dict['mfgp_std'] = mfgp_std

        # 2-7. All 6 methods - DNN only vs DNN+BLR+PCA
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
                preds[f'{method}_dnn_pred'] = pred_dnn
            except Exception as e:
                print(f"        Error: {e}")

            # DNN + BLR + PCA
            print(f"\n  [{3 + i*2}/{len(methods)*2 + 1}] {method} + BLR + PCA(dim={PCA_DIM})...")
            try:
                pred_blr, std_blr = create_and_train_model_with_blr_pca(method, data, device, seed, params, pca_dim=PCA_DIM)
                rmse, r2 = calc_metrics(y_test_hf, pred_blr[test_idx])
                avg_std = np.mean(std_blr[test_idx])
                print(f"        RMSE={rmse:.4f}, R²={r2:.4f}, Avg σ={avg_std:.4f}")
                fold_results.update({f'{method}_blr_rmse': rmse, f'{method}_blr_r2': r2, f'{method}_blr_avg_std': avg_std})
                preds[f'{method}_blr_pred'] = pred_blr
                stds_dict[f'{method}_blr_std'] = std_blr
            except Exception as e:
                print(f"        Error: {e}")
                import traceback
                traceback.print_exc()

        # Visualization
        fold_dir = create_comparison_visualization(fold_idx, seed, data, preds, stds_dict, param_space, output_dir)
        print(f"\n  Visualization saved: {fold_dir}")

        results_list.append(fold_results)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    import pandas as pd
    df = pd.DataFrame(results_list)

    print(f"\n{'Model':<30} {'Avg RMSE':<12} {'Avg R²':<12} {'Avg σ':<12}")
    print('-' * 70)

    # MFGP
    if 'mfgp_r2' in df.columns:
        print(f"{'MFGP':<30} {df['mfgp_rmse'].mean():.4f}       {df['mfgp_r2'].mean():.4f}       {df['mfgp_avg_std'].mean():.4f}")

    print()
    for method in methods:
        # DNN only
        dnn_r2_col = f'{method}_dnn_r2'
        if dnn_r2_col in df.columns and df[dnn_r2_col].notna().any():
            print(f"{method + ' (DNN)':<30} {df[f'{method}_dnn_rmse'].mean():.4f}       {df[dnn_r2_col].mean():.4f}       {'N/A':<12}")

        # BLR + PCA
        blr_r2_col = f'{method}_blr_r2'
        if blr_r2_col in df.columns and df[blr_r2_col].notna().any():
            print(f"{method + ' + BLR+PCA':<30} {df[f'{method}_blr_rmse'].mean():.4f}       {df[blr_r2_col].mean():.4f}       {df[f'{method}_blr_avg_std'].mean():.4f}")

    # Save results
    df.to_csv(output_dir / 'results_blr_pca.csv', index=False)
    print(f"\nResults saved to: {output_dir / 'results_blr_pca.csv'}")

    # Create summary chart
    create_summary_chart(df, output_dir)
    print(f"Summary chart saved: {output_dir / 'summary_comparison.png'}")


if __name__ == '__main__':
    main()
