#!/usr/bin/env python3
"""
6가지 Multi-Fidelity 학습 방법 + Residual-Based Uncertainty (Strict Version)

데이터 유출 방지를 위한 엄격한 검증:
1. 각 fold에서 train/test 분리 명확
2. 불확실성 추정 시 테스트 데이터의 레이블 사용 안 함
3. 모든 학습은 오직 train 데이터만 사용

검증 포인트:
- DNN 훈련: LF 72개 + HF 9개 (train only)
- 불확실성 추정: HF 9개의 잔차로 base_std 계산 (train only)
- 테스트 시: 예측값과 불확실성만 출력 (test labels 미사용)

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
# Residual-Based Uncertainty (Strict - No Data Leakage)
# ============================================================================

class ResidualUncertaintyStrict:
    """
    엄격한 잔차 기반 불확실성 추정 (데이터 유출 없음)

    학습 단계:
    - HF train 데이터의 DNN 잔차로 base_std 계산
    - HF train 데이터 위치 저장

    예측 단계:
    - base_std와 거리만 사용 (테스트 레이블 미사용)
    """

    def __init__(self, distance_scale=2.0, min_std=0.1):
        self.distance_scale = distance_scale
        self.min_std = min_std
        self.X_train = None
        self.base_std = None
        self.fitted = False

    def fit(self, X_train, y_train, y_pred_train):
        """
        학습 (오직 train 데이터만 사용)

        Args:
            X_train: HF 학습 입력 (N_train, D) - train only
            y_train: HF 학습 타겟 (N_train,) - train only
            y_pred_train: DNN 예측 on train (N_train,) - train only
        """
        self.X_train = np.asarray(X_train, dtype=np.float32)

        # 잔차 계산 (train 데이터만)
        residuals = np.asarray(y_train) - np.asarray(y_pred_train)
        self.base_std = max(np.std(residuals), self.min_std)

        self.fitted = True

        # 검증 출력
        print(f"        [UQ] Fitted on {len(X_train)} train samples, base_std={self.base_std:.4f}")

    def predict_std(self, X_test):
        """
        불확실성 예측 (테스트 레이블 미사용)

        Args:
            X_test: 테스트 입력 (N_test, D)

        Returns:
            std: 예측 불확실성 (N_test,)
        """
        if not self.fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        X_test = np.asarray(X_test, dtype=np.float32)
        stds = []

        for x in X_test:
            # 학습 데이터와의 거리 계산 (레이블 미사용)
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            min_dist = np.min(distances)

            # 거리 기반 불확실성
            distance_factor = min_dist / self.distance_scale
            std = self.base_std * (1.0 + distance_factor)
            stds.append(std)

        return np.array(stds)


# ============================================================================
# Helper Functions
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
            best_params[method] = {'params': best_p, 'r2': best_value}
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
                    'names': [org, cat, ani]
                })

    return lookup, all_combinations, param_space


def generate_data(lookup, all_combinations, n_lofi=72, n_hifi=9, seed=42):
    """
    데이터 생성 - 명확한 train/test 분리
    """
    set_seeds(seed)
    rng = np.random.default_rng(seed)

    n_total = len(all_combinations)

    # LF train: 72개 랜덤 선택
    lofi_idx = rng.choice(n_total, size=n_lofi, replace=False)

    # HF train: 9개 랜덤 선택
    hifi_idx = rng.choice(n_total, size=n_hifi, replace=False)

    # LF train data
    X_low, y_low = [], []
    for idx in lofi_idx:
        c = all_combinations[idx]
        X_low.append(c['labels'])
        y_low.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_gga']))

    # HF train data
    X_high, y_high = [], []
    for idx in hifi_idx:
        c = all_combinations[idx]
        X_high.append(c['labels'])
        y_high.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_hse06']))

    # All data (for evaluation)
    X_all, y_all_hf, y_all_lf = [], [], []
    for c in all_combinations:
        X_all.append(c['labels'])
        y_all_hf.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_hse06']))
        y_all_lf.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_gga']))

    return {
        'X_low': np.array(X_low, dtype=np.float32),      # LF train input
        'y_low': np.array(y_low, dtype=np.float32),      # LF train target
        'X_high': np.array(X_high, dtype=np.float32),    # HF train input
        'y_high': np.array(y_high, dtype=np.float32),    # HF train target
        'X_all': np.array(X_all, dtype=np.float32),      # All inputs
        'y_all': np.array(y_all_hf, dtype=np.float32),   # All HF targets (for eval only)
        'y_all_lf': np.array(y_all_lf, dtype=np.float32),
        'hifi_idx': hifi_idx,  # HF train indices
        'lofi_idx': lofi_idx   # LF train indices
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
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.0, activation='relu'):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh() if activation == 'tanh' else nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        return self.out_layer(self.feature_net(x))


class HFNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.0, activation='relu', residual_mode=True):
        super().__init__()
        self.residual_mode = residual_mode
        layers = []
        in_dim = input_dim + 1
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh() if activation == 'tanh' else nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, y_lf):
        combined = torch.cat([x, y_lf], dim=-1)
        delta = self.out_layer(self.feature_net(combined))
        return y_lf + delta if self.residual_mode else delta


class MultiFidelityNetwork(nn.Module):
    def __init__(self, input_dim, lf_hidden, hf_hidden, lf_layers=2, hf_layers=2,
                 dropout=0.0, activation='relu', residual_mode=True):
        super().__init__()
        self.lf_network = LFNetwork(input_dim, lf_hidden, lf_layers, dropout, activation)
        self.hf_network = HFNetwork(input_dim, hf_hidden, hf_layers, dropout, activation, residual_mode)

    def forward_lf(self, x):
        return self.lf_network(x)

    def forward_hf(self, x, y_lf=None):
        if y_lf is None:
            y_lf = self.lf_network(x)
        return self.hf_network(x, y_lf)


# ============================================================================
# Training Functions
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
    for _ in range(epochs):
        optimizer.zero_grad()
        lf_loss = loss_fn(model.forward_lf(X_lf_t), y_lf_t)
        y_lf_for_hf = model.forward_lf(X_hf_t).detach()
        hf_loss = loss_fn(model.forward_hf(X_hf_t, y_lf_for_hf), y_hf_t)
        ((1 - alpha) * lf_loss + alpha * hf_loss).backward()
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
    for _ in range(epochs):
        lf_optimizer.zero_grad()
        (loss_fn(model.forward_lf(X_lf_t), y_lf_t) * gradient_scale_lf).backward()
        lf_optimizer.step()

        hf_optimizer.zero_grad()
        with torch.no_grad():
            y_lf_for_hf = model.forward_lf(X_hf_t)
        (loss_fn(model.forward_hf(X_hf_t, y_lf_for_hf), y_hf_t) * gradient_scale_hf).backward()
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
    for _ in range(lf_epochs):
        lf_optimizer.zero_grad()
        loss_fn(model.forward_lf(X_lf_t), y_lf_t).backward()
        lf_optimizer.step()

    for param in model.lf_network.parameters():
        param.requires_grad = False

    hf_optimizer = optim.Adam(model.hf_network.parameters(), lr=hf_lr, weight_decay=weight_decay)
    for _ in range(hf_epochs):
        hf_optimizer.zero_grad()
        with torch.no_grad():
            y_lf_for_hf = model.forward_lf(X_hf_t)
        loss_fn(model.forward_hf(X_hf_t, y_lf_for_hf), y_hf_t).backward()
        hf_optimizer.step()

    for param in model.lf_network.parameters():
        param.requires_grad = True


def train_progressive(model, X_lf, y_lf, X_hf, y_hf, params, device):
    lf_lr = params.get('lf_lr', 1e-3)
    hf_lr = params.get('hf_lr', 1e-4)
    lf_epochs = params.get('prog_lf_epochs', params.get('lf_epochs', 200))
    hf_epochs_per_phase = params.get('hf_epochs_per_phase', 50)
    weight_decay = params.get('weight_decay', 1e-4)
    lr_decay = params.get('lr_decay_per_phase', 0.7)

    loss_fn = nn.MSELoss()
    X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(device)
    y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(device)
    X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(device)
    y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(device)

    lf_optimizer = optim.Adam(model.lf_network.parameters(), lr=lf_lr, weight_decay=weight_decay)
    model.train()
    for _ in range(lf_epochs):
        lf_optimizer.zero_grad()
        loss_fn(model.forward_lf(X_lf_t), y_lf_t).backward()
        lf_optimizer.step()

    for param in model.parameters():
        param.requires_grad = False

    hf_layers = []
    for name, module in model.hf_network.named_children():
        if name == 'feature_net':
            for sub in module:
                if isinstance(sub, nn.Linear):
                    hf_layers.append(sub)
        elif name == 'out_layer':
            hf_layers.append(module)

    current_lr = hf_lr
    for layer in reversed(hf_layers):
        for param in layer.parameters():
            param.requires_grad = True
        trainable = [p for p in model.hf_network.parameters() if p.requires_grad]
        hf_opt = optim.Adam(trainable, lr=current_lr, weight_decay=weight_decay)
        for _ in range(hf_epochs_per_phase):
            hf_opt.zero_grad()
            with torch.no_grad():
                y_lf_for_hf = model.forward_lf(X_hf_t)
            loss_fn(model.forward_hf(X_hf_t, y_lf_for_hf), y_hf_t).backward()
            hf_opt.step()
        current_lr *= lr_decay

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
        else:
            alpha = alpha_start + (alpha_end - alpha_start) * progress

        optimizer.zero_grad()
        lf_loss = loss_fn(model.forward_lf(X_lf_t), y_lf_t)
        y_lf_for_hf = model.forward_lf(X_hf_t)
        hf_loss = loss_fn(model.forward_hf(X_hf_t, y_lf_for_hf), y_hf_t)
        ((1 - alpha) * lf_loss + alpha * hf_loss).backward()
        optimizer.step()


def train_two_stage_joint(model, X_lf, y_lf, X_hf, y_hf, params, device):
    alpha = params.get('alpha_twostage', params.get('alpha', 0.5))
    lf_lr = params.get('lf_lr', 1e-3)
    hf_lr = params.get('hf_lr', 1e-4)
    lf_warmup = params.get('lf_warmup_epochs', 100)
    joint_epochs = params.get('joint_epochs', 100)
    weight_decay = params.get('weight_decay', 1e-4)

    loss_fn = nn.MSELoss()
    X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(device)
    y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(device)
    X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(device)
    y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(device)

    lf_optimizer = optim.Adam(model.lf_network.parameters(), lr=lf_lr, weight_decay=weight_decay)
    model.train()
    for _ in range(lf_warmup):
        lf_optimizer.zero_grad()
        loss_fn(model.forward_lf(X_lf_t), y_lf_t).backward()
        lf_optimizer.step()

    joint_optimizer = optim.Adam([
        {'params': model.lf_network.parameters(), 'lr': lf_lr * 0.1, 'weight_decay': weight_decay},
        {'params': model.hf_network.parameters(), 'lr': hf_lr, 'weight_decay': weight_decay}
    ])

    for _ in range(joint_epochs):
        joint_optimizer.zero_grad()
        lf_loss = loss_fn(model.forward_lf(X_lf_t), y_lf_t)
        y_lf_for_hf = model.forward_lf(X_hf_t)
        hf_loss = loss_fn(model.forward_hf(X_hf_t, y_lf_for_hf), y_hf_t)
        ((1 - alpha) * lf_loss + alpha * hf_loss).backward()
        joint_optimizer.step()


# ============================================================================
# Model Runner (Strict - No Data Leakage)
# ============================================================================

def train_and_predict_strict(method, data, device, seed, params):
    """
    엄격한 훈련 및 예측 (데이터 유출 없음)

    단계:
    1. DNN 훈련 (LF train + HF train만 사용)
    2. HF train에 대한 DNN 예측 → 잔차 계산
    3. 불확실성 추정기 학습 (HF train 잔차만 사용)
    4. 전체 데이터 예측 (테스트 레이블 미사용)
    """
    set_seeds(seed)

    # 모델 생성
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

    # Step 1: DNN 훈련 (train 데이터만)
    train_funcs = {
        'joint': train_joint,
        'gradient_scaling': train_gradient_scaling,
        'sequential': train_sequential,
        'progressive': train_progressive,
        'curriculum': train_curriculum,
        'two_stage_joint': train_two_stage_joint
    }

    train_funcs[method](model, data['X_low'], data['y_low'], data['X_high'], data['y_high'], params, device)

    # Step 2: HF train에 대한 DNN 예측
    model.eval()
    X_hf_t = torch.tensor(data['X_high'], dtype=torch.float32).to(device)

    with torch.no_grad():
        dnn_pred_hf_train = model.forward_hf(X_hf_t).cpu().numpy().flatten()

    # Step 3: 불확실성 추정기 학습 (HF train만)
    uq = ResidualUncertaintyStrict(distance_scale=2.0, min_std=0.1)
    uq.fit(data['X_high'], data['y_high'], dnn_pred_hf_train)

    # Step 4: 전체 데이터 예측 (테스트 레이블 미사용)
    X_all_t = torch.tensor(data['X_all'], dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred_all = model.forward_hf(X_all_t).cpu().numpy().flatten()

    std_all = uq.predict_std(data['X_all'])

    return y_pred_all, std_all


def run_mfgp(data, device, seed):
    if not MFGP_AVAILABLE:
        return None, None
    set_seeds(seed)
    mfgp = MultiFidelityGP(input_dim=data['X_low'].shape[1], device=device)
    mfgp.fit(data['X_low'], data['y_low'], data['X_high'], data['y_high'])
    return mfgp.predict(data['X_all'], return_std=True)


# ============================================================================
# Visualization
# ============================================================================

def create_summary_chart(results_df, output_dir):
    methods = ['mfgp', 'joint', 'gradient_scaling', 'sequential', 'progressive', 'curriculum', 'two_stage_joint']
    labels = ['MFGP', 'Joint', 'GradScale', 'Seq', 'Prog', 'Curr', '2Stage']
    colors = ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#d62728', '#8c564b', '#17becf']

    r2_means = [results_df[f'{m}_r2'].mean() if f'{m}_r2' in results_df.columns else 0 for m in methods]
    r2_stds = [results_df[f'{m}_r2'].std() if f'{m}_r2' in results_df.columns else 0 for m in methods]
    std_means = [results_df[f'{m}_avg_std'].mean() if f'{m}_avg_std' in results_df.columns else 0 for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x_pos = np.arange(len(methods))
    bars = ax1.bar(x_pos, r2_means, yerr=r2_stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('R² Score')
    ax1.set_title(f'HF Prediction R² ({len(results_df)}-fold CV)\nDNN + Residual-Based UQ (Strict)')
    ax1.set_ylim(-0.5, 1.0)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for bar, mean in zip(bars, r2_means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{mean:.3f}',
                ha='center', fontsize=9, fontweight='bold')

    bars2 = ax2.bar(x_pos, std_means, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Average Uncertainty (σ)')
    ax2.set_title('Average Predicted Uncertainty')
    for bar, mean in zip(bars2, std_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{mean:.3f}',
                ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'summary_strict.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Data: {N_LOFI} LF + {N_HIFI} HF")
    print(f"Folds: {len(SEEDS)}")
    print("\n" + "="*70)
    print("STRICT MODE: No data leakage - all operations use train data only")
    print("="*70)

    print("\nLoading BO Best Parameters...")
    best_params = load_all_best_params()

    lookup, all_combinations, param_space = load_base_data()
    print(f"\nTotal compositions: {len(all_combinations)}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / 'visualizations' / f'{timestamp}_residual_uq_strict'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    results_list = []
    methods = ['joint', 'gradient_scaling', 'sequential', 'progressive', 'curriculum', 'two_stage_joint']

    for fold_idx, seed in enumerate(SEEDS, 1):
        print(f"\n{'='*70}")
        print(f"Fold {fold_idx}/{len(SEEDS)} (seed={seed})")
        print('='*70)

        data = generate_data(lookup, all_combinations, n_lofi=N_LOFI, n_hifi=N_HIFI, seed=seed)

        # Test mask: HF train이 아닌 모든 데이터
        test_mask = np.ones(len(data['X_all']), dtype=bool)
        test_mask[data['hifi_idx']] = False
        test_idx = np.where(test_mask)[0]

        # Test 레이블 (평가용으로만 사용 - 학습에는 절대 사용 안 함)
        y_test = data['y_all'][test_idx]

        print(f"  Train: {N_LOFI} LF + {N_HIFI} HF")
        print(f"  Test: {len(test_idx)} samples (labels used ONLY for evaluation)")

        fold_results = {'fold': fold_idx, 'seed': seed}

        # MFGP
        print(f"\n  [1/7] MFGP...")
        mfgp_pred, mfgp_std = run_mfgp(data, device, seed)
        if mfgp_pred is not None:
            mfgp_rmse, mfgp_r2 = calc_metrics(y_test, mfgp_pred[test_idx])
            mfgp_avg_std = np.mean(mfgp_std[test_idx])
            print(f"        RMSE={mfgp_rmse:.4f}, R²={mfgp_r2:.4f}, Avg σ={mfgp_avg_std:.4f}")
            fold_results.update({'mfgp_rmse': mfgp_rmse, 'mfgp_r2': mfgp_r2, 'mfgp_avg_std': mfgp_avg_std})

        # 6 Methods
        for i, method in enumerate(methods, 2):
            method_info = best_params.get(method, {})
            params = method_info.get('params', {})
            bo_r2 = method_info.get('r2', 0)

            print(f"\n  [{i}/7] {method} + ResidualUQ (BO R²={bo_r2:.4f})...")

            try:
                pred, std = train_and_predict_strict(method, data, device, seed, params)

                # 평가 (테스트 레이블은 여기서만 사용)
                rmse, r2 = calc_metrics(y_test, pred[test_idx])
                avg_std = np.mean(std[test_idx])

                print(f"        RMSE={rmse:.4f}, R²={r2:.4f}, Avg σ={avg_std:.4f}")
                fold_results.update({
                    f'{method}_rmse': rmse,
                    f'{method}_r2': r2,
                    f'{method}_avg_std': avg_std
                })
            except Exception as e:
                print(f"        Error: {e}")
                import traceback
                traceback.print_exc()

        results_list.append(fold_results)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY (Strict Mode - No Data Leakage)")
    print('='*70)

    import pandas as pd
    df = pd.DataFrame(results_list)

    print(f"\n{'Model':<25} {'Avg RMSE':<12} {'Avg R²':<12} {'Avg σ':<12}")
    print('-' * 70)

    for method in ['mfgp'] + methods:
        r2_col = f'{method}_r2'
        if r2_col in df.columns and df[r2_col].notna().any():
            print(f"{method:<25} {df[f'{method}_rmse'].mean():.4f}       "
                  f"{df[r2_col].mean():.4f}       {df[f'{method}_avg_std'].mean():.4f}")

    df.to_csv(output_dir / 'results_strict.csv', index=False)
    print(f"\nResults saved to: {output_dir / 'results_strict.csv'}")

    create_summary_chart(df, output_dir)
    print(f"Summary chart saved: {output_dir / 'summary_strict.png'}")


if __name__ == '__main__':
    main()
