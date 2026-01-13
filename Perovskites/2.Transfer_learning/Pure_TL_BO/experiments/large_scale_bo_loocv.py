#!/usr/bin/env python3
"""
Large-Scale Bayesian Optimization with Leave-One-Out CV (No Data Leakage)

데이터 유출 방지:
- 하이퍼파라미터 최적화: HF 9개에 대한 LOO-CV로 validation
- 최종 평가: 별도의 test set (하이퍼파라미터 선택에 사용 안 함)

LOO-CV 방식:
- HF 9개 중 1개를 validation으로, 나머지 8개로 훈련
- 9번 반복하여 평균 validation R² 계산
- 이 값으로 하이퍼파라미터 최적화

Author: Claude Code
Date: 2025-12-11
"""

import sys
import os
from pathlib import Path

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.samplers import TPESampler
import json
import pickle
from datetime import datetime
from typing import Dict, List
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# Configuration
# ============================================================================

N_LOFI = 72
N_HIFI = 9
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]


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
                    'names': [org, cat, ani]
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

    # 전체 데이터 (최종 테스트용)
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


TRAIN_FUNCS = {
    'joint': train_joint,
    'gradient_scaling': train_gradient_scaling,
    'sequential': train_sequential,
    'progressive': train_progressive,
    'curriculum': train_curriculum,
    'two_stage_joint': train_two_stage_joint
}


# ============================================================================
# Leave-One-Out Cross Validation for HF data
# ============================================================================

def loocv_evaluate(method, data, arch_params, train_params, device, seed):
    """
    HF 데이터에 대한 Leave-One-Out Cross Validation

    HF 9개 중 1개를 validation으로, 나머지 8개로 훈련
    9번 반복하여 validation 예측 수집 → R² 계산

    데이터 유출 없음:
    - 각 fold에서 validation sample은 훈련에 사용되지 않음
    - LF 데이터는 전체 72개 사용 (HF와 독립)
    """
    X_hf = data['X_high']
    y_hf = data['y_high']
    X_lf = data['X_low']
    y_lf = data['y_low']

    n_hf = len(X_hf)
    val_preds = np.zeros(n_hf)

    for i in range(n_hf):
        # LOO split: i번째를 validation으로
        val_idx = i
        train_idx = [j for j in range(n_hf) if j != i]

        X_hf_train = X_hf[train_idx]
        y_hf_train = y_hf[train_idx]
        X_hf_val = X_hf[val_idx:val_idx+1]

        # 모델 생성 (매 fold마다 새로 생성)
        set_seeds(seed + i)  # fold마다 다른 시드
        model = MultiFidelityNetwork(
            input_dim=3,
            lf_hidden=arch_params['lf_hidden'],
            hf_hidden=arch_params['hf_hidden'],
            lf_layers=arch_params['lf_layers'],
            hf_layers=arch_params['hf_layers'],
            dropout=arch_params['dropout'],
            activation=arch_params['activation'],
            residual_mode=True
        )
        model.to(device)

        # 훈련 (LF 전체 + HF train 8개)
        train_func = TRAIN_FUNCS[method]
        train_func(model, X_lf, y_lf, X_hf_train, y_hf_train, train_params, device)

        # Validation 예측 (1개)
        model.eval()
        X_val_t = torch.tensor(X_hf_val, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_val_pred = model.forward_hf(X_val_t).cpu().numpy().flatten()

        val_preds[i] = y_val_pred[0]

        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    # LOO-CV R² 계산
    loocv_r2 = r2_score(y_hf, val_preds)
    loocv_rmse = np.sqrt(mean_squared_error(y_hf, val_preds))

    return loocv_r2, loocv_rmse, val_preds


# ============================================================================
# Optuna Objective with LOO-CV
# ============================================================================

def create_objective_loocv(lookup, all_combinations, n_outer_folds=10, device='cpu'):
    """
    LOO-CV 기반 Optuna objective function

    Outer loop: 10개 시드로 데이터 생성
    Inner loop: 각 시드에서 HF 9개에 대한 LOO-CV

    하이퍼파라미터 선택 기준: LOO-CV R² 평균
    """

    def objective(trial):
        # 방법론 선택
        method = trial.suggest_categorical('method', [
            'joint', 'gradient_scaling', 'sequential',
            'progressive', 'curriculum', 'two_stage_joint'
        ])

        # 아키텍처 파라미터
        arch_params = {
            'lf_hidden': trial.suggest_int('lf_hidden', 32, 128, step=16),
            'hf_hidden': trial.suggest_int('hf_hidden', 32, 128, step=16),
            'lf_layers': trial.suggest_int('lf_layers', 1, 3),
            'hf_layers': trial.suggest_int('hf_layers', 1, 3),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5, step=0.1),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh'])
        }

        # 훈련 파라미터
        train_params = {
            'lf_lr': trial.suggest_float('lf_lr', 1e-4, 1e-2, log=True),
            'hf_lr': trial.suggest_float('hf_lr', 1e-5, 1e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        }

        # 방법론별 파라미터
        if method == 'joint':
            train_params['alpha'] = trial.suggest_float('alpha', 0.1, 0.9, step=0.1)
            train_params['epochs'] = trial.suggest_int('epochs', 100, 400, step=50)

        elif method == 'sequential':
            train_params['lf_epochs'] = trial.suggest_int('lf_epochs', 100, 300, step=50)
            train_params['hf_epochs'] = trial.suggest_int('hf_epochs', 50, 200, step=50)

        elif method == 'two_stage_joint':
            train_params['alpha_twostage'] = trial.suggest_float('alpha_twostage', 0.3, 0.9, step=0.1)
            train_params['lf_warmup_epochs'] = trial.suggest_int('lf_warmup_epochs', 50, 150, step=25)
            train_params['joint_epochs'] = trial.suggest_int('joint_epochs', 50, 200, step=50)

        elif method == 'curriculum':
            train_params['alpha_start'] = trial.suggest_float('alpha_start', 0.0, 0.3, step=0.1)
            train_params['alpha_end'] = trial.suggest_float('alpha_end', 0.7, 0.95, step=0.05)
            train_params['curriculum_epochs'] = trial.suggest_int('curriculum_epochs', 150, 400, step=50)
            train_params['curriculum_schedule'] = trial.suggest_categorical('curriculum_schedule',
                                                                            ['linear', 'cosine'])

        elif method == 'progressive':
            train_params['prog_lf_epochs'] = trial.suggest_int('prog_lf_epochs', 100, 300, step=50)
            train_params['hf_epochs_per_phase'] = trial.suggest_int('hf_epochs_per_phase', 25, 100, step=25)
            train_params['lr_decay_per_phase'] = trial.suggest_float('lr_decay_per_phase', 0.5, 0.9, step=0.1)

        elif method == 'gradient_scaling':
            train_params['gs_epochs'] = trial.suggest_int('gs_epochs', 100, 400, step=50)
            train_params['gradient_scale_lf'] = trial.suggest_float('gradient_scale_lf', 0.5, 2.0, step=0.25)
            train_params['gradient_scale_hf'] = trial.suggest_float('gradient_scale_hf', 0.5, 2.0, step=0.25)

        # Outer CV: 여러 시드로 데이터 생성
        loocv_r2_scores = []

        for fold_idx, seed in enumerate(SEEDS[:n_outer_folds]):
            data = generate_data(lookup, all_combinations, n_lofi=N_LOFI, n_hifi=N_HIFI, seed=seed)

            try:
                # Inner CV: LOO-CV on HF data
                loocv_r2, _, _ = loocv_evaluate(method, data, arch_params, train_params, device, seed)
                loocv_r2_scores.append(loocv_r2)
            except Exception as e:
                print(f"Error in fold {fold_idx}: {e}")
                loocv_r2_scores.append(-1.0)

        mean_loocv_r2 = np.mean(loocv_r2_scores)

        # Pruning
        trial.report(mean_loocv_r2, 0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return mean_loocv_r2

    return objective


# ============================================================================
# Main
# ============================================================================

def run_bo_loocv(n_trials=200, n_outer_folds=10, device='cpu', save_dir='results/bo_loocv'):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("="*70)
    print("Bayesian Optimization with Leave-One-Out CV (No Data Leakage)")
    print("="*70)
    print(f"Trials: {n_trials}")
    print(f"Outer folds: {n_outer_folds}")
    print(f"LOO-CV: 9 HF samples (8 train, 1 val) x 9 folds")
    print(f"Device: {device}")
    print("="*70)

    print("\nLoading data...")
    lookup, all_combinations, _ = load_base_data()
    print(f"Total compositions: {len(all_combinations)}")

    # Optuna Study
    sampler = TPESampler(seed=42, multivariate=True)
    study = optuna.create_study(
        study_name=f"bo_loocv_{timestamp}",
        direction='maximize',
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20)
    )

    objective = create_objective_loocv(lookup, all_combinations, n_outer_folds, device)

    print(f"\nStarting optimization...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, gc_after_trial=True)

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nBest LOO-CV R²: {study.best_value:.4f}")
    print(f"\nBest Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save
    results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': n_trials,
        'n_outer_folds': n_outer_folds,
        'timestamp': timestamp,
        'validation_method': 'LOO-CV on HF 9 samples',
        'all_trials': [
            {'number': t.number, 'value': t.value, 'params': t.params, 'state': str(t.state)}
            for t in study.trials
        ]
    }

    results_path = os.path.join(save_dir, f'bo_loocv_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    # Method statistics
    print("\n" + "="*70)
    print("METHOD STATISTICS")
    print("="*70)

    method_results = {}
    for trial in study.trials:
        if trial.value is not None and trial.state == optuna.trial.TrialState.COMPLETE:
            method = trial.params.get('method', 'unknown')
            if method not in method_results:
                method_results[method] = []
            method_results[method].append(trial.value)

    for method, values in sorted(method_results.items()):
        print(f"\n{method}:")
        print(f"  Count: {len(values)}, Mean R²: {np.mean(values):.4f}, Max R²: {np.max(values):.4f}")

    return study, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=200)
    parser.add_argument('--n-outer-folds', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-dir', type=str, default='results/bo_loocv')
    args = parser.parse_args()

    run_bo_loocv(
        n_trials=args.n_trials,
        n_outer_folds=args.n_outer_folds,
        device=args.device,
        save_dir=args.save_dir
    )
