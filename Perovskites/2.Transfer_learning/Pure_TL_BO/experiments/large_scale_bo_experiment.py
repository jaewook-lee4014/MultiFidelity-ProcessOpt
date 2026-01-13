#!/usr/bin/env python3
"""
Large-Scale Bayesian Optimization Experiment
방법론과 하이퍼파라미터를 동시에 탐색하는 종합 BO 실험

기존 프로젝트 설정 사용:
- LF 데이터: 72개 (GGA)
- HF 데이터: 9개 (HSE06)
- 10 Folds: 고정된 시드 [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
- Test: 전체 192개 조성에 대해 HF 예측 평가

핵심 문제 해결:
- Residual Learning: HF = LF + delta 구조로 높은 LF-HF 상관관계 활용

방법론:
1. joint: 기존 방식 (alpha 가중치)
2. sequential: LF 완전 학습 후 HF 학습 (별도 단계)
3. two_stage_joint: LF 먼저 학습 후 joint로 미세조정
4. curriculum: epoch에 따라 alpha 점진적 증가
5. progressive: 레이어별 점진적 해동
6. gradient_scaling: 독립적 gradient scaling

Author: Claude Code
Date: 2025-12-08 (Modified)
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
model_comparison_dir = current_dir.parent.parent / 'model_comparison'
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(model_comparison_dir))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.samplers import TPESampler
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# 기존 프로젝트 설정
# ============================================================================

N_LOFI = 72
N_HIFI = 9
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]  # 10 Folds


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_base_data():
    """기존 프로젝트의 데이터 로딩 함수"""
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
    """기존 프로젝트의 데이터 생성 함수"""
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

    # 전체 데이터 (테스트용)
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
# Multi-Fidelity 네트워크 구조
# ============================================================================

class LFNetwork(nn.Module):
    """Low-Fidelity 네트워크"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2,
                 dropout: float = 0.0, activation: str = 'relu'):
        super().__init__()
        self.activation_name = activation

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
    """High-Fidelity 네트워크 - Residual Learning (delta = HF - LF)"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2,
                 dropout: float = 0.0, activation: str = 'relu',
                 residual_mode: bool = True):
        super().__init__()
        self.activation_name = activation
        self.residual_mode = residual_mode

        # HF는 [X, y_LF]를 입력으로 받음
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
    """통합 Multi-Fidelity 네트워크"""
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

    def forward_both(self, x):
        y_lf = self.lf_network(x)
        y_hf = self.hf_network(x, y_lf)
        return y_lf, y_hf


# ============================================================================
# 다양한 학습 방법론
# ============================================================================

class TrainingMethod:
    def __init__(self, model: MultiFidelityNetwork, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)

    def train(self, X_lf, y_lf, X_hf, y_hf, params: Dict) -> Dict:
        raise NotImplementedError


class JointTraining(TrainingMethod):
    """Joint Training (alpha 가중치)"""
    def train(self, X_lf, y_lf, X_hf, y_hf, params: Dict) -> Dict:
        alpha = params.get('alpha', 0.5)
        lf_lr = params.get('lf_lr', 1e-3)
        hf_lr = params.get('hf_lr', 1e-4)
        epochs = params.get('epochs', 200)
        weight_decay = params.get('weight_decay', 1e-4)

        optimizer = optim.Adam([
            {'params': self.model.lf_network.parameters(), 'lr': lf_lr, 'weight_decay': weight_decay},
            {'params': self.model.hf_network.parameters(), 'lr': hf_lr, 'weight_decay': weight_decay}
        ])

        loss_fn = nn.MSELoss()

        X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(self.device)
        y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(self.device)
        X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(self.device)
        y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(self.device)

        losses = {'lf': [], 'hf': [], 'total': []}

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            y_lf_pred = self.model.forward_lf(X_lf_t)
            lf_loss = loss_fn(y_lf_pred, y_lf_t)

            y_lf_for_hf = self.model.forward_lf(X_hf_t).detach()
            y_hf_pred = self.model.forward_hf(X_hf_t, y_lf_for_hf)
            hf_loss = loss_fn(y_hf_pred, y_hf_t)

            total_loss = (1 - alpha) * lf_loss + alpha * hf_loss
            total_loss.backward()
            optimizer.step()

            losses['lf'].append(lf_loss.item())
            losses['hf'].append(hf_loss.item())
            losses['total'].append(total_loss.item())

        return losses


class SequentialTraining(TrainingMethod):
    """Sequential Training: LF 완전 학습 후 HF 학습"""
    def train(self, X_lf, y_lf, X_hf, y_hf, params: Dict) -> Dict:
        lf_lr = params.get('lf_lr', 1e-3)
        hf_lr = params.get('hf_lr', 1e-4)
        lf_epochs = params.get('lf_epochs', 200)
        hf_epochs = params.get('hf_epochs', 100)
        weight_decay = params.get('weight_decay', 1e-4)

        loss_fn = nn.MSELoss()

        X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(self.device)
        y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(self.device)
        X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(self.device)
        y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(self.device)

        losses = {'lf': [], 'hf': [], 'total': []}

        # Stage 1: LF 학습
        lf_optimizer = optim.Adam(self.model.lf_network.parameters(), lr=lf_lr, weight_decay=weight_decay)

        self.model.train()
        for epoch in range(lf_epochs):
            lf_optimizer.zero_grad()
            y_lf_pred = self.model.forward_lf(X_lf_t)
            lf_loss = loss_fn(y_lf_pred, y_lf_t)
            lf_loss.backward()
            lf_optimizer.step()
            losses['lf'].append(lf_loss.item())

        # Stage 2: LF 동결, HF 학습
        for param in self.model.lf_network.parameters():
            param.requires_grad = False

        hf_optimizer = optim.Adam(self.model.hf_network.parameters(), lr=hf_lr, weight_decay=weight_decay)

        for epoch in range(hf_epochs):
            hf_optimizer.zero_grad()

            with torch.no_grad():
                y_lf_for_hf = self.model.forward_lf(X_hf_t)

            y_hf_pred = self.model.forward_hf(X_hf_t, y_lf_for_hf)
            hf_loss = loss_fn(y_hf_pred, y_hf_t)
            hf_loss.backward()
            hf_optimizer.step()
            losses['hf'].append(hf_loss.item())

        for param in self.model.lf_network.parameters():
            param.requires_grad = True

        final_lf_loss = losses['lf'][-1] if losses['lf'] else 0
        final_hf_loss = losses['hf'][-1] if losses['hf'] else 0
        losses['total'] = [final_lf_loss + final_hf_loss]

        return losses


class TwoStageJointTraining(TrainingMethod):
    """Two-Stage Joint Training: LF 먼저 학습 후 Joint 미세조정"""
    def train(self, X_lf, y_lf, X_hf, y_hf, params: Dict) -> Dict:
        alpha = params.get('alpha', 0.5)
        lf_lr = params.get('lf_lr', 1e-3)
        hf_lr = params.get('hf_lr', 1e-4)
        lf_warmup_epochs = params.get('lf_warmup_epochs', 100)
        joint_epochs = params.get('joint_epochs', 100)
        weight_decay = params.get('weight_decay', 1e-4)

        loss_fn = nn.MSELoss()

        X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(self.device)
        y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(self.device)
        X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(self.device)
        y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(self.device)

        losses = {'lf': [], 'hf': [], 'total': []}

        # Stage 1: LF Warmup
        lf_optimizer = optim.Adam(self.model.lf_network.parameters(), lr=lf_lr, weight_decay=weight_decay)

        self.model.train()
        for epoch in range(lf_warmup_epochs):
            lf_optimizer.zero_grad()
            y_lf_pred = self.model.forward_lf(X_lf_t)
            lf_loss = loss_fn(y_lf_pred, y_lf_t)
            lf_loss.backward()
            lf_optimizer.step()
            losses['lf'].append(lf_loss.item())

        # Stage 2: Joint Training
        joint_lf_lr = lf_lr * 0.1
        joint_optimizer = optim.Adam([
            {'params': self.model.lf_network.parameters(), 'lr': joint_lf_lr, 'weight_decay': weight_decay},
            {'params': self.model.hf_network.parameters(), 'lr': hf_lr, 'weight_decay': weight_decay}
        ])

        for epoch in range(joint_epochs):
            joint_optimizer.zero_grad()

            y_lf_pred = self.model.forward_lf(X_lf_t)
            lf_loss = loss_fn(y_lf_pred, y_lf_t)

            y_lf_for_hf = self.model.forward_lf(X_hf_t)
            y_hf_pred = self.model.forward_hf(X_hf_t, y_lf_for_hf)
            hf_loss = loss_fn(y_hf_pred, y_hf_t)

            total_loss = (1 - alpha) * lf_loss + alpha * hf_loss
            total_loss.backward()
            joint_optimizer.step()

            losses['lf'].append(lf_loss.item())
            losses['hf'].append(hf_loss.item())
            losses['total'].append(total_loss.item())

        return losses


class CurriculumTraining(TrainingMethod):
    """Curriculum Training: alpha를 점진적으로 증가"""
    def train(self, X_lf, y_lf, X_hf, y_hf, params: Dict) -> Dict:
        alpha_start = params.get('alpha_start', 0.1)
        alpha_end = params.get('alpha_end', 0.9)
        lf_lr = params.get('lf_lr', 1e-3)
        hf_lr = params.get('hf_lr', 1e-4)
        epochs = params.get('epochs', 200)
        weight_decay = params.get('weight_decay', 1e-4)
        curriculum_schedule = params.get('curriculum_schedule', 'linear')

        loss_fn = nn.MSELoss()

        X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(self.device)
        y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(self.device)
        X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(self.device)
        y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(self.device)

        optimizer = optim.Adam([
            {'params': self.model.lf_network.parameters(), 'lr': lf_lr, 'weight_decay': weight_decay},
            {'params': self.model.hf_network.parameters(), 'lr': hf_lr, 'weight_decay': weight_decay}
        ])

        losses = {'lf': [], 'hf': [], 'total': [], 'alpha': []}

        self.model.train()
        for epoch in range(epochs):
            progress = epoch / max(epochs - 1, 1)
            if curriculum_schedule == 'cosine':
                alpha = alpha_start + (alpha_end - alpha_start) * (1 - np.cos(progress * np.pi)) / 2
            elif curriculum_schedule == 'step':
                if progress < 0.33:
                    alpha = alpha_start
                elif progress < 0.66:
                    alpha = (alpha_start + alpha_end) / 2
                else:
                    alpha = alpha_end
            else:  # linear
                alpha = alpha_start + (alpha_end - alpha_start) * progress

            optimizer.zero_grad()

            y_lf_pred = self.model.forward_lf(X_lf_t)
            lf_loss = loss_fn(y_lf_pred, y_lf_t)

            y_lf_for_hf = self.model.forward_lf(X_hf_t)
            y_hf_pred = self.model.forward_hf(X_hf_t, y_lf_for_hf)
            hf_loss = loss_fn(y_hf_pred, y_hf_t)

            total_loss = (1 - alpha) * lf_loss + alpha * hf_loss
            total_loss.backward()
            optimizer.step()

            losses['lf'].append(lf_loss.item())
            losses['hf'].append(hf_loss.item())
            losses['total'].append(total_loss.item())
            losses['alpha'].append(alpha)

        return losses


class ProgressiveUnfreezingTraining(TrainingMethod):
    """Progressive Unfreezing: 레이어별 점진적 해동"""
    def train(self, X_lf, y_lf, X_hf, y_hf, params: Dict) -> Dict:
        lf_lr = params.get('lf_lr', 1e-3)
        hf_lr = params.get('hf_lr', 1e-4)
        lf_epochs = params.get('lf_epochs', 200)
        hf_epochs_per_phase = params.get('hf_epochs_per_phase', 50)
        weight_decay = params.get('weight_decay', 1e-4)
        lr_decay_per_phase = params.get('lr_decay_per_phase', 0.7)

        loss_fn = nn.MSELoss()

        X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(self.device)
        y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(self.device)
        X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(self.device)
        y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(self.device)

        losses = {'lf': [], 'hf': [], 'total': []}

        # Stage 1: LF 완전 학습
        lf_optimizer = optim.Adam(self.model.lf_network.parameters(), lr=lf_lr, weight_decay=weight_decay)

        self.model.train()
        for epoch in range(lf_epochs):
            lf_optimizer.zero_grad()
            y_lf_pred = self.model.forward_lf(X_lf_t)
            lf_loss = loss_fn(y_lf_pred, y_lf_t)
            lf_loss.backward()
            lf_optimizer.step()
            losses['lf'].append(lf_loss.item())

        # Stage 2: Progressive Unfreezing for HF
        for param in self.model.lf_network.parameters():
            param.requires_grad = False
        for param in self.model.hf_network.parameters():
            param.requires_grad = False

        hf_layers = []
        for name, module in self.model.hf_network.named_children():
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

            trainable_params = [p for p in self.model.hf_network.parameters() if p.requires_grad]
            hf_optimizer = optim.Adam(trainable_params, lr=current_lr, weight_decay=weight_decay)

            for epoch in range(hf_epochs_per_phase):
                hf_optimizer.zero_grad()

                with torch.no_grad():
                    y_lf_for_hf = self.model.forward_lf(X_hf_t)

                y_hf_pred = self.model.forward_hf(X_hf_t, y_lf_for_hf)
                hf_loss = loss_fn(y_hf_pred, y_hf_t)
                hf_loss.backward()
                hf_optimizer.step()
                losses['hf'].append(hf_loss.item())

            current_lr *= lr_decay_per_phase

        for param in self.model.parameters():
            param.requires_grad = True

        final_lf_loss = losses['lf'][-1] if losses['lf'] else 0
        final_hf_loss = losses['hf'][-1] if losses['hf'] else 0
        losses['total'] = [final_lf_loss + final_hf_loss]

        return losses


class GradientScalingTraining(TrainingMethod):
    """Gradient Scaling: LF/HF gradient를 독립적으로 스케일링"""
    def train(self, X_lf, y_lf, X_hf, y_hf, params: Dict) -> Dict:
        lf_lr = params.get('lf_lr', 1e-3)
        hf_lr = params.get('hf_lr', 1e-4)
        epochs = params.get('epochs', 200)
        weight_decay = params.get('weight_decay', 1e-4)
        gradient_scale_lf = params.get('gradient_scale_lf', 1.0)
        gradient_scale_hf = params.get('gradient_scale_hf', 1.0)

        loss_fn = nn.MSELoss()

        X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(self.device)
        y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(self.device)
        X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(self.device)
        y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(self.device)

        lf_optimizer = optim.Adam(self.model.lf_network.parameters(), lr=lf_lr, weight_decay=weight_decay)
        hf_optimizer = optim.Adam(self.model.hf_network.parameters(), lr=hf_lr, weight_decay=weight_decay)

        losses = {'lf': [], 'hf': [], 'total': []}

        self.model.train()
        for epoch in range(epochs):
            # LF 업데이트
            lf_optimizer.zero_grad()
            y_lf_pred = self.model.forward_lf(X_lf_t)
            lf_loss = loss_fn(y_lf_pred, y_lf_t)
            scaled_lf_loss = lf_loss * gradient_scale_lf
            scaled_lf_loss.backward()
            lf_optimizer.step()

            # HF 업데이트
            hf_optimizer.zero_grad()
            with torch.no_grad():
                y_lf_for_hf = self.model.forward_lf(X_hf_t)
            y_hf_pred = self.model.forward_hf(X_hf_t, y_lf_for_hf)
            hf_loss = loss_fn(y_hf_pred, y_hf_t)
            scaled_hf_loss = hf_loss * gradient_scale_hf
            scaled_hf_loss.backward()
            hf_optimizer.step()

            losses['lf'].append(lf_loss.item())
            losses['hf'].append(hf_loss.item())
            losses['total'].append(lf_loss.item() + hf_loss.item())

        return losses


# ============================================================================
# 평가 함수
# ============================================================================

def evaluate_model(model: MultiFidelityNetwork, X_test: np.ndarray, y_lf_test: np.ndarray,
                   y_hf_test: np.ndarray, device: str) -> Dict:
    """모델 평가 - 예측된 y_lf 값을 사용하여 HF 예측 (공정한 평가)"""
    model.eval()

    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        # LF 예측
        y_lf_pred_t = model.forward_lf(X_t)
        y_lf_pred = y_lf_pred_t.cpu().numpy().flatten()
        # HF 예측: 모델이 예측한 y_lf 값 사용 (공정한 평가)
        y_hf_pred = model.forward_hf(X_t, y_lf_pred_t).cpu().numpy().flatten()

    results = {
        'lf_r2': r2_score(y_lf_test, y_lf_pred),
        'hf_r2': r2_score(y_hf_test, y_hf_pred),
        'lf_rmse': np.sqrt(mean_squared_error(y_lf_test, y_lf_pred)),
        'hf_rmse': np.sqrt(mean_squared_error(y_hf_test, y_hf_pred)),
        'lf_pred': y_lf_pred,
        'hf_pred': y_hf_pred
    }

    return results


# ============================================================================
# Optuna Objective Function
# ============================================================================

def create_objective(lookup, all_combinations, n_folds: int = 10, device: str = 'cpu'):
    """Optuna objective function 생성 - 기존 프로젝트 데이터셋 사용"""

    def objective(trial: optuna.Trial) -> float:
        # 방법론 선택
        method = trial.suggest_categorical('method', [
            'joint', 'sequential', 'two_stage_joint', 'curriculum',
            'progressive', 'gradient_scaling'
        ])

        # 공통 하이퍼파라미터
        lf_hidden = trial.suggest_int('lf_hidden', 32, 128, step=16)
        hf_hidden = trial.suggest_int('hf_hidden', 32, 128, step=16)
        lf_layers = trial.suggest_int('lf_layers', 1, 3)
        hf_layers = trial.suggest_int('hf_layers', 1, 3)
        dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
        activation = trial.suggest_categorical('activation', ['relu', 'tanh'])

        lf_lr = trial.suggest_float('lf_lr', 1e-4, 1e-2, log=True)
        hf_lr = trial.suggest_float('hf_lr', 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

        # 방법론별 파라미터
        params = {
            'lf_lr': lf_lr,
            'hf_lr': hf_lr,
            'weight_decay': weight_decay
        }

        if method == 'joint':
            params['alpha'] = trial.suggest_float('alpha', 0.1, 0.9, step=0.1)
            params['epochs'] = trial.suggest_int('epochs', 100, 400, step=50)

        elif method == 'sequential':
            params['lf_epochs'] = trial.suggest_int('lf_epochs', 100, 300, step=50)
            params['hf_epochs'] = trial.suggest_int('hf_epochs', 50, 200, step=50)

        elif method == 'two_stage_joint':
            params['alpha'] = trial.suggest_float('alpha_twostage', 0.3, 0.9, step=0.1)
            params['lf_warmup_epochs'] = trial.suggest_int('lf_warmup_epochs', 50, 150, step=25)
            params['joint_epochs'] = trial.suggest_int('joint_epochs', 50, 200, step=50)

        elif method == 'curriculum':
            params['alpha_start'] = trial.suggest_float('alpha_start', 0.0, 0.3, step=0.1)
            params['alpha_end'] = trial.suggest_float('alpha_end', 0.7, 0.95, step=0.05)
            params['epochs'] = trial.suggest_int('curriculum_epochs', 150, 400, step=50)
            params['curriculum_schedule'] = trial.suggest_categorical('curriculum_schedule',
                                                                       ['linear', 'cosine', 'step'])

        elif method == 'progressive':
            params['lf_epochs'] = trial.suggest_int('prog_lf_epochs', 100, 300, step=50)
            params['hf_epochs_per_phase'] = trial.suggest_int('hf_epochs_per_phase', 25, 100, step=25)
            params['lr_decay_per_phase'] = trial.suggest_float('lr_decay_per_phase', 0.5, 0.9, step=0.1)

        elif method == 'gradient_scaling':
            params['epochs'] = trial.suggest_int('gs_epochs', 100, 400, step=50)
            params['gradient_scale_lf'] = trial.suggest_float('gradient_scale_lf', 0.5, 2.0, step=0.25)
            params['gradient_scale_hf'] = trial.suggest_float('gradient_scale_hf', 0.5, 2.0, step=0.25)

        # 10-Fold Cross Validation (기존 시드 사용)
        hf_r2_scores = []

        for fold_idx, seed in enumerate(SEEDS[:n_folds]):
            # 기존 방식으로 데이터 생성
            data = generate_data(lookup, all_combinations, n_lofi=N_LOFI, n_hifi=N_HIFI, seed=seed)

            # 모델 생성
            model = MultiFidelityNetwork(
                input_dim=3,  # [organic, cation, anion]
                lf_hidden=lf_hidden,
                hf_hidden=hf_hidden,
                lf_layers=lf_layers,
                hf_layers=hf_layers,
                dropout=dropout,
                activation=activation,
                residual_mode=True  # Residual Learning 사용
            )

            # 학습 방법론 선택
            if method == 'joint':
                trainer = JointTraining(model, device)
            elif method == 'sequential':
                trainer = SequentialTraining(model, device)
            elif method == 'two_stage_joint':
                trainer = TwoStageJointTraining(model, device)
            elif method == 'curriculum':
                trainer = CurriculumTraining(model, device)
            elif method == 'progressive':
                trainer = ProgressiveUnfreezingTraining(model, device)
            elif method == 'gradient_scaling':
                trainer = GradientScalingTraining(model, device)

            # 학습 (LF 72개, HF 9개)
            try:
                trainer.train(data['X_low'], data['y_low'], data['X_high'], data['y_high'], params)
            except Exception as e:
                import traceback
                print(f"Training error in fold {fold_idx} (seed={seed}): {e}")
                traceback.print_exc()
                hf_r2_scores.append(-1.0)
                del model, trainer
                continue

            # 평가 (전체 192개에 대해 HF 예측)
            results = evaluate_model(model, data['X_all'], data['y_all_lf'], data['y_all'], device)
            hf_r2_scores.append(results['hf_r2'])

            # 메모리 정리
            del model, trainer
            if device == 'cuda':
                torch.cuda.empty_cache()

        mean_hf_r2 = np.mean(hf_r2_scores)

        # Pruning 체크
        trial.report(mean_hf_r2, 0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return mean_hf_r2

    return objective


# ============================================================================
# Main Experiment
# ============================================================================

def run_large_scale_bo(n_trials: int = 500, n_folds: int = 10,
                       save_dir: str = 'results/large_scale_bo',
                       device: str = 'cpu', n_jobs: int = 1):
    """대규모 BO 실험 실행"""

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("="*60)
    print("Large-Scale Bayesian Optimization Experiment")
    print("="*60)
    print(f"Total trials: {n_trials}")
    print(f"Folds: {n_folds} (Seeds: {SEEDS[:n_folds]})")
    print(f"Data: {N_LOFI} LF + {N_HIFI} HF")
    print(f"Device: {device}")
    print(f"n_jobs: {n_jobs}")
    print("="*60)

    # 데이터 로드
    print("\nLoading data...")
    lookup, all_combinations, param_space = load_base_data()
    print(f"Total compositions: {len(all_combinations)}")

    # Optuna Study 생성
    study_name = f"mf_bo_{timestamp}"
    sampler = TPESampler(seed=42, multivariate=True)

    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )

    # Objective function 생성
    objective = create_objective(lookup, all_combinations, n_folds=n_folds, device=device)

    # 최적화 실행
    print(f"\nStarting optimization with {n_trials} trials...")
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
        gc_after_trial=True
    )

    # 결과 저장
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)

    print(f"\nBest HF R²: {study.best_value:.4f}")
    print(f"\nBest Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 결과 파일 저장
    results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': n_trials,
        'n_folds': n_folds,
        'seeds': SEEDS[:n_folds],
        'data_config': {'n_lofi': N_LOFI, 'n_hifi': N_HIFI},
        'timestamp': timestamp,
        'all_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state)
            }
            for t in study.trials
        ]
    }

    results_path = os.path.join(save_dir, f'bo_results_v2_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Study 저장
    study_path = os.path.join(save_dir, f'study_v2_{timestamp}.pkl')
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    print(f"Study saved to: {study_path}")

    # 방법론별 통계
    print("\n" + "="*60)
    print("METHOD-WISE STATISTICS")
    print("="*60)

    method_results = {}
    for trial in study.trials:
        if trial.value is not None and trial.state == optuna.trial.TrialState.COMPLETE:
            method = trial.params.get('method', 'unknown')
            if method not in method_results:
                method_results[method] = []
            method_results[method].append(trial.value)

    for method, values in sorted(method_results.items()):
        print(f"\n{method}:")
        print(f"  Count: {len(values)}")
        print(f"  Mean R²: {np.mean(values):.4f}")
        print(f"  Max R²: {np.max(values):.4f}")
        print(f"  Std: {np.std(values):.4f}")

    return study, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Large-Scale BO Experiment')
    parser.add_argument('--n-trials', type=int, default=500, help='Number of trials')
    parser.add_argument('--n-folds', type=int, default=10, help='Number of folds (max 10)')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('--save-dir', type=str, default='results/large_scale_bo', help='Save directory')

    args = parser.parse_args()

    run_large_scale_bo(
        n_trials=args.n_trials,
        n_folds=min(args.n_folds, 10),  # 최대 10 folds
        device=args.device,
        n_jobs=args.n_jobs,
        save_dir=args.save_dir
    )
