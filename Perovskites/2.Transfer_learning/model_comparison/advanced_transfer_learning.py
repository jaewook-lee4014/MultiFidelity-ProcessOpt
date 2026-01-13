#!/usr/bin/env python3
"""
Advanced Transfer Learning Methods for Multi-Fidelity Optimization

6 New Transfer Learning Techniques:
1. Knowledge Distillation (KD) - DNGO-KD, BNN-KD
2. Domain Adaptation with MMD (DA) - DNGO-DA
3. Soft Parameter Sharing - MF-DNN-Soft
4. Pseudo-Labeling (PL) - DNGO-PL
5. Adapter-based Transfer - DNGO-Adapter
6. MAML (Model-Agnostic Meta-Learning) - DNGO-MAML

Author: Claude Code
Date: 2025-12-12
Updated: 2025-12-16 (Added MAML)
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

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


# ============================================================================
# Network Architectures
# ============================================================================

class LFNetwork(nn.Module):
    """Low-Fidelity Network"""
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

    def extract_features(self, x):
        return self.feature_net(x)


class HFNetwork(nn.Module):
    """High-Fidelity Network with Residual Learning"""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.0,
                 activation='relu', residual_mode=True):
        super().__init__()
        self.residual_mode = residual_mode
        layers = []
        in_dim = input_dim + 1  # +1 for y_lf
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

    def extract_features(self, x, y_lf):
        combined = torch.cat([x, y_lf], dim=-1)
        return self.feature_net(combined)


class MultiFidelityNetwork(nn.Module):
    """Multi-Fidelity Network"""
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


class AdapterLayer(nn.Module):
    """Adapter Layer for parameter-efficient transfer"""
    def __init__(self, input_dim, bottleneck_dim=16):
        super().__init__()
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, input_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        return x + self.up(self.activation(self.down(x)))


class AdapterNetwork(nn.Module):
    """Network with Adapter layers for HF adaptation"""
    def __init__(self, input_dim, hidden_dim, num_layers=2, bottleneck_dim=16,
                 dropout=0.0, activation='relu'):
        super().__init__()
        # Frozen feature extractor (pretrained on LF)
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh() if activation == 'tanh' else nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)

        # Adapter layers (trainable on HF)
        self.adapters = nn.ModuleList([
            AdapterLayer(hidden_dim, bottleneck_dim) for _ in range(num_layers)
        ])

        self.out_layer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, use_adapters=True):
        h = x
        layer_idx = 0
        for module in self.feature_net:
            h = module(h)
            if isinstance(module, (nn.ReLU, nn.Tanh)) and use_adapters:
                if layer_idx < len(self.adapters):
                    h = self.adapters[layer_idx](h)
                    layer_idx += 1
        return self.out_layer(h)

    def freeze_backbone(self):
        for param in self.feature_net.parameters():
            param.requires_grad = False
        for param in self.out_layer.parameters():
            param.requires_grad = False

    def unfreeze_adapters(self):
        for adapter in self.adapters:
            for param in adapter.parameters():
                param.requires_grad = True


# ============================================================================
# Loss Functions for Advanced Transfer Learning
# ============================================================================

def mmd_loss(source_features, target_features, kernel='rbf', bandwidth=1.0):
    """
    Maximum Mean Discrepancy (MMD) loss for domain adaptation.
    Measures distribution difference between source (LF) and target (HF) features.
    """
    def rbf_kernel(x, y, bandwidth):
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=-1)
        return torch.exp(-dist_sq / (2 * bandwidth ** 2))

    n_s = source_features.size(0)
    n_t = target_features.size(0)

    if kernel == 'rbf':
        k_ss = rbf_kernel(source_features, source_features, bandwidth)
        k_tt = rbf_kernel(target_features, target_features, bandwidth)
        k_st = rbf_kernel(source_features, target_features, bandwidth)

        mmd = (k_ss.sum() / (n_s * n_s) +
               k_tt.sum() / (n_t * n_t) -
               2 * k_st.sum() / (n_s * n_t))
    else:
        # Linear kernel
        mean_s = source_features.mean(dim=0)
        mean_t = target_features.mean(dim=0)
        mmd = torch.sum((mean_s - mean_t) ** 2)

    return mmd


def knowledge_distillation_loss(student_logits, teacher_logits, temperature=3.0):
    """
    Knowledge Distillation loss using soft labels from teacher (LF model).
    For regression, we use MSE between softened predictions.
    """
    # For regression, apply temperature scaling to create "soft" targets
    soft_student = student_logits / temperature
    soft_teacher = teacher_logits / temperature

    # KL divergence approximation for regression
    kd_loss = F.mse_loss(soft_student, soft_teacher) * (temperature ** 2)

    return kd_loss


def coral_loss(source_features, target_features):
    """
    CORAL (Correlation Alignment) loss for domain adaptation.
    Aligns second-order statistics (covariance) between domains.
    """
    d = source_features.size(1)

    # Source covariance
    source_mean = source_features.mean(dim=0, keepdim=True)
    source_centered = source_features - source_mean
    cov_source = (source_centered.t() @ source_centered) / (source_features.size(0) - 1)

    # Target covariance
    target_mean = target_features.mean(dim=0, keepdim=True)
    target_centered = target_features - target_mean
    cov_target = (target_centered.t() @ target_centered) / (target_features.size(0) - 1)

    # Frobenius norm of covariance difference
    loss = torch.sum((cov_source - cov_target) ** 2) / (4 * d * d)

    return loss


# ============================================================================
# Training Methods
# ============================================================================

def train_knowledge_distillation(model, X_lf, y_lf, X_hf, y_hf, params, device):
    """
    Knowledge Distillation Training

    1. Train LF network as teacher
    2. Train HF network with:
       - Hard label loss (actual HF targets)
       - Soft label loss (LF predictions as guidance)
    """
    alpha_kd = params.get('alpha_kd', 0.3)  # Weight for KD loss
    temperature = params.get('temperature', 3.0)
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

    # Stage 1: Train LF network (Teacher)
    lf_optimizer = optim.Adam(model.lf_network.parameters(), lr=lf_lr, weight_decay=weight_decay)
    model.train()
    for _ in range(lf_epochs):
        lf_optimizer.zero_grad()
        loss_fn(model.forward_lf(X_lf_t), y_lf_t).backward()
        lf_optimizer.step()

    # Freeze LF network (teacher)
    for param in model.lf_network.parameters():
        param.requires_grad = False

    # Stage 2: Train HF network (Student) with Knowledge Distillation
    hf_optimizer = optim.Adam(model.hf_network.parameters(), lr=hf_lr, weight_decay=weight_decay)

    for _ in range(hf_epochs):
        hf_optimizer.zero_grad()

        # Get teacher predictions (LF predictions on HF inputs)
        with torch.no_grad():
            teacher_pred = model.forward_lf(X_hf_t)

        # Get student predictions (HF network)
        student_pred = model.forward_hf(X_hf_t, teacher_pred)

        # Hard label loss (actual HF targets)
        hard_loss = loss_fn(student_pred, y_hf_t)

        # Soft label loss (match LF predictions scaled by offset)
        # For residual learning, we encourage the delta to be smooth
        kd_loss = knowledge_distillation_loss(student_pred, teacher_pred + (y_hf_t - teacher_pred).mean(), temperature)

        # Combined loss
        total_loss = (1 - alpha_kd) * hard_loss + alpha_kd * kd_loss
        total_loss.backward()
        hf_optimizer.step()

    for param in model.lf_network.parameters():
        param.requires_grad = True


def train_domain_adaptation_mmd(model, X_lf, y_lf, X_hf, y_hf, params, device):
    """
    Domain Adaptation with Maximum Mean Discrepancy (MMD)

    Aligns feature distributions between LF and HF data.
    Loss = L_task + lambda * L_MMD
    """
    lambda_mmd = params.get('lambda_mmd', 0.1)
    lf_lr = params.get('lf_lr', 1e-3)
    hf_lr = params.get('hf_lr', 1e-4)
    lf_epochs = params.get('lf_epochs', 200)
    hf_epochs = params.get('hf_epochs', 100)
    weight_decay = params.get('weight_decay', 1e-4)
    mmd_bandwidth = params.get('mmd_bandwidth', 1.0)

    loss_fn = nn.MSELoss()

    X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(device)
    y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(device)
    X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(device)
    y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(device)

    # Stage 1: Pretrain on LF
    lf_optimizer = optim.Adam(model.lf_network.parameters(), lr=lf_lr, weight_decay=weight_decay)
    model.train()
    for _ in range(lf_epochs):
        lf_optimizer.zero_grad()
        loss_fn(model.forward_lf(X_lf_t), y_lf_t).backward()
        lf_optimizer.step()

    # Stage 2: Finetune with MMD regularization
    optimizer = optim.Adam([
        {'params': model.lf_network.parameters(), 'lr': lf_lr * 0.1},
        {'params': model.hf_network.parameters(), 'lr': hf_lr}
    ], weight_decay=weight_decay)

    for _ in range(hf_epochs):
        optimizer.zero_grad()

        # Extract features from both domains
        lf_features = model.lf_network.extract_features(X_lf_t)

        with torch.no_grad():
            y_lf_for_hf = model.forward_lf(X_hf_t)
        hf_features = model.hf_network.extract_features(X_hf_t, y_lf_for_hf)

        # Task loss
        y_hf_pred = model.forward_hf(X_hf_t, y_lf_for_hf)
        task_loss = loss_fn(y_hf_pred, y_hf_t)

        # MMD loss (align feature distributions)
        # Use a subset of LF features to match HF sample size
        lf_subset_idx = torch.randperm(len(lf_features))[:len(hf_features)]
        lf_features_subset = lf_features[lf_subset_idx]

        # Need to project features to same dimension for MMD
        # Use mean pooling if dimensions differ
        if lf_features_subset.shape[1] != hf_features.shape[1]:
            min_dim = min(lf_features_subset.shape[1], hf_features.shape[1])
            lf_features_subset = lf_features_subset[:, :min_dim]
            hf_features = hf_features[:, :min_dim]

        mmd = mmd_loss(lf_features_subset, hf_features, bandwidth=mmd_bandwidth)

        # Total loss
        total_loss = task_loss + lambda_mmd * mmd
        total_loss.backward()
        optimizer.step()


def train_soft_parameter_sharing(model, X_lf, y_lf, X_hf, y_hf, params, device):
    """
    Soft Parameter Sharing

    Instead of hard sharing, regularize parameters to be similar:
    Loss = L_LF + L_HF + lambda * ||theta_LF - theta_HF||^2
    """
    lambda_soft = params.get('lambda_soft', 0.01)
    alpha = params.get('alpha', 0.5)
    lf_lr = params.get('lf_lr', 1e-3)
    hf_lr = params.get('hf_lr', 1e-4)
    epochs = params.get('epochs', 200)
    weight_decay = params.get('weight_decay', 1e-4)

    loss_fn = nn.MSELoss()

    X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(device)
    y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(device)
    X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(device)
    y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(device)

    optimizer = optim.Adam([
        {'params': model.lf_network.parameters(), 'lr': lf_lr},
        {'params': model.hf_network.parameters(), 'lr': hf_lr}
    ], weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()

        # LF loss
        lf_loss = loss_fn(model.forward_lf(X_lf_t), y_lf_t)

        # HF loss
        y_lf_for_hf = model.forward_lf(X_hf_t)
        hf_loss = loss_fn(model.forward_hf(X_hf_t, y_lf_for_hf), y_hf_t)

        # Soft parameter sharing regularization
        # Compare corresponding layers
        param_diff = 0.0
        lf_params = list(model.lf_network.feature_net.parameters())
        hf_params = list(model.hf_network.feature_net.parameters())

        # Only compare first layer (input dimensions differ for later layers)
        if len(lf_params) > 0 and len(hf_params) > 0:
            # Compare output weights of first linear layer (both have hidden_dim outputs)
            lf_first = lf_params[0]  # [hidden_dim, input_dim]
            hf_first = hf_params[0]  # [hidden_dim, input_dim + 1]

            # Compare overlapping part
            min_in = min(lf_first.shape[1], hf_first.shape[1])
            param_diff = torch.sum((lf_first[:, :min_in] - hf_first[:, :min_in]) ** 2)

        # Total loss
        total_loss = (1 - alpha) * lf_loss + alpha * hf_loss + lambda_soft * param_diff
        total_loss.backward()
        optimizer.step()


def train_pseudo_labeling(model, X_lf, y_lf, X_hf, y_hf, params, device):
    """
    Pseudo-Labeling

    1. Train LF model
    2. Generate pseudo-HF labels for unlabeled data using LF predictions + learned offset
    3. Train HF model with real + pseudo labels
    """
    confidence_threshold = params.get('confidence_threshold', 0.8)
    lf_lr = params.get('lf_lr', 1e-3)
    hf_lr = params.get('hf_lr', 1e-4)
    lf_epochs = params.get('lf_epochs', 200)
    hf_epochs = params.get('hf_epochs', 100)
    weight_decay = params.get('weight_decay', 1e-4)
    pseudo_weight = params.get('pseudo_weight', 0.5)

    loss_fn = nn.MSELoss()

    X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(device)
    y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(device)
    X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(device)
    y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(device)

    # Stage 1: Train LF model
    lf_optimizer = optim.Adam(model.lf_network.parameters(), lr=lf_lr, weight_decay=weight_decay)
    model.train()
    for _ in range(lf_epochs):
        lf_optimizer.zero_grad()
        loss_fn(model.forward_lf(X_lf_t), y_lf_t).backward()
        lf_optimizer.step()

    # Compute LF-HF offset from available HF data
    model.eval()
    with torch.no_grad():
        lf_pred_on_hf = model.forward_lf(X_hf_t)
        offset = (y_hf_t - lf_pred_on_hf).mean()
        offset_std = (y_hf_t - lf_pred_on_hf).std()

    # Generate pseudo-labels for LF data points not in HF
    with torch.no_grad():
        lf_pred_all = model.forward_lf(X_lf_t)
        pseudo_hf_labels = lf_pred_all + offset

    # Select confident pseudo-labels (those close to mean offset)
    residuals = torch.abs(pseudo_hf_labels - lf_pred_all - offset)
    confident_mask = (residuals < offset_std * confidence_threshold).squeeze()

    X_pseudo = X_lf_t[confident_mask]
    y_pseudo = pseudo_hf_labels[confident_mask]

    # Freeze LF network
    for param in model.lf_network.parameters():
        param.requires_grad = False

    # Stage 2: Train HF with real + pseudo labels
    hf_optimizer = optim.Adam(model.hf_network.parameters(), lr=hf_lr, weight_decay=weight_decay)

    model.train()
    for _ in range(hf_epochs):
        hf_optimizer.zero_grad()

        # Real HF loss
        with torch.no_grad():
            y_lf_for_hf = model.forward_lf(X_hf_t)
        real_loss = loss_fn(model.forward_hf(X_hf_t, y_lf_for_hf), y_hf_t)

        # Pseudo label loss
        if len(X_pseudo) > 0:
            with torch.no_grad():
                y_lf_pseudo = model.forward_lf(X_pseudo)
            pseudo_loss = loss_fn(model.forward_hf(X_pseudo, y_lf_pseudo), y_pseudo)
            total_loss = real_loss + pseudo_weight * pseudo_loss
        else:
            total_loss = real_loss

        total_loss.backward()
        hf_optimizer.step()

    for param in model.lf_network.parameters():
        param.requires_grad = True


def train_adapter(adapter_model, X_lf, y_lf, X_hf, y_hf, params, device):
    """
    Adapter-based Transfer Learning

    1. Pretrain full network on LF data
    2. Freeze backbone, only train adapter layers on HF data

    Note: Uses AdapterNetwork instead of MultiFidelityNetwork
    """
    lf_lr = params.get('lf_lr', 1e-3)
    adapter_lr = params.get('adapter_lr', 1e-3)
    lf_epochs = params.get('lf_epochs', 200)
    adapter_epochs = params.get('adapter_epochs', 100)
    weight_decay = params.get('weight_decay', 1e-4)

    loss_fn = nn.MSELoss()

    X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(device)
    y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(device)
    X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(device)
    y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(device)

    # Stage 1: Pretrain on LF (without adapters)
    optimizer = optim.Adam(adapter_model.parameters(), lr=lf_lr, weight_decay=weight_decay)
    adapter_model.train()
    for _ in range(lf_epochs):
        optimizer.zero_grad()
        loss_fn(adapter_model(X_lf_t, use_adapters=False), y_lf_t).backward()
        optimizer.step()

    # Stage 2: Freeze backbone, train only adapters
    adapter_model.freeze_backbone()
    adapter_model.unfreeze_adapters()

    # Also add a trainable output layer for HF
    hf_out = nn.Linear(adapter_model.out_layer.in_features, 1, bias=False).to(device)

    adapter_params = []
    for adapter in adapter_model.adapters:
        adapter_params.extend(adapter.parameters())
    adapter_params.extend(hf_out.parameters())

    adapter_optimizer = optim.Adam(adapter_params, lr=adapter_lr, weight_decay=weight_decay)

    for _ in range(adapter_epochs):
        adapter_optimizer.zero_grad()

        # Forward with adapters
        h = X_hf_t
        layer_idx = 0
        for module in adapter_model.feature_net:
            h = module(h)
            if isinstance(module, (nn.ReLU, nn.Tanh)):
                if layer_idx < len(adapter_model.adapters):
                    h = adapter_model.adapters[layer_idx](h)
                    layer_idx += 1

        pred = hf_out(h)
        loss = loss_fn(pred, y_hf_t)
        loss.backward()
        adapter_optimizer.step()

    # Store HF output layer
    adapter_model.hf_out = hf_out


def train_maml(model, X_lf, y_lf, X_hf, y_hf, params, device):
    """
    Model-Agnostic Meta-Learning (MAML) for Multi-Fidelity Transfer

    MAML learns an initialization that can quickly adapt to new tasks.
    Here, LF data is used as meta-training tasks and HF as target task.

    Algorithm:
    1. Meta-train: Learn initialization θ that can adapt to LF sub-tasks
    2. Fine-tune: Adapt θ to HF task with few gradient steps

    Key idea: θ* = θ - α∇L_task(f_θ) for each task
              θ = θ - β∇Σ_tasks L(f_θ*)

    Note: Added numerical stability improvements (gradient clipping, NaN checks)
    """
    inner_lr = params.get('inner_lr', 0.01)  # α: task-specific learning rate
    outer_lr = params.get('outer_lr', 1e-3)  # β: meta learning rate
    meta_epochs = params.get('meta_epochs', 100)
    inner_steps = params.get('inner_steps', 5)  # gradient steps per task
    n_tasks = params.get('n_tasks', 8)  # number of LF sub-tasks per meta-update
    task_size = params.get('task_size', 9)  # samples per task (same as HF)
    finetune_epochs = params.get('finetune_epochs', 50)
    finetune_lr = params.get('finetune_lr', 1e-4)
    weight_decay = params.get('weight_decay', 1e-4)
    grad_clip = params.get('grad_clip', 1.0)  # Gradient clipping for stability

    loss_fn = nn.MSELoss()

    X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(device)
    y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(device)
    X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(device)
    y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(device)

    # Ensure task_size doesn't exceed available data
    task_size = min(task_size, len(X_lf_t) - 1)
    if task_size < 4:
        task_size = 4  # Minimum task size

    # Use LF network for MAML (simpler architecture)
    meta_optimizer = optim.Adam(model.lf_network.parameters(), lr=outer_lr, weight_decay=weight_decay)

    # =========================================
    # Stage 1: Meta-Training on LF sub-tasks
    # =========================================
    model.train()

    for epoch in range(meta_epochs):
        meta_optimizer.zero_grad()

        meta_loss = 0.0
        valid_tasks = 0

        # Sample n_tasks from LF data
        for _ in range(n_tasks):
            try:
                # Sample a task (subset of LF data)
                task_idx = torch.randperm(len(X_lf_t))[:task_size]
                X_task = X_lf_t[task_idx]
                y_task = y_lf_t[task_idx]

                # Split into support and query
                support_size = max(task_size // 2, 2)
                X_support, y_support = X_task[:support_size], y_task[:support_size]
                X_query, y_query = X_task[support_size:], y_task[support_size:]

                if len(X_query) == 0:
                    continue

                # Clone model parameters for inner loop
                fast_weights = {name: param.clone() for name, param in model.lf_network.named_parameters()}

                # Inner loop: adapt to support set
                for _ in range(inner_steps):
                    # Forward with fast weights
                    pred = _forward_with_weights(model.lf_network, X_support, fast_weights)

                    # Check for NaN
                    if torch.isnan(pred).any():
                        break

                    inner_loss = loss_fn(pred, y_support)

                    # Check for NaN loss
                    if torch.isnan(inner_loss):
                        break

                    # Compute gradients w.r.t. fast_weights
                    grads = torch.autograd.grad(inner_loss, fast_weights.values(),
                                                create_graph=True, allow_unused=True)

                    # Update fast weights with gradient clipping
                    new_fast_weights = {}
                    for (name, param), grad in zip(fast_weights.items(), grads):
                        if grad is None:
                            new_fast_weights[name] = param
                        else:
                            # Clip gradient
                            grad_clipped = torch.clamp(grad, -grad_clip, grad_clip)
                            new_fast_weights[name] = param - inner_lr * grad_clipped
                    fast_weights = new_fast_weights

                # Evaluate on query set with adapted weights
                query_pred = _forward_with_weights(model.lf_network, X_query, fast_weights)

                # Check for NaN
                if torch.isnan(query_pred).any():
                    continue

                query_loss = loss_fn(query_pred, y_query)

                if not torch.isnan(query_loss):
                    meta_loss += query_loss
                    valid_tasks += 1

            except RuntimeError as e:
                # Skip this task if any error occurs
                continue

        # Meta update (only if we have valid tasks)
        if valid_tasks > 0:
            meta_loss = meta_loss / valid_tasks

            if not torch.isnan(meta_loss):
                meta_loss.backward()
                # Gradient clipping for meta-optimizer
                torch.nn.utils.clip_grad_norm_(model.lf_network.parameters(), grad_clip)
                meta_optimizer.step()

    # =========================================
    # Stage 2: Fine-tune on HF data
    # =========================================
    # Now the LF network has a good initialization for quick adaptation
    # Fine-tune HF network using the meta-learned LF features

    # Freeze LF network (use as feature extractor)
    for param in model.lf_network.parameters():
        param.requires_grad = False

    hf_optimizer = optim.Adam(model.hf_network.parameters(), lr=finetune_lr, weight_decay=weight_decay)

    for _ in range(finetune_epochs):
        hf_optimizer.zero_grad()

        with torch.no_grad():
            y_lf_pred = model.forward_lf(X_hf_t)

        y_hf_pred = model.forward_hf(X_hf_t, y_lf_pred)
        loss = loss_fn(y_hf_pred, y_hf_t)

        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.hf_network.parameters(), grad_clip)
            hf_optimizer.step()

    # Unfreeze for future use
    for param in model.lf_network.parameters():
        param.requires_grad = True


def _forward_with_weights(network, x, weights):
    """Forward pass using provided weights (for MAML inner loop)"""
    h = x
    layer_idx = 0

    # Get weight keys for each layer
    weight_keys = [k for k in weights.keys() if 'weight' in k]
    bias_keys = [k for k in weights.keys() if 'bias' in k]

    for module in network.feature_net:
        if isinstance(module, nn.Linear):
            # Find corresponding weight and bias
            w_key = f'feature_net.{layer_idx}.weight'
            b_key = f'feature_net.{layer_idx}.bias'

            if w_key in weights:
                w = weights[w_key]
                b = weights.get(b_key, None)
                h = F.linear(h, w, b)
            else:
                h = module(h)
            layer_idx += 1
        elif isinstance(module, (nn.ReLU, nn.Tanh)):
            h = module(h)
            layer_idx += 1
        elif isinstance(module, nn.Dropout):
            h = module(h)
            layer_idx += 1

    # Output layer
    out_w_key = 'out_layer.weight'
    if out_w_key in weights:
        h = F.linear(h, weights[out_w_key], None)
    else:
        h = network.out_layer(h)

    return h


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, X_test, y_test, device, model_type='mf'):
    """Evaluate model on test set"""
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        if model_type == 'mf':
            y_pred = model.forward_hf(X_t).cpu().numpy().flatten()
        elif model_type == 'adapter':
            # Use HF output layer with adapters
            h = X_t
            layer_idx = 0
            for module in model.feature_net:
                h = module(h)
                if isinstance(module, (nn.ReLU, nn.Tanh)):
                    if layer_idx < len(model.adapters):
                        h = model.adapters[layer_idx](h)
                        layer_idx += 1
            y_pred = model.hf_out(h).cpu().numpy().flatten()
        else:
            y_pred = model(X_t).cpu().numpy().flatten()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {'rmse': rmse, 'r2': r2, 'y_pred': y_pred}


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(method_name, train_func, data, params, device, seed, model_type='mf'):
    """Run single experiment"""
    set_seeds(seed)

    if model_type == 'adapter':
        model = AdapterNetwork(
            input_dim=3,
            hidden_dim=params.get('hidden_dim', 64),
            num_layers=params.get('num_layers', 2),
            bottleneck_dim=params.get('bottleneck_dim', 16),
            dropout=params.get('dropout', 0.0),
            activation=params.get('activation', 'relu')
        )
    else:
        model = MultiFidelityNetwork(
            input_dim=3,
            lf_hidden=params.get('lf_hidden', 64),
            hf_hidden=params.get('hf_hidden', 64),
            lf_layers=params.get('lf_layers', 2),
            hf_layers=params.get('hf_layers', 2),
            dropout=params.get('dropout', 0.0),
            activation=params.get('activation', 'relu'),
            residual_mode=True
        )
    model.to(device)

    # Train
    train_func(model, data['X_low'], data['y_low'], data['X_high'], data['y_high'], params, device)

    # Evaluate on test set (all data except HF train)
    test_mask = np.ones(len(data['X_all']), dtype=bool)
    test_mask[data['hifi_idx']] = False

    X_test = data['X_all'][test_mask]
    y_test = data['y_all'][test_mask]

    results = evaluate_model(model, X_test, y_test, device, model_type)

    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Data: {N_LOFI} LF + {N_HIFI} HF")
    print(f"Folds: {len(SEEDS)}")

    # Load data
    print("\nLoading data...")
    lookup, all_combinations, _ = load_base_data()
    print(f"Total compositions: {len(all_combinations)}")

    # Define methods
    methods = {
        'knowledge_distillation': {
            'train_func': train_knowledge_distillation,
            'params': {
                'alpha_kd': 0.3, 'temperature': 3.0,
                'lf_lr': 1e-3, 'hf_lr': 1e-4,
                'lf_epochs': 200, 'hf_epochs': 100,
                'lf_hidden': 64, 'hf_hidden': 64,
                'lf_layers': 2, 'hf_layers': 2
            },
            'model_type': 'mf'
        },
        'domain_adaptation_mmd': {
            'train_func': train_domain_adaptation_mmd,
            'params': {
                'lambda_mmd': 0.1, 'mmd_bandwidth': 1.0,
                'lf_lr': 1e-3, 'hf_lr': 1e-4,
                'lf_epochs': 200, 'hf_epochs': 100,
                'lf_hidden': 64, 'hf_hidden': 64,
                'lf_layers': 2, 'hf_layers': 2
            },
            'model_type': 'mf'
        },
        'soft_parameter_sharing': {
            'train_func': train_soft_parameter_sharing,
            'params': {
                'lambda_soft': 0.01, 'alpha': 0.5,
                'lf_lr': 1e-3, 'hf_lr': 1e-4,
                'epochs': 200,
                'lf_hidden': 64, 'hf_hidden': 64,
                'lf_layers': 2, 'hf_layers': 2
            },
            'model_type': 'mf'
        },
        'pseudo_labeling': {
            'train_func': train_pseudo_labeling,
            'params': {
                'confidence_threshold': 0.8, 'pseudo_weight': 0.5,
                'lf_lr': 1e-3, 'hf_lr': 1e-4,
                'lf_epochs': 200, 'hf_epochs': 100,
                'lf_hidden': 64, 'hf_hidden': 64,
                'lf_layers': 2, 'hf_layers': 2
            },
            'model_type': 'mf'
        },
        'adapter': {
            'train_func': train_adapter,
            'params': {
                'lf_lr': 1e-3, 'adapter_lr': 1e-3,
                'lf_epochs': 200, 'adapter_epochs': 100,
                'hidden_dim': 64, 'num_layers': 2,
                'bottleneck_dim': 16
            },
            'model_type': 'adapter'
        },
        'maml': {
            'train_func': train_maml,
            'params': {
                'inner_lr': 0.01, 'outer_lr': 1e-3,
                'meta_epochs': 100, 'inner_steps': 5,
                'n_tasks': 8, 'task_size': 9,
                'finetune_epochs': 50, 'finetune_lr': 1e-4,
                'lf_hidden': 64, 'hf_hidden': 64,
                'lf_layers': 2, 'hf_layers': 2
            },
            'model_type': 'mf'
        }
    }

    # Run experiments
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / 'visualizations' / f'{timestamp}_advanced_tl'
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {method: [] for method in methods}

    print("\n" + "="*70)
    print("Running Advanced Transfer Learning Experiments")
    print("="*70)

    for fold_idx, seed in enumerate(SEEDS, 1):
        print(f"\nFold {fold_idx}/{len(SEEDS)} (seed={seed})")
        print("-"*50)

        data = generate_data(lookup, all_combinations, n_lofi=N_LOFI, n_hifi=N_HIFI, seed=seed)

        for method_name, config in methods.items():
            try:
                results = run_experiment(
                    method_name,
                    config['train_func'],
                    data,
                    config['params'],
                    device,
                    seed,
                    config['model_type']
                )
                all_results[method_name].append(results)
                print(f"  {method_name:<30}: R²={results['r2']:.4f}, RMSE={results['rmse']:.4f}")
            except Exception as e:
                print(f"  {method_name:<30}: ERROR - {e}")
                import traceback
                traceback.print_exc()
                all_results[method_name].append({'r2': np.nan, 'rmse': np.nan})

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Method':<30} {'Mean R²':<12} {'Std R²':<12} {'Mean RMSE':<12}")
    print("-"*70)

    summary_data = []
    for method_name, results in all_results.items():
        r2_values = [r['r2'] for r in results if not np.isnan(r['r2'])]
        rmse_values = [r['rmse'] for r in results if not np.isnan(r['rmse'])]

        if r2_values:
            mean_r2 = np.mean(r2_values)
            std_r2 = np.std(r2_values)
            mean_rmse = np.mean(rmse_values)
            print(f"{method_name:<30} {mean_r2:<12.4f} {std_r2:<12.4f} {mean_rmse:<12.4f}")
            summary_data.append({
                'method': method_name,
                'mean_r2': mean_r2,
                'std_r2': std_r2,
                'mean_rmse': mean_rmse
            })

    # Save results
    import pandas as pd

    # Detailed results
    detailed_results = []
    for method_name, results in all_results.items():
        for fold_idx, r in enumerate(results, 1):
            detailed_results.append({
                'method': method_name,
                'fold': fold_idx,
                'seed': SEEDS[fold_idx-1],
                'r2': r['r2'],
                'rmse': r['rmse']
            })

    df_detailed = pd.DataFrame(detailed_results)
    df_detailed.to_csv(output_dir / 'detailed_results.csv', index=False)

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_dir / 'summary_results.csv', index=False)

    print(f"\nResults saved to: {output_dir}")

    return all_results


if __name__ == '__main__':
    main()
