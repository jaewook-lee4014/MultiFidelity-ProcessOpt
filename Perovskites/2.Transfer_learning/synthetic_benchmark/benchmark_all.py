#!/usr/bin/env python
"""
Unified Multi-Fidelity Benchmark Comparison

Compares 13 models on synthetic benchmarks (Branin-2D, Park-4D):

Models (from advanced_transfer_learning.py):
1. MFGP - BoTorch SingleTaskMultiFidelityGP (EI selection)
2. Sequential - Sequential Training (pretrain LF → finetune HF)
3. Progressive - Progressive layer unfreezing
4. Curriculum - Curriculum learning (easy → hard)
5. Two-Stage Joint - Two-stage joint training
6. DNGO-Joint - Joint training with gradient balancing
7. DNGO-Gradient - Gradient scaling between LF/HF
8. Knowledge Distillation - KD from LF to HF
9. Domain Adaptation (MMD) - MMD-based domain adaptation
10. Soft Parameter Sharing - Soft weight sharing regularization
11. Pseudo-Labeling - Self-training with pseudo labels
12. Adapter - Adapter-based transfer learning

Based on: "Best Practices for Multi-Fidelity Bayesian Optimization" (Nature Comp Science)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Tuple, Callable, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import norm
import argparse

# BoTorch
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

# Local imports
from synthetic_functions_mfbo import (
    branin_hf, branin_lf, park_hf, park_lf,
    SCENARIOS, FUNCTIONS
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# =============================================================================
# Network Architectures
# =============================================================================

class LFNetwork(nn.Module):
    """Low-Fidelity Network"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.out_layer(self.feature_net(x))

    def extract_features(self, x):
        return self.feature_net(x)


class HFNetwork(nn.Module):
    """High-Fidelity Network with Residual Learning"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        layers = []
        in_dim = input_dim + 1  # +1 for y_lf
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x, y_lf):
        combined = torch.cat([x, y_lf], dim=-1)
        delta = self.out_layer(self.feature_net(combined))
        return y_lf + delta

    def extract_features(self, x, y_lf):
        combined = torch.cat([x, y_lf], dim=-1)
        return self.feature_net(combined)


class AdapterLayer(nn.Module):
    """Adapter Layer for parameter-efficient transfer"""
    def __init__(self, input_dim, bottleneck_dim=16):
        super().__init__()
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, x):
        return x + self.up(F.relu(self.down(x)))


# =============================================================================
# Base Model Class
# =============================================================================

class BaseModel:
    """Base class for all multi-fidelity models"""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        raise NotImplementedError

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


# =============================================================================
# 1. MFGP (BoTorch)
# =============================================================================

class MFGP(BaseModel):
    """Multi-Fidelity GP using BoTorch"""

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        n_lf, n_hf = len(X_lf), len(X_hf)

        X_lf_fid = np.hstack([X_lf, np.zeros((n_lf, 1))])
        X_hf_fid = np.hstack([X_hf, np.ones((n_hf, 1))])

        X_all = np.vstack([X_lf_fid, X_hf_fid])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])

        train_X = torch.tensor(X_all, dtype=torch.float64).to(device)
        train_Y = torch.tensor(y_all, dtype=torch.float64).unsqueeze(-1).to(device)

        self.model = SingleTaskMultiFidelityGP(
            train_X, train_Y,
            data_fidelities=[self.input_dim],
            outcome_transform=Standardize(m=1)
        ).to(device)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self.is_fitted = True

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_fid = np.hstack([X, np.ones((len(X), 1))])
        X_tensor = torch.tensor(X_fid, dtype=torch.float64).to(device)

        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X_tensor)
            mean = posterior.mean.cpu().numpy().flatten()
            std = posterior.variance.sqrt().cpu().numpy().flatten()

        return mean, np.maximum(std, 1e-6)


# =============================================================================
# 2. Sequential (Pretrain LF → Finetune HF)
# =============================================================================

class Sequential(BaseModel):
    """Sequential Training: Pretrain on LF, then finetune on HF"""

    def __init__(self, input_dim, hidden_dim=64, lf_epochs=200, hf_epochs=100):
        super().__init__(input_dim, hidden_dim)
        self.lf_epochs = lf_epochs
        self.hf_epochs = hf_epochs

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]

        X_lf_t = torch.FloatTensor(X_lf_s).to(device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(device)

        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(device)

        # Stage 1: Train LF
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        # Freeze LF
        for p in self.lf_net.parameters():
            p.requires_grad = False

        # Stage 2: Train HF
        opt = torch.optim.Adam(self.hf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.hf_epochs):
            opt.zero_grad()
            with torch.no_grad():
                y_lf_pred = self.lf_net(X_hf_t)
            y_hf_pred = self.hf_net(X_hf_t, y_lf_pred)
            F.mse_loss(y_hf_pred, y_hf_t).backward()
            opt.step()

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(device)

        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()

        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1


# =============================================================================
# 3. Progressive (Layer unfreezing)
# =============================================================================

class Progressive(BaseModel):
    """Progressive Training: Gradually unfreeze layers"""

    def __init__(self, input_dim, hidden_dim=64, epochs_per_stage=50):
        super().__init__(input_dim, hidden_dim)
        self.epochs_per_stage = epochs_per_stage

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]

        X_lf_t = torch.FloatTensor(X_lf_s).to(device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(device)

        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(device)

        # Stage 1: Pretrain LF
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3)
        for _ in range(self.epochs_per_stage * 2):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        # Stage 2: Freeze all, train HF output only
        for p in self.lf_net.parameters():
            p.requires_grad = False
        for p in self.hf_net.feature_net.parameters():
            p.requires_grad = False

        opt = torch.optim.Adam(self.hf_net.out_layer.parameters(), lr=1e-3)
        for _ in range(self.epochs_per_stage):
            opt.zero_grad()
            with torch.no_grad():
                y_lf_pred = self.lf_net(X_hf_t)
            y_hf_pred = self.hf_net(X_hf_t, y_lf_pred)
            F.mse_loss(y_hf_pred, y_hf_t).backward()
            opt.step()

        # Stage 3: Unfreeze last HF layer
        for p in list(self.hf_net.feature_net.parameters())[-2:]:
            p.requires_grad = True
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.hf_net.parameters()), lr=1e-4)
        for _ in range(self.epochs_per_stage):
            opt.zero_grad()
            with torch.no_grad():
                y_lf_pred = self.lf_net(X_hf_t)
            y_hf_pred = self.hf_net(X_hf_t, y_lf_pred)
            F.mse_loss(y_hf_pred, y_hf_t).backward()
            opt.step()

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1


# =============================================================================
# 4. Curriculum (Easy to hard)
# =============================================================================

class Curriculum(BaseModel):
    """Curriculum Learning: Train on easy samples first"""

    def __init__(self, input_dim, hidden_dim=64, epochs=200):
        super().__init__(input_dim, hidden_dim)
        self.epochs = epochs

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]

        X_lf_t = torch.FloatTensor(X_lf_s).to(device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(device)

        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(device)

        # Pretrain LF
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3)
        for _ in range(self.epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        for p in self.lf_net.parameters():
            p.requires_grad = False

        # Curriculum: sort HF by residual magnitude (easy first)
        with torch.no_grad():
            y_lf_pred = self.lf_net(X_hf_t)
            residuals = torch.abs(y_hf_t - y_lf_pred).squeeze()
        sorted_idx = torch.argsort(residuals)

        # Train HF with curriculum
        opt = torch.optim.Adam(self.hf_net.parameters(), lr=1e-3)
        n_hf = len(X_hf_t)
        for epoch in range(self.epochs):
            # Gradually increase difficulty
            n_use = min(n_hf, max(2, int((epoch + 1) / self.epochs * n_hf)))
            idx = sorted_idx[:n_use]

            opt.zero_grad()
            with torch.no_grad():
                y_lf_sub = self.lf_net(X_hf_t[idx])
            y_hf_pred = self.hf_net(X_hf_t[idx], y_lf_sub)
            F.mse_loss(y_hf_pred, y_hf_t[idx]).backward()
            opt.step()

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1


# =============================================================================
# 5. Two-Stage Joint
# =============================================================================

class TwoStageJoint(BaseModel):
    """Two-Stage Joint Training"""

    def __init__(self, input_dim, hidden_dim=64, stage1_epochs=100, stage2_epochs=100):
        super().__init__(input_dim, hidden_dim)
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]

        X_lf_t = torch.FloatTensor(X_lf_s).to(device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(device)

        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(device)

        # Stage 1: Train LF only
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3)
        for _ in range(self.stage1_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        # Stage 2: Joint training with balanced loss
        opt = torch.optim.Adam(
            list(self.lf_net.parameters()) + list(self.hf_net.parameters()),
            lr=1e-4
        )
        for _ in range(self.stage2_epochs):
            opt.zero_grad()
            lf_loss = F.mse_loss(self.lf_net(X_lf_t), y_lf_t)
            y_lf_pred = self.lf_net(X_hf_t)
            hf_loss = F.mse_loss(self.hf_net(X_hf_t, y_lf_pred), y_hf_t)
            (0.3 * lf_loss + 0.7 * hf_loss).backward()
            opt.step()

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1


# =============================================================================
# 6. DNGO-Joint (Joint training)
# =============================================================================

class DNGOJoint(BaseModel):
    """DNGO-Joint: Joint training with gradient balancing"""

    def __init__(self, input_dim, hidden_dim=64, epochs=300, alpha=0.5):
        super().__init__(input_dim, hidden_dim)
        self.epochs = epochs
        self.alpha = alpha

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]

        X_lf_t = torch.FloatTensor(X_lf_s).to(device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(device)

        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(device)

        opt = torch.optim.Adam(
            list(self.lf_net.parameters()) + list(self.hf_net.parameters()),
            lr=1e-3, weight_decay=1e-4
        )

        for _ in range(self.epochs):
            opt.zero_grad()
            lf_loss = F.mse_loss(self.lf_net(X_lf_t), y_lf_t)
            with torch.no_grad():
                y_lf_pred = self.lf_net(X_hf_t)
            hf_loss = F.mse_loss(self.hf_net(X_hf_t, y_lf_pred), y_hf_t)
            ((1 - self.alpha) * lf_loss + self.alpha * hf_loss).backward()
            opt.step()

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1


# =============================================================================
# 7. DNGO-Gradient (Gradient scaling)
# =============================================================================

class DNGOGradient(BaseModel):
    """DNGO-Gradient: Gradient scaling between LF/HF"""

    def __init__(self, input_dim, hidden_dim=64, epochs=300):
        super().__init__(input_dim, hidden_dim)
        self.epochs = epochs

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]

        X_lf_t = torch.FloatTensor(X_lf_s).to(device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(device)

        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(device)

        # Use different learning rates for LF and HF
        opt = torch.optim.Adam([
            {'params': self.lf_net.parameters(), 'lr': 1e-3},
            {'params': self.hf_net.parameters(), 'lr': 5e-4}
        ], weight_decay=1e-4)

        for _ in range(self.epochs):
            opt.zero_grad()
            lf_loss = F.mse_loss(self.lf_net(X_lf_t), y_lf_t)
            y_lf_pred = self.lf_net(X_hf_t)
            hf_loss = F.mse_loss(self.hf_net(X_hf_t, y_lf_pred), y_hf_t)
            (lf_loss + hf_loss).backward()
            opt.step()

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1


# =============================================================================
# 8. Knowledge Distillation
# =============================================================================

class KnowledgeDistillation(BaseModel):
    """Knowledge Distillation from LF teacher to HF student"""

    def __init__(self, input_dim, hidden_dim=64, lf_epochs=200, hf_epochs=100, alpha_kd=0.3, temperature=3.0):
        super().__init__(input_dim, hidden_dim)
        self.lf_epochs = lf_epochs
        self.hf_epochs = hf_epochs
        self.alpha_kd = alpha_kd
        self.temperature = temperature

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]

        X_lf_t = torch.FloatTensor(X_lf_s).to(device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(device)

        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(device)

        # Train LF teacher
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        for p in self.lf_net.parameters():
            p.requires_grad = False

        # Train HF with KD
        opt = torch.optim.Adam(self.hf_net.parameters(), lr=1e-4)
        for _ in range(self.hf_epochs):
            opt.zero_grad()
            with torch.no_grad():
                teacher_pred = self.lf_net(X_hf_t)
            student_pred = self.hf_net(X_hf_t, teacher_pred)

            hard_loss = F.mse_loss(student_pred, y_hf_t)
            soft_student = student_pred / self.temperature
            soft_teacher = teacher_pred / self.temperature
            kd_loss = F.mse_loss(soft_student, soft_teacher) * (self.temperature ** 2)

            ((1 - self.alpha_kd) * hard_loss + self.alpha_kd * kd_loss).backward()
            opt.step()

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1


# =============================================================================
# 9. Domain Adaptation (MMD)
# =============================================================================

class DomainAdaptationMMD(BaseModel):
    """Domain Adaptation with Maximum Mean Discrepancy"""

    def __init__(self, input_dim, hidden_dim=64, lf_epochs=200, hf_epochs=100, lambda_mmd=0.1):
        super().__init__(input_dim, hidden_dim)
        self.lf_epochs = lf_epochs
        self.hf_epochs = hf_epochs
        self.lambda_mmd = lambda_mmd

    def _mmd_loss(self, source, target, bandwidth=1.0):
        def rbf_kernel(x, y):
            diff = x.unsqueeze(1) - y.unsqueeze(0)
            dist_sq = torch.sum(diff ** 2, dim=-1)
            return torch.exp(-dist_sq / (2 * bandwidth ** 2))

        k_ss = rbf_kernel(source, source)
        k_tt = rbf_kernel(target, target)
        k_st = rbf_kernel(source, target)
        return k_ss.mean() + k_tt.mean() - 2 * k_st.mean()

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]

        X_lf_t = torch.FloatTensor(X_lf_s).to(device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(device)

        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(device)

        # Pretrain LF
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        # Finetune with MMD
        opt = torch.optim.Adam([
            {'params': self.lf_net.parameters(), 'lr': 1e-4},
            {'params': self.hf_net.parameters(), 'lr': 1e-4}
        ])

        for _ in range(self.hf_epochs):
            opt.zero_grad()
            lf_features = self.lf_net.extract_features(X_lf_t)
            with torch.no_grad():
                y_lf_pred = self.lf_net(X_hf_t)
            hf_features = self.hf_net.extract_features(X_hf_t, y_lf_pred)

            task_loss = F.mse_loss(self.hf_net(X_hf_t, y_lf_pred), y_hf_t)

            # Match feature dimensions
            min_n = min(len(lf_features), len(hf_features))
            mmd = self._mmd_loss(lf_features[:min_n], hf_features[:min_n])

            (task_loss + self.lambda_mmd * mmd).backward()
            opt.step()

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1


# =============================================================================
# 10. Soft Parameter Sharing
# =============================================================================

class SoftParameterSharing(BaseModel):
    """Soft Parameter Sharing with regularization"""

    def __init__(self, input_dim, hidden_dim=64, epochs=200, lambda_soft=0.01):
        super().__init__(input_dim, hidden_dim)
        self.epochs = epochs
        self.lambda_soft = lambda_soft

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]

        X_lf_t = torch.FloatTensor(X_lf_s).to(device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(device)

        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(device)

        opt = torch.optim.Adam(
            list(self.lf_net.parameters()) + list(self.hf_net.parameters()),
            lr=1e-3
        )

        for _ in range(self.epochs):
            opt.zero_grad()
            lf_loss = F.mse_loss(self.lf_net(X_lf_t), y_lf_t)
            y_lf_pred = self.lf_net(X_hf_t)
            hf_loss = F.mse_loss(self.hf_net(X_hf_t, y_lf_pred), y_hf_t)

            # Soft parameter sharing regularization
            lf_params = list(self.lf_net.feature_net.parameters())
            hf_params = list(self.hf_net.feature_net.parameters())
            param_diff = 0.0
            if len(lf_params) > 0 and len(hf_params) > 0:
                lf_w = lf_params[0]
                hf_w = hf_params[0]
                min_in = min(lf_w.shape[1], hf_w.shape[1])
                param_diff = torch.sum((lf_w[:, :min_in] - hf_w[:, :min_in]) ** 2)

            (0.5 * lf_loss + 0.5 * hf_loss + self.lambda_soft * param_diff).backward()
            opt.step()

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1


# =============================================================================
# 11. Pseudo-Labeling
# =============================================================================

class PseudoLabeling(BaseModel):
    """Pseudo-Labeling: Self-training with pseudo labels"""

    def __init__(self, input_dim, hidden_dim=64, lf_epochs=200, hf_epochs=100, pseudo_weight=0.5):
        super().__init__(input_dim, hidden_dim)
        self.lf_epochs = lf_epochs
        self.hf_epochs = hf_epochs
        self.pseudo_weight = pseudo_weight

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]

        X_lf_t = torch.FloatTensor(X_lf_s).to(device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(device)

        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(device)

        # Train LF
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        # Compute offset
        self.lf_net.eval()
        with torch.no_grad():
            lf_pred_on_hf = self.lf_net(X_hf_t)
            offset = (y_hf_t - lf_pred_on_hf).mean()
            lf_pred_all = self.lf_net(X_lf_t)
            pseudo_labels = lf_pred_all + offset

        for p in self.lf_net.parameters():
            p.requires_grad = False

        # Train HF with real + pseudo labels
        opt = torch.optim.Adam(self.hf_net.parameters(), lr=1e-4)
        for _ in range(self.hf_epochs):
            opt.zero_grad()
            with torch.no_grad():
                y_lf_for_hf = self.lf_net(X_hf_t)
            real_loss = F.mse_loss(self.hf_net(X_hf_t, y_lf_for_hf), y_hf_t)

            with torch.no_grad():
                y_lf_pseudo = self.lf_net(X_lf_t)
            pseudo_loss = F.mse_loss(self.hf_net(X_lf_t, y_lf_pseudo), pseudo_labels)

            (real_loss + self.pseudo_weight * pseudo_loss).backward()
            opt.step()

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1


# =============================================================================
# 12. Adapter
# =============================================================================

class Adapter(BaseModel):
    """Adapter-based Transfer Learning"""

    def __init__(self, input_dim, hidden_dim=64, lf_epochs=200, adapter_epochs=100, bottleneck_dim=16):
        super().__init__(input_dim, hidden_dim)
        self.lf_epochs = lf_epochs
        self.adapter_epochs = adapter_epochs
        self.bottleneck_dim = bottleneck_dim

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]

        X_lf_t = torch.FloatTensor(X_lf_s).to(device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(device)

        # Build network with adapters
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        ).to(device)
        self.out_layer = nn.Linear(self.hidden_dim, 1).to(device)
        self.adapters = nn.ModuleList([
            AdapterLayer(self.hidden_dim, self.bottleneck_dim) for _ in range(2)
        ]).to(device)
        self.hf_out = nn.Linear(self.hidden_dim, 1).to(device)

        # Pretrain on LF
        opt = torch.optim.Adam(
            list(self.backbone.parameters()) + list(self.out_layer.parameters()), lr=1e-3
        )
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            h = self.backbone(X_lf_t)
            F.mse_loss(self.out_layer(h), y_lf_t).backward()
            opt.step()

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.out_layer.parameters():
            p.requires_grad = False

        # Train adapters on HF
        opt = torch.optim.Adam(
            list(self.adapters.parameters()) + list(self.hf_out.parameters()), lr=1e-3
        )
        for _ in range(self.adapter_epochs):
            opt.zero_grad()
            h = X_hf_t
            adapter_idx = 0
            for i, module in enumerate(self.backbone):
                h = module(h)
                if isinstance(module, nn.Tanh) and adapter_idx < len(self.adapters):
                    h = self.adapters[adapter_idx](h)
                    adapter_idx += 1
            F.mse_loss(self.hf_out(h), y_hf_t).backward()
            opt.step()

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(device)
        self.backbone.eval()
        with torch.no_grad():
            h = X_t
            adapter_idx = 0
            for module in self.backbone:
                h = module(h)
                if isinstance(module, nn.Tanh) and adapter_idx < len(self.adapters):
                    h = self.adapters[adapter_idx](h)
                    adapter_idx += 1
            mean_s = self.hf_out(h).cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1


# =============================================================================
# Acquisition Functions
# =============================================================================

def expected_improvement(mean, std, y_best, xi=0.01):
    with np.errstate(divide='ignore', invalid='ignore'):
        imp = y_best - mean - xi
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std < 1e-10] = 0.0
    return ei


def select_next(mean, std, y_best, sampled_indices, use_ei=True):
    if use_ei:
        scores = expected_improvement(mean, std, y_best)
        scores[list(sampled_indices)] = -np.inf
        return np.argmax(scores)
    else:
        mean_masked = mean.copy()
        mean_masked[list(sampled_indices)] = np.inf
        return np.argmin(mean_masked)


# =============================================================================
# Benchmark Classes
# =============================================================================

class SyntheticBenchmark:
    """Synthetic benchmark (Branin, Park)"""

    def __init__(self, name, hf_func, lf_func, dim, alpha, cost_ratio, f_star, grid_size=50):
        self.name = name
        self.hf_func = hf_func
        self.lf_func = lf_func
        self.dim = dim
        self.alpha = alpha
        self.cost_ratio = cost_ratio
        self.f_star = f_star
        self.grid_size = grid_size
        self._create_grid()
        # R² = Pearson correlation squared (correct informativeness measure)
        corr = np.corrcoef(self.y_hf, self.y_lf)[0, 1]
        self.r2 = corr ** 2
        print(f"Created {name}: {self.n_candidates} grid points, R²={self.r2:.3f}, ρ={cost_ratio}")

    def _create_grid(self):
        if self.dim == 2:
            axes = [np.linspace(0, 1, self.grid_size) for _ in range(2)]
            grids = np.meshgrid(*axes)
            self.X = np.column_stack([g.ravel() for g in grids])
        else:
            n_per_dim = int(np.ceil(self.grid_size ** 0.5))
            axes = [np.linspace(0, 1, n_per_dim) for _ in range(self.dim)]
            grids = np.meshgrid(*axes, indexing='ij')
            self.X = np.column_stack([g.ravel() for g in grids])

        self.n_candidates = len(self.X)
        self.y_hf = self.hf_func(self.X).flatten()
        self.y_lf = self.lf_func(self.X, self.alpha).flatten()

    def evaluate_hf(self, indices):
        return self.y_hf[indices.astype(int).flatten()]

    def evaluate_lf(self, indices):
        return self.y_lf[indices.astype(int).flatten()]


# =============================================================================
# BO Loop
# =============================================================================

def run_bo(benchmark, model_class, budget, seed=42):
    """Run BO on benchmark"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    rho = benchmark.cost_ratio
    n_candidates = benchmark.n_candidates

    # Initial sampling
    init_budget = 0.1 * budget
    n_init_hf = max(2, int(init_budget * 0.5 / 1.0))
    n_init_lf = max(2, int(init_budget * 0.5 / rho))

    all_indices = np.arange(n_candidates)
    np.random.shuffle(all_indices)

    lf_indices = set(all_indices[:n_init_lf].tolist())
    hf_indices = set(all_indices[n_init_lf:n_init_lf + n_init_hf].tolist())

    X_lf = benchmark.X[list(lf_indices)]
    y_lf = benchmark.evaluate_lf(np.array(list(lf_indices)))
    X_hf = benchmark.X[list(hf_indices)]
    y_hf = benchmark.evaluate_hf(np.array(list(hf_indices)))

    current_budget = n_init_lf * rho + n_init_hf * 1.0

    lf_per_hf = max(1, int(1.0 / rho))
    lf_counter = 0

    regrets = [max(0, y_hf.min() - benchmark.f_star)]
    budgets = [current_budget]

    iteration = 0
    max_iter = 500
    use_ei = (model_class == MFGP)

    while current_budget < budget and iteration < max_iter:
        iteration += 1
        remaining = budget - current_budget

        # Decide fidelity
        if remaining >= 1.0:
            if remaining >= rho and lf_counter < lf_per_hf:
                eval_hf = False
                cost = rho
                lf_counter += 1
            else:
                eval_hf = True
                cost = 1.0
                lf_counter = 0
        elif remaining >= rho:
            eval_hf = False
            cost = rho
        else:
            break

        try:
            model = model_class(benchmark.X.shape[1])
            model.fit(X_lf, y_lf, X_hf, y_hf)
            mean, std = model.predict(benchmark.X)

            sampled = lf_indices | hf_indices
            y_best = y_hf.min()
            next_idx = select_next(mean, std, y_best, sampled, use_ei)

            if eval_hf:
                y_new = benchmark.evaluate_hf(np.array([next_idx]))
                hf_indices.add(next_idx)
                X_hf = benchmark.X[list(hf_indices)]
                y_hf = benchmark.evaluate_hf(np.array(list(hf_indices)))
            else:
                y_new = benchmark.evaluate_lf(np.array([next_idx]))
                lf_indices.add(next_idx)
                X_lf = benchmark.X[list(lf_indices)]
                y_lf = benchmark.evaluate_lf(np.array(list(lf_indices)))

            current_budget += cost

        except Exception as e:
            available = set(range(n_candidates)) - (lf_indices | hf_indices)
            if available:
                next_idx = np.random.choice(list(available))
                if eval_hf:
                    hf_indices.add(next_idx)
                    X_hf = benchmark.X[list(hf_indices)]
                    y_hf = benchmark.evaluate_hf(np.array(list(hf_indices)))
                else:
                    lf_indices.add(next_idx)
                    X_lf = benchmark.X[list(lf_indices)]
                    y_lf = benchmark.evaluate_lf(np.array(list(lf_indices)))
            current_budget += cost

        regrets.append(max(0, y_hf.min() - benchmark.f_star))
        budgets.append(current_budget)

    return {
        'regrets': regrets,
        'budgets': budgets,
        'final_regret': regrets[-1],
        'n_hf': len(hf_indices),
        'n_lf': len(lf_indices),
        'best_y': y_hf.min()
    }


# =============================================================================
# Main
# =============================================================================

def run_all_benchmarks(n_seeds=3, base_seed=42):
    """Run all benchmarks with multiple seeds"""

    # Define benchmarks (Branin only for quick test)
    benchmarks = {
        'Branin-Fav': SyntheticBenchmark(
            'Branin-Fav', branin_hf, branin_lf, dim=2,
            alpha=0.8, cost_ratio=0.1, f_star=0.397887, grid_size=50
        ),
        'Branin-Unfav': SyntheticBenchmark(
            'Branin-Unfav', branin_hf, branin_lf, dim=2,
            alpha=0.1, cost_ratio=0.5, f_star=0.397887, grid_size=50
        ),
    }

    # Budget
    budgets = {'Branin-Fav': 50, 'Branin-Unfav': 50}

    # All 12 models
    models = {
        'MFGP': MFGP,
        'Sequential': Sequential,
        'Progressive': Progressive,
        'Curriculum': Curriculum,
        'Two-Stage Joint': TwoStageJoint,
        'DNGO-Joint': DNGOJoint,
        'DNGO-Gradient': DNGOGradient,
        'Knowledge Distillation': KnowledgeDistillation,
        'Domain Adaptation (MMD)': DomainAdaptationMMD,
        'Soft Parameter Sharing': SoftParameterSharing,
        'Pseudo-Labeling': PseudoLabeling,
        'Adapter': Adapter,
    }

    seeds = [base_seed + i for i in range(n_seeds)]
    results = []

    for bench_name, benchmark in benchmarks.items():
        print(f"\n{'='*60}")
        print(f"Benchmark: {bench_name}")
        print(f"{'='*60}")

        for model_name, model_class in models.items():
            for seed in seeds:
                print(f"  {model_name}, seed={seed}...", end=' ')
                try:
                    result = run_bo(benchmark, model_class, budgets[bench_name], seed)
                    results.append({
                        'benchmark': bench_name,
                        'model': model_name,
                        'seed': seed,
                        'final_regret': result['final_regret'],
                        'n_hf': result['n_hf'],
                        'n_lf': result['n_lf'],
                        'best_y': result['best_y'],
                    })
                    print(f"regret={result['final_regret']:.4f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    results.append({
                        'benchmark': bench_name,
                        'model': model_name,
                        'seed': seed,
                        'final_regret': np.nan,
                        'n_hf': 0, 'n_lf': 0, 'best_y': np.nan,
                    })

    return pd.DataFrame(results)


def visualize_results(df, output_dir):
    """Visualize comparison results"""
    benchmarks = df['benchmark'].unique()
    models = df['model'].unique()

    # Color by category
    def get_color(name):
        if name == 'MFGP':
            return '#FF6B6B'
        elif 'DNGO' in name:
            return '#4ECDC4'
        elif name in ['Sequential', 'Progressive', 'Curriculum', 'Two-Stage Joint']:
            return '#45B7D1'
        else:
            return '#96CEB4'

    for bench_name in benchmarks:
        df_bench = df[df['benchmark'] == bench_name]

        # Compute mean regret per model
        summary = df_bench.groupby('model')['final_regret'].agg(['mean', 'std']).reset_index()
        summary = summary.sort_values('mean')

        fig, ax = plt.subplots(figsize=(14, 8))
        colors = [get_color(m) for m in summary['model']]

        x = np.arange(len(summary))
        bars = ax.barh(x, summary['mean'], xerr=summary['std'], capsize=3,
                       color=colors, edgecolor='black', linewidth=1)

        ax.set_yticks(x)
        ax.set_yticklabels(summary['model'], fontsize=10)
        ax.set_xlabel('Final Regret (mean ± std)', fontsize=12)
        ax.set_title(f'{bench_name}: Model Comparison ({len(df_bench["seed"].unique())} seeds)',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add values
        for i, (mean, std) in enumerate(zip(summary['mean'], summary['std'])):
            ax.text(mean + std + 0.01, i, f'{mean:.4f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / f'{bench_name}_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {bench_name}_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='Unified MF Benchmark')
    parser.add_argument('--n-seeds', type=int, default=3, help='Number of seeds')
    parser.add_argument('--base-seed', type=int, default=42, help='Base seed')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'benchmark_results_{timestamp}')
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Unified Multi-Fidelity Benchmark Comparison")
    print("=" * 80)
    print(f"Models: 12 models")
    print(f"Seeds: {args.n_seeds}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    df = run_all_benchmarks(args.n_seeds, args.base_seed)

    df.to_csv(output_dir / 'results.csv', index=False)
    print(f"\nSaved: results.csv")

    print("\nGenerating visualizations...")
    visualize_results(df, output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY (Mean Final Regret)")
    print("=" * 80)

    for bench in df['benchmark'].unique():
        print(f"\n{bench}:")
        summary = df[df['benchmark'] == bench].groupby('model')['final_regret'].agg(['mean', 'std'])
        summary = summary.sort_values('mean')
        for model, row in summary.iterrows():
            print(f"  {model:<30}: {row['mean']:.4f} ± {row['std']:.4f}")

    print("\n" + "=" * 80)
    print("Done!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
