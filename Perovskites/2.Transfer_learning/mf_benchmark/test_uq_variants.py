#!/usr/bin/env python
"""
UQ Variants Comparison Test

Compare 5 Sequential model variants:
1. Baseline - Standard Sequential (std=0.1 fixed)
2. +SpectralNorm - Spectral Normalization only
3. +BLR - Last-Layer Bayesian Linear Regression only
4. +LayerNorm - LayerNorm only
5. +All - SpectralNorm + BLR + LayerNorm

Test on Branin-2D (Favorable scenario) with single seed for quick validation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import norm
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# Synthetic Functions (Branin-2D)
# =============================================================================

def branin_hf(X: np.ndarray) -> np.ndarray:
    """High-fidelity Branin function"""
    x1, x2 = X[:, 0], X[:, 1]
    x1_scaled = 15 * x1 - 5
    x2_scaled = 15 * x2
    a, b, c = 1, 5.1 / (4 * np.pi**2), 5 / np.pi
    r, s, t = 6, 10, 1 / (8 * np.pi)
    term1 = a * (x2_scaled - b * x1_scaled**2 + c * x1_scaled - r)**2
    term2 = s * (1 - t) * np.cos(x1_scaled)
    return (term1 + term2 + s).reshape(-1)

def branin_lf(X: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    """Low-fidelity Branin function"""
    x1, x2 = X[:, 0], X[:, 1]
    x1_scaled = 15 * x1 - 5
    x2_scaled = 15 * x2
    a, b, c = 1, 5.1 / (4 * np.pi**2), 5 / np.pi
    r, s, t = 6, 10, 1 / (8 * np.pi)
    term1 = a * (x2_scaled - b * x1_scaled**2 + c * x1_scaled - r)**2
    term2 = s * (1 - t) * np.cos(x1_scaled)
    hf = term1 + term2 + s
    return (alpha * hf + (1 - alpha) * (10 * x1_scaled + 5 * x2_scaled)).reshape(-1)


# =============================================================================
# Network Architectures
# =============================================================================

class LFNetwork_Baseline(nn.Module):
    """Baseline LF Network (Tanh, no normalization)"""
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


class LFNetwork_SpectralNorm(nn.Module):
    """LF Network with Spectral Normalization"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(spectral_norm(nn.Linear(in_dim, hidden_dim)))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        self.out_layer = spectral_norm(nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.out_layer(self.feature_net(x))

    def extract_features(self, x):
        return self.feature_net(x)


class LFNetwork_LayerNorm(nn.Module):
    """LF Network with LayerNorm"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.out_layer(self.feature_net(x))

    def extract_features(self, x):
        return self.feature_net(x)


class LFNetwork_All(nn.Module):
    """LF Network with SpectralNorm + LayerNorm"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(spectral_norm(nn.Linear(in_dim, hidden_dim)))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        self.out_layer = spectral_norm(nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.out_layer(self.feature_net(x))

    def extract_features(self, x):
        return self.feature_net(x)


class HFNetwork(nn.Module):
    """HF Network (same for all variants - learns delta)"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, use_spectral=False, use_layernorm=False):
        super().__init__()
        layers = []
        in_dim = input_dim + 1  # x + y_lf
        for _ in range(num_layers):
            linear = nn.Linear(in_dim, hidden_dim)
            if use_spectral:
                linear = spectral_norm(linear)
            layers.append(linear)
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        out_linear = nn.Linear(hidden_dim, 1)
        self.out_layer = spectral_norm(out_linear) if use_spectral else out_linear

    def forward(self, x, y_lf):
        combined = torch.cat([x, y_lf], dim=-1)
        delta = self.out_layer(self.feature_net(combined))
        return y_lf + delta

    def extract_features(self, x, y_lf):
        combined = torch.cat([x, y_lf], dim=-1)
        return self.feature_net(combined)


# =============================================================================
# Model Classes
# =============================================================================

class SequentialBaseline:
    """Baseline: std=0.1 fixed"""
    def __init__(self, input_dim, hidden_dim=64, lf_epochs=200, hf_epochs=100):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lf_epochs = lf_epochs
        self.hf_epochs = hf_epochs
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

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

        # LF training
        self.lf_net = LFNetwork_Baseline(self.input_dim, self.hidden_dim).to(device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        # Freeze LF
        for p in self.lf_net.parameters():
            p.requires_grad = False

        # HF training
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(device)
        opt = torch.optim.Adam(self.hf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.hf_epochs):
            opt.zero_grad()
            with torch.no_grad():
                y_lf_pred = self.lf_net(X_hf_t)
            y_hf_pred = self.hf_net(X_hf_t, y_lf_pred)
            F.mse_loss(y_hf_pred, y_hf_t).backward()
            opt.step()

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(device)

        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()

        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1  # Fixed std


class SequentialSpectralNorm:
    """SpectralNorm only: distance-preserving features"""
    def __init__(self, input_dim, hidden_dim=64, lf_epochs=200, hf_epochs=100):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lf_epochs = lf_epochs
        self.hf_epochs = hf_epochs
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

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

        # LF with Spectral Norm
        self.lf_net = LFNetwork_SpectralNorm(self.input_dim, self.hidden_dim).to(device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        for p in self.lf_net.parameters():
            p.requires_grad = False

        # HF with Spectral Norm
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim, use_spectral=True).to(device)
        opt = torch.optim.Adam(self.hf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.hf_epochs):
            opt.zero_grad()
            with torch.no_grad():
                y_lf_pred = self.lf_net(X_hf_t)
            y_hf_pred = self.hf_net(X_hf_t, y_lf_pred)
            F.mse_loss(y_hf_pred, y_hf_t).backward()
            opt.step()

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(device)

        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()

        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1  # Still fixed std (SpectralNorm only)


class SequentialBLR:
    """BLR only: Last-layer Bayesian Linear Regression"""
    def __init__(self, input_dim, hidden_dim=64, lf_epochs=200, hf_epochs=100,
                 alpha_blr=1.0, beta_blr=25.0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lf_epochs = lf_epochs
        self.hf_epochs = hf_epochs
        self.alpha_blr = alpha_blr
        self.beta_blr = beta_blr
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

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

        # LF training
        self.lf_net = LFNetwork_Baseline(self.input_dim, self.hidden_dim).to(device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        for p in self.lf_net.parameters():
            p.requires_grad = False

        # HF training
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(device)
        opt = torch.optim.Adam(self.hf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.hf_epochs):
            opt.zero_grad()
            with torch.no_grad():
                y_lf_pred = self.lf_net(X_hf_t)
            y_hf_pred = self.hf_net(X_hf_t, y_lf_pred)
            F.mse_loss(y_hf_pred, y_hf_t).backward()
            opt.step()

        # Fit BLR on HF features
        self.hf_net.eval()
        with torch.no_grad():
            y_lf_pred = self.lf_net(X_hf_t)
            Phi = self.hf_net.extract_features(X_hf_t, y_lf_pred).cpu().numpy()

        # BLR closed form
        A = self.alpha_blr * np.eye(self.hidden_dim) + self.beta_blr * Phi.T @ Phi
        self.A_inv = np.linalg.inv(A + 1e-6 * np.eye(A.shape[0]))
        self.m = self.beta_blr * self.A_inv @ Phi.T @ y_hf_s

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(device)

        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            Phi = self.hf_net.extract_features(X_t, y_lf).cpu().numpy()

        # BLR prediction
        mean_s = Phi @ self.m
        var_s = 1.0 / self.beta_blr + np.sum((Phi @ self.A_inv) * Phi, axis=1)
        std_s = np.sqrt(np.maximum(var_s, 1e-10))

        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        std = std_s * self.scaler_y.scale_[0]

        return mean, np.maximum(std, 1e-6)


class SequentialLayerNorm:
    """LayerNorm only: stable training with small batches"""
    def __init__(self, input_dim, hidden_dim=64, lf_epochs=200, hf_epochs=100):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lf_epochs = lf_epochs
        self.hf_epochs = hf_epochs
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

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

        # LF with LayerNorm
        self.lf_net = LFNetwork_LayerNorm(self.input_dim, self.hidden_dim).to(device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        for p in self.lf_net.parameters():
            p.requires_grad = False

        # HF with LayerNorm
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim, use_layernorm=True).to(device)
        opt = torch.optim.Adam(self.hf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.hf_epochs):
            opt.zero_grad()
            with torch.no_grad():
                y_lf_pred = self.lf_net(X_hf_t)
            y_hf_pred = self.hf_net(X_hf_t, y_lf_pred)
            F.mse_loss(y_hf_pred, y_hf_t).backward()
            opt.step()

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(device)

        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()

        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1  # Fixed std


class SequentialAll:
    """All: SpectralNorm + BLR + LayerNorm"""
    def __init__(self, input_dim, hidden_dim=64, lf_epochs=200, hf_epochs=100,
                 alpha_blr=1.0, beta_blr=25.0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lf_epochs = lf_epochs
        self.hf_epochs = hf_epochs
        self.alpha_blr = alpha_blr
        self.beta_blr = beta_blr
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

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

        # LF with SpectralNorm + LayerNorm
        self.lf_net = LFNetwork_All(self.input_dim, self.hidden_dim).to(device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        for p in self.lf_net.parameters():
            p.requires_grad = False

        # HF with SpectralNorm + LayerNorm
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim,
                                use_spectral=True, use_layernorm=True).to(device)
        opt = torch.optim.Adam(self.hf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.hf_epochs):
            opt.zero_grad()
            with torch.no_grad():
                y_lf_pred = self.lf_net(X_hf_t)
            y_hf_pred = self.hf_net(X_hf_t, y_lf_pred)
            F.mse_loss(y_hf_pred, y_hf_t).backward()
            opt.step()

        # Fit BLR on HF features
        self.hf_net.eval()
        with torch.no_grad():
            y_lf_pred = self.lf_net(X_hf_t)
            Phi = self.hf_net.extract_features(X_hf_t, y_lf_pred).cpu().numpy()

        # BLR closed form
        A = self.alpha_blr * np.eye(self.hidden_dim) + self.beta_blr * Phi.T @ Phi
        self.A_inv = np.linalg.inv(A + 1e-6 * np.eye(A.shape[0]))
        self.m = self.beta_blr * self.A_inv @ Phi.T @ y_hf_s

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(device)

        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            Phi = self.hf_net.extract_features(X_t, y_lf).cpu().numpy()

        # BLR prediction
        mean_s = Phi @ self.m
        var_s = 1.0 / self.beta_blr + np.sum((Phi @ self.A_inv) * Phi, axis=1)
        std_s = np.sqrt(np.maximum(var_s, 1e-10))

        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        std = std_s * self.scaler_y.scale_[0]

        return mean, np.maximum(std, 1e-6)


# =============================================================================
# BO Utilities
# =============================================================================

def expected_improvement(mean: np.ndarray, std: np.ndarray, y_best: float,
                         xi: float = 0.01) -> np.ndarray:
    """Expected Improvement acquisition function"""
    std = np.maximum(std, 1e-6)
    z = (y_best - mean - xi) / std
    ei = (y_best - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
    return np.maximum(ei, 0)


def run_single_bo(model_class, model_name: str, X_lf: np.ndarray, y_lf: np.ndarray,
                  X_hf: np.ndarray, y_hf: np.ndarray, X_grid: np.ndarray,
                  f_hf, f_lf, budget: float = 30, cost_ratio: float = 0.1,
                  seed: int = 42) -> Dict:
    """Run single BO experiment"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    dim = X_grid.shape[1]
    cost_hf, cost_lf = 1.0, cost_ratio

    # Copy initial data
    X_lf_curr = X_lf.copy()
    y_lf_curr = y_lf.copy()
    X_hf_curr = X_hf.copy()
    y_hf_curr = y_hf.copy()

    # Track sampled indices
    sampled_hf = set()
    sampled_lf = set()
    for i, x in enumerate(X_grid):
        for xh in X_hf_curr:
            if np.allclose(x, xh):
                sampled_hf.add(i)
        for xl in X_lf_curr:
            if np.allclose(x, xl):
                sampled_lf.add(i)

    # Initial budget
    spent = len(X_lf_curr) * cost_lf + len(X_hf_curr) * cost_hf

    # BO loop
    history = []
    lf_counter = 0
    step = 0

    while spent < budget:
        step += 1

        # Decide fidelity (round-robin: 2 LF then 1 HF)
        lf_counter += 1
        eval_hf = (lf_counter >= 3)
        if eval_hf:
            lf_counter = 0

        cost_next = cost_hf if eval_hf else cost_lf
        if spent + cost_next > budget:
            break

        # Fit model
        model = model_class(dim)
        model.fit(X_lf_curr, y_lf_curr, X_hf_curr, y_hf_curr)

        # Get predictions
        mean, std = model.predict(X_grid)

        # Select next point
        y_best = y_hf_curr.min()

        # Create mask for unsampled points
        if eval_hf:
            mask = np.array([i not in sampled_hf for i in range(len(X_grid))])
        else:
            mask = np.array([i not in sampled_lf for i in range(len(X_grid))])

        if not mask.any():
            break

        # Use EI if model has proper uncertainty, else argmin
        has_real_uncertainty = 'BLR' in model_name or 'All' in model_name

        if has_real_uncertainty:
            ei = expected_improvement(mean, std, y_best)
            ei[~mask] = -np.inf
            next_idx = np.argmax(ei)
        else:
            mean_masked = mean.copy()
            mean_masked[~mask] = np.inf
            next_idx = np.argmin(mean_masked)

        # Evaluate
        x_next = X_grid[next_idx:next_idx+1]
        if eval_hf:
            y_next = f_hf(x_next)
            X_hf_curr = np.vstack([X_hf_curr, x_next])
            y_hf_curr = np.concatenate([y_hf_curr, y_next])
            sampled_hf.add(next_idx)
        else:
            y_next = f_lf(x_next, alpha=0.8)
            X_lf_curr = np.vstack([X_lf_curr, x_next])
            y_lf_curr = np.concatenate([y_lf_curr, y_next])
            sampled_lf.add(next_idx)

        spent += cost_next

        # Record
        best_hf = y_hf_curr.min()
        history.append({
            'step': step,
            'budget': spent,
            'best_hf': best_hf,
            'n_hf': len(X_hf_curr),
            'n_lf': len(X_lf_curr)
        })

    # Final metrics
    f_star = 0.397887  # Branin global minimum
    final_regret = y_hf_curr.min() - f_star

    return {
        'model': model_name,
        'final_regret': final_regret,
        'final_best': y_hf_curr.min(),
        'n_hf': len(X_hf_curr),
        'n_lf': len(X_lf_curr),
        'history': history
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("UQ Variants Comparison Test")
    print("=" * 70)
    print(f"Device: {device}")
    print()

    # Setup
    np.random.seed(42)
    torch.manual_seed(42)

    dim = 2
    n_grid = 50
    x1 = np.linspace(0, 1, n_grid)
    x2 = np.linspace(0, 1, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])

    # Initial samples
    n_init_lf = 10
    n_init_hf = 3

    idx_lf = np.random.choice(len(X_grid), n_init_lf, replace=False)
    X_lf_init = X_grid[idx_lf]
    y_lf_init = branin_lf(X_lf_init, alpha=0.8)

    idx_hf = np.random.choice(idx_lf, n_init_hf, replace=False)
    X_hf_init = X_grid[idx_hf]
    y_hf_init = branin_hf(X_hf_init)

    print(f"Grid: {n_grid}x{n_grid} = {len(X_grid)} points")
    print(f"Initial: LF={n_init_lf}, HF={n_init_hf}")
    print(f"Budget: 30 (cost_ratio=0.1)")
    print()

    # Models to compare
    models = [
        (SequentialBaseline, "Baseline"),
        (SequentialSpectralNorm, "+SpectralNorm"),
        (SequentialBLR, "+BLR"),
        (SequentialLayerNorm, "+LayerNorm"),
        (SequentialAll, "+All"),
    ]

    # Run experiments
    results = []
    for model_class, model_name in models:
        print(f"Running {model_name}...", end=" ", flush=True)
        result = run_single_bo(
            model_class, model_name,
            X_lf_init.copy(), y_lf_init.copy(),
            X_hf_init.copy(), y_hf_init.copy(),
            X_grid, branin_hf, branin_lf,
            budget=30, cost_ratio=0.1, seed=42
        )
        results.append(result)
        print(f"Regret: {result['final_regret']:.6f}, HF: {result['n_hf']}, LF: {result['n_lf']}")

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Model':<20} {'Regret':>12} {'Best':>12} {'HF':>6} {'LF':>6}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<20} {r['final_regret']:>12.6f} {r['final_best']:>12.4f} {r['n_hf']:>6} {r['n_lf']:>6}")

    # Find best
    best_result = min(results, key=lambda x: x['final_regret'])
    print()
    print(f"Best: {best_result['model']} (regret={best_result['final_regret']:.6f})")

    # Plot convergence
    plt.figure(figsize=(10, 6))
    for r in results:
        budgets = [h['budget'] for h in r['history']]
        bests = [h['best_hf'] for h in r['history']]
        plt.plot(budgets, bests, '-o', label=r['model'], markersize=3)

    plt.axhline(y=0.397887, color='gray', linestyle='--', alpha=0.5, label='Global min')
    plt.xlabel('Budget')
    plt.ylabel('Best HF value')
    plt.title('UQ Variants Comparison on Branin-2D (Favorable)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'uq_variants_test_{timestamp}')
    output_dir.mkdir(exist_ok=True)

    plt.savefig(output_dir / 'convergence.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved to {output_dir}/")

    # Save results
    import json
    with open(output_dir / 'results.json', 'w') as f:
        json.dump([{k: v for k, v in r.items() if k != 'history'} for r in results], f, indent=2)

    plt.show()


if __name__ == '__main__':
    main()
