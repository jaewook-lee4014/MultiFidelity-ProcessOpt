#!/usr/bin/env python
"""
Adaptive Fidelity Selection Benchmark

Based on benchmark_parallel_lf_blr.py but replaces round-robin fidelity scheduling
with cost-weighted EI adaptive selection. Tests whether DNN vs MFGP conclusions
hold under adaptive scheduling (reviewer question).

Key changes from lf_blr:
- MFGP.predict_lf() returns actual LF posterior (fidelity=0) instead of HF
- DNN models get HF-BLR (predict_hf_blr) in addition to existing LF-BLR
- adaptive_select_next() jointly selects (candidate, fidelity) via cost-weighted EI
- Asymmetric masking: HF excludes sampled_hf only; LF excludes sampled_lf | sampled_hf
- Fidelity decision logging per step

Usage:
    python benchmark_adaptive.py --n-seeds 20 --n-workers 48
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube
from scipy.spatial.distance import cdist

# RDKit for molecular descriptors
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import argparse
import multiprocessing as mp
from multiprocessing import Pool, Manager
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Set multiprocessing start method for CUDA compatibility
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

# BoTorch
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

# Local imports
from synthetic_functions import (
    branin_hf, branin_lf, park_hf, park_lf,
    SCENARIOS, FUNCTIONS
)
from baselines import SparseMFGP, NARGP, DKLMultiFidelity


# =============================================================================
# Network Architectures
# =============================================================================

class LFNetwork(nn.Module):
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
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        layers = []
        in_dim = input_dim + 1
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
    def __init__(self, input_dim, bottleneck_dim=16):
        super().__init__()
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, x):
        return x + self.up(F.relu(self.down(x)))


# =============================================================================
# Bayesian Linear Regression for Last Layer
# =============================================================================

class BayesianLinearRegression:
    """Bayesian Linear Regression for uncertainty quantification on last layer."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha  # Prior precision
        self.beta = beta    # Noise precision
        self.m_N = None     # Posterior mean
        self.S_N = None     # Posterior covariance
        self.fitted = False

    def fit(self, Phi: np.ndarray, y: np.ndarray):
        Phi = np.asarray(Phi)
        y = np.asarray(y).flatten()

        N, D = Phi.shape

        # Add bias term
        Phi_bias = np.hstack([Phi, np.ones((N, 1))])
        D_bias = D + 1

        # Prior
        S_0_inv = self.alpha * np.eye(D_bias)

        # Posterior covariance
        S_N_inv = S_0_inv + self.beta * (Phi_bias.T @ Phi_bias)

        # Regularization for numerical stability
        S_N_inv += 1e-6 * np.eye(D_bias)

        try:
            self.S_N = np.linalg.inv(S_N_inv)
        except np.linalg.LinAlgError:
            self.S_N = np.linalg.pinv(S_N_inv)

        # Posterior mean
        self.m_N = self.beta * (self.S_N @ Phi_bias.T @ y)
        self.fitted = True

    def predict(self, Phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("BLR not fitted yet")

        Phi = np.asarray(Phi)
        N = Phi.shape[0]

        # Add bias term
        Phi_bias = np.hstack([Phi, np.ones((N, 1))])

        # Predictive mean
        mean = Phi_bias @ self.m_N

        # Predictive variance
        var = np.zeros(N)
        for i in range(N):
            phi_i = Phi_bias[i:i+1].T
            var[i] = 1.0 / self.beta + phi_i.T @ self.S_N @ phi_i

        std = np.sqrt(np.maximum(var, 1e-10))

        return mean.flatten(), std.flatten()


# =============================================================================
# Model Classes with LF-BLR + HF-BLR Support
# =============================================================================

class BaseModel:
    def __init__(self, input_dim: int, hidden_dim: int = 64, device=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # LF-BLR components
        self.lf_blr = None
        self.has_lf_blr = False

        # HF-BLR components (new for adaptive)
        self.hf_blr = None
        self.has_hf_blr = False

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        raise NotImplementedError

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def predict_lf(self, X) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def predict_hf_blr(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """Predict HF values with BLR uncertainty using HF network features."""
        raise NotImplementedError

    def _fit_lf_blr(self, X_lf_t, y_lf):
        """Fit BLR on LF network's last layer features"""
        if not hasattr(self, 'lf_net'):
            return

        self.lf_net.eval()
        with torch.no_grad():
            features = self.lf_net.extract_features(X_lf_t).cpu().numpy()

        self.lf_blr = BayesianLinearRegression(alpha=1.0, beta=1.0)
        self.lf_blr.fit(features, y_lf)
        self.has_lf_blr = True

    def _fit_hf_blr(self, X_hf_t, y_hf_pred_lf, y_hf_s):
        """Fit BLR on HF network's last layer features.

        Args:
            X_hf_t: Scaled HF input tensor on device
            y_hf_pred_lf: LF predictions on HF inputs (tensor on device)
            y_hf_s: Scaled HF targets (numpy)
        """
        if not hasattr(self, 'hf_net'):
            return

        self.hf_net.eval()
        with torch.no_grad():
            features = self.hf_net.extract_features(X_hf_t, y_hf_pred_lf).cpu().numpy()

        self.hf_blr = BayesianLinearRegression(alpha=1.0, beta=1.0)
        self.hf_blr.fit(features, y_hf_s)
        self.has_hf_blr = True

    def _predict_hf_blr_impl(self, X):
        """Shared HF-BLR prediction for standard LF+HF network models."""
        if not self.has_hf_blr:
            raise RuntimeError("HF-BLR not fitted")

        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        self.hf_net.eval()

        with torch.no_grad():
            y_lf_pred = self.lf_net(X_t)
            features = self.hf_net.extract_features(X_t, y_lf_pred).cpu().numpy()

        mean_s, std_s = self.hf_blr.predict(features)
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        std = std_s * self.scaler_y.scale_[0]
        return mean, np.maximum(std, 1e-6)


class MFGP(BaseModel):
    """MFGP with proper LF prediction via fidelity=0"""

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        n_lf, n_hf = len(X_lf), len(X_hf)
        X_lf_fid = np.hstack([X_lf, np.zeros((n_lf, 1))])
        X_hf_fid = np.hstack([X_hf, np.ones((n_hf, 1))])
        X_all = np.vstack([X_lf_fid, X_hf_fid])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        train_X = torch.tensor(X_all, dtype=torch.float64).to(self.device)
        train_Y = torch.tensor(y_all, dtype=torch.float64).unsqueeze(-1).to(self.device)
        self.model = SingleTaskMultiFidelityGP(
            train_X, train_Y,
            data_fidelities=[self.input_dim],
            outcome_transform=Standardize(m=1)
        ).to(self.device)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self.is_fitted = True

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_fid = np.hstack([X, np.ones((len(X), 1))])
        X_tensor = torch.tensor(X_fid, dtype=torch.float64).to(self.device)
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X_tensor)
            mean = posterior.mean.cpu().numpy().flatten()
            std = posterior.variance.sqrt().cpu().numpy().flatten()
        return mean, np.maximum(std, 1e-6)

    def predict_lf(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """Actual LF posterior via fidelity=0 (fixed from lf_blr version)"""
        X_fid = np.hstack([X, np.zeros((len(X), 1))])
        X_tensor = torch.tensor(X_fid, dtype=torch.float64).to(self.device)
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X_tensor)
            mean = posterior.mean.cpu().numpy().flatten()
            std = posterior.variance.sqrt().cpu().numpy().flatten()
        return mean, np.maximum(std, 1e-6)


class Sequential(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, lf_epochs=200, hf_epochs=100, device=None):
        super().__init__(input_dim, hidden_dim, device)
        self.lf_epochs = lf_epochs
        self.hf_epochs = hf_epochs

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()
        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        # Fit LF-BLR after LF training
        self._fit_lf_blr(X_lf_t, y_lf_s)

        for p in self.lf_net.parameters():
            p.requires_grad = False
        opt = torch.optim.Adam(self.hf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.hf_epochs):
            opt.zero_grad()
            with torch.no_grad():
                y_lf_pred = self.lf_net(X_hf_t)
            y_hf_pred = self.hf_net(X_hf_t, y_lf_pred)
            F.mse_loss(y_hf_pred, y_hf_t).backward()
            opt.step()

        # Fit HF-BLR after HF training
        if len(X_hf) >= 3:
            try:
                self.lf_net.eval()
                with torch.no_grad():
                    y_lf_pred = self.lf_net(X_hf_t)
                self._fit_hf_blr(X_hf_t, y_lf_pred, y_hf_s)
            except Exception:
                pass

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()

        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()

        if self.has_lf_blr:
            mean_s, std_s = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return mean, np.maximum(std, 1e-6)
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return mean, np.ones_like(mean) * 0.1

    def predict_hf_blr(self, X):
        return self._predict_hf_blr_impl(X)


class Progressive(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, epochs_per_stage=50, device=None):
        super().__init__(input_dim, hidden_dim, device)
        self.epochs_per_stage = epochs_per_stage

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()
        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3)
        for _ in range(self.epochs_per_stage * 2):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        # Fit LF-BLR
        self._fit_lf_blr(X_lf_t, y_lf_s)

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

        # Fit HF-BLR
        if len(X_hf) >= 3:
            try:
                self.lf_net.eval()
                with torch.no_grad():
                    y_lf_pred = self.lf_net(X_hf_t)
                self._fit_hf_blr(X_hf_t, y_lf_pred, y_hf_s)
            except Exception:
                pass

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()

        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()

        if self.has_lf_blr:
            mean_s, std_s = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return mean, np.maximum(std, 1e-6)
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return mean, np.ones_like(mean) * 0.1

    def predict_hf_blr(self, X):
        return self._predict_hf_blr_impl(X)


class Curriculum(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, epochs=200, device=None):
        super().__init__(input_dim, hidden_dim, device)
        self.epochs = epochs

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()
        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3)
        for _ in range(self.epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        # Fit LF-BLR
        self._fit_lf_blr(X_lf_t, y_lf_s)

        for p in self.lf_net.parameters():
            p.requires_grad = False
        with torch.no_grad():
            y_lf_pred = self.lf_net(X_hf_t)
            residuals = torch.abs(y_hf_t - y_lf_pred).squeeze()
        sorted_idx = torch.argsort(residuals)
        opt = torch.optim.Adam(self.hf_net.parameters(), lr=1e-3)
        n_hf = len(X_hf_t)
        for epoch in range(self.epochs):
            n_use = min(n_hf, max(2, int((epoch + 1) / self.epochs * n_hf)))
            idx = sorted_idx[:n_use]
            opt.zero_grad()
            with torch.no_grad():
                y_lf_sub = self.lf_net(X_hf_t[idx])
            y_hf_pred = self.hf_net(X_hf_t[idx], y_lf_sub)
            F.mse_loss(y_hf_pred, y_hf_t[idx]).backward()
            opt.step()

        # Fit HF-BLR
        if len(X_hf) >= 3:
            try:
                self.lf_net.eval()
                with torch.no_grad():
                    y_lf_pred = self.lf_net(X_hf_t)
                self._fit_hf_blr(X_hf_t, y_lf_pred, y_hf_s)
            except Exception:
                pass

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()

        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()

        if self.has_lf_blr:
            mean_s, std_s = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return mean, np.maximum(std, 1e-6)
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return mean, np.ones_like(mean) * 0.1

    def predict_hf_blr(self, X):
        return self._predict_hf_blr_impl(X)


class TwoStageJoint(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, stage1_epochs=100, stage2_epochs=100, device=None):
        super().__init__(input_dim, hidden_dim, device)
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()
        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3)
        for _ in range(self.stage1_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        # Fit LF-BLR after stage 1
        self._fit_lf_blr(X_lf_t, y_lf_s)

        opt = torch.optim.Adam(list(self.lf_net.parameters()) + list(self.hf_net.parameters()), lr=1e-4)
        for _ in range(self.stage2_epochs):
            opt.zero_grad()
            lf_loss = F.mse_loss(self.lf_net(X_lf_t), y_lf_t)
            y_lf_pred = self.lf_net(X_hf_t)
            hf_loss = F.mse_loss(self.hf_net(X_hf_t, y_lf_pred), y_hf_t)
            (0.3 * lf_loss + 0.7 * hf_loss).backward()
            opt.step()

        # Re-fit LF-BLR after joint training
        self._fit_lf_blr(X_lf_t, y_lf_s)

        # Fit HF-BLR
        if len(X_hf) >= 3:
            try:
                self.lf_net.eval()
                with torch.no_grad():
                    y_lf_pred = self.lf_net(X_hf_t)
                self._fit_hf_blr(X_hf_t, y_lf_pred, y_hf_s)
            except Exception:
                pass

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()

        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()

        if self.has_lf_blr:
            mean_s, std_s = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return mean, np.maximum(std, 1e-6)
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return mean, np.ones_like(mean) * 0.1

    def predict_hf_blr(self, X):
        return self._predict_hf_blr_impl(X)


class DNGOJoint(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, epochs=300, alpha=0.5, device=None):
        super().__init__(input_dim, hidden_dim, device)
        self.epochs = epochs
        self.alpha = alpha

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()
        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(list(self.lf_net.parameters()) + list(self.hf_net.parameters()), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.epochs):
            opt.zero_grad()
            lf_loss = F.mse_loss(self.lf_net(X_lf_t), y_lf_t)
            with torch.no_grad():
                y_lf_pred = self.lf_net(X_hf_t)
            hf_loss = F.mse_loss(self.hf_net(X_hf_t, y_lf_pred), y_hf_t)
            ((1 - self.alpha) * lf_loss + self.alpha * hf_loss).backward()
            opt.step()

        # Fit LF-BLR
        self._fit_lf_blr(X_lf_t, y_lf_s)

        # Fit HF-BLR
        if len(X_hf) >= 3:
            try:
                self.lf_net.eval()
                with torch.no_grad():
                    y_lf_pred = self.lf_net(X_hf_t)
                self._fit_hf_blr(X_hf_t, y_lf_pred, y_hf_s)
            except Exception:
                pass

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()

        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()

        if self.has_lf_blr:
            mean_s, std_s = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return mean, np.maximum(std, 1e-6)
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return mean, np.ones_like(mean) * 0.1

    def predict_hf_blr(self, X):
        return self._predict_hf_blr_impl(X)


class DNGOGradient(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, epochs=300, device=None):
        super().__init__(input_dim, hidden_dim, device)
        self.epochs = epochs

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()
        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
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

        # Fit LF-BLR
        self._fit_lf_blr(X_lf_t, y_lf_s)

        # Fit HF-BLR
        if len(X_hf) >= 3:
            try:
                self.lf_net.eval()
                with torch.no_grad():
                    y_lf_pred = self.lf_net(X_hf_t)
                self._fit_hf_blr(X_hf_t, y_lf_pred, y_hf_s)
            except Exception:
                pass

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()

        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()

        if self.has_lf_blr:
            mean_s, std_s = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return mean, np.maximum(std, 1e-6)
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return mean, np.ones_like(mean) * 0.1

    def predict_hf_blr(self, X):
        return self._predict_hf_blr_impl(X)


class KnowledgeDistillation(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, lf_epochs=200, hf_epochs=100, alpha_kd=0.3, temperature=3.0, device=None):
        super().__init__(input_dim, hidden_dim, device)
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
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        # Fit LF-BLR
        self._fit_lf_blr(X_lf_t, y_lf_s)

        for p in self.lf_net.parameters():
            p.requires_grad = False
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

        # Fit HF-BLR
        if len(X_hf) >= 3:
            try:
                self.lf_net.eval()
                with torch.no_grad():
                    y_lf_pred = self.lf_net(X_hf_t)
                self._fit_hf_blr(X_hf_t, y_lf_pred, y_hf_s)
            except Exception:
                pass

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()

        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()

        if self.has_lf_blr:
            mean_s, std_s = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return mean, np.maximum(std, 1e-6)
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return mean, np.ones_like(mean) * 0.1

    def predict_hf_blr(self, X):
        return self._predict_hf_blr_impl(X)


class DomainAdaptationMMD(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, lf_epochs=200, hf_epochs=100, lambda_mmd=0.1, device=None):
        super().__init__(input_dim, hidden_dim, device)
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
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        # Fit LF-BLR after initial LF training
        self._fit_lf_blr(X_lf_t, y_lf_s)

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
            min_n = min(len(lf_features), len(hf_features))
            mmd = self._mmd_loss(lf_features[:min_n], hf_features[:min_n])
            (task_loss + self.lambda_mmd * mmd).backward()
            opt.step()

        # Re-fit LF-BLR after joint training
        self._fit_lf_blr(X_lf_t, y_lf_s)

        # Fit HF-BLR
        if len(X_hf) >= 3:
            try:
                self.lf_net.eval()
                with torch.no_grad():
                    y_lf_pred = self.lf_net(X_hf_t)
                self._fit_hf_blr(X_hf_t, y_lf_pred, y_hf_s)
            except Exception:
                pass

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()

        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()

        if self.has_lf_blr:
            mean_s, std_s = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return mean, np.maximum(std, 1e-6)
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return mean, np.ones_like(mean) * 0.1

    def predict_hf_blr(self, X):
        return self._predict_hf_blr_impl(X)


class SoftParameterSharing(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, epochs=200, lambda_soft=0.01, device=None):
        super().__init__(input_dim, hidden_dim, device)
        self.epochs = epochs
        self.lambda_soft = lambda_soft

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()
        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(list(self.lf_net.parameters()) + list(self.hf_net.parameters()), lr=1e-3)
        for _ in range(self.epochs):
            opt.zero_grad()
            lf_loss = F.mse_loss(self.lf_net(X_lf_t), y_lf_t)
            y_lf_pred = self.lf_net(X_hf_t)
            hf_loss = F.mse_loss(self.hf_net(X_hf_t, y_lf_pred), y_hf_t)
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

        # Fit LF-BLR
        self._fit_lf_blr(X_lf_t, y_lf_s)

        # Fit HF-BLR
        if len(X_hf) >= 3:
            try:
                self.lf_net.eval()
                with torch.no_grad():
                    y_lf_pred = self.lf_net(X_hf_t)
                self._fit_hf_blr(X_hf_t, y_lf_pred, y_hf_s)
            except Exception:
                pass

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()

        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()

        if self.has_lf_blr:
            mean_s, std_s = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return mean, np.maximum(std, 1e-6)
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return mean, np.ones_like(mean) * 0.1

    def predict_hf_blr(self, X):
        return self._predict_hf_blr_impl(X)


class PseudoLabeling(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, lf_epochs=200, hf_epochs=100, pseudo_weight=0.5, device=None):
        super().__init__(input_dim, hidden_dim, device)
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
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        # Fit LF-BLR
        self._fit_lf_blr(X_lf_t, y_lf_s)

        self.lf_net.eval()
        with torch.no_grad():
            lf_pred_on_hf = self.lf_net(X_hf_t)
            offset = (y_hf_t - lf_pred_on_hf).mean()
            lf_pred_all = self.lf_net(X_lf_t)
            pseudo_labels = lf_pred_all + offset
        for p in self.lf_net.parameters():
            p.requires_grad = False
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

        # Fit HF-BLR
        if len(X_hf) >= 3:
            try:
                self.lf_net.eval()
                with torch.no_grad():
                    y_lf_pred = self.lf_net(X_hf_t)
                self._fit_hf_blr(X_hf_t, y_lf_pred, y_hf_s)
            except Exception:
                pass

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()

        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()

        if self.has_lf_blr:
            mean_s, std_s = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return mean, np.maximum(std, 1e-6)
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return mean, np.ones_like(mean) * 0.1

    def predict_hf_blr(self, X):
        return self._predict_hf_blr_impl(X)


class Adapter(BaseModel):
    """Adapter model - uses backbone instead of lf_net, needs special handling"""

    def __init__(self, input_dim, hidden_dim=64, lf_epochs=200, adapter_epochs=100, bottleneck_dim=16, device=None):
        super().__init__(input_dim, hidden_dim, device)
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
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        ).to(self.device)
        self.out_layer = nn.Linear(self.hidden_dim, 1).to(self.device)
        self.adapters = nn.ModuleList([AdapterLayer(self.hidden_dim, self.bottleneck_dim) for _ in range(2)]).to(self.device)
        self.hf_out = nn.Linear(self.hidden_dim, 1).to(self.device)
        opt = torch.optim.Adam(list(self.backbone.parameters()) + list(self.out_layer.parameters()), lr=1e-3)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            h = self.backbone(X_lf_t)
            F.mse_loss(self.out_layer(h), y_lf_t).backward()
            opt.step()

        # Fit LF-BLR on backbone features
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(X_lf_t).cpu().numpy()
        self.lf_blr = BayesianLinearRegression(alpha=1.0, beta=1.0)
        self.lf_blr.fit(features, y_lf_s)
        self.has_lf_blr = True

        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.out_layer.parameters():
            p.requires_grad = False
        opt = torch.optim.Adam(list(self.adapters.parameters()) + list(self.hf_out.parameters()), lr=1e-3)
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

        # Fit HF-BLR on adapter features
        if len(X_hf) >= 3:
            try:
                self.backbone.eval()
                with torch.no_grad():
                    h = X_hf_t
                    adapter_idx = 0
                    for module in self.backbone:
                        h = module(h)
                        if isinstance(module, nn.Tanh) and adapter_idx < len(self.adapters):
                            h = self.adapters[adapter_idx](h)
                            adapter_idx += 1
                    hf_features = h.cpu().numpy()
                self.hf_blr = BayesianLinearRegression(alpha=1.0, beta=1.0)
                self.hf_blr.fit(hf_features, y_hf_s)
                self.has_hf_blr = True
            except Exception:
                pass

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
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

    def predict_lf(self, X):
        """Predict LF using backbone + BLR"""
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.backbone.eval()

        with torch.no_grad():
            features = self.backbone(X_t).cpu().numpy()

        if self.has_lf_blr:
            mean_s, std_s = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return mean, np.maximum(std, 1e-6)
        else:
            with torch.no_grad():
                mean_s = self.out_layer(self.backbone(X_t)).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return mean, np.ones_like(mean) * 0.1

    def predict_hf_blr(self, X):
        """Predict HF using adapter features + BLR"""
        if not self.has_hf_blr:
            raise RuntimeError("HF-BLR not fitted")

        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.backbone.eval()

        with torch.no_grad():
            h = X_t
            adapter_idx = 0
            for module in self.backbone:
                h = module(h)
                if isinstance(module, nn.Tanh) and adapter_idx < len(self.adapters):
                    h = self.adapters[adapter_idx](h)
                    adapter_idx += 1
            features = h.cpu().numpy()

        mean_s, std_s = self.hf_blr.predict(features)
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        std = std_s * self.scaler_y.scale_[0]
        return mean, np.maximum(std, 1e-6)


# =============================================================================
# Benchmark Classes (same as lf_blr)
# =============================================================================

class SyntheticBenchmark:
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
        corr = np.corrcoef(self.y_hf, self.y_lf)[0, 1]
        self.r2 = corr ** 2

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


class ChemistryBenchmark:
    def __init__(self, name, csv_path, cost_ratio, use_smiles=False, minimize=True, negate=False, pca_dim=10):
        self.name = name
        self.cost_ratio = cost_ratio
        self.minimize = minimize
        self.negate = negate
        self.pca_dim = pca_dim
        df = pd.read_csv(csv_path)
        if use_smiles:
            smiles_col = [c for c in df.columns if 'smiles' in c.lower()]
            if smiles_col:
                self.X = self._smiles_to_rdkit_features(df[smiles_col[0]].values)
            else:
                self.X = self._smiles_to_rdkit_features(df.iloc[:, 0].values)
        else:
            feature_cols = [c for c in df.columns if c not in ['HF', 'LF']]
            self.X = df[feature_cols].values
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        self.y_hf = df['HF'].values.flatten()
        self.y_lf = df['LF'].values.flatten()

        if negate:
            self.y_hf = -self.y_hf
            self.y_lf = -self.y_lf

        self.f_star = self.y_hf.min()
        self.n_candidates = len(self.X)
        self.dim = self.X.shape[1]
        corr = np.corrcoef(self.y_hf, self.y_lf)[0, 1]
        self.r2 = corr ** 2

    def _smiles_to_rdkit_features(self, smiles_list):
        descriptor_names = [desc[0] for desc in Descriptors._descList]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

        features = []
        valid_indices = []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                desc = calc.CalcDescriptors(mol)
                features.append(desc)
                valid_indices.append(i)
            else:
                features.append([0.0] * len(descriptor_names))
                valid_indices.append(i)

        features = np.array(features, dtype=np.float64)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        pca = PCA(n_components=self.pca_dim)
        features_pca = pca.fit_transform(features_scaled)

        return features_pca

    def evaluate_hf(self, indices):
        return self.y_hf[indices.astype(int).flatten()]

    def evaluate_lf(self, indices):
        return self.y_lf[indices.astype(int).flatten()]


# =============================================================================
# Initial Sampling Methods (same as lf_blr)
# =============================================================================

def furthest_point_sampling(X: np.ndarray, n_samples: int, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    n_candidates = len(X)
    n_samples = min(n_samples, n_candidates)
    selected = [np.random.randint(n_candidates)]
    for _ in range(n_samples - 1):
        selected_X = X[selected]
        distances = cdist(X, selected_X, metric='euclidean')
        min_distances = distances.min(axis=1)
        min_distances[selected] = -np.inf
        next_idx = np.argmax(min_distances)
        selected.append(next_idx)
    return np.array(selected)


def latin_hypercube_sampling(bounds: np.ndarray, n_samples: int, seed: int = 42) -> np.ndarray:
    n_dims = len(bounds)
    sampler = LatinHypercube(d=n_dims, seed=seed)
    samples = sampler.random(n=n_samples)
    for i in range(n_dims):
        samples[:, i] = bounds[i, 0] + samples[:, i] * (bounds[i, 1] - bounds[i, 0])
    return samples


def find_nearest_candidates(X_candidates: np.ndarray, X_samples: np.ndarray) -> np.ndarray:
    distances = cdist(X_samples, X_candidates, metric='euclidean')
    return np.argmin(distances, axis=1)


# =============================================================================
# Acquisition Function
# =============================================================================

def expected_improvement(mean, std, y_best, xi=0.01):
    with np.errstate(divide='ignore', invalid='ignore'):
        imp = y_best - mean - xi
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std < 1e-10] = 0.0
    return ei


# =============================================================================
# Adaptive Fidelity Selection (Alpha-weighted Cost-weighted EI)
# =============================================================================

GP_MODELS = (MFGP, SparseMFGP, NARGP, DKLMultiFidelity)


def compute_alpha(y_lf_paired, y_hf_paired):
    """
    Compute LF-HF correlation from paired observations.
    alpha = |corr(y_lf, y_hf)| clipped to [0.1, 1.0].
    Returns 0.5 if insufficient data (<3 pairs).
    """
    if len(y_lf_paired) < 3 or len(y_hf_paired) < 3:
        return 0.5  # prior before enough data
    try:
        corr = np.corrcoef(y_lf_paired, y_hf_paired)[0, 1]
        alpha = np.clip(np.abs(corr), 0.1, 1.0)
        if np.isnan(alpha):
            return 0.5
        return float(alpha)
    except Exception:
        return 0.5


def adaptive_select_next(model, X_candidates, y_best_hf, y_best_lf, rho,
                         sampled_lf, sampled_hf, remaining_budget,
                         is_gp_model, alpha, consecutive_lf_count=0):
    """
    Alpha-weighted cost-weighted EI for joint (candidate, fidelity) selection.

    HF path: score_hf = EI_hf(x; y_best_hf) / 1.0
    LF path: score_lf = alpha * EI_lf(x; y_best_lf) / rho

    alpha = |corr(y_lf, y_hf)| from paired data, discounts LF EI to prevent
    scale imbalance that caused degenerate all-LF behavior.

    Asymmetric masking:
      - HF: exclude sampled_hf only (LF->HF promotion allowed)
      - LF: exclude sampled_lf | sampled_hf

    Safety: force HF evaluation after max_lf_streak consecutive LF steps
    to ensure sufficient HF data for model calibration.

    Returns: (next_idx, eval_hf: bool, cost: float, score_max: float)
    """
    max_lf_streak = max(3, int(2.0 / rho))  # rho=0.1 -> 20, rho=0.5 -> 4

    # --- HF predictions + EI ---
    if is_gp_model:
        mean_hf, std_hf = model.predict(X_candidates)
    else:
        if model.has_hf_blr:
            mean_hf, std_hf = model.predict_hf_blr(X_candidates)
        else:
            # Fallback: use DNN point prediction with LF std as proxy
            # (constant 0.1 is too small, causing near-zero HF EI)
            mean_hf, _ = model.predict(X_candidates)
            _, std_lf_proxy = model.predict_lf(X_candidates)
            std_hf = np.maximum(std_lf_proxy, 0.1)

    ei_hf = expected_improvement(mean_hf, std_hf, y_best_hf)
    score_hf = ei_hf / 1.0  # cost_hf = 1.0

    # --- LF predictions + EI ---
    mean_lf, std_lf = model.predict_lf(X_candidates)
    ei_lf = expected_improvement(mean_lf, std_lf, y_best_lf)
    score_lf = (alpha * ei_lf) / rho  # alpha-discounted, cost-weighted

    # --- Asymmetric masking ---
    # HF: only exclude already-HF-evaluated (allows LF->HF promotion)
    if sampled_hf:
        score_hf[list(sampled_hf)] = -np.inf
    # LF: exclude both LF and HF evaluated
    excluded_lf = sampled_lf | sampled_hf
    if excluded_lf:
        score_lf[list(excluded_lf)] = -np.inf

    # --- Budget check ---
    if remaining_budget < 1.0:
        score_hf[:] = -np.inf
    if remaining_budget < rho:
        score_lf[:] = -np.inf

    # --- Safety: force HF after too many consecutive LF ---
    if consecutive_lf_count >= max_lf_streak and remaining_budget >= 1.0:
        best_hf_idx = int(np.argmax(score_hf))
        if score_hf[best_hf_idx] > -np.inf:
            return best_hf_idx, True, 1.0, float(score_hf[best_hf_idx])

    # --- Joint argmax ---
    best_hf_idx = int(np.argmax(score_hf))
    best_lf_idx = int(np.argmax(score_lf))

    best_hf_score = float(score_hf[best_hf_idx])
    best_lf_score = float(score_lf[best_lf_idx])

    # Both infeasible
    if best_hf_score == -np.inf and best_lf_score == -np.inf:
        return -1, True, 0.0, -np.inf

    if best_hf_score >= best_lf_score:
        return best_hf_idx, True, 1.0, best_hf_score
    else:
        return best_lf_idx, False, rho, best_lf_score


# =============================================================================
# Adaptive BO Loop
# =============================================================================

def run_bo_adaptive(benchmark, model_class, budget, seed=42, device=None,
                    sampling_method='fps'):
    """
    Run Bayesian Optimization with alpha-weighted cost-weighted EI adaptive
    fidelity selection.

    Key difference from round-robin (lf_blr):
    - Joint (candidate, fidelity) selection via cost-weighted EI
    - alpha = |corr(y_lf, y_hf)| from paired data discounts LF EI
    - HF EI uses y_best_hf incumbent; LF EI uses y_best_lf incumbent
    - Asymmetric masking: HF excludes sampled_hf; LF excludes sampled_lf|sampled_hf
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    rho = benchmark.cost_ratio
    n_candidates = benchmark.n_candidates
    is_gp = (model_class in GP_MODELS)

    init_budget = 0.1 * budget
    n_init_hf = max(2, int(init_budget * 0.5 / 1.0))
    n_init_lf = max(2, int(init_budget * 0.5 / rho))
    n_init_total = n_init_lf + n_init_hf

    # Initial sampling
    if sampling_method == 'lhs':
        bounds = np.array([[0, 1]] * benchmark.dim)
        lhs_samples = latin_hypercube_sampling(bounds, n_init_total, seed)
        X_min, X_max = benchmark.X.min(axis=0), benchmark.X.max(axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1
        lhs_samples_scaled = X_min + lhs_samples * X_range
        init_indices = find_nearest_candidates(benchmark.X, lhs_samples_scaled)
        init_indices = list(dict.fromkeys(init_indices))
        if len(init_indices) < n_init_total:
            remaining = n_init_total - len(init_indices)
            available = set(range(n_candidates)) - set(init_indices)
            if available:
                extra = furthest_point_sampling(
                    benchmark.X[list(available)],
                    remaining,
                    seed + 1000
                )
                extra_indices = [list(available)[i] for i in extra]
                init_indices.extend(extra_indices)
    else:
        init_indices = furthest_point_sampling(benchmark.X, n_init_total, seed).tolist()

    lf_indices = set(init_indices[:n_init_lf])
    hf_indices = set(init_indices[n_init_lf:n_init_lf + n_init_hf])

    X_lf = benchmark.X[list(lf_indices)]
    y_lf = benchmark.evaluate_lf(np.array(list(lf_indices)))
    X_hf = benchmark.X[list(hf_indices)]
    y_hf = benchmark.evaluate_hf(np.array(list(hf_indices)))

    current_budget = n_init_lf * rho + n_init_hf * 1.0

    regrets = [max(0, y_hf.min() - benchmark.f_star)]
    budgets = [current_budget]
    fidelity_decisions = []

    iteration = 0
    max_iter = 500
    consecutive_lf_count = 0

    while current_budget < budget and iteration < max_iter:
        iteration += 1
        remaining = budget - current_budget

        if remaining < min(rho, 1.0):
            break

        try:
            model = model_class(benchmark.X.shape[1], device=device)
            model.fit(X_lf, y_lf, X_hf, y_hf)

            y_best_hf = y_hf.min()
            y_best_lf = y_lf.min()

            # Compute alpha from paired observations (indices in both LF and HF)
            paired_indices = sorted(lf_indices & hf_indices)
            if len(paired_indices) >= 3:
                y_lf_paired = benchmark.evaluate_lf(np.array(paired_indices))
                y_hf_paired = benchmark.evaluate_hf(np.array(paired_indices))
                alpha = compute_alpha(y_lf_paired, y_hf_paired)
            else:
                alpha = 0.5  # default prior

            next_idx, eval_hf, cost, score_max = adaptive_select_next(
                model, benchmark.X, y_best_hf, y_best_lf, rho,
                lf_indices, hf_indices, remaining, is_gp, alpha,
                consecutive_lf_count
            )

            if next_idx < 0:
                break

            if eval_hf:
                hf_indices.add(next_idx)
                X_hf = benchmark.X[list(hf_indices)]
                y_hf = benchmark.evaluate_hf(np.array(list(hf_indices)))
                consecutive_lf_count = 0
            else:
                lf_indices.add(next_idx)
                X_lf = benchmark.X[list(lf_indices)]
                y_lf = benchmark.evaluate_lf(np.array(list(lf_indices)))
                consecutive_lf_count += 1

            current_budget += cost

        except Exception as e:
            available = set(range(n_candidates)) - (lf_indices | hf_indices)
            if available:
                next_idx = np.random.choice(list(available))
                if remaining >= 1.0:
                    eval_hf = True
                    cost = 1.0
                    hf_indices.add(next_idx)
                    X_hf = benchmark.X[list(hf_indices)]
                    y_hf = benchmark.evaluate_hf(np.array(list(hf_indices)))
                    consecutive_lf_count = 0
                elif remaining >= rho:
                    eval_hf = False
                    cost = rho
                    lf_indices.add(next_idx)
                    X_lf = benchmark.X[list(lf_indices)]
                    y_lf = benchmark.evaluate_lf(np.array(list(lf_indices)))
                    consecutive_lf_count += 1
                else:
                    break
                current_budget += cost
                score_max = float('nan')
                alpha = float('nan')
            else:
                break

        regrets.append(max(0, y_hf.min() - benchmark.f_star))
        budgets.append(current_budget)
        fidelity_decisions.append({
            'step': iteration,
            'fidelity_chosen': 1 if eval_hf else 0,
            'candidate_idx': int(next_idx),
            'cost': cost,
            'score_max': score_max,
            'alpha': alpha,
            'consecutive_lf_count': consecutive_lf_count,
            'n_hf_so_far': len(hf_indices),
            'n_lf_so_far': len(lf_indices),
            'remaining_budget': remaining - cost,
        })

    return {
        'regrets': regrets,
        'budgets': budgets,
        'final_regret': regrets[-1],
        'n_hf': len(hf_indices),
        'n_lf': len(lf_indices),
        'best_y': y_hf.min(),
        'fidelity_decisions': fidelity_decisions,
    }


# =============================================================================
# Parallel Worker Function
# =============================================================================

def run_combination(args):
    """Worker function for parallel execution"""
    bench_name, bench_config, model_name, model_class, budget, seeds, output_dir, worker_id = args

    if torch.cuda.is_available():
        torch.cuda.init()
        device = torch.device('cuda:0')
        _ = torch.zeros(1).to(device)
    else:
        device = torch.device('cpu')

    print(f"[Worker {worker_id}] {bench_name} + {model_name}: Using {device}", flush=True)

    # Create benchmark
    if bench_config['type'] == 'synthetic':
        benchmark = SyntheticBenchmark(
            bench_name,
            bench_config['hf_func'],
            bench_config['lf_func'],
            bench_config['dim'],
            bench_config['alpha'],
            bench_config['cost_ratio'],
            bench_config['f_star'],
            bench_config['grid_size']
        )
    else:
        benchmark = ChemistryBenchmark(
            bench_name,
            bench_config['csv_path'],
            bench_config['cost_ratio'],
            bench_config['use_smiles'],
            bench_config['minimize'],
            bench_config.get('negate', False)
        )

    sampling_method = 'lhs' if bench_config['type'] == 'synthetic' else 'fps'

    results_summary = []
    results_trajectory = []
    results_fidelity = []
    start_time = time.time()

    for i, seed in enumerate(seeds):
        seed_start = time.time()
        try:
            result = run_bo_adaptive(benchmark, model_class, budget, seed, device, sampling_method)
            seed_elapsed = time.time() - seed_start

            results_summary.append({
                'benchmark': bench_name,
                'model': model_name,
                'strategy': 'adaptive',
                'seed': seed,
                'final_regret': result['final_regret'],
                'n_hf': result['n_hf'],
                'n_lf': result['n_lf'],
                'best_y': result['best_y'],
                'elapsed_sec': round(seed_elapsed, 3),
            })

            for b, r in zip(result['budgets'], result['regrets']):
                results_trajectory.append({
                    'benchmark': bench_name,
                    'model': model_name,
                    'strategy': 'adaptive',
                    'seed': seed,
                    'budget': round(b, 2),
                    'regret': r,
                })

            for fd in result['fidelity_decisions']:
                fd_row = {
                    'benchmark': bench_name,
                    'model': model_name,
                    'seed': seed,
                }
                fd_row.update(fd)
                results_fidelity.append(fd_row)

        except Exception as e:
            seed_elapsed = time.time() - seed_start
            print(f"[Worker {worker_id}] {bench_name} + {model_name} seed={seed}: ERROR {e}", flush=True)
            results_summary.append({
                'benchmark': bench_name,
                'model': model_name,
                'strategy': 'adaptive',
                'seed': seed,
                'final_regret': np.nan,
                'n_hf': 0,
                'n_lf': 0,
                'best_y': np.nan,
                'elapsed_sec': round(seed_elapsed, 3),
            })

    elapsed = time.time() - start_time

    df_summary = pd.DataFrame(results_summary)
    summary_file = output_dir / f'summary_{bench_name}_{model_name.replace(" ", "_")}.csv'
    df_summary.to_csv(summary_file, index=False)

    df_trajectory = pd.DataFrame(results_trajectory)
    trajectory_file = output_dir / f'trajectory_{bench_name}_{model_name.replace(" ", "_")}.csv'
    df_trajectory.to_csv(trajectory_file, index=False)

    if results_fidelity:
        df_fidelity = pd.DataFrame(results_fidelity)
        fidelity_file = output_dir / f'fidelity_{bench_name}_{model_name.replace(" ", "_")}.csv'
        df_fidelity.to_csv(fidelity_file, index=False)

    return {
        'bench_name': bench_name,
        'model_name': model_name,
        'n_seeds': len(seeds),
        'elapsed': elapsed,
        'results_summary': results_summary,
        'results_trajectory': results_trajectory,
        'results_fidelity': results_fidelity,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Adaptive Fidelity Selection Benchmark')
    parser.add_argument('--n-seeds', type=int, default=20, help='Number of seeds')
    parser.add_argument('--base-seed', type=int, default=42, help='Base seed')
    parser.add_argument('--n-workers', type=int, default=48, help='Number of parallel workers')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                        help='Phase 1: 5 benchmarks x 5 models; Phase 2: full 7x15')
    parser.add_argument('--benchmarks', type=str, default=None,
                        help='Comma-separated benchmark names to run (overrides phase selection)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Shared output directory (for array jobs writing to same dir)')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model names to run (default: all for phase)')
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f'benchmark_adaptive_{timestamp}')
    output_dir.mkdir(exist_ok=True)

    data_dir = Path(__file__).parent / 'data'

    print("=" * 80)
    print("Adaptive Fidelity Selection Benchmark")
    print("=" * 80)
    print(f"Workers: {args.n_workers}")
    print(f"Seeds: {args.n_seeds}")
    print(f"Phase: {args.phase}")
    print(f"Output: {output_dir}")
    print()
    print("Adaptive Strategy:")
    print("  - Alpha-weighted cost-weighted EI for joint (candidate, fidelity) selection")
    print("  - alpha = |corr(y_lf, y_hf)| from paired data, discounts LF EI")
    print("  - HF EI uses y_best_hf; LF EI uses y_best_lf (separate incumbents)")
    print("  - Asymmetric masking: HF excludes sampled_hf; LF excludes sampled_lf|sampled_hf")

    print(f"\nCUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")

    bench_configs = {
        'Branin-Fav': {
            'type': 'synthetic', 'hf_func': branin_hf, 'lf_func': branin_lf,
            'dim': 2, 'alpha': 0.8, 'cost_ratio': 0.1, 'f_star': 0.397887, 'grid_size': 50
        },
        'Branin-Unfav': {
            'type': 'synthetic', 'hf_func': branin_hf, 'lf_func': branin_lf,
            'dim': 2, 'alpha': 0.1, 'cost_ratio': 0.5, 'f_star': 0.397887, 'grid_size': 50
        },
        'Park-Fav': {
            'type': 'synthetic', 'hf_func': park_hf, 'lf_func': park_lf,
            'dim': 4, 'alpha': 0.6, 'cost_ratio': 0.1, 'f_star': 0.0, 'grid_size': 10
        },
        'Park-Unfav': {
            'type': 'synthetic', 'hf_func': park_hf, 'lf_func': park_lf,
            'dim': 4, 'alpha': 0.0, 'cost_ratio': 0.5, 'f_star': 0.0, 'grid_size': 10
        },
        'COFs': {
            'type': 'chemistry', 'csv_path': data_dir / 'cofs.csv',
            'cost_ratio': 0.065, 'use_smiles': False, 'minimize': True, 'negate': True
        },
        'FreeSolv': {
            'type': 'chemistry', 'csv_path': data_dir / 'freesolv.csv',
            'cost_ratio': 0.1, 'use_smiles': True, 'minimize': True, 'negate': False
        },
        'Polarizability': {
            'type': 'chemistry', 'csv_path': data_dir / 'polarizability.csv',
            'cost_ratio': 0.167, 'use_smiles': True, 'minimize': True, 'negate': True
        },
    }

    budgets = {
        'Branin-Fav': 50, 'Branin-Unfav': 50,
        'Park-Fav': 50, 'Park-Unfav': 50,
        'COFs': 30, 'FreeSolv': 50, 'Polarizability': 30,
    }

    all_models = {
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

    if args.phase == 1:
        # Phase 1: 5 benchmarks x 5 models = 25 combinations
        selected_benchmarks = ['Branin-Fav', 'Park-Unfav', 'COFs', 'FreeSolv', 'Polarizability']
        selected_models = ['MFGP', 'Sequential', 'Curriculum', 'DNGO-Joint', 'NARGP']
        bench_configs = {k: v for k, v in bench_configs.items() if k in selected_benchmarks}

        # NARGP is from baselines, not local
        models = {}
        for name in selected_models:
            if name == 'NARGP':
                models[name] = NARGP
            else:
                models[name] = all_models[name]
    else:
        # Phase 2: full 7 benchmarks x all models (11 DNN + MFGP + GP baselines)
        models = dict(all_models)
        models['NARGP'] = NARGP
        models['Sparse MFGP'] = SparseMFGP
        models['DKL Multi-Fidelity'] = DKLMultiFidelity

    # Filter benchmarks if specified via --benchmarks
    if args.benchmarks:
        selected = [b.strip() for b in args.benchmarks.split(',')]
        bench_configs = {k: v for k, v in bench_configs.items() if k in selected}
        print(f"Filtered benchmarks: {list(bench_configs.keys())}")

    # Filter models if specified via --models
    if args.models:
        selected_m = [m.strip() for m in args.models.split(',')]
        models = {k: v for k, v in models.items() if k in selected_m}
        print(f"Filtered models: {list(models.keys())}")

    seeds = [args.base_seed + i for i in range(args.n_seeds)]

    tasks = []
    worker_id = 0
    for bench_name, bench_config in bench_configs.items():
        for model_name, model_class in models.items():
            tasks.append((
                bench_name, bench_config, model_name, model_class,
                budgets[bench_name], seeds, output_dir, worker_id
            ))
            worker_id += 1

    total_combinations = len(tasks)
    total_runs = total_combinations * args.n_seeds

    print(f"\nCombinations: {total_combinations}")
    print(f"Total runs: {total_runs}")
    print("=" * 80)

    start_time = time.time()

    with Pool(processes=args.n_workers) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(run_combination, tasks)):
            results.append(result)
            elapsed = time.time() - start_time
            completed = i + 1
            eta = (elapsed / completed) * (total_combinations - completed) if completed > 0 else 0
            print(f"[{completed}/{total_combinations}] {result['bench_name']} + {result['model_name']}: "
                  f"{result['elapsed']:.1f}s ({result['n_seeds']} seeds) | ETA: {eta/60:.1f}min")

    total_time = time.time() - start_time

    all_summary = []
    all_trajectory = []
    all_fidelity = []
    for r in results:
        all_summary.extend(r['results_summary'])
        all_trajectory.extend(r['results_trajectory'])
        all_fidelity.extend(r['results_fidelity'])

    df_summary = pd.DataFrame(all_summary)
    df_summary.to_csv(output_dir / 'results_summary.csv', index=False)

    df_trajectory = pd.DataFrame(all_trajectory)
    df_trajectory.to_csv(output_dir / 'results_trajectory.csv', index=False)

    df_fidelity = pd.DataFrame(all_fidelity)
    df_fidelity.to_csv(output_dir / 'fidelity_decisions.csv', index=False)

    print("\n" + "=" * 80)
    print(f"Completed in {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Results saved to: {output_dir}")
    print(f"  - results_summary.csv: {len(df_summary)} rows")
    print(f"  - results_trajectory.csv: {len(df_trajectory)} rows")
    print(f"  - fidelity_decisions.csv: {len(df_fidelity)} rows")
    print("=" * 80)

    print("\nSUMMARY (Mean Final Regret)")
    print("-" * 60)
    for bench in sorted(df_summary['benchmark'].unique()):
        print(f"\n{bench}:")
        summary = df_summary[df_summary['benchmark'] == bench].groupby('model')['final_regret'].agg(['mean', 'std'])
        summary = summary.sort_values('mean')
        for model_name, row in summary.head(5).iterrows():
            print(f"  {model_name:<30}: {row['mean']:.4f} +/- {row['std']:.4f}")

    # Fidelity allocation summary
    if len(df_fidelity) > 0:
        print("\nFIDELITY ALLOCATION (mean HF ratio per model)")
        print("-" * 60)
        for bench in sorted(df_fidelity['benchmark'].unique()):
            bench_data = df_fidelity[df_fidelity['benchmark'] == bench]
            print(f"\n{bench}:")
            for model_name in sorted(bench_data['model'].unique()):
                model_data = bench_data[bench_data['model'] == model_name]
                hf_ratio = model_data['fidelity_chosen'].mean()
                print(f"  {model_name:<30}: HF={hf_ratio:.1%}, LF={1-hf_ratio:.1%}")


if __name__ == '__main__':
    main()
