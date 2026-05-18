#!/usr/bin/env python
"""
LF-BLR 기반 Multi-Fidelity 모델 비교

핵심 아이디어:
- LF 평가 시: LF 모델에 BLR 붙여서 불확실성 계산 → EI로 다음 점 선택
- HF 평가 시: HF 모델은 BLR 없이 → argmin(mean)으로 다음 점 선택

모델:
1. MFGP - BoTorch SingleTaskMultiFidelityGP (기준)
2. Sequential-Vanilla - 모든 예측기에서 BLR 제거, argmin(mean) 사용
3. Sequential-LF-BLR - LF에만 BLR, HF에는 BLR 없음
   - LF 평가: EI (LF BLR 불확실성 기반)
   - HF 평가: argmin(mean)
4. Joint-Vanilla - Joint training, BLR 없음, argmin(mean) 사용
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import argparse
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from scipy.stats import norm
import pandas as pd

# BoTorch for MFGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

from synthetic_functions_mfbo import branin_hf, branin_lf, SCENARIOS, FUNCTIONS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# =============================================================================
# Acquisition Functions
# =============================================================================

def expected_improvement(mean: np.ndarray, std: np.ndarray,
                         y_best: float, xi: float = 0.01) -> np.ndarray:
    """Expected Improvement acquisition function"""
    with np.errstate(divide='ignore', invalid='ignore'):
        imp = y_best - mean - xi
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std < 1e-10] = 0.0
    return ei


def get_sampled_indices(X_grid: np.ndarray, X_sampled: np.ndarray) -> set:
    """
    Find grid indices that match sampled points exactly.

    Args:
        X_grid: Grid points (n_grid, dim)
        X_sampled: Already sampled points (n_sampled, dim)

    Returns:
        Set of grid indices that have been sampled
    """
    if len(X_sampled) == 0:
        return set()

    sampled_indices = set()
    for x in X_sampled:
        # Find exact match in grid (using small tolerance for float comparison)
        distances = np.sqrt(np.sum((X_grid - x)**2, axis=1))
        min_idx = np.argmin(distances)
        if distances[min_idx] < 1e-10:  # Exact match
            sampled_indices.add(min_idx)

    return sampled_indices


def select_exploitation(mean: np.ndarray, X_grid: np.ndarray,
                        sampled_indices: set = None) -> np.ndarray:
    """
    Pure exploitation: argmin(mean), excluding already sampled points

    Args:
        mean: Predicted mean values
        X_grid: Grid points
        sampled_indices: Set of grid indices to exclude
    """
    mean_masked = mean.copy()

    if sampled_indices is not None and len(sampled_indices) > 0:
        for idx in sampled_indices:
            mean_masked[idx] = np.inf

    best_idx = np.argmin(mean_masked)
    return X_grid[best_idx:best_idx+1], best_idx


def select_ei(mean: np.ndarray, std: np.ndarray, y_best: float,
              X_grid: np.ndarray, sampled_indices: set = None) -> np.ndarray:
    """
    EI-based selection: argmax(EI), excluding already sampled points

    Args:
        mean: Predicted mean values
        std: Predicted std values
        y_best: Best observed value so far
        X_grid: Grid points
        sampled_indices: Set of grid indices to exclude
    """
    ei = expected_improvement(mean, std, y_best)

    if sampled_indices is not None and len(sampled_indices) > 0:
        for idx in sampled_indices:
            ei[idx] = -np.inf

    best_idx = np.argmax(ei)
    return X_grid[best_idx:best_idx+1], best_idx


# =============================================================================
# MFGP (BoTorch)
# =============================================================================

class GP_MFGP:
    """Multi-Fidelity GP using BoTorch"""

    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = None
        self.is_fitted = False

    def fit(self, X_lf: np.ndarray, y_lf: np.ndarray,
            X_hf: np.ndarray, y_hf: np.ndarray):
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

    def predict(self, X: np.ndarray, fidelity: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Predict at given fidelity (0=LF, 1=HF)"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_fid = np.hstack([X, np.ones((len(X), 1)) * fidelity])
        X_tensor = torch.tensor(X_fid, dtype=torch.float64).to(device)

        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X_tensor)
            mean = posterior.mean.cpu().numpy().flatten()
            std = posterior.variance.sqrt().cpu().numpy().flatten()

        return mean, np.maximum(std, 1e-6)


# =============================================================================
# Sequential-Vanilla: No BLR anywhere, argmin(mean) only
# =============================================================================

class Sequential_Vanilla:
    """
    Sequential Training WITHOUT any BLR
    - LF network: x -> y_lf (direct output)
    - HF network: (x, y_lf) -> delta
    - Selection: always argmin(mean)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 lf_lr: float = 1e-3, hf_lr: float = 1e-3,
                 lf_epochs: int = 300, hf_epochs: int = 200,
                 num_layers: int = 2, l2_lambda: float = 1e-3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lf_lr = lf_lr
        self.hf_lr = hf_lr
        self.lf_epochs = lf_epochs
        self.hf_epochs = hf_epochs
        self.num_layers = num_layers
        self.l2_lambda = l2_lambda
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def _build_network(self, in_dim: int, out_dim: int):
        layers = []
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.Tanh())
            in_dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, out_dim))
        return nn.Sequential(*layers).to(device)

    def fit(self, X_lf: np.ndarray, y_lf: np.ndarray,
            X_hf: np.ndarray, y_hf: np.ndarray):
        # Scale
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])

        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

        X_lf_scaled = X_scaled[:len(X_lf)]
        X_hf_scaled = X_scaled[len(X_lf):]
        y_lf_scaled = y_scaled[:len(y_lf)]
        y_hf_scaled = y_scaled[len(y_lf):]

        X_lf_t = torch.FloatTensor(X_lf_scaled).to(device)
        y_lf_t = torch.FloatTensor(y_lf_scaled).view(-1, 1).to(device)
        X_hf_t = torch.FloatTensor(X_hf_scaled).to(device)
        y_hf_t = torch.FloatTensor(y_hf_scaled).view(-1, 1).to(device)

        # Build networks
        self.lf_network = self._build_network(self.input_dim, 1)
        self.hf_network = self._build_network(self.input_dim + 1, 1)

        loss_fn = nn.MSELoss()

        # Stage 1: Train LF
        lf_optimizer = torch.optim.Adam(
            self.lf_network.parameters(), lr=self.lf_lr, weight_decay=self.l2_lambda
        )

        self.lf_network.train()
        for _ in range(self.lf_epochs):
            lf_optimizer.zero_grad()
            y_lf_pred = self.lf_network(X_lf_t)
            loss = loss_fn(y_lf_pred, y_lf_t)
            loss.backward()
            lf_optimizer.step()

        # Stage 2: Freeze LF, Train HF
        for param in self.lf_network.parameters():
            param.requires_grad = False
        self.lf_network.eval()

        hf_optimizer = torch.optim.Adam(
            self.hf_network.parameters(), lr=self.hf_lr, weight_decay=self.l2_lambda
        )

        self.hf_network.train()
        for _ in range(self.hf_epochs):
            hf_optimizer.zero_grad()

            with torch.no_grad():
                y_lf_for_hf = self.lf_network(X_hf_t)

            hf_input = torch.cat([X_hf_t, y_lf_for_hf], dim=1)
            delta = self.hf_network(hf_input)
            y_hf_pred = y_lf_for_hf + delta

            loss = loss_fn(y_hf_pred, y_hf_t)
            loss.backward()
            hf_optimizer.step()

        self.is_fitted = True

    def predict_hf(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict HF (no uncertainty)"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_scaled = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(device)

        self.lf_network.eval()
        self.hf_network.eval()

        with torch.no_grad():
            y_lf_pred = self.lf_network(X_t)
            hf_input = torch.cat([X_t, y_lf_pred], dim=1)
            delta = self.hf_network(hf_input)
            y_hf_pred = y_lf_pred + delta
            mean_scaled = y_hf_pred.cpu().numpy().flatten()

        mean = self.scaler_y.inverse_transform(mean_scaled.reshape(-1, 1)).flatten()
        std = np.ones_like(mean) * 0.1  # Dummy

        return mean, std

    def predict_lf(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict LF (no uncertainty)"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_scaled = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(device)

        self.lf_network.eval()

        with torch.no_grad():
            y_lf_pred = self.lf_network(X_t)
            mean_scaled = y_lf_pred.cpu().numpy().flatten()

        mean = self.scaler_y.inverse_transform(mean_scaled.reshape(-1, 1)).flatten()
        std = np.ones_like(mean) * 0.1  # Dummy

        return mean, std


# =============================================================================
# Sequential-LF-BLR: BLR only on LF for exploration, HF uses argmin(mean)
# =============================================================================

class Sequential_LF_BLR:
    """
    Sequential Training with BLR ONLY on LF

    - LF network: x -> features -> BLR -> (mean, std) for LF
    - HF network: (x, y_lf) -> delta (no BLR)

    Selection Strategy:
    - LF evaluation: Use LF BLR uncertainty → EI → argmax(EI)
    - HF evaluation: Use HF mean → argmin(mean)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 feature_dim: int = 32,
                 lf_lr: float = 1e-3, hf_lr: float = 1e-3,
                 lf_epochs: int = 300, hf_epochs: int = 200,
                 num_layers: int = 2, l2_lambda: float = 1e-3,
                 alpha_blr: float = 1.0, beta_blr: float = 25.0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.lf_lr = lf_lr
        self.hf_lr = hf_lr
        self.lf_epochs = lf_epochs
        self.hf_epochs = hf_epochs
        self.num_layers = num_layers
        self.l2_lambda = l2_lambda
        self.alpha_blr = alpha_blr
        self.beta_blr = beta_blr
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def _build_lf_feature_network(self):
        """LF network outputs features for BLR"""
        layers = []
        in_dim = self.input_dim
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.Tanh())
            in_dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, self.feature_dim))
        return nn.Sequential(*layers).to(device)

    def _build_hf_network(self):
        """HF network: (x, y_lf) -> delta (direct output, no BLR)"""
        layers = []
        in_dim = self.input_dim + 1
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.Tanh())
            in_dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, 1))
        return nn.Sequential(*layers).to(device)

    def fit(self, X_lf: np.ndarray, y_lf: np.ndarray,
            X_hf: np.ndarray, y_hf: np.ndarray):
        # Scale
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])

        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

        X_lf_scaled = X_scaled[:len(X_lf)]
        X_hf_scaled = X_scaled[len(X_lf):]
        y_lf_scaled = y_scaled[:len(y_lf)]
        y_hf_scaled = y_scaled[len(y_lf):]

        X_lf_t = torch.FloatTensor(X_lf_scaled).to(device)
        y_lf_t = torch.FloatTensor(y_lf_scaled).view(-1, 1).to(device)
        X_hf_t = torch.FloatTensor(X_hf_scaled).to(device)
        y_hf_t = torch.FloatTensor(y_hf_scaled).view(-1, 1).to(device)

        # Build networks
        self.lf_feature_network = self._build_lf_feature_network()
        self.hf_network = self._build_hf_network()

        # ============ Stage 1: Train LF feature network ============
        # First, train feature network with a linear head
        lf_head = nn.Linear(self.feature_dim, 1).to(device)

        lf_optimizer = torch.optim.Adam(
            list(self.lf_feature_network.parameters()) + list(lf_head.parameters()),
            lr=self.lf_lr, weight_decay=self.l2_lambda
        )

        loss_fn = nn.MSELoss()
        self.lf_feature_network.train()
        lf_head.train()

        for _ in range(self.lf_epochs):
            lf_optimizer.zero_grad()
            Phi = self.lf_feature_network(X_lf_t)
            y_pred = lf_head(Phi)
            loss = loss_fn(y_pred, y_lf_t)
            loss.backward()
            lf_optimizer.step()

        # After training, fit BLR on the learned features
        self.lf_feature_network.eval()
        with torch.no_grad():
            Phi_lf = self.lf_feature_network(X_lf_t).cpu().numpy()

        # BLR closed form on learned features
        A = self.alpha_blr * np.eye(self.feature_dim) + self.beta_blr * Phi_lf.T @ Phi_lf
        self.lf_A_inv = np.linalg.inv(A)
        self.lf_m = self.beta_blr * self.lf_A_inv @ Phi_lf.T @ y_lf_scaled

        # ============ Stage 2: Train HF network (no BLR) ============
        hf_optimizer = torch.optim.Adam(
            self.hf_network.parameters(), lr=self.hf_lr, weight_decay=self.l2_lambda
        )

        loss_fn = nn.MSELoss()
        self.hf_network.train()

        for _ in range(self.hf_epochs):
            hf_optimizer.zero_grad()

            # Get LF predictions for HF points
            with torch.no_grad():
                Phi_hf = self.lf_feature_network(X_hf_t).cpu().numpy()
                y_lf_for_hf = Phi_hf @ self.lf_m
                y_lf_for_hf_t = torch.FloatTensor(y_lf_for_hf).view(-1, 1).to(device)

            hf_input = torch.cat([X_hf_t, y_lf_for_hf_t], dim=1)
            delta = self.hf_network(hf_input)
            y_hf_pred = y_lf_for_hf_t + delta

            loss = loss_fn(y_hf_pred, y_hf_t)
            loss.backward()
            hf_optimizer.step()

        self.is_fitted = True

    def predict_lf(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict LF with BLR uncertainty
        Returns: (mean, std) in original scale
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_scaled = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(device)

        self.lf_feature_network.eval()
        with torch.no_grad():
            Phi = self.lf_feature_network(X_t).cpu().numpy()

        # BLR prediction
        mean_scaled = Phi @ self.lf_m

        # BLR variance
        var_scaled = 1.0 / self.beta_blr + np.sum((Phi @ self.lf_A_inv) * Phi, axis=1)
        std_scaled = np.sqrt(np.maximum(var_scaled, 1e-10))

        # Inverse transform
        mean = self.scaler_y.inverse_transform(mean_scaled.reshape(-1, 1)).flatten()
        std = std_scaled * self.scaler_y.scale_[0]  # Scale std by y scaler

        return mean, np.maximum(std, 1e-6)

    def predict_hf(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict HF (no BLR, just mean)
        Returns: (mean, dummy_std)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_scaled = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(device)

        self.lf_feature_network.eval()
        self.hf_network.eval()

        with torch.no_grad():
            # LF prediction
            Phi = self.lf_feature_network(X_t).cpu().numpy()
            y_lf_pred_scaled = Phi @ self.lf_m
            y_lf_pred_t = torch.FloatTensor(y_lf_pred_scaled).view(-1, 1).to(device)

            # HF prediction (residual)
            hf_input = torch.cat([X_t, y_lf_pred_t], dim=1)
            delta = self.hf_network(hf_input)
            y_hf_pred_scaled = y_lf_pred_t + delta
            mean_scaled = y_hf_pred_scaled.cpu().numpy().flatten()

        mean = self.scaler_y.inverse_transform(mean_scaled.reshape(-1, 1)).flatten()
        std = np.ones_like(mean) * 0.1  # Dummy for HF (no BLR)

        return mean, std


# =============================================================================
# Joint-Vanilla: Joint training, no BLR
# =============================================================================

class Joint_Vanilla:
    """Joint Training WITHOUT BLR, argmin(mean) only"""

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 lr: float = 1e-3, epochs: int = 300,
                 alpha: float = 0.3, l2_lambda: float = 1e-3,
                 num_layers: int = 2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha
        self.l2_lambda = l2_lambda
        self.num_layers = num_layers
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def _build_network(self, in_dim: int, out_dim: int):
        layers = []
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.Tanh())
            in_dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, out_dim))
        return nn.Sequential(*layers).to(device)

    def fit(self, X_lf: np.ndarray, y_lf: np.ndarray,
            X_hf: np.ndarray, y_hf: np.ndarray):
        # Scale
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])

        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

        X_lf_scaled = X_scaled[:len(X_lf)]
        X_hf_scaled = X_scaled[len(X_lf):]
        y_lf_scaled = y_scaled[:len(y_lf)]
        y_hf_scaled = y_scaled[len(y_lf):]

        X_lf_t = torch.FloatTensor(X_lf_scaled).to(device)
        y_lf_t = torch.FloatTensor(y_lf_scaled).view(-1, 1).to(device)
        X_hf_t = torch.FloatTensor(X_hf_scaled).to(device)
        y_hf_t = torch.FloatTensor(y_hf_scaled).view(-1, 1).to(device)

        self.lf_network = self._build_network(self.input_dim, 1)
        self.hf_network = self._build_network(self.input_dim + 1, 1)

        optimizer = torch.optim.Adam(
            list(self.lf_network.parameters()) + list(self.hf_network.parameters()),
            lr=self.lr, weight_decay=self.l2_lambda
        )
        loss_fn = nn.MSELoss()

        for _ in range(self.epochs):
            optimizer.zero_grad()

            # LF
            y_lf_pred = self.lf_network(X_lf_t)
            lf_loss = loss_fn(y_lf_pred, y_lf_t)

            # HF (residual)
            with torch.no_grad():
                y_lf_for_hf = self.lf_network(X_hf_t)

            hf_input = torch.cat([X_hf_t, y_lf_for_hf], dim=1)
            delta = self.hf_network(hf_input)
            y_hf_pred = y_lf_for_hf + delta
            hf_loss = loss_fn(y_hf_pred, y_hf_t)

            total_loss = (1 - self.alpha) * lf_loss + self.alpha * hf_loss
            total_loss.backward()
            optimizer.step()

        self.is_fitted = True

    def predict_hf(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_scaled = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(device)

        self.lf_network.eval()
        self.hf_network.eval()

        with torch.no_grad():
            y_lf_pred = self.lf_network(X_t)
            hf_input = torch.cat([X_t, y_lf_pred], dim=1)
            delta = self.hf_network(hf_input)
            y_hf_pred = y_lf_pred + delta
            mean_scaled = y_hf_pred.cpu().numpy().flatten()

        mean = self.scaler_y.inverse_transform(mean_scaled.reshape(-1, 1)).flatten()
        std = np.ones_like(mean) * 0.1

        return mean, std


# =============================================================================
# Main Comparison
# =============================================================================

def run_comparison(seed: int, total_budget: float = 50.0, save_interval: int = 5,
                   scenario_name: str = 'favorable'):
    """
    Compare 4 models:
    1. MFGP (baseline)
    2. Sequential-Vanilla (no BLR, argmin)
    3. Sequential-LF-BLR (LF: EI, HF: argmin)
    4. Joint-Vanilla (no BLR, argmin)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    scenario = SCENARIOS[scenario_name]
    alpha = scenario['alpha_branin']
    rho = scenario['rho']
    f_star = FUNCTIONS['Branin-2D']['f_star']

    dim = 2
    cost_hf = 1.0
    cost_lf = rho

    lf_hf_ratio = int(1.0 / rho)
    n_init_hf = 3
    n_init_lf = n_init_hf * lf_hf_ratio

    # Grid
    n_grid = 100
    x1_grid = np.linspace(0, 1, n_grid)
    x2_grid = np.linspace(0, 1, n_grid)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])

    y_hf_true = branin_hf(X_grid).reshape(n_grid, n_grid)
    y_lf_true = branin_lf(X_grid, alpha).reshape(n_grid, n_grid)

    # Initial data (shared) - select from grid to ensure proper exclusion
    init_lf_indices = np.random.choice(len(X_grid), n_init_lf, replace=False)
    X_lf_init = X_grid[init_lf_indices]
    y_lf_init = branin_lf(X_lf_init, alpha).flatten()

    # Select HF initial points from remaining grid points
    remaining_indices = np.setdiff1d(np.arange(len(X_grid)), init_lf_indices)
    init_hf_indices = np.random.choice(remaining_indices, n_init_hf, replace=False)
    X_hf_init = X_grid[init_hf_indices]
    y_hf_init = branin_hf(X_hf_init).flatten()

    init_budget = n_init_lf * cost_lf + n_init_hf * cost_hf

    model_keys = ['mfgp', 'seq_vanilla', 'seq_lf_blr', 'joint_vanilla']
    model_names = ['MFGP', 'Seq-Vanilla', 'Seq-LF-BLR', 'Joint-Vanilla']

    data = {k: {
        'X_lf': X_lf_init.copy(), 'y_lf': y_lf_init.copy(),
        'X_hf': X_hf_init.copy(), 'y_hf': y_hf_init.copy(),
        'budget': init_budget
    } for k in model_keys}

    lf_counters = {k: 0 for k in model_keys}
    comparison_data = []

    # Track HF selections for each model
    hf_selection_records = {k: [] for k in model_keys}

    # Record initial HF points
    for k in model_keys:
        for i in range(len(X_hf_init)):
            hf_selection_records[k].append({
                'step': 0,
                'x1': X_hf_init[i, 0],
                'x2': X_hf_init[i, 1],
                'y_hf': y_hf_init[i],
                'is_initial': True
            })

    max_iterations = 300
    iteration = 0
    lf_per_hf = max(1, int(1.0 / rho))

    def get_eval_decision(remaining, lf_counter):
        if remaining >= cost_hf:
            if remaining >= cost_lf and lf_counter < lf_per_hf:
                return False, cost_lf, lf_counter + 1
            else:
                return True, cost_hf, 0
        elif remaining >= cost_lf:
            return False, cost_lf, lf_counter
        else:
            return None, 0, lf_counter

    print(f"\nScenario: {scenario_name} (alpha={alpha}, rho={rho})")
    print(f"Initial: LF={n_init_lf}, HF={n_init_hf}, Budget={init_budget:.1f}")
    print(f"Models: {model_names}")
    print("-" * 80)

    while iteration < max_iterations:
        iteration += 1

        all_done = all(data[m]['budget'] >= total_budget for m in model_keys)
        if all_done:
            break

        y_hf_true_flat = y_hf_true.flatten()
        step_info = {'step': iteration}

        for model_key in model_keys:
            d = data[model_key]
            remaining = total_budget - d['budget']

            eval_hf, cost, lf_counters[model_key] = get_eval_decision(
                remaining, lf_counters[model_key]
            )

            if eval_hf is None:
                continue

            try:
                # Get indices of already sampled points (exact grid match only)
                X_sampled = np.vstack([d['X_lf'], d['X_hf']])
                sampled_indices = get_sampled_indices(X_grid, X_sampled)

                # Create and fit model
                if model_key == 'mfgp':
                    model = GP_MFGP(dim)
                    model.fit(d['X_lf'], d['y_lf'], d['X_hf'], d['y_hf'])

                    # MFGP uses EI at HF fidelity
                    mean, std = model.predict(X_grid, fidelity=1.0)
                    y_best = d['y_hf'].min()
                    x_next, _ = select_ei(mean, std, y_best, X_grid, sampled_indices)

                elif model_key == 'seq_vanilla':
                    model = Sequential_Vanilla(dim)
                    model.fit(d['X_lf'], d['y_lf'], d['X_hf'], d['y_hf'])

                    # Always argmin(mean) for both LF and HF
                    mean, std = model.predict_hf(X_grid)
                    x_next, _ = select_exploitation(mean, X_grid, sampled_indices)

                elif model_key == 'seq_lf_blr':
                    model = Sequential_LF_BLR(dim)
                    model.fit(d['X_lf'], d['y_lf'], d['X_hf'], d['y_hf'])

                    if eval_hf:
                        # HF evaluation: argmin(mean)
                        mean, std = model.predict_hf(X_grid)
                        x_next, _ = select_exploitation(mean, X_grid, sampled_indices)
                    else:
                        # LF evaluation: EI based on LF uncertainty
                        mean_lf, std_lf = model.predict_lf(X_grid)
                        y_best_lf = d['y_lf'].min()
                        x_next, _ = select_ei(mean_lf, std_lf, y_best_lf, X_grid, sampled_indices)
                        # Use HF prediction for metrics
                        mean, std = model.predict_hf(X_grid)

                elif model_key == 'joint_vanilla':
                    model = Joint_Vanilla(dim)
                    model.fit(d['X_lf'], d['y_lf'], d['X_hf'], d['y_hf'])

                    mean, std = model.predict_hf(X_grid)
                    x_next, _ = select_exploitation(mean, X_grid, sampled_indices)

                # Evaluate
                x_new = x_next.reshape(1, -1)
                if eval_hf:
                    y_new = branin_hf(x_new).flatten()
                    d['X_hf'] = np.vstack([d['X_hf'], x_new])
                    d['y_hf'] = np.append(d['y_hf'], y_new)

                    # Record HF selection
                    hf_selection_records[model_key].append({
                        'step': iteration,
                        'x1': x_new[0, 0],
                        'x2': x_new[0, 1],
                        'y_hf': y_new[0],
                        'is_initial': False
                    })
                else:
                    y_new = branin_lf(x_new, alpha).flatten()
                    d['X_lf'] = np.vstack([d['X_lf'], x_new])
                    d['y_lf'] = np.append(d['y_lf'], y_new)

                d['budget'] += cost

                # Metrics
                r2 = 1 - np.sum((y_hf_true_flat - mean)**2) / np.sum((y_hf_true_flat - np.mean(y_hf_true_flat))**2)
                regret = max(0, d['y_hf'].min() - f_star)

                step_info[f'{model_key}_mean'] = mean.reshape(n_grid, n_grid)
                step_info[f'{model_key}_std'] = std.reshape(n_grid, n_grid)
                step_info[f'{model_key}_x_next'] = x_next
                step_info[f'{model_key}_r2'] = r2
                step_info[f'{model_key}_regret'] = regret
                step_info[f'{model_key}_best'] = d['y_hf'].min()
                step_info[f'{model_key}_X_hf'] = d['X_hf'].copy()
                step_info[f'{model_key}_X_lf'] = d['X_lf'].copy()
                step_info[f'{model_key}_eval_type'] = 'HF' if eval_hf else 'LF'

            except Exception as e:
                print(f"  {model_key} error at step {iteration}: {e}")
                import traceback
                traceback.print_exc()

        # Save at intervals
        if iteration % save_interval == 0:
            comparison_data.append(step_info)
            regrets = {m: max(0, data[m]['y_hf'].min() - f_star) for m in model_keys}
            print(f"  Step {iteration:3d}: " + " | ".join(
                f"{model_names[i]}={regrets[k]:.4f}" for i, k in enumerate(model_keys)
            ))

    return {
        'comparison_data': comparison_data,
        'y_hf_true': y_hf_true,
        'y_lf_true': y_lf_true,
        'X1': X1, 'X2': X2,
        'f_star': f_star,
        'alpha': alpha,
        'rho': rho,
        'scenario': scenario_name,
        'model_keys': model_keys,
        'model_names': model_names,
        'hf_selection_records': hf_selection_records,
    }


def visualize_results(results: dict, output_dir: Path):
    """Visualize comparison results"""
    comparison_data = results['comparison_data']
    y_hf_true = results['y_hf_true']
    X1, X2 = results['X1'], results['X2']
    f_star = results['f_star']
    model_keys = results['model_keys']
    model_names = results['model_names']

    colors = {'mfgp': 'green', 'seq_vanilla': 'blue',
              'seq_lf_blr': 'red', 'joint_vanilla': 'purple'}

    # Global optima
    global_optima = [
        ((-np.pi + 5) / 15, 12.275 / 15),
        ((np.pi + 5) / 15, 2.275 / 15),
        ((9.42478 + 5) / 15, 2.475 / 15),
    ]
    opt_x1 = [o[0] for o in global_optima]
    opt_x2 = [o[1] for o in global_optima]

    if len(comparison_data) == 0:
        print("No data to visualize!")
        return

    # Select steps
    n_show = min(4, len(comparison_data))
    indices = np.linspace(0, len(comparison_data) - 1, n_show, dtype=int)
    selected = [comparison_data[i] for i in indices]

    for data in selected:
        step = data['step']

        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 5, figure=fig, hspace=0.3, wspace=0.3)

        # Row 1: Mean predictions (4 models + True HF)
        for col, (key, name) in enumerate(zip(model_keys, model_names)):
            if f'{key}_mean' not in data:
                continue

            ax = fig.add_subplot(gs[0, col])
            mean = data[f'{key}_mean']
            r2 = data.get(f'{key}_r2', 0)
            best = data.get(f'{key}_best', 0)
            eval_type = data.get(f'{key}_eval_type', '?')

            im = ax.contourf(X1, X2, mean, levels=30, cmap='viridis')
            ax.scatter(data[f'{key}_X_hf'][:, 0], data[f'{key}_X_hf'][:, 1],
                       c='red', s=50, marker='o', edgecolors='white', label='HF')
            ax.scatter(data[f'{key}_X_lf'][:, 0], data[f'{key}_X_lf'][:, 1],
                       c='blue', s=30, marker='^', edgecolors='white', alpha=0.5, label='LF')
            ax.scatter(data[f'{key}_x_next'][0, 0], data[f'{key}_x_next'][0, 1],
                       c='lime', s=200, marker='*', edgecolors='black', linewidths=2)
            ax.scatter(opt_x1, opt_x2, c='yellow', s=100, marker='*', edgecolors='black')
            ax.set_title(f'{name} [{eval_type}]\nR²={r2:.3f}, best={best:.4f}', fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)
            if col == 0:
                ax.legend(loc='lower left', fontsize=6)

        # True HF
        ax = fig.add_subplot(gs[0, 4])
        im = ax.contourf(X1, X2, y_hf_true, levels=30, cmap='viridis')
        ax.scatter(opt_x1, opt_x2, c='yellow', s=100, marker='*', edgecolors='black')
        ax.set_title(f'True HF (f*={f_star:.4f})', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Row 2: Error maps
        for col, (key, name) in enumerate(zip(model_keys, model_names)):
            if f'{key}_mean' not in data:
                continue

            ax = fig.add_subplot(gs[1, col])
            mean = data[f'{key}_mean']
            error = np.abs(mean - y_hf_true)
            mae = error.mean()

            im = ax.contourf(X1, X2, error, levels=30, cmap='Reds')
            ax.scatter(data[f'{key}_X_hf'][:, 0], data[f'{key}_X_hf'][:, 1],
                       c='blue', s=40, marker='o', edgecolors='white')
            ax.set_title(f'{name} Error\nMAE={mae:.3f}', fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Row 2, col 4: Sampled points comparison
        ax = fig.add_subplot(gs[1, 4])
        ax.contourf(X1, X2, y_hf_true, levels=30, cmap='viridis', alpha=0.5)
        for key, name in zip(model_keys, model_names):
            if f'{key}_X_hf' in data:
                ax.scatter(data[f'{key}_X_hf'][:, 0], data[f'{key}_X_hf'][:, 1],
                           c=colors[key], s=40, marker='o', edgecolors='white', alpha=0.7,
                           label=f'{name} ({len(data[f"{key}_X_hf"])})')
        ax.scatter(opt_x1, opt_x2, c='yellow', s=150, marker='*', edgecolors='black')
        ax.set_title('HF Sample Points', fontsize=9)
        ax.legend(loc='lower left', fontsize=6)

        # Row 3: Regret bar + R² bar
        ax = fig.add_subplot(gs[2, :2])
        regrets = [data.get(f'{k}_regret', 0) for k in model_keys]
        bars = ax.bar(model_names, regrets, color=[colors[k] for k in model_keys], alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_ylabel('Regret')
        ax.set_title('Simple Regret', fontsize=10)
        for bar, reg in zip(bars, regrets):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{reg:.4f}', ha='center', va='bottom', fontsize=8)

        ax = fig.add_subplot(gs[2, 2:4])
        r2s = [data.get(f'{k}_r2', 0) for k in model_keys]
        bars = ax.bar(model_names, r2s, color=[colors[k] for k in model_keys], alpha=0.7)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax.set_ylabel('R²')
        ax.set_ylim(0, 1.1)
        ax.set_title('Prediction Accuracy', fontsize=10)
        for bar, r2 in zip(bars, r2s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{r2:.3f}', ha='center', va='bottom', fontsize=8)

        # Legend for strategy
        ax = fig.add_subplot(gs[2, 4])
        ax.axis('off')
        strategy_text = """Selection Strategy:

MFGP: EI (GP uncertainty)
Seq-Vanilla: argmin(mean) always
Seq-LF-BLR:
  - LF: EI (BLR uncertainty)
  - HF: argmin(mean)
Joint-Vanilla: argmin(mean) always
"""
        ax.text(0.1, 0.5, strategy_text, fontsize=9, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle(f'Step {step} | LF-BLR Comparison | {results["scenario"]}',
                     fontsize=12, fontweight='bold', y=0.995)

        plt.savefig(output_dir / f'lf_blr_step_{step:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: lf_blr_step_{step:03d}.png")


def visualize_convergence(results: dict, output_dir: Path):
    """Plot convergence curves"""
    comparison_data = results['comparison_data']
    model_keys = results['model_keys']
    model_names = results['model_names']

    colors = {'mfgp': 'green', 'seq_vanilla': 'blue',
              'seq_lf_blr': 'red', 'joint_vanilla': 'purple'}

    if len(comparison_data) == 0:
        return

    steps = [d['step'] for d in comparison_data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Regret
    ax = axes[0]
    for key, name in zip(model_keys, model_names):
        regrets = [d.get(f'{key}_regret', np.nan) for d in comparison_data]
        ax.semilogy(steps, regrets, '-o', color=colors[key], label=name, markersize=4)
    ax.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='Target 0.01')
    ax.set_xlabel('Step')
    ax.set_ylabel('Simple Regret (log)')
    ax.set_title('Convergence (Regret)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # R²
    ax = axes[1]
    for key, name in zip(model_keys, model_names):
        r2s = [d.get(f'{key}_r2', np.nan) for d in comparison_data]
        ax.plot(steps, r2s, '-o', color=colors[key], label=name, markersize=4)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('R²')
    ax.set_title('Prediction Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Final results
    final = comparison_data[-1]
    final_regrets = {k: final.get(f'{k}_regret', np.nan) for k in model_keys}
    winner = min(final_regrets, key=final_regrets.get)
    winner_name = model_names[model_keys.index(winner)]

    fig.suptitle(f'LF-BLR Comparison | Winner: {winner_name} ({final_regrets[winner]:.4f})',
                 fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / 'convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: convergence.png")


def main():
    parser = argparse.ArgumentParser(description='LF-BLR MF Model Comparison')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--budget', type=float, default=50, help='Total budget')
    parser.add_argument('--save-interval', type=int, default=5, help='Save interval')
    parser.add_argument('--scenario', type=str, default='favorable',
                        choices=['favorable', 'unfavorable'], help='Scenario')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'lf_blr_comparison_{timestamp}')
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("LF-BLR Multi-Fidelity Model Comparison")
    print("=" * 80)
    print("Models:")
    print("  1. MFGP         - GP with full uncertainty, EI selection")
    print("  2. Seq-Vanilla  - No BLR, argmin(mean) always")
    print("  3. Seq-LF-BLR   - LF: BLR + EI, HF: argmin(mean)")
    print("  4. Joint-Vanilla- No BLR, argmin(mean) always")
    print("-" * 80)
    print(f"Seed: {args.seed}")
    print(f"Budget: {args.budget}")
    print(f"Scenario: {args.scenario}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    results = run_comparison(
        args.seed, args.budget, args.save_interval, args.scenario
    )

    print("\nGenerating visualizations...")
    visualize_results(results, output_dir)
    visualize_convergence(results, output_dir)

    # Save HF selection records to CSV
    print("\nSaving HF selection records to CSV...")
    hf_records = results['hf_selection_records']
    model_keys = results['model_keys']
    model_names = results['model_names']

    for key, name in zip(model_keys, model_names):
        df = pd.DataFrame(hf_records[key])
        csv_path = output_dir / f'hf_selections_{key}.csv'
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path.name} ({len(df)} HF points)")

    # Also save combined CSV with model column
    all_records = []
    for key, name in zip(model_keys, model_names):
        for record in hf_records[key]:
            record_copy = record.copy()
            record_copy['model'] = name
            all_records.append(record_copy)

    df_all = pd.DataFrame(all_records)
    df_all.to_csv(output_dir / 'hf_selections_all.csv', index=False)
    print(f"  Saved: hf_selections_all.csv ({len(df_all)} total records)")

    print("\n" + "=" * 80)
    print("Done!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
