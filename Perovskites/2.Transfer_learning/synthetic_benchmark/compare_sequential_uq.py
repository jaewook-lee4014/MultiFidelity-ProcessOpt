#!/usr/bin/env python
"""
Multi-Fidelity 모델 비교 (독립 BO 루프):
1. MFGP (BoTorch SingleTaskMultiFidelityGP)
2. DNGO_Sequential_BLR
3. DNGO_Sequential_MCDropout

각 모델이 초기 데이터만 공유하고 이후 독립적으로 BO 진행
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import argparse
from scipy.stats import norm, spearmanr
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from typing import Tuple

# BoTorch for MFGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

# Plotly for 3D visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from synthetic_functions_mfbo import branin_hf, branin_lf, SCENARIOS, FUNCTIONS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# =============================================================================
# MFGP (BoTorch)
# =============================================================================

class GP_MFGP:
    """
    Multi-Fidelity GP using BoTorch's SingleTaskMultiFidelityGP
    - Fidelity kernel learns LF-HF correlation
    - Native GP uncertainty quantification
    """

    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = None
        self.is_fitted = False

    def fit(self, X_lf: np.ndarray, y_lf: np.ndarray,
            X_hf: np.ndarray, y_hf: np.ndarray):
        # Add fidelity column: LF=0, HF=1
        n_lf, n_hf = len(X_lf), len(X_hf)

        X_lf_fid = np.hstack([X_lf, np.zeros((n_lf, 1))])
        X_hf_fid = np.hstack([X_hf, np.ones((n_hf, 1))])

        X_all = np.vstack([X_lf_fid, X_hf_fid])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])

        train_X = torch.tensor(X_all, dtype=torch.float64).to(device)
        train_Y = torch.tensor(y_all, dtype=torch.float64).unsqueeze(-1).to(device)

        self.model = SingleTaskMultiFidelityGP(
            train_X, train_Y,
            data_fidelities=[self.input_dim],  # Last column is fidelity
            outcome_transform=Standardize(m=1)
        ).to(device)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        # Predict at HF (fidelity=1)
        X_fid = np.hstack([X, np.ones((len(X), 1))])
        X_tensor = torch.tensor(X_fid, dtype=torch.float64).to(device)

        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X_tensor)
            mean = posterior.mean.cpu().numpy().flatten()
            std = posterior.variance.sqrt().cpu().numpy().flatten()

        return mean, np.maximum(std, 1e-6)


# =============================================================================
# DNGO_Sequential with MC Dropout
# =============================================================================

class DNGO_Sequential_MCDropout:
    """
    DNGO Sequential with MC Dropout for uncertainty estimation

    - Stage 1: Train LF network on LF data (with dropout)
    - Stage 2: Freeze LF, train HF network on HF data (with dropout)
    - Prediction: MC sampling with dropout enabled
    - L2 regularization applied via weight_decay in optimizer
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 lf_lr: float = 1e-3, hf_lr: float = 1e-3,
                 lf_epochs: int = 300, hf_epochs: int = 200,
                 num_layers: int = 2, l2_lambda: float = 1e-3,
                 dropout: float = 0.1, n_mc_samples: int = 50):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lf_lr = lf_lr
        self.hf_lr = hf_lr
        self.lf_epochs = lf_epochs
        self.hf_epochs = hf_epochs
        self.num_layers = num_layers
        self.l2_lambda = l2_lambda  # L2 regularization strength
        self.dropout = dropout
        self.n_mc_samples = n_mc_samples
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def _build_lf_network(self):
        """LF network with dropout"""
        layers = []
        in_dim = self.input_dim
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(self.dropout))
            in_dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, 1))
        return nn.Sequential(*layers).to(device)

    def _build_hf_network(self):
        """HF network with dropout - outputs prediction directly"""
        layers = []
        in_dim = self.input_dim + 1  # +1 for LF prediction
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(self.dropout))
            in_dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, 1))  # Direct output, not features
        return nn.Sequential(*layers).to(device)

    def fit(self, X_lf: np.ndarray, y_lf: np.ndarray,
            X_hf: np.ndarray, y_hf: np.ndarray):
        # Scale data
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

        self.lf_network = self._build_lf_network()
        self.hf_network = self._build_hf_network()

        loss_fn = nn.MSELoss()

        # Stage 1: Train LF network with L2 regularization
        lf_optimizer = torch.optim.Adam(
            self.lf_network.parameters(),
            lr=self.lf_lr,
            weight_decay=self.l2_lambda  # L2 regularization
        )

        self.lf_network.train()
        for epoch in range(self.lf_epochs):
            lf_optimizer.zero_grad()
            y_lf_pred = self.lf_network(X_lf_t)
            lf_loss = loss_fn(y_lf_pred, y_lf_t)
            lf_loss.backward()
            lf_optimizer.step()

        # Stage 2: Freeze LF, train HF
        for param in self.lf_network.parameters():
            param.requires_grad = False
        self.lf_network.eval()

        hf_optimizer = torch.optim.Adam(
            self.hf_network.parameters(),
            lr=self.hf_lr,
            weight_decay=self.l2_lambda  # L2 regularization
        )

        self.hf_network.train()
        for epoch in range(self.hf_epochs):
            hf_optimizer.zero_grad()

            with torch.no_grad():
                y_lf_for_hf = self.lf_network(X_hf_t)

            hf_input = torch.cat([X_hf_t, y_lf_for_hf], dim=1)
            delta = self.hf_network(hf_input)
            y_hf_pred = y_lf_for_hf + delta  # Residual learning

            hf_loss = loss_fn(y_hf_pred, y_hf_t)
            hf_loss.backward()
            hf_optimizer.step()

        for param in self.lf_network.parameters():
            param.requires_grad = True

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """MC Dropout prediction"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_scaled = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(device)

        # Keep dropout active for MC sampling
        self.lf_network.train()  # Enable dropout
        self.hf_network.train()  # Enable dropout

        predictions = []
        for _ in range(self.n_mc_samples):
            with torch.no_grad():
                y_lf_pred = self.lf_network(X_t)
                hf_input = torch.cat([X_t, y_lf_pred], dim=1)
                delta = self.hf_network(hf_input)
                y_hf_pred = y_lf_pred + delta
                predictions.append(y_hf_pred.cpu().numpy().flatten())

        predictions = np.array(predictions)  # [n_samples, n_points]
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)

        # Inverse transform
        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
        std = std * self.scaler_y.scale_[0]

        return mean, np.maximum(std, 1e-6)


class DNGO_Sequential_BLR:
    """
    DNGO Sequential with BLR for uncertainty estimation (original)
    - L2 regularization applied via weight_decay in optimizer
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 lf_lr: float = 1e-3, hf_lr: float = 1e-3,
                 lf_epochs: int = 300, hf_epochs: int = 200,
                 num_layers: int = 2, l2_lambda: float = 1e-3,
                 feature_dim: int = 50, blr_alpha: float = 1.0, blr_beta: float = 25.0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lf_lr = lf_lr
        self.hf_lr = hf_lr
        self.lf_epochs = lf_epochs
        self.hf_epochs = hf_epochs
        self.num_layers = num_layers
        self.l2_lambda = l2_lambda  # L2 regularization strength
        self.feature_dim = feature_dim
        self.blr_alpha = blr_alpha
        self.blr_beta = blr_beta
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def _build_lf_network(self):
        layers = []
        in_dim = self.input_dim
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.Tanh())
            in_dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, 1))
        return nn.Sequential(*layers).to(device)

    def _build_hf_network(self):
        layers = []
        in_dim = self.input_dim + 1
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.Tanh())
            in_dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, self.feature_dim))
        return nn.Sequential(*layers).to(device)

    def fit(self, X_lf: np.ndarray, y_lf: np.ndarray,
            X_hf: np.ndarray, y_hf: np.ndarray):
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

        self.lf_network = self._build_lf_network()
        self.hf_network = self._build_hf_network()

        loss_fn = nn.MSELoss()

        # Stage 1: Train LF with L2 regularization
        lf_optimizer = torch.optim.Adam(
            self.lf_network.parameters(), lr=self.lf_lr, weight_decay=self.l2_lambda
        )

        self.lf_network.train()
        for epoch in range(self.lf_epochs):
            lf_optimizer.zero_grad()
            y_lf_pred = self.lf_network(X_lf_t)
            lf_loss = loss_fn(y_lf_pred, y_lf_t)
            lf_loss.backward()
            lf_optimizer.step()

        # Stage 2: Freeze LF, train HF with L2 regularization
        for param in self.lf_network.parameters():
            param.requires_grad = False
        self.lf_network.eval()

        hf_optimizer = torch.optim.Adam(
            self.hf_network.parameters(), lr=self.hf_lr, weight_decay=self.l2_lambda
        )

        self.hf_network.train()
        for epoch in range(self.hf_epochs):
            hf_optimizer.zero_grad()

            with torch.no_grad():
                y_lf_for_hf = self.lf_network(X_hf_t)

            hf_input = torch.cat([X_hf_t, y_lf_for_hf], dim=1)
            hf_features = self.hf_network(hf_input)
            delta = hf_features.mean(dim=1, keepdim=True)
            y_hf_pred = y_lf_for_hf + delta

            hf_loss = loss_fn(y_hf_pred, y_hf_t)
            hf_loss.backward()
            hf_optimizer.step()

        for param in self.lf_network.parameters():
            param.requires_grad = True

        # BLR on HF features
        self.lf_network.eval()
        self.hf_network.eval()
        with torch.no_grad():
            y_lf_for_hf = self.lf_network(X_hf_t)
            hf_input = torch.cat([X_hf_t, y_lf_for_hf], dim=1)
            Phi = self.hf_network(hf_input).cpu().numpy()

        alpha_blr, beta = self.blr_alpha, self.blr_beta
        A = alpha_blr * np.eye(Phi.shape[1]) + beta * Phi.T @ Phi
        self.A_inv = np.linalg.inv(A + 1e-6 * np.eye(A.shape[0]))

        y_lf_for_hf_np = y_lf_for_hf.cpu().numpy().flatten()
        residual = y_hf_scaled - y_lf_for_hf_np
        self.m = beta * self.A_inv @ Phi.T @ residual
        self.beta = beta
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_scaled = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(device)

        self.lf_network.eval()
        self.hf_network.eval()
        with torch.no_grad():
            y_lf_pred = self.lf_network(X_t)
            hf_input = torch.cat([X_t, y_lf_pred], dim=1)
            Phi = self.hf_network(hf_input).cpu().numpy()
            y_lf_np = y_lf_pred.cpu().numpy().flatten()

        residual_mean = Phi @ self.m
        mean = y_lf_np + residual_mean

        var = 1/self.beta + np.sum(Phi @ self.A_inv * Phi, axis=1)
        std = np.sqrt(np.maximum(var, 1e-6))

        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
        std = std * self.scaler_y.scale_[0]

        return mean, std


# =============================================================================
# Comparison Functions
# =============================================================================

def expected_improvement(mean: np.ndarray, std: np.ndarray,
                         y_best: float, xi: float = 0.01) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        imp = y_best - mean - xi
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std < 1e-10] = 0.0
    return ei


def run_comparison(seed: int, total_budget: float = 50.0, save_interval: int = 5,
                   dropout: float = 0.1, n_mc_samples: int = 50, l2_lambda: float = 1e-3):
    """
    Compare MFGP vs BLR vs MC Dropout
    Each model runs its own independent BO loop (only initial data is shared)

    Args:
        l2_lambda: L2 regularization strength for DNGO models (default: 1e-3)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Setup - favorable scenario
    scenario = SCENARIOS['favorable']
    alpha = scenario['alpha_branin']
    rho = scenario['rho']
    f_star = FUNCTIONS['Branin-2D']['f_star']

    dim = 2
    cost_hf = 1.0
    cost_lf = rho

    lf_hf_ratio = int(1.0 / rho)
    n_init_hf = 2
    n_init_lf = n_init_hf * lf_hf_ratio

    # Grid
    n_grid = 100
    x1_grid = np.linspace(0, 1, n_grid)
    x2_grid = np.linspace(0, 1, n_grid)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])

    y_hf_true = branin_hf(X_grid).reshape(n_grid, n_grid)
    y_lf_true = branin_lf(X_grid, alpha).reshape(n_grid, n_grid)

    opt_idx = np.argmin(y_hf_true)
    opt_x1, opt_x2 = X1.ravel()[opt_idx], X2.ravel()[opt_idx]
    print(f"True optimum at: ({opt_x1:.3f}, {opt_x2:.3f}), f*={f_star:.4f}")

    # Initial samples (SHARED between all models)
    X_lf_init = np.random.uniform(0, 1, (n_init_lf, dim))
    y_lf_init = branin_lf(X_lf_init, alpha).flatten()
    X_hf_init = np.random.uniform(0, 1, (n_init_hf, dim))
    y_hf_init = branin_hf(X_hf_init).flatten()

    init_budget = n_init_lf * cost_lf + n_init_hf * cost_hf

    # Separate data for each model (independent BO loops)
    # MFGP
    X_lf_mfgp, y_lf_mfgp = X_lf_init.copy(), y_lf_init.copy()
    X_hf_mfgp, y_hf_mfgp = X_hf_init.copy(), y_hf_init.copy()
    # BLR
    X_lf_blr, y_lf_blr = X_lf_init.copy(), y_lf_init.copy()
    X_hf_blr, y_hf_blr = X_hf_init.copy(), y_hf_init.copy()
    # MC Dropout
    X_lf_mc, y_lf_mc = X_lf_init.copy(), y_lf_init.copy()
    X_hf_mc, y_hf_mc = X_hf_init.copy(), y_hf_init.copy()

    budget_mfgp = init_budget
    budget_blr = init_budget
    budget_mc = init_budget

    comparison_data = []
    max_iterations = 200
    iteration = 0
    lf_per_hf = max(1, int(1.0 / rho))
    lf_counter_mfgp = 0
    lf_counter_blr = 0
    lf_counter_mc = 0

    def get_eval_decision(remaining, lf_counter):
        """Determine whether to evaluate HF or LF"""
        if remaining >= cost_hf:
            if remaining >= cost_lf and lf_counter < lf_per_hf:
                return False, cost_lf, lf_counter + 1  # eval LF
            else:
                return True, cost_hf, 0  # eval HF
        elif remaining >= cost_lf:
            return False, cost_lf, lf_counter
        else:
            return None, 0, lf_counter  # done

    while (budget_mfgp < total_budget or budget_blr < total_budget or budget_mc < total_budget) and iteration < max_iterations:
        iteration += 1

        # Get decisions for each model
        eval_hf_mfgp, cost_mfgp, lf_counter_mfgp = get_eval_decision(total_budget - budget_mfgp, lf_counter_mfgp)
        eval_hf_blr, cost_blr, lf_counter_blr = get_eval_decision(total_budget - budget_blr, lf_counter_blr)
        eval_hf_mc, cost_mc, lf_counter_mc = get_eval_decision(total_budget - budget_mc, lf_counter_mc)

        # All done?
        if eval_hf_mfgp is None and eval_hf_blr is None and eval_hf_mc is None:
            break

        y_hf_true_flat = y_hf_true.flatten()

        try:
            # ========== MFGP Model ==========
            mean_mfgp, std_mfgp, ei_mfgp, x_next_mfgp = None, None, None, None
            r2_mfgp, calib_mfgp, errors_mfgp = None, None, None

            if eval_hf_mfgp is not None:
                model_mfgp = GP_MFGP(dim)
                model_mfgp.fit(X_lf_mfgp, y_lf_mfgp, X_hf_mfgp, y_hf_mfgp)
                mean_mfgp, std_mfgp = model_mfgp.predict(X_grid)

                y_best_mfgp = y_hf_mfgp.min()
                ei_mfgp = expected_improvement(mean_mfgp, std_mfgp, y_best_mfgp)
                x_next_mfgp = X_grid[np.argmax(ei_mfgp)]

                r2_mfgp = 1 - np.sum((y_hf_true_flat - mean_mfgp)**2) / np.sum((y_hf_true_flat - np.mean(y_hf_true_flat))**2)
                errors_mfgp = np.abs(y_hf_true_flat - mean_mfgp)
                calib_mfgp, _ = spearmanr(errors_mfgp, std_mfgp)

                # Update MFGP data
                x_new_mfgp = x_next_mfgp.reshape(1, -1)
                if eval_hf_mfgp:
                    y_new_mfgp = branin_hf(x_new_mfgp).flatten()
                    X_hf_mfgp = np.vstack([X_hf_mfgp, x_new_mfgp])
                    y_hf_mfgp = np.append(y_hf_mfgp, y_new_mfgp)
                else:
                    y_new_mfgp = branin_lf(x_new_mfgp, alpha).flatten()
                    X_lf_mfgp = np.vstack([X_lf_mfgp, x_new_mfgp])
                    y_lf_mfgp = np.append(y_lf_mfgp, y_new_mfgp)
                budget_mfgp += cost_mfgp

            # ========== BLR Model ==========
            mean_blr, std_blr, ei_blr, x_next_blr = None, None, None, None
            r2_blr, calib_blr, errors_blr = None, None, None

            if eval_hf_blr is not None:
                model_blr = DNGO_Sequential_BLR(dim, l2_lambda=l2_lambda)
                model_blr.fit(X_lf_blr, y_lf_blr, X_hf_blr, y_hf_blr)
                mean_blr, std_blr = model_blr.predict(X_grid)

                y_best_blr = y_hf_blr.min()
                ei_blr = expected_improvement(mean_blr, std_blr, y_best_blr)
                x_next_blr = X_grid[np.argmax(ei_blr)]

                r2_blr = 1 - np.sum((y_hf_true_flat - mean_blr)**2) / np.sum((y_hf_true_flat - np.mean(y_hf_true_flat))**2)
                errors_blr = np.abs(y_hf_true_flat - mean_blr)
                calib_blr, _ = spearmanr(errors_blr, std_blr)

                # Update BLR data
                x_new_blr = x_next_blr.reshape(1, -1)
                if eval_hf_blr:
                    y_new_blr = branin_hf(x_new_blr).flatten()
                    X_hf_blr = np.vstack([X_hf_blr, x_new_blr])
                    y_hf_blr = np.append(y_hf_blr, y_new_blr)
                else:
                    y_new_blr = branin_lf(x_new_blr, alpha).flatten()
                    X_lf_blr = np.vstack([X_lf_blr, x_new_blr])
                    y_lf_blr = np.append(y_lf_blr, y_new_blr)
                budget_blr += cost_blr

            # ========== MC Dropout Model ==========
            mean_mc, std_mc, ei_mc, x_next_mc = None, None, None, None
            r2_mc, calib_mc, errors_mc = None, None, None

            if eval_hf_mc is not None:
                model_mc = DNGO_Sequential_MCDropout(dim, dropout=dropout, n_mc_samples=n_mc_samples, l2_lambda=l2_lambda)
                model_mc.fit(X_lf_mc, y_lf_mc, X_hf_mc, y_hf_mc)
                mean_mc, std_mc = model_mc.predict(X_grid)

                y_best_mc = y_hf_mc.min()
                ei_mc = expected_improvement(mean_mc, std_mc, y_best_mc)
                x_next_mc = X_grid[np.argmax(ei_mc)]

                r2_mc = 1 - np.sum((y_hf_true_flat - mean_mc)**2) / np.sum((y_hf_true_flat - np.mean(y_hf_true_flat))**2)
                errors_mc = np.abs(y_hf_true_flat - mean_mc)
                calib_mc, _ = spearmanr(errors_mc, std_mc)

                # Update MC data
                x_new_mc = x_next_mc.reshape(1, -1)
                if eval_hf_mc:
                    y_new_mc = branin_hf(x_new_mc).flatten()
                    X_hf_mc = np.vstack([X_hf_mc, x_new_mc])
                    y_hf_mc = np.append(y_hf_mc, y_new_mc)
                else:
                    y_new_mc = branin_lf(x_new_mc, alpha).flatten()
                    X_lf_mc = np.vstack([X_lf_mc, x_new_mc])
                    y_lf_mc = np.append(y_lf_mc, y_new_mc)
                budget_mc += cost_mc

            # Save comparison data (only when all 3 models have data)
            if iteration % save_interval == 0 and mean_mfgp is not None and mean_blr is not None and mean_mc is not None:
                comparison_data.append({
                    'step': iteration,
                    # MFGP data
                    'X_lf_mfgp': X_lf_mfgp.copy(), 'y_lf_mfgp': y_lf_mfgp.copy(),
                    'X_hf_mfgp': X_hf_mfgp.copy(), 'y_hf_mfgp': y_hf_mfgp.copy(),
                    'budget_mfgp': budget_mfgp,
                    'regret_mfgp': max(0, y_hf_mfgp.min() - f_star),
                    'mean_mfgp': mean_mfgp.reshape(n_grid, n_grid),
                    'std_mfgp': std_mfgp.reshape(n_grid, n_grid),
                    'ei_mfgp': ei_mfgp.reshape(n_grid, n_grid),
                    'x_next_mfgp': x_next_mfgp,
                    'r2_mfgp': r2_mfgp,
                    'calib_mfgp': calib_mfgp,
                    'errors_mfgp': errors_mfgp,
                    # BLR data
                    'X_lf_blr': X_lf_blr.copy(), 'y_lf_blr': y_lf_blr.copy(),
                    'X_hf_blr': X_hf_blr.copy(), 'y_hf_blr': y_hf_blr.copy(),
                    'budget_blr': budget_blr,
                    'regret_blr': max(0, y_hf_blr.min() - f_star),
                    'mean_blr': mean_blr.reshape(n_grid, n_grid),
                    'std_blr': std_blr.reshape(n_grid, n_grid),
                    'ei_blr': ei_blr.reshape(n_grid, n_grid),
                    'x_next_blr': x_next_blr,
                    'r2_blr': r2_blr,
                    'calib_blr': calib_blr,
                    'errors_blr': errors_blr,
                    # MC data
                    'X_lf_mc': X_lf_mc.copy(), 'y_lf_mc': y_lf_mc.copy(),
                    'X_hf_mc': X_hf_mc.copy(), 'y_hf_mc': y_hf_mc.copy(),
                    'budget_mc': budget_mc,
                    'y_best_mc': y_hf_mc.min(),
                    'regret_mc': max(0, y_hf_mc.min() - f_star),
                    'mean_mc': mean_mc.reshape(n_grid, n_grid),
                    'std_mc': std_mc.reshape(n_grid, n_grid),
                    'ei_mc': ei_mc.reshape(n_grid, n_grid),
                    'x_next_mc': x_next_mc,
                    'r2_mc': r2_mc,
                    'calib_mc': calib_mc,
                    'errors_mc': errors_mc,
                })

        except Exception as e:
            print(f"  Step {iteration} error: {e}")
            import traceback
            traceback.print_exc()

        if iteration % 10 == 0:
            regret_mfgp = max(0, y_hf_mfgp.min() - f_star)
            regret_blr = max(0, y_hf_blr.min() - f_star)
            regret_mc = max(0, y_hf_mc.min() - f_star)
            print(f"  Step {iteration}: MFGP={regret_mfgp:.4f} | BLR={regret_blr:.4f} | MC={regret_mc:.4f}")

    return {
        'comparison_data': comparison_data,
        'y_hf_true': y_hf_true,
        'y_lf_true': y_lf_true,
        'X1': X1, 'X2': X2,
        'x1_grid': x1_grid, 'x2_grid': x2_grid,
        'alpha': alpha, 'rho': rho,
        'f_star': f_star,
        'opt_x1': opt_x1, 'opt_x2': opt_x2,
    }


def visualize_comparison(results: dict, output_dir: Path):
    """Visualize MFGP vs BLR vs MC Dropout comparison (independent BO loops)"""
    comparison_data = results['comparison_data']
    y_hf_true = results['y_hf_true']
    X1, X2 = results['X1'], results['X2']
    f_star = results['f_star']

    # Branin has 3 global optima (normalized coordinates)
    global_optima = [
        ((-np.pi + 5) / 15, 12.275 / 15),    # ≈ (0.124, 0.818)
        ((np.pi + 5) / 15, 2.275 / 15),       # ≈ (0.543, 0.152)
        ((9.42478 + 5) / 15, 2.475 / 15),     # ≈ (0.962, 0.165)
    ]
    opt_x1_list = [opt[0] for opt in global_optima]
    opt_x2_list = [opt[1] for opt in global_optima]

    if len(comparison_data) == 0:
        print("No comparison data!")
        return

    n_steps = min(len(comparison_data), 4)
    indices = np.linspace(0, len(comparison_data) - 1, n_steps, dtype=int)
    selected_data = [comparison_data[i] for i in indices]

    for data in selected_data:
        step = data['step']
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Find best points for each model
        best_idx_mfgp = np.argmin(data['y_hf_mfgp'])
        best_idx_blr = np.argmin(data['y_hf_blr'])
        best_idx_mc = np.argmin(data['y_hf_mc'])

        # Row 1: Mean predictions (MFGP, BLR, MC, True)
        ax1 = fig.add_subplot(gs[0, 0])
        c1 = ax1.contourf(X1, X2, data['mean_mfgp'], levels=30, cmap='viridis')
        ax1.scatter(data['X_hf_mfgp'][:, 0], data['X_hf_mfgp'][:, 1], c='red', s=40, marker='o', edgecolors='white')
        ax1.scatter(data['X_hf_mfgp'][best_idx_mfgp, 0], data['X_hf_mfgp'][best_idx_mfgp, 1],
                    c='lime', s=150, marker='*', edgecolors='black', linewidths=2, zorder=10)
        ax1.scatter(opt_x1_list, opt_x2_list, c='yellow', s=200, marker='*', edgecolors='black', linewidths=2)
        ax1.set_title(f'MFGP (R²={data["r2_mfgp"]:.3f}, best={data["y_hf_mfgp"][best_idx_mfgp]:.4f})', fontsize=10)
        plt.colorbar(c1, ax=ax1, fraction=0.046)

        ax2 = fig.add_subplot(gs[0, 1])
        c2 = ax2.contourf(X1, X2, data['mean_blr'], levels=30, cmap='viridis')
        ax2.scatter(data['X_hf_blr'][:, 0], data['X_hf_blr'][:, 1], c='red', s=40, marker='o', edgecolors='white')
        ax2.scatter(data['X_hf_blr'][best_idx_blr, 0], data['X_hf_blr'][best_idx_blr, 1],
                    c='cyan', s=150, marker='*', edgecolors='black', linewidths=2, zorder=10)
        ax2.scatter(opt_x1_list, opt_x2_list, c='yellow', s=200, marker='*', edgecolors='black', linewidths=2)
        ax2.set_title(f'Seq-BLR (R²={data["r2_blr"]:.3f}, best={data["y_hf_blr"][best_idx_blr]:.4f})', fontsize=10)
        plt.colorbar(c2, ax=ax2, fraction=0.046)

        ax3 = fig.add_subplot(gs[0, 2])
        c3 = ax3.contourf(X1, X2, data['mean_mc'], levels=30, cmap='viridis')
        ax3.scatter(data['X_hf_mc'][:, 0], data['X_hf_mc'][:, 1], c='red', s=40, marker='o', edgecolors='white')
        ax3.scatter(data['X_hf_mc'][best_idx_mc, 0], data['X_hf_mc'][best_idx_mc, 1],
                    c='orange', s=150, marker='*', edgecolors='black', linewidths=2, zorder=10)
        ax3.scatter(opt_x1_list, opt_x2_list, c='yellow', s=200, marker='*', edgecolors='black', linewidths=2)
        ax3.set_title(f'Seq-MC (R²={data["r2_mc"]:.3f}, best={data["y_hf_mc"][best_idx_mc]:.4f})', fontsize=10)
        plt.colorbar(c3, ax=ax3, fraction=0.046)

        ax4 = fig.add_subplot(gs[0, 3])
        c4 = ax4.contourf(X1, X2, y_hf_true, levels=30, cmap='viridis')
        ax4.scatter(opt_x1_list, opt_x2_list, c='yellow', s=200, marker='*', edgecolors='black', linewidths=2)
        ax4.set_title(f'True HF (3 optima, f*={f_star:.4f})', fontsize=11)
        plt.colorbar(c4, ax=ax4, fraction=0.046)

        # Row 2: Uncertainty (STD)
        ax5 = fig.add_subplot(gs[1, 0])
        c5 = ax5.contourf(X1, X2, data['std_mfgp'], levels=30, cmap='Reds')
        ax5.scatter(data['X_hf_mfgp'][:, 0], data['X_hf_mfgp'][:, 1], c='blue', s=40, marker='o', edgecolors='white')
        ax5.scatter(opt_x1_list, opt_x2_list, c='yellow', s=200, marker='*', edgecolors='black', linewidths=2)
        ax5.set_title(f'MFGP Std ({data["std_mfgp"].mean():.3f})', fontsize=11)
        plt.colorbar(c5, ax=ax5, fraction=0.046)

        ax6 = fig.add_subplot(gs[1, 1])
        c6 = ax6.contourf(X1, X2, data['std_blr'], levels=30, cmap='Reds')
        ax6.scatter(data['X_hf_blr'][:, 0], data['X_hf_blr'][:, 1], c='blue', s=40, marker='o', edgecolors='white')
        ax6.scatter(opt_x1_list, opt_x2_list, c='yellow', s=200, marker='*', edgecolors='black', linewidths=2)
        ax6.set_title(f'Seq-BLR Std ({data["std_blr"].mean():.3f})', fontsize=11)
        plt.colorbar(c6, ax=ax6, fraction=0.046)

        ax7 = fig.add_subplot(gs[1, 2])
        c7 = ax7.contourf(X1, X2, data['std_mc'], levels=30, cmap='Reds')
        ax7.scatter(data['X_hf_mc'][:, 0], data['X_hf_mc'][:, 1], c='blue', s=40, marker='o', edgecolors='white')
        ax7.scatter(opt_x1_list, opt_x2_list, c='yellow', s=200, marker='*', edgecolors='black', linewidths=2)
        ax7.set_title(f'Seq-MC Std ({data["std_mc"].mean():.3f})', fontsize=11)
        plt.colorbar(c7, ax=ax7, fraction=0.046)

        ax8 = fig.add_subplot(gs[1, 3])
        ax8.hist(data['std_mfgp'].flatten(), bins=50, alpha=0.5, label=f'MFGP ({data["std_mfgp"].mean():.3f})', color='green')
        ax8.hist(data['std_blr'].flatten(), bins=50, alpha=0.5, label=f'BLR ({data["std_blr"].mean():.3f})', color='blue')
        ax8.hist(data['std_mc'].flatten(), bins=50, alpha=0.5, label=f'MC ({data["std_mc"].mean():.3f})', color='red')
        ax8.set_xlabel('Std')
        ax8.set_title('Std Distribution', fontsize=11)
        ax8.legend(fontsize=8)

        # Row 3: EI
        ax9 = fig.add_subplot(gs[2, 0])
        c9 = ax9.contourf(X1, X2, data['ei_mfgp'], levels=30, cmap='plasma')
        ax9.scatter(data['x_next_mfgp'][0], data['x_next_mfgp'][1], c='lime', s=200, marker='*', edgecolors='black')
        ax9.scatter(opt_x1_list, opt_x2_list, c='yellow', s=150, marker='*', edgecolors='black')
        ax9.set_title(f'MFGP EI ({data["ei_mfgp"].max():.4f})', fontsize=11)
        plt.colorbar(c9, ax=ax9, fraction=0.046)

        ax10 = fig.add_subplot(gs[2, 1])
        c10 = ax10.contourf(X1, X2, data['ei_blr'], levels=30, cmap='plasma')
        ax10.scatter(data['x_next_blr'][0], data['x_next_blr'][1], c='lime', s=200, marker='*', edgecolors='black')
        ax10.scatter(opt_x1_list, opt_x2_list, c='yellow', s=150, marker='*', edgecolors='black')
        ax10.set_title(f'Seq-BLR EI ({data["ei_blr"].max():.4f})', fontsize=11)
        plt.colorbar(c10, ax=ax10, fraction=0.046)

        ax11 = fig.add_subplot(gs[2, 2])
        c11 = ax11.contourf(X1, X2, data['ei_mc'], levels=30, cmap='plasma')
        ax11.scatter(data['x_next_mc'][0], data['x_next_mc'][1], c='lime', s=200, marker='*', edgecolors='black')
        ax11.scatter(opt_x1_list, opt_x2_list, c='yellow', s=150, marker='*', edgecolors='black')
        ax11.set_title(f'Seq-MC EI ({data["ei_mc"].max():.4f})', fontsize=11)
        plt.colorbar(c11, ax=ax11, fraction=0.046)

        ax12 = fig.add_subplot(gs[2, 3])
        ax12.contourf(X1, X2, y_hf_true, levels=30, cmap='viridis', alpha=0.5)
        ax12.scatter(data['X_hf_mfgp'][:, 0], data['X_hf_mfgp'][:, 1], c='green', s=40, marker='o',
                     edgecolors='white', label=f'MFGP ({len(data["X_hf_mfgp"])})')
        ax12.scatter(data['X_hf_blr'][:, 0], data['X_hf_blr'][:, 1], c='blue', s=40, marker='s',
                     edgecolors='white', label=f'BLR ({len(data["X_hf_blr"])})')
        ax12.scatter(data['X_hf_mc'][:, 0], data['X_hf_mc'][:, 1], c='red', s=40, marker='^',
                     edgecolors='white', label=f'MC ({len(data["X_hf_mc"])})')
        # Best points for each
        ax12.scatter(data['X_hf_mfgp'][best_idx_mfgp, 0], data['X_hf_mfgp'][best_idx_mfgp, 1],
                     c='lime', s=120, marker='*', edgecolors='black', linewidths=2, zorder=10)
        ax12.scatter(data['X_hf_blr'][best_idx_blr, 0], data['X_hf_blr'][best_idx_blr, 1],
                     c='cyan', s=120, marker='*', edgecolors='black', linewidths=2, zorder=10)
        ax12.scatter(data['X_hf_mc'][best_idx_mc, 0], data['X_hf_mc'][best_idx_mc, 1],
                     c='orange', s=120, marker='*', edgecolors='black', linewidths=2, zorder=10)
        ax12.scatter(opt_x1_list, opt_x2_list, c='yellow', s=200, marker='*', edgecolors='black', label='3 Optima')
        ax12.set_title('Sampled HF & Best Points', fontsize=11)
        ax12.legend(loc='lower left', fontsize=6)

        # Row 4: Calibration
        ax13 = fig.add_subplot(gs[3, 0])
        ax13.scatter(data['std_mfgp'].flatten(), data['errors_mfgp'], alpha=0.3, s=5, c='green')
        ax13.set_xlabel('Predicted Std')
        ax13.set_ylabel('|Error|')
        ax13.set_title(f'MFGP Calib (ρ={data["calib_mfgp"]:.3f})', fontsize=11)
        max_val = max(data['std_mfgp'].max(), data['errors_mfgp'].max())
        ax13.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)

        ax14 = fig.add_subplot(gs[3, 1])
        ax14.scatter(data['std_blr'].flatten(), data['errors_blr'], alpha=0.3, s=5, c='blue')
        ax14.set_xlabel('Predicted Std')
        ax14.set_ylabel('|Error|')
        ax14.set_title(f'Seq-BLR Calib (ρ={data["calib_blr"]:.3f})', fontsize=11)
        max_val = max(data['std_blr'].max(), data['errors_blr'].max())
        ax14.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)

        ax15 = fig.add_subplot(gs[3, 2])
        ax15.scatter(data['std_mc'].flatten(), data['errors_mc'], alpha=0.3, s=5, c='red')
        ax15.set_xlabel('Predicted Std')
        ax15.set_ylabel('|Error|')
        ax15.set_title(f'Seq-MC Calib (ρ={data["calib_mc"]:.3f})', fontsize=11)
        max_val = max(data['std_mc'].max(), data['errors_mc'].max())
        ax15.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)

        # Summary bar chart
        ax16 = fig.add_subplot(gs[3, 3])
        models = ['MFGP', 'Seq-BLR', 'Seq-MC']
        regrets = [data['regret_mfgp'], data['regret_blr'], data['regret_mc']]
        colors = ['green', 'blue', 'red']
        bars = ax16.bar(models, regrets, color=colors, alpha=0.7)
        ax16.set_ylabel('Regret')
        ax16.set_title('Regret Comparison', fontsize=11)
        for bar, reg in zip(bars, regrets):
            ax16.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                      f'{reg:.4f}', ha='center', va='bottom', fontsize=9)

        fig.suptitle(f'Step {step} | MFGP: {data["regret_mfgp"]:.4f} | BLR: {data["regret_blr"]:.4f} | MC: {data["regret_mc"]:.4f}',
                     fontsize=14, fontweight='bold', y=0.995)

        plt.savefig(output_dir / f'comparison_step_{step:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: comparison_step_{step:03d}.png")


def visualize_summary(results: dict, output_dir: Path):
    """Summary metrics with regret comparison for 3 models"""
    comparison_data = results['comparison_data']
    f_star = results['f_star']
    if len(comparison_data) == 0:
        return

    steps = [d['step'] for d in comparison_data]
    # MFGP
    r2_mfgp = [d['r2_mfgp'] for d in comparison_data]
    calib_mfgp = [d['calib_mfgp'] for d in comparison_data]
    std_mean_mfgp = [d['std_mfgp'].mean() for d in comparison_data]
    regret_mfgp = [d['regret_mfgp'] for d in comparison_data]
    # BLR
    r2_blr = [d['r2_blr'] for d in comparison_data]
    calib_blr = [d['calib_blr'] for d in comparison_data]
    std_mean_blr = [d['std_blr'].mean() for d in comparison_data]
    regret_blr = [d['regret_blr'] for d in comparison_data]
    # MC
    r2_mc = [d['r2_mc'] for d in comparison_data]
    calib_mc = [d['calib_mc'] for d in comparison_data]
    std_mean_mc = [d['std_mc'].mean() for d in comparison_data]
    regret_mc = [d['regret_mc'] for d in comparison_data]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Regret comparison (most important!)
    ax1 = axes[0, 0]
    ax1.plot(steps, regret_mfgp, 'g-o', label='MFGP', linewidth=2, markersize=6)
    ax1.plot(steps, regret_blr, 'b-s', label='Seq-BLR', linewidth=2, markersize=6)
    ax1.plot(steps, regret_mc, 'r-^', label='Seq-MC', linewidth=2, markersize=6)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, label=f'f*={f_star:.4f}')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Regret')
    ax1.set_title('Simple Regret (lower is better)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(steps, r2_mfgp, 'g-o', label='MFGP', linewidth=2, markersize=6)
    ax2.plot(steps, r2_blr, 'b-s', label='Seq-BLR', linewidth=2, markersize=6)
    ax2.plot(steps, r2_mc, 'r-^', label='Seq-MC', linewidth=2, markersize=6)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('R²')
    ax2.set_title('Prediction Accuracy (R²)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.plot(steps, calib_mfgp, 'g-o', label='MFGP', linewidth=2, markersize=6)
    ax3.plot(steps, calib_blr, 'b-s', label='Seq-BLR', linewidth=2, markersize=6)
    ax3.plot(steps, calib_mc, 'r-^', label='Seq-MC', linewidth=2, markersize=6)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Spearman ρ')
    ax3.set_title('Uncertainty Calibration', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.plot(steps, std_mean_mfgp, 'g-o', label='MFGP', linewidth=2, markersize=6)
    ax4.plot(steps, std_mean_blr, 'b-s', label='Seq-BLR', linewidth=2, markersize=6)
    ax4.plot(steps, std_mean_mc, 'r-^', label='Seq-MC', linewidth=2, markersize=6)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Mean Std')
    ax4.set_title('Average Uncertainty', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Final comparison
    final_regrets = {'MFGP': regret_mfgp[-1], 'Seq-BLR': regret_blr[-1], 'Seq-MC': regret_mc[-1]}
    winner = min(final_regrets, key=final_regrets.get)

    fig.suptitle(f'MFGP={regret_mfgp[-1]:.4f} | BLR={regret_blr[-1]:.4f} | MC={regret_mc[-1]:.4f} | Winner: {winner}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: summary_comparison.png")


def visualize_3d(results: dict, output_dir: Path):
    """Interactive 3D visualization using Plotly"""
    comparison_data = results['comparison_data']
    y_hf_true = results['y_hf_true']
    X1, X2 = results['X1'], results['X2']
    f_star = results['f_star']

    # Branin has 3 global optima (normalized coordinates)
    # Original: (-π, 12.275), (π, 2.275), (9.42478, 2.475)
    # Normalized: x1 = (x1_orig + 5) / 15, x2 = x2_orig / 15
    global_optima = [
        ((-np.pi + 5) / 15, 12.275 / 15),    # ≈ (0.124, 0.818)
        ((np.pi + 5) / 15, 2.275 / 15),       # ≈ (0.543, 0.152)
        ((9.42478 + 5) / 15, 2.475 / 15),     # ≈ (0.962, 0.165)
    ]

    if len(comparison_data) == 0:
        print("No comparison data for 3D visualization!")
        return

    # Use final step data
    data = comparison_data[-1]
    step = data['step']

    # Find best point for each algorithm
    best_idx_mfgp = np.argmin(data['y_hf_mfgp'])
    best_mfgp = (data['X_hf_mfgp'][best_idx_mfgp, 0], data['X_hf_mfgp'][best_idx_mfgp, 1], data['y_hf_mfgp'][best_idx_mfgp])

    best_idx_blr = np.argmin(data['y_hf_blr'])
    best_blr = (data['X_hf_blr'][best_idx_blr, 0], data['X_hf_blr'][best_idx_blr, 1], data['y_hf_blr'][best_idx_blr])

    best_idx_mc = np.argmin(data['y_hf_mc'])
    best_mc = (data['X_hf_mc'][best_idx_mc, 0], data['X_hf_mc'][best_idx_mc, 1], data['y_hf_mc'][best_idx_mc])

    # Downsample for performance (every 2nd point)
    skip = 2
    X1_ds = X1[::skip, ::skip]
    X2_ds = X2[::skip, ::skip]
    y_true_ds = y_hf_true[::skip, ::skip]
    mean_mfgp_ds = data['mean_mfgp'][::skip, ::skip]
    mean_blr_ds = data['mean_blr'][::skip, ::skip]
    mean_mc_ds = data['mean_mc'][::skip, ::skip]

    # Create 2x2 subplot with 3D surfaces
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}],
               [{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=[
            f'True HF (f*={f_star:.4f})',
            f'MFGP (R²={data["r2_mfgp"]:.3f}, Regret={data["regret_mfgp"]:.4f})',
            f'Seq-BLR (R²={data["r2_blr"]:.3f}, Regret={data["regret_blr"]:.4f})',
            f'Seq-MC (R²={data["r2_mc"]:.3f}, Regret={data["regret_mc"]:.4f})'
        ],
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )

    # Common colorscale
    colorscale = 'Viridis'
    zmin = min(y_true_ds.min(), mean_mfgp_ds.min(), mean_blr_ds.min(), mean_mc_ds.min())
    zmax = max(y_true_ds.max(), mean_mfgp_ds.max(), mean_blr_ds.max(), mean_mc_ds.max())

    # True HF surface
    fig.add_trace(
        go.Surface(x=X1_ds, y=X2_ds, z=y_true_ds, colorscale=colorscale,
                   cmin=zmin, cmax=zmax, showscale=False, name='True HF',
                   opacity=0.9),
        row=1, col=1
    )
    # 3 Global optima points
    opt_x1_list = [opt[0] for opt in global_optima]
    opt_x2_list = [opt[1] for opt in global_optima]
    fig.add_trace(
        go.Scatter3d(x=opt_x1_list, y=opt_x2_list, z=[f_star]*3,
                     mode='markers', marker=dict(size=10, color='yellow', symbol='diamond',
                                                  line=dict(color='black', width=2)),
                     name='Global Optima (3)'),
        row=1, col=1
    )

    # MFGP surface
    fig.add_trace(
        go.Surface(x=X1_ds, y=X2_ds, z=mean_mfgp_ds, colorscale=colorscale,
                   cmin=zmin, cmax=zmax, showscale=False, name='MFGP',
                   opacity=0.9),
        row=1, col=2
    )
    # MFGP sampled HF points
    fig.add_trace(
        go.Scatter3d(x=data['X_hf_mfgp'][:, 0], y=data['X_hf_mfgp'][:, 1],
                     z=data['y_hf_mfgp'],
                     mode='markers', marker=dict(size=4, color='lightgreen'),
                     name='MFGP HF'),
        row=1, col=2
    )
    # MFGP best point found
    fig.add_trace(
        go.Scatter3d(x=[best_mfgp[0]], y=[best_mfgp[1]], z=[best_mfgp[2]],
                     mode='markers', marker=dict(size=12, color='lime', symbol='star',
                                                  line=dict(color='black', width=2)),
                     name=f'MFGP Best ({best_mfgp[2]:.4f})'),
        row=1, col=2
    )

    # Seq-BLR surface
    fig.add_trace(
        go.Surface(x=X1_ds, y=X2_ds, z=mean_blr_ds, colorscale=colorscale,
                   cmin=zmin, cmax=zmax, showscale=False, name='Seq-BLR',
                   opacity=0.9),
        row=2, col=1
    )
    # BLR sampled HF points
    fig.add_trace(
        go.Scatter3d(x=data['X_hf_blr'][:, 0], y=data['X_hf_blr'][:, 1],
                     z=data['y_hf_blr'],
                     mode='markers', marker=dict(size=4, color='lightblue'),
                     name='BLR HF'),
        row=2, col=1
    )
    # BLR best point found
    fig.add_trace(
        go.Scatter3d(x=[best_blr[0]], y=[best_blr[1]], z=[best_blr[2]],
                     mode='markers', marker=dict(size=12, color='cyan', symbol='star',
                                                  line=dict(color='black', width=2)),
                     name=f'BLR Best ({best_blr[2]:.4f})'),
        row=2, col=1
    )

    # Seq-MC surface
    fig.add_trace(
        go.Surface(x=X1_ds, y=X2_ds, z=mean_mc_ds, colorscale=colorscale,
                   cmin=zmin, cmax=zmax, showscale=True, name='Seq-MC',
                   opacity=0.9),
        row=2, col=2
    )
    # MC sampled HF points
    fig.add_trace(
        go.Scatter3d(x=data['X_hf_mc'][:, 0], y=data['X_hf_mc'][:, 1],
                     z=data['y_hf_mc'],
                     mode='markers', marker=dict(size=4, color='lightsalmon'),
                     name='MC HF'),
        row=2, col=2
    )
    # MC best point found
    fig.add_trace(
        go.Scatter3d(x=[best_mc[0]], y=[best_mc[1]], z=[best_mc[2]],
                     mode='markers', marker=dict(size=12, color='orange', symbol='star',
                                                  line=dict(color='black', width=2)),
                     name=f'MC Best ({best_mc[2]:.4f})'),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Step {step} | 3D Surface Comparison<br>'
                 f'<sup>MFGP: {data["regret_mfgp"]:.4f} | BLR: {data["regret_blr"]:.4f} | MC: {data["regret_mc"]:.4f}</sup>',
            x=0.5, font=dict(size=16)
        ),
        height=900,
        width=1200,
        showlegend=True,
        legend=dict(x=1.02, y=0.5)
    )

    # Update all scenes
    scene_settings = dict(
        xaxis_title='X1',
        yaxis_title='X2',
        zaxis_title='f(x)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=0.7)
    )
    fig.update_scenes(scene_settings)

    # Save as HTML
    html_path = output_dir / f'3d_comparison_step_{step:03d}.html'
    fig.write_html(str(html_path))
    print(f"  Saved: {html_path.name}")

    # Also create comparison: True vs Each model (overlay)
    fig2 = go.Figure()

    # True surface (semi-transparent)
    fig2.add_trace(
        go.Surface(x=X1_ds, y=X2_ds, z=y_true_ds, colorscale='Greys',
                   opacity=0.4, name='True HF', showscale=False)
    )

    # MFGP surface
    fig2.add_trace(
        go.Surface(x=X1_ds, y=X2_ds, z=mean_mfgp_ds, colorscale='Greens',
                   opacity=0.6, name='MFGP', showscale=False)
    )

    # Add error heatmap on base
    error_mfgp = np.abs(data['mean_mfgp'] - y_hf_true)[::skip, ::skip]
    error_blr = np.abs(data['mean_blr'] - y_hf_true)[::skip, ::skip]
    error_mc = np.abs(data['mean_mc'] - y_hf_true)[::skip, ::skip]

    fig2.update_layout(
        title=f'MFGP vs True HF | Mean Error: {error_mfgp.mean():.4f}',
        scene=dict(
            xaxis_title='X1',
            yaxis_title='X2',
            zaxis_title='f(x)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
        ),
        height=700,
        width=900
    )
    fig2.write_html(str(output_dir / f'3d_mfgp_vs_true_step_{step:03d}.html'))

    # BLR vs True
    fig3 = go.Figure()
    fig3.add_trace(
        go.Surface(x=X1_ds, y=X2_ds, z=y_true_ds, colorscale='Greys',
                   opacity=0.4, name='True HF', showscale=False)
    )
    fig3.add_trace(
        go.Surface(x=X1_ds, y=X2_ds, z=mean_blr_ds, colorscale='Blues',
                   opacity=0.6, name='Seq-BLR', showscale=False)
    )
    fig3.update_layout(
        title=f'Seq-BLR vs True HF | Mean Error: {error_blr.mean():.4f}',
        scene=dict(
            xaxis_title='X1',
            yaxis_title='X2',
            zaxis_title='f(x)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
        ),
        height=700,
        width=900
    )
    fig3.write_html(str(output_dir / f'3d_blr_vs_true_step_{step:03d}.html'))

    # MC vs True
    fig4 = go.Figure()
    fig4.add_trace(
        go.Surface(x=X1_ds, y=X2_ds, z=y_true_ds, colorscale='Greys',
                   opacity=0.4, name='True HF', showscale=False)
    )
    fig4.add_trace(
        go.Surface(x=X1_ds, y=X2_ds, z=mean_mc_ds, colorscale='Reds',
                   opacity=0.6, name='Seq-MC', showscale=False)
    )
    fig4.update_layout(
        title=f'Seq-MC vs True HF | Mean Error: {error_mc.mean():.4f}',
        scene=dict(
            xaxis_title='X1',
            yaxis_title='X2',
            zaxis_title='f(x)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
        ),
        height=700,
        width=900
    )
    fig4.write_html(str(output_dir / f'3d_mc_vs_true_step_{step:03d}.html'))

    # All 3 models overlaid on True
    fig5 = go.Figure()
    fig5.add_trace(
        go.Surface(x=X1_ds, y=X2_ds, z=y_true_ds, colorscale='Greys',
                   opacity=0.3, name='True HF', showscale=False)
    )
    fig5.add_trace(
        go.Surface(x=X1_ds, y=X2_ds, z=mean_mfgp_ds, colorscale='Greens',
                   opacity=0.5, name='MFGP', showscale=False)
    )
    fig5.add_trace(
        go.Surface(x=X1_ds, y=X2_ds, z=mean_blr_ds, colorscale='Blues',
                   opacity=0.5, name='Seq-BLR', showscale=False)
    )
    fig5.add_trace(
        go.Surface(x=X1_ds, y=X2_ds, z=mean_mc_ds, colorscale='Reds',
                   opacity=0.5, name='Seq-MC', showscale=False)
    )
    # 3 Global Optima
    fig5.add_trace(
        go.Scatter3d(x=opt_x1_list, y=opt_x2_list, z=[f_star]*3,
                     mode='markers', marker=dict(size=12, color='yellow', symbol='diamond',
                                                  line=dict(color='black', width=2)),
                     name=f'Global Optima (f*={f_star:.4f})')
    )
    # Best points from each algorithm
    fig5.add_trace(
        go.Scatter3d(x=[best_mfgp[0]], y=[best_mfgp[1]], z=[best_mfgp[2]],
                     mode='markers', marker=dict(size=10, color='lime', symbol='star',
                                                  line=dict(color='black', width=2)),
                     name=f'MFGP Best ({best_mfgp[2]:.4f})')
    )
    fig5.add_trace(
        go.Scatter3d(x=[best_blr[0]], y=[best_blr[1]], z=[best_blr[2]],
                     mode='markers', marker=dict(size=10, color='cyan', symbol='star',
                                                  line=dict(color='black', width=2)),
                     name=f'BLR Best ({best_blr[2]:.4f})')
    )
    fig5.add_trace(
        go.Scatter3d(x=[best_mc[0]], y=[best_mc[1]], z=[best_mc[2]],
                     mode='markers', marker=dict(size=10, color='orange', symbol='star',
                                                  line=dict(color='black', width=2)),
                     name=f'MC Best ({best_mc[2]:.4f})')
    )
    fig5.update_layout(
        title=f'All Models vs True HF | Step {step}<br>'
              f'<sup>Best Found: MFGP={best_mfgp[2]:.4f}, BLR={best_blr[2]:.4f}, MC={best_mc[2]:.4f} | f*={f_star:.4f}</sup>',
        scene=dict(
            xaxis_title='X1',
            yaxis_title='X2',
            zaxis_title='f(x)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
        ),
        height=800,
        width=1000,
        showlegend=True
    )
    fig5.write_html(str(output_dir / f'3d_all_models_step_{step:03d}.html'))
    print(f"  Saved: 3d_all_models_step_{step:03d}.html")


def main():
    parser = argparse.ArgumentParser(description='MFGP vs Seq-BLR vs Seq-MC Comparison')
    parser.add_argument('--seed', type=int, default=1004, help='Random seed')
    parser.add_argument('--budget', type=float, default=50, help='Total budget')
    parser.add_argument('--save-interval', type=int, default=5, help='Save interval')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for MC')
    parser.add_argument('--n-mc-samples', type=int, default=50, help='MC samples')
    parser.add_argument('--l2-lambda', type=float, default=1e-3, help='L2 regularization strength for DNGO models')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'compare_mf_models_{timestamp}')
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Multi-Fidelity Model Comparison (Independent BO)")
    print("=" * 60)
    print("Models: MFGP, DNGO_Sequential_BLR, DNGO_Sequential_MCDropout")
    print(f"Seed: {args.seed}")
    print(f"Budget: {args.budget}")
    print(f"MC Dropout: {args.dropout}, MC Samples: {args.n_mc_samples}")
    print(f"L2 Lambda: {args.l2_lambda}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    results = run_comparison(
        args.seed, args.budget, args.save_interval,
        dropout=args.dropout, n_mc_samples=args.n_mc_samples, l2_lambda=args.l2_lambda
    )

    print("\nGenerating visualizations...")
    visualize_comparison(results, output_dir)
    visualize_summary(results, output_dir)
    visualize_3d(results, output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
