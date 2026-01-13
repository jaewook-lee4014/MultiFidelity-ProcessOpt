#!/usr/bin/env python3
"""
Bayesian Optimization with Online HP Tuning

Features:
- 6 SF models + 12 MF models = 18 total models
- LOOCV-based HP tuning every N iterations (default: at start only for 50-iter runs)
- Synthetic test functions: Branin-2D, Park-4D
- EI visualization for seed 0

Online HP Tuning Strategy:
- For DNN-based models, tune HP using LOOCV on current data
- Tune at iteration 0 (with initial data)
- For longer runs (>50 iter), tune periodically
"""

import numpy as np
import torch
import torch.nn as nn
import json
import os
import optuna
from datetime import datetime
from typing import Dict, List, Tuple, Callable, Optional
from scipy.stats import norm, qmc
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Local imports
from synthetic_functions import branin_hf, branin_lf, park_hf, park_lf, find_global_minimum

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# HP TUNING VIA LOOCV
# =============================================================================

def loocv_score(model_class, hp: Dict, X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute LOOCV MSE for a model with given hyperparameters

    Returns negative MSE (for maximization in Optuna)
    """
    n = len(X)
    if n < 3:  # Need at least 3 points for meaningful LOOCV
        return -1e10

    errors = []
    for i in range(n):
        # Leave one out
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i:i+1]
        y_test = y[i]

        try:
            # Create and fit model
            model = model_class(input_dim=X.shape[1], **hp)
            model.fit(X_train, y_train)

            # Predict
            pred, _ = model.predict(X_test)
            errors.append((pred[0] - y_test) ** 2)
        except Exception as e:
            errors.append(1e6)  # Penalty for failed models

    return -np.mean(errors)  # Negative MSE for maximization


def tune_hyperparameters(model_name: str, X: np.ndarray, y: np.ndarray,
                         n_trials: int = 20) -> Dict:
    """
    Tune hyperparameters using LOOCV with Optuna

    Returns best hyperparameters for the model
    """
    dim = X.shape[1]

    def objective(trial):
        if model_name == 'GP':
            hp = {
                'length_scale': trial.suggest_float('length_scale', 0.1, 5.0),
                'noise_level': trial.suggest_float('noise_level', 1e-4, 0.1, log=True),
                'n_restarts': trial.suggest_int('n_restarts', 3, 10),
            }
            return loocv_score(GPModel, hp, X, y)

        elif model_name == 'DNGO':
            hp = {
                'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
                'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                'epochs': trial.suggest_int('epochs', 100, 500),
                'blr_alpha': trial.suggest_float('blr_alpha', 0.01, 1.0),
                'blr_beta': trial.suggest_float('blr_beta', 0.5, 5.0),
            }
            return loocv_score(DNGOModel, hp, X, y)

        elif model_name == 'BNN':
            hp = {
                'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
                'num_layers': trial.suggest_int('num_layers', 1, 3),
                'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                'epochs': trial.suggest_int('epochs', 100, 400),
                'kl_weight': trial.suggest_float('kl_weight', 0.01, 0.5),
            }
            return loocv_score(BNNModel, hp, X, y)

        elif model_name == 'MC-Dropout':
            hp = {
                'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                'epochs': trial.suggest_int('epochs', 100, 500),
            }
            return loocv_score(MCDropoutModel, hp, X, y)

        elif model_name == 'Deep Ensemble':
            hp = {
                'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
                'n_ensemble': trial.suggest_int('n_ensemble', 3, 7),
                'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                'epochs': trial.suggest_int('epochs', 100, 400),
            }
            return loocv_score(DeepEnsembleModel, hp, X, y)

        elif model_name == 'SNGP':
            hp = {
                'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
                'n_inducing': trial.suggest_int('n_inducing', 10, 50),
                'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                'epochs': trial.suggest_int('epochs', 100, 400),
            }
            return loocv_score(SNGPModel, hp, X, y)

        else:
            return -1e10

    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params


# =============================================================================
# SF UQ MODELS (6 models)
# =============================================================================

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


class GPModel:
    """Gaussian Process with Matern kernel"""

    def __init__(self, input_dim: int = None, length_scale: float = 1.0,
                 noise_level: float = 0.01, n_restarts: int = 5):
        self.kernel = Matern(length_scale=length_scale, nu=2.5) + \
                      WhiteKernel(noise_level=noise_level)
        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=n_restarts,
            normalize_y=True,
            random_state=42
        )
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = self.scaler_x.fit_transform(X)
        self.y_train = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        self.model.fit(self.X_train, self.y_train)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler_x.transform(X)
        mean, std = self.model.predict(X_scaled, return_std=True)
        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
        std = std * self.scaler_y.scale_[0]
        return mean, np.maximum(std, 1e-6)


class DNGOModel:
    """Deep Networks for Global Optimization"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, lr: float = 0.005,
                 epochs: int = 200, blr_alpha: float = 0.1, blr_beta: float = 2.0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.blr_alpha = blr_alpha
        self.blr_beta = blr_beta
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 50)
        ).to(device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        X_t = torch.FloatTensor(X_scaled).to(device)
        y_t = torch.FloatTensor(y_scaled).to(device)

        self.network = self._build_network()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        self.network.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            features = self.network(X_t)
            pred = features.mean(dim=1)
            loss = nn.MSELoss()(pred, y_t)
            loss.backward()
            optimizer.step()

        self.network.eval()
        with torch.no_grad():
            self.features = self.network(X_t).cpu().numpy()
        self.y_train = y_scaled

        # Bayesian Linear Regression
        Phi = self.features
        A = self.blr_alpha * np.eye(Phi.shape[1]) + self.blr_beta * Phi.T @ Phi
        self.A_inv = np.linalg.inv(A + 1e-6 * np.eye(A.shape[0]))
        self.m = self.blr_beta * self.A_inv @ Phi.T @ self.y_train

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_scaled = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(device)

        self.network.eval()
        with torch.no_grad():
            Phi = self.network(X_t).cpu().numpy()

        mean = Phi @ self.m
        var = np.array([phi @ self.A_inv @ phi for phi in Phi]) / self.blr_beta
        std = np.sqrt(np.maximum(var, 1e-8))

        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
        std = std * self.scaler_y.scale_[0]

        return mean, std


class BNNModel:
    """Bayesian Neural Network with variational inference"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 lr: float = 0.01, epochs: int = 200, kl_weight: float = 0.1,
                 n_samples: int = 20):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.kl_weight = kl_weight
        self.n_samples = n_samples
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        X_t = torch.FloatTensor(X_scaled).to(device)
        y_t = torch.FloatTensor(y_scaled).to(device)

        # Build variational parameters
        self.mu = nn.ParameterList()
        self.log_sigma = nn.ParameterList()

        in_dim = self.input_dim
        for _ in range(self.num_layers):
            self.mu.append(nn.Parameter(torch.randn(in_dim, self.hidden_dim) * 0.1))
            self.log_sigma.append(nn.Parameter(torch.ones(in_dim, self.hidden_dim) * -3))
            in_dim = self.hidden_dim

        self.mu.append(nn.Parameter(torch.randn(self.hidden_dim, 1) * 0.1))
        self.log_sigma.append(nn.Parameter(torch.ones(self.hidden_dim, 1) * -3))

        self.mu = self.mu.to(device)
        self.log_sigma = self.log_sigma.to(device)

        optimizer = torch.optim.Adam(list(self.mu) + list(self.log_sigma), lr=self.lr)

        for _ in range(self.epochs):
            optimizer.zero_grad()

            kl = 0
            preds = []
            for _ in range(5):
                W = []
                for mu, log_sigma in zip(self.mu, self.log_sigma):
                    sigma = torch.exp(log_sigma)
                    eps = torch.randn_like(mu)
                    w = mu + sigma * eps
                    W.append(w)
                    kl += 0.5 * torch.sum(sigma**2 + mu**2 - 2*log_sigma - 1)

                h = X_t
                for i, w in enumerate(W[:-1]):
                    h = torch.tanh(h @ w)
                pred = h @ W[-1]
                preds.append(pred)

            preds = torch.stack(preds)
            mean_pred = preds.mean(dim=0).squeeze()

            nll = 0.5 * torch.mean((mean_pred - y_t)**2)
            loss = nll + self.kl_weight * kl / len(X)

            loss.backward()
            optimizer.step()

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_scaled = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(device)

        preds = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                W = []
                for mu, log_sigma in zip(self.mu, self.log_sigma):
                    sigma = torch.exp(log_sigma)
                    eps = torch.randn_like(mu)
                    w = mu + sigma * eps
                    W.append(w)

                h = X_t
                for w in W[:-1]:
                    h = torch.tanh(h @ w)
                pred = h @ W[-1]
                preds.append(pred.cpu().numpy())

        preds = np.array(preds).squeeze()
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)

        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
        std = std * self.scaler_y.scale_[0]

        return mean, np.maximum(std, 1e-6)


class MCDropoutModel:
    """MC-Dropout for uncertainty estimation"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.2,
                 lr: float = 0.005, epochs: int = 200, n_samples: int = 20):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.epochs = epochs
        self.n_samples = n_samples
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 1)
        ).to(device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        X_t = torch.FloatTensor(X_scaled).to(device)
        y_t = torch.FloatTensor(y_scaled).to(device)

        self.network = self._build_network()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        self.network.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = self.network(X_t).squeeze()
            loss = nn.MSELoss()(pred, y_t)
            loss.backward()
            optimizer.step()

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_scaled = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(device)

        self.network.train()  # Enable dropout during prediction
        preds = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.network(X_t).cpu().numpy()
                preds.append(pred)

        preds = np.array(preds).squeeze()
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)

        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
        std = std * self.scaler_y.scale_[0]

        return mean, np.maximum(std, 1e-6)


class DeepEnsembleModel:
    """Deep Ensemble for uncertainty estimation"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, n_ensemble: int = 5,
                 lr: float = 0.005, epochs: int = 200):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_ensemble = n_ensemble
        self.lr = lr
        self.epochs = epochs
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        ).to(device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        X_t = torch.FloatTensor(X_scaled).to(device)
        y_t = torch.FloatTensor(y_scaled).to(device)

        self.networks = []
        for i in range(self.n_ensemble):
            network = self._build_network()
            optimizer = torch.optim.Adam(network.parameters(), lr=self.lr)

            # Bootstrap sampling
            idx = np.random.choice(len(X), len(X), replace=True)
            X_boot = X_t[idx]
            y_boot = y_t[idx]

            network.train()
            for _ in range(self.epochs):
                optimizer.zero_grad()
                pred = network(X_boot).squeeze()
                loss = nn.MSELoss()(pred, y_boot)
                loss.backward()
                optimizer.step()

            self.networks.append(network)

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_scaled = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(device)

        preds = []
        for network in self.networks:
            network.eval()
            with torch.no_grad():
                pred = network(X_t).cpu().numpy()
                preds.append(pred)

        preds = np.array(preds).squeeze()
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)

        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
        std = std * self.scaler_y.scale_[0]

        return mean, np.maximum(std, 1e-6)


class SNGPModel:
    """Spectral-normalized Neural GP"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, n_inducing: int = 20,
                 lr: float = 0.005, epochs: int = 200):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_inducing = n_inducing
        self.lr = lr
        self.epochs = epochs
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        X_t = torch.FloatTensor(X_scaled).to(device)
        y_t = torch.FloatTensor(y_scaled).to(device)

        # Feature extractor with spectral normalization
        self.feature_net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.input_dim, self.hidden_dim)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
        ).to(device)

        # Random Fourier features for GP
        self.n_features = min(self.n_inducing, len(X))
        self.omega = torch.randn(self.hidden_dim, self.n_features).to(device) * 0.5
        self.b = torch.rand(self.n_features).to(device) * 2 * np.pi

        optimizer = torch.optim.Adam(self.feature_net.parameters(), lr=self.lr)

        self.feature_net.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            h = self.feature_net(X_t)
            phi = torch.cos(h @ self.omega + self.b) * np.sqrt(2.0 / self.n_features)
            pred = phi.mean(dim=1)
            loss = nn.MSELoss()(pred, y_t)
            loss.backward()
            optimizer.step()

        # Compute GP posterior
        self.feature_net.eval()
        with torch.no_grad():
            h = self.feature_net(X_t)
            self.Phi = torch.cos(h @ self.omega + self.b) * np.sqrt(2.0 / self.n_features)
            self.Phi = self.Phi.cpu().numpy()

        # BLR on features
        alpha, beta = 0.1, 2.0
        A = alpha * np.eye(self.n_features) + beta * self.Phi.T @ self.Phi
        self.A_inv = np.linalg.inv(A + 1e-6 * np.eye(A.shape[0]))
        self.m = beta * self.A_inv @ self.Phi.T @ y_scaled
        self.beta = beta

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_scaled = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(device)

        self.feature_net.eval()
        with torch.no_grad():
            h = self.feature_net(X_t)
            Phi = torch.cos(h @ self.omega + self.b) * np.sqrt(2.0 / self.n_features)
            Phi = Phi.cpu().numpy()

        mean = Phi @ self.m
        var = np.array([phi @ self.A_inv @ phi for phi in Phi]) / self.beta
        std = np.sqrt(np.maximum(var, 1e-8))

        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
        std = std * self.scaler_y.scale_[0]

        return mean, std


# =============================================================================
# MODEL REGISTRY
# =============================================================================

SF_MODELS = {
    'GP': {'class': GPModel, 'needs_hp_tuning': True},
    'DNGO': {'class': DNGOModel, 'needs_hp_tuning': True},
    'BNN': {'class': BNNModel, 'needs_hp_tuning': True},
    'MC-Dropout': {'class': MCDropoutModel, 'needs_hp_tuning': True},
    'Deep Ensemble': {'class': DeepEnsembleModel, 'needs_hp_tuning': True},
    'SNGP': {'class': SNGPModel, 'needs_hp_tuning': True},
}

# Default HP (used when data is too small for tuning)
DEFAULT_HP = {
    'GP': {'length_scale': 1.0, 'noise_level': 0.01, 'n_restarts': 5},
    'DNGO': {'hidden_dim': 64, 'lr': 0.005, 'epochs': 200, 'blr_alpha': 0.1, 'blr_beta': 2.0},
    'BNN': {'hidden_dim': 64, 'num_layers': 2, 'lr': 0.01, 'epochs': 200, 'kl_weight': 0.1},
    'MC-Dropout': {'hidden_dim': 64, 'dropout_rate': 0.2, 'lr': 0.005, 'epochs': 200},
    'Deep Ensemble': {'hidden_dim': 64, 'n_ensemble': 5, 'lr': 0.005, 'epochs': 200},
    'SNGP': {'hidden_dim': 64, 'n_inducing': 20, 'lr': 0.005, 'epochs': 200},
}


def create_model(model_name: str, input_dim: int, hp: Dict = None):
    """Create model with given hyperparameters"""
    if hp is None:
        hp = DEFAULT_HP.get(model_name, {})

    model_class = SF_MODELS[model_name]['class']
    return model_class(input_dim=input_dim, **hp)


# =============================================================================
# ACQUISITION FUNCTIONS
# =============================================================================

def expected_improvement(X: np.ndarray, model, y_best: float, xi: float = 0.01) -> np.ndarray:
    """Expected Improvement acquisition function"""
    mean, std = model.predict(X)
    std = np.maximum(std, 1e-8)
    z = (y_best - mean - xi) / std
    ei = (y_best - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
    return np.maximum(ei, 0)


def optimize_acquisition(acq_func: Callable, bounds: np.ndarray,
                         n_restarts: int = 10, n_random: int = 1000) -> np.ndarray:
    """Optimize acquisition function"""
    dim = bounds.shape[0]
    X_random = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_random, dim))
    acq_values = acq_func(X_random)
    best_idx = np.argmax(acq_values)
    best_x = X_random[best_idx]
    best_val = acq_values[best_idx]

    top_idx = np.argsort(acq_values)[-n_restarts:]
    for idx in top_idx:
        try:
            result = minimize(
                lambda x: -acq_func(x.reshape(1, -1))[0],
                X_random[idx],
                method='L-BFGS-B',
                bounds=[(bounds[i, 0], bounds[i, 1]) for i in range(dim)]
            )
            if -result.fun > best_val:
                best_val = -result.fun
                best_x = result.x
        except:
            pass
    return best_x


# =============================================================================
# BO WITH ONLINE HP TUNING
# =============================================================================

def run_bo_with_online_hp(
    objective: Callable,
    model_name: str,
    bounds: np.ndarray,
    f_star: float,
    n_init: int = 5,
    n_iterations: int = 50,
    hp_tune_interval: int = 50,  # Tune HP every N iterations (50 = only at start)
    hp_tune_trials: int = 15,
    seed: int = 0,
    verbose: bool = False
) -> Dict:
    """
    Run BO with online HP tuning via LOOCV

    Args:
        hp_tune_interval: How often to tune HP (50 means only at start for 50-iter runs)
        hp_tune_trials: Number of Optuna trials for HP search
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    dim = bounds.shape[0]

    # Initial samples using LHS
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    X_init = sampler.random(n=n_init)
    X_init = qmc.scale(X_init, bounds[:, 0], bounds[:, 1])
    y_init = objective(X_init).flatten()

    X = X_init.copy()
    y = y_init.copy()

    # Track metrics
    regrets = [y.min() - f_star]
    best_values = [y.min()]

    # Initial HP tuning (if enough data)
    current_hp = DEFAULT_HP.get(model_name, {}).copy()
    if len(X) >= 5 and SF_MODELS[model_name]['needs_hp_tuning']:
        if verbose:
            print(f"    Initial HP tuning with {len(X)} samples...")
        try:
            current_hp = tune_hyperparameters(model_name, X, y, n_trials=hp_tune_trials)
            if verbose:
                print(f"    Tuned HP: {current_hp}")
        except Exception as e:
            if verbose:
                print(f"    HP tuning failed, using defaults: {e}")

    for i in range(n_iterations):
        try:
            # Periodic HP tuning
            if i > 0 and i % hp_tune_interval == 0 and SF_MODELS[model_name]['needs_hp_tuning']:
                if verbose:
                    print(f"    Re-tuning HP at iteration {i} with {len(X)} samples...")
                try:
                    current_hp = tune_hyperparameters(model_name, X, y, n_trials=hp_tune_trials)
                except:
                    pass  # Keep current HP if tuning fails

            # Create and fit model with current HP
            model = create_model(model_name, dim, current_hp)
            model.fit(X, y)

            # Optimize acquisition function
            y_best = y.min()
            x_next = optimize_acquisition(
                lambda x: expected_improvement(x, model, y_best),
                bounds,
                n_restarts=5,
                n_random=500
            )

            # Evaluate objective
            y_next = objective(x_next.reshape(1, -1)).flatten()[0]

            # Update data
            X = np.vstack([X, x_next])
            y = np.append(y, y_next)

            # Track metrics
            regrets.append(y.min() - f_star)
            best_values.append(y.min())

            if verbose and (i + 1) % 10 == 0:
                print(f"    Iter {i+1}/{n_iterations}: best={y.min():.4f}, regret={regrets[-1]:.4f}")

        except Exception as e:
            if verbose:
                print(f"    Iter {i+1}: Error - {e}")
            regrets.append(regrets[-1] if regrets else np.inf)
            best_values.append(best_values[-1] if best_values else np.inf)

    return {
        'regrets': np.array(regrets),
        'best_values': np.array(best_values),
        'X': X,
        'y': y,
        'f_star': f_star,
        'final_hp': current_hp
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_benchmark(n_seeds: int = 5, n_iterations: int = 50,
                  hp_tune_interval: int = 50, hp_tune_trials: int = 15,
                  verbose: bool = True):
    """Run benchmark with online HP tuning"""

    print("=" * 70)
    print("SF BO Benchmark with Online HP Tuning (LOOCV)")
    print("=" * 70)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Seeds: {n_seeds}")
    print(f"Iterations: {n_iterations}")
    print(f"HP Tune Interval: Every {hp_tune_interval} iterations")
    print(f"HP Tune Trials: {hp_tune_trials}")
    print(f"Models: {list(SF_MODELS.keys())}")
    print()

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results_online_hp_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Test functions
    test_functions = {
        'Branin-2D': {
            'func': branin_hf,
            'dim': 2,
            'bounds': np.array([[0, 1], [0, 1]]),
            'f_star': 0.397887,
        },
        'Park-4D': {
            'func': park_hf,
            'dim': 4,
            'bounds': np.array([[0, 1]] * 4),
            'f_star': None,
        }
    }

    # Compute Park-4D minimum
    print("Computing Park-4D global minimum...")
    _, park_min = find_global_minimum(park_hf, 4, n_random=20000, n_local=50)
    test_functions['Park-4D']['f_star'] = park_min
    print(f"  Park-4D f* = {park_min:.4f}")
    print()

    # Run experiments
    results = {}
    all_summaries = []

    for func_name, func_info in test_functions.items():
        print(f"\n{'=' * 70}")
        print(f"Test Function: {func_name} (dim={func_info['dim']}, f*={func_info['f_star']:.4f})")
        print(f"{'=' * 70}")

        results[func_name] = {}

        for model_name in SF_MODELS.keys():
            print(f"\n  Model: {model_name}")

            all_regrets = []
            all_best_values = []
            successful_seeds = 0

            for seed in range(n_seeds):
                print(f"    Seed {seed}/{n_seeds}...", end="", flush=True)

                try:
                    result = run_bo_with_online_hp(
                        objective=func_info['func'],
                        model_name=model_name,
                        bounds=func_info['bounds'],
                        f_star=func_info['f_star'],
                        n_init=5,
                        n_iterations=n_iterations,
                        hp_tune_interval=hp_tune_interval,
                        hp_tune_trials=hp_tune_trials,
                        seed=seed,
                        verbose=False
                    )
                    all_regrets.append(result['regrets'])
                    all_best_values.append(result['best_values'])
                    successful_seeds += 1

                    print(f" regret={result['regrets'][-1]:.4f}")

                except Exception as e:
                    print(f" FAILED: {e}")

            if all_regrets:
                regrets = np.array(all_regrets)
                best_values = np.array(all_best_values)

                results[func_name][model_name] = {
                    'regrets_mean': regrets.mean(axis=0),
                    'regrets_std': regrets.std(axis=0),
                    'regrets_median': np.median(regrets, axis=0),
                    'final_regret_mean': float(regrets[:, -1].mean()),
                    'final_regret_std': float(regrets[:, -1].std()),
                    'final_regret_median': float(np.median(regrets[:, -1])),
                    'n_successful': successful_seeds
                }

                all_summaries.append({
                    'Function': func_name,
                    'Model': model_name,
                    'Final Regret (Mean)': regrets[:, -1].mean(),
                    'Final Regret (Std)': regrets[:, -1].std(),
                    'Final Regret (Median)': np.median(regrets[:, -1]),
                    'Successful Seeds': successful_seeds
                })

                print(f"    Final: regret={regrets[:, -1].mean():.4f} ± {regrets[:, -1].std():.4f}")

    # Save results
    results_serializable = {}
    for func_name, func_results in results.items():
        results_serializable[func_name] = {}
        for model_name, model_results in func_results.items():
            results_serializable[func_name][model_name] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in model_results.items()
            }

    with open(f"{save_dir}/results.json", 'w') as f:
        json.dump(results_serializable, f, indent=2)

    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv(f"{save_dir}/summary.csv", index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY (with Online HP Tuning)")
    print("=" * 70)

    for func_name in test_functions:
        print(f"\n{func_name}:")
        print("-" * 60)
        print(f"{'Model':<18} {'Mean Regret':>15} {'Std':>12} {'Median':>12}")
        print("-" * 60)

        if func_name in results:
            sorted_models = sorted(
                results[func_name].items(),
                key=lambda x: x[1]['final_regret_mean']
            )
            for model_name, model_results in sorted_models:
                print(f"{model_name:<18} "
                      f"{model_results['final_regret_mean']:>15.6f} "
                      f"{model_results['final_regret_std']:>12.6f} "
                      f"{model_results['final_regret_median']:>12.6f}")

    print(f"\nResults saved to: {save_dir}")

    return results, save_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='BO with Online HP Tuning')
    parser.add_argument('--seeds', type=int, default=5, help='Number of random seeds')
    parser.add_argument('--iterations', type=int, default=50, help='Number of BO iterations')
    parser.add_argument('--hp-interval', type=int, default=50, help='HP tuning interval')
    parser.add_argument('--hp-trials', type=int, default=15, help='Optuna trials for HP tuning')
    args = parser.parse_args()

    results, save_dir = run_benchmark(
        n_seeds=args.seeds,
        n_iterations=args.iterations,
        hp_tune_interval=args.hp_interval,
        hp_tune_trials=args.hp_trials,
        verbose=True
    )
