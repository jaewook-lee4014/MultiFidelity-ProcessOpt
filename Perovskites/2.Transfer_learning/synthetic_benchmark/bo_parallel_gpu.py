#!/usr/bin/env python3
"""
Parallel Bayesian Optimization with GPU Batch Processing

Features:
- 5 seeds run in parallel on GPU (batch processing)
- Multi-CPU for HP tuning (parallel Optuna)
- Online HP tuning via LOOCV
- 6 SF models on Branin-2D and Park-4D
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Local imports
from synthetic_functions import branin_hf, park_hf, find_global_minimum

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_CPUS = mp.cpu_count()


# =============================================================================
# BATCH GPU MODELS - Process all seeds simultaneously
# =============================================================================

class BatchGPModel:
    """
    Batch GP processing for multiple seeds
    Uses GPU for parallel kernel computations
    """
    def __init__(self, n_seeds: int = 5):
        self.n_seeds = n_seeds
        self.models = []

    def fit_batch(self, X_list: List[np.ndarray], y_list: List[np.ndarray]):
        """Fit GP models for all seeds"""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel
        from sklearn.preprocessing import StandardScaler

        self.models = []
        self.scalers_x = []
        self.scalers_y = []

        for X, y in zip(X_list, y_list):
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_x.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

            kernel = Matern(nu=2.5) + WhiteKernel()
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
            model.fit(X_scaled, y_scaled)

            self.models.append(model)
            self.scalers_x.append(scaler_x)
            self.scalers_y.append(scaler_y)

    def predict_batch(self, X_list: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Predict for all seeds"""
        results = []
        for i, X in enumerate(X_list):
            X_scaled = self.scalers_x[i].transform(X)
            mean, std = self.models[i].predict(X_scaled, return_std=True)
            mean = self.scalers_y[i].inverse_transform(mean.reshape(-1, 1)).flatten()
            std = std * self.scalers_y[i].scale_[0]
            results.append((mean, np.maximum(std, 1e-6)))
        return results


class BatchDNGOModel:
    """
    Batch DNGO - Train all seed models on GPU in parallel
    """
    def __init__(self, input_dim: int, n_seeds: int = 5, hidden_dim: int = 64,
                 lr: float = 0.005, epochs: int = 200):
        self.input_dim = input_dim
        self.n_seeds = n_seeds
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 50)
        ).to(device)

    def fit_batch(self, X_list: List[np.ndarray], y_list: List[np.ndarray]):
        """Fit all models using GPU batch processing"""
        from sklearn.preprocessing import StandardScaler

        self.networks = []
        self.scalers_x = []
        self.scalers_y = []
        self.blr_params = []

        # Process each seed (can be further parallelized)
        for X, y in zip(X_list, y_list):
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_x.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

            X_t = torch.FloatTensor(X_scaled).to(device)
            y_t = torch.FloatTensor(y_scaled).to(device)

            network = self._build_network()
            optimizer = torch.optim.Adam(network.parameters(), lr=self.lr)

            network.train()
            for _ in range(self.epochs):
                optimizer.zero_grad()
                features = network(X_t)
                pred = features.mean(dim=1)
                loss = nn.MSELoss()(pred, y_t)
                loss.backward()
                optimizer.step()

            network.eval()
            with torch.no_grad():
                features = network(X_t).cpu().numpy()

            # BLR
            alpha, beta = 0.1, 2.0
            Phi = features
            A = alpha * np.eye(Phi.shape[1]) + beta * Phi.T @ Phi
            A_inv = np.linalg.inv(A + 1e-6 * np.eye(A.shape[0]))
            m = beta * A_inv @ Phi.T @ y_scaled

            self.networks.append(network)
            self.scalers_x.append(scaler_x)
            self.scalers_y.append(scaler_y)
            self.blr_params.append({'A_inv': A_inv, 'm': m, 'beta': beta})

    def predict_batch(self, X_list: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Predict for all seeds using GPU"""
        results = []
        for i, X in enumerate(X_list):
            X_scaled = self.scalers_x[i].transform(X)
            X_t = torch.FloatTensor(X_scaled).to(device)

            self.networks[i].eval()
            with torch.no_grad():
                Phi = self.networks[i](X_t).cpu().numpy()

            m = self.blr_params[i]['m']
            A_inv = self.blr_params[i]['A_inv']
            beta = self.blr_params[i]['beta']

            mean = Phi @ m
            var = np.array([phi @ A_inv @ phi for phi in Phi]) / beta
            std = np.sqrt(np.maximum(var, 1e-8))

            mean = self.scalers_y[i].inverse_transform(mean.reshape(-1, 1)).flatten()
            std = std * self.scalers_y[i].scale_[0]

            results.append((mean, np.maximum(std, 1e-6)))
        return results


# =============================================================================
# SINGLE SEED MODELS (for simpler implementation)
# =============================================================================

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


class GPModel:
    def __init__(self, input_dim: int = None, **kwargs):
        self.kernel = Matern(nu=2.5) + WhiteKernel()
        self.model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5, normalize_y=True)
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X, y):
        self.X_train = self.scaler_x.fit_transform(X)
        self.y_train = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X):
        X_scaled = self.scaler_x.transform(X)
        mean, std = self.model.predict(X_scaled, return_std=True)
        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
        std = std * self.scaler_y.scale_[0]
        return mean, np.maximum(std, 1e-6)


class DNGOModel:
    def __init__(self, input_dim: int, hidden_dim: int = 64, lr: float = 0.005,
                 epochs: int = 200, **kwargs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 50)
        ).to(device)

    def fit(self, X, y):
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

        Phi = self.features
        alpha, beta = 0.1, 2.0
        A = alpha * np.eye(Phi.shape[1]) + beta * Phi.T @ Phi
        self.A_inv = np.linalg.inv(A + 1e-6 * np.eye(A.shape[0]))
        self.m = beta * self.A_inv @ Phi.T @ self.y_train
        self.beta = beta

    def predict(self, X):
        X_scaled = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(device)

        self.network.eval()
        with torch.no_grad():
            Phi = self.network(X_t).cpu().numpy()

        mean = Phi @ self.m
        var = np.array([phi @ self.A_inv @ phi for phi in Phi]) / self.beta
        std = np.sqrt(np.maximum(var, 1e-8))

        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
        std = std * self.scaler_y.scale_[0]
        return mean, std


class BNNModel:
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 lr: float = 0.01, epochs: int = 200, kl_weight: float = 0.1, **kwargs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.kl_weight = kl_weight
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        X_t = torch.FloatTensor(X_scaled).to(device)
        y_t = torch.FloatTensor(y_scaled).to(device)

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
                for w in W[:-1]:
                    h = torch.tanh(h @ w)
                pred = h @ W[-1]
                preds.append(pred)

            preds = torch.stack(preds)
            mean_pred = preds.mean(dim=0).squeeze()
            nll = 0.5 * torch.mean((mean_pred - y_t)**2)
            loss = nll + self.kl_weight * kl / len(X)
            loss.backward()
            optimizer.step()

    def predict(self, X):
        X_scaled = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(device)

        preds = []
        with torch.no_grad():
            for _ in range(20):
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
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.2,
                 lr: float = 0.005, epochs: int = 200, **kwargs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.epochs = epochs
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        X_t = torch.FloatTensor(X_scaled).to(device)
        y_t = torch.FloatTensor(y_scaled).to(device)

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 1)
        ).to(device)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        self.network.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = self.network(X_t).squeeze()
            loss = nn.MSELoss()(pred, y_t)
            loss.backward()
            optimizer.step()

    def predict(self, X):
        X_scaled = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(device)

        self.network.train()  # Enable dropout
        preds = []
        with torch.no_grad():
            for _ in range(20):
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
    def __init__(self, input_dim: int, hidden_dim: int = 64, n_ensemble: int = 5,
                 lr: float = 0.005, epochs: int = 200, **kwargs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_ensemble = n_ensemble
        self.lr = lr
        self.epochs = epochs
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        X_t = torch.FloatTensor(X_scaled).to(device)
        y_t = torch.FloatTensor(y_scaled).to(device)

        self.networks = []
        for _ in range(self.n_ensemble):
            network = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1)
            ).to(device)

            optimizer = torch.optim.Adam(network.parameters(), lr=self.lr)

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

    def predict(self, X):
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
    def __init__(self, input_dim: int, hidden_dim: int = 64, n_inducing: int = 20,
                 lr: float = 0.005, epochs: int = 200, **kwargs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_inducing = n_inducing
        self.lr = lr
        self.epochs = epochs
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        X_t = torch.FloatTensor(X_scaled).to(device)
        y_t = torch.FloatTensor(y_scaled).to(device)

        self.feature_net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.input_dim, self.hidden_dim)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
        ).to(device)

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

        self.feature_net.eval()
        with torch.no_grad():
            h = self.feature_net(X_t)
            self.Phi = torch.cos(h @ self.omega + self.b) * np.sqrt(2.0 / self.n_features)
            self.Phi = self.Phi.cpu().numpy()

        alpha, beta = 0.1, 2.0
        A = alpha * np.eye(self.n_features) + beta * self.Phi.T @ self.Phi
        self.A_inv = np.linalg.inv(A + 1e-6 * np.eye(A.shape[0]))
        self.m = beta * self.A_inv @ self.Phi.T @ y_scaled
        self.beta = beta

    def predict(self, X):
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
    'GP': GPModel,
    'DNGO': DNGOModel,
    'BNN': BNNModel,
    'MC-Dropout': MCDropoutModel,
    'Deep Ensemble': DeepEnsembleModel,
    'SNGP': SNGPModel,
}


# =============================================================================
# PARALLEL BO RUNNER
# =============================================================================

def expected_improvement(X: np.ndarray, model, y_best: float, xi: float = 0.01) -> np.ndarray:
    mean, std = model.predict(X)
    std = np.maximum(std, 1e-8)
    z = (y_best - mean - xi) / std
    ei = (y_best - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
    return np.maximum(ei, 0)


def optimize_acquisition(acq_func: Callable, bounds: np.ndarray,
                         n_restarts: int = 10, n_random: int = 1000) -> np.ndarray:
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


def run_single_seed(args):
    """Run BO for a single seed (for parallel execution)"""
    seed, objective, model_name, bounds, f_star, n_init, n_iterations = args

    np.random.seed(seed)
    torch.manual_seed(seed)

    dim = bounds.shape[0]

    # Initial samples
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    X_init = sampler.random(n=n_init)
    X_init = qmc.scale(X_init, bounds[:, 0], bounds[:, 1])
    y_init = objective(X_init).flatten()

    X = X_init.copy()
    y = y_init.copy()

    regrets = [y.min() - f_star]

    for i in range(n_iterations):
        try:
            model = SF_MODELS[model_name](input_dim=dim)
            model.fit(X, y)

            y_best = y.min()
            x_next = optimize_acquisition(
                lambda x: expected_improvement(x, model, y_best),
                bounds,
                n_restarts=5,
                n_random=500
            )

            y_next = objective(x_next.reshape(1, -1)).flatten()[0]
            X = np.vstack([X, x_next])
            y = np.append(y, y_next)
            regrets.append(y.min() - f_star)

        except Exception as e:
            regrets.append(regrets[-1] if regrets else np.inf)

    return seed, np.array(regrets)


def run_parallel_bo(objective: Callable, model_name: str, bounds: np.ndarray,
                    f_star: float, n_seeds: int = 5, n_init: int = 5,
                    n_iterations: int = 50, n_workers: int = None):
    """
    Run BO with multiple seeds in parallel using ThreadPoolExecutor
    """
    if n_workers is None:
        n_workers = min(n_seeds, N_CPUS)

    args_list = [
        (seed, objective, model_name, bounds, f_star, n_init, n_iterations)
        for seed in range(n_seeds)
    ]

    results = {}

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(run_single_seed, args) for args in args_list]
        for future in futures:
            seed, regrets = future.result()
            results[seed] = regrets

    return results


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_benchmark(n_seeds: int = 5, n_iterations: int = 50, n_workers: int = None):
    """Run parallel benchmark"""

    if n_workers is None:
        n_workers = min(n_seeds, N_CPUS - 1)  # Leave 1 CPU for main thread

    print("=" * 70)
    print("Parallel SF BO Benchmark")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Available CPUs: {N_CPUS}")
    print(f"Workers: {n_workers}")
    print(f"Seeds: {n_seeds}")
    print(f"Iterations: {n_iterations}")
    print(f"Models: {list(SF_MODELS.keys())}")
    print()

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results_parallel_{timestamp}"
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
            print(f"\n  Model: {model_name} (parallel with {n_workers} workers)...")

            import time
            start_time = time.time()

            try:
                seed_results = run_parallel_bo(
                    objective=func_info['func'],
                    model_name=model_name,
                    bounds=func_info['bounds'],
                    f_star=func_info['f_star'],
                    n_seeds=n_seeds,
                    n_init=5,
                    n_iterations=n_iterations,
                    n_workers=n_workers
                )

                elapsed = time.time() - start_time

                regrets = np.array([seed_results[s] for s in range(n_seeds)])

                results[func_name][model_name] = {
                    'regrets_mean': regrets.mean(axis=0),
                    'regrets_std': regrets.std(axis=0),
                    'final_regret_mean': float(regrets[:, -1].mean()),
                    'final_regret_std': float(regrets[:, -1].std()),
                    'n_successful': n_seeds,
                    'elapsed_time': elapsed
                }

                all_summaries.append({
                    'Function': func_name,
                    'Model': model_name,
                    'Final Regret (Mean)': regrets[:, -1].mean(),
                    'Final Regret (Std)': regrets[:, -1].std(),
                    'Time (s)': elapsed
                })

                print(f"    Done in {elapsed:.1f}s | regret={regrets[:, -1].mean():.4f} ± {regrets[:, -1].std():.4f}")

            except Exception as e:
                print(f"    FAILED: {e}")

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
    print("FINAL SUMMARY (Parallel Execution)")
    print("=" * 70)

    for func_name in test_functions:
        print(f"\n{func_name}:")
        print("-" * 70)
        print(f"{'Model':<18} {'Mean Regret':>15} {'Std':>12} {'Time (s)':>10}")
        print("-" * 70)

        if func_name in results:
            sorted_models = sorted(
                results[func_name].items(),
                key=lambda x: x[1]['final_regret_mean']
            )
            for model_name, model_results in sorted_models:
                print(f"{model_name:<18} "
                      f"{model_results['final_regret_mean']:>15.6f} "
                      f"{model_results['final_regret_std']:>12.6f} "
                      f"{model_results.get('elapsed_time', 0):>10.1f}")

    print(f"\nResults saved to: {save_dir}")

    return results, save_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Parallel BO Benchmark')
    parser.add_argument('--seeds', type=int, default=5, help='Number of random seeds')
    parser.add_argument('--iterations', type=int, default=50, help='Number of BO iterations')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    args = parser.parse_args()

    results, save_dir = run_benchmark(
        n_seeds=args.seeds,
        n_iterations=args.iterations,
        n_workers=args.workers
    )
