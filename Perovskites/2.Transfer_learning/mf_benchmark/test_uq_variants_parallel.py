#!/usr/bin/env python
"""
UQ Variants Comparison Test - Parallel Version (benchmark_parallel.py style)

Compare 5 Sequential model variants:
1. Baseline - Standard Sequential (std=0.1 fixed)
2. +SpectralNorm - Spectral Normalization only
3. +BLR - Last-Layer Bayesian Linear Regression only
4. +LayerNorm - LayerNorm only
5. +All - SpectralNorm + BLR + LayerNorm

Output format matches benchmark_parallel.py:
- results_summary.csv: final metrics per (model, seed)
- results_trajectory.csv: budget vs regret per step

Usage:
    python test_uq_variants_parallel.py --n-seeds 10 --n-workers 3 --benchmark freesolv
    nohup python test_uq_variants_parallel.py --n-seeds 10 --n-workers 3 > uq_test.log 2>&1 &
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import norm
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("Warning: RDKit not available, chemistry benchmarks disabled")


# =============================================================================
# Synthetic Functions (Branin-2D)
# =============================================================================

def branin_hf(X: np.ndarray) -> np.ndarray:
    x1, x2 = X[:, 0], X[:, 1]
    x1_scaled = 15 * x1 - 5
    x2_scaled = 15 * x2
    a, b, c = 1, 5.1 / (4 * np.pi**2), 5 / np.pi
    r, s, t = 6, 10, 1 / (8 * np.pi)
    term1 = a * (x2_scaled - b * x1_scaled**2 + c * x1_scaled - r)**2
    term2 = s * (1 - t) * np.cos(x1_scaled)
    return (term1 + term2 + s).reshape(-1)

def branin_lf(X: np.ndarray, alpha: float = 0.8) -> np.ndarray:
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
# Chemistry Benchmark (FreeSolv)
# =============================================================================

def get_rdkit_descriptors(smiles_list: List[str], n_components: int = 10) -> Tuple[np.ndarray, List[int]]:
    if not HAS_RDKIT:
        raise RuntimeError("RDKit not available")

    desc_names = [desc[0] for desc in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    features = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            desc = calc.CalcDescriptors(mol)
            if not any(np.isnan(desc)) and not any(np.isinf(desc)):
                features.append(desc)
                valid_indices.append(i)

    features = np.array(features)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=min(n_components, features_scaled.shape[1]))
    features_pca = pca.fit_transform(features_scaled)

    return features_pca, valid_indices


def load_freesolv_benchmark(data_dir: Path) -> Dict:
    csv_path = data_dir / 'freesolv.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"FreeSolv data not found at {csv_path}")

    df = pd.read_csv(csv_path)
    smiles = df['smiles'].tolist()
    X, valid_idx = get_rdkit_descriptors(smiles, n_components=10)

    y_hf = df['HF'].values[valid_idx]
    y_lf = df['LF'].values[valid_idx]

    return {
        'X': X,
        'y_hf': y_hf,
        'y_lf': y_lf,
        'cost_ratio': 0.1,
        'f_star': y_hf.min(),
        'name': 'FreeSolv'
    }


# =============================================================================
# Network Architectures
# =============================================================================

class LFNetwork_Baseline(nn.Module):
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
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, use_spectral=False, use_layernorm=False):
        super().__init__()
        layers = []
        in_dim = input_dim + 1
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

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SequentialBaseline:
    name = "Baseline"

    def __init__(self, input_dim, hidden_dim=64, lf_epochs=200, hf_epochs=100):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lf_epochs = lf_epochs
        self.hf_epochs = hf_epochs
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.device = get_device()

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

        self.lf_net = LFNetwork_Baseline(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        for p in self.lf_net.parameters():
            p.requires_grad = False

        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
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
        X_t = torch.FloatTensor(X_s).to(self.device)

        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()

        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1


class SequentialSpectralNorm(SequentialBaseline):
    name = "+SpectralNorm"

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

        self.lf_net = LFNetwork_SpectralNorm(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        for p in self.lf_net.parameters():
            p.requires_grad = False

        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim, use_spectral=True).to(self.device)
        opt = torch.optim.Adam(self.hf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.hf_epochs):
            opt.zero_grad()
            with torch.no_grad():
                y_lf_pred = self.lf_net(X_hf_t)
            y_hf_pred = self.hf_net(X_hf_t, y_lf_pred)
            F.mse_loss(y_hf_pred, y_hf_t).backward()
            opt.step()


class SequentialBLR(SequentialBaseline):
    name = "+BLR"

    def __init__(self, input_dim, hidden_dim=64, lf_epochs=200, hf_epochs=100,
                 alpha_blr=1.0, beta_blr=25.0):
        super().__init__(input_dim, hidden_dim, lf_epochs, hf_epochs)
        self.alpha_blr = alpha_blr
        self.beta_blr = beta_blr

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]
        self.y_hf_s = y_hf_s

        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)

        self.lf_net = LFNetwork_Baseline(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        for p in self.lf_net.parameters():
            p.requires_grad = False

        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.hf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.hf_epochs):
            opt.zero_grad()
            with torch.no_grad():
                y_lf_pred = self.lf_net(X_hf_t)
            y_hf_pred = self.hf_net(X_hf_t, y_lf_pred)
            F.mse_loss(y_hf_pred, y_hf_t).backward()
            opt.step()

        # Fit BLR
        self.hf_net.eval()
        with torch.no_grad():
            y_lf_pred = self.lf_net(X_hf_t)
            Phi = self.hf_net.extract_features(X_hf_t, y_lf_pred).cpu().numpy()

        A = self.alpha_blr * np.eye(self.hidden_dim) + self.beta_blr * Phi.T @ Phi
        self.A_inv = np.linalg.inv(A + 1e-6 * np.eye(A.shape[0]))
        self.m = self.beta_blr * self.A_inv @ Phi.T @ y_hf_s

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)

        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            Phi = self.hf_net.extract_features(X_t, y_lf).cpu().numpy()

        mean_s = Phi @ self.m
        var_s = 1.0 / self.beta_blr + np.sum((Phi @ self.A_inv) * Phi, axis=1)
        std_s = np.sqrt(np.maximum(var_s, 1e-10))

        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        std = std_s * self.scaler_y.scale_[0]

        return mean, np.maximum(std, 1e-6)


class SequentialLayerNorm(SequentialBaseline):
    name = "+LayerNorm"

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

        self.lf_net = LFNetwork_LayerNorm(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        for p in self.lf_net.parameters():
            p.requires_grad = False

        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim, use_layernorm=True).to(self.device)
        opt = torch.optim.Adam(self.hf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.hf_epochs):
            opt.zero_grad()
            with torch.no_grad():
                y_lf_pred = self.lf_net(X_hf_t)
            y_hf_pred = self.hf_net(X_hf_t, y_lf_pred)
            F.mse_loss(y_hf_pred, y_hf_t).backward()
            opt.step()


class SequentialAll(SequentialBLR):
    name = "+All"

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]
        self.y_hf_s = y_hf_s

        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)

        self.lf_net = LFNetwork_All(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        for p in self.lf_net.parameters():
            p.requires_grad = False

        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim,
                                use_spectral=True, use_layernorm=True).to(self.device)
        opt = torch.optim.Adam(self.hf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.hf_epochs):
            opt.zero_grad()
            with torch.no_grad():
                y_lf_pred = self.lf_net(X_hf_t)
            y_hf_pred = self.hf_net(X_hf_t, y_lf_pred)
            F.mse_loss(y_hf_pred, y_hf_t).backward()
            opt.step()

        # Fit BLR
        self.hf_net.eval()
        with torch.no_grad():
            y_lf_pred = self.lf_net(X_hf_t)
            Phi = self.hf_net.extract_features(X_hf_t, y_lf_pred).cpu().numpy()

        A = self.alpha_blr * np.eye(self.hidden_dim) + self.beta_blr * Phi.T @ Phi
        self.A_inv = np.linalg.inv(A + 1e-6 * np.eye(A.shape[0]))
        self.m = self.beta_blr * self.A_inv @ Phi.T @ y_hf_s


# =============================================================================
# BO Utilities
# =============================================================================

def expected_improvement(mean: np.ndarray, std: np.ndarray, y_best: float,
                         xi: float = 0.01) -> np.ndarray:
    std = np.maximum(std, 1e-6)
    z = (y_best - mean - xi) / std
    ei = (y_best - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
    return np.maximum(ei, 0)


# =============================================================================
# Worker function for parallel execution
# =============================================================================

def run_single_combination(args):
    """Run all seeds for one (benchmark, model) combination"""
    model_class, model_name, benchmark_data, seeds, budget, output_dir = args

    bench_name = benchmark_data['name']
    X = benchmark_data['X']
    y_hf = benchmark_data['y_hf']
    y_lf = benchmark_data['y_lf']
    cost_ratio = benchmark_data['cost_ratio']
    f_star = benchmark_data['f_star']
    dim = X.shape[1]
    n_points = len(X)

    results_summary = []
    results_trajectory = []

    start_time = time.time()

    for seed in seeds:
        seed_start = time.time()

        try:
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Initial samples
            n_init_lf = min(20, n_points // 5)
            n_init_hf = min(5, n_points // 10)

            idx_all = np.random.permutation(n_points)
            idx_lf = idx_all[:n_init_lf]
            idx_hf = idx_lf[:n_init_hf]

            X_lf_curr = X[idx_lf].copy()
            y_lf_curr = y_lf[idx_lf].copy()
            X_hf_curr = X[idx_hf].copy()
            y_hf_curr = y_hf[idx_hf].copy()

            sampled_lf = set(idx_lf)
            sampled_hf = set(idx_hf)

            cost_hf, cost_lf = 1.0, cost_ratio
            spent = len(X_lf_curr) * cost_lf + len(X_hf_curr) * cost_hf

            lf_counter = 0
            budgets = [spent]
            regrets = [y_hf_curr.min() - f_star]

            while spent < budget:
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
                mean, std = model.predict(X)

                # Select next point
                y_best = y_hf_curr.min()

                if eval_hf:
                    mask = np.array([i not in sampled_hf for i in range(n_points)])
                else:
                    mask = np.array([i not in sampled_lf for i in range(n_points)])

                if not mask.any():
                    break

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
                if eval_hf:
                    X_hf_curr = np.vstack([X_hf_curr, X[next_idx:next_idx+1]])
                    y_hf_curr = np.concatenate([y_hf_curr, y_hf[next_idx:next_idx+1]])
                    sampled_hf.add(next_idx)
                else:
                    X_lf_curr = np.vstack([X_lf_curr, X[next_idx:next_idx+1]])
                    y_lf_curr = np.concatenate([y_lf_curr, y_lf[next_idx:next_idx+1]])
                    sampled_lf.add(next_idx)

                spent += cost_next
                budgets.append(spent)
                regrets.append(y_hf_curr.min() - f_star)

            seed_elapsed = time.time() - seed_start
            final_regret = y_hf_curr.min() - f_star

            results_summary.append({
                'benchmark': bench_name,
                'model': model_name,
                'seed': seed,
                'final_regret': final_regret,
                'n_hf': len(X_hf_curr),
                'n_lf': len(X_lf_curr),
                'best_y': y_hf_curr.min(),
                'elapsed_sec': round(seed_elapsed, 3),
            })

            for b, r in zip(budgets, regrets):
                results_trajectory.append({
                    'benchmark': bench_name,
                    'model': model_name,
                    'seed': seed,
                    'budget': round(b, 2),
                    'regret': r,
                })

            print(f"  {model_name} seed={seed}: regret={final_regret:.4f}, "
                  f"HF={len(X_hf_curr)}, LF={len(X_lf_curr)}, time={seed_elapsed:.1f}s")

        except Exception as e:
            seed_elapsed = time.time() - seed_start
            print(f"  {model_name} seed={seed}: ERROR - {e}")
            results_summary.append({
                'benchmark': bench_name,
                'model': model_name,
                'seed': seed,
                'final_regret': np.nan,
                'n_hf': 0,
                'n_lf': 0,
                'best_y': np.nan,
                'elapsed_sec': round(seed_elapsed, 3),
            })

    elapsed = time.time() - start_time

    # Save individual results
    df_summary = pd.DataFrame(results_summary)
    summary_file = output_dir / f'summary_{bench_name}_{model_name.replace(" ", "_").replace("+", "plus")}.csv'
    df_summary.to_csv(summary_file, index=False)

    df_trajectory = pd.DataFrame(results_trajectory)
    trajectory_file = output_dir / f'trajectory_{bench_name}_{model_name.replace(" ", "_").replace("+", "plus")}.csv'
    df_trajectory.to_csv(trajectory_file, index=False)

    return {
        'bench_name': bench_name,
        'model_name': model_name,
        'n_seeds': len(seeds),
        'elapsed': elapsed,
        'results_summary': results_summary,
        'results_trajectory': results_trajectory
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='UQ Variants Parallel Test')
    parser.add_argument('--n-seeds', type=int, default=10, help='Number of seeds')
    parser.add_argument('--base-seed', type=int, default=42, help='Base seed')
    parser.add_argument('--n-workers', type=int, default=3, help='Number of parallel workers')
    parser.add_argument('--benchmark', type=str, default='branin',
                        choices=['branin', 'freesolv'], help='Benchmark to use')
    parser.add_argument('--budget', type=float, default=30, help='Total budget')
    args = parser.parse_args()

    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'uq_variants_{args.benchmark}_{timestamp}')
    output_dir.mkdir(exist_ok=True)

    # Data directory
    data_dir = Path(__file__).parent / 'data'

    print("=" * 80)
    print("UQ Variants Parallel Test")
    print("=" * 80)
    print(f"Device: {get_device()}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Seeds: {args.n_seeds} (base={args.base_seed})")
    print(f"Workers: {args.n_workers}")
    print(f"Budget: {args.budget}")
    print(f"Output: {output_dir}")
    print()

    # Prepare benchmark data
    if args.benchmark == 'branin':
        np.random.seed(42)
        n_grid = 50
        x1 = np.linspace(0, 1, n_grid)
        x2 = np.linspace(0, 1, n_grid)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.column_stack([X1.ravel(), X2.ravel()])

        benchmark_data = {
            'X': X,
            'y_hf': branin_hf(X),
            'y_lf': branin_lf(X, alpha=0.8),
            'cost_ratio': 0.1,
            'f_star': 0.397887,
            'name': 'Branin-2D'
        }
    else:  # freesolv
        benchmark_data = load_freesolv_benchmark(data_dir)

    print(f"Data points: {len(benchmark_data['X'])}")
    print(f"Input dim: {benchmark_data['X'].shape[1]}")
    print(f"Cost ratio: {benchmark_data['cost_ratio']}")
    print()

    # Models to compare
    model_classes = [
        (SequentialBaseline, "Baseline"),
        (SequentialSpectralNorm, "+SpectralNorm"),
        (SequentialBLR, "+BLR"),
        (SequentialLayerNorm, "+LayerNorm"),
        (SequentialAll, "+All"),
    ]

    seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))

    # Prepare tasks
    tasks = []
    for model_class, model_name in model_classes:
        tasks.append((model_class, model_name, benchmark_data, seeds, args.budget, output_dir))

    print(f"Models: {len(model_classes)}")
    print(f"Total seeds per model: {len(seeds)}")
    print()
    print("-" * 80)

    # Run in parallel
    start_time = time.time()

    if args.n_workers > 1:
        with Pool(args.n_workers) as pool:
            all_results = pool.map(run_single_combination, tasks)
    else:
        all_results = [run_single_combination(t) for t in tasks]

    total_elapsed = time.time() - start_time

    print("-" * 80)
    print()

    # Aggregate all results
    all_summary = []
    all_trajectory = []
    for r in all_results:
        all_summary.extend(r['results_summary'])
        all_trajectory.extend(r['results_trajectory'])

    # Save combined results
    df_summary = pd.DataFrame(all_summary)
    df_summary.to_csv(output_dir / 'results_summary.csv', index=False)

    df_trajectory = pd.DataFrame(all_trajectory)
    df_trajectory.to_csv(output_dir / 'results_trajectory.csv', index=False)

    # Print summary
    print("=" * 80)
    print("Summary (mean ± std)")
    print("=" * 80)
    print(f"{'Model':<20} {'Regret':>20} {'Min':>12} {'Max':>12}")
    print("-" * 70)

    for model_class, model_name in model_classes:
        model_data = df_summary[df_summary['model'] == model_name]['final_regret'].dropna()
        if len(model_data) > 0:
            mean_r = model_data.mean()
            std_r = model_data.std()
            min_r = model_data.min()
            max_r = model_data.max()
            print(f"{model_name:<20} {mean_r:>8.4f} ± {std_r:<8.4f} {min_r:>12.4f} {max_r:>12.4f}")
        else:
            print(f"{model_name:<20} {'N/A':>20} {'N/A':>12} {'N/A':>12}")

    # Find best
    mean_regrets = df_summary.groupby('model')['final_regret'].mean()
    best_model = mean_regrets.idxmin()
    print()
    print(f"Best: {best_model} (mean regret = {mean_regrets[best_model]:.4f})")

    # Plot convergence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Bar plot of final regret
    ax = axes[0]
    model_names = [m[1] for m in model_classes]
    means = [df_summary[df_summary['model'] == m]['final_regret'].mean() for m in model_names]
    stds = [df_summary[df_summary['model'] == m]['final_regret'].std() for m in model_names]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax.bar(model_names, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_ylabel('Simple Regret')
    ax.set_title(f'Final Regret ({args.n_seeds} seeds)')
    ax.grid(True, alpha=0.3, axis='y')

    # Right: Convergence curves
    ax = axes[1]
    for (model_class, model_name), color in zip(model_classes, colors):
        model_traj = df_trajectory[df_trajectory['model'] == model_name]
        if len(model_traj) > 0:
            # Average across seeds
            avg_traj = model_traj.groupby('budget')['regret'].agg(['mean', 'std']).reset_index()
            ax.plot(avg_traj['budget'], avg_traj['mean'], '-', color=color, label=model_name, linewidth=2)
            ax.fill_between(avg_traj['budget'],
                           avg_traj['mean'] - avg_traj['std'],
                           avg_traj['mean'] + avg_traj['std'],
                           color=color, alpha=0.2)

    ax.set_xlabel('Budget')
    ax.set_ylabel('Simple Regret')
    ax.set_title('Convergence (mean ± std)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'UQ Variants on {benchmark_data["name"]} (budget={args.budget})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=150, bbox_inches='tight')

    print()
    print(f"Total time: {total_elapsed:.1f}s")
    print()
    print(f"Results saved to: {output_dir}")
    print(f"  - results_summary.csv: {len(df_summary)} rows (final metrics)")
    print(f"  - results_trajectory.csv: {len(df_trajectory)} rows (budget vs regret)")
    print(f"  - comparison.png: visualization")


if __name__ == '__main__':
    # CUDA multiprocessing fix
    mp.set_start_method('spawn', force=True)
    main()
