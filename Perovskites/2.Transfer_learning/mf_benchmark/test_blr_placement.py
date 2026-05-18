#!/usr/bin/env python
"""
BLR Placement Comparison: HF-BLR vs LF-BLR vs Dual-BLR

Compare three BLR strategies:
1. HF-BLR: BLR on HF network, EI for both LF and HF selection (current approach)
2. LF-BLR: BLR on LF network for LF selection (EI), HF uses argmin (original design)
3. Dual-BLR: BLR on both LF and HF networks, each uses its own EI

Usage:
    # Single benchmark
    python test_blr_placement.py --n-seeds 10 --n-workers 3 --benchmarks freesolv

    # Multiple benchmarks (comma-separated)
    python test_blr_placement.py --n-seeds 10 --n-workers 3 --benchmarks park,cofs,polarizability

    # All benchmarks
    python test_blr_placement.py --n-seeds 10 --n-workers 3 --benchmarks all
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# Synthetic Functions (Branin-2D, Park-4D)
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


def park_hf(X: np.ndarray) -> np.ndarray:
    """Park function (4D) - High Fidelity"""
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    term1 = x1 / 2 * (np.sqrt(1 + (x2 + x3**2) * x4 / x1**2) - 1)
    term2 = (x1 + 3 * x4) * np.exp(1 + np.sin(x3))
    return (term1 + term2).reshape(-1)

def park_lf(X: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Park function (4D) - Low Fidelity"""
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    term1 = (1 + np.sin(x1) / 10) * park_hf(X) - 2 * x1 + x2**2 + x3**2 + 0.5
    # Blend with HF based on alpha
    hf = park_hf(X)
    return (alpha * hf + (1 - alpha) * term1).reshape(-1)


# =============================================================================
# Chemistry Benchmarks (FreeSolv, COFs, Polarizability)
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


def load_cofs_benchmark(data_dir: Path) -> Dict:
    """Load COFs benchmark (no SMILES, use feature columns directly)"""
    csv_path = data_dir / 'cofs.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"COFs data not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # COFs uses feature columns, not SMILES
    feature_cols = [c for c in df.columns if c not in ['HF', 'LF']]
    X = df[feature_cols].values

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Negate for maximization -> minimization
    y_hf = -df['HF'].values
    y_lf = -df['LF'].values

    return {
        'X': X,
        'y_hf': y_hf,
        'y_lf': y_lf,
        'cost_ratio': 0.065,
        'f_star': y_hf.min(),
        'name': 'COFs'
    }


def load_polarizability_benchmark(data_dir: Path) -> Dict:
    """Load Polarizability benchmark (uses SMILES)"""
    csv_path = data_dir / 'polarizability.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Polarizability data not found at {csv_path}")

    df = pd.read_csv(csv_path)
    smiles = df['smiles'].tolist()
    X, valid_idx = get_rdkit_descriptors(smiles, n_components=10)

    # Negate for maximization -> minimization
    y_hf = -df['HF'].values[valid_idx]
    y_lf = -df['LF'].values[valid_idx]

    return {
        'X': X,
        'y_hf': y_hf,
        'y_lf': y_lf,
        'cost_ratio': 0.167,
        'f_star': y_hf.min(),
        'name': 'Polarizability'
    }


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


# =============================================================================
# Model Classes
# =============================================================================

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SequentialBaseline:
    """Baseline: No BLR, fixed std=0.1"""
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

        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
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

    def predict_lf(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)

        self.lf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            mean_s = y_lf.cpu().numpy().flatten()

        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1

    def predict_hf(self, X) -> Tuple[np.ndarray, np.ndarray]:
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


class SequentialHFBLR(SequentialBaseline):
    """HF-BLR: BLR on HF network, used for both LF and HF selection"""
    name = "HF-BLR"

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

        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
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

        # Fit BLR on HF features
        self.hf_net.eval()
        with torch.no_grad():
            y_lf_pred = self.lf_net(X_hf_t)
            Phi = self.hf_net.extract_features(X_hf_t, y_lf_pred).cpu().numpy()

        A = self.alpha_blr * np.eye(self.hidden_dim) + self.beta_blr * Phi.T @ Phi
        self.A_inv = np.linalg.inv(A + 1e-6 * np.eye(A.shape[0]))
        self.m = self.beta_blr * self.A_inv @ Phi.T @ y_hf_s

    def predict_hf(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """HF prediction with BLR uncertainty"""
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


class SequentialLFBLR(SequentialBaseline):
    """LF-BLR: BLR on LF network for LF selection, HF uses argmin (original design)"""
    name = "LF-BLR"

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
        self.y_lf_s = y_lf_s

        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)

        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        # Fit BLR on LF features (using LF data - more abundant)
        self.lf_net.eval()
        with torch.no_grad():
            Phi_lf = self.lf_net.extract_features(X_lf_t).cpu().numpy()

        A = self.alpha_blr * np.eye(self.hidden_dim) + self.beta_blr * Phi_lf.T @ Phi_lf
        self.A_inv_lf = np.linalg.inv(A + 1e-6 * np.eye(A.shape[0]))
        self.m_lf = self.beta_blr * self.A_inv_lf @ Phi_lf.T @ y_lf_s

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

    def predict_lf(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """LF prediction with BLR uncertainty"""
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)

        self.lf_net.eval()
        with torch.no_grad():
            Phi = self.lf_net.extract_features(X_t).cpu().numpy()

        mean_s = Phi @ self.m_lf
        var_s = 1.0 / self.beta_blr + np.sum((Phi @ self.A_inv_lf) * Phi, axis=1)
        std_s = np.sqrt(np.maximum(var_s, 1e-10))

        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        std = std_s * self.scaler_y.scale_[0]

        return mean, np.maximum(std, 1e-6)

    def predict_hf(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """HF prediction (no BLR, just mean)"""
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)

        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()

        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1  # No BLR on HF


class SequentialDualBLR(SequentialBaseline):
    """Dual-BLR: BLR on both LF and HF networks"""
    name = "Dual-BLR"

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
        self.y_lf_s = y_lf_s
        self.y_hf_s = y_hf_s

        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)

        # Train LF network
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        # Fit BLR on LF features
        self.lf_net.eval()
        with torch.no_grad():
            Phi_lf = self.lf_net.extract_features(X_lf_t).cpu().numpy()

        A_lf = self.alpha_blr * np.eye(self.hidden_dim) + self.beta_blr * Phi_lf.T @ Phi_lf
        self.A_inv_lf = np.linalg.inv(A_lf + 1e-6 * np.eye(A_lf.shape[0]))
        self.m_lf = self.beta_blr * self.A_inv_lf @ Phi_lf.T @ y_lf_s

        for p in self.lf_net.parameters():
            p.requires_grad = False

        # Train HF network
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
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
            Phi_hf = self.hf_net.extract_features(X_hf_t, y_lf_pred).cpu().numpy()

        A_hf = self.alpha_blr * np.eye(self.hidden_dim) + self.beta_blr * Phi_hf.T @ Phi_hf
        self.A_inv_hf = np.linalg.inv(A_hf + 1e-6 * np.eye(A_hf.shape[0]))
        self.m_hf = self.beta_blr * self.A_inv_hf @ Phi_hf.T @ y_hf_s

    def predict_lf(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """LF prediction with BLR uncertainty"""
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)

        self.lf_net.eval()
        with torch.no_grad():
            Phi = self.lf_net.extract_features(X_t).cpu().numpy()

        mean_s = Phi @ self.m_lf
        var_s = 1.0 / self.beta_blr + np.sum((Phi @ self.A_inv_lf) * Phi, axis=1)
        std_s = np.sqrt(np.maximum(var_s, 1e-10))

        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        std = std_s * self.scaler_y.scale_[0]

        return mean, np.maximum(std, 1e-6)

    def predict_hf(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """HF prediction with BLR uncertainty"""
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)

        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            Phi = self.hf_net.extract_features(X_t, y_lf).cpu().numpy()

        mean_s = Phi @ self.m_hf
        var_s = 1.0 / self.beta_blr + np.sum((Phi @ self.A_inv_hf) * Phi, axis=1)
        std_s = np.sqrt(np.maximum(var_s, 1e-10))

        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        std = std_s * self.scaler_y.scale_[0]

        return mean, np.maximum(std, 1e-6)


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

                # Select next point based on model type
                if eval_hf:
                    # HF selection: Use HF prediction
                    mask = np.array([i not in sampled_hf for i in range(n_points)])
                    mean_hf, std_hf = model.predict_hf(X)

                    if model_name in ['HF-BLR', 'Dual-BLR']:
                        # HF-BLR and Dual-BLR use EI for HF selection
                        y_best = y_hf_curr.min()
                        ei = expected_improvement(mean_hf, std_hf, y_best)
                        ei[~mask] = -np.inf
                        next_idx = np.argmax(ei)
                    else:
                        # LF-BLR uses argmin for HF selection (original design)
                        mean_masked = mean_hf.copy()
                        mean_masked[~mask] = np.inf
                        next_idx = np.argmin(mean_masked)
                else:
                    # LF selection: Use LF prediction
                    mask = np.array([i not in sampled_lf for i in range(n_points)])

                    if model_name in ['LF-BLR', 'Dual-BLR']:
                        # LF-BLR and Dual-BLR use LF prediction + BLR for LF selection
                        mean_lf, std_lf = model.predict_lf(X)
                        y_best_lf = y_lf_curr.min()
                        ei = expected_improvement(mean_lf, std_lf, y_best_lf)
                        ei[~mask] = -np.inf
                        next_idx = np.argmax(ei)
                    else:
                        # HF-BLR uses HF prediction for LF selection too
                        mean_hf, std_hf = model.predict_hf(X)
                        y_best = y_hf_curr.min()
                        ei = expected_improvement(mean_hf, std_hf, y_best)
                        ei[~mask] = -np.inf
                        next_idx = np.argmax(ei)

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
            import traceback
            traceback.print_exc()
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
    summary_file = output_dir / f'summary_{bench_name}_{model_name.replace(" ", "_").replace("-", "_")}.csv'
    df_summary.to_csv(summary_file, index=False)

    df_trajectory = pd.DataFrame(results_trajectory)
    trajectory_file = output_dir / f'trajectory_{bench_name}_{model_name.replace(" ", "_").replace("-", "_")}.csv'
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
    parser = argparse.ArgumentParser(description='BLR Placement Comparison')
    parser.add_argument('--n-seeds', type=int, default=10, help='Number of seeds')
    parser.add_argument('--base-seed', type=int, default=42, help='Base seed')
    parser.add_argument('--n-workers', type=int, default=3, help='Number of parallel workers')
    parser.add_argument('--benchmarks', type=str, default='park,cofs,polarizability',
                        help='Benchmarks to use (comma-separated or "all")')
    args = parser.parse_args()

    # Default budgets per benchmark (from benchmark_parallel.py)
    BUDGETS = {
        'branin': 50,
        'park': 50,
        'freesolv': 50,
        'cofs': 30,
        'polarizability': 30,
    }

    ALL_BENCHMARKS = ['branin', 'park', 'freesolv', 'cofs', 'polarizability']

    # Parse benchmarks
    if args.benchmarks.lower() == 'all':
        benchmark_list = ALL_BENCHMARKS
    else:
        benchmark_list = [b.strip() for b in args.benchmarks.split(',')]
        for b in benchmark_list:
            if b not in ALL_BENCHMARKS:
                raise ValueError(f"Unknown benchmark: {b}. Choose from {ALL_BENCHMARKS}")

    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    benchmarks_str = '_'.join(benchmark_list[:3])  # Use first 3 for dir name
    if len(benchmark_list) > 3:
        benchmarks_str += f'_etc{len(benchmark_list)}'
    output_dir = Path(f'blr_placement_{benchmarks_str}_{timestamp}')
    output_dir.mkdir(exist_ok=True)

    # Data directory
    data_dir = Path(__file__).parent / 'data'

    print("=" * 80)
    print("BLR Placement Comparison: HF-BLR vs LF-BLR vs Dual-BLR")
    print("=" * 80)
    print(f"Device: {get_device()}")
    print(f"Benchmarks: {', '.join(benchmark_list)}")
    print(f"Seeds: {args.n_seeds} (base={args.base_seed})")
    print(f"Workers: {args.n_workers}")
    print(f"Output: {output_dir}")
    print()

    # Load all benchmark data
    def load_benchmark(bench_name):
        if bench_name == 'branin':
            np.random.seed(42)
            n_grid = 50
            x1 = np.linspace(0, 1, n_grid)
            x2 = np.linspace(0, 1, n_grid)
            X1, X2 = np.meshgrid(x1, x2)
            X = np.column_stack([X1.ravel(), X2.ravel()])
            return {
                'X': X,
                'y_hf': branin_hf(X),
                'y_lf': branin_lf(X, alpha=0.8),
                'cost_ratio': 0.1,
                'f_star': 0.397887,
                'name': 'Branin-2D'
            }
        elif bench_name == 'park':
            np.random.seed(42)
            n_grid = 10
            axes = [np.linspace(0.01, 1, n_grid) for _ in range(4)]
            grids = np.meshgrid(*axes, indexing='ij')
            X = np.column_stack([g.ravel() for g in grids])
            return {
                'X': X,
                'y_hf': park_hf(X),
                'y_lf': park_lf(X, alpha=0.6),
                'cost_ratio': 0.1,
                'f_star': park_hf(X).min(),
                'name': 'Park-4D'
            }
        elif bench_name == 'freesolv':
            return load_freesolv_benchmark(data_dir)
        elif bench_name == 'cofs':
            return load_cofs_benchmark(data_dir)
        elif bench_name == 'polarizability':
            return load_polarizability_benchmark(data_dir)

    # Print benchmark info
    print("Benchmark details:")
    benchmark_data_list = []
    for bench_name in benchmark_list:
        bench_data = load_benchmark(bench_name)
        benchmark_data_list.append((bench_name, bench_data))
        print(f"  {bench_data['name']}: {len(bench_data['X'])} points, "
              f"dim={bench_data['X'].shape[1]}, budget={BUDGETS[bench_name]}")
    print()

    # Models to compare (3 BLR variants)
    model_classes = [
        (SequentialHFBLR, "HF-BLR"),
        (SequentialLFBLR, "LF-BLR"),
        (SequentialDualBLR, "Dual-BLR"),
    ]

    print("Models:")
    print("  1. HF-BLR: BLR on HF, EI for both LF and HF (current approach)")
    print("  2. LF-BLR: BLR on LF for LF selection (EI), argmin for HF (original design)")
    print("  3. Dual-BLR: BLR on both LF and HF, EI for both")
    print()

    seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))

    # Prepare tasks for all (benchmark, model) combinations
    tasks = []
    for bench_name, bench_data in benchmark_data_list:
        budget = BUDGETS[bench_name]
        for model_class, model_name in model_classes:
            tasks.append((model_class, model_name, bench_data, seeds, budget, output_dir))

    total_combinations = len(benchmark_list) * len(model_classes)
    print(f"Total tasks: {total_combinations} ({len(benchmark_list)} benchmarks × {len(model_classes)} models)")
    print(f"Total runs: {total_combinations * args.n_seeds}")
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

    # Print summary per benchmark
    print("=" * 80)
    print("Summary per Benchmark (mean ± std)")
    print("=" * 80)

    for bench_name, bench_data in benchmark_data_list:
        print(f"\n{bench_data['name']} (budget={BUDGETS[bench_name]}):")
        print(f"{'Model':<15} {'Regret':>20} {'Success Rate':>15}")
        print("-" * 55)

        bench_df = df_summary[df_summary['benchmark'] == bench_data['name']]
        for model_class, model_name in model_classes:
            model_data = bench_df[bench_df['model'] == model_name]['final_regret'].dropna()
            if len(model_data) > 0:
                mean_r = model_data.mean()
                std_r = model_data.std()
                success_rate = (model_data == 0).sum() / len(model_data) * 100
                print(f"{model_name:<15} {mean_r:>8.4f} ± {std_r:<8.4f} {success_rate:>12.0f}%")
            else:
                print(f"{model_name:<15} {'N/A':>20} {'N/A':>15}")

        # Best for this benchmark
        bench_means = bench_df.groupby('model')['final_regret'].mean()
        if len(bench_means) > 0:
            best = bench_means.idxmin()
            print(f"  Best: {best} ({bench_means[best]:.4f})")

    # Overall summary
    print("\n" + "=" * 80)
    print("Overall Summary (across all benchmarks)")
    print("=" * 80)
    print(f"{'Model':<15} {'Mean Regret':>15} {'Avg Success':>15}")
    print("-" * 50)

    for model_class, model_name in model_classes:
        model_data = df_summary[df_summary['model'] == model_name]['final_regret'].dropna()
        if len(model_data) > 0:
            mean_r = model_data.mean()
            success_rate = (model_data == 0).sum() / len(model_data) * 100
            print(f"{model_name:<15} {mean_r:>15.4f} {success_rate:>14.0f}%")

    # Find overall best
    mean_regrets = df_summary.groupby('model')['final_regret'].mean()
    best_model = mean_regrets.idxmin()
    print()
    print(f"Overall Best: {best_model} (mean regret = {mean_regrets[best_model]:.4f})")

    # Plot comparison - one subplot per benchmark
    n_benchmarks = len(benchmark_list)
    fig, axes = plt.subplots(2, n_benchmarks, figsize=(5*n_benchmarks, 10))
    if n_benchmarks == 1:
        axes = axes.reshape(2, 1)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    model_names = [m[1] for m in model_classes]

    for i, (bench_name, bench_data) in enumerate(benchmark_data_list):
        bench_df_summary = df_summary[df_summary['benchmark'] == bench_data['name']]
        bench_df_traj = df_trajectory[df_trajectory['benchmark'] == bench_data['name']]

        # Top row: Bar plot
        ax = axes[0, i]
        means = [bench_df_summary[bench_df_summary['model'] == m]['final_regret'].mean() for m in model_names]
        stds = [bench_df_summary[bench_df_summary['model'] == m]['final_regret'].std() for m in model_names]
        ax.bar(model_names, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
        ax.set_ylabel('Simple Regret')
        ax.set_title(f'{bench_data["name"]}')
        ax.grid(True, alpha=0.3, axis='y')

        # Bottom row: Convergence
        ax = axes[1, i]
        for (model_class, model_name), color in zip(model_classes, colors):
            model_traj = bench_df_traj[bench_df_traj['model'] == model_name]
            if len(model_traj) > 0:
                avg_traj = model_traj.groupby('budget')['regret'].agg(['mean', 'std']).reset_index()
                ax.plot(avg_traj['budget'], avg_traj['mean'], '-', color=color, label=model_name, linewidth=2)
                ax.fill_between(avg_traj['budget'],
                               avg_traj['mean'] - avg_traj['std'],
                               avg_traj['mean'] + avg_traj['std'],
                               color=color, alpha=0.2)
        ax.set_xlabel('Budget')
        ax.set_ylabel('Simple Regret')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'BLR Placement Comparison ({args.n_seeds} seeds)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=150, bbox_inches='tight')

    print()
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print()
    print(f"Results saved to: {output_dir}")
    print(f"  - results_summary.csv: {len(df_summary)} rows")
    print(f"  - results_trajectory.csv: {len(df_trajectory)} rows")
    print(f"  - comparison.png: visualization")


if __name__ == '__main__':
    # CUDA multiprocessing fix
    mp.set_start_method('spawn', force=True)
    main()
