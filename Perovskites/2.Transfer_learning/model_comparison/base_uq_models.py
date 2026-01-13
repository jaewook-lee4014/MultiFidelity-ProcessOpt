#!/usr/bin/env python3
"""
Base Uncertainty Quantification (UQ) Models for Multi-Fidelity Optimization

Implements various DNN-based UQ methods:
1. GP (Gaussian Process) - baseline
2. DNGO (Deep Networks for Global Optimization) - DNN + BLR
3. BNN (Bayesian Neural Network) - Variational Inference
4. MC-Dropout - Dropout at inference
5. Deep Ensemble - Multiple independent networks  [NEW]
6. SNGP (Spectral Normalized Neural Gaussian Process) [NEW]

Author: Claude Code
Date: 2025-12-17
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
# Base Network Architecture
# ============================================================================

class BaseNetwork(nn.Module):
    """Base DNN for feature extraction"""
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
        self.out_layer = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        return self.out_layer(self.feature_net(x))

    def extract_features(self, x):
        return self.feature_net(x)


# ============================================================================
# 1. Deep Ensemble
# ============================================================================

class DeepEnsemble(nn.Module):
    """
    Deep Ensemble for Uncertainty Quantification

    Reference: Lakshminarayanan et al., "Simple and Scalable Predictive
               Uncertainty Estimation using Deep Ensembles", NeurIPS 2017

    Key ideas:
    - Train M independent neural networks with different random initializations
    - Each network predicts mean and variance (heteroscedastic)
    - Final prediction: mixture of Gaussians
    - Uncertainty = predictive variance (aleatoric + epistemic)
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, n_ensemble=5,
                 dropout=0.0, activation='relu'):
        super().__init__()
        self.n_ensemble = n_ensemble
        self.models = nn.ModuleList([
            EnsembleMember(input_dim, hidden_dim, num_layers, dropout, activation)
            for _ in range(n_ensemble)
        ])

    def forward(self, x, return_individual=False):
        """
        Forward pass through all ensemble members

        Returns:
            mean: Ensemble mean prediction
            var: Predictive variance (uncertainty)
            individual: (optional) predictions from each member
        """
        means = []
        vars = []

        for model in self.models:
            mu, var = model(x)
            means.append(mu)
            vars.append(var)

        means = torch.stack(means, dim=0)  # [n_ensemble, batch, 1]
        vars = torch.stack(vars, dim=0)

        # Mixture of Gaussians
        # Mean: average of means
        ensemble_mean = means.mean(dim=0)

        # Variance: E[var] + Var[mean] (law of total variance)
        # = average variance + variance of means
        avg_var = vars.mean(dim=0)  # Aleatoric uncertainty (average)
        var_of_means = means.var(dim=0)  # Epistemic uncertainty
        ensemble_var = avg_var + var_of_means

        if return_individual:
            return ensemble_mean, ensemble_var, means, vars
        return ensemble_mean, ensemble_var

    def predict_with_uncertainty(self, x):
        """Return predictions with uncertainty estimates"""
        self.eval()
        with torch.no_grad():
            mean, var = self.forward(x)
        return mean.cpu().numpy(), var.cpu().numpy()


class EnsembleMember(nn.Module):
    """
    Single ensemble member that predicts both mean and variance
    (Heteroscedastic regression)
    """
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

        # Two output heads: mean and log-variance
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.feature_net(x)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        # Ensure positive variance with softplus
        var = F.softplus(logvar) + 1e-6
        return mean, var


def train_deep_ensemble(model, X_train, y_train, params, device):
    """
    Train Deep Ensemble with negative log-likelihood loss

    Each member is trained independently with different initialization
    """
    epochs = params.get('epochs', 200)
    lr = params.get('lr', 1e-3)
    weight_decay = params.get('weight_decay', 1e-4)
    batch_size = params.get('batch_size', 32)

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

    # Train each ensemble member independently
    for member_idx, member in enumerate(model.models):
        # Reset member weights for independent training
        for m in member.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        optimizer = optim.Adam(member.parameters(), lr=lr, weight_decay=weight_decay)

        member.train()
        for epoch in range(epochs):
            # Mini-batch training
            perm = torch.randperm(len(X_t))
            for i in range(0, len(X_t), batch_size):
                batch_idx = perm[i:i+batch_size]
                X_batch = X_t[batch_idx]
                y_batch = y_t[batch_idx]

                optimizer.zero_grad()

                mean, var = member(X_batch)

                # Negative log-likelihood loss (Gaussian)
                # -log p(y|x) = 0.5 * (log(var) + (y - mean)^2 / var)
                nll_loss = 0.5 * (torch.log(var) + (y_batch - mean)**2 / var)
                loss = nll_loss.mean()

                loss.backward()
                optimizer.step()


class DeepEnsembleMultiFidelity(nn.Module):
    """
    Deep Ensemble with Multi-Fidelity Transfer Learning

    Architecture:
    - LF Ensemble: Pretrained on low-fidelity data
    - HF Ensemble: Fine-tuned on high-fidelity data with residual learning
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, n_ensemble=5,
                 dropout=0.0, activation='relu'):
        super().__init__()
        self.n_ensemble = n_ensemble

        # LF ensemble (feature extractors)
        self.lf_ensemble = DeepEnsemble(
            input_dim, hidden_dim, num_layers, n_ensemble, dropout, activation
        )

        # HF ensemble (residual predictors)
        # Input: x + y_lf prediction
        self.hf_ensemble = nn.ModuleList([
            EnsembleMember(input_dim + 1, hidden_dim, num_layers, dropout, activation)
            for _ in range(n_ensemble)
        ])

    def forward_lf(self, x):
        return self.lf_ensemble(x)

    def forward_hf(self, x, y_lf=None):
        """Forward through HF ensemble with residual connection"""
        if y_lf is None:
            y_lf, _ = self.forward_lf(x)

        # Concatenate input with LF prediction
        x_hf = torch.cat([x, y_lf], dim=-1)

        means = []
        vars = []
        for member in self.hf_ensemble:
            delta_mean, delta_var = member(x_hf)
            # Residual: y_hf = y_lf + delta
            hf_mean = y_lf + delta_mean
            means.append(hf_mean)
            vars.append(delta_var)

        means = torch.stack(means, dim=0)
        vars = torch.stack(vars, dim=0)

        ensemble_mean = means.mean(dim=0)
        avg_var = vars.mean(dim=0)
        var_of_means = means.var(dim=0)
        ensemble_var = avg_var + var_of_means

        return ensemble_mean, ensemble_var


def train_deep_ensemble_mf(model, X_lf, y_lf, X_hf, y_hf, params, device):
    """Train multi-fidelity Deep Ensemble"""
    lf_epochs = params.get('lf_epochs', 200)
    hf_epochs = params.get('hf_epochs', 100)
    lf_lr = params.get('lf_lr', 1e-3)
    hf_lr = params.get('hf_lr', 1e-4)
    weight_decay = params.get('weight_decay', 1e-4)

    X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(device)
    y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(device)
    X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(device)
    y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(device)

    # Stage 1: Train LF ensemble
    train_deep_ensemble(model.lf_ensemble, X_lf, y_lf,
                        {'epochs': lf_epochs, 'lr': lf_lr, 'weight_decay': weight_decay},
                        device)

    # Stage 2: Train HF ensemble (residual)
    # Freeze LF ensemble
    for param in model.lf_ensemble.parameters():
        param.requires_grad = False

    for member_idx, member in enumerate(model.hf_ensemble):
        # Initialize
        for m in member.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        optimizer = optim.Adam(member.parameters(), lr=hf_lr, weight_decay=weight_decay)

        member.train()
        for epoch in range(hf_epochs):
            optimizer.zero_grad()

            # Get LF predictions
            with torch.no_grad():
                y_lf_pred, _ = model.forward_lf(X_hf_t)

            # HF prediction
            x_hf_input = torch.cat([X_hf_t, y_lf_pred], dim=-1)
            delta_mean, delta_var = member(x_hf_input)
            hf_mean = y_lf_pred + delta_mean

            # NLL loss
            nll_loss = 0.5 * (torch.log(delta_var) + (y_hf_t - hf_mean)**2 / delta_var)
            loss = nll_loss.mean()

            loss.backward()
            optimizer.step()

    # Unfreeze
    for param in model.lf_ensemble.parameters():
        param.requires_grad = True


# ============================================================================
# 2. SNGP (Spectral Normalized Neural Gaussian Process)
# ============================================================================

class SpectralNormLinear(nn.Module):
    """
    Linear layer with Spectral Normalization

    Spectral normalization constrains the Lipschitz constant of the layer,
    making the network "distance-aware" in feature space.

    Reference: Miyato et al., "Spectral Normalization for GANs", ICLR 2018
    """
    def __init__(self, in_features, out_features, spectral_norm_bound=0.95, n_power_iterations=1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.spectral_norm_bound = spectral_norm_bound
        self.n_power_iterations = n_power_iterations

        # Initialize u and v for power iteration
        self.register_buffer('u', torch.randn(out_features))
        self.register_buffer('v', torch.randn(in_features))

    def _spectral_norm(self):
        """Compute spectral norm using power iteration"""
        weight = self.linear.weight.detach()  # Detach to avoid gradient issues
        u = self.u.clone()
        v = self.v.clone()

        for _ in range(self.n_power_iterations):
            v_new = F.normalize(torch.mv(weight.t(), u), dim=0)
            u_new = F.normalize(torch.mv(weight, v_new), dim=0)
            u = u_new
            v = v_new

        sigma = torch.dot(u, torch.mv(weight, v))

        # Update buffers (no gradient tracking needed)
        with torch.no_grad():
            self.u.copy_(u)
            self.v.copy_(v)

        return sigma

    def forward(self, x):
        # Compute spectral norm (for buffer updates only during training)
        if self.training:
            with torch.no_grad():
                self._spectral_norm()

        # Compute sigma with gradient tracking for forward pass
        weight = self.linear.weight
        sigma = torch.dot(self.u, torch.mv(weight, self.v))

        # Scale weight to have spectral norm <= bound
        if sigma > self.spectral_norm_bound:
            weight_scaled = weight * (self.spectral_norm_bound / sigma)
        else:
            weight_scaled = weight

        return F.linear(x, weight_scaled, self.linear.bias)


class RandomFourierFeatures(nn.Module):
    """
    Random Fourier Features for approximating RBF kernel GP

    Reference: Rahimi & Recht, "Random Features for Large-Scale Kernel Machines", NeurIPS 2007

    φ(x) = sqrt(2/D) * cos(Wx + b)
    where W ~ N(0, 1/length_scale^2), b ~ Uniform(0, 2π)
    """
    def __init__(self, in_features, num_features=1024, length_scale=1.0):
        super().__init__()
        self.num_features = num_features
        self.length_scale = length_scale

        # Random weights (fixed after initialization)
        self.register_buffer('W', torch.randn(in_features, num_features) / length_scale)
        self.register_buffer('b', torch.rand(num_features) * 2 * np.pi)

    def forward(self, x):
        # φ(x) = sqrt(2/D) * cos(xW + b)
        projection = torch.matmul(x, self.W) + self.b
        features = np.sqrt(2.0 / self.num_features) * torch.cos(projection)
        return features


class SNGP(nn.Module):
    """
    Spectral Normalized Neural Gaussian Process (SNGP)

    Reference: Liu et al., "Simple and Principled Uncertainty Estimation with
               Deterministic Deep Learning via Distance Awareness", NeurIPS 2020

    Architecture:
    1. Spectral normalized hidden layers (distance-aware)
    2. Random Fourier Features layer
    3. Gaussian Process output layer (Laplace approximation)

    Key benefits:
    - Single forward pass for uncertainty
    - Well-calibrated predictive uncertainty
    - Effective OOD detection
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_inducing=1024,
                 spectral_norm_bound=0.95, dropout=0.0, activation='relu',
                 ridge_penalty=1.0, length_scale=1.0):
        super().__init__()
        self.ridge_penalty = ridge_penalty

        # Spectral normalized feature extractor
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(SpectralNormLinear(in_dim, hidden_dim, spectral_norm_bound))
            layers.append(nn.Tanh() if activation == 'tanh' else nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)

        # Random Fourier Features for GP approximation
        self.rff = RandomFourierFeatures(hidden_dim, num_inducing, length_scale)

        # GP output layer parameters (learned during training)
        self.num_inducing = num_inducing
        self.register_buffer('precision_matrix', torch.eye(num_inducing) * ridge_penalty)
        self.register_buffer('mean_weights', torch.zeros(num_inducing))

        # Output scaling
        self.output_scale = nn.Parameter(torch.ones(1))
        self.output_bias = nn.Parameter(torch.zeros(1))

    def extract_features(self, x):
        """Extract spectral-normalized features"""
        return self.feature_net(x)

    def forward(self, x, return_uncertainty=True):
        """
        Forward pass with optional uncertainty estimation

        Returns:
            mean: Predicted mean
            var: Predictive variance (if return_uncertainty=True)
        """
        # Extract features
        features = self.extract_features(x)

        # Random Fourier Features
        phi = self.rff(features)  # [batch, num_inducing]

        # GP mean prediction: μ = φ(x)^T * β
        mean = torch.matmul(phi, self.mean_weights.unsqueeze(-1))
        mean = mean * self.output_scale + self.output_bias

        if return_uncertainty:
            # GP variance: σ² = φ(x)^T * Σ * φ(x)
            # where Σ = (Φ^T Φ + λI)^{-1}
            # Using Woodbury identity for efficiency
            precision_phi = torch.matmul(phi, self.precision_matrix)  # [batch, num_inducing]
            var = (precision_phi * phi).sum(dim=-1, keepdim=True)  # [batch, 1]
            var = var * (self.output_scale ** 2) + 1e-6  # Scale and add jitter

            return mean, var

        return mean

    def fit_gp_layer(self, X, y):
        """
        Fit GP output layer using training data

        Computes posterior mean and precision matrix
        """
        with torch.no_grad():
            features = self.extract_features(X)
            phi = self.rff(features)  # [N, D]

            # Precision matrix: Λ = Φ^T Φ + λI
            precision = torch.matmul(phi.t(), phi) + self.ridge_penalty * torch.eye(
                self.num_inducing, device=phi.device
            )

            # Solve for mean weights: β = Λ^{-1} Φ^T y
            phi_t_y = torch.matmul(phi.t(), y)

            # Use Cholesky decomposition for numerical stability
            try:
                L = torch.linalg.cholesky(precision)
                mean_weights = torch.cholesky_solve(phi_t_y, L).squeeze()
            except:
                # Fallback to pseudoinverse
                mean_weights = torch.matmul(
                    torch.linalg.pinv(precision), phi_t_y
                ).squeeze()

            # Store covariance (inverse of precision) for uncertainty
            try:
                covariance = torch.cholesky_inverse(L)
            except:
                covariance = torch.linalg.pinv(precision)

            self.precision_matrix.copy_(covariance)
            self.mean_weights.copy_(mean_weights)

    def predict_with_uncertainty(self, x):
        """Return predictions with uncertainty estimates"""
        self.eval()
        with torch.no_grad():
            mean, var = self.forward(x, return_uncertainty=True)
        return mean.cpu().numpy(), var.cpu().numpy()


def train_sngp(model, X_train, y_train, params, device):
    """
    Train SNGP model

    Two-stage training:
    1. Train feature extractor with standard regression loss
    2. Fit GP output layer on extracted features
    """
    epochs = params.get('epochs', 200)
    lr = params.get('lr', 1e-3)
    weight_decay = params.get('weight_decay', 1e-4)

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

    # Stage 1: Train feature extractor
    # Use a temporary output layer for training
    temp_out = nn.Linear(model.num_inducing, 1).to(device)

    # Combine parameters
    train_params = list(model.feature_net.parameters()) + list(temp_out.parameters())
    train_params.append(model.output_scale)
    train_params.append(model.output_bias)

    optimizer = optim.Adam(train_params, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        features = model.extract_features(X_t)
        phi = model.rff(features)
        pred = temp_out(phi) * model.output_scale + model.output_bias

        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()

    # Stage 2: Fit GP output layer
    model.eval()
    model.fit_gp_layer(X_t, y_t)


class SNGPMultiFidelity(nn.Module):
    """
    SNGP with Multi-Fidelity Transfer Learning

    Architecture:
    - LF SNGP: Pretrained on low-fidelity data
    - HF SNGP: Fine-tuned on high-fidelity data with residual learning
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_inducing=512,
                 spectral_norm_bound=0.95, dropout=0.0, activation='relu',
                 ridge_penalty=1.0, length_scale=1.0):
        super().__init__()

        # LF SNGP
        self.lf_sngp = SNGP(
            input_dim, hidden_dim, num_layers, num_inducing,
            spectral_norm_bound, dropout, activation, ridge_penalty, length_scale
        )

        # HF SNGP (takes x + y_lf as input)
        self.hf_sngp = SNGP(
            input_dim + 1, hidden_dim, num_layers, num_inducing,
            spectral_norm_bound, dropout, activation, ridge_penalty, length_scale
        )

    def forward_lf(self, x, return_uncertainty=True):
        return self.lf_sngp(x, return_uncertainty)

    def forward_hf(self, x, y_lf=None, return_uncertainty=True):
        """Forward through HF SNGP with residual connection"""
        if y_lf is None:
            y_lf, _ = self.forward_lf(x, return_uncertainty=True)

        # Concatenate input with LF prediction
        x_hf = torch.cat([x, y_lf], dim=-1)
        delta_mean, delta_var = self.hf_sngp(x_hf, return_uncertainty=True)

        # Residual: y_hf = y_lf + delta
        hf_mean = y_lf + delta_mean

        if return_uncertainty:
            return hf_mean, delta_var
        return hf_mean


def train_sngp_mf(model, X_lf, y_lf, X_hf, y_hf, params, device):
    """Train multi-fidelity SNGP"""
    lf_epochs = params.get('lf_epochs', 200)
    hf_epochs = params.get('hf_epochs', 100)
    lf_lr = params.get('lf_lr', 1e-3)
    hf_lr = params.get('hf_lr', 1e-4)
    weight_decay = params.get('weight_decay', 1e-4)

    X_lf_t = torch.tensor(X_lf, dtype=torch.float32).to(device)
    y_lf_t = torch.tensor(y_lf, dtype=torch.float32).view(-1, 1).to(device)
    X_hf_t = torch.tensor(X_hf, dtype=torch.float32).to(device)
    y_hf_t = torch.tensor(y_hf, dtype=torch.float32).view(-1, 1).to(device)

    # Stage 1: Train LF SNGP
    train_sngp(model.lf_sngp, X_lf, y_lf,
               {'epochs': lf_epochs, 'lr': lf_lr, 'weight_decay': weight_decay},
               device)

    # Stage 2: Train HF SNGP (residual)
    # Freeze LF SNGP
    for param in model.lf_sngp.parameters():
        param.requires_grad = False

    # Get LF predictions for HF inputs
    model.lf_sngp.eval()
    with torch.no_grad():
        y_lf_pred, _ = model.forward_lf(X_hf_t)

    # Prepare HF training data (residuals)
    X_hf_input = torch.cat([X_hf_t, y_lf_pred], dim=-1)
    y_hf_residual = y_hf_t - y_lf_pred  # Train to predict residual

    # Train HF SNGP feature extractor
    temp_out = nn.Linear(model.hf_sngp.num_inducing, 1).to(device)
    train_params = list(model.hf_sngp.feature_net.parameters()) + list(temp_out.parameters())
    train_params.append(model.hf_sngp.output_scale)
    train_params.append(model.hf_sngp.output_bias)

    optimizer = optim.Adam(train_params, lr=hf_lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    model.hf_sngp.train()
    for epoch in range(hf_epochs):
        optimizer.zero_grad()

        features = model.hf_sngp.extract_features(X_hf_input)
        phi = model.hf_sngp.rff(features)
        pred = temp_out(phi) * model.hf_sngp.output_scale + model.hf_sngp.output_bias

        loss = loss_fn(pred, y_hf_residual)
        loss.backward()
        optimizer.step()

    # Fit GP layer
    model.hf_sngp.eval()
    model.hf_sngp.fit_gp_layer(X_hf_input, y_hf_residual)

    # Unfreeze
    for param in model.lf_sngp.parameters():
        param.requires_grad = True


# ============================================================================
# Evaluation Utilities
# ============================================================================

def evaluate_uq_model(model, X_test, y_test, device, model_type='ensemble'):
    """Evaluate UQ model on test set"""
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        if model_type in ['ensemble', 'ensemble_mf']:
            if model_type == 'ensemble_mf':
                y_pred, var = model.forward_hf(X_t)
            else:
                y_pred, var = model(X_t)
            y_pred = y_pred.cpu().numpy().flatten()
            uncertainty = np.sqrt(var.cpu().numpy().flatten())
        elif model_type in ['sngp', 'sngp_mf']:
            if model_type == 'sngp_mf':
                y_pred, var = model.forward_hf(X_t)
            else:
                y_pred, var = model(X_t)
            y_pred = y_pred.cpu().numpy().flatten()
            uncertainty = np.sqrt(var.cpu().numpy().flatten())
        else:
            y_pred = model(X_t).cpu().numpy().flatten()
            uncertainty = None

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        'rmse': rmse,
        'r2': r2,
        'y_pred': y_pred,
        'uncertainty': uncertainty
    }


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_base_model_comparison():
    """Run comparison of base UQ models"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Data: {N_LOFI} LF + {N_HIFI} HF")
    print(f"Folds: {len(SEEDS)}")

    # Load data
    print("\nLoading data...")
    lookup, all_combinations, _ = load_base_data()
    print(f"Total compositions: {len(all_combinations)}")

    # Define models to compare
    models_config = {
        'Deep Ensemble (MF)': {
            'model_class': DeepEnsembleMultiFidelity,
            'train_func': train_deep_ensemble_mf,
            'model_type': 'ensemble_mf',
            'params': {
                'input_dim': 3, 'hidden_dim': 64, 'num_layers': 2,
                'n_ensemble': 5, 'dropout': 0.0, 'activation': 'relu',
                'lf_epochs': 200, 'hf_epochs': 100,
                'lf_lr': 1e-3, 'hf_lr': 1e-4, 'weight_decay': 1e-4
            }
        },
        'SNGP (MF)': {
            'model_class': SNGPMultiFidelity,
            'train_func': train_sngp_mf,
            'model_type': 'sngp_mf',
            'params': {
                'input_dim': 3, 'hidden_dim': 64, 'num_layers': 2,
                'num_inducing': 256, 'spectral_norm_bound': 0.95,
                'dropout': 0.0, 'activation': 'relu',
                'ridge_penalty': 1.0, 'length_scale': 1.0,
                'lf_epochs': 200, 'hf_epochs': 100,
                'lf_lr': 1e-3, 'hf_lr': 1e-4, 'weight_decay': 1e-4
            }
        },
    }

    # Run experiments
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / 'visualizations' / f'{timestamp}_base_uq_models'
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {name: [] for name in models_config}

    print("\n" + "="*70)
    print("Running Base UQ Model Comparison")
    print("="*70)

    for fold_idx, seed in enumerate(SEEDS, 1):
        print(f"\nFold {fold_idx}/{len(SEEDS)} (seed={seed})")
        print("-"*50)

        data = generate_data(lookup, all_combinations, n_lofi=N_LOFI, n_hifi=N_HIFI, seed=seed)

        # Test set (all data except HF train)
        test_mask = np.ones(len(data['X_all']), dtype=bool)
        test_mask[data['hifi_idx']] = False
        X_test = data['X_all'][test_mask]
        y_test = data['y_all'][test_mask]

        for model_name, config in models_config.items():
            set_seeds(seed)
            try:
                # Create model
                model_params = {k: v for k, v in config['params'].items()
                               if k in ['input_dim', 'hidden_dim', 'num_layers',
                                        'n_ensemble', 'num_inducing', 'spectral_norm_bound',
                                        'dropout', 'activation', 'ridge_penalty', 'length_scale']}
                model = config['model_class'](**model_params).to(device)

                # Train
                config['train_func'](
                    model, data['X_low'], data['y_low'],
                    data['X_high'], data['y_high'],
                    config['params'], device
                )

                # Evaluate
                results = evaluate_uq_model(model, X_test, y_test, device, config['model_type'])
                all_results[model_name].append(results)

                print(f"  {model_name:<25}: R²={results['r2']:.4f}, RMSE={results['rmse']:.4f}")

            except Exception as e:
                print(f"  {model_name:<25}: ERROR - {e}")
                import traceback
                traceback.print_exc()
                all_results[model_name].append({'r2': np.nan, 'rmse': np.nan})

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Model':<25} {'Mean R²':<12} {'Std R²':<12} {'Mean RMSE':<12}")
    print("-"*70)

    import pandas as pd
    summary_data = []
    for model_name, results in all_results.items():
        r2_values = [r['r2'] for r in results if not np.isnan(r.get('r2', np.nan))]
        rmse_values = [r['rmse'] for r in results if not np.isnan(r.get('rmse', np.nan))]

        if r2_values:
            mean_r2 = np.mean(r2_values)
            std_r2 = np.std(r2_values)
            mean_rmse = np.mean(rmse_values)
            print(f"{model_name:<25} {mean_r2:<12.4f} {std_r2:<12.4f} {mean_rmse:<12.4f}")
            summary_data.append({
                'model': model_name,
                'mean_r2': mean_r2,
                'std_r2': std_r2,
                'mean_rmse': mean_rmse
            })

    # Save results
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_dir / 'summary_results.csv', index=False)

    print(f"\nResults saved to: {output_dir}")

    return all_results


if __name__ == '__main__':
    run_base_model_comparison()
