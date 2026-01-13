#!/usr/bin/env python
"""
Multi-Fidelity UQ Models for Bayesian Optimization

12 MF Models = 6 UQ Models × 2 Transfer Learning Approaches

UQ Models:
1. GP (Gaussian Process)
2. DNGO (Deep Networks for Global Optimization)
3. BNN (Bayesian Neural Network)
4. MC-Dropout
5. Deep Ensemble
6. SNGP (Spectral-Normalized GP)

Transfer Learning Approaches:
1. MFGP: Multi-Fidelity GP with fidelity correlation
2. TL (Transfer Learning): Pretrain on LF, fine-tune on HF

Naming: {UQ}_{TL} e.g., GP_MFGP, DNGO_TL
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Callable
from sklearn.preprocessing import StandardScaler

# BoTorch imports for MFGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# BASE CLASSES
# =============================================================================

class BaseMFModel:
    """Base class for Multi-Fidelity models"""

    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.is_fitted = False

    def fit(self, X_lf: np.ndarray, y_lf: np.ndarray,
            X_hf: np.ndarray, y_hf: np.ndarray):
        """Fit with both LF and HF data"""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict at HF level: returns (mean, std)"""
        raise NotImplementedError


# =============================================================================
# 1. MFGP-BASED MODELS (BoTorch SingleTaskMultiFidelityGP)
# =============================================================================

class GP_MFGP(BaseMFModel):
    """
    Multi-Fidelity GP using BoTorch's SingleTaskMultiFidelityGP
    Learns correlation between LF and HF data
    """

    def __init__(self, input_dim: int):
        super().__init__(input_dim)
        self.model = None

    def fit(self, X_lf: np.ndarray, y_lf: np.ndarray,
            X_hf: np.ndarray, y_hf: np.ndarray):
        # Prepare data with fidelity dimension
        n_lf, n_hf = len(X_lf), len(X_hf)

        # Add fidelity column (0=LF, 1=HF)
        X_lf_fid = np.hstack([X_lf, np.zeros((n_lf, 1))])
        X_hf_fid = np.hstack([X_hf, np.ones((n_hf, 1))])

        train_X = np.vstack([X_lf_fid, X_hf_fid])
        train_Y = np.concatenate([y_lf.flatten(), y_hf.flatten()]).reshape(-1, 1)

        train_X = torch.tensor(train_X, dtype=torch.double).to(device)
        train_Y = torch.tensor(train_Y, dtype=torch.double).to(device)

        # Create and fit MFGP
        self.model = SingleTaskMultiFidelityGP(
            train_X=train_X,
            train_Y=train_Y,
            outcome_transform=Standardize(m=1),
            data_fidelities=[self.input_dim]
        ).to(device)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        # Predict at HF level (fidelity=1)
        X_fid = np.hstack([X, np.ones((len(X), 1))])
        X_tensor = torch.tensor(X_fid, dtype=torch.double).to(device)

        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X_tensor)
            mean = posterior.mean.cpu().numpy().flatten()
            std = np.sqrt(posterior.variance.cpu().numpy().flatten())

        return mean, np.maximum(std, 1e-6)


# =============================================================================
# 2. TRANSFER LEARNING MODELS (Pretrain on LF, Fine-tune on HF)
# =============================================================================

class DNGO_MFGP(BaseMFModel):
    """
    DNGO with MFGP-style approach:
    - Train feature extractor on combined LF+HF data
    - Use BLR on HF predictions
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 lr: float = 0.01, epochs: int = 300):
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.input_dim + 1, self.hidden_dim),  # +1 for fidelity
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 50)
        ).to(device)

    def fit(self, X_lf: np.ndarray, y_lf: np.ndarray,
            X_hf: np.ndarray, y_hf: np.ndarray):
        n_lf, n_hf = len(X_lf), len(X_hf)

        # Add fidelity dimension
        X_lf_fid = np.hstack([X_lf, np.zeros((n_lf, 1))])
        X_hf_fid = np.hstack([X_hf, np.ones((n_hf, 1))])

        X_all = np.vstack([X_lf_fid, X_hf_fid])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])

        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

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

        # BLR on HF data only
        self.network.eval()
        X_hf_scaled = self.scaler_x.transform(X_hf_fid)
        X_hf_t = torch.FloatTensor(X_hf_scaled).to(device)
        y_hf_scaled = self.scaler_y.transform(y_hf.reshape(-1, 1)).flatten()

        with torch.no_grad():
            Phi = self.network(X_hf_t).cpu().numpy()

        # BLR parameters
        alpha, beta = 0.1, 2.0
        A = alpha * np.eye(Phi.shape[1]) + beta * Phi.T @ Phi
        self.A_inv = np.linalg.inv(A + 1e-6 * np.eye(A.shape[0]))
        self.m = beta * self.A_inv @ Phi.T @ y_hf_scaled
        self.beta = beta
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        # Predict at HF level
        X_fid = np.hstack([X, np.ones((len(X), 1))])
        X_scaled = self.scaler_x.transform(X_fid)
        X_t = torch.FloatTensor(X_scaled).to(device)

        self.network.eval()
        with torch.no_grad():
            Phi = self.network(X_t).cpu().numpy()

        mean = Phi @ self.m
        var = 1/self.beta + np.sum(Phi @ self.A_inv * Phi, axis=1)
        std = np.sqrt(np.maximum(var, 1e-6))

        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
        std = std * self.scaler_y.scale_[0]

        return mean, std


class DNGO_Joint(BaseMFModel):
    """
    DNGO with Joint Training (Best performer in model_comparison)
    - Train LF and HF networks jointly with alpha weighting
    - loss = (1-alpha)*LF_loss + alpha*HF_loss
    - HF network takes LF prediction as additional input (residual learning)
    - BLR on HF features for uncertainty
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 lr: float = 0.01, epochs: int = 300, alpha: float = 0.2):
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha  # Weight for HF loss
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def _build_lf_network(self):
        """LF network: predicts LF values"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        ).to(device)

    def _build_hf_network(self):
        """HF network: takes x and LF prediction, outputs HF feature"""
        return nn.Sequential(
            nn.Linear(self.input_dim + 1, self.hidden_dim),  # +1 for LF prediction
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 50)  # Feature dimension for BLR
        ).to(device)

    def fit(self, X_lf: np.ndarray, y_lf: np.ndarray,
            X_hf: np.ndarray, y_hf: np.ndarray):
        # Combine and scale
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

        optimizer = torch.optim.Adam(
            list(self.lf_network.parameters()) + list(self.hf_network.parameters()),
            lr=self.lr
        )
        loss_fn = nn.MSELoss()

        # Joint training
        for _ in range(self.epochs):
            optimizer.zero_grad()

            # LF loss
            y_lf_pred = self.lf_network(X_lf_t)
            lf_loss = loss_fn(y_lf_pred, y_lf_t)

            # HF loss (with LF prediction as input)
            with torch.no_grad():
                y_lf_for_hf = self.lf_network(X_hf_t)
            hf_input = torch.cat([X_hf_t, y_lf_for_hf], dim=1)
            hf_features = self.hf_network(hf_input)
            y_hf_pred = hf_features.mean(dim=1, keepdim=True)
            hf_loss = loss_fn(y_hf_pred, y_hf_t)

            # Combined loss
            total_loss = (1 - self.alpha) * lf_loss + self.alpha * hf_loss
            total_loss.backward()
            optimizer.step()

        # BLR on HF features
        self.lf_network.eval()
        self.hf_network.eval()
        with torch.no_grad():
            y_lf_for_hf = self.lf_network(X_hf_t)
            hf_input = torch.cat([X_hf_t, y_lf_for_hf], dim=1)
            Phi = self.hf_network(hf_input).cpu().numpy()

        alpha_blr, beta = 0.1, 2.0
        A = alpha_blr * np.eye(Phi.shape[1]) + beta * Phi.T @ Phi
        self.A_inv = np.linalg.inv(A + 1e-6 * np.eye(A.shape[0]))
        self.m = beta * self.A_inv @ Phi.T @ y_hf_scaled
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

        mean = Phi @ self.m
        var = 1/self.beta + np.sum(Phi @ self.A_inv * Phi, axis=1)
        std = np.sqrt(np.maximum(var, 1e-6))

        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
        std = std * self.scaler_y.scale_[0]

        return mean, std


class DNGO_TL(BaseMFModel):
    """
    DNGO with Transfer Learning:
    - Pretrain feature extractor on LF data
    - Fine-tune on HF data
    - BLR on HF features
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 lr: float = 0.01, pretrain_epochs: int = 200, finetune_epochs: int = 100):
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.scaler_x = StandardScaler()
        self.scaler_y_lf = StandardScaler()
        self.scaler_y_hf = StandardScaler()

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 50)
        ).to(device)

    def fit(self, X_lf: np.ndarray, y_lf: np.ndarray,
            X_hf: np.ndarray, y_hf: np.ndarray):
        # Pretrain on LF
        X_lf_scaled = self.scaler_x.fit_transform(X_lf)
        y_lf_scaled = self.scaler_y_lf.fit_transform(y_lf.reshape(-1, 1)).flatten()

        X_lf_t = torch.FloatTensor(X_lf_scaled).to(device)
        y_lf_t = torch.FloatTensor(y_lf_scaled).to(device)

        self.network = self._build_network()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        # Pretrain
        self.network.train()
        for _ in range(self.pretrain_epochs):
            optimizer.zero_grad()
            features = self.network(X_lf_t)
            pred = features.mean(dim=1)
            loss = nn.MSELoss()(pred, y_lf_t)
            loss.backward()
            optimizer.step()

        # Fine-tune on HF
        X_hf_scaled = self.scaler_x.transform(X_hf)
        y_hf_scaled = self.scaler_y_hf.fit_transform(y_hf.reshape(-1, 1)).flatten()

        X_hf_t = torch.FloatTensor(X_hf_scaled).to(device)
        y_hf_t = torch.FloatTensor(y_hf_scaled).to(device)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr * 0.1)
        for _ in range(self.finetune_epochs):
            optimizer.zero_grad()
            features = self.network(X_hf_t)
            pred = features.mean(dim=1)
            loss = nn.MSELoss()(pred, y_hf_t)
            loss.backward()
            optimizer.step()

        # BLR on HF
        self.network.eval()
        with torch.no_grad():
            Phi = self.network(X_hf_t).cpu().numpy()

        alpha, beta = 0.1, 2.0
        A = alpha * np.eye(Phi.shape[1]) + beta * Phi.T @ Phi
        self.A_inv = np.linalg.inv(A + 1e-6 * np.eye(A.shape[0]))
        self.m = beta * self.A_inv @ Phi.T @ y_hf_scaled
        self.beta = beta
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
        var = 1/self.beta + np.sum(Phi @ self.A_inv * Phi, axis=1)
        std = np.sqrt(np.maximum(var, 1e-6))

        mean = self.scaler_y_hf.inverse_transform(mean.reshape(-1, 1)).flatten()
        std = std * self.scaler_y_hf.scale_[0]

        return mean, std


class BNN_MFGP(BaseMFModel):
    """BNN with MFGP-style fidelity encoding"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 lr: float = 0.02, epochs: int = 300, kl_weight: float = 0.2, n_samples: int = 20):
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.kl_weight = kl_weight
        self.n_samples = n_samples
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X_lf: np.ndarray, y_lf: np.ndarray,
            X_hf: np.ndarray, y_hf: np.ndarray):
        n_lf, n_hf = len(X_lf), len(X_hf)

        # Add fidelity dimension
        X_lf_fid = np.hstack([X_lf, np.zeros((n_lf, 1))])
        X_hf_fid = np.hstack([X_hf, np.ones((n_hf, 1))])

        X_all = np.vstack([X_lf_fid, X_hf_fid])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])

        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

        X_t = torch.FloatTensor(X_scaled).to(device)
        y_t = torch.FloatTensor(y_scaled).to(device)

        # Build variational parameters
        self.mu = nn.ParameterList()
        self.log_sigma = nn.ParameterList()

        in_dim = self.input_dim + 1  # +1 for fidelity
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
            loss = nll + self.kl_weight * kl / len(X_all)

            loss.backward()
            optimizer.step()

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        # Predict at HF level
        X_fid = np.hstack([X, np.ones((len(X), 1))])
        X_scaled = self.scaler_x.transform(X_fid)
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

        preds = np.stack(preds)
        mean = preds.mean(axis=0).flatten()
        std = preds.std(axis=0).flatten()

        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
        std = std * self.scaler_y.scale_[0]

        return mean, np.maximum(std, 1e-6)


class BNN_TL(BaseMFModel):
    """BNN with Transfer Learning: Pretrain on LF, fine-tune on HF"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 lr: float = 0.02, pretrain_epochs: int = 200, finetune_epochs: int = 100,
                 kl_weight: float = 0.2, n_samples: int = 20):
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.kl_weight = kl_weight
        self.n_samples = n_samples
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def _train_epoch(self, X_t, y_t, optimizer, n_data):
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
        loss = nll + self.kl_weight * kl / n_data

        loss.backward()
        optimizer.step()

    def fit(self, X_lf: np.ndarray, y_lf: np.ndarray,
            X_hf: np.ndarray, y_hf: np.ndarray):
        # Scale LF data
        X_lf_scaled = self.scaler_x.fit_transform(X_lf)
        y_lf_scaled = self.scaler_y.fit_transform(y_lf.reshape(-1, 1)).flatten()

        X_lf_t = torch.FloatTensor(X_lf_scaled).to(device)
        y_lf_t = torch.FloatTensor(y_lf_scaled).to(device)

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

        # Pretrain on LF
        optimizer = torch.optim.Adam(list(self.mu) + list(self.log_sigma), lr=self.lr)
        for _ in range(self.pretrain_epochs):
            self._train_epoch(X_lf_t, y_lf_t, optimizer, len(X_lf))

        # Fine-tune on HF
        X_hf_scaled = self.scaler_x.transform(X_hf)
        y_hf_scaled = self.scaler_y.transform(y_hf.reshape(-1, 1)).flatten()

        X_hf_t = torch.FloatTensor(X_hf_scaled).to(device)
        y_hf_t = torch.FloatTensor(y_hf_scaled).to(device)

        optimizer = torch.optim.Adam(list(self.mu) + list(self.log_sigma), lr=self.lr * 0.1)
        for _ in range(self.finetune_epochs):
            self._train_epoch(X_hf_t, y_hf_t, optimizer, len(X_hf))

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

        preds = np.stack(preds)
        mean = preds.mean(axis=0).flatten()
        std = preds.std(axis=0).flatten()

        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
        std = std * self.scaler_y.scale_[0]

        return mean, np.maximum(std, 1e-6)


class MCDropout_MFGP(BaseMFModel):
    """MC-Dropout with MFGP-style fidelity encoding"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 1,
                 dropout: float = 0.15, lr: float = 0.02, epochs: int = 300, n_samples: int = 50):
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.n_samples = n_samples
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def _build_network(self):
        layers = []
        in_dim = self.input_dim + 1  # +1 for fidelity
        for _ in range(self.num_layers):
            layers.extend([
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            in_dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, 1))
        return nn.Sequential(*layers).to(device)

    def fit(self, X_lf: np.ndarray, y_lf: np.ndarray,
            X_hf: np.ndarray, y_hf: np.ndarray):
        n_lf, n_hf = len(X_lf), len(X_hf)

        X_lf_fid = np.hstack([X_lf, np.zeros((n_lf, 1))])
        X_hf_fid = np.hstack([X_hf, np.ones((n_hf, 1))])

        X_all = np.vstack([X_lf_fid, X_hf_fid])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])

        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

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

        X_fid = np.hstack([X, np.ones((len(X), 1))])
        X_scaled = self.scaler_x.transform(X_fid)
        X_t = torch.FloatTensor(X_scaled).to(device)

        self.network.train()  # Keep dropout active
        preds = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.network(X_t).cpu().numpy()
                preds.append(pred)

        preds = np.stack(preds)
        mean = preds.mean(axis=0).flatten()
        std = preds.std(axis=0).flatten()

        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
        std = std * self.scaler_y.scale_[0]

        return mean, np.maximum(std, 1e-6)


class MCDropout_TL(BaseMFModel):
    """MC-Dropout with Transfer Learning"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 1,
                 dropout: float = 0.15, lr: float = 0.02,
                 pretrain_epochs: int = 200, finetune_epochs: int = 100, n_samples: int = 50):
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.n_samples = n_samples
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def _build_network(self):
        layers = []
        in_dim = self.input_dim
        for _ in range(self.num_layers):
            layers.extend([
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            in_dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, 1))
        return nn.Sequential(*layers).to(device)

    def fit(self, X_lf: np.ndarray, y_lf: np.ndarray,
            X_hf: np.ndarray, y_hf: np.ndarray):
        # Pretrain on LF
        X_lf_scaled = self.scaler_x.fit_transform(X_lf)
        y_lf_scaled = self.scaler_y.fit_transform(y_lf.reshape(-1, 1)).flatten()

        X_lf_t = torch.FloatTensor(X_lf_scaled).to(device)
        y_lf_t = torch.FloatTensor(y_lf_scaled).to(device)

        self.network = self._build_network()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        self.network.train()
        for _ in range(self.pretrain_epochs):
            optimizer.zero_grad()
            pred = self.network(X_lf_t).squeeze()
            loss = nn.MSELoss()(pred, y_lf_t)
            loss.backward()
            optimizer.step()

        # Fine-tune on HF
        X_hf_scaled = self.scaler_x.transform(X_hf)
        y_hf_scaled = self.scaler_y.transform(y_hf.reshape(-1, 1)).flatten()

        X_hf_t = torch.FloatTensor(X_hf_scaled).to(device)
        y_hf_t = torch.FloatTensor(y_hf_scaled).to(device)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr * 0.1)
        for _ in range(self.finetune_epochs):
            optimizer.zero_grad()
            pred = self.network(X_hf_t).squeeze()
            loss = nn.MSELoss()(pred, y_hf_t)
            loss.backward()
            optimizer.step()

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_scaled = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(device)

        self.network.train()
        preds = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.network(X_t).cpu().numpy()
                preds.append(pred)

        preds = np.stack(preds)
        mean = preds.mean(axis=0).flatten()
        std = preds.std(axis=0).flatten()

        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
        std = std * self.scaler_y.scale_[0]

        return mean, np.maximum(std, 1e-6)


class DeepEnsemble_MFGP(BaseMFModel):
    """Deep Ensemble with MFGP-style fidelity encoding"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1,
                 n_ensemble: int = 3, lr: float = 0.01, epochs: int = 300):
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_ensemble = n_ensemble
        self.lr = lr
        self.epochs = epochs
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def _build_member(self):
        layers = []
        in_dim = self.input_dim + 1
        for _ in range(self.num_layers):
            layers.extend([nn.Linear(in_dim, self.hidden_dim), nn.ReLU()])
            in_dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, 2))  # mean + log_var
        return nn.Sequential(*layers).to(device)

    def fit(self, X_lf: np.ndarray, y_lf: np.ndarray,
            X_hf: np.ndarray, y_hf: np.ndarray):
        n_lf, n_hf = len(X_lf), len(X_hf)

        X_lf_fid = np.hstack([X_lf, np.zeros((n_lf, 1))])
        X_hf_fid = np.hstack([X_hf, np.ones((n_hf, 1))])

        X_all = np.vstack([X_lf_fid, X_hf_fid])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])

        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

        X_t = torch.FloatTensor(X_scaled).to(device)
        y_t = torch.FloatTensor(y_scaled).to(device)

        self.networks = []
        for i in range(self.n_ensemble):
            torch.manual_seed(i * 1000)
            net = self._build_member()
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)

            net.train()
            for _ in range(self.epochs):
                optimizer.zero_grad()
                out = net(X_t)
                mean = out[:, 0]
                log_var = out[:, 1]
                var = F.softplus(log_var) + 1e-6
                loss = 0.5 * torch.mean(torch.log(var) + (y_t - mean)**2 / var)
                loss.backward()
                optimizer.step()

            self.networks.append(net)

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_fid = np.hstack([X, np.ones((len(X), 1))])
        X_scaled = self.scaler_x.transform(X_fid)
        X_t = torch.FloatTensor(X_scaled).to(device)

        means, vars = [], []
        for net in self.networks:
            net.eval()
            with torch.no_grad():
                out = net(X_t)
                mean = out[:, 0].cpu().numpy()
                log_var = out[:, 1].cpu().numpy()
                var = np.exp(log_var) + 1e-6
                means.append(mean)
                vars.append(var)

        means = np.stack(means)
        vars = np.stack(vars)

        ensemble_mean = means.mean(axis=0)
        ensemble_var = vars.mean(axis=0) + means.var(axis=0)

        ensemble_mean = self.scaler_y.inverse_transform(ensemble_mean.reshape(-1, 1)).flatten()
        ensemble_std = np.sqrt(ensemble_var) * self.scaler_y.scale_[0]

        return ensemble_mean, np.maximum(ensemble_std, 1e-6)


class DeepEnsemble_TL(BaseMFModel):
    """Deep Ensemble with Transfer Learning"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1,
                 n_ensemble: int = 3, lr: float = 0.01,
                 pretrain_epochs: int = 200, finetune_epochs: int = 100):
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_ensemble = n_ensemble
        self.lr = lr
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def _build_member(self):
        layers = []
        in_dim = self.input_dim
        for _ in range(self.num_layers):
            layers.extend([nn.Linear(in_dim, self.hidden_dim), nn.ReLU()])
            in_dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, 2))
        return nn.Sequential(*layers).to(device)

    def fit(self, X_lf: np.ndarray, y_lf: np.ndarray,
            X_hf: np.ndarray, y_hf: np.ndarray):
        X_lf_scaled = self.scaler_x.fit_transform(X_lf)
        y_lf_scaled = self.scaler_y.fit_transform(y_lf.reshape(-1, 1)).flatten()

        X_lf_t = torch.FloatTensor(X_lf_scaled).to(device)
        y_lf_t = torch.FloatTensor(y_lf_scaled).to(device)

        X_hf_scaled = self.scaler_x.transform(X_hf)
        y_hf_scaled = self.scaler_y.transform(y_hf.reshape(-1, 1)).flatten()

        X_hf_t = torch.FloatTensor(X_hf_scaled).to(device)
        y_hf_t = torch.FloatTensor(y_hf_scaled).to(device)

        self.networks = []
        for i in range(self.n_ensemble):
            torch.manual_seed(i * 1000)
            net = self._build_member()

            # Pretrain on LF
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
            net.train()
            for _ in range(self.pretrain_epochs):
                optimizer.zero_grad()
                out = net(X_lf_t)
                mean = out[:, 0]
                log_var = out[:, 1]
                var = F.softplus(log_var) + 1e-6
                loss = 0.5 * torch.mean(torch.log(var) + (y_lf_t - mean)**2 / var)
                loss.backward()
                optimizer.step()

            # Fine-tune on HF
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr * 0.1)
            for _ in range(self.finetune_epochs):
                optimizer.zero_grad()
                out = net(X_hf_t)
                mean = out[:, 0]
                log_var = out[:, 1]
                var = F.softplus(log_var) + 1e-6
                loss = 0.5 * torch.mean(torch.log(var) + (y_hf_t - mean)**2 / var)
                loss.backward()
                optimizer.step()

            self.networks.append(net)

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_scaled = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(device)

        means, vars = [], []
        for net in self.networks:
            net.eval()
            with torch.no_grad():
                out = net(X_t)
                mean = out[:, 0].cpu().numpy()
                log_var = out[:, 1].cpu().numpy()
                var = np.exp(log_var) + 1e-6
                means.append(mean)
                vars.append(var)

        means = np.stack(means)
        vars = np.stack(vars)

        ensemble_mean = means.mean(axis=0)
        ensemble_var = vars.mean(axis=0) + means.var(axis=0)

        ensemble_mean = self.scaler_y.inverse_transform(ensemble_mean.reshape(-1, 1)).flatten()
        ensemble_std = np.sqrt(ensemble_var) * self.scaler_y.scale_[0]

        return ensemble_mean, np.maximum(ensemble_std, 1e-6)


class SNGP_MFGP(BaseMFModel):
    """SNGP with MFGP-style fidelity encoding"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_inducing: int = 512,
                 lr: float = 0.001, epochs: int = 150):
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.num_inducing = num_inducing
        self.lr = lr
        self.epochs = epochs
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X_lf: np.ndarray, y_lf: np.ndarray,
            X_hf: np.ndarray, y_hf: np.ndarray):
        n_lf, n_hf = len(X_lf), len(X_hf)

        X_lf_fid = np.hstack([X_lf, np.zeros((n_lf, 1))])
        X_hf_fid = np.hstack([X_hf, np.ones((n_hf, 1))])

        X_all = np.vstack([X_lf_fid, X_hf_fid])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])

        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()

        X_t = torch.FloatTensor(X_scaled).to(device)
        y_t = torch.FloatTensor(y_scaled).to(device)

        # Backbone with spectral norm
        self.backbone = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.input_dim + 1, self.hidden_dim)),
            nn.ReLU()
        ).to(device)

        # RFF
        self.W = torch.randn(self.hidden_dim, self.num_inducing).to(device)
        self.b = torch.rand(self.num_inducing).to(device) * 2 * np.pi
        self.beta = nn.Parameter(torch.zeros(self.num_inducing, 1).to(device))

        optimizer = torch.optim.Adam(
            list(self.backbone.parameters()) + [self.beta], lr=self.lr
        )

        self.backbone.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            h = self.backbone(X_t)
            rff = torch.cos(h @ self.W + self.b) * np.sqrt(2.0 / self.num_inducing)
            mean = rff @ self.beta
            loss = nn.MSELoss()(mean.squeeze(), y_t)
            loss.backward()
            optimizer.step()

        # Compute precision for uncertainty
        self.backbone.eval()
        with torch.no_grad():
            h = self.backbone(X_t)
            rff = torch.cos(h @ self.W + self.b) * np.sqrt(2.0 / self.num_inducing)
            Phi = rff.cpu().numpy()
            self.precision = 0.1 * np.eye(self.num_inducing) + Phi.T @ Phi
            self.cov = np.linalg.inv(self.precision + 1e-6 * np.eye(self.num_inducing))

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_fid = np.hstack([X, np.ones((len(X), 1))])
        X_scaled = self.scaler_x.transform(X_fid)
        X_t = torch.FloatTensor(X_scaled).to(device)

        self.backbone.eval()
        with torch.no_grad():
            h = self.backbone(X_t)
            rff = torch.cos(h @ self.W + self.b) * np.sqrt(2.0 / self.num_inducing)
            mean = (rff @ self.beta).cpu().numpy().flatten()

            Phi = rff.cpu().numpy()
            var = np.sum(Phi @ self.cov * Phi, axis=1) + 1e-6
            std = np.sqrt(var)

        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
        std = std * self.scaler_y.scale_[0]

        return mean, np.maximum(std, 1e-6)


class SNGP_TL(BaseMFModel):
    """SNGP with Transfer Learning"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_inducing: int = 512,
                 lr: float = 0.001, pretrain_epochs: int = 100, finetune_epochs: int = 50):
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.num_inducing = num_inducing
        self.lr = lr
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X_lf: np.ndarray, y_lf: np.ndarray,
            X_hf: np.ndarray, y_hf: np.ndarray):
        X_lf_scaled = self.scaler_x.fit_transform(X_lf)
        y_lf_scaled = self.scaler_y.fit_transform(y_lf.reshape(-1, 1)).flatten()

        X_lf_t = torch.FloatTensor(X_lf_scaled).to(device)
        y_lf_t = torch.FloatTensor(y_lf_scaled).to(device)

        # Backbone
        self.backbone = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(self.input_dim, self.hidden_dim)),
            nn.ReLU()
        ).to(device)

        self.W = torch.randn(self.hidden_dim, self.num_inducing).to(device)
        self.b = torch.rand(self.num_inducing).to(device) * 2 * np.pi
        self.beta = nn.Parameter(torch.zeros(self.num_inducing, 1).to(device))

        # Pretrain
        optimizer = torch.optim.Adam(
            list(self.backbone.parameters()) + [self.beta], lr=self.lr
        )

        self.backbone.train()
        for _ in range(self.pretrain_epochs):
            optimizer.zero_grad()
            h = self.backbone(X_lf_t)
            rff = torch.cos(h @ self.W + self.b) * np.sqrt(2.0 / self.num_inducing)
            mean = rff @ self.beta
            loss = nn.MSELoss()(mean.squeeze(), y_lf_t)
            loss.backward()
            optimizer.step()

        # Fine-tune
        X_hf_scaled = self.scaler_x.transform(X_hf)
        y_hf_scaled = self.scaler_y.transform(y_hf.reshape(-1, 1)).flatten()

        X_hf_t = torch.FloatTensor(X_hf_scaled).to(device)
        y_hf_t = torch.FloatTensor(y_hf_scaled).to(device)

        optimizer = torch.optim.Adam(
            list(self.backbone.parameters()) + [self.beta], lr=self.lr * 0.1
        )

        for _ in range(self.finetune_epochs):
            optimizer.zero_grad()
            h = self.backbone(X_hf_t)
            rff = torch.cos(h @ self.W + self.b) * np.sqrt(2.0 / self.num_inducing)
            mean = rff @ self.beta
            loss = nn.MSELoss()(mean.squeeze(), y_hf_t)
            loss.backward()
            optimizer.step()

        # Compute covariance on HF
        self.backbone.eval()
        with torch.no_grad():
            h = self.backbone(X_hf_t)
            rff = torch.cos(h @ self.W + self.b) * np.sqrt(2.0 / self.num_inducing)
            Phi = rff.cpu().numpy()
            self.precision = 0.1 * np.eye(self.num_inducing) + Phi.T @ Phi
            self.cov = np.linalg.inv(self.precision + 1e-6 * np.eye(self.num_inducing))

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_scaled = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(device)

        self.backbone.eval()
        with torch.no_grad():
            h = self.backbone(X_t)
            rff = torch.cos(h @ self.W + self.b) * np.sqrt(2.0 / self.num_inducing)
            mean = (rff @ self.beta).cpu().numpy().flatten()

            Phi = rff.cpu().numpy()
            var = np.sum(Phi @ self.cov * Phi, axis=1) + 1e-6
            std = np.sqrt(var)

        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
        std = std * self.scaler_y.scale_[0]

        return mean, np.maximum(std, 1e-6)


# =============================================================================
# MODEL REGISTRY
# =============================================================================

MF_MODEL_REGISTRY = {
    # MFGP-style (fidelity correlation)
    'GP_MFGP': GP_MFGP,
    'DNGO_MFGP': DNGO_MFGP,
    'BNN_MFGP': BNN_MFGP,
    'MCDropout_MFGP': MCDropout_MFGP,
    'DeepEnsemble_MFGP': DeepEnsemble_MFGP,
    'SNGP_MFGP': SNGP_MFGP,

    # Joint Training (best in model_comparison: R²=0.78)
    'DNGO_Joint': DNGO_Joint,

    # Transfer Learning (pretrain LF, finetune HF)
    'DNGO_TL': DNGO_TL,
    'BNN_TL': BNN_TL,
    'MCDropout_TL': MCDropout_TL,
    'DeepEnsemble_TL': DeepEnsemble_TL,
    'SNGP_TL': SNGP_TL,
}

# Note: GP_TL doesn't make sense (GP doesn't have pretrain/finetune)
# So we have 11 MF models: 6 MFGP + 5 TL


def create_mf_model(model_name: str, input_dim: int) -> BaseMFModel:
    """Create MF model instance"""
    if model_name not in MF_MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MF_MODEL_REGISTRY.keys())}")

    return MF_MODEL_REGISTRY[model_name](input_dim)


def get_all_mf_models() -> list:
    """Get list of all MF model names"""
    return list(MF_MODEL_REGISTRY.keys())
