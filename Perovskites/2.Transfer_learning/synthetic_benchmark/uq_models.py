"""
UQ Models for Single-Fidelity Bayesian Optimization

Models (from hp_tuning_base_uq_models.py results):
1. GP (Gaussian Process) - R²=0.8393, Best overall calibration
2. DNGO - R²=0.7900, Over-conservative
3. BNN - R²=0.8307, Good balance
4. MC-Dropout - R²=0.8260, Poor uncertainty (under-confident)
5. Deep Ensemble - R²=0.8406, Best overall
6. SNGP - R²=0.7494, Over-conservative

Uses best hyperparameters from HF-only evaluation on perovskite data.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# 1. GP (Gaussian Process)
# =============================================================================

class GPModel:
    """
    Gaussian Process with Matern kernel

    Best params from HP tuning:
        length_scale: 1.859
        noise_level: 0.0146
        kernel: Matern
        n_restarts: 7
    """

    def __init__(self, length_scale: float = 1.859, noise_level: float = 0.0146,
                 n_restarts: int = 7):
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


# =============================================================================
# 2. DNGO (Deep Networks for Global Optimization)
# =============================================================================

class DNGOModel:
    """
    Deep Networks for Global Optimization

    Best params from HP tuning:
        hidden_dim: 128
        lr: 0.00836
        epochs: 389
        blr_alpha: 0.1016
        blr_beta: 2.685
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 lr: float = 0.00836, epochs: int = 389,
                 blr_alpha: float = 0.1016, blr_beta: float = 2.685):
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
            nn.Linear(self.hidden_dim, 50)  # Feature dim
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

        # Extract features for BLR
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
        var = 1/self.blr_beta + np.sum(Phi @ self.A_inv * Phi, axis=1)
        std = np.sqrt(np.maximum(var, 1e-6))

        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
        std = std * self.scaler_y.scale_[0]

        return mean, std


# =============================================================================
# 3. BNN (Bayesian Neural Network)
# =============================================================================

class BNNModel:
    """
    Bayesian Neural Network with variational inference

    Best params from HP tuning:
        hidden_dim: 128
        num_layers: 2
        lr: 0.0181
        epochs: 330
        kl_weight: 0.239
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, lr: float = 0.0181, epochs: int = 330,
                 kl_weight: float = 0.239, n_samples: int = 20):
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
                for i, w in enumerate(W[:-1]):
                    h = torch.tanh(h @ w)
                pred = h @ W[-1]
                preds.append(pred.cpu().numpy())

        preds = np.stack(preds)
        mean = preds.mean(axis=0).flatten()
        std = preds.std(axis=0).flatten()

        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).flatten()
        std = std * self.scaler_y.scale_[0]

        return mean, np.maximum(std, 1e-6)


# =============================================================================
# 4. MC-Dropout
# =============================================================================

class MCDropoutModel:
    """
    MC-Dropout for uncertainty quantification

    Best params from HP tuning:
        hidden_dim: 256
        num_layers: 1
        dropout: 0.1435
        lr: 0.0162
        epochs: 279
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 num_layers: int = 1, dropout: float = 0.1435,
                 lr: float = 0.0162, epochs: int = 279, n_samples: int = 50):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.n_samples = n_samples
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

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


# =============================================================================
# 5. Deep Ensemble
# =============================================================================

class DeepEnsembleModel:
    """
    Deep Ensemble for uncertainty quantification

    Best params from HP tuning:
        hidden_dim: 64
        num_layers: 1
        n_ensemble: 3
        dropout: 0.0388
        lr: 0.00828
        epochs: 350
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 1, n_ensemble: int = 3,
                 dropout: float = 0.0388, lr: float = 0.00828, epochs: int = 350):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_ensemble = n_ensemble
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def _build_member(self):
        layers = []
        in_dim = self.input_dim
        for _ in range(self.num_layers):
            layers.extend([
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            in_dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, 2))  # mean + log_var
        return nn.Sequential(*layers).to(device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

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
        aleatoric = vars.mean(axis=0)
        epistemic = means.var(axis=0)
        ensemble_var = aleatoric + epistemic

        ensemble_mean = self.scaler_y.inverse_transform(ensemble_mean.reshape(-1, 1)).flatten()
        ensemble_std = np.sqrt(ensemble_var) * self.scaler_y.scale_[0]

        return ensemble_mean, np.maximum(ensemble_std, 1e-6)


# =============================================================================
# 6. SNGP (Spectral-Normalized GP)
# =============================================================================

class SNGPModel:
    """
    Spectral-Normalized Gaussian Process

    Best params from HP tuning:
        hidden_dim: 64
        num_layers: 1
        num_inducing: 512
        spectral_norm_bound: 0.84
        ridge_penalty: 0.1
        length_scale: 1.07
        lr: 0.000139
        epochs: 122
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 1, num_inducing: int = 512,
                 spectral_norm_bound: float = 0.84, ridge_penalty: float = 0.1,
                 length_scale: float = 1.07, lr: float = 0.000139, epochs: int = 122):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_inducing = num_inducing
        self.spectral_norm_bound = spectral_norm_bound
        self.ridge_penalty = ridge_penalty
        self.length_scale = length_scale
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

        # Backbone with spectral normalization
        layers = []
        in_dim = self.input_dim
        for _ in range(self.num_layers):
            layers.extend([
                nn.utils.spectral_norm(nn.Linear(in_dim, self.hidden_dim)),
                nn.ReLU()
            ])
            in_dim = self.hidden_dim
        self.backbone = nn.Sequential(*layers).to(device)

        # Random Fourier Features
        self.W = torch.randn(self.hidden_dim, self.num_inducing).to(device) / self.length_scale
        self.b = torch.rand(self.num_inducing).to(device) * 2 * np.pi

        # Output weight
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

        # Compute precision matrix for uncertainty
        self.backbone.eval()
        with torch.no_grad():
            h = self.backbone(X_t)
            rff = torch.cos(h @ self.W + self.b) * np.sqrt(2.0 / self.num_inducing)
            Phi = rff.cpu().numpy()
            self.precision = self.ridge_penalty * np.eye(self.num_inducing) + Phi.T @ Phi
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

MODEL_REGISTRY = {
    'GP': {
        'class': GPModel,
        'kwargs': {'length_scale': 1.859, 'noise_level': 0.0146, 'n_restarts': 7},
        'requires_input_dim': False
    },
    'DNGO': {
        'class': DNGOModel,
        'kwargs': {'hidden_dim': 128, 'lr': 0.00836, 'epochs': 389,
                  'blr_alpha': 0.1016, 'blr_beta': 2.685},
        'requires_input_dim': True
    },
    'BNN': {
        'class': BNNModel,
        'kwargs': {'hidden_dim': 128, 'num_layers': 2, 'lr': 0.0181,
                  'epochs': 330, 'kl_weight': 0.239, 'n_samples': 20},
        'requires_input_dim': True
    },
    'MC-Dropout': {
        'class': MCDropoutModel,
        'kwargs': {'hidden_dim': 256, 'num_layers': 1, 'dropout': 0.1435,
                  'lr': 0.0162, 'epochs': 279, 'n_samples': 50},
        'requires_input_dim': True
    },
    'Deep Ensemble': {
        'class': DeepEnsembleModel,
        'kwargs': {'hidden_dim': 64, 'num_layers': 1, 'n_ensemble': 3,
                  'dropout': 0.0388, 'lr': 0.00828, 'epochs': 350},
        'requires_input_dim': True
    },
    'SNGP': {
        'class': SNGPModel,
        'kwargs': {'hidden_dim': 64, 'num_layers': 1, 'num_inducing': 512,
                  'spectral_norm_bound': 0.84, 'ridge_penalty': 0.1,
                  'length_scale': 1.07, 'lr': 0.000139, 'epochs': 122},
        'requires_input_dim': True
    }
}


def create_model(model_name: str, input_dim: int):
    """Create model instance with best hyperparameters"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    config = MODEL_REGISTRY[model_name]
    kwargs = config['kwargs'].copy()

    if config['requires_input_dim']:
        kwargs['input_dim'] = input_dim

    return config['class'](**kwargs)
