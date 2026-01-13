"""
Base Uncertainty Quantification Models for Comparison
Implements: Deep Ensemble, SNGP (Spectral-Normalized GP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import math


class SpectralNormLinear(nn.Module):
    """Linear layer with spectral normalization for SNGP"""

    def __init__(self, in_features: int, out_features: int,
                 spectral_norm_bound: float = 0.95, n_power_iterations: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.spectral_norm_bound = spectral_norm_bound
        self.n_power_iterations = n_power_iterations

        # Initialize u and v for power iteration
        self.register_buffer('u', torch.randn(out_features))
        self.register_buffer('v', torch.randn(in_features))

    def _spectral_norm(self):
        """Apply spectral normalization"""
        weight = self.linear.weight.detach()  # Detach to avoid gradient issues
        u = self.u.clone()
        v = self.v.clone()

        for _ in range(self.n_power_iterations):
            v = F.normalize(torch.mv(weight.t(), u), dim=0)
            u = F.normalize(torch.mv(weight, v), dim=0)

        sigma = torch.dot(u, torch.mv(weight, v))

        # Update buffers without gradients
        with torch.no_grad():
            self.u.copy_(u)
            self.v.copy_(v)

        return sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self._spectral_norm()
        if sigma > self.spectral_norm_bound:
            weight = self.linear.weight * (self.spectral_norm_bound / sigma)
        else:
            weight = self.linear.weight
        return F.linear(x, weight, self.linear.bias)


class RandomFourierFeatures(nn.Module):
    """Random Fourier Features for approximating RBF kernel in SNGP"""

    def __init__(self, in_features: int, num_features: int = 1024,
                 length_scale: float = 1.0):
        super().__init__()
        self.num_features = num_features
        self.length_scale = length_scale

        # Random projection matrix
        self.register_buffer(
            'W', torch.randn(in_features, num_features) / length_scale
        )
        self.register_buffer(
            'b', torch.rand(num_features) * 2 * math.pi
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = torch.mm(x, self.W) + self.b
        return torch.cos(proj) * math.sqrt(2.0 / self.num_features)


class SNGPOutputLayer(nn.Module):
    """GP output layer for SNGP with mean-field approximation"""

    def __init__(self, num_features: int, num_outputs: int = 1,
                 ridge_penalty: float = 1.0, momentum: float = 0.999):
        super().__init__()
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.ridge_penalty = ridge_penalty
        self.momentum = momentum

        # Learnable parameters
        self.beta = nn.Parameter(torch.zeros(num_features, num_outputs))

        # Running covariance statistics
        self.register_buffer('precision_matrix',
                           torch.eye(num_features) * ridge_penalty)
        self.register_buffer('covariance_matrix',
                           torch.eye(num_features) / ridge_penalty)

    def reset_precision(self):
        """Reset precision matrix to prior"""
        self.precision_matrix.copy_(torch.eye(self.num_features) * self.ridge_penalty)
        self.covariance_matrix.copy_(torch.eye(self.num_features) / self.ridge_penalty)

    def update_precision(self, features: torch.Tensor):
        """Update precision matrix with new features (for training)"""
        with torch.no_grad():
            batch_precision = torch.mm(features.t(), features)
            self.precision_matrix.mul_(self.momentum).add_(
                batch_precision, alpha=(1 - self.momentum)
            )
            # Update covariance approximation periodically
            try:
                self.covariance_matrix.copy_(
                    torch.linalg.inv(self.precision_matrix +
                                   torch.eye(self.num_features, device=features.device) * 1e-6)
                )
            except:
                pass  # Keep old covariance if inversion fails

    def forward(self, features: torch.Tensor,
                return_variance: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mean = torch.mm(features, self.beta)

        if return_variance:
            # Compute predictive variance
            var = torch.sum(
                torch.mm(features, self.covariance_matrix) * features,
                dim=1, keepdim=True
            )
            var = F.softplus(var) + 1e-6
            return mean, var
        return mean, None


class SNGP(nn.Module):
    """
    Spectral-Normalized Gaussian Process
    Combines spectral normalization with GP output layer for uncertainty
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64],
                 num_rff_features: int = 512, length_scale: float = 1.0,
                 spectral_norm_bound: float = 0.95, ridge_penalty: float = 1.0,
                 dropout_rate: float = 0.1):
        super().__init__()

        # Build spectral-normalized backbone
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(SpectralNormLinear(prev_dim, hidden_dim, spectral_norm_bound))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Random Fourier Features layer
        self.rff = RandomFourierFeatures(prev_dim, num_rff_features, length_scale)

        # GP output layer
        self.gp_output = SNGPOutputLayer(num_rff_features, 1, ridge_penalty)

    def forward(self, x: torch.Tensor,
                return_variance: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        features = self.backbone(x)
        rff_features = self.rff(features)
        return self.gp_output(rff_features, return_variance)

    def fit(self, X: torch.Tensor, y: torch.Tensor,
            epochs: int = 200, lr: float = 0.001, batch_size: int = 32,
            verbose: bool = False):
        """Train the SNGP model"""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.gp_output.reset_precision()

        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()

                # Forward pass
                features = self.backbone(batch_x)
                rff_features = self.rff(features)
                mean, var = self.gp_output(rff_features, return_variance=True)

                # Negative log-likelihood loss
                loss = 0.5 * torch.mean(
                    torch.log(var) + (batch_y - mean)**2 / var
                )

                loss.backward()
                optimizer.step()

                # Update precision matrix
                with torch.no_grad():
                    self.gp_output.update_precision(rff_features.detach())

                total_loss += loss.item()

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    def predict(self, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty"""
        self.eval()
        with torch.no_grad():
            mean, var = self.forward(X, return_variance=True)
            return mean.cpu().numpy(), np.sqrt(var.cpu().numpy())


class EnsembleMember(nn.Module):
    """Single member of the deep ensemble"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64],
                 dropout_rate: float = 0.1):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, 1)
        self.var_head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        mean = self.mean_head(features)
        log_var = self.var_head(features)
        var = F.softplus(log_var) + 1e-6
        return mean, var


class DeepEnsemble(nn.Module):
    """
    Deep Ensemble for uncertainty quantification
    Uses mixture of Gaussians for predictive uncertainty
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64],
                 n_members: int = 5, dropout_rate: float = 0.1):
        super().__init__()
        self.n_members = n_members

        self.members = nn.ModuleList([
            EnsembleMember(input_dim, hidden_dims, dropout_rate)
            for _ in range(n_members)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mixture mean and variance"""
        means = []
        vars = []

        for member in self.members:
            mean, var = member(x)
            means.append(mean)
            vars.append(var)

        means = torch.stack(means, dim=0)  # (n_members, batch, 1)
        vars = torch.stack(vars, dim=0)

        # Mixture of Gaussians: combine epistemic and aleatoric uncertainty
        ensemble_mean = means.mean(dim=0)

        # Total variance = mean of variances + variance of means
        aleatoric = vars.mean(dim=0)
        epistemic = means.var(dim=0)
        ensemble_var = aleatoric + epistemic

        return ensemble_mean, ensemble_var

    def fit(self, X: torch.Tensor, y: torch.Tensor,
            epochs: int = 200, lr: float = 0.001, batch_size: int = 32,
            verbose: bool = False):
        """Train the ensemble with adversarial training for diversity"""
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for i, member in enumerate(self.members):
            member.train()
            optimizer = torch.optim.Adam(member.parameters(), lr=lr)

            for epoch in range(epochs):
                total_loss = 0
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()

                    mean, var = member(batch_x)

                    # Negative log-likelihood loss (heteroscedastic)
                    loss = 0.5 * torch.mean(
                        torch.log(var) + (batch_y - mean)**2 / var
                    )

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if verbose and (epoch + 1) % 50 == 0:
                    print(f"Member {i+1}, Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    def predict(self, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty"""
        self.eval()
        with torch.no_grad():
            mean, var = self.forward(X)
            return mean.cpu().numpy(), np.sqrt(var.cpu().numpy())


class MCDropout(nn.Module):
    """
    MC-Dropout for uncertainty quantification
    Uses dropout at inference time for uncertainty estimation
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64],
                 dropout_rate: float = 0.2, n_samples: int = 50):
        super().__init__()
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.output(features)

    def fit(self, X: torch.Tensor, y: torch.Tensor,
            epochs: int = 200, lr: float = 0.001, batch_size: int = 32,
            verbose: bool = False):
        """Train the MC-Dropout model"""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self.forward(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    def predict(self, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with MC-Dropout uncertainty"""
        self.train()  # Keep dropout active
        predictions = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.forward(X)
                predictions.append(pred.cpu().numpy())

        predictions = np.stack(predictions, axis=0)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)

        return mean, std
