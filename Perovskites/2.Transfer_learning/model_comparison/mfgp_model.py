#!/usr/bin/env python
"""
Multi-Fidelity Gaussian Process (MFGP) Model using BoTorch

This implements the standard MFGP approach used in multi-fidelity Bayesian optimization.
"""
import numpy as np
import torch
from typing import Tuple, Optional, Dict

# BoTorch imports
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll


class MultiFidelityGP:
    """
    Multi-Fidelity Gaussian Process model using BoTorch's SingleTaskMultiFidelityGP.

    This model learns correlations between low-fidelity and high-fidelity data
    to make better predictions on high-fidelity targets.
    """

    def __init__(self, input_dim: int, device: str = 'cpu'):
        """
        Initialize MFGP model.

        Args:
            input_dim: Number of input features (excluding fidelity dimension)
            device: 'cpu' or 'cuda'
        """
        self.input_dim = input_dim
        self.device = torch.device(device)
        self.model = None
        self.mll = None
        self.is_fitted = False

        # Store training data for reference
        self.X_train = None
        self.y_train = None

    def _prepare_data(self, X_low: np.ndarray, y_low: np.ndarray,
                      X_high: np.ndarray, y_high: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for MFGP by adding fidelity dimension.

        Low fidelity: fidelity = 0
        High fidelity: fidelity = 1

        Args:
            X_low: Low-fidelity input features [N_lf, d]
            y_low: Low-fidelity targets [N_lf]
            X_high: High-fidelity input features [N_hf, d]
            y_high: High-fidelity targets [N_hf]

        Returns:
            train_X: Combined inputs with fidelity dimension [N_lf + N_hf, d + 1]
            train_Y: Combined targets [N_lf + N_hf, 1]
        """
        X_low = np.asarray(X_low, dtype=np.float64)
        y_low = np.asarray(y_low, dtype=np.float64).flatten()
        X_high = np.asarray(X_high, dtype=np.float64)
        y_high = np.asarray(y_high, dtype=np.float64).flatten()

        n_low = len(X_low)
        n_high = len(X_high)

        # Add fidelity dimension (0 for low, 1 for high)
        fidelity_low = np.zeros((n_low, 1))
        fidelity_high = np.ones((n_high, 1))

        X_low_with_fid = np.hstack([X_low, fidelity_low])
        X_high_with_fid = np.hstack([X_high, fidelity_high])

        # Combine all data
        train_X = np.vstack([X_low_with_fid, X_high_with_fid])
        train_Y = np.concatenate([y_low, y_high]).reshape(-1, 1)

        # Convert to tensors
        train_X = torch.tensor(train_X, dtype=torch.double).to(self.device)
        train_Y = torch.tensor(train_Y, dtype=torch.double).to(self.device)

        return train_X, train_Y

    def fit(self, X_low: np.ndarray, y_low: np.ndarray,
            X_high: np.ndarray, y_high: np.ndarray,
            verbose: bool = False) -> Dict:
        """
        Fit the MFGP model.

        Args:
            X_low: Low-fidelity input features
            y_low: Low-fidelity targets
            X_high: High-fidelity input features
            y_high: High-fidelity targets
            verbose: Print training progress

        Returns:
            Dictionary with training info
        """
        # Prepare data
        train_X, train_Y = self._prepare_data(X_low, y_low, X_high, y_high)

        self.X_train = train_X
        self.y_train = train_Y

        if verbose:
            print(f"    MFGP: Training with {len(X_low)} LF + {len(X_high)} HF samples")

        # Create model
        # The fidelity dimension is the last one (index = input_dim)
        self.model = SingleTaskMultiFidelityGP(
            train_X=train_X,
            train_Y=train_Y,
            outcome_transform=Standardize(m=1),
            data_fidelities=[self.input_dim]  # Index of fidelity dimension
        )
        self.model = self.model.to(self.device)

        # Create MLL
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        # Fit model using BoTorch's fit function
        try:
            fit_gpytorch_mll(self.mll)
            self.is_fitted = True

            if verbose:
                print(f"    MFGP: Model fitted successfully")

        except Exception as e:
            if verbose:
                print(f"    MFGP: Fitting failed with error: {e}")
            raise e

        return {'status': 'success', 'n_low': len(X_low), 'n_high': len(X_high)}

    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict at high-fidelity level.

        Args:
            X: Input features [N, d] (without fidelity dimension)
            return_std: Whether to return standard deviation

        Returns:
            mean: Predicted mean [N]
            std: Predicted standard deviation [N] (if return_std=True)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=np.float64)
        n_test = len(X)

        # Add high-fidelity indicator (fidelity = 1)
        fidelity = np.ones((n_test, 1))
        X_with_fid = np.hstack([X, fidelity])

        X_tensor = torch.tensor(X_with_fid, dtype=torch.double).to(self.device)

        # Get posterior
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X_tensor)
            mean = posterior.mean.cpu().numpy().flatten()

            if return_std:
                variance = posterior.variance.cpu().numpy().flatten()
                std = np.sqrt(variance)
                return mean, std
            else:
                return mean, None

    def predict_with_fidelity(self, X: np.ndarray, fidelity: float = 1.0,
                               return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict at specified fidelity level.

        Args:
            X: Input features [N, d]
            fidelity: Fidelity level (0=low, 1=high)
            return_std: Whether to return standard deviation

        Returns:
            mean: Predicted mean [N]
            std: Predicted standard deviation [N] (if return_std=True)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=np.float64)
        n_test = len(X)

        # Add fidelity indicator
        fidelity_col = np.full((n_test, 1), fidelity)
        X_with_fid = np.hstack([X, fidelity_col])

        X_tensor = torch.tensor(X_with_fid, dtype=torch.double).to(self.device)

        # Get posterior
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X_tensor)
            mean = posterior.mean.cpu().numpy().flatten()

            if return_std:
                variance = posterior.variance.cpu().numpy().flatten()
                std = np.sqrt(variance)
                return mean, std
            else:
                return mean, None


class StandardGP:
    """
    Standard single-fidelity Gaussian Process (for comparison).
    Uses only high-fidelity data.
    """

    def __init__(self, input_dim: int, device: str = 'cpu'):
        self.input_dim = input_dim
        self.device = torch.device(device)
        self.model = None
        self.mll = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> Dict:
        """Fit standard GP on high-fidelity data only."""
        from botorch.models import SingleTaskGP

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten().reshape(-1, 1)

        train_X = torch.tensor(X, dtype=torch.double).to(self.device)
        train_Y = torch.tensor(y, dtype=torch.double).to(self.device)

        if verbose:
            print(f"    Standard GP: Training with {len(X)} HF samples only")

        self.model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            outcome_transform=Standardize(m=1)
        )
        self.model = self.model.to(self.device)

        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        try:
            fit_gpytorch_mll(self.mll)
            self.is_fitted = True
            if verbose:
                print(f"    Standard GP: Model fitted successfully")
        except Exception as e:
            if verbose:
                print(f"    Standard GP: Fitting failed: {e}")
            raise e

        return {'status': 'success', 'n_samples': len(X)}

    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict with standard GP."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=np.float64)
        X_tensor = torch.tensor(X, dtype=torch.double).to(self.device)

        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X_tensor)
            mean = posterior.mean.cpu().numpy().flatten()

            if return_std:
                variance = posterior.variance.cpu().numpy().flatten()
                std = np.sqrt(variance)
                return mean, std
            else:
                return mean, None
