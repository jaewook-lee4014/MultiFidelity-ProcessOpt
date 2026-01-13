#!/usr/bin/env python3
"""
Hyperparameter Tuning for Base UQ Models (No Transfer Learning)

Compares fundamental uncertainty quantification methods:
1. GP (Gaussian Process) - baseline
2. DNGO (Deep Networks for Global Optimization)
3. BNN (Bayesian Neural Network)
4. MC-Dropout
5. Deep Ensemble
6. SNGP (Spectral Normalized Neural Gaussian Process)

Data Split: HF data only, 6:2:2 (Train:Val:Test)
- Train: 115 samples (~60%)
- Val: 38 samples (~20%) - for HP tuning
- Test: 39 samples (~20%) - final evaluation

Evaluation Metrics:
1. Prediction Quality:
   - R², RMSE, MAE

2. Uncertainty Quality:
   - NLL (Negative Log-Likelihood): Lower is better
   - Calibration Error (ECE): How well uncertainty matches actual errors
   - Sharpness: Average uncertainty (lower = more confident)
   - Spearman Correlation: Uncertainty vs |Error| correlation
   - Coverage@90%: % of predictions within 90% CI

Author: Claude Code
Date: 2025-12-17
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import json
import optuna
from optuna.samplers import TPESampler
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Import base UQ models
from base_uq_models import (
    set_seeds, load_base_data,
    DeepEnsemble, DeepEnsembleMultiFidelity,
    SNGP, SNGPMultiFidelity,
    train_deep_ensemble, train_deep_ensemble_mf,
    train_sngp, train_sngp_mf,
)

# Import existing models
sys.path.append(str(Path(__file__).parent.parent / 'Pure_TL_BO'))
from DNGO.models import TransferLearningDNN, BayesianLinearRegression
from BNN.bnn_models import TransferLearningBNN

# ============================================================================
# Configuration
# ============================================================================

N_OPTUNA_TRIALS = 200
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# HF data split ratio (6:2:2)
HF_TRAIN_RATIO = 0.6
HF_VAL_RATIO = 0.2
HF_TEST_RATIO = 0.2


# ============================================================================
# Uncertainty Evaluation Metrics
# ============================================================================

def calculate_nll(y_true, y_pred, y_std):
    """
    Negative Log-Likelihood (Gaussian assumption)

    NLL = 0.5 * (log(2π) + log(σ²) + (y - μ)²/σ²)

    Lower is better - measures both accuracy and calibration
    """
    y_std = np.maximum(y_std, 1e-6)  # Prevent log(0)
    nll = 0.5 * (np.log(2 * np.pi) + np.log(y_std**2) + ((y_true - y_pred)**2) / (y_std**2))
    return np.mean(nll)


def calculate_calibration_error(y_true, y_pred, y_std, n_bins=10):
    """
    Expected Calibration Error (ECE)

    Measures how well predicted confidence intervals match observed frequencies.
    For regression: checks if p% CI contains p% of true values.

    Lower is better (0 = perfectly calibrated)
    """
    # Calculate z-scores
    y_std = np.maximum(y_std, 1e-6)
    z_scores = np.abs(y_true - y_pred) / y_std

    # Expected confidence levels
    confidence_levels = np.linspace(0.1, 0.99, n_bins)

    calibration_errors = []
    for conf in confidence_levels:
        # Critical value for this confidence level
        z_crit = norm.ppf((1 + conf) / 2)
        # Observed coverage
        observed_coverage = np.mean(z_scores <= z_crit)
        # Calibration error
        calibration_errors.append(np.abs(observed_coverage - conf))

    return np.mean(calibration_errors)


def calculate_sharpness(y_std):
    """
    Sharpness (Average uncertainty)

    Lower is better - but only meaningful if calibration is good.
    A model that always predicts large uncertainty will be well-calibrated
    but not sharp.
    """
    return np.mean(y_std)


def calculate_uncertainty_correlation(y_true, y_pred, y_std):
    """
    Spearman correlation between uncertainty and |error|

    Higher is better - uncertainty should correlate with actual errors.
    A good UQ model should be more uncertain when it makes larger errors.
    """
    abs_errors = np.abs(y_true - y_pred)
    corr, p_value = spearmanr(y_std, abs_errors)
    return corr if not np.isnan(corr) else 0.0


def calculate_coverage(y_true, y_pred, y_std, confidence=0.90):
    """
    Coverage at specified confidence level

    What percentage of true values fall within the predicted CI?
    Should be close to the confidence level for well-calibrated models.
    """
    y_std = np.maximum(y_std, 1e-6)
    z_crit = norm.ppf((1 + confidence) / 2)  # e.g., 1.645 for 90%

    within_ci = np.abs(y_true - y_pred) <= z_crit * y_std
    return np.mean(within_ci) * 100


def calculate_crps(y_true, y_pred, y_std):
    """
    Continuous Ranked Probability Score (CRPS)

    Proper scoring rule that measures both calibration and sharpness.
    For Gaussian predictions: CRPS = σ * [z*Φ(z) + φ(z) - 1/√π]
    where z = (y - μ)/σ, Φ is CDF, φ is PDF

    Lower is better.
    """
    y_std = np.maximum(y_std, 1e-6)
    z = (y_true - y_pred) / y_std

    crps = y_std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return np.mean(crps)


def calculate_all_metrics(y_true, y_pred, y_std):
    """Calculate all evaluation metrics"""
    metrics = {
        # Prediction quality
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),

        # Uncertainty quality
        'nll': calculate_nll(y_true, y_pred, y_std),
        'calibration_error': calculate_calibration_error(y_true, y_pred, y_std),
        'sharpness': calculate_sharpness(y_std),
        'uncertainty_correlation': calculate_uncertainty_correlation(y_true, y_pred, y_std),
        'coverage_90': calculate_coverage(y_true, y_pred, y_std, 0.90),
        'coverage_95': calculate_coverage(y_true, y_pred, y_std, 0.95),
        'crps': calculate_crps(y_true, y_pred, y_std),
    }
    return metrics


# ============================================================================
# Data Generation (HF only, no transfer learning)
# ============================================================================

def generate_hf_data_split(lookup, all_combinations, seed=42):
    """
    Generate HF-only data with 6:2:2 split

    Total: 192 compositions
    - Train: ~115 samples (60%)
    - Val: ~38 samples (20%)
    - Test: ~39 samples (20%)
    """
    set_seeds(seed)
    rng = np.random.default_rng(seed)

    n_total = len(all_combinations)

    # Shuffle all indices
    all_indices = np.arange(n_total)
    rng.shuffle(all_indices)

    n_train = int(n_total * HF_TRAIN_RATIO)
    n_val = int(n_total * HF_VAL_RATIO)

    train_idx = all_indices[:n_train]
    val_idx = all_indices[n_train:n_train + n_val]
    test_idx = all_indices[n_train + n_val:]

    def get_data(indices):
        X, y = [], []
        for idx in indices:
            c = all_combinations[idx]
            X.append(c['labels'])
            y.append(np.amin(lookup[c['names'][0].capitalize()][c['names'][1]][c['names'][2]]['bandgap_hse06']))
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    X_train, y_train = get_data(train_idx)
    X_val, y_val = get_data(val_idx)
    X_test, y_test = get_data(test_idx)

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx
    }


# ============================================================================
# Model Training Functions (No Transfer Learning)
# ============================================================================

def train_gp(params, X_train, y_train):
    """Train Gaussian Process"""
    try:
        length_scale = params.get('length_scale', 1.0)
        noise_level = params.get('noise_level', 0.1)
        kernel_type = params.get('kernel', 'rbf')

        if kernel_type == 'rbf':
            kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
        else:  # matern
            kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=2.5) + WhiteKernel(noise_level=noise_level)

        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=params.get('n_restarts', 5),
            normalize_y=True,
            random_state=42
        )
        gp.fit(X_train, y_train)
        return gp
    except Exception as e:
        return None


def predict_gp(model, X):
    """Predict with GP"""
    y_pred, y_std = model.predict(X, return_std=True)
    return y_pred, y_std


def train_dngo(params, X_train, y_train, device):
    """Train DNGO (without transfer learning)"""
    try:
        model = TransferLearningDNN(
            input_dim=X_train.shape[1],
            hidden_dim=params.get('hidden_dim', 64),
            device=device
        )

        # Direct training on HF data (no pretrain/finetune split)
        model.pretrain(
            X_train, y_train,
            epochs=params.get('epochs', 300),
            lr=params.get('lr', 1e-3),
            verbose=False
        )

        # BLR for uncertainty
        features = model.extract_features(X_train)
        blr = BayesianLinearRegression(
            alpha=params.get('blr_alpha', 1.0),
            beta=params.get('blr_beta', 25.0)
        )
        blr.fit(features, y_train)

        return model, blr
    except:
        return None, None


def predict_dngo(model, blr, X):
    """Predict with DNGO"""
    features = model.extract_features(X)
    y_pred, y_std = [], []
    for i in range(len(X)):
        mean, var = blr.predict(features[i])
        y_pred.append(mean)
        y_std.append(np.sqrt(max(var, 1e-8)))
    return np.array(y_pred), np.array(y_std)


def train_bnn(params, X_train, y_train, device):
    """Train BNN (without transfer learning)"""
    try:
        hidden_dims = [params.get('hidden_dim', 64)] * params.get('num_layers', 2)

        model = TransferLearningBNN(
            input_dim=X_train.shape[1],
            hidden_dims=hidden_dims,
            device=device,
            prior_pi=params.get('prior_pi', 0.5),
            prior_sigma1=params.get('prior_sigma1', 1.0),
            prior_sigma2=params.get('prior_sigma2', 0.002),
            kl_weight=params.get('kl_weight', 1.0),
            transfer_mode='consistent_bnn'
        )

        # Direct training
        model.pretrain(
            X_train, y_train,
            epochs=params.get('epochs', 300),
            lr=params.get('lr', 1e-3),
            verbose=False
        )

        return model
    except:
        return None


def predict_bnn(model, X, n_samples=50):
    """Predict with BNN"""
    return model.predict(X, n_samples=n_samples)


def train_mc_dropout(params, X_train, y_train, device):
    """Train MC-Dropout model"""
    try:
        hidden_dim = params.get('hidden_dim', 64)
        num_layers = params.get('num_layers', 2)
        dropout = params.get('dropout', 0.1)

        layers = []
        in_dim = X_train.shape[1]
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))

        model = nn.Sequential(*layers).to(device)

        optimizer = optim.Adam(model.parameters(), lr=params.get('lr', 1e-3),
                               weight_decay=params.get('weight_decay', 1e-4))
        loss_fn = nn.MSELoss()

        X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

        model.train()
        for _ in range(params.get('epochs', 300)):
            optimizer.zero_grad()
            loss = loss_fn(model(X_t), y_t)
            loss.backward()
            optimizer.step()

        return model
    except:
        return None


def predict_mc_dropout(model, X, device, n_samples=50):
    """Predict with MC-Dropout"""
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    model.train()  # Keep dropout active

    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(X_t).cpu().numpy()
            predictions.append(pred)

    predictions = np.array(predictions)
    y_pred = predictions.mean(axis=0).flatten()
    y_std = predictions.std(axis=0).flatten()

    return y_pred, y_std


def train_deep_ensemble_simple(params, X_train, y_train, device):
    """Train Deep Ensemble (without transfer learning)"""
    try:
        model = DeepEnsemble(
            input_dim=X_train.shape[1],
            hidden_dim=params.get('hidden_dim', 64),
            num_layers=params.get('num_layers', 2),
            n_ensemble=params.get('n_ensemble', 5),
            dropout=params.get('dropout', 0.0),
            activation=params.get('activation', 'relu')
        ).to(device)

        train_deep_ensemble(
            model, X_train, y_train,
            {
                'epochs': params.get('epochs', 300),
                'lr': params.get('lr', 1e-3),
                'weight_decay': params.get('weight_decay', 1e-4),
                'batch_size': params.get('batch_size', 32)
            },
            device
        )

        return model
    except:
        return None


def predict_deep_ensemble_simple(model, X, device):
    """Predict with Deep Ensemble"""
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        y_pred, var = model(X_t)
        y_pred = y_pred.cpu().numpy().flatten()
        y_std = np.sqrt(var.cpu().numpy().flatten())
    return y_pred, y_std


def train_sngp_simple(params, X_train, y_train, device):
    """Train SNGP (without transfer learning)"""
    try:
        model = SNGP(
            input_dim=X_train.shape[1],
            hidden_dim=params.get('hidden_dim', 64),
            num_layers=params.get('num_layers', 2),
            num_inducing=params.get('num_inducing', 256),
            spectral_norm_bound=params.get('spectral_norm_bound', 0.95),
            dropout=params.get('dropout', 0.0),
            activation=params.get('activation', 'relu'),
            ridge_penalty=params.get('ridge_penalty', 1.0),
            length_scale=params.get('length_scale', 1.0)
        ).to(device)

        train_sngp(
            model, X_train, y_train,
            {
                'epochs': params.get('epochs', 300),
                'lr': params.get('lr', 1e-3),
                'weight_decay': params.get('weight_decay', 1e-4)
            },
            device
        )

        return model
    except:
        return None


def predict_sngp_simple(model, X, device):
    """Predict with SNGP"""
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        y_pred, var = model(X_t, return_uncertainty=True)
        y_pred = y_pred.cpu().numpy().flatten()
        y_std = np.sqrt(var.cpu().numpy().flatten())
    return y_pred, y_std


# ============================================================================
# Optuna Objectives
# ============================================================================

def create_gp_objective(data):
    def objective(trial):
        params = {
            'length_scale': trial.suggest_float('length_scale', 0.1, 10.0, log=True),
            'noise_level': trial.suggest_float('noise_level', 1e-4, 1.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'matern']),
            'n_restarts': trial.suggest_int('n_restarts', 1, 10),
        }

        model = train_gp(params, data['X_train'], data['y_train'])
        if model is None:
            return -10.0

        y_pred, y_std = predict_gp(model, data['X_val'])
        if np.isnan(y_pred).any():
            return -10.0

        return r2_score(data['y_val'], y_pred)
    return objective


def create_dngo_objective(data, device):
    def objective(trial):
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128, 256]),
            'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
            'epochs': trial.suggest_int('epochs', 100, 500),
            'blr_alpha': trial.suggest_float('blr_alpha', 0.1, 10.0, log=True),
            'blr_beta': trial.suggest_float('blr_beta', 1.0, 100.0, log=True),
        }

        model, blr = train_dngo(params, data['X_train'], data['y_train'], device)
        if model is None:
            return -10.0

        y_pred, y_std = predict_dngo(model, blr, data['X_val'])
        if np.isnan(y_pred).any():
            return -10.0

        return r2_score(data['y_val'], y_pred)
    return objective


def create_bnn_objective(data, device):
    def objective(trial):
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
            'epochs': trial.suggest_int('epochs', 100, 400),
            'kl_weight': trial.suggest_float('kl_weight', 0.01, 10.0, log=True),
            'prior_pi': trial.suggest_float('prior_pi', 0.1, 0.9),
            'prior_sigma1': trial.suggest_float('prior_sigma1', 0.1, 2.0),
            'prior_sigma2': trial.suggest_float('prior_sigma2', 0.001, 0.1, log=True),
        }

        model = train_bnn(params, data['X_train'], data['y_train'], device)
        if model is None:
            return -10.0

        y_pred, y_std = predict_bnn(model, data['X_val'])
        if np.isnan(y_pred).any():
            return -10.0

        return r2_score(data['y_val'], y_pred)
    return objective


def create_mc_dropout_objective(data, device):
    def objective(trial):
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128, 256]),
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'dropout': trial.suggest_float('dropout', 0.05, 0.5),
            'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
            'epochs': trial.suggest_int('epochs', 100, 500),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        }

        model = train_mc_dropout(params, data['X_train'], data['y_train'], device)
        if model is None:
            return -10.0

        y_pred, y_std = predict_mc_dropout(model, data['X_val'], device)
        if np.isnan(y_pred).any():
            return -10.0

        return r2_score(data['y_val'], y_pred)
    return objective


def create_deep_ensemble_objective(data, device):
    def objective(trial):
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'n_ensemble': trial.suggest_categorical('n_ensemble', [3, 5, 7, 10]),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
            'epochs': trial.suggest_int('epochs', 100, 400),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        }

        model = train_deep_ensemble_simple(params, data['X_train'], data['y_train'], device)
        if model is None:
            return -10.0

        y_pred, y_std = predict_deep_ensemble_simple(model, data['X_val'], device)
        if np.isnan(y_pred).any():
            return -10.0

        return r2_score(data['y_val'], y_pred)
    return objective


def create_sngp_objective(data, device):
    def objective(trial):
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'num_inducing': trial.suggest_categorical('num_inducing', [128, 256, 512]),
            'spectral_norm_bound': trial.suggest_float('spectral_norm_bound', 0.8, 0.99),
            'ridge_penalty': trial.suggest_float('ridge_penalty', 0.1, 10.0, log=True),
            'length_scale': trial.suggest_float('length_scale', 0.1, 5.0),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
            'epochs': trial.suggest_int('epochs', 100, 400),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        }

        model = train_sngp_simple(params, data['X_train'], data['y_train'], device)
        if model is None:
            return -10.0

        y_pred, y_std = predict_sngp_simple(model, data['X_val'], device)
        if np.isnan(y_pred).any():
            return -10.0

        return r2_score(data['y_val'], y_pred)
    return objective


# ============================================================================
# Final Evaluation Functions
# ============================================================================

def final_eval_gp(params, data):
    model = train_gp(params, data['X_train'], data['y_train'])
    y_pred, y_std = predict_gp(model, data['X_test'])
    return y_pred, y_std


def final_eval_dngo(params, data, device):
    model, blr = train_dngo(params, data['X_train'], data['y_train'], device)
    y_pred, y_std = predict_dngo(model, blr, data['X_test'])
    return y_pred, y_std


def final_eval_bnn(params, data, device):
    model = train_bnn(params, data['X_train'], data['y_train'], device)
    y_pred, y_std = predict_bnn(model, data['X_test'])
    return y_pred, y_std


def final_eval_mc_dropout(params, data, device):
    model = train_mc_dropout(params, data['X_train'], data['y_train'], device)
    y_pred, y_std = predict_mc_dropout(model, data['X_test'], device)
    return y_pred, y_std


def final_eval_deep_ensemble(params, data, device):
    model = train_deep_ensemble_simple(params, data['X_train'], data['y_train'], device)
    y_pred, y_std = predict_deep_ensemble_simple(model, data['X_test'], device)
    return y_pred, y_std


def final_eval_sngp(params, data, device):
    model = train_sngp_simple(params, data['X_train'], data['y_train'], device)
    y_pred, y_std = predict_sngp_simple(model, data['X_test'], device)
    return y_pred, y_std


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*70)
    print("Base UQ Models - HP Tuning & Comparison (HF Data Only)")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Optuna Trials: {N_OPTUNA_TRIALS}")
    print(f"Data Split (HF only): {HF_TRAIN_RATIO*100:.0f}%:{HF_VAL_RATIO*100:.0f}%:{HF_TEST_RATIO*100:.0f}%")
    print(f"CV Folds: {len(SEEDS)}")
    print("\nEvaluation Metrics:")
    print("  - Prediction: R², RMSE, MAE")
    print("  - Uncertainty: NLL, Calibration Error, Sharpness, Correlation, Coverage, CRPS")

    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / 'visualizations' / f'{timestamp}_base_uq_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    lookup, all_combinations, _ = load_base_data()
    print(f"Total HF compositions: {len(all_combinations)}")

    # Use first seed for HP tuning
    data = generate_hf_data_split(lookup, all_combinations, seed=42)
    print(f"Train: {len(data['X_train'])}, Val: {len(data['X_val'])}, Test: {len(data['X_test'])}")

    # Models to compare
    models = {
        'GP': {
            'objective_fn': lambda d: create_gp_objective(d),
            'final_eval_fn': final_eval_gp,
            'needs_device': False,
        },
        'DNGO': {
            'objective_fn': lambda d: create_dngo_objective(d, DEVICE),
            'final_eval_fn': lambda p, d: final_eval_dngo(p, d, DEVICE),
            'needs_device': True,
        },
        'BNN': {
            'objective_fn': lambda d: create_bnn_objective(d, DEVICE),
            'final_eval_fn': lambda p, d: final_eval_bnn(p, d, DEVICE),
            'needs_device': True,
        },
        'MC-Dropout': {
            'objective_fn': lambda d: create_mc_dropout_objective(d, DEVICE),
            'final_eval_fn': lambda p, d: final_eval_mc_dropout(p, d, DEVICE),
            'needs_device': True,
        },
        'Deep Ensemble': {
            'objective_fn': lambda d: create_deep_ensemble_objective(d, DEVICE),
            'final_eval_fn': lambda p, d: final_eval_deep_ensemble(p, d, DEVICE),
            'needs_device': True,
        },
        'SNGP': {
            'objective_fn': lambda d: create_sngp_objective(d, DEVICE),
            'final_eval_fn': lambda p, d: final_eval_sngp(p, d, DEVICE),
            'needs_device': True,
        },
    }

    all_results = {}
    best_params_all = {}

    for model_name, config in models.items():
        print(f"\n{'='*70}")
        print(f"Tuning: {model_name}")
        print("="*70)

        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
        )

        objective = config['objective_fn'](data)

        try:
            study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True,
                           catch=(ValueError, RuntimeError))
        except Exception as e:
            print(f"  Error during tuning: {e}")
            continue

        best_params = study.best_params
        best_val_r2 = study.best_value
        best_params_all[model_name] = best_params

        print(f"\nBest Val R²: {best_val_r2:.4f}")
        print(f"Best params: {json.dumps(best_params, indent=2)}")

        # Save Optuna trials
        study_df = study.trials_dataframe()
        study_df.to_csv(output_dir / f'{model_name.lower().replace(" ", "_")}_optuna_trials.csv', index=False)

        # Final evaluation across all folds
        print(f"\nFinal evaluation on TEST set ({len(SEEDS)} folds):")
        fold_results = []

        for fold_idx, seed in enumerate(SEEDS, 1):
            set_seeds(seed)
            fold_data = generate_hf_data_split(lookup, all_combinations, seed=seed)

            try:
                if config['needs_device']:
                    y_pred, y_std = config['final_eval_fn'](best_params, fold_data)
                else:
                    y_pred, y_std = config['final_eval_fn'](best_params, fold_data)

                if np.isnan(y_pred).any() or np.isnan(y_std).any():
                    raise ValueError("NaN in predictions")

                metrics = calculate_all_metrics(fold_data['y_test'], y_pred, y_std)
                metrics['fold'] = fold_idx
                metrics['seed'] = seed
                fold_results.append(metrics)

                print(f"  Fold {fold_idx}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, "
                      f"NLL={metrics['nll']:.4f}, CalErr={metrics['calibration_error']:.4f}, "
                      f"Cov90={metrics['coverage_90']:.1f}%")

            except Exception as e:
                print(f"  Fold {fold_idx}: ERROR - {e}")
                fold_results.append({
                    'fold': fold_idx, 'seed': seed,
                    'r2': np.nan, 'rmse': np.nan, 'mae': np.nan,
                    'nll': np.nan, 'calibration_error': np.nan, 'sharpness': np.nan,
                    'uncertainty_correlation': np.nan, 'coverage_90': np.nan,
                    'coverage_95': np.nan, 'crps': np.nan
                })

        # Aggregate results
        df_folds = pd.DataFrame(fold_results)
        valid_folds = df_folds.dropna()

        if len(valid_folds) > 0:
            summary = {
                'mean_r2': valid_folds['r2'].mean(),
                'std_r2': valid_folds['r2'].std(),
                'mean_rmse': valid_folds['rmse'].mean(),
                'std_rmse': valid_folds['rmse'].std(),
                'mean_mae': valid_folds['mae'].mean(),
                'mean_nll': valid_folds['nll'].mean(),
                'std_nll': valid_folds['nll'].std(),
                'mean_calibration_error': valid_folds['calibration_error'].mean(),
                'mean_sharpness': valid_folds['sharpness'].mean(),
                'mean_uncertainty_correlation': valid_folds['uncertainty_correlation'].mean(),
                'mean_coverage_90': valid_folds['coverage_90'].mean(),
                'mean_coverage_95': valid_folds['coverage_95'].mean(),
                'mean_crps': valid_folds['crps'].mean(),
                'best_val_r2': best_val_r2,
                'n_valid_folds': len(valid_folds),
            }

            print(f"\n{model_name} Summary:")
            print(f"  R² = {summary['mean_r2']:.4f} ± {summary['std_r2']:.4f}")
            print(f"  RMSE = {summary['mean_rmse']:.4f}")
            print(f"  NLL = {summary['mean_nll']:.4f} ± {summary['std_nll']:.4f}")
            print(f"  Calibration Error = {summary['mean_calibration_error']:.4f}")
            print(f"  Coverage@90% = {summary['mean_coverage_90']:.1f}%")
            print(f"  Uncertainty Correlation = {summary['mean_uncertainty_correlation']:.4f}")
        else:
            summary = {k: np.nan for k in ['mean_r2', 'std_r2', 'mean_rmse', 'std_rmse',
                                           'mean_mae', 'mean_nll', 'std_nll',
                                           'mean_calibration_error', 'mean_sharpness',
                                           'mean_uncertainty_correlation',
                                           'mean_coverage_90', 'mean_coverage_95', 'mean_crps']}
            summary['best_val_r2'] = best_val_r2
            summary['n_valid_folds'] = 0

        all_results[model_name] = {
            'summary': summary,
            'fold_results': fold_results,
            'best_params': best_params
        }

        # Save fold results
        df_folds.to_csv(output_dir / f'{model_name.lower().replace(" ", "_")}_fold_results.csv', index=False)

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "="*100)
    print("FINAL SUMMARY (Test Set Results)")
    print("="*100)

    print(f"\n{'Model':<15} {'R²':<15} {'RMSE':<10} {'NLL':<12} {'CalErr':<10} {'Cov90%':<10} {'UncCorr':<10}")
    print("-"*100)

    summary_data = []
    for model_name, results in sorted(all_results.items(), key=lambda x: -x[1]['summary'].get('mean_r2', -999)):
        s = results['summary']
        print(f"{model_name:<15} {s['mean_r2']:.4f}±{s['std_r2']:.4f}  {s['mean_rmse']:.4f}    "
              f"{s['mean_nll']:.4f}±{s['std_nll']:.4f} {s['mean_calibration_error']:.4f}    "
              f"{s['mean_coverage_90']:.1f}%     {s['mean_uncertainty_correlation']:.4f}")

        summary_data.append({
            'model': model_name,
            **s
        })

    # Save summary
    pd.DataFrame(summary_data).to_csv(output_dir / 'summary_results.csv', index=False)

    # Save best params
    with open(output_dir / 'best_params_all.json', 'w') as f:
        json.dump(best_params_all, f, indent=2)

    # Save full results
    with open(output_dir / 'full_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\nResults saved to: {output_dir}")

    # Print metric explanations
    print("\n" + "="*70)
    print("METRIC EXPLANATIONS")
    print("="*70)
    print("""
Prediction Quality:
  - R²: Coefficient of determination (higher is better, max=1)
  - RMSE: Root Mean Squared Error (lower is better)
  - MAE: Mean Absolute Error (lower is better)

Uncertainty Quality:
  - NLL: Negative Log-Likelihood (lower is better)
         Measures both accuracy and calibration jointly

  - Calibration Error: Expected Calibration Error (lower is better, 0=perfect)
         How well predicted confidence intervals match observed frequencies

  - Sharpness: Average predicted uncertainty (lower is better, but only if calibrated)
         A sharp model makes confident predictions

  - Uncertainty Correlation: Spearman corr between |error| and uncertainty (higher is better)
         Good UQ should be more uncertain when making larger errors

  - Coverage@90%: % of true values within 90% CI (ideal=90%)
         Should be close to 90% for well-calibrated models

  - CRPS: Continuous Ranked Probability Score (lower is better)
         Proper scoring rule measuring both calibration and sharpness
""")

    return all_results


if __name__ == '__main__':
    main()
