#!/usr/bin/env python
"""
Hyperparameter Optimization for Multi-Fidelity UQ Models using Optuna (Bayesian Optimization)

Uses 5-fold CV with Optuna TPE sampler for efficient HP search.
Based on model_comparison/hp_tuning_*.py implementations.

Optimization is done on HF data only (LF is auxiliary).
"""

import numpy as np
import torch
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler
from typing import Dict, Tuple, Optional, List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Default number of Optuna trials per optimization
N_OPTUNA_TRIALS = 30  # Balance between thoroughness and speed


# =============================================================================
# OPTUNA OBJECTIVE CREATORS (per model type)
# Based on model_comparison/hp_tuning_base_uq_models.py
# =============================================================================

def create_dngo_objective(model_class, X_lf, y_lf, X_hf, y_hf, input_dim):
    """DNGO objective for Optuna - from hp_tuning_base_uq_models.py"""
    def objective(trial):
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128, 256]),
            'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
            'blr_alpha': trial.suggest_float('blr_alpha', 0.1, 10.0, log=True),
            'blr_beta': trial.suggest_float('blr_beta', 1.0, 100.0, log=True),
        }

        score = kfold_cv_score_mf(model_class, X_lf, y_lf, X_hf, y_hf, params, input_dim)
        return score  # Minimize MSE

    return objective


def create_dngo_joint_objective(model_class, X_lf, y_lf, X_hf, y_hf, input_dim):
    """DNGO_Joint objective for Optuna - from hp_tuning_advanced_tl.py"""
    def objective(trial):
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256]),
            'lr': trial.suggest_float('lr', 1e-4, 5e-2, log=True),
            'alpha': trial.suggest_float('alpha', 0.05, 0.95),  # HF loss weight
            'epochs': trial.suggest_int('epochs', 100, 500),
        }

        score = kfold_cv_score_mf(model_class, X_lf, y_lf, X_hf, y_hf, params, input_dim)
        return score

    return objective


def create_bnn_objective(model_class, X_lf, y_lf, X_hf, y_hf, input_dim):
    """BNN objective for Optuna - from hp_tuning_base_uq_models.py"""
    def objective(trial):
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
            'kl_weight': trial.suggest_float('kl_weight', 0.01, 10.0, log=True),
            'prior_pi': trial.suggest_float('prior_pi', 0.1, 0.9),
            'prior_sigma1': trial.suggest_float('prior_sigma1', 0.1, 2.0),
            'prior_sigma2': trial.suggest_float('prior_sigma2', 0.001, 0.1, log=True),
        }

        score = kfold_cv_score_mf(model_class, X_lf, y_lf, X_hf, y_hf, params, input_dim)
        return score

    return objective


def create_mcdropout_objective(model_class, X_lf, y_lf, X_hf, y_hf, input_dim):
    """MC-Dropout objective for Optuna - from hp_tuning_base_uq_models.py"""
    def objective(trial):
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128, 256]),
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'dropout': trial.suggest_float('dropout', 0.05, 0.5),
            'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        }

        score = kfold_cv_score_mf(model_class, X_lf, y_lf, X_hf, y_hf, params, input_dim)
        return score

    return objective


def create_deep_ensemble_objective(model_class, X_lf, y_lf, X_hf, y_hf, input_dim):
    """Deep Ensemble objective for Optuna - from hp_tuning_base_uq_models.py"""
    def objective(trial):
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'n_ensemble': trial.suggest_categorical('n_ensemble', [3, 5, 7, 10]),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3),
            'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        }

        score = kfold_cv_score_mf(model_class, X_lf, y_lf, X_hf, y_hf, params, input_dim)
        return score

    return objective


def create_sngp_objective(model_class, X_lf, y_lf, X_hf, y_hf, input_dim):
    """SNGP objective for Optuna - from hp_tuning_base_uq_models.py"""
    def objective(trial):
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'num_inducing': trial.suggest_categorical('num_inducing', [128, 256, 512]),
            'spectral_norm_bound': trial.suggest_float('spectral_norm_bound', 0.8, 0.99),
            'ridge_penalty': trial.suggest_float('ridge_penalty', 0.1, 10.0, log=True),
            'length_scale': trial.suggest_float('length_scale', 0.1, 5.0),
            'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        }

        score = kfold_cv_score_mf(model_class, X_lf, y_lf, X_hf, y_hf, params, input_dim)
        return score

    return objective


# Mapping from model type to objective creator
OBJECTIVE_CREATORS = {
    'DNGO': create_dngo_objective,
    'DNGO_Joint': create_dngo_joint_objective,
    'BNN': create_bnn_objective,
    'MCDropout': create_mcdropout_objective,
    'DeepEnsemble': create_deep_ensemble_objective,
    'SNGP': create_sngp_objective,
}


def get_model_type(model_name: str) -> str:
    """Extract base model type from full name"""
    if model_name.startswith('GP_'):
        return 'GP'
    elif model_name == 'DNGO_Joint':
        return 'DNGO_Joint'
    elif model_name.startswith('DNGO_'):
        return 'DNGO'
    elif model_name.startswith('BNN_'):
        return 'BNN'
    elif model_name.startswith('MCDropout_'):
        return 'MCDropout'
    elif model_name.startswith('DeepEnsemble_'):
        return 'DeepEnsemble'
    elif model_name.startswith('SNGP_'):
        return 'SNGP'
    else:
        raise ValueError(f"Unknown model: {model_name}")


def is_tl_model(model_name: str) -> bool:
    """Check if model uses Transfer Learning approach"""
    return model_name.endswith('_TL')


# =============================================================================
# 5-FOLD CV EVALUATION FOR MF MODELS
# =============================================================================

def kfold_cv_score_mf(model_class, X_lf: np.ndarray, y_lf: np.ndarray,
                      X_hf: np.ndarray, y_hf: np.ndarray,
                      hp_config: Dict, input_dim: int, n_folds: int = 5) -> float:
    """
    Compute 5-fold CV score on HF data for MF model

    For MF models, we do k-fold CV on HF data while keeping all LF data.
    Much faster than LOOCV (5 models instead of N models).
    """
    n_hf = len(X_hf)
    if n_hf < n_folds:
        return float('inf')

    errors = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X_hf):
        X_hf_train = X_hf[train_idx]
        y_hf_train = y_hf[train_idx]
        X_hf_test = X_hf[test_idx]
        y_hf_test = y_hf[test_idx]

        try:
            # Create model with current HP config
            model = model_class(input_dim, **hp_config)
            model.fit(X_lf, y_lf, X_hf_train, y_hf_train)

            mean, _ = model.predict(X_hf_test)
            fold_errors = (mean - y_hf_test) ** 2
            errors.extend(fold_errors.tolist())
        except Exception:
            errors.append(float('inf'))

    return np.mean(errors)


def bayesian_optimize_hp(model_class, model_name: str,
                         X_lf: np.ndarray, y_lf: np.ndarray,
                         X_hf: np.ndarray, y_hf: np.ndarray,
                         input_dim: int,
                         n_trials: int = N_OPTUNA_TRIALS) -> Tuple[Dict, float]:
    """
    Bayesian Optimization with Optuna TPE sampler for MF model hyperparameters

    Returns: (best_config, best_score)
    """
    model_type = get_model_type(model_name)

    # GP doesn't need HP optimization (uses MLL)
    if model_type == 'GP':
        return {}, 0.0

    # Get objective creator for this model type
    objective_creator = OBJECTIVE_CREATORS.get(model_type)
    if objective_creator is None:
        return {}, float('inf')

    # Create Optuna study with TPE sampler (Bayesian optimization)
    study = optuna.create_study(
        direction='minimize',  # Minimize MSE
        sampler=TPESampler(seed=42)
    )

    # Create objective function
    objective = objective_creator(model_class, X_lf, y_lf, X_hf, y_hf, input_dim)

    # Run optimization
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                       catch=(ValueError, RuntimeError))
    except Exception:
        pass

    if len(study.trials) == 0 or study.best_trial is None:
        return {}, float('inf')

    return study.best_params, study.best_value


# =============================================================================
# HP OPTIMIZER CLASS
# =============================================================================

class MFHyperparameterOptimizer:
    """
    Manages hyperparameter optimization for MF models using Optuna (Bayesian Optimization).

    Triggers optimization every `optimize_interval` HF data points.
    Uses TPE (Tree-structured Parzen Estimator) for efficient search.
    Stores optimization history for analysis.
    """

    def __init__(self, model_name: str, input_dim: int,
                 optimize_interval: int = 20, n_trials: int = N_OPTUNA_TRIALS):
        self.model_name = model_name
        self.input_dim = input_dim
        self.optimize_interval = optimize_interval
        self.n_trials = n_trials
        self.model_type = get_model_type(model_name)
        self.is_tl = is_tl_model(model_name)

        self.best_hp = self._get_default_hp()
        self.best_score = float('inf')
        self.last_optimize_n = 0

        # History for analysis
        self.hp_history = []  # List of (n_hf, hp_config, cv_score)

    def _get_default_hp(self) -> Dict:
        """Get default hyperparameters for model type (from model_comparison results)"""
        defaults = {
            'GP': {},
            'DNGO': {'hidden_dim': 64, 'lr': 0.001, 'blr_alpha': 1.0, 'blr_beta': 25.0},
            'DNGO_Joint': {'hidden_dim': 64, 'lr': 0.001, 'alpha': 0.2, 'epochs': 300},
            'BNN': {'hidden_dim': 64, 'num_layers': 2, 'lr': 0.001, 'kl_weight': 1.0,
                    'prior_pi': 0.5, 'prior_sigma1': 1.0, 'prior_sigma2': 0.002},
            'MCDropout': {'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.1, 'lr': 0.001,
                          'weight_decay': 1e-4},
            'DeepEnsemble': {'hidden_dim': 64, 'num_layers': 2, 'n_ensemble': 5, 'lr': 0.001,
                             'dropout': 0.0, 'weight_decay': 1e-4},
            'SNGP': {'hidden_dim': 64, 'num_layers': 2, 'num_inducing': 256, 'lr': 0.001,
                     'spectral_norm_bound': 0.95, 'ridge_penalty': 1.0, 'length_scale': 1.0,
                     'weight_decay': 1e-4},
        }
        return defaults.get(self.model_type, {})

    def should_optimize(self, n_hf: int) -> bool:
        """Check if HP optimization should be triggered"""
        if self.model_type == 'GP':
            return False
        return (n_hf - self.last_optimize_n) >= self.optimize_interval

    def optimize(self, model_class, X_lf: np.ndarray, y_lf: np.ndarray,
                 X_hf: np.ndarray, y_hf: np.ndarray) -> Dict:
        """
        Run Bayesian HP optimization with Optuna and return best config
        """
        if self.model_type == 'GP':
            return {}

        best_hp, best_score = bayesian_optimize_hp(
            model_class, self.model_name,
            X_lf, y_lf, X_hf, y_hf,
            self.input_dim, self.n_trials
        )

        n_hf = len(X_hf)
        if best_hp:
            self.best_hp = best_hp
            self.best_score = best_score

        # Record history
        self.hp_history.append({
            'n_hf': n_hf,
            'hp_config': self.best_hp.copy(),
            'cv_score': best_score
        })

        self.last_optimize_n = n_hf
        return self.best_hp

    def get_hp(self) -> Dict:
        """Get current best hyperparameters"""
        return self.best_hp

    def get_history(self) -> List[Dict]:
        """Get optimization history"""
        return self.hp_history

    def get_summary(self) -> Dict:
        """Get summary of HP optimization"""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'final_hp': self.best_hp,
            'final_cv_score': self.best_score,
            'n_optimizations': len(self.hp_history),
            'history': self.hp_history
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_mf_model_with_hp(model_name: str, input_dim: int,
                            hp_config: Dict = None):
    """Create MF model with specified hyperparameters"""
    from mf_uq_models import MF_MODEL_REGISTRY

    model_class = MF_MODEL_REGISTRY[model_name]

    if hp_config:
        return model_class(input_dim, **hp_config)
    else:
        return model_class(input_dim)


def get_model_class(model_name: str):
    """Get model class from registry"""
    from mf_uq_models import MF_MODEL_REGISTRY
    return MF_MODEL_REGISTRY[model_name]
