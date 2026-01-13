"""
Model Evaluators for Transfer Learning Comparison

Evaluates DNGO and BNN models on pre-generated datasets.
Uses transfer learning: pretrain on lofi data, finetune on hifi data.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import spearmanr

# Add parent path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR.parent / 'Pure_TL_BO'))

from DNGO.models import TransferLearningDNN, BayesianLinearRegression
from BNN.bnn_models import TransferLearningBNN


@dataclass
class EvaluationMetrics:
    """Metrics for model evaluation"""
    mse: float
    rmse: float
    mae: float
    r2: float
    spearman_corr: float
    uncertainty_calibration: float  # Spearman corr between |error| and uncertainty
    best_prediction: float
    best_actual: float
    coverage_90: float  # % of predictions within 90% CI


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray = None
) -> EvaluationMetrics:
    """Calculate comprehensive evaluation metrics"""
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_errors)

    # R2 score
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Spearman correlation
    spearman_corr, _ = spearmanr(y_true, y_pred)

    # Uncertainty calibration (if uncertainties provided)
    if y_std is not None and len(y_std) > 2:
        unc_calibration, _ = spearmanr(abs_errors, y_std)
        # Coverage: % within 1.645 * std (90% CI for normal distribution)
        within_ci = np.abs(errors) <= 1.645 * y_std
        coverage_90 = np.mean(within_ci) * 100
    else:
        unc_calibration = 0.0
        coverage_90 = 0.0

    # Best predictions
    best_idx = np.argmin(y_pred)

    return EvaluationMetrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        r2=r2,
        spearman_corr=spearman_corr if not np.isnan(spearman_corr) else 0.0,
        uncertainty_calibration=unc_calibration if not np.isnan(unc_calibration) else 0.0,
        best_prediction=y_pred[best_idx],
        best_actual=y_true[best_idx],
        coverage_90=coverage_90
    )


class DNGOEvaluator:
    """Evaluator for DNGO (Deep Networks for Global Optimization)"""

    def __init__(
        self,
        hidden_layers: int = 2,
        hidden_dim: int = 64,
        pretrain_epochs: int = 200,
        finetune_epochs: int = 100,
        pretrain_lr: float = 1e-3,
        finetune_lr: float = 1e-4,
        device: str = 'cuda'
    ):
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.pretrain_lr = pretrain_lr
        self.finetune_lr = finetune_lr
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.model = None
        self.blr = None

    def train(
        self,
        X_low: np.ndarray,
        y_low: np.ndarray,
        X_high: np.ndarray,
        y_high: np.ndarray,
        verbose: bool = False
    ) -> Dict:
        """Train DNGO with transfer learning"""
        input_dim = X_low.shape[1]

        # Create model - TransferLearningDNN uses hidden_dim only (not hidden_layers)
        # hidden_layers is handled internally via _build_default_model
        self.model = TransferLearningDNN(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            device=self.device
        )

        # Pretrain on low-fidelity data
        if verbose:
            print(f"  Pretraining on {len(X_low)} lofi samples...")

        pretrain_loss = self.model.pretrain(
            X_low, y_low,
            epochs=self.pretrain_epochs,
            lr=self.pretrain_lr,
            verbose=False
        )

        # Finetune on high-fidelity data
        if verbose:
            print(f"  Finetuning on {len(X_high)} hifi samples...")

        finetune_loss = self.model.finetune(
            X_high, y_high,
            epochs=self.finetune_epochs,
            lr=self.finetune_lr,
            verbose=False
        )

        # Setup BLR for uncertainty estimation
        features = self.model.extract_features(X_high)
        self.blr = BayesianLinearRegression(alpha=1.0, beta=25.0)
        self.blr.fit(features, y_high)

        return {
            'pretrain_loss': pretrain_loss,
            'finetune_loss': finetune_loss
        }

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimation"""
        features = self.model.extract_features(X)

        means = []
        stds = []
        for i in range(len(X)):
            mean, var = self.blr.predict(features[i])
            means.append(mean)
            stds.append(np.sqrt(max(var, 1e-8)))

        return np.array(means), np.array(stds)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> EvaluationMetrics:
        """Evaluate model on test data"""
        y_pred, y_std = self.predict(X_test)
        return calculate_metrics(y_test, y_pred, y_std)


class BNNEvaluator:
    """
    Evaluator for Bayesian Neural Network using TransferLearningBNN

    Now supports two transfer modes:
    - 'consistent_bnn': Full BNN for both pretrain and finetune (recommended)
    - 'dngo_style': Deterministic feature extractor + BNN head
    """

    def __init__(
        self,
        hidden_layers: int = 2,
        hidden_dim: int = 64,
        pretrain_epochs: int = 200,
        finetune_epochs: int = 100,
        pretrain_lr: float = 1e-3,
        finetune_lr: float = 1e-4,
        kl_weight: float = 1.0,
        device: str = 'cuda',
        transfer_mode: str = 'consistent_bnn',
        prior_pi: float = 0.5,
        prior_sigma1: float = 1.0,
        prior_sigma2: float = 0.002
    ):
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.pretrain_lr = pretrain_lr
        self.finetune_lr = finetune_lr
        self.kl_weight = kl_weight
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.transfer_mode = transfer_mode

        # Scale Mixture Prior parameters (Blundell et al. 2015)
        self.prior_pi = prior_pi
        self.prior_sigma1 = prior_sigma1
        self.prior_sigma2 = prior_sigma2

        self.model = None

    def train(
        self,
        X_low: np.ndarray,
        y_low: np.ndarray,
        X_high: np.ndarray,
        y_high: np.ndarray,
        verbose: bool = False
    ) -> Dict:
        """Train BNN with transfer learning"""
        input_dim = X_low.shape[1]

        # Create hidden_dims list based on hidden_layers and hidden_dim
        hidden_dims = [self.hidden_dim] * self.hidden_layers

        # Create model with new API
        self.model = TransferLearningBNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            device=self.device,
            prior_pi=self.prior_pi,
            prior_sigma1=self.prior_sigma1,
            prior_sigma2=self.prior_sigma2,
            kl_weight=self.kl_weight,
            transfer_mode=self.transfer_mode
        )

        # Pretrain on low-fidelity data
        if verbose:
            print(f"  Pretraining on {len(X_low)} lofi samples (mode: {self.transfer_mode})...")

        self.model.pretrain(
            X_low, y_low,
            epochs=self.pretrain_epochs,
            lr=self.pretrain_lr,
            verbose=False
        )
        pretrain_loss = self.model.pretrain_losses[-1] if self.model.pretrain_losses else 0.0

        # Finetune on high-fidelity data
        if verbose:
            print(f"  Finetuning on {len(X_high)} hifi samples...")

        self.model.finetune(
            X_high, y_high,
            epochs=self.finetune_epochs,
            lr=self.finetune_lr,
            kl_weight=self.kl_weight,
            verbose=False
        )
        finetune_loss = self.model.finetune_losses[-1] if self.model.finetune_losses else 0.0

        return {
            'pretrain_loss': pretrain_loss,
            'finetune_loss': finetune_loss
        }

    def predict(self, X: np.ndarray, n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimation using MC sampling"""
        return self.model.predict(X, n_samples=n_samples)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> EvaluationMetrics:
        """Evaluate model on test data"""
        y_pred, y_std = self.predict(X_test)
        return calculate_metrics(y_test, y_pred, y_std)


def run_evaluation(
    dataset: Dict,
    model_type: str,
    model_config: Dict = None,
    test_all_combinations: bool = True,
    all_X: np.ndarray = None,
    all_y: np.ndarray = None,
    all_y_lofi: np.ndarray = None,
    verbose: bool = False
) -> Dict:
    """
    Run evaluation on a single dataset

    Args:
        dataset: Dataset dictionary with X_low, y_low, X_high, y_high
        model_type: 'dngo' or 'bnn'
        model_config: Model configuration dictionary
        test_all_combinations: If True, test on all combinations
        all_X: All X combinations for testing
        all_y: All y values for testing (HIFI)
        all_y_lofi: All y values for testing (LOFI) - for diagnosing pretrain vs finetune
        verbose: Print progress

    Returns:
        Evaluation results dictionary
    """
    if model_config is None:
        model_config = {}

    # Create evaluator
    if model_type.lower() == 'dngo':
        evaluator = DNGOEvaluator(**model_config)
    elif model_type.lower() == 'bnn':
        evaluator = BNNEvaluator(**model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train
    train_result = evaluator.train(
        dataset['X_low'], dataset['y_low'],
        dataset['X_high'], dataset['y_high'],
        verbose=verbose
    )

    # Evaluate on high-fidelity training data (seen)
    metrics_train = evaluator.evaluate(dataset['X_high'], dataset['y_high'])

    # Evaluate on all combinations (if provided)
    if test_all_combinations and all_X is not None and all_y is not None:
        metrics_all = evaluator.evaluate(all_X, all_y)
    else:
        metrics_all = None

    # Evaluate on LOFI targets (for diagnosing pretrain quality)
    if test_all_combinations and all_X is not None and all_y_lofi is not None:
        metrics_all_lofi = evaluator.evaluate(all_X, all_y_lofi)
    else:
        metrics_all_lofi = None

    # Get predictions for all combinations
    if all_X is not None:
        y_pred_all, y_std_all = evaluator.predict(all_X)
    else:
        y_pred_all, y_std_all = None, None

    return {
        'model_type': model_type,
        'train_result': train_result,
        'metrics_train': metrics_train,
        'metrics_all': metrics_all,
        'metrics_all_lofi': metrics_all_lofi,
        'y_pred_all': y_pred_all,
        'y_std_all': y_std_all,
        'dataset_info': {
            'n_lofi': dataset['n_lofi'],
            'n_hifi': dataset['n_hifi'],
            'actual_cost': dataset['actual_cost']
        }
    }


def run_all_evaluations(
    datasets_dir: str,
    output_dir: str,
    model_configs: Dict[str, Dict] = None,
    verbose: bool = True
) -> Dict:
    """
    Run all evaluations on all datasets

    Args:
        datasets_dir: Directory containing datasets
        output_dir: Directory to save results
        model_configs: Model configurations for each model type
        verbose: Print progress

    Returns:
        All evaluation results
    """
    # Load datasets
    with open(os.path.join(datasets_dir, 'all_datasets.pkl'), 'rb') as f:
        all_datasets = pickle.load(f)

    with open(os.path.join(datasets_dir, 'param_space.json'), 'r') as f:
        param_space = json.load(f)

    with open(os.path.join(datasets_dir, 'label_maps.json'), 'r') as f:
        label_maps = json.load(f)

    # Load lookup table for generating test data
    lookup_path = SCRIPT_DIR.parent.parent / '0.Data' / 'lookup_table.pkl'
    with open(lookup_path, 'rb') as f:
        lookup = pickle.load(f)

    # Generate all combinations for testing
    organics = param_space['organic']
    cations = param_space['cation']
    anions = param_space['anion']

    all_X = []
    all_y = []
    all_y_lofi = []

    for i, org in enumerate(organics, 1):
        for j, cat in enumerate(cations, 1):
            for k, ani in enumerate(anions, 1):
                all_X.append([i, j, k])
                # Get high-fidelity bandgap (HSE06)
                bandgap_hifi = np.amin(lookup[org.capitalize()][cat][ani]['bandgap_hse06'])
                all_y.append(bandgap_hifi)
                # Get low-fidelity bandgap (GGA) for pretrain diagnosis
                bandgap_lofi = np.amin(lookup[org.capitalize()][cat][ani]['bandgap_gga'])
                all_y_lofi.append(bandgap_lofi)

    all_X = np.array(all_X, dtype=np.float32)
    all_y = np.array(all_y, dtype=np.float32)
    all_y_lofi = np.array(all_y_lofi, dtype=np.float32)

    if verbose:
        print(f"Total combinations for testing: {len(all_X)}")
        print(f"HIFI bandgap range: [{all_y.min():.3f}, {all_y.max():.3f}]")
        print(f"LOFI bandgap range: [{all_y_lofi.min():.3f}, {all_y_lofi.max():.3f}]")

    # Default model configs
    if model_configs is None:
        model_configs = {
            'dngo': {
                'hidden_layers': 2,
                'hidden_dim': 64,
                'pretrain_epochs': 200,
                'finetune_epochs': 100,
                'pretrain_lr': 1e-3,
                'finetune_lr': 1e-4
            },
            'bnn': {
                'hidden_layers': 2,
                'hidden_dim': 64,
                'pretrain_epochs': 200,
                'finetune_epochs': 100,
                'pretrain_lr': 1e-3,
                'finetune_lr': 1e-4,
                'kl_weight': 1.0
            }
        }

    # Run evaluations
    all_results = {}

    for config_name, datasets in all_datasets.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating {config_name}")
            print(f"{'='*60}")

        config_results = []

        for set_idx, dataset in enumerate(datasets):
            if verbose:
                print(f"\n  Dataset {set_idx + 1}/{len(datasets)}")

            set_results = {}

            for model_type in ['dngo', 'bnn']:
                if verbose:
                    print(f"    Running {model_type.upper()}...")

                result = run_evaluation(
                    dataset=dataset,
                    model_type=model_type,
                    model_config=model_configs.get(model_type, {}),
                    test_all_combinations=True,
                    all_X=all_X,
                    all_y=all_y,
                    all_y_lofi=all_y_lofi,
                    verbose=False
                )

                set_results[model_type] = result

                if verbose:
                    metrics_hifi = result['metrics_all']
                    metrics_lofi = result['metrics_all_lofi']
                    print(f"      [HIFI] RMSE: {metrics_hifi.rmse:.4f}, R2: {metrics_hifi.r2:.4f}, "
                          f"Spearman: {metrics_hifi.spearman_corr:.4f}")
                    print(f"      [LOFI] RMSE: {metrics_lofi.rmse:.4f}, R2: {metrics_lofi.r2:.4f}, "
                          f"Spearman: {metrics_lofi.spearman_corr:.4f}")

            config_results.append(set_results)

        all_results[config_name] = config_results

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, 'evaluation_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)

    if verbose:
        print(f"\nResults saved to: {results_path}")

    return all_results


if __name__ == '__main__':
    # Run evaluation
    datasets_dir = SCRIPT_DIR / 'datasets'
    output_dir = SCRIPT_DIR / 'results'

    results = run_all_evaluations(
        str(datasets_dir),
        str(output_dir),
        verbose=True
    )
