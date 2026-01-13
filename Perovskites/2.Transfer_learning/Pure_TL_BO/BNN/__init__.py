"""
Bayesian Neural Network (BNN) module for Transfer Learning
"""

from .bnn_models import TransferLearningBNN, OnlineTransferLearningBNN
from .optimization_bnn import (
    single_optimization_run_bnn,
    multiple_optimization_runs_bnn,
    train_dual_bnn_models,
    # BNN-OL functions
    single_optimization_run_bnn_ol,
    multiple_optimization_runs_bnn_ol,
    train_bnn_ol_models,
    update_bnn_ol_online,
    recommend_next_bnn_ol
)

__all__ = [
    'TransferLearningBNN',
    'OnlineTransferLearningBNN',
    'single_optimization_run_bnn',
    'multiple_optimization_runs_bnn',
    'train_dual_bnn_models',
    # BNN-OL
    'single_optimization_run_bnn_ol',
    'multiple_optimization_runs_bnn_ol',
    'train_bnn_ol_models',
    'update_bnn_ol_online',
    'recommend_next_bnn_ol'
]