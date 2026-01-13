#!/usr/bin/env python3
"""
Test script for Deep Ensemble and SNGP models

Quick test to verify the implementations work correctly
before running full experiments.

Author: Claude Code
Date: 2025-12-17
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from base_uq_models import (
    load_base_data, generate_data, set_seeds,
    DeepEnsemble, DeepEnsembleMultiFidelity,
    SNGP, SNGPMultiFidelity,
    train_deep_ensemble, train_deep_ensemble_mf,
    train_sngp, train_sngp_mf,
    evaluate_uq_model
)


def test_deep_ensemble():
    """Test Deep Ensemble model"""
    print("=" * 60)
    print("Testing Deep Ensemble")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create simple test data
    set_seeds(42)
    X_train = np.random.randn(50, 3).astype(np.float32)
    y_train = np.sin(X_train[:, 0]) + 0.1 * np.random.randn(50).astype(np.float32)
    X_test = np.random.randn(20, 3).astype(np.float32)
    y_test = np.sin(X_test[:, 0])

    # Test single-fidelity Deep Ensemble
    print("\n1. Testing single-fidelity Deep Ensemble...")
    model = DeepEnsemble(
        input_dim=3, hidden_dim=32, num_layers=2,
        n_ensemble=3, dropout=0.0, activation='relu'
    ).to(device)

    train_deep_ensemble(model, X_train, y_train,
                       {'epochs': 50, 'lr': 1e-3}, device)

    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        mean, var = model(X_t)

    print(f"   Mean shape: {mean.shape}")
    print(f"   Var shape: {var.shape}")
    print(f"   Mean predictions: {mean[:3].cpu().numpy().flatten()}")
    print(f"   Uncertainties (std): {torch.sqrt(var[:3]).cpu().numpy().flatten()}")
    print("   ✓ Single-fidelity Deep Ensemble works!")

    return True


def test_deep_ensemble_mf():
    """Test Multi-Fidelity Deep Ensemble"""
    print("\n" + "=" * 60)
    print("Testing Multi-Fidelity Deep Ensemble")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load real data
    print("\nLoading perovskite data...")
    lookup, all_combinations, _ = load_base_data()
    data = generate_data(lookup, all_combinations, n_lofi=72, n_hifi=9, seed=42)

    print(f"LF data: {len(data['X_low'])} samples")
    print(f"HF data: {len(data['X_high'])} samples")

    # Create model
    model = DeepEnsembleMultiFidelity(
        input_dim=3, hidden_dim=64, num_layers=2,
        n_ensemble=5, dropout=0.0, activation='relu'
    ).to(device)

    # Train
    print("\nTraining MF Deep Ensemble...")
    train_deep_ensemble_mf(
        model, data['X_low'], data['y_low'],
        data['X_high'], data['y_high'],
        {'lf_epochs': 100, 'hf_epochs': 50, 'lf_lr': 1e-3, 'hf_lr': 1e-4},
        device
    )

    # Evaluate
    test_mask = np.ones(len(data['X_all']), dtype=bool)
    test_mask[data['hifi_idx']] = False
    X_test = data['X_all'][test_mask]
    y_test = data['y_all'][test_mask]

    results = evaluate_uq_model(model, X_test, y_test, device, 'ensemble_mf')

    print(f"\n   R² Score: {results['r2']:.4f}")
    print(f"   RMSE: {results['rmse']:.4f}")
    print(f"   Mean uncertainty: {np.mean(results['uncertainty']):.4f}")
    print("   ✓ MF Deep Ensemble works!")

    return results


def test_sngp():
    """Test SNGP model"""
    print("\n" + "=" * 60)
    print("Testing SNGP")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create simple test data
    set_seeds(42)
    X_train = np.random.randn(50, 3).astype(np.float32)
    y_train = np.sin(X_train[:, 0]) + 0.1 * np.random.randn(50).astype(np.float32)
    X_test = np.random.randn(20, 3).astype(np.float32)
    y_test = np.sin(X_test[:, 0])

    # Test single-fidelity SNGP
    print("\n1. Testing single-fidelity SNGP...")
    model = SNGP(
        input_dim=3, hidden_dim=32, num_layers=2,
        num_inducing=128, spectral_norm_bound=0.95,
        dropout=0.0, activation='relu',
        ridge_penalty=1.0, length_scale=1.0
    ).to(device)

    train_sngp(model, X_train, y_train,
              {'epochs': 50, 'lr': 1e-3}, device)

    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        mean, var = model(X_t, return_uncertainty=True)

    print(f"   Mean shape: {mean.shape}")
    print(f"   Var shape: {var.shape}")
    print(f"   Mean predictions: {mean[:3].cpu().numpy().flatten()}")
    print(f"   Uncertainties (std): {torch.sqrt(var[:3]).cpu().numpy().flatten()}")
    print("   ✓ Single-fidelity SNGP works!")

    return True


def test_sngp_mf():
    """Test Multi-Fidelity SNGP"""
    print("\n" + "=" * 60)
    print("Testing Multi-Fidelity SNGP")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load real data
    print("\nLoading perovskite data...")
    lookup, all_combinations, _ = load_base_data()
    data = generate_data(lookup, all_combinations, n_lofi=72, n_hifi=9, seed=42)

    # Create model
    model = SNGPMultiFidelity(
        input_dim=3, hidden_dim=64, num_layers=2,
        num_inducing=256, spectral_norm_bound=0.95,
        dropout=0.0, activation='relu',
        ridge_penalty=1.0, length_scale=1.0
    ).to(device)

    # Train
    print("\nTraining MF SNGP...")
    train_sngp_mf(
        model, data['X_low'], data['y_low'],
        data['X_high'], data['y_high'],
        {'lf_epochs': 100, 'hf_epochs': 50, 'lf_lr': 1e-3, 'hf_lr': 1e-4},
        device
    )

    # Evaluate
    test_mask = np.ones(len(data['X_all']), dtype=bool)
    test_mask[data['hifi_idx']] = False
    X_test = data['X_all'][test_mask]
    y_test = data['y_all'][test_mask]

    results = evaluate_uq_model(model, X_test, y_test, device, 'sngp_mf')

    print(f"\n   R² Score: {results['r2']:.4f}")
    print(f"   RMSE: {results['rmse']:.4f}")
    print(f"   Mean uncertainty: {np.mean(results['uncertainty']):.4f}")
    print("   ✓ MF SNGP works!")

    return results


def main():
    print("=" * 60)
    print("Testing New UQ Models: Deep Ensemble & SNGP")
    print("=" * 60)

    # Test all models
    test_deep_ensemble()
    ensemble_results = test_deep_ensemble_mf()

    test_sngp()
    sngp_results = test_sngp_mf()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"\nDeep Ensemble (MF): R²={ensemble_results['r2']:.4f}, RMSE={ensemble_results['rmse']:.4f}")
    print(f"SNGP (MF):          R²={sngp_results['r2']:.4f}, RMSE={sngp_results['rmse']:.4f}")

    print("\n✓ All tests passed!")


if __name__ == '__main__':
    main()
