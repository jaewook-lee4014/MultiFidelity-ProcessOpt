#!/usr/bin/env python
"""
CPU vs GPU speed comparison for HP optimization
"""
import time
import torch
import numpy as np
from mf_hyperparameter_optimization import MFHyperparameterOptimizer, get_model_class
from synthetic_functions_mfbo import branin_hf, branin_lf, SCENARIOS

def run_hp_optimization(device_name, n_trials=20):
    """Run HP optimization on specified device"""
    # Force device globally
    import mf_hyperparameter_optimization as mf_hp
    original_device = mf_hp.device
    mf_hp.device = torch.device(device_name)

    # Generate synthetic data
    np.random.seed(42)
    scenario = SCENARIOS['favorable']
    alpha = scenario['alpha_branin']

    # LF data (50 points)
    n_lf = 50
    X_lf = np.random.uniform(0, 1, (n_lf, 2))
    y_lf = branin_lf(X_lf, alpha).flatten()

    # HF data (20 points)
    n_hf = 20
    X_hf = np.random.uniform(0, 1, (n_hf, 2))
    y_hf = branin_hf(X_hf).flatten()

    print(f"\n{'='*50}")
    print(f"Running on {device_name.upper()}")
    print(f"{'='*50}")
    print(f"LF data: {n_lf} points")
    print(f"HF data: {n_hf} points")
    print(f"Trials: {n_trials}")

    # Get model class
    model_class = get_model_class('DNGO_Joint')

    # Create optimizer
    optimizer = MFHyperparameterOptimizer(
        model_name='DNGO_Joint',
        input_dim=2,
        optimize_interval=20,
        n_trials=n_trials
    )

    # Run optimization with timing
    start_time = time.time()
    best_hp = optimizer.optimize(model_class, X_lf, y_lf, X_hf, y_hf)
    elapsed = time.time() - start_time

    best_score = optimizer.best_score

    print(f"\nResults ({device_name}):")
    print(f"  Best score: {best_score:.4f}")
    print(f"  Best params: {best_hp}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Time per trial: {elapsed/n_trials:.3f}s")

    # Restore original device
    mf_hp.device = original_device

    return elapsed, best_score, best_hp

if __name__ == "__main__":
    n_trials = 20

    # Test CPU
    cpu_time, cpu_score, cpu_params = run_hp_optimization('cpu', n_trials)

    # Test GPU (if available)
    if torch.cuda.is_available():
        gpu_time, gpu_score, gpu_params = run_hp_optimization('cuda', n_trials)

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"CPU time: {cpu_time:.2f}s ({cpu_time/n_trials:.3f}s per trial)")
        print(f"GPU time: {gpu_time:.2f}s ({gpu_time/n_trials:.3f}s per trial)")
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        if gpu_time < cpu_time:
            print(f"Speedup: {speedup:.2f}x (GPU faster)")
        else:
            print(f"Speedup: {1/speedup:.2f}x (CPU faster)")
    else:
        print("\nGPU not available for comparison")
