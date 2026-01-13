#!/usr/bin/env python3
"""
Single-Fidelity Bayesian Optimization Benchmark

Compares 6 UQ models on synthetic test functions:
- Branin-2D
- Park-4D

Evaluation metrics:
- Simple regret: r_t = f(x_best) - f*
- Best found value over iterations
- Convergence curves

Reference: "Best Practices for Multi-Fidelity BO" (Nature Comp Science)
"""

import numpy as np
import torch
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Callable
from scipy.stats import norm, qmc
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Local imports
from synthetic_functions import branin_hf, park_hf, find_global_minimum
from uq_models import create_model, MODEL_REGISTRY


# =============================================================================
# ACQUISITION FUNCTIONS
# =============================================================================

def expected_improvement(X: np.ndarray, model, y_best: float, xi: float = 0.01) -> np.ndarray:
    """
    Expected Improvement acquisition function

    EI(x) = (y_best - μ(x) - ξ) * Φ(z) + σ(x) * φ(z)
    where z = (y_best - μ(x) - ξ) / σ(x)

    Args:
        X: Points to evaluate [N, d]
        model: UQ model with predict(X) -> (mean, std)
        y_best: Best observed value (minimization)
        xi: Exploration-exploitation trade-off

    Returns:
        EI values [N,]
    """
    mean, std = model.predict(X)
    std = np.maximum(std, 1e-8)

    z = (y_best - mean - xi) / std
    ei = (y_best - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
    ei = np.maximum(ei, 0)  # EI is always non-negative

    return ei


def lower_confidence_bound(X: np.ndarray, model, beta: float = 2.0) -> np.ndarray:
    """
    Lower Confidence Bound (for minimization)

    LCB(x) = μ(x) - β * σ(x)
    """
    mean, std = model.predict(X)
    return mean - beta * std


# =============================================================================
# ACQUISITION OPTIMIZATION
# =============================================================================

def optimize_acquisition(acq_func: Callable, bounds: np.ndarray,
                         n_restarts: int = 10, n_random: int = 1000) -> np.ndarray:
    """
    Optimize acquisition function using random search + L-BFGS-B

    Args:
        acq_func: Acquisition function to maximize
        bounds: Search bounds [d, 2]
        n_restarts: Number of local optimization restarts
        n_random: Number of random samples for initialization

    Returns:
        Best point found [d,]
    """
    dim = bounds.shape[0]

    # Random search phase
    X_random = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_random, dim))
    acq_values = acq_func(X_random)
    best_idx = np.argmax(acq_values)
    best_x = X_random[best_idx]
    best_val = acq_values[best_idx]

    # Local optimization from top candidates
    top_idx = np.argsort(acq_values)[-n_restarts:]

    for idx in top_idx:
        try:
            result = minimize(
                lambda x: -acq_func(x.reshape(1, -1))[0],
                X_random[idx],
                method='L-BFGS-B',
                bounds=[(bounds[i, 0], bounds[i, 1]) for i in range(dim)]
            )
            if -result.fun > best_val:
                best_val = -result.fun
                best_x = result.x
        except:
            pass

    return best_x


# =============================================================================
# SINGLE-FIDELITY BO
# =============================================================================

def run_single_fidelity_bo(
    objective: Callable,
    model_name: str,
    bounds: np.ndarray,
    f_star: float,
    n_init: int = 5,
    n_iterations: int = 50,
    seed: int = 0,
    verbose: bool = False
) -> Dict:
    """
    Run single-fidelity Bayesian optimization

    Args:
        objective: Black-box function to minimize
        model_name: Name of UQ model to use
        bounds: Search bounds [d, 2]
        f_star: True global minimum (for regret computation)
        n_init: Number of initial random samples
        n_iterations: Number of BO iterations
        seed: Random seed
        verbose: Print progress

    Returns:
        Dictionary with results:
            - regrets: Simple regret at each iteration
            - best_values: Best found value at each iteration
            - X: All evaluated points
            - y: All function values
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    dim = bounds.shape[0]

    # Initial samples using Latin Hypercube Sampling (LHS)
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    X_init = sampler.random(n=n_init)
    X_init = qmc.scale(X_init, bounds[:, 0], bounds[:, 1])
    y_init = objective(X_init).flatten()

    X = X_init.copy()
    y = y_init.copy()

    # Track metrics
    regrets = [y.min() - f_star]
    best_values = [y.min()]

    for i in range(n_iterations):
        try:
            # Create and fit model
            model = create_model(model_name, dim)
            model.fit(X, y)

            # Optimize acquisition function
            y_best = y.min()
            x_next = optimize_acquisition(
                lambda x: expected_improvement(x, model, y_best),
                bounds,
                n_restarts=5,
                n_random=500
            )

            # Evaluate objective
            y_next = objective(x_next.reshape(1, -1)).flatten()[0]

            # Update data
            X = np.vstack([X, x_next])
            y = np.append(y, y_next)

            # Track metrics
            regrets.append(y.min() - f_star)
            best_values.append(y.min())

            if verbose and (i + 1) % 10 == 0:
                print(f"    Iter {i+1}/{n_iterations}: "
                      f"best={y.min():.4f}, regret={regrets[-1]:.4f}")

        except Exception as e:
            if verbose:
                print(f"    Iter {i+1}: Error - {e}")
            regrets.append(regrets[-1] if regrets else np.inf)
            best_values.append(best_values[-1] if best_values else np.inf)

    return {
        'regrets': np.array(regrets),
        'best_values': np.array(best_values),
        'X': X,
        'y': y,
        'f_star': f_star
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_benchmark(n_seeds: int = 20, n_iterations: int = 50, verbose: bool = True):
    """
    Run full benchmark experiment

    Args:
        n_seeds: Number of random seeds
        n_iterations: Number of BO iterations per run
        verbose: Print progress

    Returns:
        results: Nested dict with all results
        save_dir: Path to saved results
    """
    print("=" * 70)
    print("Single-Fidelity BO Benchmark")
    print("=" * 70)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Seeds: {n_seeds}")
    print(f"Iterations: {n_iterations}")
    print(f"Models: {list(MODEL_REGISTRY.keys())}")
    print()

    # Test functions
    test_functions = {
        'Branin-2D': {
            'func': branin_hf,
            'dim': 2,
            'bounds': np.array([[0, 1], [0, 1]]),
            'f_star': 0.397887  # Known global minimum
        },
        'Park-4D': {
            'func': park_hf,
            'dim': 4,
            'bounds': np.array([[0, 1]] * 4),
            'f_star': None  # Will compute
        }
    }

    # Compute Park-4D minimum
    print("Computing Park-4D global minimum...")
    _, park_min = find_global_minimum(park_hf, 4, n_random=20000, n_local=50)
    test_functions['Park-4D']['f_star'] = park_min
    print(f"  Park-4D f* = {park_min:.4f}")
    print()

    # Run experiments
    results = {}
    all_summaries = []

    for func_name, func_info in test_functions.items():
        print(f"\n{'=' * 70}")
        print(f"Test Function: {func_name} (dim={func_info['dim']}, f*={func_info['f_star']:.4f})")
        print(f"{'=' * 70}")

        results[func_name] = {}

        for model_name in MODEL_REGISTRY.keys():
            print(f"\n  Model: {model_name}")

            all_regrets = []
            all_best_values = []
            successful_seeds = 0

            for seed in range(n_seeds):
                if verbose and seed % 5 == 0:
                    print(f"    Seed {seed}/{n_seeds}...", end="", flush=True)

                try:
                    result = run_single_fidelity_bo(
                        objective=func_info['func'],
                        model_name=model_name,
                        bounds=func_info['bounds'],
                        f_star=func_info['f_star'],
                        n_init=5,
                        n_iterations=n_iterations,
                        seed=seed,
                        verbose=False
                    )
                    all_regrets.append(result['regrets'])
                    all_best_values.append(result['best_values'])
                    successful_seeds += 1

                    if verbose and seed % 5 == 0:
                        print(f" regret={result['regrets'][-1]:.4f}")

                except Exception as e:
                    if verbose and seed % 5 == 0:
                        print(f" FAILED: {e}")

            if all_regrets:
                regrets = np.array(all_regrets)
                best_values = np.array(all_best_values)

                results[func_name][model_name] = {
                    'regrets_mean': regrets.mean(axis=0),
                    'regrets_std': regrets.std(axis=0),
                    'regrets_median': np.median(regrets, axis=0),
                    'best_values_mean': best_values.mean(axis=0),
                    'best_values_std': best_values.std(axis=0),
                    'final_regret_mean': float(regrets[:, -1].mean()),
                    'final_regret_std': float(regrets[:, -1].std()),
                    'final_regret_median': float(np.median(regrets[:, -1])),
                    'n_successful': successful_seeds
                }

                all_summaries.append({
                    'Function': func_name,
                    'Model': model_name,
                    'Final Regret (Mean)': regrets[:, -1].mean(),
                    'Final Regret (Std)': regrets[:, -1].std(),
                    'Final Regret (Median)': np.median(regrets[:, -1]),
                    'Successful Seeds': successful_seeds
                })

                print(f"    Final: regret={regrets[:, -1].mean():.4f} ± {regrets[:, -1].std():.4f} "
                      f"(median={np.median(regrets[:, -1]):.4f}, n={successful_seeds})")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Save raw results
    results_serializable = {}
    for func_name, func_results in results.items():
        results_serializable[func_name] = {}
        for model_name, model_results in func_results.items():
            results_serializable[func_name][model_name] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in model_results.items()
            }

    with open(f"{save_dir}/results.json", 'w') as f:
        json.dump(results_serializable, f, indent=2)

    # Save summary
    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv(f"{save_dir}/summary.csv", index=False)

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for func_name in test_functions:
        print(f"\n{func_name}:")
        print("-" * 60)
        print(f"{'Model':<18} {'Mean Regret':>15} {'Std':>12} {'Median':>12}")
        print("-" * 60)

        if func_name in results:
            sorted_models = sorted(
                results[func_name].items(),
                key=lambda x: x[1]['final_regret_mean']
            )
            for model_name, model_results in sorted_models:
                print(f"{model_name:<18} "
                      f"{model_results['final_regret_mean']:>15.6f} "
                      f"{model_results['final_regret_std']:>12.6f} "
                      f"{model_results['final_regret_median']:>12.6f}")

    print(f"\nResults saved to: {save_dir}")

    # Generate plots
    generate_plots(results, save_dir)

    return results, save_dir


def generate_plots(results: Dict, save_dir: str):
    """Generate convergence plots"""
    print("\nGenerating plots...")

    colors = {
        'GP': '#1f77b4',
        'DNGO': '#ff7f0e',
        'BNN': '#2ca02c',
        'MC-Dropout': '#d62728',
        'Deep Ensemble': '#9467bd',
        'SNGP': '#8c564b'
    }

    for func_name, func_results in results.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Mean regret curves
        ax1 = axes[0]
        for model_name, model_results in func_results.items():
            mean = model_results['regrets_mean']
            std = model_results['regrets_std']
            x = np.arange(len(mean))

            ax1.plot(x, mean, label=model_name, color=colors.get(model_name, 'gray'), linewidth=2)
            ax1.fill_between(x, mean - std, mean + std, alpha=0.2, color=colors.get(model_name, 'gray'))

        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Simple Regret', fontsize=12)
        ax1.set_title(f'{func_name}: Regret Convergence', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Final regret comparison
        ax2 = axes[1]
        models = list(func_results.keys())
        final_means = [func_results[m]['final_regret_mean'] for m in models]
        final_stds = [func_results[m]['final_regret_std'] for m in models]

        # Sort by mean
        sorted_idx = np.argsort(final_means)
        models = [models[i] for i in sorted_idx]
        final_means = [final_means[i] for i in sorted_idx]
        final_stds = [final_stds[i] for i in sorted_idx]

        bars = ax2.barh(models, final_means, xerr=final_stds,
                        color=[colors.get(m, 'gray') for m in models],
                        capsize=5, alpha=0.8)
        ax2.set_xlabel('Final Regret', fontsize=12)
        ax2.set_title(f'{func_name}: Final Regret Comparison', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{func_name.replace(' ', '_').lower()}_convergence.png", dpi=150)
        plt.savefig(f"{save_dir}/{func_name.replace(' ', '_').lower()}_convergence.pdf")
        plt.close()

    # Combined comparison plot
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 6))
    if len(results) == 1:
        axes = [axes]

    for ax, (func_name, func_results) in zip(axes, results.items()):
        models = list(func_results.keys())
        final_means = [func_results[m]['final_regret_mean'] for m in models]
        final_stds = [func_results[m]['final_regret_std'] for m in models]

        sorted_idx = np.argsort(final_means)
        models = [models[i] for i in sorted_idx]
        final_means = [final_means[i] for i in sorted_idx]
        final_stds = [final_stds[i] for i in sorted_idx]

        bars = ax.barh(models, final_means, xerr=final_stds,
                       color=[colors.get(m, 'gray') for m in models],
                       capsize=5, alpha=0.8)
        ax.set_xlabel('Final Regret (lower is better)', fontsize=12)
        ax.set_title(f'{func_name}', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Single-Fidelity BO: Model Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/combined_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{save_dir}/combined_comparison.pdf", bbox_inches='tight')
    plt.close()

    print(f"Plots saved to: {save_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Single-Fidelity BO Benchmark')
    parser.add_argument('--seeds', type=int, default=20, help='Number of random seeds')
    parser.add_argument('--iterations', type=int, default=50, help='Number of BO iterations')
    parser.add_argument('--quiet', action='store_true', help='Reduce output')
    args = parser.parse_args()

    results, save_dir = run_benchmark(
        n_seeds=args.seeds,
        n_iterations=args.iterations,
        verbose=not args.quiet
    )
