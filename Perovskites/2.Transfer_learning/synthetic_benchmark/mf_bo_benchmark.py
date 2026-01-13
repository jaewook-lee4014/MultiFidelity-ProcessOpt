#!/usr/bin/env python
"""
Multi-Fidelity Bayesian Optimization Benchmark

Based on: "Best Practices for Multi-Fidelity Bayesian Optimization" (Nature Comp Science)

Test Function: Branin-2D only (Park excluded due to R² constraints)

Scenarios (from paper):
1. Favorable: α=0.8, ρ=0.1 (LF cheap & informative, R² ≈ 0.97)
2. Unfavorable: α=0.1, ρ=0.5 (LF expensive & less informative, R² ≈ 0.56)

Cost Model (from paper):
- HF cost = 1 (always)
- LF cost = ρ (0.1 for favorable, 0.5 for unfavorable)

MF Models (6 MFGP + 5 TL = 11 total):
1. GP_MFGP (no HP optimization needed - uses MLL)
2. DNGO_MFGP, DNGO_TL
3. BNN_MFGP, BNN_TL
4. MCDropout_MFGP, MCDropout_TL
5. DeepEnsemble_MFGP, DeepEnsemble_TL
6. SNGP_MFGP, SNGP_TL

HP Optimization:
- LOOCV-based optimization every 20 HF data points
- Each model type has different HP to optimize
- GP_MFGP uses automatic MLL optimization (no manual HP tuning)
"""

import numpy as np
import torch
import json
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Tuple, List, Dict
from scipy.stats import norm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import multiprocessing as mp

# Local imports
from synthetic_functions_mfbo import (
    branin_hf, branin_lf, SCENARIOS, FUNCTIONS
)
from mf_uq_models import create_mf_model, get_all_mf_models, MF_MODEL_REGISTRY
from mf_hyperparameter_optimization import (
    MFHyperparameterOptimizer, create_mf_model_with_hp, get_model_class
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Default budget
TOTAL_BUDGET = 50
HP_OPTIMIZE_INTERVAL = 20  # Optimize HP every 20 HF data points


def expected_improvement(mean: np.ndarray, std: np.ndarray,
                         y_best: float, xi: float = 0.01) -> np.ndarray:
    """Expected Improvement acquisition function"""
    with np.errstate(divide='ignore', invalid='ignore'):
        imp = y_best - mean - xi
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std < 1e-10] = 0.0
    return ei


def compute_prediction_metrics(model, X: np.ndarray, y_true: np.ndarray) -> Dict:
    """
    Compute prediction accuracy metrics (R², MSE, MAE)

    Args:
        model: Trained model with predict() method
        X: Input features
        y_true: Ground truth values

    Returns:
        Dictionary with r2, mse, mae
    """
    try:
        mean, std = model.predict(X)
        mean = mean.flatten()
        y_true = y_true.flatten()

        r2 = r2_score(y_true, mean)
        mse = mean_squared_error(y_true, mean)
        mae = mean_absolute_error(y_true, mean)

        # Also compute mean uncertainty (std)
        mean_std = np.mean(std)

        return {
            'r2': float(r2),
            'mse': float(mse),
            'mae': float(mae),
            'mean_std': float(mean_std)
        }
    except Exception as e:
        return {
            'r2': float('nan'),
            'mse': float('nan'),
            'mae': float('nan'),
            'mean_std': float('nan')
        }


def run_mf_bo_single(seed: int, f_hf: Callable, f_lf: Callable,
                     model_name: str, bounds: np.ndarray, f_star: float,
                     alpha: float, rho: float,
                     n_init_lf: int = 5, n_init_hf: int = 2,
                     total_budget: float = TOTAL_BUDGET,
                     hp_optimize_interval: int = HP_OPTIMIZE_INTERVAL,
                     n_test: int = 200) -> Dict:
    """
    Run single MF BO experiment with online HP optimization
    Now also tracks prediction accuracy metrics (R², MSE, MAE) on train and test data
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    dim = bounds.shape[0]
    n_candidates = 1000

    # Cost model from paper: HF = 1, LF = rho
    cost_hf = 1.0
    cost_lf = rho

    # Generate fixed test set (HF values only, for fair comparison)
    # Use different seed range to avoid overlap with training data
    np.random.seed(seed + 10000)
    X_test = np.random.uniform(0, 1, (n_test, dim))
    y_test = f_hf(X_test).flatten()
    np.random.seed(seed)  # Reset seed for reproducibility

    # Initial cost
    current_budget = n_init_lf * cost_lf + n_init_hf * cost_hf

    # Initial samples
    X_lf = np.random.uniform(0, 1, (n_init_lf, dim))
    y_lf = f_lf(X_lf, alpha).flatten()

    X_hf = np.random.uniform(0, 1, (n_init_hf, dim))
    y_hf = f_hf(X_hf).flatten()

    # Track regret over budget
    budget_history = [current_budget]
    regret_history = [max(0, y_hf.min() - f_star)]

    # Track prediction metrics over budget
    train_metrics_history = []  # Metrics on HF training data
    test_metrics_history = []   # Metrics on held-out test data

    # HP optimizer
    hp_optimizer = MFHyperparameterOptimizer(
        model_name, dim, optimize_interval=hp_optimize_interval
    )
    model_class = get_model_class(model_name)

    # BO loop
    max_iterations = 500
    iteration = 0
    last_hp_optimize_n = 0
    model = None  # Keep track of last model for metrics

    while current_budget < total_budget and iteration < max_iterations:
        iteration += 1

        remaining = total_budget - current_budget

        if remaining >= cost_hf:
            if remaining >= cost_lf and iteration % 2 == 0:
                eval_hf = False
                cost = cost_lf
            else:
                eval_hf = True
                cost = cost_hf
        elif remaining >= cost_lf:
            eval_hf = False
            cost = cost_lf
        else:
            break

        try:
            # Check if HP optimization should be triggered (only on HF data increase)
            n_hf_current = len(X_hf)
            if hp_optimizer.should_optimize(n_hf_current):
                hp_optimizer.optimize(model_class, X_lf, y_lf, X_hf, y_hf)

            # Create model with current best HP
            current_hp = hp_optimizer.get_hp()
            if current_hp:
                model = create_mf_model_with_hp(model_name, dim, current_hp)
            else:
                model = create_mf_model(model_name, dim)

            model.fit(X_lf, y_lf, X_hf, y_hf)

            # Compute prediction metrics on train (HF) and test data
            train_metrics = compute_prediction_metrics(model, X_hf, y_hf)
            test_metrics = compute_prediction_metrics(model, X_test, y_test)
            train_metrics_history.append(train_metrics)
            test_metrics_history.append(test_metrics)

            X_cand = np.random.uniform(0, 1, (n_candidates, dim))
            mean, std = model.predict(X_cand)

            y_best = y_hf.min()
            ei = expected_improvement(mean, std, y_best)

            best_idx = np.argmax(ei)
            x_new = X_cand[best_idx:best_idx+1]

        except Exception as e:
            x_new = np.random.uniform(0, 1, (1, dim))
            # Record NaN metrics on failure
            nan_metrics = {'r2': float('nan'), 'mse': float('nan'), 'mae': float('nan'), 'mean_std': float('nan')}
            train_metrics_history.append(nan_metrics)
            test_metrics_history.append(nan_metrics)

        if eval_hf:
            y_new = f_hf(x_new).flatten()
            X_hf = np.vstack([X_hf, x_new])
            y_hf = np.append(y_hf, y_new)
        else:
            y_new = f_lf(x_new, alpha).flatten()
            X_lf = np.vstack([X_lf, x_new])
            y_lf = np.append(y_lf, y_new)

        current_budget += cost
        budget_history.append(current_budget)
        regret_history.append(max(0, y_hf.min() - f_star))

    # Extract final metrics
    final_train_metrics = train_metrics_history[-1] if train_metrics_history else {}
    final_test_metrics = test_metrics_history[-1] if test_metrics_history else {}

    return {
        'seed': seed,
        'budget_history': budget_history,
        'regret_history': regret_history,
        'final_regret': regret_history[-1],
        'n_lf': len(X_lf),
        'n_hf': len(X_hf),
        'total_cost': current_budget,
        'hp_summary': hp_optimizer.get_summary(),
        # New: prediction metrics history
        'train_metrics_history': train_metrics_history,
        'test_metrics_history': test_metrics_history,
        'final_train_metrics': final_train_metrics,
        'final_test_metrics': final_test_metrics
    }


def run_single_model_scenario(args):
    """Run a single model on a single scenario (for thread pool)"""
    model_name, scenario_name, scenario, f_hf, f_lf, bounds, f_star, n_seeds, total_budget = args

    alpha = scenario['alpha_branin']
    rho = scenario['rho']

    results = []
    for seed in range(n_seeds):
        try:
            result = run_mf_bo_single(
                seed=seed,
                f_hf=f_hf,
                f_lf=f_lf,
                model_name=model_name,
                bounds=bounds,
                f_star=f_star,
                alpha=alpha,
                rho=rho,
                total_budget=total_budget
            )
            results.append(result)
        except Exception as e:
            print(f"      {model_name}/{scenario_name}/seed{seed} failed: {e}")

    if not results:
        return None

    # Aggregate results
    max_budget = max(r['budget_history'][-1] for r in results)
    budget_points = np.arange(0, max_budget + 0.1, 0.1).tolist()

    all_regrets = []
    for r in results:
        regrets_interp = np.interp(budget_points, r['budget_history'], r['regret_history'])
        all_regrets.append(regrets_interp)

    all_regrets = np.array(all_regrets)
    regrets_mean = all_regrets.mean(axis=0)
    regrets_std = all_regrets.std(axis=0)

    final_regrets = [r['final_regret'] for r in results]
    n_lf_avg = np.mean([r['n_lf'] for r in results])
    n_hf_avg = np.mean([r['n_hf'] for r in results])

    # Collect HP summaries
    hp_summaries = [r['hp_summary'] for r in results]

    # Aggregate prediction metrics (Train and Test)
    # Interpolate metrics history to common budget points for averaging
    def aggregate_metrics_history(results, metric_key, budget_points):
        """Aggregate a specific metric across all seeds"""
        all_r2 = []
        all_mse = []
        all_mae = []
        all_std = []

        for r in results:
            metrics_hist = r.get(metric_key, [])
            if not metrics_hist:
                continue

            # Create budget points for this run (excluding initial)
            # metrics are recorded starting from iteration 1
            run_budget = r['budget_history'][1:]  # Skip initial budget
            if len(run_budget) != len(metrics_hist):
                # Align lengths
                min_len = min(len(run_budget), len(metrics_hist))
                run_budget = run_budget[:min_len]
                metrics_hist = metrics_hist[:min_len]

            r2_vals = [m['r2'] for m in metrics_hist]
            mse_vals = [m['mse'] for m in metrics_hist]
            mae_vals = [m['mae'] for m in metrics_hist]
            std_vals = [m['mean_std'] for m in metrics_hist]

            # Interpolate to common budget points
            r2_interp = np.interp(budget_points, run_budget, r2_vals)
            mse_interp = np.interp(budget_points, run_budget, mse_vals)
            mae_interp = np.interp(budget_points, run_budget, mae_vals)
            std_interp = np.interp(budget_points, run_budget, std_vals)

            all_r2.append(r2_interp)
            all_mse.append(mse_interp)
            all_mae.append(mae_interp)
            all_std.append(std_interp)

        if not all_r2:
            return None

        return {
            'r2_mean': np.nanmean(all_r2, axis=0).tolist(),
            'r2_std': np.nanstd(all_r2, axis=0).tolist(),
            'mse_mean': np.nanmean(all_mse, axis=0).tolist(),
            'mse_std': np.nanstd(all_mse, axis=0).tolist(),
            'mae_mean': np.nanmean(all_mae, axis=0).tolist(),
            'mae_std': np.nanstd(all_mae, axis=0).tolist(),
            'uncertainty_mean': np.nanmean(all_std, axis=0).tolist(),
            'uncertainty_std': np.nanstd(all_std, axis=0).tolist(),
        }

    train_metrics_agg = aggregate_metrics_history(results, 'train_metrics_history', budget_points)
    test_metrics_agg = aggregate_metrics_history(results, 'test_metrics_history', budget_points)

    # Final metrics (averaged across seeds)
    final_train_r2 = np.nanmean([r['final_train_metrics'].get('r2', np.nan) for r in results])
    final_train_mse = np.nanmean([r['final_train_metrics'].get('mse', np.nan) for r in results])
    final_train_mae = np.nanmean([r['final_train_metrics'].get('mae', np.nan) for r in results])

    final_test_r2 = np.nanmean([r['final_test_metrics'].get('r2', np.nan) for r in results])
    final_test_mse = np.nanmean([r['final_test_metrics'].get('mse', np.nan) for r in results])
    final_test_mae = np.nanmean([r['final_test_metrics'].get('mae', np.nan) for r in results])

    # Std of final metrics
    final_train_r2_std = np.nanstd([r['final_train_metrics'].get('r2', np.nan) for r in results])
    final_test_r2_std = np.nanstd([r['final_test_metrics'].get('r2', np.nan) for r in results])

    return {
        'model_name': model_name,
        'scenario_name': scenario_name,
        'budget_points': budget_points,
        'regrets_mean': regrets_mean.tolist(),
        'regrets_std': regrets_std.tolist(),
        'final_regret_mean': np.mean(final_regrets),
        'final_regret_std': np.std(final_regrets),
        'n_successful': len(results),
        'n_lf_avg': n_lf_avg,
        'n_hf_avg': n_hf_avg,
        'alpha': alpha,
        'rho': rho,
        'hp_summaries': hp_summaries,
        # New: prediction metrics
        'train_metrics': train_metrics_agg,
        'test_metrics': test_metrics_agg,
        'final_train_r2': float(final_train_r2),
        'final_train_mse': float(final_train_mse),
        'final_train_mae': float(final_train_mae),
        'final_train_r2_std': float(final_train_r2_std),
        'final_test_r2': float(final_test_r2),
        'final_test_mse': float(final_test_mse),
        'final_test_mae': float(final_test_mae),
        'final_test_r2_std': float(final_test_r2_std),
    }


def main():
    parser = argparse.ArgumentParser(description='MF BO Benchmark (Branin, HP Optimization)')
    parser.add_argument('--seeds', type=int, default=5, help='Number of random seeds')
    parser.add_argument('--budget', type=float, default=50, help='Total cost budget')
    parser.add_argument('--workers', type=int, default=None, help='Parallel workers (default: auto)')
    parser.add_argument('--scenario', type=str, default='both',
                        choices=['favorable', 'unfavorable', 'both'])
    parser.add_argument('--hp-interval', type=int, default=20,
                        help='HP optimization interval (HF data points)')
    args = parser.parse_args()

    global TOTAL_BUDGET, HP_OPTIMIZE_INTERVAL
    TOTAL_BUDGET = args.budget
    HP_OPTIMIZE_INTERVAL = args.hp_interval

    # Auto-detect workers
    if args.workers is None:
        args.workers = min(mp.cpu_count(), 10)

    print("=" * 70)
    print("Multi-Fidelity BO Benchmark (Branin, Online HP Optimization)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")
    print(f"Budget: {TOTAL_BUDGET}")
    print(f"Workers: {args.workers}")
    print(f"Scenario: {args.scenario}")
    print(f"HP Optimize Interval: {HP_OPTIMIZE_INTERVAL} HF data points")

    # Select MF models based on model_comparison results
    # From all_models_comparison_summary.csv (R² ranking):
    #   1. Pseudo-Labeling: 0.7939
    #   2. DNGO-Joint: 0.7802  <-- Best non-pseudo-labeling
    #   3. DNGO-Gradient: 0.7792
    #   ...
    #   12. MFGP: 0.3732 (worst)
    #
    # User request: Compare DNGO-Joint, MFGP, and SF models
    # - DNGO_Joint: Best performer in model_comparison
    # - GP_MFGP: MFGP baseline (despite low performance)
    mf_models = ['GP_MFGP', 'DNGO_Joint']
    print(f"MF Models ({len(mf_models)}): {mf_models}")
    print("  - GP_MFGP: MFGP baseline")
    print("  - DNGO_Joint: Best TL approach from model_comparison (R²=0.78)")

    # Branin only
    f_hf = branin_hf
    f_lf = branin_lf
    bounds = np.array([[0, 1], [0, 1]])
    f_star = FUNCTIONS['Branin-2D']['f_star']

    print(f"\nTest Function: Branin-2D (dim=2, f*={f_star:.4f})")

    # Scenarios to run
    if args.scenario == 'both':
        scenarios_to_run = ['favorable', 'unfavorable']
    else:
        scenarios_to_run = [args.scenario]

    # Print scenario info
    print("\n" + "-" * 70)
    print("Scenario Configuration:")
    print("-" * 70)
    for scenario_name in scenarios_to_run:
        s = SCENARIOS[scenario_name]
        print(f"  {scenario_name.upper()}: ρ={s['rho']}, α={s['alpha_branin']}")

    # Print HP info
    print("\n" + "-" * 70)
    print("Hyperparameter Optimization:")
    print("-" * 70)
    print("  GP_MFGP: No HP optimization (uses MLL)")
    print("  DNGO: hidden_dim, lr, alpha, beta")
    print("  BNN: hidden_dim, num_layers, lr, kl_weight")
    print("  MCDropout: hidden_dim, num_layers, dropout, lr")
    print("  DeepEnsemble: hidden_dim, num_layers, n_ensemble, lr")
    print("  SNGP: hidden_dim, num_inducing, lr")

    # Prepare all tasks
    tasks = []
    for scenario_name in scenarios_to_run:
        scenario = SCENARIOS[scenario_name]
        for model_name in mf_models:
            tasks.append((
                model_name, scenario_name, scenario,
                f_hf, f_lf, bounds, f_star,
                args.seeds, TOTAL_BUDGET
            ))

    print(f"\nTotal tasks: {len(tasks)} (models × scenarios)")
    print(f"Running {args.workers} tasks in parallel...")
    print("=" * 70)

    # Run all tasks in parallel
    start_time = datetime.now()
    all_results = {s: {} for s in scenarios_to_run}

    import time
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(run_single_model_scenario, task): task for task in tasks}

        completed = 0
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            model_name, scenario_name = task[0], task[1]

            try:
                result = future.result(timeout=1800)  # 30 min timeout (HP opt takes longer)
                if result:
                    result['elapsed_time'] = time.time() - t0
                    all_results[scenario_name][model_name] = result
                    completed += 1
                    print(f"  [{completed}/{len(tasks)}] {scenario_name}/{model_name}: "
                          f"regret={result['final_regret_mean']:.4f}±{result['final_regret_std']:.4f}")
                else:
                    print(f"  [{completed+1}/{len(tasks)}] {scenario_name}/{model_name}: FAILED")
                    completed += 1
            except Exception as e:
                print(f"  [{completed+1}/{len(tasks)}] {scenario_name}/{model_name}: ERROR - {e}")
                completed += 1

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results_mf_branin_{timestamp}")
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    for scenario_name, scenario_results in all_results.items():
        if not scenario_results:
            continue

        print(f"\n{scenario_name.upper()} (ρ={SCENARIOS[scenario_name]['rho']}, α={SCENARIOS[scenario_name]['alpha_branin']}):")
        print("-" * 100)
        print(f"{'Model':<20} {'Regret':>10} {'Train R²':>10} {'Test R²':>10} {'Test MSE':>10} {'Test MAE':>10} {'LF#':>6} {'HF#':>6}")
        print("-" * 100)

        sorted_models = sorted(scenario_results.items(), key=lambda x: x[1]['final_regret_mean'])
        for model_name, result in sorted_models:
            print(f"{model_name:<20} {result['final_regret_mean']:>10.4f} "
                  f"{result.get('final_train_r2', float('nan')):>10.4f} "
                  f"{result.get('final_test_r2', float('nan')):>10.4f} "
                  f"{result.get('final_test_mse', float('nan')):>10.4f} "
                  f"{result.get('final_test_mae', float('nan')):>10.4f} "
                  f"{result['n_lf_avg']:>6.1f} {result['n_hf_avg']:>6.1f}")

    # CSV summary (extended with prediction metrics)
    with open(results_dir / "summary.csv", "w") as f:
        f.write("scenario,model,alpha,rho,mean_regret,std_regret,"
                "train_r2,train_r2_std,train_mse,train_mae,"
                "test_r2,test_r2_std,test_mse,test_mae,"
                "n_lf,n_hf\n")
        for scenario_name, scenario_results in all_results.items():
            for model_name, result in scenario_results.items():
                f.write(f"{scenario_name},{model_name},"
                        f"{result['alpha']},{result['rho']},"
                        f"{result['final_regret_mean']:.6f},{result['final_regret_std']:.6f},"
                        f"{result.get('final_train_r2', float('nan')):.6f},"
                        f"{result.get('final_train_r2_std', float('nan')):.6f},"
                        f"{result.get('final_train_mse', float('nan')):.6f},"
                        f"{result.get('final_train_mae', float('nan')):.6f},"
                        f"{result.get('final_test_r2', float('nan')):.6f},"
                        f"{result.get('final_test_r2_std', float('nan')):.6f},"
                        f"{result.get('final_test_mse', float('nan')):.6f},"
                        f"{result.get('final_test_mae', float('nan')):.6f},"
                        f"{result['n_lf_avg']:.1f},{result['n_hf_avg']:.1f}\n")

    # Save HP optimization results
    hp_results = {}
    for scenario_name, scenario_results in all_results.items():
        hp_results[scenario_name] = {}
        for model_name, result in scenario_results.items():
            if 'hp_summaries' in result:
                hp_results[scenario_name][model_name] = result['hp_summaries']

    with open(results_dir / "hp_optimization.json", "w") as f:
        json.dump(hp_results, f, indent=2)

    print(f"\nResults saved to: {results_dir}")
    print(f"  - results.json: Full results")
    print(f"  - summary.csv: Summary table")
    print(f"  - hp_optimization.json: HP optimization history")
    print(f"Total time: {datetime.now() - start_time}")


if __name__ == "__main__":
    main()
