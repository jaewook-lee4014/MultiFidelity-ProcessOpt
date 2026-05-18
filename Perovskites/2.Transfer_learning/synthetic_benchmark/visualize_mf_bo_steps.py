#!/usr/bin/env python
"""
MF BO Step-by-Step Visualization

Visualizes model predictions at each BO step:
- Surrogate mean prediction
- ±2σ uncertainty band
- True HF function values
- EI acquisition function
- Train (LF/HF) data points

Similar style to model_comparison visualization
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import argparse
import json
import pickle
from scipy.stats import norm
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Local imports
from synthetic_functions_mfbo import branin_hf, branin_lf, SCENARIOS, FUNCTIONS
from mf_uq_models import create_mf_model
from mf_hyperparameter_optimization import (
    MFHyperparameterOptimizer, create_mf_model_with_hp, get_model_class
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def expected_improvement(mean: np.ndarray, std: np.ndarray,
                         y_best: float, xi: float = 0.01) -> np.ndarray:
    """Expected Improvement acquisition function"""
    with np.errstate(divide='ignore', invalid='ignore'):
        imp = y_best - mean - xi
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std < 1e-10] = 0.0
    return ei


def create_1d_grid(n_points=200):
    """Create 1D grid for cross-section visualization"""
    return np.linspace(0, 1, n_points)


def run_mf_bo_with_step_tracking(seed: int, model_name: str, scenario_name: str,
                                  total_budget: float = 50.0,
                                  save_interval: int = 5):
    """
    Run MF BO and save predictions at each step

    Returns:
        step_data: List of dicts with predictions at each step
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Setup
    scenario = SCENARIOS[scenario_name]
    alpha = scenario['alpha_branin']
    rho = scenario['rho']
    f_star = FUNCTIONS['Branin-2D']['f_star']

    dim = 2
    cost_hf = 1.0
    cost_lf = rho

    # Initial samples based on cost ratio (LF:HF = 1/rho : 1)
    # FAVORABLE (rho=0.1): LF:HF = 10:1, so 10 LF per 1 HF
    # UNFAVORABLE (rho=0.5): LF:HF = 2:1, so 2 LF per 1 HF
    lf_hf_ratio = int(1.0 / rho)
    n_init_hf = 2
    n_init_lf = n_init_hf * lf_hf_ratio  # FAVORABLE: 20, UNFAVORABLE: 4

    # Create evaluation grid (for plotting)
    n_grid = 100
    x1_grid = np.linspace(0, 1, n_grid)
    x2_grid = np.linspace(0, 1, n_grid)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])

    # True functions on grid
    y_hf_true = branin_hf(X_grid).reshape(n_grid, n_grid)
    y_lf_true = branin_lf(X_grid, alpha).reshape(n_grid, n_grid)

    # Initial samples
    X_lf = np.random.uniform(0, 1, (n_init_lf, dim))
    y_lf = branin_lf(X_lf, alpha).flatten()

    X_hf = np.random.uniform(0, 1, (n_init_hf, dim))
    y_hf = branin_hf(X_hf).flatten()

    current_budget = n_init_lf * cost_lf + n_init_hf * cost_hf

    # HP optimizer
    hp_optimizer = MFHyperparameterOptimizer(model_name, dim, optimize_interval=20)
    model_class = get_model_class(model_name)

    # Storage
    step_data = []

    # Save initial state
    step_data.append({
        'step': 0,
        'X_lf': X_lf.copy(),
        'y_lf': y_lf.copy(),
        'X_hf': X_hf.copy(),
        'y_hf': y_hf.copy(),
        'budget': current_budget,
        'regret': max(0, y_hf.min() - f_star),
        'y_best': y_hf.min(),
        'mean_grid': None,
        'std_grid': None,
        'ei_grid': None,
        'x_next': None,
        'eval_type': None
    })

    # BO loop
    max_iterations = 200
    iteration = 0

    # Cost-based fidelity selection ratio
    # For every 1 HF evaluation, do (1/rho) LF evaluations
    # FAVORABLE (rho=0.1): 10 LF per 1 HF
    # UNFAVORABLE (rho=0.5): 2 LF per 1 HF
    lf_per_hf = max(1, int(1.0 / rho))
    lf_counter = 0  # Track LF evaluations since last HF

    while current_budget < total_budget and iteration < max_iterations:
        iteration += 1

        remaining = total_budget - current_budget

        if remaining >= cost_hf:
            # Cost-based fidelity selection
            if remaining >= cost_lf and lf_counter < lf_per_hf:
                eval_hf = False
                cost = cost_lf
                lf_counter += 1
            else:
                eval_hf = True
                cost = cost_hf
                lf_counter = 0  # Reset counter after HF evaluation
        elif remaining >= cost_lf:
            eval_hf = False
            cost = cost_lf
        else:
            break

        try:
            # HP optimization check
            n_hf_current = len(X_hf)
            if hp_optimizer.should_optimize(n_hf_current):
                hp_optimizer.optimize(model_class, X_lf, y_lf, X_hf, y_hf)

            # Create and fit model
            current_hp = hp_optimizer.get_hp()
            if current_hp:
                model = create_mf_model_with_hp(model_name, dim, current_hp)
            else:
                model = create_mf_model(model_name, dim)

            model.fit(X_lf, y_lf, X_hf, y_hf)

            # Predict on grid
            mean_grid, std_grid = model.predict(X_grid)
            mean_grid_2d = mean_grid.reshape(n_grid, n_grid)
            std_grid_2d = std_grid.reshape(n_grid, n_grid)

            # EI on grid
            y_best = y_hf.min()
            ei_grid = expected_improvement(mean_grid, std_grid, y_best)
            ei_grid_2d = ei_grid.reshape(n_grid, n_grid)

            # Find next point
            best_idx = np.argmax(ei_grid)
            x_new = X_grid[best_idx:best_idx+1]

        except Exception as e:
            print(f"  Step {iteration} error: {e}")
            x_new = np.random.uniform(0, 1, (1, dim))
            mean_grid_2d = None
            std_grid_2d = None
            ei_grid_2d = None

        # Save step data (every save_interval steps or last step)
        if iteration % save_interval == 0 or remaining - cost < cost_lf:
            step_data.append({
                'step': iteration,
                'X_lf': X_lf.copy(),
                'y_lf': y_lf.copy(),
                'X_hf': X_hf.copy(),
                'y_hf': y_hf.copy(),
                'budget': current_budget,
                'regret': max(0, y_hf.min() - f_star),
                'y_best': y_hf.min(),
                'mean_grid': mean_grid_2d,
                'std_grid': std_grid_2d,
                'ei_grid': ei_grid_2d,
                'x_next': x_new.copy() if x_new is not None else None,
                'eval_type': 'HF' if eval_hf else 'LF'
            })

        # Evaluate new point
        if eval_hf:
            y_new = branin_hf(x_new).flatten()
            X_hf = np.vstack([X_hf, x_new])
            y_hf = np.append(y_hf, y_new)
        else:
            y_new = branin_lf(x_new, alpha).flatten()
            X_lf = np.vstack([X_lf, x_new])
            y_lf = np.append(y_lf, y_new)

        current_budget += cost

    # Save final state
    step_data.append({
        'step': iteration,
        'X_lf': X_lf.copy(),
        'y_lf': y_lf.copy(),
        'X_hf': X_hf.copy(),
        'y_hf': y_hf.copy(),
        'budget': current_budget,
        'regret': max(0, y_hf.min() - f_star),
        'y_best': y_hf.min(),
        'mean_grid': None,
        'std_grid': None,
        'ei_grid': None,
        'x_next': None,
        'eval_type': None
    })

    return {
        'step_data': step_data,
        'y_hf_true': y_hf_true,
        'y_lf_true': y_lf_true,
        'X1': X1,
        'X2': X2,
        'x1_grid': x1_grid,
        'x2_grid': x2_grid,
        'alpha': alpha,
        'rho': rho,
        'f_star': f_star
    }


def visualize_all_steps(results: dict, model_name: str, scenario_name: str,
                        output_dir: Path):
    """
    Create multi-panel visualization similar to model_comparison style

    Shows prediction, uncertainty, and EI at each saved step
    """
    step_data = results['step_data']
    y_hf_true = results['y_hf_true']
    y_lf_true = results['y_lf_true']
    X1, X2 = results['X1'], results['X2']
    x1_grid = results['x1_grid']
    f_star = results['f_star']
    alpha = results['alpha']
    rho = results['rho']

    # Filter steps with predictions
    valid_steps = [s for s in step_data if s['mean_grid'] is not None]

    if len(valid_steps) == 0:
        print("No valid steps with predictions found!")
        return

    n_steps = len(valid_steps)

    # Create figure: 3 rows (mean, std, EI) x n_steps columns
    fig = plt.figure(figsize=(5 * min(n_steps, 6), 15))
    n_cols = min(n_steps, 6)

    # Select evenly spaced steps if too many
    if n_steps > n_cols:
        indices = np.linspace(0, n_steps - 1, n_cols, dtype=int)
        selected_steps = [valid_steps[i] for i in indices]
    else:
        selected_steps = valid_steps

    gs = GridSpec(3, len(selected_steps), figure=fig, hspace=0.25, wspace=0.3)

    for col_idx, step in enumerate(selected_steps):
        step_num = step['step']
        mean_grid = step['mean_grid']
        std_grid = step['std_grid']
        ei_grid = step['ei_grid']
        X_lf = step['X_lf']
        X_hf = step['X_hf']
        x_next = step['x_next']
        y_best = step['y_best']
        regret = step['regret']
        budget = step['budget']
        eval_type = step['eval_type']

        # Row 1: Surrogate Mean
        ax1 = fig.add_subplot(gs[0, col_idx])
        c1 = ax1.contourf(X1, X2, mean_grid, levels=30, cmap='viridis')
        ax1.scatter(X_lf[:, 0], X_lf[:, 1], c='blue', s=40, marker='^',
                    edgecolors='white', linewidths=1, label='LF', zorder=4)
        ax1.scatter(X_hf[:, 0], X_hf[:, 1], c='red', s=60, marker='o',
                    edgecolors='white', linewidths=1.5, label='HF', zorder=5)
        if x_next is not None:
            ax1.scatter(x_next[0, 0], x_next[0, 1], c='lime', s=200, marker='*',
                        edgecolors='black', linewidths=2, label=f'Next ({eval_type})', zorder=6)

        ax1.set_title(f'Step {step_num}\nμ(x) | Budget={budget:.1f}', fontsize=10)
        ax1.set_xlabel('x₁')
        ax1.set_ylabel('x₂')
        if col_idx == 0:
            ax1.legend(loc='lower left', fontsize=7)
        plt.colorbar(c1, ax=ax1, fraction=0.046)

        # Row 2: Uncertainty
        ax2 = fig.add_subplot(gs[1, col_idx])
        c2 = ax2.contourf(X1, X2, std_grid, levels=30, cmap='Reds')
        ax2.scatter(X_lf[:, 0], X_lf[:, 1], c='blue', s=40, marker='^',
                    edgecolors='white', linewidths=1, zorder=4)
        ax2.scatter(X_hf[:, 0], X_hf[:, 1], c='red', s=60, marker='o',
                    edgecolors='white', linewidths=1.5, zorder=5)
        if x_next is not None:
            ax2.scatter(x_next[0, 0], x_next[0, 1], c='lime', s=200, marker='*',
                        edgecolors='black', linewidths=2, zorder=6)

        ax2.set_title(f'σ(x) | LF={len(X_lf)}, HF={len(X_hf)}', fontsize=10)
        ax2.set_xlabel('x₁')
        ax2.set_ylabel('x₂')
        plt.colorbar(c2, ax=ax2, fraction=0.046)

        # Row 3: EI
        ax3 = fig.add_subplot(gs[2, col_idx])
        c3 = ax3.contourf(X1, X2, ei_grid, levels=30, cmap='plasma')
        ax3.scatter(X_lf[:, 0], X_lf[:, 1], c='blue', s=40, marker='^',
                    edgecolors='white', linewidths=1, zorder=4)
        ax3.scatter(X_hf[:, 0], X_hf[:, 1], c='red', s=60, marker='o',
                    edgecolors='white', linewidths=1.5, zorder=5)
        if x_next is not None:
            ax3.scatter(x_next[0, 0], x_next[0, 1], c='lime', s=200, marker='*',
                        edgecolors='black', linewidths=2, zorder=6)

        ax3.set_title(f'EI(x) | Regret={regret:.4f}', fontsize=10)
        ax3.set_xlabel('x₁')
        ax3.set_ylabel('x₂')
        plt.colorbar(c3, ax=ax3, fraction=0.046)

    fig.suptitle(f'{model_name} | {scenario_name.upper()} (α={alpha}, ρ={rho}) | f*={f_star:.4f}',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(output_dir / f'{model_name}_{scenario_name}_steps_2d.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {model_name}_{scenario_name}_steps_2d.png")


def visualize_1d_cross_section(results: dict, model_name: str, scenario_name: str,
                               output_dir: Path, x2_fixed: float = 0.5):
    """
    1D cross-section visualization at fixed x2

    Similar to model_comparison style: prediction line, uncertainty band, true values
    """
    step_data = results['step_data']
    y_hf_true = results['y_hf_true']
    y_lf_true = results['y_lf_true']
    x1_grid = results['x1_grid']
    f_star = results['f_star']
    alpha = results['alpha']
    rho = results['rho']

    # Filter steps with predictions
    valid_steps = [s for s in step_data if s['mean_grid'] is not None]

    if len(valid_steps) == 0:
        print("No valid steps with predictions found!")
        return

    n_steps = len(valid_steps)

    # Select steps to show (same as 2D: max 6)
    n_show = min(n_steps, 6)
    if n_steps > n_show:
        indices = np.linspace(0, n_steps - 1, n_show, dtype=int)
        selected_steps = [valid_steps[i] for i in indices]
    else:
        selected_steps = valid_steps

    # Get cross-section index
    n_grid = len(x1_grid)
    x2_idx = int(x2_fixed * (n_grid - 1))

    # True function cross-sections
    y_hf_cross = y_hf_true[:, x2_idx]
    y_lf_cross = y_lf_true[:, x2_idx]

    # Create figure
    fig, axes = plt.subplots(len(selected_steps), 1, figsize=(14, 3.5 * len(selected_steps)))
    if len(selected_steps) == 1:
        axes = [axes]

    colors = {'mean': '#2ca02c', 'uncertainty': '#2ca02c', 'hf_true': 'black',
              'lf_true': '#1f77b4', 'hf_data': 'red', 'lf_data': 'blue'}

    for ax_idx, step in enumerate(selected_steps):
        ax = axes[ax_idx]

        step_num = step['step']
        mean_grid = step['mean_grid']
        std_grid = step['std_grid']
        X_lf = step['X_lf']
        y_lf = step['y_lf']
        X_hf = step['X_hf']
        y_hf = step['y_hf']
        y_best = step['y_best']
        regret = step['regret']
        budget = step['budget']

        # Cross-section predictions
        mean_cross = mean_grid[:, x2_idx]
        std_cross = std_grid[:, x2_idx]

        # Uncertainty band
        ax.fill_between(x1_grid,
                        mean_cross - 2*std_cross,
                        mean_cross + 2*std_cross,
                        alpha=0.3, color=colors['uncertainty'], label='±2σ')

        # Prediction
        ax.plot(x1_grid, mean_cross, color=colors['mean'], linewidth=2,
                label=f'{model_name} Predicted')

        # True HF
        ax.plot(x1_grid, y_hf_cross, 'k-', linewidth=1.5, label='True HF', alpha=0.7)

        # True LF
        ax.plot(x1_grid, y_lf_cross, '--', color=colors['lf_true'], linewidth=1,
                label='True LF', alpha=0.6)

        # HF training data (near x2=x2_fixed)
        hf_mask = np.abs(X_hf[:, 1] - x2_fixed) < 0.15
        if hf_mask.any():
            ax.scatter(X_hf[hf_mask, 0], y_hf[hf_mask], c=colors['hf_data'],
                       s=150, marker='*', edgecolors='darkred', linewidths=1.5,
                       label=f'Train HF ({len(X_hf)})', zorder=6)

        # LF training data (near x2=x2_fixed)
        lf_mask = np.abs(X_lf[:, 1] - x2_fixed) < 0.15
        if lf_mask.any():
            ax.scatter(X_lf[lf_mask, 0], y_lf[lf_mask], c=colors['lf_data'],
                       s=80, marker='^', edgecolors='darkblue', linewidths=1,
                       label=f'Train LF ({len(X_lf)})', zorder=5)

        # Calculate R² metrics
        # 1. 2D full grid R² (Test set - entire grid vs True HF)
        mean_grid_flat = mean_grid.flatten()
        y_hf_true_flat = y_hf_true.flatten()
        y_lf_true_flat = y_lf_true.flatten()

        r2_2d_hf = 1 - np.sum((y_hf_true_flat - mean_grid_flat)**2) / np.sum((y_hf_true_flat - np.mean(y_hf_true_flat))**2)
        r2_2d_lf = 1 - np.sum((y_lf_true_flat - mean_grid_flat)**2) / np.sum((y_lf_true_flat - np.mean(y_lf_true_flat))**2)

        # 2. Train set R² (predictions at training points)
        # Need to get predictions at training points from the grid (nearest neighbor interpolation)
        def get_grid_predictions(X_points, mean_grid, n_grid=100):
            """Get predictions from grid for given points"""
            idx_x1 = np.clip((X_points[:, 0] * (n_grid - 1)).astype(int), 0, n_grid - 1)
            idx_x2 = np.clip((X_points[:, 1] * (n_grid - 1)).astype(int), 0, n_grid - 1)
            return mean_grid[idx_x1, idx_x2]

        pred_hf_train = get_grid_predictions(X_hf, mean_grid)
        pred_lf_train = get_grid_predictions(X_lf, mean_grid)

        r2_train_hf = 1 - np.sum((y_hf - pred_hf_train)**2) / np.sum((y_hf - np.mean(y_hf))**2) if len(y_hf) > 1 else 0
        r2_train_lf = 1 - np.sum((y_lf - pred_lf_train)**2) / np.sum((y_lf - np.mean(y_lf))**2) if len(y_lf) > 1 else 0

        ax.set_ylabel('f(x)', fontsize=11)
        ax.set_title(f'Step {step_num} | Budget={budget:.1f} | Regret={regret:.4f}\n'
                     f'Train R²: HF={r2_train_hf:.3f}, LF={r2_train_lf:.3f} | '
                     f'Test R² (2D): HF={r2_2d_hf:.3f}, LF={r2_2d_lf:.3f}', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(f'x₁ (at x₂={x2_fixed})', fontsize=11)

    fig.suptitle(f'{model_name} | {scenario_name.upper()} (α={alpha}, ρ={rho}) | '
                 f'Cross-section at x₂={x2_fixed}',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    plt.savefig(output_dir / f'{model_name}_{scenario_name}_steps_1d.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {model_name}_{scenario_name}_steps_1d.png")


def visualize_regret_curve(all_results: dict, output_dir: Path):
    """Plot regret curves for all models and scenarios"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'GP_MFGP': '#1f77b4', 'DNGO_Joint': '#2ca02c'}

    for ax_idx, scenario in enumerate(['favorable', 'unfavorable']):
        ax = axes[ax_idx]

        for model_name, color in colors.items():
            key = f"{model_name}_{scenario}"
            if key in all_results:
                step_data = all_results[key]['step_data']
                budgets = [s['budget'] for s in step_data]
                regrets = [s['regret'] for s in step_data]

                ax.semilogy(budgets, regrets, '-o', color=color, linewidth=2,
                           markersize=4, label=model_name)

        ax.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='Target: 0.01')
        ax.set_xlabel('Budget', fontsize=11)
        ax.set_ylabel('Simple Regret (log)', fontsize=11)
        ax.set_title(f'{scenario.upper()}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('MF BO Regret Curves: GP_MFGP vs DNGO_Joint', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(output_dir / 'regret_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: regret_comparison.png")


def run_single_task(args_tuple):
    """Wrapper function for parallel execution (must be picklable)"""
    seed, model_name, scenario_name, total_budget, save_interval = args_tuple

    # Set device for this process
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"  [START] {model_name} | {scenario_name}")

    try:
        results = run_mf_bo_with_step_tracking(
            seed=seed,
            model_name=model_name,
            scenario_name=scenario_name,
            total_budget=total_budget,
            save_interval=save_interval
        )

        final_step = results['step_data'][-1]
        print(f"  [DONE] {model_name} | {scenario_name} | "
              f"Regret={final_step['regret']:.4f}, HF={len(final_step['X_hf'])}")

        return model_name, scenario_name, results

    except Exception as e:
        print(f"  [ERROR] {model_name} | {scenario_name}: {e}")
        import traceback
        traceback.print_exc()
        return model_name, scenario_name, None


def main():
    parser = argparse.ArgumentParser(description='MF BO Step Visualization')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--budget', type=float, default=50, help='Total budget')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='Save predictions every N steps')
    parser.add_argument('--models', type=str, default='GP_MFGP,DNGO_Joint,DNGO_Sequential',
                        help='Comma-separated list of models (e.g., DNGO_Joint or GP_MFGP,DNGO_Joint,DNGO_Sequential)')
    parser.add_argument('--scenarios', type=str, default='favorable,unfavorable',
                        help='Comma-separated list of scenarios (e.g., favorable or favorable,unfavorable)')
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel execution')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    args = parser.parse_args()

    # Models and scenarios
    models = [m.strip() for m in args.models.split(',')]
    scenarios = [s.strip() for s in args.scenarios.split(',')]

    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'viz_mf_bo_steps_{timestamp}')
    output_dir.mkdir(exist_ok=True)

    # Determine number of workers
    n_workers = args.workers if args.workers else min(multiprocessing.cpu_count(), len(models) * len(scenarios))

    print("=" * 60)
    print("MF BO Step-by-Step Visualization")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Budget: {args.budget}")
    print(f"Save interval: {args.save_interval}")
    print(f"Models: {models}")
    print(f"Scenarios: {scenarios}")
    print(f"Parallel: {args.parallel} (workers: {n_workers})")
    print(f"Output: {output_dir}")
    print("=" * 60)

    all_results = {}

    if args.parallel:
        # ========== PARALLEL EXECUTION ==========
        print(f"\nRunning {len(models) * len(scenarios)} tasks in parallel...")

        # Create task list
        tasks = [
            (args.seed, model_name, scenario_name, args.budget, args.save_interval)
            for model_name in models
            for scenario_name in scenarios
        ]

        # Use 'spawn' method for CUDA compatibility
        ctx = multiprocessing.get_context('spawn')

        # Run in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            futures = {executor.submit(run_single_task, task): task for task in tasks}

            for future in as_completed(futures):
                model_name, scenario_name, results = future.result()
                if results is not None:
                    all_results[f"{model_name}_{scenario_name}"] = results

        # Generate visualizations after all tasks complete
        print("\nGenerating visualizations...")
        for key, results in all_results.items():
            model_name, scenario_name = key.rsplit('_', 1)
            # Handle case where model name contains underscore
            for m in models:
                for s in scenarios:
                    if key == f"{m}_{s}":
                        model_name, scenario_name = m, s
                        break

            visualize_all_steps(results, model_name, scenario_name, output_dir)
            visualize_1d_cross_section(results, model_name, scenario_name, output_dir)

    else:
        # ========== SEQUENTIAL EXECUTION ==========
        for model_name in models:
            for scenario_name in scenarios:
                print(f"\n--- {model_name} | {scenario_name} ---")

                # Run BO with step tracking
                results = run_mf_bo_with_step_tracking(
                    seed=args.seed,
                    model_name=model_name,
                    scenario_name=scenario_name,
                    total_budget=args.budget,
                    save_interval=args.save_interval
                )

                all_results[f"{model_name}_{scenario_name}"] = results

                # 2D contour plots
                visualize_all_steps(results, model_name, scenario_name, output_dir)

                # 1D cross-section plots
                visualize_1d_cross_section(results, model_name, scenario_name, output_dir)

                # Print summary
                final_step = results['step_data'][-1]
                print(f"  Final: Budget={final_step['budget']:.1f}, "
                      f"Regret={final_step['regret']:.4f}, "
                      f"LF={len(final_step['X_lf'])}, HF={len(final_step['X_hf'])}")

    # Regret comparison
    visualize_regret_curve(all_results, output_dir)

    # Save results
    save_data = {}
    for key, results in all_results.items():
        save_data[key] = {
            'step_data': [
                {
                    'step': s['step'],
                    'budget': s['budget'],
                    'regret': s['regret'],
                    'y_best': s['y_best'],
                    'n_lf': len(s['X_lf']),
                    'n_hf': len(s['X_hf']),
                }
                for s in results['step_data']
            ],
            'alpha': results['alpha'],
            'rho': results['rho'],
            'f_star': results['f_star']
        }

    with open(output_dir / 'step_summary.json', 'w') as f:
        json.dump(save_data, f, indent=2)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
