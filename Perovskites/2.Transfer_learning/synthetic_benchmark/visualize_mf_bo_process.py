#!/usr/bin/env python
"""
MF BO Process Visualization

Visualizes the step-by-step BO process:
1. Surrogate model prediction (mean + uncertainty)
2. EI acquisition function
3. Next point selection
4. LF vs HF data points
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import argparse
from scipy.stats import norm

# Local imports
from synthetic_functions_mfbo import branin_hf, branin_lf, SCENARIOS, FUNCTIONS
from mf_uq_models import create_mf_model
from mf_hyperparameter_optimization import (
    MFHyperparameterOptimizer, create_mf_model_with_hp, get_model_class
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def expected_improvement(mean: np.ndarray, std: np.ndarray,
                         y_best: float, xi: float = 0.01) -> np.ndarray:
    """Expected Improvement acquisition function"""
    with np.errstate(divide='ignore', invalid='ignore'):
        imp = y_best - mean - xi
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std < 1e-10] = 0.0
    return ei


def create_grid(n_points=100):
    """Create 2D grid for visualization"""
    x1 = np.linspace(0, 1, n_points)
    x2 = np.linspace(0, 1, n_points)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    return X1, X2, X_grid


def run_mf_bo_with_visualization(seed: int, model_name: str, scenario_name: str,
                                  total_budget: float = 50.0,
                                  save_dir: Path = None,
                                  viz_steps: list = None):
    """
    Run MF BO with step-by-step visualization

    Args:
        seed: Random seed
        model_name: Model to use ('GP_MFGP' or 'DNGO_Joint')
        scenario_name: 'favorable' or 'unfavorable'
        total_budget: Total cost budget
        save_dir: Directory to save visualizations
        viz_steps: List of steps to visualize (e.g., [1, 5, 10, 20, 30])
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Setup
    scenario = SCENARIOS[scenario_name]
    alpha = scenario['alpha_branin']
    rho = scenario['rho']
    f_star = FUNCTIONS['Branin-2D']['f_star']

    dim = 2
    n_init_lf = 5
    n_init_hf = 2

    cost_hf = 1.0
    cost_lf = rho

    # Create grid for visualization
    X1, X2, X_grid = create_grid(100)

    # True functions on grid
    y_hf_true = branin_hf(X_grid).reshape(100, 100)
    y_lf_true = branin_lf(X_grid, alpha).reshape(100, 100)

    # Initial samples
    X_lf = np.random.uniform(0, 1, (n_init_lf, dim))
    y_lf = branin_lf(X_lf, alpha).flatten()

    X_hf = np.random.uniform(0, 1, (n_init_hf, dim))
    y_hf = branin_hf(X_hf).flatten()

    current_budget = n_init_lf * cost_lf + n_init_hf * cost_hf

    # HP optimizer
    hp_optimizer = MFHyperparameterOptimizer(model_name, dim, optimize_interval=20)
    model_class = get_model_class(model_name)

    # Default viz steps if not provided
    if viz_steps is None:
        viz_steps = [1, 5, 10, 15, 20, 25, 30]

    # Storage for visualization
    history = {
        'X_lf': [X_lf.copy()],
        'y_lf': [y_lf.copy()],
        'X_hf': [X_hf.copy()],
        'y_hf': [y_hf.copy()],
        'budget': [current_budget],
        'regret': [max(0, y_hf.min() - f_star)],
        'eval_type': []  # 'lf' or 'hf'
    }

    # BO loop
    max_iterations = 200
    iteration = 0

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

            # Predict on grid for visualization
            mean_grid, std_grid = model.predict(X_grid)
            mean_grid = mean_grid.reshape(100, 100)
            std_grid = std_grid.reshape(100, 100)

            # EI on grid
            y_best = y_hf.min()
            ei_grid = expected_improvement(mean_grid.ravel(), std_grid.ravel(), y_best)
            ei_grid = ei_grid.reshape(100, 100)

            # Find next point
            best_idx = np.argmax(ei_grid)
            best_i, best_j = best_idx // 100, best_idx % 100
            x_new = X_grid[best_idx:best_idx+1]

            # Visualize if this step is in viz_steps
            if iteration in viz_steps:
                visualize_step(
                    iteration=iteration,
                    X1=X1, X2=X2,
                    y_hf_true=y_hf_true,
                    y_lf_true=y_lf_true,
                    mean_grid=mean_grid,
                    std_grid=std_grid,
                    ei_grid=ei_grid,
                    X_lf=X_lf, y_lf=y_lf,
                    X_hf=X_hf, y_hf=y_hf,
                    x_new=x_new,
                    eval_hf=eval_hf,
                    y_best=y_best,
                    f_star=f_star,
                    model_name=model_name,
                    scenario_name=scenario_name,
                    current_budget=current_budget,
                    total_budget=total_budget,
                    save_dir=save_dir,
                    alpha=alpha,
                    rho=rho
                )

        except Exception as e:
            print(f"  Step {iteration} error: {e}")
            x_new = np.random.uniform(0, 1, (1, dim))

        # Evaluate new point
        if eval_hf:
            y_new = branin_hf(x_new).flatten()
            X_hf = np.vstack([X_hf, x_new])
            y_hf = np.append(y_hf, y_new)
            history['eval_type'].append('hf')
        else:
            y_new = branin_lf(x_new, alpha).flatten()
            X_lf = np.vstack([X_lf, x_new])
            y_lf = np.append(y_lf, y_new)
            history['eval_type'].append('lf')

        current_budget += cost

        # Update history
        history['X_lf'].append(X_lf.copy())
        history['y_lf'].append(y_lf.copy())
        history['X_hf'].append(X_hf.copy())
        history['y_hf'].append(y_hf.copy())
        history['budget'].append(current_budget)
        history['regret'].append(max(0, y_hf.min() - f_star))

    # Final summary plot
    visualize_summary(history, model_name, scenario_name, f_star, save_dir)

    return history


def visualize_step(iteration, X1, X2, y_hf_true, y_lf_true,
                   mean_grid, std_grid, ei_grid,
                   X_lf, y_lf, X_hf, y_hf,
                   x_new, eval_hf, y_best, f_star,
                   model_name, scenario_name,
                   current_budget, total_budget,
                   save_dir, alpha, rho):
    """Visualize a single BO step"""

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle(f'{model_name} | {scenario_name.upper()} (α={alpha}, ρ={rho}) | '
                 f'Step {iteration} | Budget: {current_budget:.1f}/{total_budget}',
                 fontsize=14, fontweight='bold')

    # 1. True HF function
    ax1 = fig.add_subplot(gs[0, 0])
    c1 = ax1.contourf(X1, X2, y_hf_true, levels=30, cmap='viridis')
    ax1.scatter(X_hf[:, 0], X_hf[:, 1], c='red', s=80, marker='o',
                edgecolors='white', linewidths=2, label='HF data', zorder=5)
    ax1.scatter(X_lf[:, 0], X_lf[:, 1], c='blue', s=50, marker='^',
                edgecolors='white', linewidths=1, label='LF data', zorder=4)
    ax1.scatter(x_new[0, 0], x_new[0, 1], c='yellow', s=200, marker='*',
                edgecolors='black', linewidths=2, label='Next point', zorder=6)
    ax1.set_title(f'True HF Function (f*={f_star:.4f})')
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.legend(loc='upper right', fontsize=8)
    plt.colorbar(c1, ax=ax1)

    # 2. True LF function
    ax2 = fig.add_subplot(gs[0, 1])
    c2 = ax2.contourf(X1, X2, y_lf_true, levels=30, cmap='viridis')
    ax2.scatter(X_lf[:, 0], X_lf[:, 1], c='blue', s=50, marker='^',
                edgecolors='white', linewidths=1)
    ax2.set_title(f'True LF Function (α={alpha})')
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    plt.colorbar(c2, ax=ax2)

    # 3. Surrogate Mean
    ax3 = fig.add_subplot(gs[0, 2])
    c3 = ax3.contourf(X1, X2, mean_grid, levels=30, cmap='viridis')
    ax3.scatter(X_hf[:, 0], X_hf[:, 1], c='red', s=80, marker='o',
                edgecolors='white', linewidths=2)
    ax3.scatter(X_lf[:, 0], X_lf[:, 1], c='blue', s=50, marker='^',
                edgecolors='white', linewidths=1)
    ax3.scatter(x_new[0, 0], x_new[0, 1], c='yellow', s=200, marker='*',
                edgecolors='black', linewidths=2)
    ax3.set_title('Surrogate Mean μ(x)')
    ax3.set_xlabel('x₁')
    ax3.set_ylabel('x₂')
    plt.colorbar(c3, ax=ax3)

    # 4. Surrogate Std
    ax4 = fig.add_subplot(gs[0, 3])
    c4 = ax4.contourf(X1, X2, std_grid, levels=30, cmap='Reds')
    ax4.scatter(X_hf[:, 0], X_hf[:, 1], c='red', s=80, marker='o',
                edgecolors='white', linewidths=2)
    ax4.scatter(X_lf[:, 0], X_lf[:, 1], c='blue', s=50, marker='^',
                edgecolors='white', linewidths=1)
    ax4.scatter(x_new[0, 0], x_new[0, 1], c='yellow', s=200, marker='*',
                edgecolors='black', linewidths=2)
    ax4.set_title('Uncertainty σ(x)')
    ax4.set_xlabel('x₁')
    ax4.set_ylabel('x₂')
    plt.colorbar(c4, ax=ax4)

    # 5. EI acquisition function
    ax5 = fig.add_subplot(gs[1, 0:2])
    c5 = ax5.contourf(X1, X2, ei_grid, levels=30, cmap='plasma')
    ax5.scatter(X_hf[:, 0], X_hf[:, 1], c='red', s=80, marker='o',
                edgecolors='white', linewidths=2, label='HF')
    ax5.scatter(X_lf[:, 0], X_lf[:, 1], c='blue', s=50, marker='^',
                edgecolors='white', linewidths=1, label='LF')
    ax5.scatter(x_new[0, 0], x_new[0, 1], c='lime', s=300, marker='*',
                edgecolors='black', linewidths=2, label='Next (max EI)')
    ax5.set_title(f'Expected Improvement (EI) | y_best={y_best:.4f}')
    ax5.set_xlabel('x₁')
    ax5.set_ylabel('x₂')
    ax5.legend(loc='upper right')
    plt.colorbar(c5, ax=ax5)

    # 6. Prediction error (Mean - True HF)
    ax6 = fig.add_subplot(gs[1, 2:4])
    error = mean_grid - y_hf_true
    max_err = max(abs(error.min()), abs(error.max()))
    c6 = ax6.contourf(X1, X2, error, levels=30, cmap='RdBu_r',
                      vmin=-max_err, vmax=max_err)
    ax6.scatter(X_hf[:, 0], X_hf[:, 1], c='red', s=80, marker='o',
                edgecolors='white', linewidths=2)
    ax6.scatter(X_lf[:, 0], X_lf[:, 1], c='blue', s=50, marker='^',
                edgecolors='white', linewidths=1)
    ax6.set_title('Prediction Error (μ - f_HF)')
    ax6.set_xlabel('x₁')
    ax6.set_ylabel('x₂')
    plt.colorbar(c6, ax=ax6)

    # 7. Cross-section at x2=0.5
    ax7 = fig.add_subplot(gs[2, 0:2])
    x1_line = np.linspace(0, 1, 100)
    idx_x2 = 50  # x2 = 0.5

    ax7.plot(x1_line, y_hf_true[:, idx_x2], 'k-', linewidth=2, label='True HF')
    ax7.plot(x1_line, y_lf_true[:, idx_x2], 'b--', linewidth=1.5, label='True LF')
    ax7.plot(x1_line, mean_grid[:, idx_x2], 'r-', linewidth=2, label='Surrogate μ')
    ax7.fill_between(x1_line,
                     mean_grid[:, idx_x2] - 2*std_grid[:, idx_x2],
                     mean_grid[:, idx_x2] + 2*std_grid[:, idx_x2],
                     alpha=0.3, color='red', label='±2σ')

    # Mark data points near x2=0.5
    mask_hf = np.abs(X_hf[:, 1] - 0.5) < 0.1
    mask_lf = np.abs(X_lf[:, 1] - 0.5) < 0.1
    if mask_hf.any():
        ax7.scatter(X_hf[mask_hf, 0], y_hf[mask_hf], c='red', s=100,
                    marker='o', zorder=5, edgecolors='black')
    if mask_lf.any():
        ax7.scatter(X_lf[mask_lf, 0], y_lf[mask_lf], c='blue', s=60,
                    marker='^', zorder=5, edgecolors='black')

    ax7.set_title('Cross-section at x₂=0.5')
    ax7.set_xlabel('x₁')
    ax7.set_ylabel('f(x)')
    ax7.legend(loc='upper right', fontsize=9)
    ax7.grid(True, alpha=0.3)

    # 8. EI cross-section
    ax8 = fig.add_subplot(gs[2, 2:4])
    ax8.plot(x1_line, ei_grid[:, idx_x2], 'purple', linewidth=2)
    ax8.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Mark max EI
    max_ei_idx = np.argmax(ei_grid[:, idx_x2])
    ax8.scatter(x1_line[max_ei_idx], ei_grid[max_ei_idx, idx_x2],
                c='lime', s=200, marker='*', edgecolors='black', linewidths=2,
                zorder=5)

    ax8.set_title('EI Cross-section at x₂=0.5')
    ax8.set_xlabel('x₁')
    ax8.set_ylabel('EI(x)')
    ax8.grid(True, alpha=0.3)

    # Info text
    next_eval = "HF" if eval_hf else "LF"
    info_text = (f"LF points: {len(X_lf)} | HF points: {len(X_hf)}\n"
                 f"Current best: {y_best:.4f} | Regret: {y_best - f_star:.4f}\n"
                 f"Next evaluation: {next_eval} at ({x_new[0,0]:.3f}, {x_new[0,1]:.3f})")
    fig.text(0.02, 0.02, info_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    if save_dir:
        save_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_dir / f'step_{iteration:03d}.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: step_{iteration:03d}.png")

    plt.close()


def visualize_summary(history, model_name, scenario_name, f_star, save_dir):
    """Create summary visualization of the BO run"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    fig.suptitle(f'{model_name} | {scenario_name.upper()} | BO Summary',
                 fontsize=14, fontweight='bold')

    # 1. Regret vs Budget
    ax1 = axes[0, 0]
    budgets = history['budget']
    regrets = history['regret']
    ax1.semilogy(budgets, regrets, 'b-', linewidth=2)
    ax1.set_xlabel('Budget')
    ax1.set_ylabel('Simple Regret (log scale)')
    ax1.set_title('Regret vs Budget')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.01, color='red', linestyle='--', label='Target: 0.01')
    ax1.legend()

    # 2. Final data distribution
    ax2 = axes[0, 1]
    X_lf_final = history['X_lf'][-1]
    X_hf_final = history['X_hf'][-1]

    # Create grid for true function
    X1, X2, X_grid = create_grid(100)
    y_hf_true = branin_hf(X_grid).reshape(100, 100)

    ax2.contourf(X1, X2, y_hf_true, levels=30, cmap='viridis', alpha=0.7)
    ax2.scatter(X_lf_final[:, 0], X_lf_final[:, 1], c='blue', s=50, marker='^',
                edgecolors='white', linewidths=1, label=f'LF ({len(X_lf_final)})')
    ax2.scatter(X_hf_final[:, 0], X_hf_final[:, 1], c='red', s=80, marker='o',
                edgecolors='white', linewidths=2, label=f'HF ({len(X_hf_final)})')

    # Mark global minima
    ax2.scatter([0.124, 0.543, 0.961], [0.818, 0.152, 0.165],
                c='lime', s=150, marker='*', edgecolors='black', linewidths=2,
                label='Global minima')

    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.set_title('Final Data Distribution')
    ax2.legend(loc='upper right')

    # 3. Evaluation type over iterations
    ax3 = axes[1, 0]
    eval_types = history['eval_type']
    hf_evals = [1 if t == 'hf' else 0 for t in eval_types]
    lf_evals = [1 if t == 'lf' else 0 for t in eval_types]

    iters = range(1, len(eval_types) + 1)
    ax3.bar(iters, hf_evals, color='red', alpha=0.7, label='HF eval')
    ax3.bar(iters, [-x for x in lf_evals], color='blue', alpha=0.7, label='LF eval')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Evaluation Type')
    ax3.set_title('HF vs LF Evaluations')
    ax3.legend()
    ax3.set_ylim(-1.5, 1.5)

    # 4. Best observed vs iteration
    ax4 = axes[1, 1]
    y_hf_history = history['y_hf']
    best_so_far = []
    for y_hf in y_hf_history:
        best_so_far.append(y_hf.min())

    ax4.plot(range(len(best_so_far)), best_so_far, 'r-', linewidth=2, marker='o', markersize=4)
    ax4.axhline(y=f_star, color='green', linestyle='--', linewidth=2, label=f'f*={f_star:.4f}')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Best Observed HF Value')
    ax4.set_title('Best HF Value Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_dir:
        plt.savefig(save_dir / 'summary.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: summary.png")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize MF BO Process')
    parser.add_argument('--model', type=str, default='GP_MFGP',
                        choices=['GP_MFGP', 'DNGO_Joint'],
                        help='Model to visualize')
    parser.add_argument('--scenario', type=str, default='favorable',
                        choices=['favorable', 'unfavorable'],
                        help='Scenario to run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--budget', type=float, default=50, help='Total budget')
    parser.add_argument('--steps', type=str, default='1,5,10,15,20,25,30',
                        help='Steps to visualize (comma-separated)')
    args = parser.parse_args()

    # Parse visualization steps
    viz_steps = [int(s) for s in args.steps.split(',')]

    # Create save directory
    save_dir = Path(f'viz_mf_bo_{args.model}_{args.scenario}_seed{args.seed}')
    save_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("MF BO Process Visualization")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Scenario: {args.scenario}")
    print(f"Seed: {args.seed}")
    print(f"Budget: {args.budget}")
    print(f"Visualize steps: {viz_steps}")
    print(f"Save to: {save_dir}")
    print("=" * 60)

    history = run_mf_bo_with_visualization(
        seed=args.seed,
        model_name=args.model,
        scenario_name=args.scenario,
        total_budget=args.budget,
        save_dir=save_dir,
        viz_steps=viz_steps
    )

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Final regret: {history['regret'][-1]:.6f}")
    print(f"LF points: {len(history['X_lf'][-1])}")
    print(f"HF points: {len(history['X_hf'][-1])}")
    print(f"Total cost: {history['budget'][-1]:.1f}")
    print(f"Visualizations saved to: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
