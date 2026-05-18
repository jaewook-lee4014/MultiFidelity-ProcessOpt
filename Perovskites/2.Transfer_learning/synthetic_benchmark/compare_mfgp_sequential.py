#!/usr/bin/env python
"""
MFGP vs DNGO_Sequential 상세 비교 시각화

R²는 높은데 Regret이 높은 이유를 분석:
1. Uncertainty 비교 (std 분포)
2. EI landscape 비교
3. Uncertainty Calibration (|error| vs std)
4. 최적점 근처 zoom-in
5. Acquisition 궤적 비교
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import argparse
from scipy.stats import norm, spearmanr
from datetime import datetime

from synthetic_functions_mfbo import branin_hf, branin_lf, SCENARIOS, FUNCTIONS
from mf_uq_models import create_mf_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


def expected_improvement(mean: np.ndarray, std: np.ndarray,
                         y_best: float, xi: float = 0.01) -> np.ndarray:
    """Expected Improvement acquisition function"""
    with np.errstate(divide='ignore', invalid='ignore'):
        imp = y_best - mean - xi
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std < 1e-10] = 0.0
    return ei


def run_comparison(seed: int, total_budget: float = 50.0, save_interval: int = 5):
    """
    동일한 데이터로 두 모델 비교 실행
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Setup - favorable scenario only
    scenario = SCENARIOS['favorable']
    alpha = scenario['alpha_branin']
    rho = scenario['rho']
    f_star = FUNCTIONS['Branin-2D']['f_star']

    dim = 2
    cost_hf = 1.0
    cost_lf = rho

    # Initial samples
    lf_hf_ratio = int(1.0 / rho)
    n_init_hf = 2
    n_init_lf = n_init_hf * lf_hf_ratio

    # Create evaluation grid
    n_grid = 100
    x1_grid = np.linspace(0, 1, n_grid)
    x2_grid = np.linspace(0, 1, n_grid)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])

    # True functions
    y_hf_true = branin_hf(X_grid).reshape(n_grid, n_grid)
    y_lf_true = branin_lf(X_grid, alpha).reshape(n_grid, n_grid)

    # Find true optimum location
    opt_idx = np.argmin(y_hf_true)
    opt_x1, opt_x2 = X1.ravel()[opt_idx], X2.ravel()[opt_idx]
    print(f"True optimum at: ({opt_x1:.3f}, {opt_x2:.3f}), f*={f_star:.4f}")

    # Initial samples
    X_lf = np.random.uniform(0, 1, (n_init_lf, dim))
    y_lf = branin_lf(X_lf, alpha).flatten()

    X_hf = np.random.uniform(0, 1, (n_init_hf, dim))
    y_hf = branin_hf(X_hf).flatten()

    current_budget = n_init_lf * cost_lf + n_init_hf * cost_hf

    # Storage for comparison
    comparison_data = []

    # BO loop
    max_iterations = 200
    iteration = 0
    lf_per_hf = max(1, int(1.0 / rho))
    lf_counter = 0

    # Track acquisition points for each model
    mfgp_acquisitions = []
    seq_acquisitions = []

    while current_budget < total_budget and iteration < max_iterations:
        iteration += 1
        remaining = total_budget - current_budget

        if remaining >= cost_hf:
            if remaining >= cost_lf and lf_counter < lf_per_hf:
                eval_hf = False
                cost = cost_lf
                lf_counter += 1
            else:
                eval_hf = True
                cost = cost_hf
                lf_counter = 0
        elif remaining >= cost_lf:
            eval_hf = False
            cost = cost_lf
        else:
            break

        try:
            # Fit both models on same data
            model_mfgp = create_mf_model('GP_MFGP', dim)
            model_seq = create_mf_model('DNGO_Sequential', dim)

            model_mfgp.fit(X_lf, y_lf, X_hf, y_hf)
            model_seq.fit(X_lf, y_lf, X_hf, y_hf)

            # Predict on grid
            mean_mfgp, std_mfgp = model_mfgp.predict(X_grid)
            mean_seq, std_seq = model_seq.predict(X_grid)

            # EI
            y_best = y_hf.min()
            ei_mfgp = expected_improvement(mean_mfgp, std_mfgp, y_best)
            ei_seq = expected_improvement(mean_seq, std_seq, y_best)

            # Next points
            x_next_mfgp = X_grid[np.argmax(ei_mfgp)]
            x_next_seq = X_grid[np.argmax(ei_seq)]

            # Calculate metrics
            y_hf_true_flat = y_hf_true.flatten()

            # R² for HF
            r2_mfgp = 1 - np.sum((y_hf_true_flat - mean_mfgp)**2) / np.sum((y_hf_true_flat - np.mean(y_hf_true_flat))**2)
            r2_seq = 1 - np.sum((y_hf_true_flat - mean_seq)**2) / np.sum((y_hf_true_flat - np.mean(y_hf_true_flat))**2)

            # Errors
            errors_mfgp = np.abs(y_hf_true_flat - mean_mfgp)
            errors_seq = np.abs(y_hf_true_flat - mean_seq)

            # Calibration: correlation between |error| and std
            calib_mfgp, _ = spearmanr(errors_mfgp, std_mfgp)
            calib_seq, _ = spearmanr(errors_seq, std_seq)

            # Save comparison data
            if iteration % save_interval == 0:
                comparison_data.append({
                    'step': iteration,
                    'X_lf': X_lf.copy(),
                    'y_lf': y_lf.copy(),
                    'X_hf': X_hf.copy(),
                    'y_hf': y_hf.copy(),
                    'budget': current_budget,
                    'y_best': y_best,
                    'regret': max(0, y_best - f_star),
                    # MFGP
                    'mean_mfgp': mean_mfgp.reshape(n_grid, n_grid),
                    'std_mfgp': std_mfgp.reshape(n_grid, n_grid),
                    'ei_mfgp': ei_mfgp.reshape(n_grid, n_grid),
                    'x_next_mfgp': x_next_mfgp,
                    'r2_mfgp': r2_mfgp,
                    'calib_mfgp': calib_mfgp,
                    'errors_mfgp': errors_mfgp,
                    # Sequential
                    'mean_seq': mean_seq.reshape(n_grid, n_grid),
                    'std_seq': std_seq.reshape(n_grid, n_grid),
                    'ei_seq': ei_seq.reshape(n_grid, n_grid),
                    'x_next_seq': x_next_seq,
                    'r2_seq': r2_seq,
                    'calib_seq': calib_seq,
                    'errors_seq': errors_seq,
                })

            # Use MFGP's suggestion for actual BO (to keep data consistent)
            x_new = x_next_mfgp.reshape(1, -1)
            mfgp_acquisitions.append(x_next_mfgp.copy())
            seq_acquisitions.append(x_next_seq.copy())

        except Exception as e:
            print(f"  Step {iteration} error: {e}")
            x_new = np.random.uniform(0, 1, (1, dim))

        # Evaluate
        if eval_hf:
            y_new = branin_hf(x_new).flatten()
            X_hf = np.vstack([X_hf, x_new])
            y_hf = np.append(y_hf, y_new)
        else:
            y_new = branin_lf(x_new, alpha).flatten()
            X_lf = np.vstack([X_lf, x_new])
            y_lf = np.append(y_lf, y_new)

        current_budget += cost

        if iteration % 10 == 0:
            print(f"  Step {iteration}: Budget={current_budget:.1f}, Regret={max(0, y_hf.min() - f_star):.4f}")

    return {
        'comparison_data': comparison_data,
        'y_hf_true': y_hf_true,
        'y_lf_true': y_lf_true,
        'X1': X1,
        'X2': X2,
        'x1_grid': x1_grid,
        'x2_grid': x2_grid,
        'alpha': alpha,
        'rho': rho,
        'f_star': f_star,
        'opt_x1': opt_x1,
        'opt_x2': opt_x2,
        'mfgp_acquisitions': mfgp_acquisitions,
        'seq_acquisitions': seq_acquisitions,
    }


def visualize_comparison(results: dict, output_dir: Path):
    """
    상세 비교 시각화
    """
    comparison_data = results['comparison_data']
    y_hf_true = results['y_hf_true']
    X1, X2 = results['X1'], results['X2']
    f_star = results['f_star']
    opt_x1, opt_x2 = results['opt_x1'], results['opt_x2']

    if len(comparison_data) == 0:
        print("No comparison data!")
        return

    # Select steps to visualize
    n_steps = min(len(comparison_data), 4)
    indices = np.linspace(0, len(comparison_data) - 1, n_steps, dtype=int)
    selected_data = [comparison_data[i] for i in indices]

    for data in selected_data:
        step = data['step']
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Row 1: Mean predictions
        ax1 = fig.add_subplot(gs[0, 0])
        c1 = ax1.contourf(X1, X2, data['mean_mfgp'], levels=30, cmap='viridis')
        ax1.scatter(data['X_hf'][:, 0], data['X_hf'][:, 1], c='red', s=60, marker='o', edgecolors='white', label='HF')
        ax1.scatter(data['X_lf'][:, 0], data['X_lf'][:, 1], c='blue', s=30, marker='^', edgecolors='white', label='LF')
        ax1.scatter(opt_x1, opt_x2, c='yellow', s=200, marker='*', edgecolors='black', linewidths=2, label='True Opt')
        ax1.scatter(data['x_next_mfgp'][0], data['x_next_mfgp'][1], c='lime', s=150, marker='X', edgecolors='black')
        ax1.set_title(f'MFGP Mean (R²={data["r2_mfgp"]:.3f})', fontsize=11)
        ax1.legend(loc='lower left', fontsize=7)
        plt.colorbar(c1, ax=ax1, fraction=0.046)

        ax2 = fig.add_subplot(gs[0, 1])
        c2 = ax2.contourf(X1, X2, data['mean_seq'], levels=30, cmap='viridis')
        ax2.scatter(data['X_hf'][:, 0], data['X_hf'][:, 1], c='red', s=60, marker='o', edgecolors='white')
        ax2.scatter(data['X_lf'][:, 0], data['X_lf'][:, 1], c='blue', s=30, marker='^', edgecolors='white')
        ax2.scatter(opt_x1, opt_x2, c='yellow', s=200, marker='*', edgecolors='black', linewidths=2)
        ax2.scatter(data['x_next_seq'][0], data['x_next_seq'][1], c='lime', s=150, marker='X', edgecolors='black')
        ax2.set_title(f'Sequential Mean (R²={data["r2_seq"]:.3f})', fontsize=11)
        plt.colorbar(c2, ax=ax2, fraction=0.046)

        # Mean difference
        ax3 = fig.add_subplot(gs[0, 2])
        mean_diff = data['mean_seq'] - data['mean_mfgp']
        vmax = max(abs(mean_diff.min()), abs(mean_diff.max()))
        c3 = ax3.contourf(X1, X2, mean_diff, levels=30, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax3.scatter(opt_x1, opt_x2, c='yellow', s=200, marker='*', edgecolors='black', linewidths=2)
        ax3.set_title('Mean Diff (Seq - MFGP)', fontsize=11)
        plt.colorbar(c3, ax=ax3, fraction=0.046)

        # True HF
        ax4 = fig.add_subplot(gs[0, 3])
        c4 = ax4.contourf(X1, X2, y_hf_true, levels=30, cmap='viridis')
        ax4.scatter(opt_x1, opt_x2, c='yellow', s=200, marker='*', edgecolors='black', linewidths=2)
        ax4.set_title(f'True HF (f*={f_star:.4f})', fontsize=11)
        plt.colorbar(c4, ax=ax4, fraction=0.046)

        # Row 2: Uncertainty (STD) - KEY COMPARISON
        ax5 = fig.add_subplot(gs[1, 0])
        c5 = ax5.contourf(X1, X2, data['std_mfgp'], levels=30, cmap='Reds')
        ax5.scatter(data['X_hf'][:, 0], data['X_hf'][:, 1], c='red', s=60, marker='o', edgecolors='white')
        ax5.scatter(data['X_lf'][:, 0], data['X_lf'][:, 1], c='blue', s=30, marker='^', edgecolors='white')
        ax5.scatter(opt_x1, opt_x2, c='yellow', s=200, marker='*', edgecolors='black', linewidths=2)
        ax5.set_title(f'MFGP Std (mean={data["std_mfgp"].mean():.3f})', fontsize=11)
        plt.colorbar(c5, ax=ax5, fraction=0.046)

        ax6 = fig.add_subplot(gs[1, 1])
        c6 = ax6.contourf(X1, X2, data['std_seq'], levels=30, cmap='Reds')
        ax6.scatter(data['X_hf'][:, 0], data['X_hf'][:, 1], c='red', s=60, marker='o', edgecolors='white')
        ax6.scatter(data['X_lf'][:, 0], data['X_lf'][:, 1], c='blue', s=30, marker='^', edgecolors='white')
        ax6.scatter(opt_x1, opt_x2, c='yellow', s=200, marker='*', edgecolors='black', linewidths=2)
        ax6.set_title(f'Sequential Std (mean={data["std_seq"].mean():.3f})', fontsize=11)
        plt.colorbar(c6, ax=ax6, fraction=0.046)

        # Std ratio
        ax7 = fig.add_subplot(gs[1, 2])
        std_ratio = data['std_seq'] / (data['std_mfgp'] + 1e-6)
        c7 = ax7.contourf(X1, X2, std_ratio, levels=30, cmap='RdBu_r',
                          vmin=0, vmax=2)
        ax7.scatter(opt_x1, opt_x2, c='yellow', s=200, marker='*', edgecolors='black', linewidths=2)
        ax7.set_title(f'Std Ratio (Seq/MFGP, mean={std_ratio.mean():.2f})', fontsize=11)
        plt.colorbar(c7, ax=ax7, fraction=0.046)

        # Std histogram
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.hist(data['std_mfgp'].flatten(), bins=50, alpha=0.6, label=f'MFGP ({data["std_mfgp"].mean():.3f})', color='blue')
        ax8.hist(data['std_seq'].flatten(), bins=50, alpha=0.6, label=f'Seq ({data["std_seq"].mean():.3f})', color='red')
        ax8.set_xlabel('Std')
        ax8.set_ylabel('Count')
        ax8.set_title('Std Distribution', fontsize=11)
        ax8.legend(fontsize=9)

        # Row 3: EI comparison
        ax9 = fig.add_subplot(gs[2, 0])
        c9 = ax9.contourf(X1, X2, data['ei_mfgp'], levels=30, cmap='plasma')
        ax9.scatter(data['x_next_mfgp'][0], data['x_next_mfgp'][1], c='lime', s=200, marker='*', edgecolors='black', linewidths=2)
        ax9.scatter(opt_x1, opt_x2, c='yellow', s=150, marker='*', edgecolors='black', linewidths=2)
        ax9.set_title(f'MFGP EI (max={data["ei_mfgp"].max():.4f})', fontsize=11)
        plt.colorbar(c9, ax=ax9, fraction=0.046)

        ax10 = fig.add_subplot(gs[2, 1])
        c10 = ax10.contourf(X1, X2, data['ei_seq'], levels=30, cmap='plasma')
        ax10.scatter(data['x_next_seq'][0], data['x_next_seq'][1], c='lime', s=200, marker='*', edgecolors='black', linewidths=2)
        ax10.scatter(opt_x1, opt_x2, c='yellow', s=150, marker='*', edgecolors='black', linewidths=2)
        ax10.set_title(f'Sequential EI (max={data["ei_seq"].max():.4f})', fontsize=11)
        plt.colorbar(c10, ax=ax10, fraction=0.046)

        # EI difference
        ax11 = fig.add_subplot(gs[2, 2])
        ei_diff = data['ei_seq'] - data['ei_mfgp']
        vmax_ei = max(abs(ei_diff.min()), abs(ei_diff.max()))
        c11 = ax11.contourf(X1, X2, ei_diff, levels=30, cmap='RdBu_r', vmin=-vmax_ei, vmax=vmax_ei)
        ax11.scatter(opt_x1, opt_x2, c='yellow', s=200, marker='*', edgecolors='black', linewidths=2)
        ax11.set_title('EI Diff (Seq - MFGP)', fontsize=11)
        plt.colorbar(c11, ax=ax11, fraction=0.046)

        # Next point comparison
        ax12 = fig.add_subplot(gs[2, 3])
        ax12.contourf(X1, X2, y_hf_true, levels=30, cmap='viridis', alpha=0.5)
        ax12.scatter(data['x_next_mfgp'][0], data['x_next_mfgp'][1], c='blue', s=200, marker='X',
                     edgecolors='black', linewidths=2, label='MFGP next')
        ax12.scatter(data['x_next_seq'][0], data['x_next_seq'][1], c='red', s=200, marker='X',
                     edgecolors='black', linewidths=2, label='Seq next')
        ax12.scatter(opt_x1, opt_x2, c='yellow', s=200, marker='*', edgecolors='black', linewidths=2, label='True Opt')
        dist_mfgp = np.sqrt((data['x_next_mfgp'][0] - opt_x1)**2 + (data['x_next_mfgp'][1] - opt_x2)**2)
        dist_seq = np.sqrt((data['x_next_seq'][0] - opt_x1)**2 + (data['x_next_seq'][1] - opt_x2)**2)
        ax12.set_title(f'Next Points (dist: MFGP={dist_mfgp:.3f}, Seq={dist_seq:.3f})', fontsize=10)
        ax12.legend(loc='lower left', fontsize=8)

        # Row 4: Calibration plots
        ax13 = fig.add_subplot(gs[3, 0])
        ax13.scatter(data['std_mfgp'].flatten(), data['errors_mfgp'], alpha=0.3, s=5, c='blue')
        ax13.set_xlabel('Predicted Std')
        ax13.set_ylabel('|Actual Error|')
        ax13.set_title(f'MFGP Calibration (ρ={data["calib_mfgp"]:.3f})', fontsize=11)
        # Add diagonal line
        max_val = max(data['std_mfgp'].max(), data['errors_mfgp'].max())
        ax13.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect')
        ax13.legend()

        ax14 = fig.add_subplot(gs[3, 1])
        ax14.scatter(data['std_seq'].flatten(), data['errors_seq'], alpha=0.3, s=5, c='red')
        ax14.set_xlabel('Predicted Std')
        ax14.set_ylabel('|Actual Error|')
        ax14.set_title(f'Sequential Calibration (ρ={data["calib_seq"]:.3f})', fontsize=11)
        max_val = max(data['std_seq'].max(), data['errors_seq'].max())
        ax14.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect')
        ax14.legend()

        # Error at optimum region
        ax15 = fig.add_subplot(gs[3, 2])
        # Zoom into optimum region (±0.2 around optimum)
        opt_mask = ((X1 - opt_x1)**2 + (X2 - opt_x2)**2) < 0.2**2
        error_mfgp_opt = np.abs(data['mean_mfgp'] - y_hf_true)
        error_seq_opt = np.abs(data['mean_seq'] - y_hf_true)

        ax15.scatter(X1[opt_mask], X2[opt_mask], c=error_mfgp_opt[opt_mask], cmap='Reds', s=30, marker='s', label='MFGP')
        ax15.scatter(opt_x1, opt_x2, c='yellow', s=200, marker='*', edgecolors='black', linewidths=2)
        ax15.set_title(f'MFGP Error near Opt (mean={error_mfgp_opt[opt_mask].mean():.4f})', fontsize=10)
        ax15.set_xlim(opt_x1 - 0.25, opt_x1 + 0.25)
        ax15.set_ylim(opt_x2 - 0.25, opt_x2 + 0.25)

        ax16 = fig.add_subplot(gs[3, 3])
        ax16.scatter(X1[opt_mask], X2[opt_mask], c=error_seq_opt[opt_mask], cmap='Reds', s=30, marker='s', label='Seq')
        ax16.scatter(opt_x1, opt_x2, c='yellow', s=200, marker='*', edgecolors='black', linewidths=2)
        ax16.set_title(f'Seq Error near Opt (mean={error_seq_opt[opt_mask].mean():.4f})', fontsize=10)
        ax16.set_xlim(opt_x1 - 0.25, opt_x1 + 0.25)
        ax16.set_ylim(opt_x2 - 0.25, opt_x2 + 0.25)

        fig.suptitle(f'Step {step} | Budget={data["budget"]:.1f} | Regret={data["regret"]:.4f} | '
                     f'LF={len(data["X_lf"])}, HF={len(data["X_hf"])}',
                     fontsize=14, fontweight='bold', y=0.995)

        plt.savefig(output_dir / f'comparison_step_{step:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: comparison_step_{step:03d}.png")


def visualize_trajectory(results: dict, output_dir: Path):
    """
    Acquisition 궤적 비교
    """
    y_hf_true = results['y_hf_true']
    X1, X2 = results['X1'], results['X2']
    opt_x1, opt_x2 = results['opt_x1'], results['opt_x2']
    f_star = results['f_star']
    mfgp_acq = np.array(results['mfgp_acquisitions'])
    seq_acq = np.array(results['seq_acquisitions'])

    if len(mfgp_acq) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # MFGP trajectory
    ax1 = axes[0]
    c1 = ax1.contourf(X1, X2, y_hf_true, levels=30, cmap='viridis', alpha=0.7)
    ax1.plot(mfgp_acq[:, 0], mfgp_acq[:, 1], 'b-', linewidth=1, alpha=0.5)
    scatter1 = ax1.scatter(mfgp_acq[:, 0], mfgp_acq[:, 1],
                           c=np.arange(len(mfgp_acq)), cmap='Blues', s=50, edgecolors='black')
    ax1.scatter(opt_x1, opt_x2, c='yellow', s=300, marker='*', edgecolors='black', linewidths=2, label='True Opt')
    ax1.set_title('MFGP Acquisition Trajectory', fontsize=12)
    ax1.legend()
    plt.colorbar(scatter1, ax=ax1, label='Step')

    # Sequential trajectory
    ax2 = axes[1]
    c2 = ax2.contourf(X1, X2, y_hf_true, levels=30, cmap='viridis', alpha=0.7)
    ax2.plot(seq_acq[:, 0], seq_acq[:, 1], 'r-', linewidth=1, alpha=0.5)
    scatter2 = ax2.scatter(seq_acq[:, 0], seq_acq[:, 1],
                           c=np.arange(len(seq_acq)), cmap='Reds', s=50, edgecolors='black')
    ax2.scatter(opt_x1, opt_x2, c='yellow', s=300, marker='*', edgecolors='black', linewidths=2, label='True Opt')
    ax2.set_title('Sequential Acquisition Trajectory', fontsize=12)
    ax2.legend()
    plt.colorbar(scatter2, ax=ax2, label='Step')

    fig.suptitle(f'Acquisition Trajectories (f*={f_star:.4f})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'acquisition_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: acquisition_trajectories.png")


def visualize_summary(results: dict, output_dir: Path):
    """
    Summary metrics across steps
    """
    comparison_data = results['comparison_data']

    if len(comparison_data) == 0:
        return

    steps = [d['step'] for d in comparison_data]
    r2_mfgp = [d['r2_mfgp'] for d in comparison_data]
    r2_seq = [d['r2_seq'] for d in comparison_data]
    calib_mfgp = [d['calib_mfgp'] for d in comparison_data]
    calib_seq = [d['calib_seq'] for d in comparison_data]
    std_mean_mfgp = [d['std_mfgp'].mean() for d in comparison_data]
    std_mean_seq = [d['std_seq'].mean() for d in comparison_data]
    ei_max_mfgp = [d['ei_mfgp'].max() for d in comparison_data]
    ei_max_seq = [d['ei_seq'].max() for d in comparison_data]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # R² over steps
    ax1 = axes[0, 0]
    ax1.plot(steps, r2_mfgp, 'b-o', label='MFGP', linewidth=2, markersize=6)
    ax1.plot(steps, r2_seq, 'r-s', label='Sequential', linewidth=2, markersize=6)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('R²')
    ax1.set_title('Prediction Accuracy (R²)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Calibration over steps
    ax2 = axes[0, 1]
    ax2.plot(steps, calib_mfgp, 'b-o', label='MFGP', linewidth=2, markersize=6)
    ax2.plot(steps, calib_seq, 'r-s', label='Sequential', linewidth=2, markersize=6)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Spearman ρ (|error| vs std)')
    ax2.set_title('Uncertainty Calibration', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mean std over steps
    ax3 = axes[1, 0]
    ax3.plot(steps, std_mean_mfgp, 'b-o', label='MFGP', linewidth=2, markersize=6)
    ax3.plot(steps, std_mean_seq, 'r-s', label='Sequential', linewidth=2, markersize=6)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Mean Std')
    ax3.set_title('Average Uncertainty', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Max EI over steps
    ax4 = axes[1, 1]
    ax4.plot(steps, ei_max_mfgp, 'b-o', label='MFGP', linewidth=2, markersize=6)
    ax4.plot(steps, ei_max_seq, 'r-s', label='Sequential', linewidth=2, markersize=6)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Max EI')
    ax4.set_title('Maximum Expected Improvement', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.suptitle('MFGP vs Sequential: Summary Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: summary_metrics.png")


def main():
    parser = argparse.ArgumentParser(description='MFGP vs Sequential Comparison')
    parser.add_argument('--seed', type=int, default=1004, help='Random seed')
    parser.add_argument('--budget', type=float, default=50, help='Total budget')
    parser.add_argument('--save-interval', type=int, default=5, help='Save interval')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'compare_mfgp_seq_{timestamp}')
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("MFGP vs DNGO_Sequential Comparison")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Budget: {args.budget}")
    print(f"Save interval: {args.save_interval}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Run comparison
    results = run_comparison(args.seed, args.budget, args.save_interval)

    # Visualizations
    print("\nGenerating visualizations...")
    visualize_comparison(results, output_dir)
    visualize_trajectory(results, output_dir)
    visualize_summary(results, output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
