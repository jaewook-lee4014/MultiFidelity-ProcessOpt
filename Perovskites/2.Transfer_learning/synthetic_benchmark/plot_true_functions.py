#!/usr/bin/env python
"""
Plot true HF/LF objective functions for Branin-2D
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Local imports
from synthetic_functions_mfbo import branin_hf, branin_lf, SCENARIOS, FUNCTIONS


def main():
    # Create grid
    n_grid = 100
    x1 = np.linspace(0, 1, n_grid)
    x2 = np.linspace(0, 1, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])

    # True HF function
    y_hf = branin_hf(X_grid).reshape(n_grid, n_grid)

    # LF functions for both scenarios
    alpha_favorable = SCENARIOS['favorable']['alpha_branin']  # 0.8
    alpha_unfavorable = SCENARIOS['unfavorable']['alpha_branin']  # 0.1

    y_lf_favorable = branin_lf(X_grid, alpha_favorable).reshape(n_grid, n_grid)
    y_lf_unfavorable = branin_lf(X_grid, alpha_unfavorable).reshape(n_grid, n_grid)

    f_star = FUNCTIONS['Branin-2D']['f_star']

    # Global minima locations (approximate)
    # Branin has 3 global minima
    minima = np.array([
        [0.1239, 0.8183],
        [0.5428, 0.1517],
        [0.9617, 0.1650]
    ])

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: True functions
    # HF
    ax = axes[0, 0]
    c = ax.contourf(X1, X2, y_hf, levels=50, cmap='viridis')
    ax.scatter(minima[:, 0], minima[:, 1], c='red', s=200, marker='*',
               edgecolors='white', linewidths=2, label=f'Global minima (f*={f_star:.4f})')
    ax.set_title(f'True HF Function (Branin)\nMin={y_hf.min():.4f}, Max={y_hf.max():.4f}', fontsize=12)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.legend(loc='upper right')
    plt.colorbar(c, ax=ax)

    # LF Favorable (α=0.8)
    ax = axes[0, 1]
    c = ax.contourf(X1, X2, y_lf_favorable, levels=50, cmap='viridis')
    ax.scatter(minima[:, 0], minima[:, 1], c='red', s=200, marker='*',
               edgecolors='white', linewidths=2)
    ax.set_title(f'LF Function (α={alpha_favorable}, FAVORABLE)\nMin={y_lf_favorable.min():.4f}, Max={y_lf_favorable.max():.4f}', fontsize=12)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    plt.colorbar(c, ax=ax)

    # LF Unfavorable (α=0.1)
    ax = axes[0, 2]
    c = ax.contourf(X1, X2, y_lf_unfavorable, levels=50, cmap='viridis')
    ax.scatter(minima[:, 0], minima[:, 1], c='red', s=200, marker='*',
               edgecolors='white', linewidths=2)
    ax.set_title(f'LF Function (α={alpha_unfavorable}, UNFAVORABLE)\nMin={y_lf_unfavorable.min():.4f}, Max={y_lf_unfavorable.max():.4f}', fontsize=12)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    plt.colorbar(c, ax=ax)

    # Row 2: Differences (HF - LF)
    # Correlation plot
    ax = axes[1, 0]
    ax.scatter(y_lf_favorable.ravel(), y_hf.ravel(), alpha=0.3, s=5, c='blue', label='Favorable')
    ax.scatter(y_lf_unfavorable.ravel(), y_hf.ravel(), alpha=0.3, s=5, c='orange', label='Unfavorable')

    # Calculate correlations
    corr_favorable = np.corrcoef(y_lf_favorable.ravel(), y_hf.ravel())[0, 1]
    corr_unfavorable = np.corrcoef(y_lf_unfavorable.ravel(), y_hf.ravel())[0, 1]

    lims = [min(y_hf.min(), y_lf_favorable.min(), y_lf_unfavorable.min()),
            max(y_hf.max(), y_lf_favorable.max(), y_lf_unfavorable.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
    ax.set_xlabel('LF Value')
    ax.set_ylabel('HF Value')
    ax.set_title(f'LF vs HF Correlation\nFavorable: r={corr_favorable:.3f}, Unfavorable: r={corr_unfavorable:.3f}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Difference (HF - LF_favorable)
    ax = axes[1, 1]
    diff_favorable = y_hf - y_lf_favorable
    max_diff = max(abs(diff_favorable.min()), abs(diff_favorable.max()))
    c = ax.contourf(X1, X2, diff_favorable, levels=50, cmap='RdBu_r',
                    vmin=-max_diff, vmax=max_diff)
    ax.scatter(minima[:, 0], minima[:, 1], c='lime', s=200, marker='*',
               edgecolors='black', linewidths=2)
    ax.set_title(f'HF - LF (Favorable, α={alpha_favorable})\nMean diff={diff_favorable.mean():.3f}, Std={diff_favorable.std():.3f}', fontsize=12)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    plt.colorbar(c, ax=ax)

    # Difference (HF - LF_unfavorable)
    ax = axes[1, 2]
    diff_unfavorable = y_hf - y_lf_unfavorable
    max_diff = max(abs(diff_unfavorable.min()), abs(diff_unfavorable.max()))
    c = ax.contourf(X1, X2, diff_unfavorable, levels=50, cmap='RdBu_r',
                    vmin=-max_diff, vmax=max_diff)
    ax.scatter(minima[:, 0], minima[:, 1], c='lime', s=200, marker='*',
               edgecolors='black', linewidths=2)
    ax.set_title(f'HF - LF (Unfavorable, α={alpha_unfavorable})\nMean diff={diff_unfavorable.mean():.3f}, Std={diff_unfavorable.std():.3f}', fontsize=12)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    plt.colorbar(c, ax=ax)

    fig.suptitle('Branin-2D: True HF and LF Functions\nf* = 0.3979 (global minimum)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir = Path('viz_mf_bo_steps_20251219_131928')
    plt.savefig(output_dir / 'true_functions.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'true_functions.png'}")

    # Also create 1D cross-section
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    x2_fixed = 0.5
    x2_idx = int(x2_fixed * (n_grid - 1))

    # Cross-section at x2=0.5
    ax = axes2[0]
    ax.plot(x1, y_hf[:, x2_idx], 'k-', linewidth=2, label='True HF')
    ax.plot(x1, y_lf_favorable[:, x2_idx], 'b--', linewidth=1.5, label=f'LF (α={alpha_favorable}, favorable)')
    ax.plot(x1, y_lf_unfavorable[:, x2_idx], 'r--', linewidth=1.5, label=f'LF (α={alpha_unfavorable}, unfavorable)')
    ax.axhline(y=f_star, color='green', linestyle=':', linewidth=1, label=f'f*={f_star:.4f}')
    ax.set_xlabel('x₁')
    ax.set_ylabel('f(x)')
    ax.set_title(f'Cross-section at x₂={x2_fixed}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cross-section at x2=0.15 (near global minima)
    x2_fixed2 = 0.15
    x2_idx2 = int(x2_fixed2 * (n_grid - 1))

    ax = axes2[1]
    ax.plot(x1, y_hf[:, x2_idx2], 'k-', linewidth=2, label='True HF')
    ax.plot(x1, y_lf_favorable[:, x2_idx2], 'b--', linewidth=1.5, label=f'LF (α={alpha_favorable}, favorable)')
    ax.plot(x1, y_lf_unfavorable[:, x2_idx2], 'r--', linewidth=1.5, label=f'LF (α={alpha_unfavorable}, unfavorable)')
    ax.axhline(y=f_star, color='green', linestyle=':', linewidth=1, label=f'f*={f_star:.4f}')
    ax.scatter([0.5428, 0.9617], [y_hf[x2_idx2, 54], y_hf[x2_idx2, 96]],
               c='red', s=100, marker='*', zorder=5, label='Near global minima')
    ax.set_xlabel('x₁')
    ax.set_ylabel('f(x)')
    ax.set_title(f'Cross-section at x₂={x2_fixed2} (near global minima)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig2.suptitle('Branin-2D: 1D Cross-sections', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(output_dir / 'true_functions_1d.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'true_functions_1d.png'}")

    plt.close('all')

    # Print summary
    print("\n" + "="*60)
    print("Branin-2D Function Summary")
    print("="*60)
    print(f"Global minimum: f* = {f_star:.4f}")
    print(f"HF range: [{y_hf.min():.4f}, {y_hf.max():.4f}]")
    print(f"LF (favorable, α=0.8) range: [{y_lf_favorable.min():.4f}, {y_lf_favorable.max():.4f}]")
    print(f"LF (unfavorable, α=0.1) range: [{y_lf_unfavorable.min():.4f}, {y_lf_unfavorable.max():.4f}]")
    print(f"\nCorrelation (LF vs HF):")
    print(f"  Favorable (α=0.8): r = {corr_favorable:.4f}")
    print(f"  Unfavorable (α=0.1): r = {corr_unfavorable:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
