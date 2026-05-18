#!/usr/bin/env python
"""
2D Visualization of Branin and Park Functions with Global Optima

- Branin: 2D function -> 1 contour plot
- Park: 4D function -> 2 contour plots (x1-x2 slice, x3-x4 slice)

Total: 3 plots
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from synthetic_functions import branin_hf, park_hf, find_global_minimum

# =============================================================================
# Branin Function (2D)
# =============================================================================

def plot_branin():
    """Plot Branin function contour with global optima"""
    # Create grid
    n_points = 200
    x1 = np.linspace(0, 1, n_points)
    x2 = np.linspace(0, 1, n_points)
    X1, X2 = np.meshgrid(x1, x2)

    # Evaluate function
    X_flat = np.column_stack([X1.ravel(), X2.ravel()])
    Z = branin_hf(X_flat).reshape(X1.shape)

    # Global optima (Branin has 3 global minima)
    # In normalized [0,1]^2 domain:
    # Original: (-π, 12.275), (π, 2.275), (9.42478, 2.475) with f* ≈ 0.397887
    # Normalized: x1 = (orig + 5) / 15, x2 = orig / 15
    optima_original = [
        (-np.pi, 12.275),
        (np.pi, 2.275),
        (9.42478, 2.475)
    ]
    optima_normalized = [
        ((x1 + 5) / 15, x2 / 15) for x1, x2 in optima_original
    ]
    f_star = 0.397887

    return X1, X2, Z, optima_normalized, f_star


def plot_park_x1x2():
    """Plot Park function slice: x1-x2 plane (x3=0.5, x4=0.5 fixed)"""
    n_points = 200
    x1 = np.linspace(0.01, 1, n_points)  # Avoid x1=0
    x2 = np.linspace(0, 1, n_points)
    X1, X2 = np.meshgrid(x1, x2)

    # Fix x3=0.5, x4=0.5
    x3_fixed = 0.5
    x4_fixed = 0.5

    X_flat = np.column_stack([
        X1.ravel(),
        X2.ravel(),
        np.full(X1.size, x3_fixed),
        np.full(X1.size, x4_fixed)
    ])
    Z = park_hf(X_flat).reshape(X1.shape)

    # Find minimum in this slice
    min_idx = np.unravel_index(np.argmin(Z), Z.shape)
    min_x1, min_x2 = X1[min_idx], X2[min_idx]
    min_val = Z[min_idx]

    return X1, X2, Z, (min_x1, min_x2), min_val, x3_fixed, x4_fixed


def plot_park_x3x4():
    """Plot Park function slice: x3-x4 plane (x1=0.5, x2=0.5 fixed)"""
    n_points = 200
    x3 = np.linspace(0, 1, n_points)
    x4 = np.linspace(0.01, 1, n_points)  # Avoid x4=0
    X3, X4 = np.meshgrid(x3, x4)

    # Fix x1=0.5, x2=0.5
    x1_fixed = 0.5
    x2_fixed = 0.5

    X_flat = np.column_stack([
        np.full(X3.size, x1_fixed),
        np.full(X3.size, x2_fixed),
        X3.ravel(),
        X4.ravel()
    ])
    Z = park_hf(X_flat).reshape(X3.shape)

    # Find minimum in this slice
    min_idx = np.unravel_index(np.argmin(Z), Z.shape)
    min_x3, min_x4 = X3[min_idx], X4[min_idx]
    min_val = Z[min_idx]

    return X3, X4, Z, (min_x3, min_x4), min_val, x1_fixed, x2_fixed


# =============================================================================
# Main Plotting
# =============================================================================

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10

# -----------------------------------------------------------------------------
# Plot 1: Branin (2D)
# -----------------------------------------------------------------------------
ax1 = axes[0]
X1, X2, Z_branin, optima, f_star_branin = plot_branin()

# Use log scale for better visualization
Z_plot = np.log10(Z_branin + 1)  # log(f + 1) to avoid log(0)

contour = ax1.contourf(X1, X2, Z_plot, levels=50, cmap='viridis')
ax1.contour(X1, X2, Z_plot, levels=15, colors='white', alpha=0.3, linewidths=0.5)

# Mark global optima
for i, (ox1, ox2) in enumerate(optima):
    if 0 <= ox1 <= 1 and 0 <= ox2 <= 1:
        ax1.scatter(ox1, ox2, c='red', s=150, marker='*', edgecolors='white',
                   linewidths=1.5, zorder=10, label='Global Optimum' if i == 0 else '')

ax1.set_xlabel('$x_1$', fontsize=LABEL_SIZE)
ax1.set_ylabel('$x_2$', fontsize=LABEL_SIZE)
ax1.set_title(f'Branin Function (2D)\n$f^* = {f_star_branin:.4f}$', fontsize=TITLE_SIZE)
ax1.tick_params(axis='both', labelsize=TICK_SIZE)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.legend(loc='upper right', fontsize=10)

# Colorbar
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(contour, cax=cax)
cbar.set_label('$\\log_{10}(f + 1)$', fontsize=LABEL_SIZE-1)
cbar.ax.tick_params(labelsize=TICK_SIZE-1)

# -----------------------------------------------------------------------------
# Plot 2: Park (x1-x2 slice)
# -----------------------------------------------------------------------------
ax2 = axes[1]
X1_p, X2_p, Z_park12, (min_x1, min_x2), min_val12, x3_f, x4_f = plot_park_x1x2()

Z_plot2 = np.log10(Z_park12 + 1)

contour2 = ax2.contourf(X1_p, X2_p, Z_plot2, levels=50, cmap='plasma')
ax2.contour(X1_p, X2_p, Z_plot2, levels=15, colors='white', alpha=0.3, linewidths=0.5)

# Mark slice minimum
ax2.scatter(min_x1, min_x2, c='cyan', s=150, marker='*', edgecolors='white',
           linewidths=1.5, zorder=10, label=f'Slice Min ({min_val12:.2f})')

ax2.set_xlabel('$x_1$', fontsize=LABEL_SIZE)
ax2.set_ylabel('$x_2$', fontsize=LABEL_SIZE)
ax2.set_title(f'Park Function ($x_1$-$x_2$ slice)\n$x_3={x3_f}$, $x_4={x4_f}$ fixed', fontsize=TITLE_SIZE)
ax2.tick_params(axis='both', labelsize=TICK_SIZE)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.legend(loc='upper right', fontsize=10)

divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.1)
cbar2 = plt.colorbar(contour2, cax=cax2)
cbar2.set_label('$\\log_{10}(f + 1)$', fontsize=LABEL_SIZE-1)
cbar2.ax.tick_params(labelsize=TICK_SIZE-1)

# -----------------------------------------------------------------------------
# Plot 3: Park (x3-x4 slice)
# -----------------------------------------------------------------------------
ax3 = axes[2]
X3_p, X4_p, Z_park34, (min_x3, min_x4), min_val34, x1_f, x2_f = plot_park_x3x4()

Z_plot3 = np.log10(Z_park34 + 1)

contour3 = ax3.contourf(X3_p, X4_p, Z_plot3, levels=50, cmap='plasma')
ax3.contour(X3_p, X4_p, Z_plot3, levels=15, colors='white', alpha=0.3, linewidths=0.5)

# Mark slice minimum
ax3.scatter(min_x3, min_x4, c='cyan', s=150, marker='*', edgecolors='white',
           linewidths=1.5, zorder=10, label=f'Slice Min ({min_val34:.2f})')

ax3.set_xlabel('$x_3$', fontsize=LABEL_SIZE)
ax3.set_ylabel('$x_4$', fontsize=LABEL_SIZE)
ax3.set_title(f'Park Function ($x_3$-$x_4$ slice)\n$x_1={x1_f}$, $x_2={x2_f}$ fixed', fontsize=TITLE_SIZE)
ax3.tick_params(axis='both', labelsize=TICK_SIZE)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.legend(loc='upper right', fontsize=10)

divider3 = make_axes_locatable(ax3)
cax3 = divider3.append_axes("right", size="5%", pad=0.1)
cbar3 = plt.colorbar(contour3, cax=cax3)
cbar3.set_label('$\\log_{10}(f + 1)$', fontsize=LABEL_SIZE-1)
cbar3.ax.tick_params(labelsize=TICK_SIZE-1)

# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------
plt.tight_layout()

out_png = 'branin_park_2d_visualization.png'
out_pdf = 'branin_park_2d_visualization.pdf'

fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_png} (300 DPI)")

fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_pdf} (vector)")

plt.close()

# =============================================================================
# Find and print global optimum for Park
# =============================================================================
print("\n" + "=" * 60)
print("Global Optima Information")
print("=" * 60)

print("\nBranin Function (2D):")
print(f"  f* = {f_star_branin:.6f}")
print("  Global optima locations (normalized [0,1]^2):")
for i, (ox1, ox2) in enumerate(optima):
    print(f"    Optimum {i+1}: x = ({ox1:.4f}, {ox2:.4f})")

print("\nPark Function (4D):")
print("  Finding global optimum...")
best_x, best_y = find_global_minimum(park_hf, 4, n_random=50000, n_local=50)
print(f"  f* ≈ {best_y:.6f}")
print(f"  x* ≈ ({best_x[0]:.4f}, {best_x[1]:.4f}, {best_x[2]:.4f}, {best_x[3]:.4f})")

print("\nDone!")
