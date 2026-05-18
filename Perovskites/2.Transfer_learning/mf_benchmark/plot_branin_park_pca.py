#!/usr/bin/env python
"""
2D Visualization of Branin and Park Functions with Global Optima

- Branin: 2D function -> direct contour plot
- Park: 4D function -> PCA reduction to 2D for visualization

Total: 2 plots
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from synthetic_functions import branin_hf, park_hf, find_global_minimum

# =============================================================================
# Settings
# =============================================================================
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10

# =============================================================================
# Branin Function (2D) - Direct plot
# =============================================================================

def plot_branin(ax):
    """Plot Branin function contour with global optima"""
    n_points = 200
    x1 = np.linspace(0, 1, n_points)
    x2 = np.linspace(0, 1, n_points)
    X1, X2 = np.meshgrid(x1, x2)

    X_flat = np.column_stack([X1.ravel(), X2.ravel()])
    Z = branin_hf(X_flat).reshape(X1.shape)

    # Global optima (normalized [0,1]^2 domain)
    optima_original = [
        (-np.pi, 12.275),
        (np.pi, 2.275),
        (9.42478, 2.475)
    ]
    optima_normalized = [
        ((x1 + 5) / 15, x2 / 15) for x1, x2 in optima_original
    ]
    f_star = 0.397887

    # Plot
    Z_plot = np.log10(Z + 1)
    contour = ax.contourf(X1, X2, Z_plot, levels=50, cmap='viridis')
    ax.contour(X1, X2, Z_plot, levels=15, colors='white', alpha=0.3, linewidths=0.5)

    # Mark global optima
    for i, (ox1, ox2) in enumerate(optima_normalized):
        if 0 <= ox1 <= 1 and 0 <= ox2 <= 1:
            ax.scatter(ox1, ox2, c='red', s=200, marker='*', edgecolors='white',
                      linewidths=2, zorder=10, label='Global Optimum' if i == 0 else '')

    ax.set_xlabel('$x_1$', fontsize=LABEL_SIZE)
    ax.set_ylabel('$x_2$', fontsize=LABEL_SIZE)
    ax.set_title(f'Branin Function (2D)\n$f^* = {f_star:.4f}$', fontsize=TITLE_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=10)

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(contour, cax=cax)
    cbar.set_label('$\\log_{10}(f + 1)$', fontsize=LABEL_SIZE-1)
    cbar.ax.tick_params(labelsize=TICK_SIZE-1)

    return f_star, optima_normalized


# =============================================================================
# Park Function (4D -> 2D via PCA)
# =============================================================================

def plot_park_pca(ax):
    """Plot Park function using PCA to reduce 4D to 2D"""
    # Sample 4D space
    n_samples = 10000
    np.random.seed(42)
    X_4d = np.random.uniform(0.01, 1, (n_samples, 4))  # Avoid 0 for numerical stability
    Y = park_hf(X_4d).flatten()

    # Find global optimum
    best_x, f_star = find_global_minimum(park_hf, 4, n_random=50000, n_local=50)

    # PCA to 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_4d)

    # Transform global optimum
    opt_2d = pca.transform(best_x.reshape(1, -1))[0]

    # Create grid in PCA space for interpolation
    n_grid = 150
    x_min, x_max = X_2d[:, 0].min() - 0.1, X_2d[:, 0].max() + 0.1
    y_min, y_max = X_2d[:, 1].min() - 0.1, X_2d[:, 1].max() + 0.1

    xi = np.linspace(x_min, x_max, n_grid)
    yi = np.linspace(y_min, y_max, n_grid)
    XI, YI = np.meshgrid(xi, yi)

    # Interpolate function values onto grid
    from scipy.interpolate import griddata
    ZI = griddata(X_2d, np.log10(Y + 1), (XI, YI), method='cubic', fill_value=np.nan)

    # Plot
    contour = ax.contourf(XI, YI, ZI, levels=50, cmap='viridis')
    ax.contour(XI, YI, ZI, levels=15, colors='white', alpha=0.3, linewidths=0.5)

    # Mark global optimum
    ax.scatter(opt_2d[0], opt_2d[1], c='red', s=200, marker='*', edgecolors='white',
              linewidths=2, zorder=10, label='Global Optimum')

    ax.set_xlabel('PC1', fontsize=LABEL_SIZE)
    ax.set_ylabel('PC2', fontsize=LABEL_SIZE)
    ax.set_title(f'Park Function (4D $\\rightarrow$ 2D PCA)\n$f^* \\approx {f_star:.4f}$', fontsize=TITLE_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.legend(loc='upper right', fontsize=10)

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(contour, cax=cax)
    cbar.set_label('$\\log_{10}(f + 1)$', fontsize=LABEL_SIZE-1)
    cbar.ax.tick_params(labelsize=TICK_SIZE-1)

    # Print PCA info
    print(f"\nPark PCA Info:")
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"  Total explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")
    print(f"  Global optimum (4D): {best_x}")
    print(f"  Global optimum (2D PCA): {opt_2d}")
    print(f"  f* = {f_star:.6f}")

    return f_star, best_x, pca


# =============================================================================
# Main
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot Branin
print("Plotting Branin function...")
f_star_branin, optima_branin = plot_branin(axes[0])

# Plot Park with PCA
print("Plotting Park function (4D -> 2D PCA)...")
f_star_park, opt_park, pca_park = plot_park_pca(axes[1])

plt.tight_layout()

# Save
out_png = 'branin_park_2d_pca.png'
out_pdf = 'branin_park_2d_pca.pdf'

fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {out_png} (300 DPI)")

fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_pdf} (vector)")

plt.close()

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"\nBranin Function (2D):")
print(f"  f* = {f_star_branin:.6f}")
print(f"  Global optima:")
for i, (ox1, ox2) in enumerate(optima_branin):
    print(f"    {i+1}: ({ox1:.4f}, {ox2:.4f})")

print(f"\nPark Function (4D):")
print(f"  f* = {f_star_park:.6f}")
print(f"  Global optimum: ({opt_park[0]:.4f}, {opt_park[1]:.4f}, {opt_park[2]:.4f}, {opt_park[3]:.4f})")

print("\nDone!")
