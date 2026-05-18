#!/usr/bin/env python
"""
Park Function Animation

Creates two animated GIFs showing how the Park function landscape changes
as the fixed variables vary from 0 to 1:

1. x1-x2 plane: x3 and x4 vary from 0 to 1
2. x3-x4 plane: x1 and x2 vary from 0 to 1

Global optimum is marked with a red star (same style as Branin).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from synthetic_functions import park_hf
from PIL import Image
import io

# =============================================================================
# Settings
# =============================================================================
n_points = 100  # Grid resolution
n_frames = 50   # Number of frames in animation
duration = 100  # milliseconds per frame (100ms = 10fps)

# Global optimum of Park function
GLOBAL_OPT = np.array([0.0, 0.0, 0.0, 0.0])
F_STAR = 0.0

TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10

# =============================================================================
# Animation 1: x1-x2 plane with x3, x4 varying
# =============================================================================

def create_x1x2_animation():
    """Create animation of x1-x2 plane as x3, x4 vary"""

    # Create grid for x1-x2
    x1 = np.linspace(0.01, 1, n_points)
    x2 = np.linspace(0, 1, n_points)
    X1, X2 = np.meshgrid(x1, x2)

    # Values for x3, x4 to sweep through
    x3_values = np.linspace(0, 1, n_frames)
    x4_values = np.linspace(0.01, 1, n_frames)

    frames = []

    for frame in range(n_frames):
        fig, ax = plt.subplots(figsize=(8, 7))

        x3_val = x3_values[frame]
        x4_val = x4_values[frame]

        X_flat = np.column_stack([
            X1.ravel(), X2.ravel(),
            np.full(X1.size, x3_val),
            np.full(X1.size, x4_val)
        ])
        Z = park_hf(X_flat).reshape(X1.shape)
        Z_plot = np.log10(Z + 1)

        contour = ax.contourf(X1, X2, Z_plot, levels=50, cmap='viridis')
        ax.contour(X1, X2, Z_plot, levels=15, colors='white', alpha=0.3, linewidths=0.5)

        # Mark global optimum (x1=0, x2=0)
        ax.scatter(0.02, 0.02, c='red', s=200, marker='*',
                  edgecolors='white', linewidths=2, zorder=10, label='Global Optimum')

        ax.set_xlabel('$x_1$', fontsize=LABEL_SIZE)
        ax.set_ylabel('$x_2$', fontsize=LABEL_SIZE)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_title(f'Park Function ($x_1$-$x_2$ plane)\n$x_3={x3_val:.2f}$, $x_4={x4_val:.2f}$, $f^*={F_STAR:.1f}$',
                    fontsize=TITLE_SIZE)

        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(contour, cax=cax)
        cbar.set_label('$\\log_{10}(f + 1)$', fontsize=LABEL_SIZE-1)

        plt.tight_layout()

        # Save frame to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, facecolor='white')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)

        if (frame + 1) % 10 == 0:
            print(f"  Frame {frame + 1}/{n_frames}")

    # Save as GIF
    output_file = 'park_x1x2_animation.gif'
    frames[0].save(
        output_file,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"Saved: {output_file}")


# =============================================================================
# Animation 2: x3-x4 plane with x1, x2 varying
# =============================================================================

def create_x3x4_animation():
    """Create animation of x3-x4 plane as x1, x2 vary"""

    # Create grid for x3-x4
    x3 = np.linspace(0, 1, n_points)
    x4 = np.linspace(0.01, 1, n_points)
    X3, X4 = np.meshgrid(x3, x4)

    # Values for x1, x2 to sweep through
    x1_values = np.linspace(0.01, 1, n_frames)
    x2_values = np.linspace(0, 1, n_frames)

    frames = []

    for frame in range(n_frames):
        fig, ax = plt.subplots(figsize=(8, 7))

        x1_val = x1_values[frame]
        x2_val = x2_values[frame]

        X_flat = np.column_stack([
            np.full(X3.size, x1_val),
            np.full(X3.size, x2_val),
            X3.ravel(),
            X4.ravel()
        ])
        Z = park_hf(X_flat).reshape(X3.shape)
        Z_plot = np.log10(Z + 1)

        contour = ax.contourf(X3, X4, Z_plot, levels=50, cmap='viridis')
        ax.contour(X3, X4, Z_plot, levels=15, colors='white', alpha=0.3, linewidths=0.5)

        # Mark global optimum (x3=0, x4=0)
        ax.scatter(0.02, 0.02, c='red', s=200, marker='*',
                  edgecolors='white', linewidths=2, zorder=10, label='Global Optimum')

        ax.set_xlabel('$x_3$', fontsize=LABEL_SIZE)
        ax.set_ylabel('$x_4$', fontsize=LABEL_SIZE)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_title(f'Park Function ($x_3$-$x_4$ plane)\n$x_1={x1_val:.2f}$, $x_2={x2_val:.2f}$, $f^*={F_STAR:.1f}$',
                    fontsize=TITLE_SIZE)

        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(contour, cax=cax)
        cbar.set_label('$\\log_{10}(f + 1)$', fontsize=LABEL_SIZE-1)

        plt.tight_layout()

        # Save frame to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, facecolor='white')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)

        if (frame + 1) % 10 == 0:
            print(f"  Frame {frame + 1}/{n_frames}")

    # Save as GIF
    output_file = 'park_x3x4_animation.gif'
    frames[0].save(
        output_file,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"Saved: {output_file}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("Creating Park function animations (GIF)...")
    print(f"  Grid resolution: {n_points}x{n_points}")
    print(f"  Frames: {n_frames}")
    print(f"  Duration: {duration}ms per frame")
    print()

    print("Animation 1: x1-x2 plane (x3, x4 varying)...")
    create_x1x2_animation()

    print("\nAnimation 2: x3-x4 plane (x1, x2 varying)...")
    create_x3x4_animation()

    print("\nDone!")
