"""
Two Model Architecture Visualization

1. DNGO Model (GP Mimic): TransferLearningDNN + BayesianLinearRegression
2. BNN Intermediate Model: TransferLearningBNN (consistent_bnn / dngo_style)

Author: Generated for model architecture comparison
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up figure
fig, axes = plt.subplots(1, 2, figsize=(18, 12))

# Color scheme
colors = {
    'input': '#E8F5E9',       # Light green
    'hidden': '#BBDEFB',       # Light blue
    'output': '#FFECB3',       # Light yellow
    'bayesian': '#F3E5F5',     # Light purple
    'freeze': '#FFCDD2',       # Light red
    'arrow': '#424242',        # Dark gray
    'text': '#212121',         # Black
    'pretrain': '#4CAF50',     # Green
    'finetune': '#2196F3',     # Blue
    'variational': '#9C27B0',  # Purple
}


def draw_layer_box(ax, x, y, width, height, label, color, sublabel=None, fontsize=10):
    """Draw a layer box with label"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=colors['text'])
    if sublabel:
        ax.text(x, y - height/2 - 0.15, sublabel, ha='center', va='top',
                fontsize=8, color='gray')


def draw_arrow(ax, start, end, color='black', style='->'):
    """Draw an arrow between two points"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=1.5))


# ============================================
# Model 1: DNGO (GP Mimic)
# ============================================
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 14)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('Model 1: DNGO (GP Mimic)\nTransferLearningDNN + BayesianLinearRegression',
              fontsize=14, fontweight='bold', pad=20)

# Input Layer
draw_layer_box(ax1, 5, 13, 3, 0.8, 'Input X', colors['input'],
               sublabel='Perovskite Features (D-dim)')

# Feature Network (TransferLearningDNN)
draw_layer_box(ax1, 5, 11, 4, 1.2, 'Feature Network', colors['hidden'],
               sublabel='Deterministic DNN')

# Layer details for feature network
layer_y = 9.5
draw_layer_box(ax1, 3, layer_y, 2.5, 0.7, 'Layer 1', colors['hidden'],
               sublabel='Linear + ReLU/Tanh')
draw_layer_box(ax1, 7, layer_y, 2.5, 0.7, 'Layer 2', colors['hidden'],
               sublabel='Linear + ReLU/Tanh')

# Arrows from Feature Network to Layers
draw_arrow(ax1, (4, 10.4), (3, 9.9))
draw_arrow(ax1, (6, 10.4), (7, 9.9))

# Hidden features
draw_layer_box(ax1, 5, 8, 3.5, 0.8, 'Hidden Features', colors['hidden'],
               sublabel='Learned Representation (H-dim)')

# Arrows connecting layers
draw_arrow(ax1, (3, 9.1), (4.5, 8.4))
draw_arrow(ax1, (7, 9.1), (5.5, 8.4))

# Output Layer (Linear, no bias)
draw_layer_box(ax1, 5, 6.5, 3, 0.8, 'Out Layer', colors['output'],
               sublabel='Linear (no bias)')

# Bayesian Linear Regression
draw_layer_box(ax1, 5, 4.8, 4, 1.2, 'Bayesian Linear\nRegression (BLR)', colors['bayesian'],
               sublabel='Posterior: P(w|X,y)')

# BLR details
ax1.text(5, 3.4, r'$P(w|D) = \mathcal{N}(\mu_N, \Sigma_N)$',
         ha='center', fontsize=10, style='italic')
ax1.text(5, 2.9, r'Prior: $\alpha=1.0$, Noise: $\beta=25.0$',
         ha='center', fontsize=9, color='gray')

# Output with uncertainty
draw_layer_box(ax1, 5, 1.8, 3.5, 0.9, r'Output: $\mu$, $\sigma^2$', colors['bayesian'],
               sublabel='Mean + Variance')

# Arrows
draw_arrow(ax1, (5, 12.6), (5, 11.6))
draw_arrow(ax1, (5, 8.4), (5, 6.9))
draw_arrow(ax1, (5, 6.1), (5, 5.4))
draw_arrow(ax1, (5, 4.2), (5, 2.3))

# Training phases annotation
ax1.add_patch(FancyBboxPatch((0.3, 8.5), 1.8, 5, boxstyle="round,pad=0.05",
                              facecolor='none', edgecolor=colors['pretrain'],
                              linewidth=2, linestyle='--'))
ax1.text(1.2, 13.8, 'Pretrain\n(Low-fi)', ha='center', fontsize=9,
         color=colors['pretrain'], fontweight='bold')

ax1.add_patch(FancyBboxPatch((0.3, 1.2), 1.8, 6.8, boxstyle="round,pad=0.05",
                              facecolor='none', edgecolor=colors['finetune'],
                              linewidth=2, linestyle='--'))
ax1.text(1.2, 8.3, 'Finetune\n(High-fi)', ha='center', fontsize=9,
         color=colors['finetune'], fontweight='bold')

# Freeze annotation
ax1.add_patch(FancyBboxPatch((7.8, 9), 2, 3.5, boxstyle="round,pad=0.05",
                              facecolor=colors['freeze'], edgecolor='red',
                              linewidth=1, alpha=0.3))
ax1.text(8.8, 12.8, 'Freeze\nOption', ha='center', fontsize=8,
         color='red', fontweight='bold')
ax1.text(8.8, 12.0, 'unfreeze_ratio\n0.0~1.0', ha='center', fontsize=7, color='darkred')


# ============================================
# Model 2: BNN Intermediate Model
# ============================================
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 14)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('Model 2: BNN Intermediate\nTransferLearningBNN (Bayesian Neural Network)',
              fontsize=14, fontweight='bold', pad=20)

# Input Layer
draw_layer_box(ax2, 5, 13, 3, 0.8, 'Input X', colors['input'],
               sublabel='Perovskite Features (D-dim)')

# Variational Layers (BNN)
draw_layer_box(ax2, 5, 11, 4.5, 1.2, 'Variational Hidden\nLayers', colors['variational'],
               sublabel='Bayesian by Backprop')

# Layer details
layer_y = 9.2
draw_layer_box(ax2, 2.5, layer_y, 2.8, 0.9, 'VarLinear 1', colors['variational'],
               sublabel=r'$w \sim \mathcal{N}(\mu, \sigma^2)$')
draw_layer_box(ax2, 7.5, layer_y, 2.8, 0.9, 'VarLinear 2', colors['variational'],
               sublabel=r'$w \sim \mathcal{N}(\mu, \sigma^2)$')

# Arrows from main block to layers
draw_arrow(ax2, (3.5, 10.4), (2.5, 9.7))
draw_arrow(ax2, (6.5, 10.4), (7.5, 9.7))

# Scale Mixture Prior
draw_layer_box(ax2, 5, 7.5, 4.2, 1.1, 'Scale Mixture Prior', colors['variational'],
               sublabel='(Blundell et al. 2015)')
ax2.text(5, 6.2, r'$P(w) = \pi\mathcal{N}(0, \sigma_1^2) + (1-\pi)\mathcal{N}(0, \sigma_2^2)$',
         ha='center', fontsize=9, style='italic')
ax2.text(5, 5.7, r'$\pi=0.5, \sigma_1=1.0, \sigma_2=0.002$',
         ha='center', fontsize=8, color='gray')

# Variational Output Layer
draw_layer_box(ax2, 5, 4.5, 3.5, 0.9, 'VarLinear Output', colors['variational'],
               sublabel='Mean + Noise Var')

# Noise modeling
draw_layer_box(ax2, 5, 3.2, 3.2, 0.7, 'Noise Model', colors['output'],
               sublabel='Homoscedastic / Heteroscedastic')

# Loss function
draw_layer_box(ax2, 5, 1.8, 4, 1.1, 'ELBO Loss', colors['bayesian'],
               sublabel='NLL + KL Divergence')
ax2.text(5, 0.5, r'$\mathcal{L} = \mathbb{E}_q[\log P(y|x,w)] - KL(q(w)||P(w))$',
         ha='center', fontsize=9, style='italic')

# Arrows
draw_arrow(ax2, (5, 12.6), (5, 11.6))
draw_arrow(ax2, (2.5, 8.75), (4, 8.0))
draw_arrow(ax2, (7.5, 8.75), (6, 8.0))
draw_arrow(ax2, (5, 6.95), (5, 4.95))
draw_arrow(ax2, (5, 4.05), (5, 3.6))
draw_arrow(ax2, (5, 2.85), (5, 2.35))

# Training modes
ax2.add_patch(FancyBboxPatch((0.2, 6.8), 2.2, 6.8, boxstyle="round,pad=0.05",
                              facecolor='none', edgecolor=colors['pretrain'],
                              linewidth=2, linestyle='--'))
ax2.text(1.3, 13.9, 'consistent_bnn', ha='center', fontsize=8,
         color=colors['pretrain'], fontweight='bold')
ax2.text(1.3, 13.4, 'Full BNN\nPretrain + Finetune', ha='center', fontsize=7,
         color=colors['pretrain'])

ax2.add_patch(FancyBboxPatch((7.6, 6.8), 2.2, 6.8, boxstyle="round,pad=0.05",
                              facecolor='none', edgecolor=colors['finetune'],
                              linewidth=2, linestyle='--'))
ax2.text(8.7, 13.9, 'dngo_style', ha='center', fontsize=8,
         color=colors['finetune'], fontweight='bold')
ax2.text(8.7, 13.4, 'Det. Feature\n+ BNN Head', ha='center', fontsize=7,
         color=colors['finetune'])

# KL Warmup annotation
ax2.text(8.5, 2.0, 'KL Warmup\n(10 epochs)', ha='center', fontsize=8,
         color='purple', style='italic')

plt.tight_layout()
plt.savefig('/cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/Pure_TL_BO/model_architectures.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/Pure_TL_BO/model_architectures.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved: model_architectures.png, model_architectures.pdf")


# ============================================
# Detailed Component Diagram
# ============================================
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 14))

# ---- Panel 1: DNGO Training Flow ----
ax = axes2[0, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('DNGO: Training Flow', fontsize=12, fontweight='bold')

# Pretrain phase
draw_layer_box(ax, 2.5, 8.5, 3, 0.8, 'Low-fi Data', colors['input'])
draw_layer_box(ax, 2.5, 6.5, 3, 1.2, 'Feature Net\n+ Out Layer', colors['hidden'])
draw_layer_box(ax, 2.5, 4.5, 3, 0.8, 'MSE Loss', colors['output'])
draw_arrow(ax, (2.5, 8.1), (2.5, 7.1))
draw_arrow(ax, (2.5, 5.9), (2.5, 4.9))
ax.text(2.5, 3.5, 'Pretrain', ha='center', fontsize=10, fontweight='bold', color=colors['pretrain'])

# Finetune phase
draw_layer_box(ax, 7.5, 8.5, 3, 0.8, 'High-fi Data', colors['input'])
draw_layer_box(ax, 7.5, 6.5, 3, 1.2, 'Feature Net\n(Frozen)', colors['freeze'])
draw_layer_box(ax, 7.5, 4.5, 3, 0.8, 'BLR fit()', colors['bayesian'])
draw_arrow(ax, (7.5, 8.1), (7.5, 7.1))
draw_arrow(ax, (7.5, 5.9), (7.5, 4.9))
ax.text(7.5, 3.5, 'Finetune', ha='center', fontsize=10, fontweight='bold', color=colors['finetune'])

# Transfer arrow
draw_arrow(ax, (4, 6.5), (6, 6.5), color=colors['arrow'], style='->')
ax.text(5, 6.8, 'Transfer', ha='center', fontsize=9, color='gray')

# ---- Panel 2: BNN Training Flow ----
ax = axes2[0, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('BNN: Training Flow (consistent_bnn)', fontsize=12, fontweight='bold')

# Pretrain phase
draw_layer_box(ax, 2.5, 8.5, 3, 0.8, 'Low-fi Data', colors['input'])
draw_layer_box(ax, 2.5, 6.5, 3, 1.2, 'Full BNN\n(All Layers)', colors['variational'])
draw_layer_box(ax, 2.5, 4.5, 3, 0.8, 'ELBO Loss', colors['bayesian'])
draw_arrow(ax, (2.5, 8.1), (2.5, 7.1))
draw_arrow(ax, (2.5, 5.9), (2.5, 4.9))
ax.text(2.5, 3.5, 'Pretrain', ha='center', fontsize=10, fontweight='bold', color=colors['pretrain'])

# Finetune phase
draw_layer_box(ax, 7.5, 8.5, 3, 0.8, 'High-fi Data', colors['input'])
draw_layer_box(ax, 7.5, 6.5, 3, 1.2, 'Same BNN\n(Warm Start)', colors['variational'])
draw_layer_box(ax, 7.5, 4.5, 3, 0.8, 'ELBO Loss', colors['bayesian'])
draw_arrow(ax, (7.5, 8.1), (7.5, 7.1))
draw_arrow(ax, (7.5, 5.9), (7.5, 4.9))
ax.text(7.5, 3.5, 'Finetune', ha='center', fontsize=10, fontweight='bold', color=colors['finetune'])

# Transfer arrow
draw_arrow(ax, (4, 6.5), (6, 6.5), color=colors['arrow'], style='->')
ax.text(5, 6.8, 'Warm Start', ha='center', fontsize=9, color='gray')

# ---- Panel 3: DNGO Uncertainty Quantification ----
ax = axes2[1, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('DNGO: Uncertainty Quantification', fontsize=12, fontweight='bold')

draw_layer_box(ax, 5, 8.5, 4, 0.8, 'Test Input x*', colors['input'])
draw_layer_box(ax, 5, 6.5, 4, 1, 'Feature Extraction\nφ(x*)', colors['hidden'])
draw_layer_box(ax, 5, 4.3, 5, 1.5, 'BLR Posterior\nPredict', colors['bayesian'])

ax.text(5, 2.5, r'$\mu(x^*) = \phi(x^*)^T \mu_N$', ha='center', fontsize=11)
ax.text(5, 1.8, r'$\sigma^2(x^*) = \frac{1}{\beta} + \phi(x^*)^T \Sigma_N \phi(x^*)$', ha='center', fontsize=11)

draw_arrow(ax, (5, 8.1), (5, 7.0))
draw_arrow(ax, (5, 6.0), (5, 5.0))

# ---- Panel 4: BNN Uncertainty Quantification ----
ax = axes2[1, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('BNN: Uncertainty Quantification', fontsize=12, fontweight='bold')

draw_layer_box(ax, 5, 8.5, 4, 0.8, 'Test Input x*', colors['input'])
draw_layer_box(ax, 5, 6.5, 4.5, 1.2, 'MC Sampling\n(N=100)', colors['variational'])
draw_layer_box(ax, 5, 4.3, 5, 1.5, 'Aggregate\nPredictions', colors['bayesian'])

ax.text(5, 2.5, r'Epistemic: $Var_{w}[f_w(x^*)]$', ha='center', fontsize=10)
ax.text(5, 1.8, r'Aleatoric: $\mathbb{E}_w[\sigma^2_{noise}]$', ha='center', fontsize=10)
ax.text(5, 1.1, r'Total: Epistemic + Aleatoric', ha='center', fontsize=10, color='gray')

draw_arrow(ax, (5, 8.1), (5, 7.1))
draw_arrow(ax, (5, 5.9), (5, 5.0))

plt.tight_layout()
plt.savefig('/cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/Pure_TL_BO/model_details.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/Pure_TL_BO/model_details.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved: model_details.png, model_details.pdf")


# ============================================
# Summary Comparison Table
# ============================================
fig3, ax3 = plt.subplots(figsize=(14, 8))
ax3.axis('off')
ax3.set_title('Model Architecture Comparison Summary', fontsize=16, fontweight='bold', pad=20)

# Table data
comparison_data = [
    ['Component', 'DNGO (GP Mimic)', 'BNN Intermediate'],
    ['Feature Extractor', 'Deterministic DNN\n(TransferLearningDNN)', 'Variational Layers\n(VariationalLinear)'],
    ['Uncertainty Model', 'Bayesian Linear Regression\n(Closed-form posterior)', 'Full BNN\n(Monte Carlo sampling)'],
    ['Prior', 'Gaussian\nα=1.0 (weight), β=25.0 (noise)', 'Scale Mixture Prior\nπN(0,σ₁²) + (1-π)N(0,σ₂²)'],
    ['Training Pretrain', 'MSE Loss\n(Full network)', 'ELBO Loss\n(Full BNN or Det. features)'],
    ['Training Finetune', 'BLR fit()\n(Features frozen)', 'ELBO Loss\n(Warm start or BNN head)'],
    ['Uncertainty Type', 'Predictive variance\n(Analytical)', 'Epistemic + Aleatoric\n(Monte Carlo)'],
    ['Freeze Support', 'unfreeze_ratio: 0.0~1.0\n(Layer-wise control)', 'unfreeze_ratio: 0.0~1.0\n(BNN layers)'],
    ['Online Learning', 'Sherman-Morrison update\n(BLR incremental)', 'Experience replay\n+ KL regularization'],
    ['Transfer Modes', 'Single mode\n(Det. Feature → BLR)', 'consistent_bnn\ndngo_style'],
]

# Draw table
table = ax3.table(
    cellText=comparison_data,
    loc='center',
    cellLoc='center',
    colWidths=[0.2, 0.4, 0.4],
)

# Style table
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)

# Header styling
for j in range(3):
    cell = table[(0, j)]
    cell.set_facecolor('#424242')
    cell.set_text_props(color='white', fontweight='bold')

# Row styling
for i in range(1, len(comparison_data)):
    for j in range(3):
        cell = table[(i, j)]
        if j == 0:
            cell.set_facecolor('#E0E0E0')
            cell.set_text_props(fontweight='bold')
        elif j == 1:
            cell.set_facecolor('#E3F2FD')
        else:
            cell.set_facecolor('#F3E5F5')

plt.savefig('/cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/Pure_TL_BO/model_comparison_table.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/Pure_TL_BO/model_comparison_table.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved: model_comparison_table.png, model_comparison_table.pdf")

print("\n✅ All visualizations generated successfully!")
print("\nFiles created:")
print("  1. model_architectures.png/pdf - Main architecture diagrams")
print("  2. model_details.png/pdf - Training flow and uncertainty quantification")
print("  3. model_comparison_table.png/pdf - Feature comparison table")
