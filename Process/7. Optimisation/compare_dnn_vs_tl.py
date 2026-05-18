"""
DNN From-Scratch vs Transfer Learning: Validation Loss Comparison
=================================================================
Case 1: Small HiFi data scenario (as in DNGO_bo.ipynb)

Compares VALIDATION LOSS on held-out HiFi data:
  1. GP baseline (sklearn, RBF kernel) on HiFi train only
  2. DNN trained from scratch on small HiFi train set only
  3. DNN with Transfer Learning: pretrain on LoFi -> finetune on small HiFi train set

Data:
  - LoFi (pretrain): 161 valid shortcut samples from DNGO runs (target: sc_CAPEX+sc_OPEX)
  - HiFi train: 26 samples from DNGO with valid shortcut cost (target: sc_CAPEX+sc_OPEX)
  - HiFi val: 316 samples from GP-BO with valid rigorous results (target: rg_CAPEX+rg_OPEX)

The val set uses RIGOROUS cost as ground truth -- this is the real HiFi objective.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from sklearn.metrics import mean_squared_error

# ============================================================
# Device setup
# ============================================================
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using Apple Silicon GPU (MPS)")
else:
    device = torch.device('cpu')
    print("Using CPU")

print(f"PyTorch version: {torch.__version__}")

# ============================================================
# Load data
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '3. Data')

input_cols_dngo = ['n1', 'Lr1', 'Hr1', 'Lr2', 'Hr2', 'T_hex', 'Lr3', 'Hr3']

# --- DNGO data (LoFi + small HiFi) ---
dfs = []
for fname in ['dngo_bo_results_251024_153407.csv',
              'dngo_bo_results_251024_153856.csv',
              'dngo_bo_results_251024_154640.csv']:
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        dfs.append(pd.read_csv(fpath))
        print(f"Loaded {fname}: {len(dfs[-1])} rows")

dngo = pd.concat(dfs, ignore_index=True)

# --- GP-BO data (large HiFi validation set) ---
gp_path = os.path.join(DATA_DIR, '6000_gp_bo_data_251024.csv')
gp = pd.read_csv(gp_path)
print(f"Loaded 6000_gp_bo_data_251024.csv: {len(gp)} rows")

# ============================================================
# Prepare LoFi train data (from DNGO, shortcut-only)
# ============================================================
lo = dngo[dngo['fidelity'] == 'low'].copy()
lo_valid = lo[lo['sc_CAPEX'] > 0].drop_duplicates(subset=input_cols_dngo)
lo_valid['target'] = lo_valid['sc_CAPEX'] + lo_valid['sc_OPEX']

X_lofi = lo_valid[input_cols_dngo].values.astype(np.float32)
y_lofi = lo_valid['target'].values.astype(np.float32)

print(f"\nLoFi train: {len(X_lofi)} samples")
print(f"  target range: [{y_lofi.min():.2f}, {y_lofi.max():.2f}], mean={y_lofi.mean():.2f}")

# ============================================================
# Prepare HiFi train data (from DNGO, small set = "case 1")
# ============================================================
hi = dngo[dngo['fidelity'] == 'high'].copy()
hi_train = hi[hi['sc_CAPEX'] > 0].drop_duplicates(subset=input_cols_dngo)
hi_train['target'] = hi_train['sc_CAPEX'] + hi_train['sc_OPEX']

X_hifi_train = hi_train[input_cols_dngo].values.astype(np.float32)
y_hifi_train = hi_train['target'].values.astype(np.float32)

print(f"\nHiFi train: {len(X_hifi_train)} samples")
print(f"  target range: [{y_hifi_train.min():.2f}, {y_hifi_train.max():.2f}], mean={y_hifi_train.mean():.2f}")

# ============================================================
# Prepare HiFi VALIDATION data (from GP-BO, rigorous results)
# ============================================================
gp_valid = gp[gp['29'].isna()].copy()
gp_valid = gp_valid[gp_valid['22'].notna() & (gp_valid['22'] > 0)]

X_hifi_val = gp_valid[['0','1','2','3','4','5','6','7']].values.astype(np.float32)
y_hifi_val_cost = (gp_valid['22'] + gp_valid['23']).values.astype(np.float32)

print(f"\nHiFi val (rigorous): {len(X_hifi_val)} samples")
print(f"  target (rg_CAPEX+rg_OPEX) range: [{y_hifi_val_cost.min():.2f}, {y_hifi_val_cost.max():.2f}], mean={y_hifi_val_cost.mean():.2f}")

# ============================================================
# Normalize inputs to [0, 1]
# ============================================================
bounds = np.array([
    [10, 50], [0, 0.9999], [0, 0.9999], [0, 0.9999],
    [0, 0.9999], [273, 350], [0, 0.9999], [0, 0.9999],
])

def normalize(X, bounds):
    return (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

X_lofi_n = normalize(X_lofi, bounds)
X_hifi_train_n = normalize(X_hifi_train, bounds)
X_hifi_val_n = normalize(X_hifi_val, bounds)

# ============================================================
# GP Baseline (sklearn, RBF kernel)
# ============================================================
print("\n" + "=" * 70)
print("GP BASELINE (sklearn, ConstantKernel * RBF)")
print("=" * 70)

gp_train_mses = []
gp_val_mses = []

for seed in range(10):
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
    gp_model = GaussianProcessRegressor(kernel=kernel, random_state=seed,
                                         n_restarts_optimizer=5, alpha=1e-6)
    gp_model.fit(X_hifi_train_n, y_hifi_train)

    gp_train_pred = gp_model.predict(X_hifi_train_n)
    gp_val_pred = gp_model.predict(X_hifi_val_n)

    gp_train_mse = mean_squared_error(y_hifi_train, gp_train_pred)
    gp_val_mse = mean_squared_error(y_hifi_val_cost, gp_val_pred)

    gp_train_mses.append(gp_train_mse)
    gp_val_mses.append(gp_val_mse)
    print(f"  GP seed {seed}: train MSE={gp_train_mse:.4f}, val MSE={gp_val_mse:.4f}")

gp_train_mse_mean = np.mean(gp_train_mses)
gp_train_mse_std = np.std(gp_train_mses)
gp_val_mse_mean = np.mean(gp_val_mses)
gp_val_mse_std = np.std(gp_val_mses)

print(f"\nGP train MSE: {gp_train_mse_mean:.4f} +/- {gp_train_mse_std:.4f}")
print(f"GP val MSE:   {gp_val_mse_mean:.4f} +/- {gp_val_mse_std:.4f}")

# ============================================================
# Model definition (matches DNGO_bo.ipynb)
# ============================================================
class TransferLearningDNN(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.out_layer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        return self.out_layer(self.feature_net(x))

# ============================================================
# Config
# ============================================================
INPUT_DIM = 8
HIDDEN_DIM = 64
PRETRAIN_EPOCHS = 200
FINETUNE_EPOCHS = 100
PRETRAIN_LR = 1e-3
FINETUNE_LR = 1e-4
NUM_SEEDS = 10

# Pre-convert val tensors
X_val_t = torch.tensor(X_hifi_val_n, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_hifi_val_cost, dtype=torch.float32).view(-1, 1).to(device)
val_loss_fn = nn.MSELoss()

def eval_val(model):
    """Compute validation MSE on held-out HiFi rigorous data."""
    model.eval()
    with torch.no_grad():
        pred = model(X_val_t)
        loss = val_loss_fn(pred, y_val_t)
    model.train()
    return loss.item()

# ============================================================
# Training functions with val loss tracking
# ============================================================
def train_from_scratch(X, y, epochs, lr, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TransferLearningDNN(INPUT_DIM, HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

    train_losses, val_losses = [], []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        val_losses.append(eval_val(model))
    return train_losses, val_losses


def train_transfer_learning(X_lo, y_lo, X_hi, y_hi,
                            pretrain_epochs, finetune_epochs,
                            pretrain_lr, finetune_lr, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TransferLearningDNN(INPUT_DIM, HIDDEN_DIM).to(device)
    loss_fn = nn.MSELoss()

    # Phase 1: Pretrain on LoFi
    optimizer_pre = optim.Adam(model.parameters(), lr=pretrain_lr)
    X_lo_t = torch.tensor(X_lo, dtype=torch.float32).to(device)
    y_lo_t = torch.tensor(y_lo, dtype=torch.float32).view(-1, 1).to(device)

    pre_train_losses, pre_val_losses = [], []
    model.train()
    for epoch in range(pretrain_epochs):
        optimizer_pre.zero_grad()
        pred = model(X_lo_t)
        loss = loss_fn(pred, y_lo_t)
        loss.backward()
        optimizer_pre.step()
        pre_train_losses.append(loss.item())
        pre_val_losses.append(eval_val(model))

    # Phase 2: Finetune on HiFi
    optimizer_ft = optim.Adam(model.parameters(), lr=finetune_lr)
    X_hi_t = torch.tensor(X_hi, dtype=torch.float32).to(device)
    y_hi_t = torch.tensor(y_hi, dtype=torch.float32).view(-1, 1).to(device)

    ft_train_losses, ft_val_losses = [], []
    model.train()
    for epoch in range(finetune_epochs):
        optimizer_ft.zero_grad()
        pred = model(X_hi_t)
        loss = loss_fn(pred, y_hi_t)
        loss.backward()
        optimizer_ft.step()
        ft_train_losses.append(loss.item())
        ft_val_losses.append(eval_val(model))

    return pre_train_losses, pre_val_losses, ft_train_losses, ft_val_losses


# ============================================================
# Run experiments
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT: DNN From-Scratch vs Transfer Learning (VAL LOSS)")
print("=" * 70)
print(f"LoFi train: {len(X_lofi_n)} | HiFi train: {len(X_hifi_train_n)} | HiFi val: {len(X_hifi_val_n)}")
print(f"Seeds: {NUM_SEEDS}")

all_scratch_train = []
all_scratch_val = []
all_pre_train = []
all_pre_val = []
all_ft_train = []
all_ft_val = []

for seed in range(NUM_SEEDS):
    print(f"\n--- Seed {seed} ---")

    total_epochs = PRETRAIN_EPOCHS + FINETUNE_EPOCHS
    s_train, s_val = train_from_scratch(
        X_hifi_train_n, y_hifi_train, total_epochs, PRETRAIN_LR, seed
    )
    all_scratch_train.append(s_train)
    all_scratch_val.append(s_val)
    print(f"  Scratch: train={s_train[-1]:.4f}, val={s_val[-1]:.4f}, best_val={min(s_val):.4f}")

    pt, pv, ft, fv = train_transfer_learning(
        X_lofi_n, y_lofi,
        X_hifi_train_n, y_hifi_train,
        PRETRAIN_EPOCHS, FINETUNE_EPOCHS,
        PRETRAIN_LR, FINETUNE_LR, seed
    )
    all_pre_train.append(pt)
    all_pre_val.append(pv)
    all_ft_train.append(ft)
    all_ft_val.append(fv)
    print(f"  TL pre:  train={pt[-1]:.4f}, val={pv[-1]:.4f}")
    print(f"  TL ft:   train={ft[-1]:.4f}, val={fv[-1]:.4f}, best_val={min(fv):.4f}")

# Convert to arrays
all_scratch_train = np.array(all_scratch_train)
all_scratch_val = np.array(all_scratch_val)
all_pre_train = np.array(all_pre_train)
all_pre_val = np.array(all_pre_val)
all_ft_train = np.array(all_ft_train)
all_ft_val = np.array(all_ft_val)

# TL val: concatenate pretrain val + finetune val
tl_val_full = np.concatenate([all_pre_val, all_ft_val], axis=1)

# ============================================================
# Plotting — Combined only (Train + Val side-by-side)
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def plot_with_ci(ax, epochs, mean, std, color, label, ls='-'):
    ax.plot(epochs, mean, color=color, linewidth=2, linestyle=ls, label=label)
    ax.fill_between(epochs, np.maximum(mean - std, 0), mean + std, color=color, alpha=0.12)

total_epochs = PRETRAIN_EPOCHS + FINETUNE_EPOCHS
epochs_all = np.arange(1, total_epochs + 1)
epochs_pre = np.arange(1, PRETRAIN_EPOCHS + 1)
epochs_ft = np.arange(PRETRAIN_EPOCHS + 1, total_epochs + 1)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Left: Train Loss ---
ax = axes[0]
plot_with_ci(ax, epochs_all, all_scratch_train.mean(0), all_scratch_train.std(0),
             'green', 'From-scratch (HiFi only)')
plot_with_ci(ax, epochs_pre, all_pre_train.mean(0), all_pre_train.std(0),
             'blue', 'TL: Pretrain (LoFi)')
plot_with_ci(ax, epochs_ft, all_ft_train.mean(0), all_ft_train.std(0),
             'red', 'TL: Finetune (HiFi)')
ax.axhline(y=gp_train_mse_mean, color='orange', ls='--', linewidth=2,
           label=f'GP baseline (train)')
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Train MSE Loss', fontsize=13)
ax.set_title('(a) Training Loss', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# --- Right: Val Loss ---
ax = axes[1]
plot_with_ci(ax, epochs_all, all_scratch_val.mean(0), all_scratch_val.std(0),
             'green', 'From-scratch (HiFi only)')
plot_with_ci(ax, epochs_pre, all_pre_val.mean(0), all_pre_val.std(0),
             'blue', 'TL: Pretrain (LoFi)')
plot_with_ci(ax, epochs_ft, all_ft_val.mean(0), all_ft_val.std(0),
             'red', 'TL: Finetune (HiFi)')
ax.axhline(y=gp_val_mse_mean, color='orange', ls='--', linewidth=2,
           label=f'GP baseline (val)')
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Val MSE Loss (HiFi rigorous)', fontsize=13)
ax.set_title('(b) Validation Loss on Held-out HiFi Data', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_combined = os.path.join(DATA_DIR, f'dnn_vs_tl_combined_{timestamp}.png')
plt.savefig(out_combined, dpi=200, bbox_inches='tight')
print(f"\nCombined plot saved to: {out_combined}")

out_pdf = os.path.join(DATA_DIR, f'dnn_vs_tl_combined_{timestamp}.pdf')
plt.savefig(out_pdf, format='pdf', dpi=600, bbox_inches='tight')
print(f"Combined PDF saved to: {out_pdf}")
plt.close(fig)

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# --- GP Baseline ---
print(f"\n--- GP Baseline (sklearn, C*RBF, 26 HiFi train) ---")
print(f"GP train MSE:  {gp_train_mse_mean:.4f} +/- {gp_train_mse_std:.4f}")
print(f"GP val MSE:    {gp_val_mse_mean:.4f} +/- {gp_val_mse_std:.4f}")

# --- Train Loss ---
scratch_final_train = all_scratch_train[:, -1]
ft_final_train = all_ft_train[:, -1]
pre_final_train = all_pre_train[:, -1]

print(f"\n--- Train Loss (MSE on training data) ---")
print(f"From-scratch final train:  {scratch_final_train.mean():.4f} +/- {scratch_final_train.std():.4f}")
print(f"TL pretrain final train:   {pre_final_train.mean():.4f} +/- {pre_final_train.std():.4f}")
print(f"TL finetune final train:   {ft_final_train.mean():.4f} +/- {ft_final_train.std():.4f}")

# --- Val Loss ---
scratch_best_val = all_scratch_val.min(axis=1)
scratch_final_val = all_scratch_val[:, -1]
ft_best_val = all_ft_val.min(axis=1)
ft_final_val = all_ft_val[:, -1]
tl_best_val = tl_val_full.min(axis=1)

print(f"\n--- Validation Loss (MSE on 316 HiFi rigorous samples) ---")
print(f"From-scratch best val:   {scratch_best_val.mean():.4f} +/- {scratch_best_val.std():.4f}")
print(f"From-scratch final val:  {scratch_final_val.mean():.4f} +/- {scratch_final_val.std():.4f}")
print(f"TL finetune best val:    {ft_best_val.mean():.4f} +/- {ft_best_val.std():.4f}")
print(f"TL finetune final val:   {ft_final_val.mean():.4f} +/- {ft_final_val.std():.4f}")
print(f"TL overall best val:     {tl_best_val.mean():.4f} +/- {tl_best_val.std():.4f}")

# --- Comparison ---
print(f"\n--- Comparison ---")
if ft_best_val.mean() < scratch_best_val.mean():
    imp = (1 - ft_best_val.mean() / scratch_best_val.mean()) * 100
    print(f"POSITIVE TRANSFER: TL best val is {imp:.1f}% lower than from-scratch best val")
else:
    deg = (ft_best_val.mean() / scratch_best_val.mean() - 1) * 100
    print(f"NEGATIVE TRANSFER: TL best val is {deg:.1f}% higher than from-scratch best val")

if ft_best_val.mean() < gp_val_mse_mean:
    imp_gp = (1 - ft_best_val.mean() / gp_val_mse_mean) * 100
    print(f"TL best val is {imp_gp:.1f}% lower than GP baseline val")
else:
    deg_gp = (ft_best_val.mean() / gp_val_mse_mean - 1) * 100
    print(f"TL best val is {deg_gp:.1f}% higher than GP baseline val")

if scratch_best_val.mean() < gp_val_mse_mean:
    imp_gp2 = (1 - scratch_best_val.mean() / gp_val_mse_mean) * 100
    print(f"From-scratch best val is {imp_gp2:.1f}% lower than GP baseline val")
else:
    deg_gp2 = (scratch_best_val.mean() / gp_val_mse_mean - 1) * 100
    print(f"From-scratch best val is {deg_gp2:.1f}% higher than GP baseline val")

tl_target = ft_best_val.mean()
scratch_val_mean = all_scratch_val.mean(0)
reached = np.where(scratch_val_mean <= tl_target)[0]
if len(reached) > 0:
    print(f"From-scratch reaches TL best val level at epoch {reached[0] + 1}")
else:
    print(f"From-scratch NEVER reaches TL best val level within {total_epochs} epochs")

print(f"\nDone.")
