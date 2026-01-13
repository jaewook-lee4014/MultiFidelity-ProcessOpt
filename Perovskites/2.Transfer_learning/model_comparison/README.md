# Model Comparison: Multi-Fidelity Transfer Learning for Perovskite Bandgap Prediction

This directory contains comprehensive experiments comparing various multi-fidelity surrogate models and transfer learning techniques for perovskite bandgap prediction.

## Overview

The goal is to predict high-fidelity (HSE06) bandgap values using limited expensive data while leveraging abundant low-fidelity (GGA) data through transfer learning approaches.

### Data Configuration
- **Low-Fidelity (LF)**: 72 GGA bandgap calculations (Cost = 1)
- **High-Fidelity (HF)**: 9 HSE06 bandgap calculations (Cost = 8)
- **Total compositions**: 72 perovskite compositions (ABX3 structure)
- **Input features**: 3-dimensional labels [organic, cation, anion]

---

## Model Taxonomy

| Model Name | Architecture | UQ Method | Fidelity | Knowledge Transfer | LF Augmentation | Training Scheme |
|------------|--------------|-----------|----------|-------------------|-----------------|-----------------|
| **Standard GP** | Gaussian Process | GP Posterior | Single (HF only) | None | No | Direct fit |
| **MFGP** | Multi-Fidelity GP | GP Posterior | Multi-Fidelity | Kernel correlation | Yes (fidelity feature) | Joint training |
| **DNGO-Base** | DNN + BLR | BLR Posterior | Single (HF only) | None | No | Direct fit |
| **DNGO-Pretrain** | DNN + BLR | BLR Posterior | Multi-Fidelity | Feature extraction | No | Pretrain LF → Finetune HF |
| **DNGO-Joint** | DNN + BLR | BLR Posterior | Multi-Fidelity | Shared features | Yes (joint loss) | Joint training (L_LF + L_HF) |
| **DNGO-Gradient** | DNN + BLR | BLR Posterior | Multi-Fidelity | Gradient scaling | Yes (scaled gradient) | Joint with gradient control |

### Extended Transfer Learning Methods (Advanced)

| Model Name | Architecture | UQ Method | Transfer Mechanism | Key Hyperparameter |
|------------|--------------|-----------|-------------------|-------------------|
| **DNGO-KD** | DNN + BLR | BLR Posterior | Knowledge Distillation (soft labels) | alpha_kd, temperature |
| **DNGO-DA** | DNN + BLR | BLR Posterior | Domain Adaptation (MMD loss) | lambda_mmd, bandwidth |
| **MF-DNN-Soft** | DNN + BLR | BLR Posterior | Soft Parameter Sharing | lambda_soft, alpha |
| **DNGO-PL** | DNN + BLR | BLR Posterior | Pseudo-Labeling | confidence_threshold |
| **DNGO-Adapter** | DNN + Adapters | BLR Posterior | Adapter Layers (frozen backbone) | bottleneck_dim |
| **DNGO-MAML** | DNN + BLR | BLR Posterior | Meta-Learning (MAML) | inner_lr, outer_lr, n_tasks |

---

## Model Descriptions

### 1. Standard GP (Baseline)
Single-fidelity Gaussian Process trained only on high-fidelity data. Serves as baseline for multi-fidelity approaches.

### 2. MFGP (Multi-Fidelity GP)
Uses BoTorch's `SingleTaskMultiFidelityGP` to learn correlations between LF and HF data through a shared kernel with fidelity indicator.

### 3. DNGO-Base
Deep Network for Global Optimization. DNN feature extractor + Bayesian Linear Regression (BLR) head for uncertainty quantification. Trained only on HF data.

### 4. DNGO-Pretrain (Feature Transfer)
Two-stage transfer learning:
1. **Stage 1**: Pretrain DNN on abundant LF data
2. **Stage 2**: Freeze/finetune features, train BLR on HF data

### 5. DNGO-Joint (Multi-task Learning)
Joint training with combined loss:
```
L_total = (1 - alpha) * L_LF + alpha * L_HF
```
Both LF and HF networks share features and are trained simultaneously.

### 6. DNGO-Gradient (Gradient Scaling)
Advanced joint training with controlled gradient flow:
```
L_total = (1 - alpha) * L_LF + alpha * L_HF
grad_LF = gradient_scale * original_grad
```
Gradient scaling factor controls how much LF task influences shared representations.

### 7. DNGO-KD (Knowledge Distillation)
Teacher-student framework:
- **Teacher**: LF network trained on abundant LF data
- **Student**: HF network learns from both hard labels (HF targets) and soft labels (teacher predictions)
```
L_total = (1 - alpha_kd) * L_hard + alpha_kd * L_soft
```

### 8. DNGO-DA (Domain Adaptation with MMD)
Aligns feature distributions between LF and HF domains using Maximum Mean Discrepancy:
```
L_total = L_task + lambda_mmd * MMD(features_LF, features_HF)
```

### 9. MF-DNN-Soft (Soft Parameter Sharing)
Instead of hard parameter sharing, regularizes LF and HF network parameters to be similar:
```
L_total = L_LF + L_HF + lambda_soft * ||theta_LF - theta_HF||^2
```

### 10. DNGO-PL (Pseudo-Labeling)
Generates pseudo-HF labels for LF data points:
1. Train LF model
2. Estimate LF→HF offset from available HF data
3. Generate pseudo-HF labels: `y_pseudo = y_LF + offset`
4. Train HF model with real + pseudo labels

### 11. DNGO-Adapter (Adapter-based Transfer)
Parameter-efficient transfer learning:
1. Pretrain backbone on LF data
2. Freeze backbone, insert small adapter layers
3. Only train adapters on HF data
```
Adapter: x → x + W_up(ReLU(W_down(x)))
```

### 12. DNGO-MAML (Model-Agnostic Meta-Learning)
Meta-learning approach for quick adaptation:
1. **Meta-training**: Learn initialization θ that adapts quickly to LF sub-tasks
2. **Fine-tuning**: Adapt θ to HF task with few gradient steps

```
Inner loop: θ'_i = θ - α∇L_task_i(f_θ)     # Adapt to each task
Outer loop: θ = θ - β∇Σ_i L(f_θ'_i)        # Update initialization
```

Key idea: MAML finds an initialization that is sensitive to task-specific gradients, enabling fast adaptation with limited HF data.

---

## Data Split Strategy (10-Fold Evaluation)

### Fold Generation
Each fold uses a different random seed to generate train/test splits:

```python
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

for seed in SEEDS:
    rng = np.random.default_rng(seed)

    # Sample 72 compositions for LF training
    lofi_idx = rng.choice(n_total, size=72, replace=False)

    # Sample 9 compositions for HF training
    hifi_idx = rng.choice(n_total, size=9, replace=False)

    # Test set: all compositions except HF training points
    test_idx = [i for i in range(n_total) if i not in hifi_idx]
```

### Important Notes
- LF and HF samples may overlap (3-5 samples typically)
- This mimics real experimental scenarios where initial DFT screens may be refined
- Test set excludes HF training points but includes LF training points

---

## Visualization Strategy

All visualizations are saved in `visualizations/` directory with timestamped folders.

### Standard Visualization Suite
Each experiment generates:

1. **Predictions by Composition** (`predictions_by_composition.png`)
   - X-axis: Composition index (sorted)
   - Shows predicted vs actual values with uncertainty bands
   - Train points marked distinctly

2. **Predictions by Value** (`predictions_by_value.png`)
   - X-axis: Sorted by true value
   - Better shows prediction quality across value range

3. **Parity Plot** (`parity_plot.png`)
   - Predicted vs True scatter plot
   - Ideal: points on diagonal

4. **R² Comparison Bar Chart** (`r2_comparison.png`)
   - Cross-method performance comparison

5. **Uncertainty Distribution** (`uncertainty_dist.png`)
   - Histogram of predicted uncertainties

6. **Summary Table** (`summary.csv`)
   - Mean, Std of R², RMSE across folds

### Multi-Panel Figures
For comprehensive comparisons, multi-panel figures show all models side-by-side:
- 5 or 6 panel layout
- Shared axes for direct comparison
- Both composition-sorted and value-sorted versions

---

## Running Jobs on HPC (SLURM)

### Basic Job Submission Script

```bash
#!/bin/bash
#SBATCH --job-name=model_comparison
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --reservation=rental_9427
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# Activate conda environment
source ~/.bashrc
conda activate pytorch_env

# GPU check
nvidia-smi

# Navigate to directory
cd /path/to/model_comparison

# Run experiment
python -u your_script.py
```

### Submit Job
```bash
sbatch submit_script.sh
```

### Monitor Jobs
```bash
squeue -u $USER          # Check job status
scancel <job_id>         # Cancel job
tail -f slurm_*.out      # Follow output
```

### Typical Resource Requirements
| Experiment Type | GPU | Memory | Time |
|-----------------|-----|--------|------|
| Single model, 10 folds | 1x GPU | 16GB | 30min |
| 6 models comparison | 1x GPU | 16GB | 2hr |
| Full BO optimization | 1x GPU | 32GB | 8hr |

---

## File Structure

```
model_comparison/
├── README.md                          # This file
├── mfgp_model.py                      # Multi-Fidelity GP implementation
├── model_evaluators.py                # Common evaluation utilities
├── dataset_generator.py               # Data loading and fold generation
│
├── # Core comparison scripts
├── run_bnn_vs_dngo.py                # BNN vs DNGO comparison
├── run_4model_comparison.py          # 4-model comparison
├── run_ol_comparison.py              # Online learning comparison
├── run_full_comparison.py            # Comprehensive comparison
├── run_true_intermediate.py          # Final 6-method comparison
├── advanced_transfer_learning.py     # Advanced TL methods (KD, DA, etc.)
│
├── # Visualization scripts
├── visualize_gradient_scaling_vs_mfgp.py
├── visualize_all_6methods.py
├── visualize_test_r2_comparison.py
├── visualize_*.py                    # Various visualization scripts
│
├── # Job submission scripts
├── submit_*.sh                       # SLURM job scripts
│
├── # Results directories
├── visualizations/                   # Timestamped visualization outputs
│   ├── YYYYMMDD_HHMMSS_experiment_name/
│   │   ├── predictions_by_composition.png
│   │   ├── predictions_by_value.png
│   │   ├── parity_plot.png
│   │   ├── summary.csv
│   │   └── ...
│   └── ...
├── results/                          # Detailed result files
├── datasets/                         # Generated dataset files
└── slurm_*.out, slurm_*.err         # Job logs
```

---

## Quick Start

### 1. Run 6-Method Comparison
```bash
# Submit job
sbatch submit_all_6methods.sh

# Or run directly (if on compute node)
python visualize_all_6methods.py
```

### 2. Run Advanced Transfer Learning Methods
```bash
python advanced_transfer_learning.py
```

### 3. Check Results
```bash
ls -la visualizations/  # Find latest timestamped folder
cat visualizations/YYYYMMDD_*/summary.csv
```

---

## References

- **MFGP**: Perdikaris et al. "Multi-fidelity modelling via recursive co-kriging"
- **DNGO**: Snoek et al. "Scalable Bayesian Optimization Using Deep Neural Networks"
- **Knowledge Distillation**: Hinton et al. "Distilling the Knowledge in a Neural Network"
- **Domain Adaptation (MMD)**: Gretton et al. "A Kernel Two-Sample Test"
- **Adapter**: Houlsby et al. "Parameter-Efficient Transfer Learning for NLP"

---

## Author
Generated with assistance from Claude Code
Date: 2025-12-16
