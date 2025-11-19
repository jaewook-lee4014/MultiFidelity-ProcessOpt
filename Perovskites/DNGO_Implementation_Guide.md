# DNGO-Based Transfer Learning Bayesian Optimization Implementation Guide

## 📋 Table of Contents
1. [Overview](#overview)
2. [Mathematical Foundations](#mathematical-foundations)
3. [System Architecture](#system-architecture)
4. [Core Components](#core-components)
5. [Implementation Details](#implementation-details)
6. [Algorithms and Workflows](#algorithms-and-workflows)
7. [Hyperparameter Optimization](#hyperparameter-optimization)
8. [API Design](#api-design)
9. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)

---

## Overview

### What is DNGO?

**DNGO (Deep Networks for Global Optimization)** is a Bayesian Optimization framework that combines:
- **Deep Neural Networks (DNNs)** for feature extraction
- **Bayesian Linear Regression (BLR)** for uncertainty quantification
- **Transfer Learning** for leveraging multi-fidelity data
- **Expected Improvement (EI)** for acquisition function

### Key Design Philosophy

```
Low-Fidelity Data (cheap) → Pretrain DNN → Extract Features
                                           ↓
High-Fidelity Data (expensive) → Fine-tune DNN → Extract Features → BLR → Predictions with Uncertainty
                                                                    ↓
                                                      Bayesian Optimization Loop (EI)
```

### Use Case in This Implementation

Optimizing perovskite solar cell materials by:
- Using GGA (low-fidelity, cost=1) bandgap calculations
- Using HSE06 (high-fidelity, cost=8) bandgap calculations
- Finding optimal composition with minimal computational cost

---

## Mathematical Foundations

### 1. Bayesian Linear Regression (BLR)

**Core Equations:**

Given features φ(x) from DNN, we model:
```
y = w^T φ(x) + ε, where ε ~ N(0, β^(-1))
```

**Prior Distribution:**
```
p(w) = N(0, α^(-1)I)
```

**Posterior Distribution (after observing data):**
```
p(w|X,y) = N(m_N, S_N)

where:
  S_N^(-1) = α·I + β·Φ^T·Φ
  m_N = β·S_N·Φ^T·y

  Φ = [φ(x_1), φ(x_2), ..., φ(x_N)]^T
```

**Predictive Distribution:**
```
p(y*|x*, X, y) = N(μ*, σ*²)

where:
  μ* = m_N^T φ(x*)
  σ*² = 1/β + φ(x*)^T S_N φ(x*)
```

**Key Parameters:**
- α = 1.0 (precision of weight prior)
- β = 25.0 (precision of noise/observation)

### 2. Expected Improvement (EI)

**Acquisition Function:**
```
EI(x) = E[max(f_best - f(x), 0)]
     = (f_best - μ(x) - ξ)·Φ(Z) + σ(x)·φ(Z)

where:
  Z = (f_best - μ(x) - ξ) / σ(x)
  Φ = CDF of standard normal
  φ = PDF of standard normal
  ξ = 0.01 (exploration parameter)
```

**Next Point Selection:**
```
x_next = argmax_x EI(x)
```

### 3. Transfer Learning Strategy

**Two-Stage Learning:**

1. **Pretrain Stage** (Low-fidelity only):
   ```
   DNN: X_low → Features
   Loss: MSE(DNN(X_low), y_low)
   Epochs: 200 (default)
   Learning Rate: 1e-3
   ```

2. **Finetune Stage** (High-fidelity):
   ```
   DNN: X_high → Features (reuse pretrained weights)
   Loss: MSE(DNN(X_high), y_high)
   Epochs: 100 (default)
   Learning Rate: 1e-4 (smaller for fine-tuning)
   ```

**BLR Models:**
- **LOFI Model**: Trained only on low-fidelity features
- **HIFI Model**: Trained on all features (low + high fidelity)

---

## System Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    DNGO Optimization System                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐      ┌──────────────────────────┐    │
│  │  Data Layer      │      │  Model Components        │    │
│  ├──────────────────┤      ├──────────────────────────┤    │
│  │ - Lookup Table   │      │ - TransferLearningDNN    │    │
│  │ - Label Maps     │      │ - BayesianLinearReg      │    │
│  │ - Param Space    │      │ - HyperparameterBO       │    │
│  └──────────────────┘      └──────────────────────────┘    │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │       Optimization Loop (optimization_base.py)        │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │ 1. Train Model (DNN Pretrain + Finetune)             │  │
│  │ 2. Fit BLR (on DNN features)                         │  │
│  │ 3. Recommend Next Point (maximize EI)                │  │
│  │ 4. Measure (query lookup table)                      │  │
│  │ 5. Update Data & Repeat                              │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Utilities & Visualization                     │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │ - Result Saving                                       │  │
│  │ - Progress Tracking                                   │  │
│  │ - Multi-fidelity Visualization                        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### File Structure

```
2.Transfer_learning/Pure_TL_BO/
├── DNGO/
│   ├── models.py                          # DNN and BLR models
│   ├── optimization_base.py               # Main optimization loop
│   ├── hyperparameter_optimization.py     # HP-BO using GP
│   ├── hyperparameter_optimization_optuna.py  # HP-BO using Optuna
│   └── visualization.py                   # Plotting utilities
├── common/
│   ├── config.py                          # Configuration parameters
│   ├── data_utils.py                      # Data loading and processing
│   ├── result_saver.py                    # Save results to disk
│   └── visualization.py                   # General plotting
└── main.py                                # Entry point
```

---

## Core Components

### Component 1: TransferLearningDNN

**Purpose:** Feature extractor using transfer learning

**Architecture:**
```python
class TransferLearningDNN:
    # Network Structure
    feature_net: Sequential[
        Linear(input_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, hidden_dim),
        ReLU(),
    ]
    out_layer: Linear(hidden_dim, 1, bias=False)

    model = Sequential(feature_net, out_layer)
```

**Key Methods:**

1. **`__init__(input_dim, hidden_dim, device, use_hyperparameter_bo)`**
   - Initialize model structure
   - Setup for hyperparameter BO if enabled

2. **`pretrain(X_low, y_low, epochs, lr, bo_trials)`**
   - Train on low-fidelity data
   - Optional: Run HP-BO to find optimal architecture
   - Optimizer: Adam with learning rate `lr`
   - Loss: MSE

3. **`finetune(X_high, y_high, epochs, lr, bo_trials)`**
   - Fine-tune on high-fidelity data
   - Keeps feature_net weights, updates all layers
   - Optional: Run HP-BO for fine-tuning parameters

4. **`predict(X)`**
   - Forward pass through full model
   - Returns: predictions (no uncertainty)

5. **`extract_features(X)`**
   - Forward pass through feature_net only
   - Returns: feature vectors φ(x)

**Dynamic Architecture (with HP-BO):**
```python
def _build_dynamic_model(params):
    layers = []
    layers.append(Linear(input_dim, params['hidden_dim']))
    layers.append(ReLU())

    for _ in range(params['hidden_layers'] - 1):
        layers.append(Linear(params['hidden_dim'], params['hidden_dim']))
        layers.append(ReLU())

    feature_net = Sequential(*layers)
    out_layer = Linear(params['hidden_dim'], 1, bias=False)
```

### Component 2: BayesianLinearRegression

**Purpose:** Uncertainty-aware predictions on DNN features

**Core Attributes:**
```python
class BayesianLinearRegression:
    alpha: float = 1.0      # Weight precision
    beta: float = 25.0      # Noise precision
    mean: ndarray           # Posterior mean m_N
    cov: ndarray            # Posterior covariance S_N
    fitted: bool            # Whether model is trained
```

**Key Methods:**

1. **`fit(X, y)`**
   ```python
   # Add bias term
   X_with_bias = column_stack([ones(N), X])

   # Prior
   S0_inv = alpha * eye(D+1)

   # Posterior
   S_N_inv = S0_inv + beta * X_with_bias.T @ X_with_bias
   S_N = inv(S_N_inv)
   m_N = beta * S_N @ X_with_bias.T @ y

   self.cov = S_N
   self.mean = m_N
   ```

2. **`predict(x)`**
   ```python
   x_with_bias = concatenate([[1], x])

   # Predictive mean
   mu = x_with_bias @ self.mean

   # Predictive variance
   var = (1/beta) + x_with_bias @ self.cov @ x_with_bias

   return mu, var
   ```

3. **`predict_batch(X)`**
   - Vectorized predictions for multiple points
   - Returns: means[], variances[]

4. **`incremental_update(X_new, y_new, weight)`**
   - Sherman-Morrison-Woodbury formula for efficient updates
   - Avoids full retraining when new data arrives
   ```python
   # Update covariance
   Sx = S_old @ x_new
   denominator = 1/beta + x_new.T @ Sx
   S_new = S_old - (Sx @ Sx.T) / denominator

   # Update mean
   prediction_error = y_new - x_new.T @ m_old
   m_new = m_old + beta * S_new @ x_new * prediction_error
   ```

### Component 3: HyperparameterBO

**Purpose:** Find optimal DNN hyperparameters using Bayesian Optimization

**Hyperparameter Search Space:**

```python
class HyperparameterSpace:
    # Depends on data_size: 'small', 'medium', 'large'

    # For 'small' data:
    hidden_layers: [1, 2, 3]
    hidden_dims: [16, 32, 64, 128]
    learning_rates: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    epochs_range: (20, 200)

    # For 'medium' data:
    hidden_layers: [1, 2, 3, 4]
    hidden_dims: [32, 64, 128, 256]
    learning_rates: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    epochs_range: (50, 500)

    # For 'large' data:
    hidden_layers: [2, 3, 4, 5]
    hidden_dims: [64, 128, 256, 512]
    learning_rates: [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    epochs_range: (100, 1000)
```

**GP Surrogate Model:**
```python
kernel = Matern(
    length_scale=0.5,
    length_scale_bounds=(1e-3, 1e3),
    nu=2.5
)

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-4,
    normalize_y=True,
    n_restarts_optimizer=3
)
```

**BO Algorithm:**
1. Initial random sampling (n_initial=5 trials)
2. For each trial:
   - Train DNN with candidate hyperparameters
   - Evaluate on validation set
   - Record validation loss
3. Fit GP on observed (hyperparams, validation_loss)
4. Optimize acquisition function (EI) to suggest next hyperparams
5. Return best hyperparameters

---

## Implementation Details

### 1. Multi-Fidelity Data Structure

**Fidelity Levels:**
- Low-fidelity (s=0.1): GGA bandgap, cost=1 unit
- High-fidelity (s=1.0): HSE06 bandgap, cost=8 units

**Data Separation:**
```python
# Separate datasets for each fidelity
X_low: ndarray   # Low-fidelity inputs
y_low: ndarray   # Low-fidelity outputs
X_high: ndarray  # High-fidelity inputs
y_high: ndarray  # High-fidelity outputs
```

**Cost Tracking:**
```python
total_cost = sum(fidelities)
# where fidelity ∈ {0.1, 1.0}
```

### 2. Fidelity Scheduling Strategy

**8:1 Ratio (default):**
```python
def select_fidelity(iteration):
    if iteration % 8 == 0:
        return 1.0  # High-fidelity
    else:
        return 0.1  # Low-fidelity
```

**Rationale:**
- Balance between exploration (cheap low-fidelity) and exploitation (accurate high-fidelity)
- Maximize information gain per unit cost

### 3. Training Workflow

**Full Training (without incremental learning):**

```python
def train_model(X_low, y_low, X_high, y_high, model_config):
    # 1. Create model
    model = TransferLearningDNN(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        device=model_config['device']
    )

    # 2. Pretrain on low-fidelity
    if len(X_low) > 0:
        model.pretrain(
            X_low, y_low,
            epochs=model_config['pretrain_epochs'],
            lr=model_config['pretrain_lr']
        )

    # 3. Finetune on high-fidelity
    if len(X_high) > 0:
        model.finetune(
            X_high, y_high,
            epochs=model_config['finetune_epochs'],
            lr=model_config['finetune_lr']
        )

    return model
```

**With Hyperparameter BO:**

```python
def train_model_with_bo(X_low, y_low, X_high, y_high,
                        pretrain_bo_trials, finetune_bo_trials):
    model = TransferLearningDNN(
        input_dim=3,
        hidden_dim=64,  # Will be overridden by BO
        use_hyperparameter_bo=True
    )

    # Pretrain with BO
    model.pretrain(
        X_low, y_low,
        bo_trials=pretrain_bo_trials,
        data_size='small'
    )
    # BO will find optimal: hidden_layers, hidden_dim, learning_rate, epochs

    # Finetune with BO
    model.finetune(
        X_high, y_high,
        bo_trials=finetune_bo_trials,
        data_size='small'
    )

    return model
```

### 4. BLR Fitting

**Transfer Learning Structure:**

```python
def fit_blr(model, X_low, X_high, y_low, y_high, alpha=1.0, beta=25.0):
    # LOFI Model (low-fidelity only)
    blr_low = None
    if len(X_low) > 0:
        features_low = model.extract_features(X_low)
        blr_low = BayesianLinearRegression(alpha, beta)
        blr_low.fit(features_low, y_low)

    # HIFI Model (transfer learning: low + high)
    blr_high = None
    if len(X_high) > 0:
        # Combine all data
        X_all = vstack([X_low, X_high]) if len(X_low) > 0 else X_high
        y_all = concatenate([y_low, y_high]) if len(y_low) > 0 else y_high

        features_all = model.extract_features(X_all)
        blr_high = BayesianLinearRegression(alpha, beta)
        blr_high.fit(features_all, y_all)
    else:
        # No high-fidelity data yet, use low-fidelity model
        blr_high = blr_low

    return blr_low, blr_high
```

**Key Insight:** HIFI model leverages transfer learning by training on features from both fidelities.

### 5. Next Point Recommendation

**Algorithm:**

```python
def recommend_next(model, blr_low, blr_high, param_ranges,
                   X_low, X_high, y_low, y_high, fidelity):
    # 1. Generate all candidate points
    all_combinations = product(*param_ranges)
    X_grid = array(all_combinations)

    # 2. Extract features
    features_grid = model.extract_features(X_grid)

    # 3. Select appropriate BLR based on fidelity
    if fidelity == 1.0:  # High-fidelity
        blr = blr_high
        y_best = min(y_high) if len(y_high) > 0 else inf
    else:  # Low-fidelity
        blr = blr_low if blr_low is not None else blr_high
        y_best = min(y_low) if len(y_low) > 0 else inf

    # 4. Predict for all candidates
    y_pred, y_std = [], []
    for phi in features_grid:
        mu, var = blr.predict(phi)
        y_pred.append(mu)
        y_std.append(sqrt(var))

    y_pred = array(y_pred)
    y_std = array(y_std)

    # 5. Compute Expected Improvement
    ei = expected_improvement(y_pred, y_std, y_best, xi=0.01)

    # 6. Filter out already measured points (at current fidelity)
    measured_points_current_fidelity = set([
        tuple(x) for x in (X_high if fidelity == 1.0 else X_low)
    ])

    valid_indices = [
        i for i, x in enumerate(X_grid)
        if tuple(x) not in measured_points_current_fidelity
    ]

    # 7. Select point with maximum EI
    if valid_indices:
        valid_ei = ei[valid_indices]
        best_valid_idx = argmax(valid_ei)
        best_idx = valid_indices[best_valid_idx]
    else:
        best_idx = argmax(ei)

    next_x = X_grid[best_idx]

    return next_x, predictions, best_idx, X_grid
```

**Separate Predictions for Visualization:**
```python
# Also compute predictions from both models for plotting
y_pred_low, ei_low = predict_with_blr(blr_low, features_grid, y_best_low)
y_pred_high, ei_high = predict_with_blr(blr_high, features_grid, y_best_high)

predictions = {
    'y_pred': y_pred,  # Current fidelity
    'y_std': y_std,
    'ei': ei,
    'y_pred_low': y_pred_low,
    'ei_low': ei_low,
    'y_pred_high': y_pred_high,
    'ei_high': ei_high
}
```

---

## Algorithms and Workflows

### Main Optimization Loop

```python
def single_optimization_run(param_space, label_maps, lookup,
                            cost_budget, num_init_design,
                            high_fidelity_ratio, min_target):
    # 1. Initialize
    init_samples = sample_param_space(param_space, num_init_design)
    init_fidelities = assign_fidelities(num_init_design, high_fidelity_ratio)
    X_low, y_low, X_high, y_high = prepare_initial_data(init_samples, init_fidelities)

    total_cost = sum(init_fidelities)
    best_so_far = min(y_high) if len(y_high) > 0 else inf
    iteration = 0

    # 2. Main loop
    while total_cost < cost_budget:
        iteration += 1

        # 2.1. Determine fidelity for this iteration
        fidelity = 1.0 if (iteration % 8 == 0) else 0.1

        # 2.2. Train model
        model = train_model(X_low, y_low, X_high, y_high, model_config)

        # 2.3. Fit BLR
        blr_low, blr_high = fit_blr(model, X_low, X_high, y_low, y_high)

        # 2.4. Recommend next point
        next_x, predictions, best_idx, X_grid = recommend_next(
            model, blr_low, blr_high, param_ranges,
            X_low, X_high, y_low, y_high, fidelity
        )

        # 2.5. Measure (query lookup table)
        measurement = measure_from_label(next_x, fidelity, label_maps, lookup)

        # 2.6. Update data
        if fidelity == 1.0:
            X_high = vstack([X_high, next_x])
            y_high = append(y_high, measurement)
            best_so_far = min(best_so_far, measurement)
        else:
            X_low = vstack([X_low, next_x])
            y_low = append(y_low, measurement)

        # 2.7. Update cost
        total_cost += fidelity

        # 2.8. Save visualization data
        save_viz_data(iteration, model, blr_low, blr_high,
                     X_grid, predictions, next_x, best_idx)

        # 2.9. Early termination
        if fidelity == 1.0 and measurement <= min_target:
            print("Target achieved!")
            break

    return {
        'total_cost': total_cost,
        'best_so_far': best_so_far,
        'iterations': iteration,
        'final_data': (X_low, y_low, X_high, y_high),
        'visualization_data': viz_data_list
    }
```

### Incremental Learning (Optional)

**Purpose:** Avoid full retraining at each iteration

**Strategy:**

1. **DNN Incremental Update:**
   ```python
   def incremental_update(model, X_new, y_new, fidelity, incremental_params):
       mode = incremental_params['mode']  # 'full', 'incremental', 'hybrid'

       if mode == 'full':
           # Always retrain from scratch
           retrain_full(model, X_all, y_all, fidelity)

       elif mode == 'incremental':
           # Update with new data only
           lr_boost = incremental_params['lr_boost_factor']
           inc_epochs = incremental_params['incremental_epochs']

           # Higher learning rate for quick adaptation
           boosted_lr = base_lr * lr_boost

           # Experience replay to avoid catastrophic forgetting
           X_combined, y_combined = prepare_replay_data(
               X_new, y_new, fidelity,
               replay_ratio=incremental_params['replay_ratio']
           )

           # Quick update
           train(model, X_combined, y_combined,
                 epochs=inc_epochs, lr=boosted_lr)

       elif mode == 'hybrid':
           # Periodic full retraining
           if iteration % incremental_params['full_retrain_interval'] == 0:
               retrain_full(model, X_all, y_all, fidelity)
           else:
               incremental_train(model, X_new, y_new, fidelity)
   ```

2. **BLR Incremental Update:**
   ```python
   def incremental_update_blr(blr, X_new, y_new):
       # Sherman-Morrison-Woodbury formula
       features_new = model.extract_features(X_new)

       for x_new, y_new in zip(features_new, y_new):
           # Update covariance
           Sx = blr.cov @ x_new
           denominator = 1/blr.beta + x_new.T @ Sx
           blr.cov = blr.cov - (Sx @ Sx.T) / denominator

           # Update mean
           error = y_new - x_new.T @ blr.mean
           blr.mean = blr.mean + blr.beta * blr.cov @ x_new * error
   ```

**Parameters:**
```python
incremental_params = {
    'mode': 'incremental',  # or 'full', 'hybrid'
    'lr_boost_factor': 2.0,
    'incremental_epochs': 10,
    'replay_ratio': 0.2,  # 20% of old data
    'weight_decay_factor': 0.9,  # Reduce importance of old data
    'full_retrain_interval': 5  # For hybrid mode
}
```

---

## Hyperparameter Optimization

### Overview

The system supports automatic hyperparameter tuning using Bayesian Optimization.

**What gets optimized:**
1. DNN architecture (hidden layers, hidden dimensions)
2. Learning rates
3. Training epochs

**When it runs:**
- At the beginning (iteration 1)
- Every N new data points (default: every 10 points)

### Implementation with Gaussian Processes

**Step 1: Hyperparameter Space Normalization**

```python
def normalize_params(params):
    # Map all hyperparameters to [0, 1]
    normalized = zeros(4)

    # Hidden layers (discrete)
    normalized[0] = (params['hidden_layers'] - 1) / (5 - 1)

    # Hidden dim (log scale)
    log_dim = log2(params['hidden_dim'])
    normalized[1] = (log_dim - log2(16)) / (log2(512) - log2(16))

    # Learning rate (log scale)
    log_lr = log10(params['learning_rate'])
    normalized[2] = (log_lr - log10(1e-5)) / (log10(1e-2) - log10(1e-5))

    # Epochs
    normalized[3] = (params['epochs'] - 20) / (500 - 20)

    return normalized
```

**Step 2: GP Surrogate**

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

kernel = Matern(length_scale=0.5, nu=2.5)
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-4,
    normalize_y=True,
    n_restarts_optimizer=3
)
```

**Step 3: BO Loop**

```python
def optimize_hyperparameters(X_train, y_train, X_val, y_val,
                             n_trials, data_size):
    param_space = HyperparameterSpace(data_size)
    X_observed = []  # Normalized hyperparams
    y_observed = []  # Validation losses

    for trial in range(n_trials):
        # Initial random sampling
        if trial < 3:
            params = param_space.sample_random()
        else:
            # Fit GP
            gp.fit(array(X_observed), array(y_observed))

            # Optimize acquisition function
            params_normalized = optimize_ei(gp, param_space)
            params = param_space.denormalize(params_normalized)

        # Evaluate
        val_loss = train_and_evaluate(params, X_train, y_train, X_val, y_val)

        # Record
        X_observed.append(param_space.normalize(params))
        y_observed.append(val_loss)

    # Return best
    best_idx = argmin(y_observed)
    best_params = denormalize(X_observed[best_idx])

    return best_params, y_observed[best_idx]
```

**Step 4: Integration with Training**

```python
if use_hyperparameter_bo and should_optimize:
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Run BO
    best_params, best_loss = optimize_hyperparameters(
        X_train, y_train, X_val, y_val,
        n_trials=5,
        data_size='small'
    )

    # Use best hyperparameters
    model._build_dynamic_model(best_params)
    model.pretrain(X, y,
                  epochs=best_params['epochs'],
                  lr=best_params['learning_rate'])
```

---

## API Design

### Core API Structure

```python
# 1. Create and train model
from DNGO.models import TransferLearningDNN, BayesianLinearRegression
from DNGO.optimization_base import train_model, fit_blr, recommend_next

# Initialize
model = TransferLearningDNN(input_dim=3, hidden_dim=64, device='cpu')

# Train
model.pretrain(X_low, y_low, epochs=200, lr=1e-3)
model.finetune(X_high, y_high, epochs=100, lr=1e-4)

# Extract features
features = model.extract_features(X_candidates)

# Fit BLR
blr_low, blr_high = fit_blr(model, X_low, X_high, y_low, y_high)

# Predict with uncertainty
mu, var = blr_high.predict(feature_vector)

# 2. Run optimization
from DNGO.optimization_base import single_optimization_run

result = single_optimization_run(
    param_space=param_space,
    label_maps=label_maps,
    lookup=lookup_table,
    cost_budget=50.0,
    num_init_design=10,
    high_fidelity_ratio=0.2,
    min_target=1.5249,
    random_state=42,
    verbose=True,
    model_config={
        'input_dim': 3,
        'hidden_dim': 64,
        'pretrain_epochs': 200,
        'finetune_epochs': 100,
        'device': 'cpu'
    }
)

# 3. Access results
print(f"Best value: {result['best_so_far']}")
print(f"Total cost: {result['total_cost']}")
print(f"Iterations: {result['iterations']}")

# Visualization data
for viz_data in result['visualization_data']:
    plot_iteration(viz_data)

# 4. Run with hyperparameter BO
result = single_optimization_run(
    param_space=param_space,
    label_maps=label_maps,
    lookup=lookup_table,
    cost_budget=50.0,
    use_hyperparameter_bo=True,
    pretrain_bo_trials=5,
    finetune_bo_trials=5,
    data_size='small',
    model_config=model_config
)

# 5. Multiple runs for statistical analysis
from DNGO.optimization_base import multiple_optimization_runs

results = multiple_optimization_runs(
    param_space=param_space,
    label_maps=label_maps,
    lookup=lookup_table,
    num_runs=100,
    cost_budget=50.0,
    save_results=True,
    results_filename='results.csv'
)

# 6. Incremental learning
result = single_optimization_run(
    param_space=param_space,
    label_maps=label_maps,
    lookup=lookup_table,
    cost_budget=50.0,
    use_incremental_learning=True,
    incremental_params={
        'mode': 'incremental',
        'lr_boost_factor': 2.0,
        'incremental_epochs': 10,
        'replay_ratio': 0.2,
        'weight_decay_factor': 0.9
    },
    model_config=model_config
)
```

---

## Step-by-Step Implementation Guide

### Phase 1: Core Components (Foundation)

#### Step 1.1: Implement BayesianLinearRegression

```python
import numpy as np

class BayesianLinearRegression:
    def __init__(self, alpha=1.0, beta=25.0):
        self.alpha = alpha  # Weight precision
        self.beta = beta    # Noise precision
        self.mean = None
        self.cov = None
        self.fitted = False

    def fit(self, X, y):
        """
        Fit Bayesian Linear Regression

        Math:
            Prior: p(w) = N(0, α^(-1)I)
            Likelihood: p(y|X,w) = N(Xw, β^(-1)I)
            Posterior: p(w|X,y) = N(m_N, S_N)
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).flatten()

        # Add bias term
        N = len(X)
        X_with_bias = np.column_stack([np.ones(N), X])

        # Prior covariance inverse
        D = X_with_bias.shape[1]
        S0_inv = self.alpha * np.eye(D)

        # Posterior covariance
        S_N_inv = S0_inv + self.beta * (X_with_bias.T @ X_with_bias)
        self.cov = np.linalg.inv(S_N_inv)

        # Posterior mean
        self.mean = self.beta * (self.cov @ X_with_bias.T @ y)

        self.fitted = True

    def predict(self, x):
        """
        Predict with uncertainty

        Returns:
            mu: Predictive mean
            var: Predictive variance
        """
        if not self.fitted:
            raise ValueError("Model not fitted")

        x = np.asarray(x, dtype=np.float32).flatten()
        x_with_bias = np.concatenate([[1.0], x])

        # Predictive distribution
        mu = x_with_bias @ self.mean
        var = (1.0 / self.beta) + (x_with_bias @ self.cov @ x_with_bias)

        return mu, var
```

**Testing BLR:**

```python
# Test
X_train = np.random.randn(50, 10)
y_train = X_train.sum(axis=1) + np.random.randn(50) * 0.1

blr = BayesianLinearRegression(alpha=1.0, beta=25.0)
blr.fit(X_train, y_train)

# Predict
x_test = np.random.randn(10)
mu, var = blr.predict(x_test)
print(f"Prediction: {mu:.4f} ± {np.sqrt(var):.4f}")
```

#### Step 1.2: Implement TransferLearningDNN

```python
import torch
import torch.nn as nn
import torch.optim as optim

class TransferLearningDNN:
    def __init__(self, input_dim, hidden_dim=64, device='cpu'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # Build model
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ).to(device).float()

        self.out_layer = nn.Linear(hidden_dim, 1, bias=False).to(device).float()
        self.model = nn.Sequential(self.feature_net, self.out_layer)

    def pretrain(self, X_low, y_low, epochs=200, lr=1e-3):
        """Pretrain on low-fidelity data"""
        X_tensor = torch.tensor(X_low, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_low, dtype=torch.float32).view(-1, 1).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self.model(X_tensor)
            loss = loss_fn(pred, y_tensor)
            loss.backward()
            optimizer.step()

    def finetune(self, X_high, y_high, epochs=100, lr=1e-4):
        """Finetune on high-fidelity data"""
        X_tensor = torch.tensor(X_high, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_high, dtype=torch.float32).view(-1, 1).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self.model(X_tensor)
            loss = loss_fn(pred, y_tensor)
            loss.backward()
            optimizer.step()

    def extract_features(self, X):
        """Extract features using feature_net"""
        self.feature_net.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            features = self.feature_net(X_tensor)
            return features.cpu().numpy()

    def predict(self, X):
        """Full model prediction"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            pred = self.model(X_tensor)
            return pred.cpu().numpy().flatten()
```

**Testing DNN:**

```python
# Test
X_low = np.random.randn(100, 3)
y_low = (X_low ** 2).sum(axis=1) + np.random.randn(100) * 0.1

X_high = np.random.randn(20, 3)
y_high = (X_high ** 2).sum(axis=1) + np.random.randn(20) * 0.05

model = TransferLearningDNN(input_dim=3, hidden_dim=64, device='cpu')
model.pretrain(X_low, y_low, epochs=100)
model.finetune(X_high, y_high, epochs=50)

features = model.extract_features(X_high)
print(f"Features shape: {features.shape}")  # (20, 64)
```

### Phase 2: Optimization Loop

#### Step 2.1: Implement Expected Improvement

```python
from scipy.stats import norm

def expected_improvement(mu, sigma, y_best, xi=0.01):
    """
    Expected Improvement acquisition function

    Args:
        mu: Predicted means
        sigma: Predicted std deviations
        y_best: Best observed value so far
        xi: Exploration parameter

    Returns:
        ei: Expected improvement values
    """
    sigma = np.maximum(sigma, 1e-8)  # Numerical stability

    z = (y_best - mu - xi) / sigma
    ei = (y_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)

    return ei
```

#### Step 2.2: Implement fit_blr

```python
def fit_blr(model, X_low, X_high, y_low, y_high, alpha=1.0, beta=25.0):
    """
    Fit BLR models for both fidelities

    Returns:
        blr_low: BLR trained on low-fidelity features only
        blr_high: BLR trained on all features (transfer learning)
    """
    # LOFI model
    blr_low = None
    if len(X_low) > 0:
        features_low = model.extract_features(X_low)
        blr_low = BayesianLinearRegression(alpha, beta)
        blr_low.fit(features_low, y_low)

    # HIFI model (with transfer learning)
    blr_high = None
    if len(X_high) > 0:
        # Combine data from both fidelities
        X_all = np.vstack([X_low, X_high]) if len(X_low) > 0 else X_high
        y_all = np.concatenate([y_low, y_high]) if len(y_low) > 0 else y_high

        features_all = model.extract_features(X_all)
        blr_high = BayesianLinearRegression(alpha, beta)
        blr_high.fit(features_all, y_all)
    else:
        blr_high = blr_low

    return blr_low, blr_high
```

#### Step 2.3: Implement recommend_next

```python
def recommend_next(model, blr_low, blr_high, X_grid,
                   X_low, X_high, y_low, y_high, fidelity):
    """
    Recommend next point to evaluate

    Args:
        model: Trained DNN
        blr_low, blr_high: Fitted BLR models
        X_grid: All candidate points
        X_low, X_high: Observed points at each fidelity
        y_low, y_high: Observed values at each fidelity
        fidelity: Current fidelity (0.1 or 1.0)

    Returns:
        next_x: Recommended point
        predictions: Dict with predictions for visualization
        best_idx: Index in X_grid
    """
    # Extract features
    features_grid = model.extract_features(X_grid)

    # Select appropriate BLR and best value
    if fidelity == 1.0:
        blr = blr_high
        y_best = np.min(y_high) if len(y_high) > 0 else np.inf
    else:
        blr = blr_low if blr_low is not None else blr_high
        y_best = np.min(y_low) if len(y_low) > 0 else np.inf

    # Predict for all candidates
    y_pred, y_std = [], []
    for phi in features_grid:
        mu, var = blr.predict(phi)
        y_pred.append(mu)
        y_std.append(np.sqrt(var))

    y_pred = np.array(y_pred)
    y_std = np.array(y_std)

    # Compute EI
    ei = expected_improvement(y_pred, y_std, y_best)

    # Filter already measured points at current fidelity
    measured = set([tuple(x) for x in (X_high if fidelity == 1.0 else X_low)])
    valid_indices = [i for i, x in enumerate(X_grid) if tuple(x) not in measured]

    # Select best
    if valid_indices:
        valid_ei = ei[valid_indices]
        best_valid_idx = np.argmax(valid_ei)
        best_idx = valid_indices[best_valid_idx]
    else:
        best_idx = np.argmax(ei)

    next_x = X_grid[best_idx]

    predictions = {
        'y_pred': y_pred,
        'y_std': y_std,
        'ei': ei
    }

    return next_x, predictions, best_idx
```

#### Step 2.4: Implement main optimization loop

```python
def single_optimization_run(X_grid, y_true_function,
                            cost_budget=50.0, num_init=10):
    """
    Main optimization loop

    Args:
        X_grid: All candidate points (N x D)
        y_true_function: Function to evaluate points
        cost_budget: Total cost budget
        num_init: Number of initial random samples

    Returns:
        results: Dict with optimization results
    """
    # Initialize
    np.random.seed(42)
    init_indices = np.random.choice(len(X_grid), num_init, replace=False)

    X_low, y_low = [], []
    X_high, y_high = [], []

    # Initial sampling (20% high-fidelity)
    for i in init_indices:
        x = X_grid[i]
        fidelity = 1.0 if np.random.rand() < 0.2 else 0.1
        y = y_true_function(x, fidelity)

        if fidelity == 1.0:
            X_high.append(x)
            y_high.append(y)
        else:
            X_low.append(x)
            y_low.append(y)

    X_low = np.array(X_low)
    y_low = np.array(y_low)
    X_high = np.array(X_high)
    y_high = np.array(y_high)

    total_cost = len(X_low) * 0.1 + len(X_high) * 1.0
    best_so_far = np.min(y_high) if len(y_high) > 0 else np.inf

    iteration = 0
    results_history = []

    # Main loop
    while total_cost < cost_budget:
        iteration += 1

        # Fidelity scheduling (8:1 ratio)
        fidelity = 1.0 if (iteration % 8 == 0) else 0.1

        # Train model
        model = TransferLearningDNN(input_dim=X_grid.shape[1], hidden_dim=64)
        if len(X_low) > 0:
            model.pretrain(X_low, y_low, epochs=200)
        if len(X_high) > 0:
            model.finetune(X_high, y_high, epochs=100)

        # Fit BLR
        blr_low, blr_high = fit_blr(model, X_low, X_high, y_low, y_high)

        # Recommend next point
        next_x, predictions, best_idx = recommend_next(
            model, blr_low, blr_high, X_grid,
            X_low, X_high, y_low, y_high, fidelity
        )

        # Evaluate
        y_new = y_true_function(next_x, fidelity)

        # Update data
        if fidelity == 1.0:
            X_high = np.vstack([X_high, next_x])
            y_high = np.append(y_high, y_new)
            best_so_far = min(best_so_far, y_new)
        else:
            X_low = np.vstack([X_low, next_x])
            y_low = np.append(y_low, y_new)

        total_cost += fidelity

        # Record
        results_history.append({
            'iteration': iteration,
            'total_cost': total_cost,
            'best_so_far': best_so_far,
            'fidelity': fidelity,
            'measurement': y_new
        })

        print(f"Iter {iteration}: Cost={total_cost:.1f}, Best={best_so_far:.4f}, EI={predictions['ei'][best_idx]:.4f}")

    return {
        'total_cost': total_cost,
        'best_so_far': best_so_far,
        'iterations': iteration,
        'history': results_history,
        'final_data': (X_low, y_low, X_high, y_high)
    }
```

### Phase 3: Advanced Features

#### Step 3.1: Add Hyperparameter BO (optional)

See the "Hyperparameter Optimization" section for implementation details.

Key steps:
1. Create `HyperparameterSpace` class for search space definition
2. Implement GP-based BO using `sklearn.gaussian_process`
3. Integrate with `TransferLearningDNN.pretrain()` and `.finetune()`

#### Step 3.2: Add Incremental Learning (optional)

Key modifications:
1. Store previous data in buffers
2. Implement experience replay mechanism
3. Use Sherman-Morrison formula for BLR updates
4. Add learning rate scheduling

#### Step 3.3: Add Visualization

```python
import matplotlib.pyplot as plt

def plot_optimization_progress(results):
    """Plot optimization results"""
    history = results['history']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Best so far
    ax = axes[0, 0]
    costs = [h['total_cost'] for h in history]
    bests = [h['best_so_far'] for h in history]
    ax.plot(costs, bests, 'b-', linewidth=2)
    ax.set_xlabel('Cumulative Cost')
    ax.set_ylabel('Best Value Found')
    ax.set_title('Optimization Progress')
    ax.grid(True, alpha=0.3)

    # Fidelity schedule
    ax = axes[0, 1]
    iterations = [h['iteration'] for h in history]
    fidelities = [h['fidelity'] for h in history]
    colors = ['red' if f == 1.0 else 'blue' for f in fidelities]
    ax.scatter(iterations, fidelities, c=colors, alpha=0.6)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fidelity')
    ax.set_title('Fidelity Scheduling')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
```

---

## Summary Checklist

### Minimum Implementation Checklist

- [ ] **BayesianLinearRegression class**
  - [ ] `fit(X, y)` with proper prior and posterior calculations
  - [ ] `predict(x)` returning (mean, variance)
  - [ ] `predict_batch(X)` for vectorized predictions

- [ ] **TransferLearningDNN class**
  - [ ] Basic architecture (2-layer feature network + output layer)
  - [ ] `pretrain(X_low, y_low)` method
  - [ ] `finetune(X_high, y_high)` method
  - [ ] `extract_features(X)` method

- [ ] **Optimization functions**
  - [ ] `expected_improvement(mu, sigma, y_best)`
  - [ ] `fit_blr(model, X_low, X_high, y_low, y_high)`
  - [ ] `recommend_next(model, blr_low, blr_high, X_grid, ...)`
  - [ ] `single_optimization_run(...)` main loop

- [ ] **Data handling**
  - [ ] Separate low/high fidelity data structures
  - [ ] Cost tracking mechanism
  - [ ] Fidelity scheduling (8:1 ratio)

- [ ] **Visualization (optional but recommended)**
  - [ ] Plot best-so-far curve
  - [ ] Plot fidelity scheduling
  - [ ] Plot predictions vs actual

### Optional Advanced Features

- [ ] **Hyperparameter Bayesian Optimization**
  - [ ] `HyperparameterSpace` class
  - [ ] GP-based hyperparameter BO
  - [ ] Integration with training loop

- [ ] **Incremental Learning**
  - [ ] DNN incremental update with experience replay
  - [ ] BLR Sherman-Morrison updates
  - [ ] Hybrid training modes

- [ ] **Multi-run Analysis**
  - [ ] `multiple_optimization_runs()` function
  - [ ] Statistical aggregation
  - [ ] Boxplot visualization

---

## References and Key Papers

1. **DNGO (Deep Networks for Global Optimization)**
   - Snoek et al., "Scalable Bayesian Optimization Using Deep Neural Networks" (2015)

2. **Bayesian Optimization**
   - Brochu et al., "A Tutorial on Bayesian Optimization" (2010)
   - Jones et al., "Efficient Global Optimization of Expensive Black-Box Functions" (1998)

3. **Transfer Learning**
   - Pan and Yang, "A Survey on Transfer Learning" (2010)

4. **Bayesian Linear Regression**
   - Bishop, "Pattern Recognition and Machine Learning" (2006), Chapter 3

---

## Contact and Support

For questions about this implementation:
- Review the source code in `2.Transfer_learning/Pure_TL_BO/`
- Check the Jupyter notebook `test_tl_bo.ipynb` for examples
- Run `python main.py --help` for command-line options

## Configuration Parameters Reference

### Default Values

```python
# Optimization
COST_BUDGET = 50.0
NUM_INIT_DESIGN = 10
HIGH_FIDELITY_RATIO = 0.2
MIN_TARGET = 1.5249  # Target bandgap

# Model
INPUT_DIM = 3
HIDDEN_DIM = 64
PRETRAIN_EPOCHS = 200
FINETUNE_EPOCHS = 100
PRETRAIN_LR = 1e-3
FINETUNE_LR = 1e-4

# BLR
ALPHA = 1.0
BETA = 25.0

# EI
XI = 0.01
```

---

**END OF GUIDE**

This guide provides a complete blueprint for reimplementing the DNGO-based optimization system in any framework or programming language. Focus on implementing the core components first (Phase 1-2), then add advanced features as needed (Phase 3).
