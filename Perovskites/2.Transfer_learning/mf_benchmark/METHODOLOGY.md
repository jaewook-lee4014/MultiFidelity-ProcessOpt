# Methodology: Transfer Learning Architectures for Scalable Multi-Fidelity Bayesian Optimization

---

## 1. Problem Setup: Multi-Fidelity Bayesian Optimization

### 1.1 Formal Definition

We consider the global optimization of an expensive black-box function:

$$\mathbf{x}^* = \arg\min_{\mathbf{x} \in \mathcal{X}} f^{(H)}(\mathbf{x})$$

where $f^{(H)}: \mathcal{X} \rightarrow \mathbb{R}$ denotes the high-fidelity (HF) objective defined over a compact search domain $\mathcal{X} \subseteq \mathbb{R}^d$.

We additionally have access to a low-fidelity (LF) approximation $f^{(L)}: \mathcal{X} \rightarrow \mathbb{R}$ satisfying:

- **Correlation**: $f^{(L)}$ is positively correlated with $f^{(H)}$, quantified via $R^2 = \text{Corr}(f^{(L)}(\mathbf{x}), f^{(H)}(\mathbf{x}))^2$
- **Cost asymmetry**: Evaluating $f^{(L)}$ costs $\rho \ll 1$ units, while $f^{(H)}$ costs 1 unit

### 1.2 Cost Model

| Fidelity | Cost | Notation |
|----------|------|----------|
| High-fidelity (HF) | $c_H = 1.0$ | e.g., HSE06 DFT |
| Low-fidelity (LF) | $c_L = \rho$ | e.g., GGA DFT |

- **Favorable regime**: $\rho = 0.1$, $R^2 > 0.9$ (cheap and informative)
- **Unfavorable regime**: $\rho = 0.5$, $R^2 < 0.75$ (expensive and uninformative)

### 1.3 Budget Constraint

Given a total computational budget $B$, the optimization is constrained by:

$$\sum_{t=1}^{T} c_{s_t} \leq B$$

where $s_t \in \{L, H\}$ is the fidelity selected at iteration $t$.

### 1.4 Performance Metric

We report **simple regret** at termination:

$$r_T = \min_{t: s_t = H} f^{(H)}(\mathbf{x}_t) - f^{(H)}(\mathbf{x}^*)$$

---

## 2. MFBO Loop and Acquisition Strategy

### 2.1 Overview

```
Input: Budget B, cost ratio rho, search space X
Output: Best found x*

1. INITIALIZE
   - Allocate 10% of B for initial sampling
   - Sample n_init_hf HF points and n_init_lf LF points
   - D_H = {(x_i, y_i^H)}, D_L = {(x_j, y_j^L)}

2. REPEAT until budget exhausted:
   a. Train surrogate: M.fit(D_L, D_H)
   b. Predict on candidates: mu, sigma = M.predict(X_cand)
   c. Compute acquisition: alpha(x) = A(mu, sigma; D_H)
   d. Select next point: x_{t+1} = argmax alpha(x)
   e. Select fidelity: s_{t+1} via round-robin schedule
   f. Evaluate: y_{t+1} = f^{(s_{t+1})}(x_{t+1})
   g. Update: D_{s_{t+1}} = D_{s_{t+1}} ∪ {(x_{t+1}, y_{t+1})}

3. RETURN argmin_{D_H} y^H
```

### 2.2 Initial Sampling

Budget allocation for initialization (10% of total budget $B$):

- $n_{\text{init}}^{H} = \max\left(2,\ \lfloor 0.1 \cdot B \cdot 0.5 / c_H \rfloor\right)$
- $n_{\text{init}}^{L} = \max\left(2,\ \lfloor 0.1 \cdot B \cdot 0.5 / c_L \rfloor\right)$

Sampling methods:
- **Continuous domains** (Branin, Park): Latin Hypercube Sampling (LHS)
- **Discrete domains** (COFs, FreeSolv, Polarizability): Furthest Point Sampling (FPS)

### 2.3 Fidelity Selection: Deterministic Round-Robin

Rather than cost-scaled acquisition, we adopt a **fixed round-robin schedule** for fair model comparison:

$$s_t = \begin{cases} L & \text{if } \text{lf\_counter} < \lfloor 1 / \rho \rfloor \\ H & \text{otherwise (reset counter)} \end{cases}$$

This yields approximately $\lfloor 1/\rho \rfloor$ LF evaluations per HF evaluation (e.g., 10:1 for $\rho = 0.1$).

### 2.4 Acquisition Function: Expected Improvement

For candidate point $\mathbf{x}$ with predictive mean $\mu(\mathbf{x})$ and standard deviation $\sigma(\mathbf{x})$:

$$\alpha_{\text{EI}}(\mathbf{x}) = (\hat{y} - \mu(\mathbf{x}) - \xi)\,\Phi(Z) + \sigma(\mathbf{x})\,\phi(Z)$$

where $Z = \frac{\hat{y} - \mu(\mathbf{x}) - \xi}{\sigma(\mathbf{x})}$, $\hat{y} = \min_{D_H} y^H$, and $\xi = 0.01$.

### 2.5 Dual Acquisition Strategy (with LF-BLR)

When equipped with LF uncertainty quantification:

| Fidelity | Acquisition | Rationale |
|----------|------------|-----------|
| LF (cheap) | $\mathbf{x}_{t+1} = \arg\max\ \alpha_{\text{EI}}^{(L)}(\mathbf{x})$ | Exploration via uncertainty |
| HF (expensive) | $\mathbf{x}_{t+1} = \arg\min\ \mu^{(H)}(\mathbf{x})$ | Pure exploitation |

---

## 3. Deep Probabilistic Surrogate for Multi-Fidelity Modeling

### 3.1 Two-Network Architecture

All DNN-based surrogates share a common backbone:

**LF Network** $g_\theta^{(L)}: \mathbb{R}^d \rightarrow \mathbb{R}$
```
x ∈ R^d → Linear(d, 64) → Tanh → Linear(64, 64) → Tanh → Linear(64, 1)
```

**HF Network** $g_\phi^{(H)}: \mathbb{R}^{d+1} \rightarrow \mathbb{R}$ (residual learner)
```
[x, y_lf] ∈ R^{d+1} → Linear(d+1, 64) → Tanh → Linear(64, 64) → Tanh → Linear(64, 1)
```

**Residual connection**: The HF prediction is formulated as
$$\hat{y}^{(H)}(\mathbf{x}) = g_\theta^{(L)}(\mathbf{x}) + g_\phi^{(H)}\left([\mathbf{x},\, g_\theta^{(L)}(\mathbf{x})]\right)$$

where the HF network learns the **fidelity residual** $\delta(\mathbf{x}) = f^{(H)}(\mathbf{x}) - f^{(L)}(\mathbf{x})$.

### 3.2 Uncertainty Quantification via Bayesian Linear Regression (BLR)

To obtain calibrated uncertainty from the DNN backbone, we apply **Bayesian Linear Regression** on the last hidden layer features of the LF network.

Let $\boldsymbol{\phi}(\mathbf{x}) \in \mathbb{R}^{D}$ denote the last-layer feature representation (before the output head) of the LF network, augmented with a bias term $\tilde{\boldsymbol{\phi}}(\mathbf{x}) = [\boldsymbol{\phi}(\mathbf{x}),\, 1] \in \mathbb{R}^{D+1}$.

**Prior**: $p(\mathbf{w}) = \mathcal{N}(\mathbf{0},\, \alpha^{-1}\mathbf{I})$

**Likelihood**: $p(y \mid \mathbf{x}, \mathbf{w}) = \mathcal{N}(y \mid \tilde{\boldsymbol{\phi}}(\mathbf{x})^\top \mathbf{w},\, \beta^{-1})$

**Posterior**: $p(\mathbf{w} \mid \mathcal{D}_L) = \mathcal{N}(\mathbf{w} \mid \mathbf{m}_N,\, \mathbf{S}_N)$

$$\mathbf{S}_N = \left(\alpha \mathbf{I} + \beta \tilde{\boldsymbol{\Phi}}^\top \tilde{\boldsymbol{\Phi}}\right)^{-1}, \qquad \mathbf{m}_N = \beta \mathbf{S}_N \tilde{\boldsymbol{\Phi}}^\top \mathbf{y}$$

**Predictive distribution** for a new input $\mathbf{x}_*$:

$$\mu_* = \tilde{\boldsymbol{\phi}}(\mathbf{x}_*)^\top \mathbf{m}_N, \qquad \sigma_*^2 = \beta^{-1} + \tilde{\boldsymbol{\phi}}(\mathbf{x}_*)^\top \mathbf{S}_N\, \tilde{\boldsymbol{\phi}}(\mathbf{x}_*)$$

**Hyperparameters**: $\alpha = 1.0$ (prior precision), $\beta = 25.0$ (noise precision).

### 3.3 Baseline: Multi-Fidelity Gaussian Process (MFGP)

As a non-DNN baseline, we employ `SingleTaskMultiFidelityGP` (BoTorch) with:
- Fidelity indicator appended as input: $\tilde{\mathbf{x}} = [\mathbf{x},\, s]$ where $s \in \{0, 1\}$
- Automatic kernel hyperparameter learning via marginal likelihood maximization
- Full GP posterior provides calibrated $\mu(\mathbf{x})$ and $\sigma(\mathbf{x})$

---

## 4. Transfer Learning for Multi-Fidelity Surrogates

### 4.1 Core Paradigm: LF Pretraining + HF Adaptation

The fundamental insight is that abundant LF data can provide useful **feature representations** for the scarce HF task:

$$\underbrace{g_\theta^{(L)} \xrightarrow{\text{train on } \mathcal{D}_L}}_{\text{Stage 1: Feature Learning}} \quad \longrightarrow \quad \underbrace{g_\phi^{(H)} \xrightarrow{\text{adapt with } \mathcal{D}_H}}_{\text{Stage 2: Knowledge Transfer}}$$

### 4.2 Taxonomy of Transfer Mechanisms

We categorize transfer learning strategies by **what is transferred**:

| Transfer Level | What is Shared | Models |
|---------------|----------------|--------|
| **Representation** | Learned features / hidden activations | Sequential, Progressive, Adapter |
| **Parameters** | Network weights (hard/soft) | Soft Parameter Sharing, Two-Stage Joint |
| **Output** | Predictions / soft targets | Knowledge Distillation, Pseudo-Labeling |
| **Data** | Augmented training data | Curriculum Learning |
| **Distribution** | Feature space alignment | Domain Adaptation (MMD) |
| **Gradient** | End-to-end gradient flow | DNGO-Joint, DNGO-Gradient |

---

## 5. Transfer Mechanisms and Architectures

### 5.1 Fine-Tuning Family

#### (a) Sequential Fine-Tuning

1. Train $g_\theta^{(L)}$ on $\mathcal{D}_L$ (200 epochs)
2. **Freeze** $\theta$; train $g_\phi^{(H)}$ on $\mathcal{D}_H$ using frozen LF predictions (100 epochs)

- LF parameters are **not updated** during HF training
- HF network receives $[\mathbf{x},\, g_\theta^{(L)}(\mathbf{x})]$ as input

#### (b) Progressive Fine-Tuning

Gradual unfreezing to mitigate catastrophic forgetting:

1. Train $g_\theta^{(L)}$ on $\mathcal{D}_L$ (100 epochs)
2. Freeze all; train HF **output layer only** (50 epochs, lr=$10^{-3}$)
3. Unfreeze last 2 layers; fine-tune (50 epochs, lr=$10^{-4}$)

- Progressively lower learning rate prevents overwriting pretrained features
- Analogous to discriminative fine-tuning in NLP

#### (c) Adapter Networks

Parameter-efficient transfer via bottleneck modules:

1. Train backbone $g_\theta^{(L)}$ on $\mathcal{D}_L$ (200 epochs)
2. **Freeze backbone**; insert and train adapter layers + HF output head (100 epochs)

**Adapter architecture** (inserted after each hidden layer):
$$\text{Adapter}(\mathbf{h}) = \mathbf{h} + \mathbf{W}_{\text{up}} \cdot \text{ReLU}(\mathbf{W}_{\text{down}} \cdot \mathbf{h})$$

where $\mathbf{W}_{\text{down}} \in \mathbb{R}^{b \times d_h}$, $\mathbf{W}_{\text{up}} \in \mathbb{R}^{d_h \times b}$, bottleneck $b = 16 \ll d_h = 64$.

- Only ~25% additional parameters are trainable
- Residual connection preserves pretrained representations

### 5.2 Parameter Sharing Family

#### (a) Two-Stage Joint Training

1. **Stage 1**: Train LF only (100 epochs)
2. **Stage 2**: Joint training on both fidelities (100 epochs)

Joint loss:
$$\mathcal{L}_{\text{joint}} = \lambda_L \cdot \mathcal{L}_{\text{LF}} + \lambda_H \cdot \mathcal{L}_{\text{HF}}, \qquad \lambda_L = 0.3,\; \lambda_H = 0.7$$

Higher HF weight ensures the model prioritizes target-fidelity accuracy.

#### (b) Soft Parameter Sharing

Concurrent training of separate LF and HF networks with a **soft regularization** penalty encouraging parameter similarity:

$$\mathcal{L} = \lambda_L \mathcal{L}_{\text{LF}} + \lambda_H \mathcal{L}_{\text{HF}} + \lambda_{\text{soft}} \sum_l \|\mathbf{W}_l^{(L)} - \mathbf{W}_l^{(H)}\|_F^2$$

where $\lambda_{\text{soft}} = 0.01$ and $\|\cdot\|_F$ is the Frobenius norm.

- Does **not** force identical weights (unlike hard sharing)
- Allows fidelity-specific specialization while maintaining structural similarity

### 5.3 Knowledge Distillation

The LF network serves as a **teacher**, and the HF network is the **student**:

$$\mathcal{L}_{\text{KD}} = (1 - \alpha) \cdot \underbrace{\text{MSE}(\hat{y}_{\text{student}},\, y^{(H)})}_{\text{hard loss}} + \alpha \cdot \underbrace{T^2 \cdot \text{MSE}\!\left(\frac{\hat{y}_{\text{student}}}{T},\, \frac{\hat{y}_{\text{teacher}}}{T}\right)}_{\text{soft loss}}$$

- **Temperature** $T = 3.0$: Smooths teacher outputs to expose inter-sample relationships
- **Distillation weight** $\alpha = 0.3$: Balances hard labels vs. soft knowledge
- $T^2$ scaling compensates for reduced gradient magnitude at high temperature

### 5.4 Semi-Supervised Transfer

#### (a) Pseudo-Labeling

1. Train $g_\theta^{(L)}$ on $\mathcal{D}_L$ (200 epochs)
2. Compute global offset: $\Delta = \frac{1}{|\mathcal{D}_H|}\sum_{i}(y_i^{(H)} - g_\theta^{(L)}(\mathbf{x}_i^{(H)}))$
3. Generate pseudo-labels: $\tilde{y}_j^{(H)} = g_\theta^{(L)}(\mathbf{x}_j^{(L)}) + \Delta$ for all LF points
4. Train HF network on combined data:

$$\mathcal{L}_{\text{PL}} = \mathcal{L}_{\text{real}} + \lambda_{\text{pseudo}} \cdot \mathcal{L}_{\text{pseudo}}, \qquad \lambda_{\text{pseudo}} = 0.5$$

- **Data augmentation effect**: Expands HF training set using shifted LF predictions
- Offset correction accounts for systematic fidelity bias

#### (b) Curriculum Learning

1. Train $g_\theta^{(L)}$ on $\mathcal{D}_L$ (200 epochs)
2. Compute residuals: $r_i = |y_i^{(H)} - g_\theta^{(L)}(\mathbf{x}_i^{(H)})|$
3. Sort HF samples by residual: easy (small $r_i$) to hard (large $r_i$)
4. Gradually introduce harder samples during HF training

- **Easy samples**: LF approximation is already good (small correction needed)
- **Hard samples**: Large fidelity gap requires substantial adaptation
- Curriculum prevents early overfitting to noisy, difficult examples

### 5.5 Domain Adaptation / Alignment

#### (a) Domain Adaptation (MMD)

1. Train $g_\theta^{(L)}$ on $\mathcal{D}_L$ (200 epochs)
2. Fine-tune HF with task loss + distribution alignment:

$$\mathcal{L}_{\text{DA}} = \mathcal{L}_{\text{task}} + \lambda_{\text{MMD}} \cdot \text{MMD}^2(\mathcal{F}_L,\, \mathcal{F}_H)$$

**Maximum Mean Discrepancy (MMD)** with RBF kernel ($\sigma = 1.0$):

$$\text{MMD}^2(\mathcal{F}_L, \mathcal{F}_H) = \frac{1}{n_L^2}\sum_{i,j} k(\mathbf{f}_i^L, \mathbf{f}_j^L) - \frac{2}{n_L n_H}\sum_{i,j} k(\mathbf{f}_i^L, \mathbf{f}_j^H) + \frac{1}{n_H^2}\sum_{i,j} k(\mathbf{f}_i^H, \mathbf{f}_j^H)$$

where $\mathcal{F}_L, \mathcal{F}_H$ are the hidden feature representations and $\lambda_{\text{MMD}} = 0.1$.

- Forces **feature space alignment** between LF and HF domains
- Reduces distribution shift when transferring representations

#### (b) DNGO-Joint (Detached Gradient)

Joint training of both fidelities from initialization (300 epochs):

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{LF}} + (1 - \alpha) \cdot \mathcal{L}_{\text{HF}}, \qquad \alpha = 0.5$$

**Key design**: The gradient from $\mathcal{L}_{\text{HF}}$ is **detached** from LF parameters:

$$\frac{\partial \mathcal{L}_{\text{HF}}}{\partial \theta^{(L)}} = 0 \quad (\text{via } \texttt{torch.no\_grad()})$$

- Prevents HF loss from corrupting learned LF representations
- LF network evolves only through its own loss signal

#### (c) DNGO-Gradient (End-to-End)

Full end-to-end gradient flow across both fidelities (300 epochs):

$$\frac{\partial \mathcal{L}}{\partial \theta^{(L)}} = \alpha \frac{\partial \mathcal{L}_{\text{LF}}}{\partial \theta^{(L)}} + (1-\alpha) \frac{\partial \mathcal{L}_{\text{HF}}}{\partial \theta^{(L)}}$$

- **Differentiated learning rates**: LF lr=$10^{-3}$, HF lr=$5 \times 10^{-4}$
- HF gradients directly shape LF features for downstream HF accuracy
- Risk: LF accuracy may degrade if HF signal dominates

---

## 6. Training and Implementation Details

### 6.1 Shared Training Configuration

| Component | Value |
|-----------|-------|
| Hidden dimension | 64 |
| Number of hidden layers | 2 |
| Activation function | Tanh |
| Optimizer | Adam |
| Weight decay (L2) | $10^{-4}$ |
| LF learning rate | $10^{-3}$ |
| HF learning rate | $10^{-3}$ ($10^{-4}$ for Progressive Stage 3) |
| Loss function | MSE |

### 6.2 Per-Model Training Schedule

| Model | LF Epochs | HF Epochs | Total Stages |
|-------|-----------|-----------|--------------|
| Sequential | 200 | 100 | 2 |
| Progressive | 100 | 50 + 50 | 3 |
| Two-Stage Joint | 100 | 100 (joint) | 2 |
| Knowledge Distillation | 200 | 100 | 2 |
| Adapter | 200 | 100 | 2 |
| Pseudo-Labeling | 200 | 100 | 2 |
| Curriculum | 200 | 100 | 2 |
| Soft Parameter Sharing | 200 (joint) | - | 1 |
| Domain Adaptation (MMD) | 200 | 100 | 2 |
| DNGO-Joint | 300 (joint) | - | 1 |
| DNGO-Gradient | 300 (joint) | - | 1 |
| MFGP | N/A (GP) | N/A (GP) | 1 |

### 6.3 Data Preprocessing

1. **Combined scaling**: LF and HF data are pooled for StandardScaler fitting
   - $\mathbf{X}_{\text{all}} = [\mathbf{X}_L;\, \mathbf{X}_H]$, $\mathbf{y}_{\text{all}} = [\mathbf{y}_L;\, \mathbf{y}_H]$
   - Ensures consistent scale across fidelities

2. **Feature representation**:
   - Continuous benchmarks: Normalized $[0, 1]^d$ coordinates
   - Molecular benchmarks: RDKit 2D descriptors $\rightarrow$ PCA ($d=10$ or $14$)

### 6.4 Model-Specific Hyperparameters

| Hyperparameter | Model | Value |
|---------------|-------|-------|
| $\alpha_{\text{KD}}$ | Knowledge Distillation | 0.3 |
| $T$ (temperature) | Knowledge Distillation | 3.0 |
| $\lambda_{\text{soft}}$ | Soft Parameter Sharing | 0.01 |
| $\lambda_{\text{MMD}}$ | Domain Adaptation | 0.1 |
| $\lambda_{\text{pseudo}}$ | Pseudo-Labeling | 0.5 |
| $\alpha$ (LF weight) | DNGO-Joint / DNGO-Gradient | 0.5 |
| $\lambda_L / \lambda_H$ | Two-Stage Joint | 0.3 / 0.7 |
| Bottleneck dim $b$ | Adapter | 16 |
| BLR $\alpha$ | LF-BLR | 1.0 |
| BLR $\beta$ | LF-BLR | 25.0 |

### 6.5 Reproducibility

- **Seeds**: 20 independent runs per configuration (seed 42--61)
- **Reporting**: Mean $\pm$ SE ($\text{SE} = \text{Std} / \sqrt{n}$)
- **Error handling**: Model fitting failures trigger random fallback selection
- **Parallel execution**: `torch.multiprocessing` with `spawn` start method (CUDA-safe)
- **Fixed architecture**: Identical hidden_dim=64, num_layers=2 across all DNN models for fair comparison

---

## 7. Benchmarks and Experimental Protocol

### 7.1 Benchmark Suite

| Benchmark | $d$ | $\rho$ | $R^2$ | $B$ | Objective | Domain |
|-----------|-----|--------|-------|-----|-----------|--------|
| Branin-Fav | 2 | 0.1 | 0.97 | 50 | Minimize | Synthetic |
| Branin-Unfav | 2 | 0.5 | 0.56 | 50 | Minimize | Synthetic |
| Park-Fav | 4 | 0.1 | 0.88 | 50 | Minimize | Synthetic |
| Park-Unfav | 4 | 0.5 | 0.42 | 50 | Minimize | Synthetic |
| COFs | 14 | 0.065 | 0.98 | 30 | Maximize | Chemistry |
| FreeSolv | 10 | 0.1 | 0.88 | 50 | Minimize | Chemistry |
| Polarizability | 10 | 0.167 | 0.99 | 30 | Maximize | Chemistry |

### 7.2 Synthetic Functions

**Branin** ($d=2$): The LF version parameterizes the quadratic coefficient $b$ as a function of $\alpha \in [0,1]$:
$$b_{\text{LF}} = \frac{5.1}{4\pi^2} - 0.1(1 - \alpha)$$

**Park** ($d=4$): The LF version modifies two terms controlled by $\alpha$:
- $x_4$ coefficient: $3 \to 3 - 1.5(1 - \alpha)$
- Exponential: $\exp(1 + \sin x_3) \to \exp(1 + \alpha \sin x_3)$

### 7.3 Experimental Volume

- **Models**: 12 (11 DNN-based + 1 MFGP baseline)
- **Benchmarks**: 7
- **Seeds**: 20
- **Total runs**: $12 \times 7 \times 20 = 1{,}680$

---

## 8. Additional Considerations

### 8.1 Residual Learning as Inductive Bias

The HF network architecture embeds a strong inductive bias:

$$\hat{y}^{(H)} = \underbrace{g^{(L)}(\mathbf{x})}_{\text{coarse estimate}} + \underbrace{\delta(\mathbf{x})}_{\text{learned correction}}$$

This is analogous to residual connections in deep learning: the HF network only needs to learn the **discrepancy**, which is typically a smoother function than $f^{(H)}$ itself. When fidelity correlation is high ($R^2 \to 1$), $\delta \to 0$ and minimal HF data suffices.

### 8.2 Computational Complexity

| Component | Complexity |
|-----------|-----------|
| DNN forward pass | $O(d \cdot h + h^2)$ per layer |
| DNN training | $O(E \cdot N \cdot L \cdot h^2)$ |
| BLR fitting | $O(N \cdot D^2 + D^3)$ |
| BLR prediction | $O(N_* \cdot D^2)$ |
| MFGP fitting | $O(N^3)$ |
| MFGP prediction | $O(N^2 \cdot N_*)$ |

where $E$ = epochs, $N$ = data size, $L$ = layers, $h$ = hidden dim, $D$ = feature dim.

DNN-based approaches scale **linearly** in $N$ (per epoch), while MFGP scales **cubically**, yielding 10--18$\times$ speedup in practice.

### 8.3 Exploration--Exploitation Trade-off

The dual acquisition strategy addresses a fundamental asymmetry in multi-fidelity settings:

- **LF evaluations are cheap** $\Rightarrow$ can afford exploration (EI with calibrated uncertainty)
- **HF evaluations are expensive** $\Rightarrow$ must exploit (greedy $\arg\min \mu^{(H)}$)

This asymmetric policy maximizes information gain per unit cost.

### 8.4 Negative Transfer

When $R^2$ is low (unfavorable regime), transfer learning can **hurt** performance if the LF representation is misleading. Architectures differ in their robustness to negative transfer:

- **Adapter / Progressive**: Freeze backbone $\Rightarrow$ limited exposure to misleading LF gradients
- **DNGO-Gradient**: Full gradient flow $\Rightarrow$ vulnerable to LF corruption
- **Pseudo-Labeling**: Offset correction partially mitigates systematic bias

---

## Appendix A. Detailed Implementation Specifications

---

### A.1 Candidate Set Construction and Acquisition Optimization

#### A.1.1 Candidate Pool: Offline Discrete Grid

All benchmarks operate over a **finite, pre-computed candidate pool** $\mathcal{X}_{\text{cand}}$. This is an **offline pool setting**, not an online continuous optimization. The entire pool is materialized at benchmark initialization.

**Synthetic benchmarks** — uniform grid in $[0,1]^d$:

| Benchmark | $d$ | Grid size per dim | $|\mathcal{X}_{\text{cand}}|$ |
|-----------|-----|-------------------|------------------------------|
| Branin (2D) | 2 | 50 | $50^2 = 2{,}500$ |
| Park (4D) | 4 | $\lceil 50^{0.5} \rceil = 8$ | $8^4 = 4{,}096$ |

**Chemistry benchmarks** — full molecular dataset (no sub-sampling):

| Benchmark | Source | $|\mathcal{X}_{\text{cand}}|$ | Feature dim |
|-----------|--------|-------------------------------|-------------|
| COFs | Structural descriptors | 608 | 14 |
| FreeSolv | SMILES → RDKit → PCA | 640 | 10 |
| Polarizability | SMILES → RDKit → PCA | 1,134 | 10 |

#### A.1.2 Acquisition Maximization

Since the candidate set is **finite and pre-computed**, acquisition optimization reduces to **exhaustive enumeration**:

$$\mathbf{x}_{t+1} = \arg\max_{\mathbf{x} \in \mathcal{X}_{\text{cand}} \setminus \mathcal{S}_t} \alpha(\mathbf{x})$$

where $\mathcal{S}_t$ is the set of previously sampled indices. No gradient-based optimization, multi-start, or Sobol sampling is needed. The surrogate predicts over the entire pool (vectorized), and the best unsampled point is selected via `argmax` (EI) or `argmin` (greedy).

**Masking**: Already-sampled points are excluded by setting $\alpha(\mathbf{x}_i) = -\infty$ for EI, or $\mu(\mathbf{x}_i) = +\infty$ for greedy argmin.

#### A.1.3 Surrogate Retraining Schedule

The surrogate is **retrained from scratch at every BO iteration**. Each iteration:

1. A fresh model instance is instantiated: `model = model_class(dim, device=device)`
2. `model.fit(X_lf, y_lf, X_hf, y_hf)` trains from randomly initialized weights
3. No warm-starting or weight inheritance between iterations

This ensures that each prediction is based on the full accumulated dataset without path-dependent artifacts. The computational cost is acceptable given the small dataset sizes (typically $<100$ points) and shallow networks (2-layer MLP, 64 hidden units).

---

### A.2 Handling of Minimize/Maximize Objectives

#### A.2.1 Negation for Maximization Tasks

Benchmarks with maximization objectives are **negated** at data loading to unify the optimization loop as minimization:

| Benchmark | Original Objective | `negate` flag | Internal Treatment |
|-----------|--------------------|---------------|-------------------|
| Branin-Fav/Unfav | Minimize | `False` | $y$ as-is |
| Park-Fav/Unfav | Minimize | `False` | $y$ as-is |
| **COFs** | **Maximize** | **`True`** | $y \leftarrow -y$ |
| FreeSolv | Minimize | `False` | $y$ as-is |
| **Polarizability** | **Maximize** | **`True`** | $y \leftarrow -y$ |

When `negate=True`:
```python
self.y_hf = -self.y_hf
self.y_lf = -self.y_lf
```

This transformation is applied **before** computing $f^* = \min y^{(H)}$ (which corresponds to $\max$ in the original space). All downstream operations—EI, argmin, regret—operate consistently in the minimization frame.

#### A.2.2 Regret Definition (Unified)

$$r_T = \min_{t: s_t = H} y_t^{(H)} - f^* = \min_{t: s_t = H} y_t^{(H)} - \min_{\mathbf{x} \in \mathcal{X}} y^{(H)}(\mathbf{x})$$

For negated benchmarks, this is equivalent to $f^*_{\text{orig}} - \max_{t} y_{t,\text{orig}}^{(H)}$ in the original maximization space.

Regret is clamped: $r_T = \max(0,\, r_T)$ to prevent numerical artifacts.

---

### A.3 Data Scaling and Information Leakage

#### A.3.1 Scaling Procedure

At **each BO iteration**, when `model.fit(X_lf, y_lf, X_hf, y_hf)` is called:

```python
X_all = np.vstack([X_lf, X_hf])
y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
X_scaled = scaler_x.fit_transform(X_all)   # fit on observed data only
y_scaled = scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()
```

**Key points regarding leakage**:

1. **Input scaling ($\mathbf{X}$)**: The `StandardScaler` is **fit only on the observed data** $\{X_{\text{lf}}, X_{\text{hf}}\}$ at each iteration, **not** on the full candidate pool. This is safe—no future information is used.

2. **Target scaling ($\mathbf{y}$)**: Similarly, $y$-scaling is fit only on observed $\{y_{\text{lf}}, y_{\text{hf}}\}$. No leakage.

3. **Feature preprocessing for chemistry benchmarks**: The feature matrix $X$ of the full pool is StandardScaler-normalized and PCA-transformed **at benchmark initialization** (before any BO iteration). This is analogous to knowing the feature space of the candidate library and does not constitute target leakage, as only input features (not $y$ values) are used.

#### A.3.2 Fidelity-Combined vs. Fidelity-Separate Scaling

Scalers are fit on **LF + HF data combined**:

$$\mu_y = \text{mean}([y^{(L)}_1, \ldots, y^{(L)}_{n_L}, y^{(H)}_1, \ldots, y^{(H)}_{n_H}])$$

**Rationale**: Since the HF network learns a residual $\delta = y^{(H)} - y^{(L)}$, both fidelities must share the same scale for the residual to be meaningful. Separate scaling would distort the fidelity gap.

#### A.3.3 Summary

| Component | Fit on | Leakage risk |
|-----------|--------|-------------|
| $X$ StandardScaler (in model) | Observed $\{X_L, X_H\}$ per iteration | None |
| $y$ StandardScaler (in model) | Observed $\{y_L, y_H\}$ per iteration | None |
| $X$ StandardScaler (chemistry features) | Full candidate pool (features only) | None (no target info) |
| PCA (chemistry features) | Full candidate pool (features only) | None (no target info) |

---

### A.4 Uncertainty Quantification: Consistency and Scope

#### A.4.1 Which $\sigma(\mathbf{x})$ Enters the Acquisition Function?

The UQ strategy differs by **fidelity selection** and **model type**:

| Condition | $\mu$ source | $\sigma$ source | Acquisition |
|-----------|-------------|-----------------|-------------|
| **LF eval, DNN model** | LF-BLR mean $\mu^{(L)}_{\text{BLR}}$ | LF-BLR std $\sigma^{(L)}_{\text{BLR}}$ | $\arg\max$ EI$(\mu^{(L)}, \sigma^{(L)}, y^{(L)}_{\text{best}})$ |
| **HF eval, DNN model** | HF residual network $\mu^{(H)}$ | *Not used* | $\arg\min\ \mu^{(H)}$ (greedy) |
| **LF eval, MFGP** | GP posterior $\mu^{(H)}_{\text{GP}}$ | GP posterior $\sigma^{(H)}_{\text{GP}}$ | $\arg\max$ EI$(\mu^{(H)}, \sigma^{(H)}, y^{(H)}_{\text{best}})$ |
| **HF eval, MFGP** | GP posterior $\mu^{(H)}_{\text{GP}}$ | *Not used* | $\arg\min\ \mu^{(H)}$ (greedy) |

#### A.4.2 HF UQ: Not Applied

BLR is applied **only to the LF network's last layer**. The HF network does **not** have BLR-based UQ because:

1. **Data scarcity**: HF data is too scarce (typically $<10$ points) for reliable BLR posterior estimation
2. **Exploitation policy**: HF selection uses greedy $\arg\min\ \mu^{(H)}$, which does not require $\sigma$
3. **Residual structure**: The HF network's output $y^{(H)} = y^{(L)} + \delta$ is a residual correction; BLR on $\delta$ features would conflate LF uncertainty with residual uncertainty

#### A.4.3 HF Prediction Mechanism

The HF predictive mean $\mu^{(H)}(\mathbf{x})$ is computed via the **residual architecture**:

$$\mu^{(H)}(\mathbf{x}) = g_\theta^{(L)}(\mathbf{x}) + g_\phi^{(H)}\!\left([\mathbf{x},\, g_\theta^{(L)}(\mathbf{x})]\right)$$

Both $g^{(L)}$ and $g^{(H)}$ are **deterministic** forward passes (no dropout, no sampling). The output is inverse-scaled to the original $y$-space.

#### A.4.4 LF-BLR Best Value ($y_{\text{best}}$)

For LF acquisition via EI, the incumbent best is:

$$y^{(L)}_{\text{best}} = \min_{i \in \mathcal{D}_L} y_i^{(L)}$$

This is the best **observed LF value** (not HF), which is appropriate since EI on LF predictions should reference the LF landscape.

---

### A.5 MFGP Baseline: Detailed Configuration

#### A.5.1 BoTorch Implementation

```python
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
```

#### A.5.2 Kernel and Noise

`SingleTaskMultiFidelityGP` defaults:

| Setting | Value |
|---------|-------|
| Kernel | Matérn-5/2 with ARD (Automatic Relevance Determination) |
| Lengthscale prior | $\text{GammaPrior}(3.0, 6.0)$ per dimension |
| Outputscale prior | $\text{GammaPrior}(2.0, 0.15)$ |
| Noise | **Learned** (inferred via MLL); $\text{GammaPrior}(1.1, 0.05)$ |
| Fidelity kernel | Downsampling kernel on fidelity dimension |

#### A.5.3 Input/Output Transformations

- **Input normalization**: Not applied externally (BoTorch handles internally via `normalize` in acquisition)
- **Outcome standardization**: `Standardize(m=1)` applied — subtracts mean, divides by std of training $y$
- **Fidelity encoding**: Appended as the last input dimension; LF $\to 0$, HF $\to 1$

#### A.5.4 Hyperparameter Optimization

```python
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)  # L-BFGS-B optimizer, default settings
```

- Optimizer: **L-BFGS-B** (via `scipy.optimize`)
- Restarts: BoTorch default (typically 1 random restart)
- Convergence: Default tolerances

#### A.5.5 Retraining

Like all other models, MFGP is **retrained from scratch at every BO iteration**. A new `SingleTaskMultiFidelityGP` instance is created each time with the accumulated dataset.

#### A.5.6 Acquisition Function (MFGP)

MFGP uses the same EI formula as DNN+LF-BLR models, but with:
- **GP posterior** $\mu, \sigma$ (calibrated, model-based uncertainty)
- EI is applied to **HF predictions** (fidelity=1), not LF
- For HF evaluation steps, MFGP also uses greedy $\arg\min$ (same as DNN models)

---

### A.6 Training Configuration: Missing Details

#### A.6.1 Batch Size

All DNN models use **full-batch gradient descent** (no mini-batching). Given the small dataset sizes (typically $N < 300$ for LF, $N < 30$ for HF), mini-batching is unnecessary and would introduce unnecessary stochasticity.

#### A.6.2 Regularization and Stability

| Technique | Status |
|-----------|--------|
| Weight decay (L2) | $10^{-4}$ (applied via `Adam(weight_decay=1e-4)`) |
| Dropout | **Not used** in any model |
| Batch normalization | **Not used** |
| Gradient clipping | **Not used** |
| Early stopping | **Not used**; all models run for fixed epoch counts |
| Learning rate scheduling | **Not used** (constant LR throughout training) |

#### A.6.3 Hyperparameter Selection Policy

All model-specific hyperparameters are **fixed** (not tuned per benchmark):

| Hyperparameter | Model | Value | Selection basis |
|---------------|-------|-------|-----------------|
| $\alpha_{\text{KD}} = 0.3$ | Knowledge Distillation | Fixed | Literature default |
| $T = 3.0$ | Knowledge Distillation | Fixed | Literature default |
| $\lambda_{\text{soft}} = 0.01$ | Soft Parameter Sharing | Fixed | Preliminary experiments |
| $\lambda_{\text{MMD}} = 0.1$ | Domain Adaptation | Fixed | Preliminary experiments |
| $\sigma_{\text{RBF}} = 1.0$ | Domain Adaptation (MMD kernel) | Fixed | Standard choice |
| $\lambda_{\text{pseudo}} = 0.5$ | Pseudo-Labeling | Fixed | Equal weighting |
| $\alpha = 0.5$ | DNGO-Joint, DNGO-Gradient | Fixed | Equal fidelity weighting |
| $\lambda_L / \lambda_H = 0.3 / 0.7$ | Two-Stage Joint | Fixed | HF-priority heuristic |
| Bottleneck dim $b = 16$ | Adapter | Fixed | $\approx d_h / 4$ |
| BLR $\alpha = 1.0$ | LF-BLR | Fixed | Uninformative prior |
| BLR $\beta = 1.0$ | LF-BLR | Fixed | Unit noise precision |

**Rationale for fixed hyperparameters**: To ensure fair comparison, all models share the same architecture ($h = 64$, 2 layers, Tanh) and all model-specific constants are fixed across benchmarks. No per-benchmark tuning is performed.

---

### A.7 Round-Robin Fidelity Schedule: Precise Implementation

#### A.7.1 LF-per-HF Calculation

The number of LF evaluations between each HF evaluation is:

$$n_{\text{lf\_per\_hf}} = \max\!\left(1,\, \lfloor 1 / \rho \rfloor\right)$$

| Benchmark | $\rho$ | $\lfloor 1/\rho \rfloor$ | Effective ratio |
|-----------|--------|--------------------------|-----------------|
| Branin-Fav | 0.1 | 10 | 10 LF : 1 HF |
| Park-Fav | 0.1 | 10 | 10 LF : 1 HF |
| COFs | 0.065 | 15 | 15 LF : 1 HF |
| Polarizability | 0.167 | 5 | 5 LF : 1 HF |
| Branin-Unfav | 0.5 | 2 | 2 LF : 1 HF |
| Park-Unfav | 0.5 | 2 | 2 LF : 1 HF |
| FreeSolv | 0.1 | 10 | 10 LF : 1 HF |

Note: For $\rho = 0.065$ (COFs), $\lfloor 1/0.065 \rfloor = 15$, which means 15 LF evaluations per HF evaluation. The cost per cycle is $15 \times 0.065 + 1 = 1.975 \approx 2$ units.

#### A.7.2 Schedule Logic

```
lf_counter = 0  (reset after each HF evaluation)

At each iteration:
  remaining = B - current_budget

  if remaining >= 1.0:
      if remaining >= rho AND lf_counter < lf_per_hf:
          → Evaluate LF (cost = rho, lf_counter++)
      else:
          → Evaluate HF (cost = 1.0, lf_counter = 0)
  elif remaining >= rho:
      → Evaluate LF (squeeze remaining budget)
  else:
      → STOP
```

#### A.7.3 Initialization vs. BO Phase

- **Initialization phase**: Indices are sampled via LHS/FPS, then split deterministically (first $n_{\text{init\_lf}}$ for LF, remainder for HF). **No round-robin** during initialization.
- **BO phase**: Round-robin starts immediately after initialization, with `lf_counter = 0`.

#### A.7.4 Budget Tracking

$$B_t = \underbrace{n_{\text{init\_lf}} \cdot \rho + n_{\text{init\_hf}} \cdot 1.0}_{\text{initialization}} + \sum_{i=1}^{t} c_{s_i}$$

The loop terminates when $B - B_t < \rho$ (cannot afford even one LF evaluation).

---

### A.8 Molecular Benchmark Details

#### A.8.1 Dataset Sizes and Feature Construction

| Benchmark | $N$ (pool) | Raw features | Feature pipeline | Final $d$ |
|-----------|-----------|-------------|------------------|-----------|
| **COFs** | 608 | 14 structural descriptors (pore diameter, void fraction, surface area, crystal density, elemental fractions B/O/C/H/Si/N/S/P/halogens/metals) | StandardScaler | 14 |
| **FreeSolv** | 640 | SMILES strings | SMILES $\to$ RDKit 2D descriptors ($\approx$210) $\to$ StandardScaler $\to$ PCA(10) | 10 |
| **Polarizability** | 1,134 | SMILES strings | SMILES $\to$ RDKit 2D descriptors ($\approx$210) $\to$ StandardScaler $\to$ PCA(10) | 10 |

#### A.8.2 RDKit Descriptor Details

For SMILES-based benchmarks (FreeSolv, Polarizability):

1. **Descriptor calculation**: All RDKit `Descriptors._descList` descriptors (approximately 210 descriptors including molecular weight, LogP, TPSA, number of H-bond donors/acceptors, ring counts, etc.)
2. **NaN handling**: `np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)`
3. **Scaling**: `StandardScaler().fit_transform(features)` on the full descriptor matrix
4. **PCA**: `PCA(n_components=10).fit_transform(features_scaled)`

#### A.8.3 PCA Fit Scope

PCA is fit on the **full candidate pool** (all molecules), not on observed-only data. This is justified because:
- PCA operates on **input features** ($X$), not targets ($y$)
- The candidate pool represents a known molecular library (analogous to knowing the search space)
- This is standard practice in fixed-pool BO settings

#### A.8.4 PCA Dimension Selection

- $d = 10$ for FreeSolv and Polarizability: Standard dimensionality for molecular descriptor PCA in similar benchmarks
- $d = 14$ for COFs: No PCA applied; the original 14 structural descriptors are used directly (already compact)

#### A.8.5 Furthest Point Sampling (FPS) for Chemistry

FPS is used for initial sampling in chemistry benchmarks (discrete molecular pools):

- **Distance metric**: Euclidean distance in the **feature space** (PCA-transformed and scaled)
- **Algorithm**: Greedy sequential — select point maximizing minimum distance to all previously selected points
- **Seed point**: Random (seed-dependent)

$$\mathbf{x}_{k+1} = \arg\max_{\mathbf{x} \in \mathcal{X} \setminus \mathcal{S}_k} \min_{\mathbf{x}' \in \mathcal{S}_k} \|\mathbf{x} - \mathbf{x}'\|_2$$

---

### A.9 Failure Handling and Fallback Policy

#### A.9.1 Failure Conditions

A model fitting or prediction failure is caught via a blanket `try/except Exception`:

```python
try:
    model = model_class(dim, device=device)
    model.fit(X_lf, y_lf, X_hf, y_hf)
    mean, std = model.predict(benchmark.X)
    # ... select next point via acquisition
except Exception as e:
    # Fallback
```

Failure modes include:
- **Numerical divergence**: NaN/Inf in network outputs (e.g., exploding gradients with Tanh saturation)
- **Linear algebra errors**: Singular matrix in BLR (`np.linalg.LinAlgError` → falls back to `pinv`)
- **GP fitting failure**: MLL optimization divergence in MFGP
- **Memory errors**: GPU OOM (rare given small model sizes)

#### A.9.2 Fallback Mechanism

On failure, a **uniformly random unsampled point** is selected:

```python
available = set(range(n_candidates)) - (lf_indices | hf_indices)
next_idx = np.random.choice(list(available))
```

- The selected fidelity follows the **pre-determined round-robin schedule** (not random)
- The budget is still consumed: `current_budget += cost`
- No error message is logged to the main output

#### A.9.3 Seed-Level Failure

If the entire seed run fails (e.g., crash during initialization), the result is recorded as:

```python
{'final_regret': np.nan, 'n_hf': 0, 'n_lf': 0, 'best_y': np.nan}
```

#### A.9.4 Impact Assessment

In practice, fallback events are **rare** (<1% of iterations across all experiments). The primary cause is MFGP fitting failure on very small datasets ($n < 5$). DNN models almost never trigger fallback due to the simplicity of the 2-layer architecture.

---

### A.10 Terminology and Naming Conventions

#### A.10.1 DNGO (Deep Networks for Global Optimization)

The name "DNGO" is adapted from Snoek et al. (2015), where deep networks are used as feature extractors with a Bayesian linear regression output layer. In our implementation:

- **DNGO-Joint**: LF and HF networks are trained **jointly** (single optimization loop). The critical design choice is **gradient detachment**: the HF loss does not backpropagate through the LF network's parameters.

  ```
  LF loss → updates θ_L only
  HF loss → updates θ_H only (LF features are computed with torch.no_grad())
  ```

  This prevents HF-induced corruption of LF feature representations.

- **DNGO-Gradient**: LF and HF networks are trained jointly with **full end-to-end gradient flow**. The HF loss directly influences LF parameters:

  ```
  LF loss → updates θ_L
  HF loss → updates θ_L AND θ_H (gradients flow through LF network)
  ```

  This allows the LF network to learn features that are specifically useful for downstream HF prediction, at the cost of potentially degrading LF-only accuracy.

#### A.10.2 Model Naming Summary

| Short name | Full name | Key mechanism |
|-----------|-----------|---------------|
| MFGP | Multi-Fidelity Gaussian Process | GP with fidelity kernel |
| Sequential | Sequential Fine-Tuning | Freeze LF → train HF |
| Progressive | Progressive Fine-Tuning | Gradual layer unfreezing |
| Two-Stage Joint | Two-Stage Joint Training | Pretrain LF → joint LF+HF |
| DNGO-Joint | Deep Network GO (Detached Joint) | Joint training, gradient detach |
| DNGO-Gradient | Deep Network GO (End-to-End) | Joint training, full gradients |
| KD | Knowledge Distillation | LF teacher → HF student |
| MMD | Domain Adaptation (MMD) | Feature alignment via MMD |
| Soft Sharing | Soft Parameter Sharing | Weight similarity regularization |
| Pseudo-Label | Pseudo-Labeling | LF-generated augmented data |
| Adapter | Adapter Networks | Bottleneck residual modules |

#### A.10.3 Abbreviations

| Abbreviation | Meaning |
|-------------|---------|
| HF | High-Fidelity |
| LF | Low-Fidelity |
| MFBO | Multi-Fidelity Bayesian Optimization |
| BLR | Bayesian Linear Regression |
| LF-BLR | BLR on LF network's last-layer features |
| EI | Expected Improvement |
| FPS | Furthest Point Sampling |
| LHS | Latin Hypercube Sampling |
| MLL | Marginal Log-Likelihood |
| MMD | Maximum Mean Discrepancy |
| ARD | Automatic Relevance Determination |
| PCA | Principal Component Analysis |

---
