# Baseline Implementation Spec — ICML 2026 Rebuttal

## Context

We have a multi-fidelity Bayesian optimization (MFBO) codebase that compares 11 DNN transfer-learning surrogates against one exact multi-fidelity GP (BoTorch `SingleTaskMultiFidelityGP`). Reviewers demand more baselines. Implement 4 new baselines that plug into the existing BO loop.

## Existing Protocol — DO NOT CHANGE

All new baselines MUST follow these rules exactly. Violating any of them makes the comparison unfair and unusable.

- **Acquisition function**: Expected Improvement (EI), maximization.
- **Fidelity schedule**: Deterministic round-robin. Step 0 = LF, Step 1 = HF, Step 2 = LF, …
- **Retraining**: From scratch every BO iteration. No warm-starting.
- **Seeds**: 20 random seeds per benchmark.
- **Benchmarks**: All 7: `Branin-Fav`, `Branin-Unfav`, `Park-Fav`, `Park-Unfav`, `FreeSolv`, `Polarizability`, `COFs`.
- **Search space**: Discrete pool-based. Candidates are removed once evaluated at any fidelity.
- **Fidelity levels**: 2 (LF=0, HF=1).
- **DNN architecture reference**: 2-layer MLP, width 64, tanh activation.
- **Reporting**: Simple regret curve (mean ± std over 20 seeds), wall-clock time per iteration, final best value.

## LF-HF Correlation Reference

Use this to sanity-check results:

| Benchmark | Correlation | Dim |
|---|---|---|
| Branin-Fav | 0.99 | 2 |
| Branin-Unfav | 0.10 | 2 |
| Park-Fav | 0.89 | 4 |
| Park-Unfav | 0.27 | 4 |
| FreeSolv | 0.94 | High (PCA) |
| Polarizability | 0.99 | High (PCA) |
| COFs | 0.94 | High (PCA) |

---

## Baseline 1: Sparse MFGP (SVGP)

### Why
Reviewers say exact GP has cubic cost and we only beat one weak baseline. This shows scalable GP also loses.

### Implementation

Input format: concatenate LF and HF data. Append fidelity indicator (0 or 1) as the last column.

```python
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

class SparseMFGP(ApproximateGP):
    def __init__(self, inducing_points, input_dim):
        """
        Args:
            inducing_points: (M, input_dim+1) tensor. Last dim = fidelity.
            input_dim: int. Feature dimensions WITHOUT fidelity.
        """
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim, active_dims=list(range(input_dim)))
        )
        self.fidelity_kernel = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)

    def forward(self, x):
        features = x[..., :-1]
        fidelity_idx = x[..., -1].long()
        mean = self.mean_module(features)
        covar = self.covar_module(features) * self.fidelity_kernel(fidelity_idx)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
```

Training:

```python
def train_sparse_mfgp(X_all, Y_all, n_inducing=100, n_epochs=500, lr=0.01):
    """
    X_all: (N_lf + N_hf, D+1). Last column is fidelity indicator.
    Y_all: (N_lf + N_hf,).
    """
    idx = torch.randperm(X_all.shape[0])[:n_inducing]
    inducing_points = X_all[idx].clone()
    input_dim = X_all.shape[1] - 1

    model = SparseMFGP(inducing_points, input_dim)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_all.shape[0])

    for _ in range(n_epochs):
        optimizer.zero_grad()
        output = model(X_all)
        loss = -mll(output, Y_all)
        loss.backward()
        optimizer.step()

    return model, likelihood
```

Acquisition: use HF posterior (fidelity=1) for EI on BOTH LF and HF turns. This matches how the existing MFGP baseline works. Reviewer SrGw flagged acquisition policy asymmetry — keep it consistent.

```python
def acquire_sparse_mfgp(model, likelihood, X_pool, best_hf_y):
    """Always query HF posterior for EI, regardless of current fidelity turn."""
    model.eval()
    likelihood.eval()
    fid_col = torch.ones(X_pool.shape[0], 1)  # always HF=1
    X_query = torch.cat([X_pool, fid_col], dim=-1)
    with torch.no_grad():
        posterior = likelihood(model(X_query))
        mu = posterior.mean
        sigma = posterior.variance.sqrt()
    z = (mu - best_hf_y) / (sigma + 1e-8)
    ei = (mu - best_hf_y) * torch.distributions.Normal(0, 1).cdf(z) \
         + sigma * torch.distributions.Normal(0, 1).log_prob(z).exp()
    return ei.argmax().item()
```

### Hyperparameters

- `n_inducing`: 100. Also run 50 and 200 for sensitivity; report in appendix.
- `lr`: 0.01
- `n_epochs`: 500
- Kernel: `ScaleKernel(RBFKernel(ARD)) * IndexKernel(num_tasks=2, rank=1)`

---

## Baseline 2: NARGP (Nonlinear Autoregressive GP)

### Why
Reviewer NQkm explicitly asked for "nonlinear multi-fidelity GP variants". NARGP (Perdikaris et al., Proc. R. Soc. A, 2017) is the standard one. It uses LF GP prediction as additional input to HF GP, enabling nonlinear cross-fidelity mapping.

### Implementation

Two-stage GP. Stage 1: fit GP on LF data. Stage 2: fit GP on HF data with augmented input `[X_hf, mu_lf(X_hf)]`.

```python
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

def train_nargp(X_lf, Y_lf, X_hf, Y_hf):
    """
    Returns: (gp_lf, gp_hf)
    Y_lf, Y_hf: (N, 1) shaped tensors.
    """
    # Stage 1: LF GP
    gp_lf = SingleTaskGP(X_lf, Y_lf)
    mll_lf = ExactMarginalLogLikelihood(gp_lf.likelihood, gp_lf)
    fit_gpytorch_mll(mll_lf)

    # Stage 2: HF GP with augmented input
    gp_lf.eval()
    with torch.no_grad():
        lf_mean = gp_lf.posterior(X_hf).mean  # (N_hf, 1)
    X_hf_aug = torch.cat([X_hf, lf_mean], dim=-1)

    gp_hf = SingleTaskGP(X_hf_aug, Y_hf)
    mll_hf = ExactMarginalLogLikelihood(gp_hf.likelihood, gp_hf)
    fit_gpytorch_mll(mll_hf)

    return gp_lf, gp_hf

def predict_nargp(gp_lf, gp_hf, X_test):
    """Returns (mean, variance) from NARGP HF posterior."""
    gp_lf.eval()
    gp_hf.eval()
    with torch.no_grad():
        lf_mean = gp_lf.posterior(X_test).mean
        X_test_aug = torch.cat([X_test, lf_mean], dim=-1)
        hf_post = gp_hf.posterior(X_test_aug)
    return hf_post.mean, hf_post.variance

def acquire_nargp(gp_lf, gp_hf, X_pool, best_hf_y):
    """EI from NARGP HF posterior."""
    mu, var = predict_nargp(gp_lf, gp_hf, X_pool)
    sigma = var.sqrt()
    z = (mu.squeeze() - best_hf_y) / (sigma.squeeze() + 1e-8)
    ei = (mu.squeeze() - best_hf_y) * torch.distributions.Normal(0, 1).cdf(z) \
         + sigma.squeeze() * torch.distributions.Normal(0, 1).log_prob(z).exp()
    return ei.argmax().item()
```

### Key constraint
Retrain BOTH gp_lf and gp_hf from scratch every BO iteration. Always fit gp_lf first, then gp_hf (order matters — gp_hf input depends on gp_lf output).

### Hyperparameters
Uses BoTorch defaults for SingleTaskGP (Matern 5/2, ARD, fit via L-BFGS). No extra hyperparameters to tune.

---

## Baseline 3: Deep Kernel Learning (DKL) + Multi-Fidelity

### Why
DKL puts a GP on top of NN features. If we give it the same architecture as our DNN surrogates (2-layer, width 64, tanh), then any performance difference is due to transfer mechanism, not representation power.

### Implementation

```python
import torch.nn as nn

class FeatureExtractor(nn.Module):
    """MUST match existing DNN surrogate architecture exactly."""
    def __init__(self, input_dim, bottleneck_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, bottleneck_dim),
        )
    def forward(self, x):
        return self.net(x)

class DKLMultiFidelityGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dim, bottleneck_dim=16):
        """
        train_x: (N, input_dim+1). Last column = fidelity indicator.
        """
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = FeatureExtractor(input_dim, bottleneck_dim)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=bottleneck_dim)
        )
        self.fidelity_kernel = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        features = x[..., :-1]
        fidelity_idx = x[..., -1].long()
        projected = self.scale_to_bounds(self.feature_extractor(features))
        mean = self.mean_module(projected)
        covar = self.covar_module(projected) * self.fidelity_kernel(fidelity_idx)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
```

Training:

```python
def train_dkl_mfgp(X_all, Y_all, input_dim, n_epochs=500):
    """
    X_all: (N, input_dim+1). Last col = fidelity.
    Y_all: (N,).
    """
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = DKLMultiFidelityGP(X_all, Y_all, likelihood, input_dim)
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': 1e-3},
        {'params': model.covar_module.parameters(), 'lr': 1e-2},
        {'params': model.mean_module.parameters(), 'lr': 1e-2},
        {'params': model.fidelity_kernel.parameters(), 'lr': 1e-2},
        {'params': likelihood.parameters(), 'lr': 1e-2},
    ])
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(n_epochs):
        optimizer.zero_grad()
        output = model(X_all)
        loss = -mll(output, Y_all)
        loss.backward()
        optimizer.step()

    return model, likelihood
```

Acquisition: same as Sparse MFGP — use HF posterior (fidelity=1) for EI.

### Hyperparameters
- Feature extractor: 2-layer, width 64, tanh, bottleneck_dim=16. **Do not change — must match existing DNN surrogates.**
- NN lr: 1e-3. GP lr: 1e-2.
- `n_epochs`: 500.

---

## Baseline 4: Successive Halving (Pool-based)

### Why
Reviewer peg1 recommended BOHB/Hyperband. Those are HPO-specific (fidelity = epochs), so direct application doesn't fit. We implement the core idea — successive halving — adapted to our pool-based 2-level setting. This is a surrogate-free baseline: no model learning at all.

### Implementation

Use this version that follows round-robin schedule to match existing protocol:

```python
import numpy as np

def successive_halving_roundrobin(X_pool, eval_lf, eval_hf, total_budget):
    """
    Round-robin compatible successive halving.
    - Odd steps (0, 2, 4, ...): LF evaluation. Pick random unevaluated candidate.
    - Even steps (1, 3, 5, ...): HF evaluation. Pick top-ranked by LF score among HF-unevaluated.

    Args:
        X_pool: (N, D) array. Full candidate pool.
        eval_lf: callable. idx -> float. Evaluates candidate at LF.
        eval_hf: callable. idx -> float. Evaluates candidate at HF.
        total_budget: int. Total number of evaluations (LF + HF combined).

    Returns:
        regret_history: list of float. Best HF value found so far at each step.
        hf_evaluated: dict. {idx: hf_score}.
    """
    n_pool = len(X_pool)
    lf_evaluated = {}    # idx -> lf_score
    hf_evaluated = {}    # idx -> hf_score
    best_hf = -np.inf
    regret_history = []

    for step in range(total_budget):
        if step % 2 == 0:  # LF turn
            unevaluated = set(range(n_pool)) - set(lf_evaluated.keys())
            if len(unevaluated) == 0:
                regret_history.append(best_hf)
                continue
            idx = np.random.choice(list(unevaluated))
            lf_evaluated[idx] = eval_lf(idx)

        else:  # HF turn
            candidates = [
                (idx, score) for idx, score in lf_evaluated.items()
                if idx not in hf_evaluated
            ]
            if len(candidates) == 0:
                regret_history.append(best_hf)
                continue
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_lf_idx = candidates[0][0]
            hf_score = eval_hf(best_lf_idx)
            hf_evaluated[best_lf_idx] = hf_score
            best_hf = max(best_hf, hf_score)

        regret_history.append(best_hf)

    return regret_history, hf_evaluated
```

### Key notes
- No model to train. No hyperparameters to tune.
- This baseline tests whether simple LF-based filtering can match learned surrogates.
- Expected to work well when LF-HF correlation is very high (Polarizability ~0.99) and poorly when correlation is low (Branin-Unfav ~0.10).
- On LF turns, candidates are selected randomly from unevaluated pool (no surrogate to guide selection).
- On HF turns, the candidate with the highest LF score (among those not yet HF-evaluated) is selected.

---

## Dependencies

```
torch>=2.0
gpytorch>=1.11
botorch>=0.10
numpy
scipy
```

All should already be in the existing environment. Verify before running.

---

## Output Checklist

After running all experiments, produce these files:

1. **Per-baseline CSV**: `results_{baseline_name}_{benchmark}_{seed}.csv` with columns `[step, fidelity, candidate_idx, observed_value, best_hf_so_far, wall_time_sec]`
2. **Aggregated CSV**: `results_summary.csv` with columns `[baseline, benchmark, mean_final_best, std_final_best, mean_time_per_iter, total_time]`
3. **Regret curves**: One plot per benchmark. Add 4 new baselines to existing plot. Use blue tones for GP variants, red tones for DNN, gray dashed for Successive Halving.

## Execution Order

Run in this order (fastest to slowest, so you can catch bugs early):

1. Successive Halving — no training, runs in seconds
2. NARGP — uses BoTorch SingleTaskGP, fastest GP
3. Sparse MFGP — variational training, moderate
4. DKL — NN + GP joint training, slowest

For each: run on `Branin-Fav` with 1 seed first as a smoke test before launching full 7×20 sweep.
