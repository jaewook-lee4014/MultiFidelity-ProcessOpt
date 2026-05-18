"""
Baselines for ICML 2026 rebuttal:
1. SparseMFGP (SVGP with IndexKernel)
2. NARGP (Nonlinear Autoregressive GP)
3. DKLMultiFidelity (Deep Kernel Learning + Multi-Fidelity)
4. SuccessiveHalving (surrogate-free baseline)
5. HF-Only Random Search (non-learning baseline)
6. LF-Screening (non-learning, surrogate-free MF baseline)
"""

import numpy as np
import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.preprocessing import StandardScaler
from typing import Tuple


# =============================================================================
# Baseline 1: Sparse MFGP (SVGP)
# =============================================================================

class _SparseMFGPModel(ApproximateGP):
    def __init__(self, inducing_points, input_dim):
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


class SparseMFGP:
    """Sparse Multi-Fidelity GP using SVGP with IndexKernel."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, device=None,
                 n_inducing: int = 100, n_epochs: int = 500, lr: float = 0.01):
        self.input_dim = input_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_inducing = n_inducing
        self.n_epochs = n_epochs
        self.lr = lr
        self.is_fitted = False

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        n_lf, n_hf = len(X_lf), len(X_hf)
        X_lf_fid = np.hstack([X_lf, np.zeros((n_lf, 1))])
        X_hf_fid = np.hstack([X_hf, np.ones((n_hf, 1))])
        X_all = np.vstack([X_lf_fid, X_hf_fid])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])

        X_t = torch.tensor(X_all, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_all, dtype=torch.float32).to(self.device)

        n_ind = min(self.n_inducing, X_t.shape[0])
        idx = torch.randperm(X_t.shape[0])[:n_ind]
        inducing_points = X_t[idx].clone()

        self.model = _SparseMFGPModel(inducing_points, self.input_dim).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=self.lr)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=X_t.shape[0])

        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            output = self.model(X_t)
            loss = -mll(output, y_t)
            loss.backward()
            optimizer.step()

        self.is_fitted = True

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_fid = np.hstack([X, np.ones((len(X), 1))])  # HF fidelity=1
        X_t = torch.tensor(X_fid, dtype=torch.float32).to(self.device)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            posterior = self.likelihood(self.model(X_t))
            mean = posterior.mean.cpu().numpy().flatten()
            std = posterior.variance.sqrt().cpu().numpy().flatten()
        return mean, np.maximum(std, 1e-6)

    def predict_lf(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_fid = np.hstack([X, np.zeros((len(X), 1))])  # LF fidelity=0
        X_t = torch.tensor(X_fid, dtype=torch.float32).to(self.device)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            posterior = self.likelihood(self.model(X_t))
            mean = posterior.mean.cpu().numpy().flatten()
            std = posterior.variance.sqrt().cpu().numpy().flatten()
        return mean, np.maximum(std, 1e-6)


# =============================================================================
# Baseline 2: NARGP (Nonlinear Autoregressive GP)
# =============================================================================

class NARGP:
    """Nonlinear Autoregressive GP (Perdikaris et al., 2017).
    Two-stage: LF GP -> augment HF input with LF predictions -> HF GP.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, device=None):
        self.input_dim = input_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_fitted = False

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        dtype = torch.float64
        X_lf_t = torch.tensor(X_lf, dtype=dtype).to(self.device)
        y_lf_t = torch.tensor(y_lf.flatten(), dtype=dtype).unsqueeze(-1).to(self.device)
        X_hf_t = torch.tensor(X_hf, dtype=dtype).to(self.device)
        y_hf_t = torch.tensor(y_hf.flatten(), dtype=dtype).unsqueeze(-1).to(self.device)

        # Stage 1: LF GP
        self.gp_lf = SingleTaskGP(X_lf_t, y_lf_t).to(self.device)
        mll_lf = ExactMarginalLogLikelihood(self.gp_lf.likelihood, self.gp_lf)
        fit_gpytorch_mll(mll_lf)

        # Stage 2: HF GP with augmented input [X_hf, mu_lf(X_hf)]
        self.gp_lf.eval()
        with torch.no_grad():
            lf_mean = self.gp_lf.posterior(X_hf_t).mean  # (N_hf, 1)
        X_hf_aug = torch.cat([X_hf_t, lf_mean], dim=-1)

        self.gp_hf = SingleTaskGP(X_hf_aug, y_hf_t).to(self.device)
        mll_hf = ExactMarginalLogLikelihood(self.gp_hf.likelihood, self.gp_hf)
        fit_gpytorch_mll(mll_hf)

        self.is_fitted = True

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        dtype = torch.float64
        X_t = torch.tensor(X, dtype=dtype).to(self.device)
        self.gp_lf.eval()
        self.gp_hf.eval()
        with torch.no_grad():
            lf_mean = self.gp_lf.posterior(X_t).mean
            X_aug = torch.cat([X_t, lf_mean], dim=-1)
            hf_post = self.gp_hf.posterior(X_aug)
            mean = hf_post.mean.cpu().numpy().flatten()
            std = hf_post.variance.sqrt().cpu().numpy().flatten()
        return mean, np.maximum(std, 1e-6)

    def predict_lf(self, X) -> Tuple[np.ndarray, np.ndarray]:
        dtype = torch.float64
        X_t = torch.tensor(X, dtype=dtype).to(self.device)
        self.gp_lf.eval()
        with torch.no_grad():
            lf_post = self.gp_lf.posterior(X_t)
            mean = lf_post.mean.cpu().numpy().flatten()
            std = lf_post.variance.sqrt().cpu().numpy().flatten()
        return mean, np.maximum(std, 1e-6)


# =============================================================================
# Baseline 3: DKL Multi-Fidelity
# =============================================================================

class _FeatureExtractor(nn.Module):
    """2-layer MLP matching existing DNN surrogate architecture."""
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


class _DKLMultiFidelityGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dim, bottleneck_dim=16):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = _FeatureExtractor(input_dim, bottleneck_dim)
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


class DKLMultiFidelity:
    """Deep Kernel Learning with Multi-Fidelity via IndexKernel."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, device=None,
                 n_epochs: int = 500, bottleneck_dim: int = 16):
        self.input_dim = input_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_epochs = n_epochs
        self.bottleneck_dim = bottleneck_dim
        self.is_fitted = False

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        n_lf, n_hf = len(X_lf), len(X_hf)
        X_lf_fid = np.hstack([X_lf, np.zeros((n_lf, 1))])
        X_hf_fid = np.hstack([X_hf, np.ones((n_hf, 1))])
        X_all = np.vstack([X_lf_fid, X_hf_fid])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])

        X_t = torch.tensor(X_all, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_all, dtype=torch.float32).to(self.device)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = _DKLMultiFidelityGP(
            X_t, y_t, self.likelihood, self.input_dim, self.bottleneck_dim
        ).to(self.device)
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': self.model.feature_extractor.parameters(), 'lr': 1e-3},
            {'params': self.model.covar_module.parameters(), 'lr': 1e-2},
            {'params': self.model.mean_module.parameters(), 'lr': 1e-2},
            {'params': self.model.fidelity_kernel.parameters(), 'lr': 1e-2},
            {'params': self.likelihood.parameters(), 'lr': 1e-2},
        ])
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            output = self.model(X_t)
            loss = -mll(output, y_t)
            loss.backward()
            optimizer.step()

        self.is_fitted = True

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_fid = np.hstack([X, np.ones((len(X), 1))])  # HF fidelity=1
        X_t = torch.tensor(X_fid, dtype=torch.float32).to(self.device)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            posterior = self.likelihood(self.model(X_t))
            mean = posterior.mean.cpu().numpy().flatten()
            std = posterior.variance.sqrt().cpu().numpy().flatten()
        return mean, np.maximum(std, 1e-6)

    def predict_lf(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_fid = np.hstack([X, np.zeros((len(X), 1))])  # LF fidelity=0
        X_t = torch.tensor(X_fid, dtype=torch.float32).to(self.device)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            posterior = self.likelihood(self.model(X_t))
            mean = posterior.mean.cpu().numpy().flatten()
            std = posterior.variance.sqrt().cpu().numpy().flatten()
        return mean, np.maximum(std, 1e-6)


# =============================================================================
# Baseline 4: Successive Halving (surrogate-free)
# =============================================================================

def run_successive_halving(benchmark, budget, seed=42):
    """
    Surrogate-free successive halving using the SAME fidelity schedule as run_bo().

    Fidelity schedule: lf_per_hf = max(1, int(1.0 / rho)) LF evals per 1 HF eval.
    Initial sampling: same 10% budget with FPS/LHS as run_bo().
    LF turn: random from unevaluated pool.
    HF turn: pick candidate with best (lowest) LF score among HF-unevaluated.

    Returns dict matching run_bo() output format.
    """
    from benchmark_parallel import (
        furthest_point_sampling, latin_hypercube_sampling,
        find_nearest_candidates,
    )

    np.random.seed(seed)
    rho = benchmark.cost_ratio
    n_candidates = benchmark.n_candidates

    # --- Initial sampling (same as run_bo) ---
    init_budget = 0.1 * budget
    n_init_hf = max(2, int(init_budget * 0.5 / 1.0))
    n_init_lf = max(2, int(init_budget * 0.5 / rho))
    n_init_total = n_init_lf + n_init_hf

    is_synthetic = hasattr(benchmark, 'dim')
    if is_synthetic:
        bounds = np.array([[0, 1]] * benchmark.dim)
        lhs_samples = latin_hypercube_sampling(bounds, n_init_total, seed)
        X_min, X_max = benchmark.X.min(axis=0), benchmark.X.max(axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1
        lhs_samples_scaled = X_min + lhs_samples * X_range
        init_indices = find_nearest_candidates(benchmark.X, lhs_samples_scaled)
        init_indices = list(dict.fromkeys(init_indices))
        if len(init_indices) < n_init_total:
            remaining = n_init_total - len(init_indices)
            available = set(range(n_candidates)) - set(init_indices)
            if available:
                extra = furthest_point_sampling(
                    benchmark.X[list(available)], remaining, seed + 1000
                )
                extra_indices = [list(available)[i] for i in extra]
                init_indices.extend(extra_indices)
    else:
        init_indices = furthest_point_sampling(benchmark.X, n_init_total, seed).tolist()

    lf_evaluated = {}
    hf_evaluated = {}
    all_sampled = set()

    for idx in init_indices[:n_init_lf]:
        lf_evaluated[idx] = benchmark.evaluate_lf(np.array([idx]))[0]
        all_sampled.add(idx)
    for idx in init_indices[n_init_lf:n_init_lf + n_init_hf]:
        hf_evaluated[idx] = benchmark.evaluate_hf(np.array([idx]))[0]
        all_sampled.add(idx)

    current_budget = n_init_lf * rho + n_init_hf * 1.0
    best_hf = min(hf_evaluated.values()) if hf_evaluated else np.inf

    regrets = [max(0, best_hf - benchmark.f_star)]
    budgets_list = [current_budget]
    step_records = []

    # --- Same fidelity schedule as run_bo ---
    lf_per_hf = max(1, int(1.0 / rho))
    lf_counter = 0
    iteration = 0
    max_iter = 500

    while current_budget < budget and iteration < max_iter:
        iteration += 1
        remaining = budget - current_budget

        if remaining >= 1.0:
            if remaining >= rho and lf_counter < lf_per_hf:
                eval_hf = False
                cost = rho
                lf_counter += 1
            else:
                eval_hf = True
                cost = 1.0
                lf_counter = 0
        elif remaining >= rho:
            eval_hf = False
            cost = rho
        else:
            break

        if eval_hf:
            # HF turn: pick best LF-scored candidate not yet HF-evaluated
            candidates = [
                (idx, score) for idx, score in lf_evaluated.items()
                if idx not in hf_evaluated
            ]
            if candidates:
                candidates.sort(key=lambda x: x[1])  # minimization
                idx = candidates[0][0]
            else:
                available = set(range(n_candidates)) - all_sampled
                if not available:
                    break
                idx = np.random.choice(list(available))
            hf_score = benchmark.evaluate_hf(np.array([idx]))[0]
            hf_evaluated[idx] = hf_score
            all_sampled.add(idx)
            best_hf = min(best_hf, hf_score)
            fidelity = 1
            observed = hf_score
        else:
            # LF turn: random from unevaluated
            unevaluated = set(range(n_candidates)) - all_sampled
            if not unevaluated:
                break
            idx = np.random.choice(list(unevaluated))
            lf_score = benchmark.evaluate_lf(np.array([idx]))[0]
            lf_evaluated[idx] = lf_score
            all_sampled.add(idx)
            fidelity = 0
            observed = lf_score

        current_budget += cost
        regrets.append(max(0, best_hf - benchmark.f_star))
        budgets_list.append(current_budget)
        step_records.append({
            'step': iteration,
            'fidelity': fidelity,
            'candidate_idx': int(idx),
            'observed_value': observed,
            'best_hf_so_far': best_hf,
            'wall_time_sec': 0.0,
        })

    return {
        'regrets': regrets if regrets else [float('inf')],
        'budgets': budgets_list if budgets_list else [0],
        'final_regret': regrets[-1] if regrets else float('inf'),
        'n_hf': len(hf_evaluated),
        'n_lf': len(lf_evaluated),
        'best_y': best_hf if hf_evaluated else np.inf,
        'step_records': step_records,
    }


# =============================================================================
# Baseline 5: HF-Only Random Search (non-learning)
# =============================================================================

def run_hf_random_search(benchmark, budget, seed=42):
    """
    Pure random search using only HF evaluations (no LF, no surrogate).

    Provides the absolute performance floor — what you get without any
    multi-fidelity or surrogate modelling.

    Initial sampling: same strategy as run_bo (LHS for synthetic, FPS for chemistry).
    Remaining budget: random HF evaluations from unevaluated candidates.

    Returns dict matching run_bo() output format.
    """
    from benchmark_parallel import (
        furthest_point_sampling, latin_hypercube_sampling,
        find_nearest_candidates,
    )

    np.random.seed(seed)
    n_candidates = benchmark.n_candidates

    # Total HF evaluations possible
    n_hf_total = int(budget / 1.0)

    # Initial sampling: 10% of budget (same as run_bo)
    n_init = max(2, int(0.1 * budget))
    n_init = min(n_init, n_hf_total)

    is_synthetic = hasattr(benchmark, 'grid_size')
    if is_synthetic:
        bounds = np.array([[0, 1]] * benchmark.dim)
        lhs_samples = latin_hypercube_sampling(bounds, n_init, seed)
        X_min, X_max = benchmark.X.min(axis=0), benchmark.X.max(axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1
        lhs_samples_scaled = X_min + lhs_samples * X_range
        init_indices = find_nearest_candidates(benchmark.X, lhs_samples_scaled)
        init_indices = list(dict.fromkeys(init_indices))
        if len(init_indices) < n_init:
            remaining = n_init - len(init_indices)
            available = set(range(n_candidates)) - set(init_indices)
            if available:
                extra = furthest_point_sampling(
                    benchmark.X[list(available)], remaining, seed + 1000
                )
                extra_indices = [list(available)[i] for i in extra]
                init_indices.extend(extra_indices)
    else:
        init_indices = furthest_point_sampling(benchmark.X, n_init, seed).tolist()

    hf_evaluated = {}
    all_sampled = set()

    # Phase 1: Initial HF evaluations
    for idx in init_indices[:n_init]:
        hf_evaluated[idx] = benchmark.evaluate_hf(np.array([idx]))[0]
        all_sampled.add(idx)

    current_budget = len(hf_evaluated) * 1.0
    best_hf = min(hf_evaluated.values()) if hf_evaluated else np.inf

    regrets = [max(0, best_hf - benchmark.f_star)]
    budgets_list = [current_budget]
    step_records = []
    iteration = 0

    # Phase 2: Random HF evaluations
    while current_budget + 1.0 <= budget:
        iteration += 1
        available = set(range(n_candidates)) - all_sampled
        if not available:
            break

        idx = np.random.choice(list(available))
        hf_score = benchmark.evaluate_hf(np.array([idx]))[0]
        hf_evaluated[idx] = hf_score
        all_sampled.add(idx)

        current_budget += 1.0
        best_hf = min(best_hf, hf_score)
        regrets.append(max(0, best_hf - benchmark.f_star))
        budgets_list.append(current_budget)
        step_records.append({
            'step': iteration,
            'fidelity': 1,
            'candidate_idx': int(idx),
            'observed_value': hf_score,
            'best_hf_so_far': best_hf,
            'wall_time_sec': 0.0,
        })

    return {
        'regrets': regrets if regrets else [float('inf')],
        'budgets': budgets_list if budgets_list else [0],
        'final_regret': regrets[-1] if regrets else float('inf'),
        'n_hf': len(hf_evaluated),
        'n_lf': 0,
        'best_y': best_hf if hf_evaluated else np.inf,
        'step_records': step_records,
    }


# =============================================================================
# Baseline 6: LF-Screening (non-learning, surrogate-free MF)
# =============================================================================

def run_lf_screening(benchmark, budget, seed=42):
    """
    LF-Screening: evaluate as many candidates as possible at LF,
    then spend remaining budget on HF evaluations of top-ranked LF candidates.

    This is the strongest possible non-learning MF baseline — it uses the
    LF fidelity directly for ranking without any surrogate model.

    Budget allocation (with dynamic reallocation when pool < n_lf_raw):
      1. Reserve minimum HF: n_reserve_hf = max(5, int(0.1 * budget))
      2. Plan LF: n_lf_raw = int((budget - n_reserve_hf) / rho)
      3. Cap by pool: n_lf = min(n_candidates, n_lf_raw)
      4. Reallocate: remaining_budget = budget - n_lf * rho → n_hf = int(remaining / 1.0)

    Returns dict matching run_bo() output format.
    """
    from benchmark_parallel import (
        furthest_point_sampling, latin_hypercube_sampling,
        find_nearest_candidates,
    )

    np.random.seed(seed)
    rho = benchmark.cost_ratio
    n_candidates = benchmark.n_candidates

    # --- Budget allocation with dynamic reallocation ---
    n_reserve_hf = max(5, int(0.1 * budget))
    lf_budget = budget - n_reserve_hf * 1.0
    n_lf_raw = int(lf_budget / rho)

    # Cap by pool size
    n_lf = min(n_candidates, n_lf_raw)

    # Dynamic reallocation: excess budget goes to HF
    actual_lf_cost = n_lf * rho
    remaining_budget = budget - actual_lf_cost
    n_hf = int(remaining_budget / 1.0)
    n_hf = max(n_hf, n_reserve_hf)  # at least the reserved amount

    # --- Initial sampling (same as run_bo) ---
    init_budget = 0.1 * budget
    n_init_hf = max(2, int(init_budget * 0.5 / 1.0))
    n_init_lf = max(2, int(init_budget * 0.5 / rho))
    n_init_total = n_init_lf + n_init_hf

    is_synthetic = hasattr(benchmark, 'grid_size')
    if is_synthetic:
        bounds = np.array([[0, 1]] * benchmark.dim)
        lhs_samples = latin_hypercube_sampling(bounds, n_init_total, seed)
        X_min, X_max = benchmark.X.min(axis=0), benchmark.X.max(axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1
        lhs_samples_scaled = X_min + lhs_samples * X_range
        init_indices = find_nearest_candidates(benchmark.X, lhs_samples_scaled)
        init_indices = list(dict.fromkeys(init_indices))
        if len(init_indices) < n_init_total:
            remaining = n_init_total - len(init_indices)
            available = set(range(n_candidates)) - set(init_indices)
            if available:
                extra = furthest_point_sampling(
                    benchmark.X[list(available)], remaining, seed + 1000
                )
                extra_indices = [list(available)[i] for i in extra]
                init_indices.extend(extra_indices)
    else:
        init_indices = furthest_point_sampling(benchmark.X, n_init_total, seed).tolist()

    lf_evaluated = {}
    hf_evaluated = {}
    all_sampled = set()

    # Initial LF evaluations
    for idx in init_indices[:n_init_lf]:
        lf_evaluated[idx] = benchmark.evaluate_lf(np.array([idx]))[0]
        all_sampled.add(idx)

    # Initial HF evaluations
    for idx in init_indices[n_init_lf:n_init_lf + n_init_hf]:
        hf_evaluated[idx] = benchmark.evaluate_hf(np.array([idx]))[0]
        all_sampled.add(idx)
        # Also get LF for these (free info for ranking, doesn't cost budget)
        if idx not in lf_evaluated:
            lf_evaluated[idx] = benchmark.evaluate_lf(np.array([idx]))[0]

    current_budget = n_init_lf * rho + n_init_hf * 1.0
    best_hf = min(hf_evaluated.values()) if hf_evaluated else np.inf

    regrets = [max(0, best_hf - benchmark.f_star)]
    budgets_list = [current_budget]
    step_records = []
    iteration = 0

    # --- Phase 1: LF Sweep ---
    # Evaluate remaining LF candidates (up to n_lf total, minus already evaluated)
    n_lf_remaining = n_lf - len(lf_evaluated)
    unevaluated_lf = list(set(range(n_candidates)) - set(lf_evaluated.keys()))
    np.random.shuffle(unevaluated_lf)
    n_lf_remaining = min(n_lf_remaining, len(unevaluated_lf))

    for i in range(n_lf_remaining):
        if current_budget + rho > budget:
            break
        idx = unevaluated_lf[i]
        iteration += 1

        lf_score = benchmark.evaluate_lf(np.array([idx]))[0]
        lf_evaluated[idx] = lf_score
        all_sampled.add(idx)

        current_budget += rho
        # Regret unchanged during LF phase (no new HF info)
        regrets.append(max(0, best_hf - benchmark.f_star))
        budgets_list.append(current_budget)
        step_records.append({
            'step': iteration,
            'fidelity': 0,
            'candidate_idx': int(idx),
            'observed_value': lf_score,
            'best_hf_so_far': best_hf,
            'wall_time_sec': 0.0,
        })

    # --- Phase 2: HF Top-k ---
    # Sort all LF-evaluated candidates by LF score (ascending = best first for minimization)
    lf_ranked = sorted(lf_evaluated.items(), key=lambda x: x[1])

    # Select top candidates not yet HF-evaluated
    hf_candidates = [idx for idx, _ in lf_ranked if idx not in hf_evaluated]

    for idx in hf_candidates:
        if current_budget + 1.0 > budget:
            break
        iteration += 1

        hf_score = benchmark.evaluate_hf(np.array([idx]))[0]
        hf_evaluated[idx] = hf_score

        current_budget += 1.0
        best_hf = min(best_hf, hf_score)
        regrets.append(max(0, best_hf - benchmark.f_star))
        budgets_list.append(current_budget)
        step_records.append({
            'step': iteration,
            'fidelity': 1,
            'candidate_idx': int(idx),
            'observed_value': hf_score,
            'best_hf_so_far': best_hf,
            'wall_time_sec': 0.0,
        })

    return {
        'regrets': regrets if regrets else [float('inf')],
        'budgets': budgets_list if budgets_list else [0],
        'final_regret': regrets[-1] if regrets else float('inf'),
        'n_hf': len(hf_evaluated),
        'n_lf': len(lf_evaluated),
        'best_y': best_hf if hf_evaluated else np.inf,
        'step_records': step_records,
    }
