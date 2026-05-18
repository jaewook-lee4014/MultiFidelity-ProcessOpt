#!/usr/bin/env python
"""
MFGP Variants Benchmark - Remaining 4 Benchmarks on 4 GPUs

Run the remaining 4 benchmarks (Park-Unfav, COFs, FreeSolv, Polarizability)
with each benchmark assigned to a specific GPU, 2 workers per GPU.

GPU Assignment:
- GPU 0: Park-Unfav (6 models x 20 seeds)
- GPU 1: COFs (6 models x 20 seeds)
- GPU 2: FreeSolv (6 models x 20 seeds)
- GPU 3: Polarizability (6 models x 20 seeds)

Each GPU runs 2 workers in parallel (so 6 models run as 3 batches of 2).

Usage:
    python benchmark_mfgp_remaining_4gpu.py --n-seeds 20 --output-dir mfgp_variants_20260204_210020
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube
from scipy.spatial.distance import cdist

# RDKit for molecular descriptors
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import argparse
import multiprocessing as mp
from multiprocessing import Pool, Process, Queue
import time
import os
import warnings
warnings.filterwarnings('ignore')

# BoTorch
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

# Local imports
from synthetic_functions import (
    branin_hf, branin_lf, park_hf, park_lf,
    SCENARIOS, FUNCTIONS
)


# =============================================================================
# MFGP VARIANTS (same as benchmark_mfgp_variants.py)
# =============================================================================

class BaseMFGP:
    """Base class for all MFGP variants"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, device=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.is_fitted = False

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        n_lf, n_hf = len(X_lf), len(X_hf)
        X_lf_fid = np.hstack([X_lf, np.zeros((n_lf, 1))])
        X_hf_fid = np.hstack([X_hf, np.ones((n_hf, 1))])
        X_all = np.vstack([X_lf_fid, X_hf_fid])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        train_X = torch.tensor(X_all, dtype=torch.float64).to(self.device)
        train_Y = torch.tensor(y_all, dtype=torch.float64).unsqueeze(-1).to(self.device)
        self.model = SingleTaskMultiFidelityGP(
            train_X, train_Y,
            data_fidelities=[self.input_dim],
            outcome_transform=Standardize(m=1)
        ).to(self.device)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self.is_fitted = True

    def predict(self, X, fidelity=1.0) -> Tuple[np.ndarray, np.ndarray]:
        X_fid = np.hstack([X, np.full((len(X), 1), fidelity)])
        X_tensor = torch.tensor(X_fid, dtype=torch.float64).to(self.device)
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X_tensor)
            mean = posterior.mean.cpu().numpy().flatten()
            std = posterior.variance.sqrt().cpu().numpy().flatten()
        return mean, np.maximum(std, 1e-6)

    def predict_lf(self, X) -> Tuple[np.ndarray, np.ndarray]:
        return self.predict(X, fidelity=0.0)


class MFGP_EI(BaseMFGP):
    def __init__(self, input_dim: int, hidden_dim: int = 64, device=None, xi: float = 0.01):
        super().__init__(input_dim, hidden_dim, device)
        self.xi = xi
        self.name = "MFGP-EI"

    def acquisition_ei(self, mean, std, y_best) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            imp = y_best - mean - self.xi
            Z = imp / std
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            ei[std < 1e-10] = 0.0
        return ei


class MFGP_UCB(BaseMFGP):
    def __init__(self, input_dim: int, hidden_dim: int = 64, device=None, beta: float = 2.0):
        super().__init__(input_dim, hidden_dim, device)
        self.beta = beta
        self.name = "MFGP-UCB"

    def acquisition_ucb(self, mean, std) -> np.ndarray:
        lcb = mean - self.beta * std
        return -lcb


class MFGP_PI(BaseMFGP):
    def __init__(self, input_dim: int, hidden_dim: int = 64, device=None, xi: float = 0.01):
        super().__init__(input_dim, hidden_dim, device)
        self.xi = xi
        self.name = "MFGP-PI"

    def acquisition_pi(self, mean, std, y_best) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = (y_best - mean - self.xi) / std
            pi = norm.cdf(Z)
            pi[std < 1e-10] = 0.0
        return pi


class MFGP_EI_Cool(BaseMFGP):
    def __init__(self, input_dim: int, hidden_dim: int = 64, device=None,
                 xi_init: float = 0.1, xi_final: float = 0.001):
        super().__init__(input_dim, hidden_dim, device)
        self.xi_init = xi_init
        self.xi_final = xi_final
        self.name = "MFGP-EI-Cool"

    def get_xi(self, progress: float) -> float:
        return self.xi_init * (1 - progress) + self.xi_final * progress

    def acquisition_ei(self, mean, std, y_best, xi) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            imp = y_best - mean - xi
            Z = imp / std
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            ei[std < 1e-10] = 0.0
        return ei


class MFGP_TS(BaseMFGP):
    def __init__(self, input_dim: int, hidden_dim: int = 64, device=None, n_samples: int = 1):
        super().__init__(input_dim, hidden_dim, device)
        self.n_samples = n_samples
        self.name = "MFGP-TS"

    def sample_posterior(self, X) -> np.ndarray:
        mean, std = self.predict(X, fidelity=1.0)
        samples = mean + std * np.random.randn(len(mean))
        return samples


class MFGP_UCB_Decay(BaseMFGP):
    def __init__(self, input_dim: int, hidden_dim: int = 64, device=None,
                 beta_init: float = 2.0, decay_type: str = 'sqrt'):
        super().__init__(input_dim, hidden_dim, device)
        self.beta_init = beta_init
        self.decay_type = decay_type
        self.name = "MFGP-UCB-Decay"

    def get_beta(self, iteration: int, total_iterations: int) -> float:
        if self.decay_type == 'sqrt':
            return self.beta_init / np.sqrt(iteration + 1)
        elif self.decay_type == 'log':
            return self.beta_init * np.sqrt(np.log(iteration + 2))
        else:
            return self.beta_init

    def acquisition_ucb(self, mean, std, beta) -> np.ndarray:
        lcb = mean - beta * std
        return -lcb


# =============================================================================
# Benchmark Classes
# =============================================================================

class SyntheticBenchmark:
    def __init__(self, name, hf_func, lf_func, dim, alpha, cost_ratio, f_star, grid_size=50):
        self.name = name
        self.hf_func = hf_func
        self.lf_func = lf_func
        self.dim = dim
        self.alpha = alpha
        self.cost_ratio = cost_ratio
        self.f_star = f_star
        self.grid_size = grid_size
        self._create_grid()
        corr = np.corrcoef(self.y_hf, self.y_lf)[0, 1]
        self.r2 = corr ** 2

    def _create_grid(self):
        if self.dim == 2:
            axes = [np.linspace(0, 1, self.grid_size) for _ in range(2)]
            grids = np.meshgrid(*axes)
            self.X = np.column_stack([g.ravel() for g in grids])
        else:
            n_per_dim = int(np.ceil(self.grid_size ** 0.5))
            axes = [np.linspace(0, 1, n_per_dim) for _ in range(self.dim)]
            grids = np.meshgrid(*axes, indexing='ij')
            self.X = np.column_stack([g.ravel() for g in grids])
        self.n_candidates = len(self.X)
        self.y_hf = self.hf_func(self.X).flatten()
        self.y_lf = self.lf_func(self.X, self.alpha).flatten()

    def evaluate_hf(self, indices):
        return self.y_hf[indices.astype(int).flatten()]

    def evaluate_lf(self, indices):
        return self.y_lf[indices.astype(int).flatten()]


class ChemistryBenchmark:
    def __init__(self, name, csv_path, cost_ratio, use_smiles=False, minimize=True, negate=False, pca_dim=10):
        self.name = name
        self.cost_ratio = cost_ratio
        self.minimize = minimize
        self.negate = negate
        self.pca_dim = pca_dim
        df = pd.read_csv(csv_path)
        if use_smiles:
            smiles_col = [c for c in df.columns if 'smiles' in c.lower()]
            if smiles_col:
                self.X = self._smiles_to_rdkit_features(df[smiles_col[0]].values)
            else:
                self.X = self._smiles_to_rdkit_features(df.iloc[:, 0].values)
        else:
            feature_cols = [c for c in df.columns if c not in ['HF', 'LF']]
            self.X = df[feature_cols].values
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        self.y_hf = df['HF'].values.flatten()
        self.y_lf = df['LF'].values.flatten()
        if negate:
            self.y_hf = -self.y_hf
            self.y_lf = -self.y_lf
        self.f_star = self.y_hf.min()
        self.n_candidates = len(self.X)
        self.dim = self.X.shape[1]
        corr = np.corrcoef(self.y_hf, self.y_lf)[0, 1]
        self.r2 = corr ** 2

    def _smiles_to_rdkit_features(self, smiles_list):
        descriptor_names = [desc[0] for desc in Descriptors._descList]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

        features = []
        valid_indices = []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                desc = calc.CalcDescriptors(mol)
                features.append(desc)
                valid_indices.append(i)
            else:
                features.append([0.0] * len(descriptor_names))
                valid_indices.append(i)

        features = np.array(features, dtype=np.float64)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        pca = PCA(n_components=self.pca_dim)
        features_pca = pca.fit_transform(features_scaled)

        return features_pca

    def evaluate_hf(self, indices):
        return self.y_hf[indices.astype(int).flatten()]

    def evaluate_lf(self, indices):
        return self.y_lf[indices.astype(int).flatten()]


# =============================================================================
# Sampling Methods
# =============================================================================

def furthest_point_sampling(X: np.ndarray, n_samples: int, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    n_candidates = len(X)
    n_samples = min(n_samples, n_candidates)
    selected = [np.random.randint(n_candidates)]
    for _ in range(n_samples - 1):
        selected_X = X[selected]
        distances = cdist(X, selected_X, metric='euclidean')
        min_distances = distances.min(axis=1)
        min_distances[selected] = -np.inf
        next_idx = np.argmax(min_distances)
        selected.append(next_idx)
    return np.array(selected)


def latin_hypercube_sampling(bounds: np.ndarray, n_samples: int, seed: int = 42) -> np.ndarray:
    n_dims = len(bounds)
    sampler = LatinHypercube(d=n_dims, seed=seed)
    samples = sampler.random(n=n_samples)
    for i in range(n_dims):
        samples[:, i] = bounds[i, 0] + samples[:, i] * (bounds[i, 1] - bounds[i, 0])
    return samples


def find_nearest_candidates(X_candidates: np.ndarray, X_samples: np.ndarray) -> np.ndarray:
    distances = cdist(X_samples, X_candidates, metric='euclidean')
    return np.argmin(distances, axis=1)


# =============================================================================
# Acquisition Functions
# =============================================================================

def expected_improvement(mean, std, y_best, xi=0.01):
    with np.errstate(divide='ignore', invalid='ignore'):
        imp = y_best - mean - xi
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std < 1e-10] = 0.0
    return ei


def probability_of_improvement(mean, std, y_best, xi=0.01):
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = (y_best - mean - xi) / std
        pi = norm.cdf(Z)
        pi[std < 1e-10] = 0.0
    return pi


def upper_confidence_bound(mean, std, beta=2.0):
    lcb = mean - beta * std
    return -lcb


# =============================================================================
# BO Loop
# =============================================================================

def run_bo_mfgp(benchmark, model_class, budget, seed=42, device=None, sampling_method='fps',
                iteration_count=None):
    np.random.seed(seed)
    torch.manual_seed(seed)

    rho = benchmark.cost_ratio
    n_candidates = benchmark.n_candidates

    init_budget = 0.1 * budget
    n_init_hf = max(2, int(init_budget * 0.5 / 1.0))
    n_init_lf = max(2, int(init_budget * 0.5 / rho))
    n_init_total = n_init_lf + n_init_hf

    if sampling_method == 'lhs':
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
                    benchmark.X[list(available)],
                    remaining,
                    seed + 1000
                )
                extra_indices = [list(available)[i] for i in extra]
                init_indices.extend(extra_indices)
    else:
        init_indices = furthest_point_sampling(benchmark.X, n_init_total, seed).tolist()

    lf_indices = set(init_indices[:n_init_lf])
    hf_indices = set(init_indices[n_init_lf:n_init_lf + n_init_hf])

    X_lf = benchmark.X[list(lf_indices)]
    y_lf = benchmark.evaluate_lf(np.array(list(lf_indices)))
    X_hf = benchmark.X[list(hf_indices)]
    y_hf = benchmark.evaluate_hf(np.array(list(hf_indices)))

    current_budget = n_init_lf * rho + n_init_hf * 1.0
    lf_per_hf = max(1, int(1.0 / rho))
    lf_counter = 0

    regrets = [max(0, y_hf.min() - benchmark.f_star)]
    budgets = [current_budget]

    iteration = 0
    max_iter = 500
    estimated_total_iter = int((budget - current_budget) / (rho + 1.0 / (lf_per_hf + 1)))

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

        try:
            model = model_class(benchmark.X.shape[1], device=device)
            model.fit(X_lf, y_lf, X_hf, y_hf)

            sampled = lf_indices | hf_indices
            y_best = y_hf.min()
            mean_hf, std_hf = model.predict(benchmark.X, fidelity=1.0)

            if eval_hf:
                mean_masked = mean_hf.copy()
                mean_masked[list(sampled)] = np.inf
                next_idx = np.argmin(mean_masked)
            else:
                if isinstance(model, MFGP_EI):
                    acq = expected_improvement(mean_hf, std_hf, y_best, xi=model.xi)
                elif isinstance(model, MFGP_UCB):
                    acq = upper_confidence_bound(mean_hf, std_hf, beta=model.beta)
                elif isinstance(model, MFGP_PI):
                    acq = probability_of_improvement(mean_hf, std_hf, y_best, xi=model.xi)
                elif isinstance(model, MFGP_EI_Cool):
                    progress = iteration / max(estimated_total_iter, 1)
                    xi = model.get_xi(progress)
                    acq = expected_improvement(mean_hf, std_hf, y_best, xi=xi)
                elif isinstance(model, MFGP_TS):
                    samples = model.sample_posterior(benchmark.X)
                    acq = -samples
                elif isinstance(model, MFGP_UCB_Decay):
                    beta = model.get_beta(iteration, estimated_total_iter)
                    acq = upper_confidence_bound(mean_hf, std_hf, beta=beta)
                else:
                    acq = expected_improvement(mean_hf, std_hf, y_best, xi=0.01)

                acq[list(sampled)] = -np.inf
                next_idx = np.argmax(acq)

            if eval_hf:
                hf_indices.add(next_idx)
                X_hf = benchmark.X[list(hf_indices)]
                y_hf = benchmark.evaluate_hf(np.array(list(hf_indices)))
            else:
                lf_indices.add(next_idx)
                X_lf = benchmark.X[list(lf_indices)]
                y_lf = benchmark.evaluate_lf(np.array(list(lf_indices)))

            current_budget += cost

        except Exception as e:
            available = set(range(n_candidates)) - (lf_indices | hf_indices)
            if available:
                next_idx = np.random.choice(list(available))
                if eval_hf:
                    hf_indices.add(next_idx)
                    X_hf = benchmark.X[list(hf_indices)]
                    y_hf = benchmark.evaluate_hf(np.array(list(hf_indices)))
                else:
                    lf_indices.add(next_idx)
                    X_lf = benchmark.X[list(lf_indices)]
                    y_lf = benchmark.evaluate_lf(np.array(list(lf_indices)))
            current_budget += cost

        regrets.append(max(0, y_hf.min() - benchmark.f_star))
        budgets.append(current_budget)

    return {
        'regrets': regrets,
        'budgets': budgets,
        'final_regret': regrets[-1],
        'n_hf': len(hf_indices),
        'n_lf': len(lf_indices),
        'best_y': y_hf.min()
    }


# =============================================================================
# GPU-specific Worker
# =============================================================================

def run_single_model_on_gpu(args):
    """Run a single model on a specific GPU"""
    bench_name, bench_config, model_name, model_class, budget, seeds, output_dir, gpu_id, worker_id = args

    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    if torch.cuda.is_available():
        torch.cuda.init()
        device = torch.device('cuda:0')  # Always 0 since we set CUDA_VISIBLE_DEVICES
        _ = torch.zeros(1).to(device)
    else:
        device = torch.device('cpu')

    print(f"[GPU {gpu_id} Worker {worker_id}] {bench_name} + {model_name}: Using {device}", flush=True)

    data_dir = Path(__file__).parent / 'data'

    # Create benchmark
    if bench_config['type'] == 'synthetic':
        benchmark = SyntheticBenchmark(
            bench_name,
            bench_config['hf_func'],
            bench_config['lf_func'],
            bench_config['dim'],
            bench_config['alpha'],
            bench_config['cost_ratio'],
            bench_config['f_star'],
            bench_config['grid_size']
        )
    else:
        benchmark = ChemistryBenchmark(
            bench_name,
            bench_config['csv_path'],
            bench_config['cost_ratio'],
            bench_config['use_smiles'],
            bench_config['minimize'],
            bench_config.get('negate', False)
        )

    sampling_method = 'lhs' if bench_config['type'] == 'synthetic' else 'fps'

    results_summary = []
    results_trajectory = []
    start_time = time.time()

    for i, seed in enumerate(seeds):
        seed_start = time.time()
        try:
            result = run_bo_mfgp(benchmark, model_class, budget, seed, device, sampling_method)
            seed_elapsed = time.time() - seed_start

            results_summary.append({
                'benchmark': bench_name,
                'model': model_name,
                'seed': seed,
                'final_regret': result['final_regret'],
                'n_hf': result['n_hf'],
                'n_lf': result['n_lf'],
                'best_y': result['best_y'],
                'elapsed_sec': round(seed_elapsed, 3),
            })

            for b, r in zip(result['budgets'], result['regrets']):
                results_trajectory.append({
                    'benchmark': bench_name,
                    'model': model_name,
                    'seed': seed,
                    'budget': round(b, 2),
                    'regret': r,
                })

            print(f"  [GPU {gpu_id}] {bench_name} + {model_name}: seed {seed} done "
                  f"(regret={result['final_regret']:.4f}, {seed_elapsed:.1f}s)", flush=True)

        except Exception as e:
            seed_elapsed = time.time() - seed_start
            results_summary.append({
                'benchmark': bench_name,
                'model': model_name,
                'seed': seed,
                'final_regret': np.nan,
                'n_hf': 0,
                'n_lf': 0,
                'best_y': np.nan,
                'elapsed_sec': round(seed_elapsed, 3),
            })
            print(f"  [GPU {gpu_id}] {bench_name} + {model_name}: seed {seed} ERROR: {e}", flush=True)

    elapsed = time.time() - start_time

    # Save per-model results
    output_dir = Path(output_dir)
    df_summary = pd.DataFrame(results_summary)
    summary_file = output_dir / f'summary_{bench_name}_{model_name.replace(" ", "_").replace("-", "_")}.csv'
    df_summary.to_csv(summary_file, index=False)

    df_trajectory = pd.DataFrame(results_trajectory)
    trajectory_file = output_dir / f'trajectory_{bench_name}_{model_name.replace(" ", "_").replace("-", "_")}.csv'
    df_trajectory.to_csv(trajectory_file, index=False)

    print(f"[GPU {gpu_id}] {bench_name} + {model_name}: COMPLETED in {elapsed:.1f}s", flush=True)

    return {
        'bench_name': bench_name,
        'model_name': model_name,
        'gpu_id': gpu_id,
        'elapsed': elapsed,
        'results_summary': results_summary,
        'results_trajectory': results_trajectory
    }


def run_benchmark_on_gpu(gpu_id, bench_name, bench_config, budget, seeds, output_dir, n_workers_per_gpu=2):
    """Run all 6 models for a benchmark on a specific GPU with multiple workers"""

    models = {
        'MFGP-EI': MFGP_EI,
        'MFGP-UCB': MFGP_UCB,
        'MFGP-PI': MFGP_PI,
        'MFGP-EI-Cool': MFGP_EI_Cool,
        'MFGP-TS': MFGP_TS,
        'MFGP-UCB-Decay': MFGP_UCB_Decay,
    }

    print(f"\n[GPU {gpu_id}] Starting {bench_name} with {n_workers_per_gpu} workers", flush=True)
    print(f"[GPU {gpu_id}] Models: {list(models.keys())}", flush=True)

    # Create tasks for this GPU
    tasks = []
    for worker_id, (model_name, model_class) in enumerate(models.items()):
        tasks.append((
            bench_name, bench_config, model_name, model_class,
            budget, seeds, output_dir, gpu_id, worker_id
        ))

    # Run with pool of workers on this GPU
    # Note: Each worker will set CUDA_VISIBLE_DEVICES to this GPU
    results = []

    # Use spawn context for CUDA compatibility
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_workers_per_gpu) as pool:
        for result in pool.imap_unordered(run_single_model_on_gpu, tasks):
            results.append(result)
            print(f"[GPU {gpu_id}] Completed: {result['model_name']} ({result['elapsed']:.1f}s)", flush=True)

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='MFGP Variants - Remaining 4 Benchmarks on 4 GPUs')
    parser.add_argument('--n-seeds', type=int, default=20, help='Number of seeds')
    parser.add_argument('--base-seed', type=int, default=42, help='Base seed')
    parser.add_argument('--n-workers-per-gpu', type=int, default=2, help='Workers per GPU')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (existing)')
    args = parser.parse_args()

    # Use existing output directory or create new one
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f'mfgp_variants_{timestamp}')
    output_dir.mkdir(exist_ok=True)

    data_dir = Path(__file__).parent / 'data'

    print("=" * 80, flush=True)
    print("MFGP Variants - Remaining 4 Benchmarks on 4 GPUs", flush=True)
    print("=" * 80, flush=True)
    print(f"Workers per GPU: {args.n_workers_per_gpu}", flush=True)
    print(f"Seeds: {args.n_seeds}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(flush=True)
    print(f"CUDA Available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}", flush=True)
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)

    # Remaining 4 benchmarks
    bench_configs = {
        'Park-Unfav': {
            'type': 'synthetic', 'hf_func': park_hf, 'lf_func': park_lf,
            'dim': 4, 'alpha': 0.0, 'cost_ratio': 0.5, 'f_star': 0.0, 'grid_size': 10
        },
        'COFs': {
            'type': 'chemistry', 'csv_path': data_dir / 'cofs.csv',
            'cost_ratio': 0.065, 'use_smiles': False, 'minimize': True, 'negate': True
        },
        'FreeSolv': {
            'type': 'chemistry', 'csv_path': data_dir / 'freesolv.csv',
            'cost_ratio': 0.1, 'use_smiles': True, 'minimize': True, 'negate': False
        },
        'Polarizability': {
            'type': 'chemistry', 'csv_path': data_dir / 'polarizability.csv',
            'cost_ratio': 0.167, 'use_smiles': True, 'minimize': True, 'negate': True
        },
    }

    budgets = {
        'Park-Unfav': 50,
        'COFs': 30,
        'FreeSolv': 50,
        'Polarizability': 30,
    }

    # GPU assignments
    gpu_assignments = {
        0: 'Park-Unfav',
        1: 'COFs',
        2: 'FreeSolv',
        3: 'Polarizability',
    }

    seeds = [args.base_seed + i for i in range(args.n_seeds)]

    print("\nGPU Assignments:", flush=True)
    for gpu_id, bench_name in gpu_assignments.items():
        print(f"  GPU {gpu_id}: {bench_name}", flush=True)
    print("=" * 80, flush=True)

    start_time = time.time()

    # Launch processes for each GPU
    processes = []
    result_queues = []

    for gpu_id, bench_name in gpu_assignments.items():
        bench_config = bench_configs[bench_name]
        budget = budgets[bench_name]

        # Run each GPU's benchmark in a separate process
        p = Process(
            target=run_benchmark_on_gpu,
            args=(gpu_id, bench_name, bench_config, budget, seeds, str(output_dir), args.n_workers_per_gpu)
        )
        processes.append((gpu_id, bench_name, p))
        p.start()
        print(f"Started process for GPU {gpu_id}: {bench_name}", flush=True)

    # Wait for all processes to complete
    for gpu_id, bench_name, p in processes:
        p.join()
        print(f"GPU {gpu_id} ({bench_name}) process completed", flush=True)

    total_time = time.time() - start_time

    print("\n" + "=" * 80, flush=True)
    print(f"All benchmarks completed in {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)
    print("=" * 80, flush=True)

    # Aggregate all results
    all_summary_files = list(output_dir.glob('summary_*.csv'))
    if all_summary_files:
        all_summaries = []
        for f in all_summary_files:
            df = pd.read_csv(f)
            all_summaries.append(df)
        df_all = pd.concat(all_summaries, ignore_index=True)
        df_all.to_csv(output_dir / 'results_summary.csv', index=False)
        print(f"Aggregated summary: {len(df_all)} rows", flush=True)

    all_trajectory_files = list(output_dir.glob('trajectory_*.csv'))
    if all_trajectory_files:
        all_trajectories = []
        for f in all_trajectory_files:
            df = pd.read_csv(f)
            all_trajectories.append(df)
        df_all_traj = pd.concat(all_trajectories, ignore_index=True)
        df_all_traj.to_csv(output_dir / 'results_trajectory.csv', index=False)
        print(f"Aggregated trajectory: {len(df_all_traj)} rows", flush=True)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
