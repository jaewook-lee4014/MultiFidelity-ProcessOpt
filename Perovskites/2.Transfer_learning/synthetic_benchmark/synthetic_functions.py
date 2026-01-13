"""
Synthetic Test Functions for Bayesian Optimization Benchmarks

Functions:
- Branin-2D: Classic 2D benchmark (global minimum f* ≈ 0.397887)
- Park-4D: 4D benchmark from MFBO literature

Reference: "Best Practices for Multi-Fidelity Bayesian Optimization" (Nature Comp Science)
"""

import numpy as np
from typing import Callable, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# =============================================================================
# BRANIN FUNCTION (2D)
# =============================================================================

def branin_hf(x: np.ndarray) -> np.ndarray:
    """
    Branin function (High-Fidelity, alpha=1)

    Input: x in [0,1]^2 (normalized domain)
    Original domain: x1 ∈ [-5, 10], x2 ∈ [0, 15]

    Global minima: f* ≈ 0.397887 at
        (x1, x2) = (-π, 12.275), (π, 2.275), (9.42478, 2.475)

    In normalized [0,1]^2:
        (0.1239, 0.8183), (0.5428, 0.1517), (0.9617, 0.165)
    """
    # Rescale from [0,1]^2 to original domain
    x1 = x[:, 0] * 15 - 5   # x1 ∈ [-5, 10]
    x2 = x[:, 1] * 15       # x2 ∈ [0, 15]

    # Branin parameters
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    result = a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s

    return result.reshape(-1, 1)


def branin_lf(x: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Branin function (Low-Fidelity with bias parameter alpha)

    alpha ∈ [0, 1]:
        - alpha = 1: identical to HF (most informative)
        - alpha → 0: increasingly biased, optimum shifts

    The modification changes:
        1. Quadratic coefficient b (surface curvature)
        2. Linear coefficient c (optimum location)
        3. Constant term r (vertical shift)
    """
    x1 = x[:, 0] * 15 - 5
    x2 = x[:, 1] * 15

    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    # Alpha-based modification (from MFBO best practices paper)
    # These modifications shift the optimum location as alpha decreases
    b_mod = b * (1 + 0.2 * (1 - alpha))      # Increase quadratic term
    c_mod = c * (0.5 + 0.5 * alpha)          # Reduce linear term
    r_mod = r * (1 - 0.3 * (1 - alpha))      # Shift constant

    result = a * (x2 - b_mod * x1**2 + c_mod * x1 - r_mod)**2 + s * (1 - t) * np.cos(x1) + s

    return result.reshape(-1, 1)


# =============================================================================
# PARK FUNCTION (4D)
# =============================================================================

def park_hf(x: np.ndarray) -> np.ndarray:
    """
    Park function (High-Fidelity, alpha=1)

    Input: x in [0,1]^4 (normalized domain)

    A challenging 4D test function commonly used in MFBO literature.
    Contains nonlinear interactions and discontinuities.
    """
    # Add small epsilon to avoid division by zero
    eps = 1e-8

    x1 = np.maximum(x[:, 0], eps)
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = np.maximum(x[:, 3], eps)

    # Park function formula
    term1 = x1 / 2 * (np.sqrt(1 + (x2 + x3**2) * x4 / (x1**2 + eps)) - 1)
    term2 = (x1 + 3 * x4) * np.exp(1 + np.sin(x3))

    # Handle numerical issues
    term1 = np.nan_to_num(term1, nan=0.0, posinf=10.0, neginf=-10.0)
    term2 = np.nan_to_num(term2, nan=0.0, posinf=100.0, neginf=-100.0)

    result = term1 + term2

    return result.reshape(-1, 1)


def park_lf(x: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Park function (Low-Fidelity with bias parameter alpha)

    alpha ∈ [0, 1]:
        - alpha = 1: identical to HF
        - alpha → 0: increasingly biased

    Modifications:
        1. Coefficient scaling on both terms
        2. Altered interaction structure
    """
    eps = 1e-8

    x1 = np.maximum(x[:, 0], eps)
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = np.maximum(x[:, 3], eps)

    # Alpha-based modification
    coef1 = 1 + 0.5 * (1 - alpha)  # Scale first term
    coef2 = 0.3 + 0.7 * alpha      # Scale second term

    # Modified Park function
    term1 = coef1 * x1 / 2 * (np.sqrt(1 + (x2 + x3**2) * x4 / (x1**2 + eps)) - 1)
    term2 = coef2 * (x1 + 3 * x4) * np.exp(1 + np.sin(x3))

    term1 = np.nan_to_num(term1, nan=0.0, posinf=10.0, neginf=-10.0)
    term2 = np.nan_to_num(term2, nan=0.0, posinf=100.0, neginf=-100.0)

    result = term1 + term2

    return result.reshape(-1, 1)


# =============================================================================
# ADDITIONAL TEST FUNCTIONS
# =============================================================================

def hartmann6_hf(x: np.ndarray) -> np.ndarray:
    """
    Hartmann-6D function (High-Fidelity)

    Input: x in [0,1]^6
    Global minimum: f* ≈ -3.32237 at (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])

    result = np.zeros(x.shape[0])
    for i in range(4):
        inner = np.sum(A[i] * (x - P[i])**2, axis=1)
        result -= alpha[i] * np.exp(-inner)

    return result.reshape(-1, 1)


def ackley_hf(x: np.ndarray) -> np.ndarray:
    """
    Ackley function (variable dimension)

    Input: x in [-32.768, 32.768]^d, normalized to [0,1]^d
    Global minimum: f* = 0 at x* = (0, ..., 0) -> (0.5, ..., 0.5) normalized
    """
    # Rescale to [-32.768, 32.768]
    x_scaled = x * 65.536 - 32.768

    d = x.shape[1]
    a = 20
    b = 0.2
    c = 2 * np.pi

    sum1 = np.sum(x_scaled**2, axis=1)
    sum2 = np.sum(np.cos(c * x_scaled), axis=1)

    result = -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e

    return result.reshape(-1, 1)


# =============================================================================
# INFORMATIVENESS METRICS
# =============================================================================

def compute_r2_informativeness(f_hf: Callable, f_lf: Callable,
                                dim: int, alpha: float,
                                n_samples: int = 100, seed: int = 42) -> float:
    """
    Compute R² between HF and LF functions (informativeness measure)

    This measures how well LF linearly predicts HF:
        - R² = 1: LF perfectly explains HF (most informative)
        - R² = 0: LF has no linear correlation with HF

    Procedure:
        1. Sample 100 points uniformly in [0,1]^d
        2. Evaluate HF and LF at each point
        3. Fit linear regression: y_hf ~ b0 + b1 * y_lf
        4. Return R² of the fit
    """
    np.random.seed(seed)

    # Uniform samples
    x = np.random.uniform(0, 1, (n_samples, dim))

    # Evaluate functions
    y_hf = f_hf(x).flatten()
    y_lf = f_lf(x, alpha).flatten()

    # Linear regression
    reg = LinearRegression()
    reg.fit(y_lf.reshape(-1, 1), y_hf)
    y_pred = reg.predict(y_lf.reshape(-1, 1))

    return r2_score(y_hf, y_pred)


def compute_rank_correlation(f_hf: Callable, f_lf: Callable,
                             dim: int, alpha: float,
                             n_samples: int = 100, seed: int = 42) -> float:
    """
    Compute Spearman rank correlation between HF and LF

    More robust to nonlinear relationships than R²
    """
    from scipy.stats import spearmanr

    np.random.seed(seed)
    x = np.random.uniform(0, 1, (n_samples, dim))

    y_hf = f_hf(x).flatten()
    y_lf = f_lf(x, alpha).flatten()

    corr, _ = spearmanr(y_hf, y_lf)
    return corr


def find_global_minimum(f: Callable, dim: int,
                        n_random: int = 10000, n_local: int = 20) -> Tuple[np.ndarray, float]:
    """
    Find global minimum of function using random search + local optimization
    """
    from scipy.optimize import minimize

    # Random search
    x_random = np.random.uniform(0, 1, (n_random, dim))
    y_random = f(x_random).flatten()
    best_idx = np.argmin(y_random)

    # Local optimization from best random points
    best_x = x_random[best_idx]
    best_y = y_random[best_idx]

    # Sort and take top candidates for local search
    sorted_idx = np.argsort(y_random)[:n_local]

    for idx in sorted_idx:
        result = minimize(
            lambda x: f(x.reshape(1, -1))[0, 0],
            x_random[idx],
            method='L-BFGS-B',
            bounds=[(0, 1)] * dim
        )
        if result.fun < best_y:
            best_x = result.x
            best_y = result.fun

    return best_x, best_y


# =============================================================================
# FUNCTION REGISTRY
# =============================================================================

SYNTHETIC_FUNCTIONS = {
    'Branin-2D': {
        'hf': branin_hf,
        'lf': branin_lf,
        'dim': 2,
        'bounds': np.array([[0, 1], [0, 1]]),
        'f_star': 0.397887,  # Known global minimum
        'description': '2D benchmark with 3 global minima'
    },
    'Park-4D': {
        'hf': park_hf,
        'lf': park_lf,
        'dim': 4,
        'bounds': np.array([[0, 1], [0, 1], [0, 1], [0, 1]]),
        'f_star': None,  # Will be computed
        'description': '4D benchmark with complex interactions'
    },
}


def get_function_info():
    """Print information about available test functions"""
    print("=" * 60)
    print("Available Synthetic Test Functions")
    print("=" * 60)

    for name, info in SYNTHETIC_FUNCTIONS.items():
        print(f"\n{name}:")
        print(f"  Dimension: {info['dim']}")
        print(f"  Description: {info['description']}")

        # Compute minimum if not known
        if info['f_star'] is None:
            _, f_star = find_global_minimum(info['hf'], info['dim'])
            print(f"  Global minimum (computed): {f_star:.4f}")
        else:
            print(f"  Global minimum: {info['f_star']:.6f}")

        # Compute informativeness for different alpha values
        print("  LF Informativeness (R²):")
        for alpha in [1.0, 0.8, 0.5, 0.2, 0.0]:
            r2 = compute_r2_informativeness(
                info['hf'], info['lf'], info['dim'], alpha
            )
            print(f"    α={alpha:.1f}: R²={r2:.4f}")


if __name__ == "__main__":
    get_function_info()
