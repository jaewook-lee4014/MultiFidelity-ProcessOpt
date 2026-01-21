"""
Synthetic Test Functions for Multi-Fidelity Bayesian Optimization Benchmarks

Based on: "Best Practices for Multi-Fidelity Bayesian Optimization" (Nature Comp Science)

Functions:
- Branin-2D: Classic 2D benchmark (f* ≈ 0.397887)
- Park-4D: 4D benchmark (f* ≈ 0)

Scenarios:
1. Favorable: α=0.6, ρ=0.1 (LF cheap & informative, R² > 0.9)
2. Unfavorable: α varies, ρ=0.5 (LF expensive & less informative, R² < 0.75)
   - Branin: α=0.1
   - Park: α=0.4

Cost Model:
- HF cost = 1 (always)
- LF cost = ρ (0.1 for favorable, 0.5 for unfavorable)
"""

import numpy as np
from typing import Callable, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# =============================================================================
# BRANIN FUNCTION (2D) - Paper SI A.2 Equation (2)
# =============================================================================

def branin_hf(x: np.ndarray) -> np.ndarray:
    """
    Branin function (High-Fidelity)

    Input: x in [0,1]^2 (normalized domain)
    Original domain: x1 ∈ [-5, 10], x2 ∈ [0, 15]

    f(x) = (x2 - (5.1/(4π²))x1² + (5/π)x1 - 6)² + 10(1 - 1/(8π))cos(x1) + 10

    Global minima: f* ≈ 0.397887
    """
    # Rescale from [0,1]^2 to original domain
    x1 = x[:, 0] * 15 - 5   # x1 ∈ [-5, 10]
    x2 = x[:, 1] * 15       # x2 ∈ [0, 15]

    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi

    result = (x2 - b * x1**2 + c * x1 - 6)**2 + 10 * (1 - 1/(8*np.pi)) * np.cos(x1) + 10

    return result.reshape(-1, 1)


def branin_lf(x: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    Branin function (Low-Fidelity) - Paper SI A.2 Equation (2)

    f(x, α) = (x2 - (5.1/(4π²) - 0.1(1-α))x1² + (5/π)x1 - 6)² + 10(1 - 1/(8π))cos(x1) + 10

    The modification: b_lf = 5.1/(4π²) - 0.1(1-α)

    α = 1.0: identical to HF
    α = 0.6: favorable scenario (R² > 0.9)
    α = 0.1: unfavorable scenario (R² < 0.75)
    """
    x1 = x[:, 0] * 15 - 5
    x2 = x[:, 1] * 15

    # Modified coefficient per paper
    b_lf = 5.1 / (4 * np.pi**2) - 0.1 * (1 - alpha)
    c = 5 / np.pi

    result = (x2 - b_lf * x1**2 + c * x1 - 6)**2 + 10 * (1 - 1/(8*np.pi)) * np.cos(x1) + 10

    return result.reshape(-1, 1)


# =============================================================================
# PARK FUNCTION (4D) - Paper SI A.2 Equation (3)
# =============================================================================

def park_hf(x: np.ndarray) -> np.ndarray:
    """
    Park function (High-Fidelity) - Paper SI A.2 Equation (3)

    Input: x in [0,1]^4

    f(x) = (x1/2) * sqrt(1 + (x2 + x3²)x4/x1² - 1) + (x1 + 3x4) * exp(1 + sin(x3))
         = (x1/2) * sqrt((x2 + x3²)x4/x1²) + (x1 + 3x4) * exp(1 + sin(x3))

    Note: -1 is INSIDE the sqrt, not outside!

    Global minimum: f* ≈ 0 (at boundary)
    """
    eps = 1e-8

    x1 = np.maximum(x[:, 0], eps)
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = np.maximum(x[:, 3], eps)

    # Park function formula - 논문: -1이 루트 안에 있음
    # sqrt(1 + (x2 + x3²)x4/x1² - 1) = sqrt((x2 + x3²)x4/x1²)
    inner = (x2 + x3**2) * x4 / (x1**2 + eps)
    term1 = (x1 / 2) * np.sqrt(np.maximum(inner, 0))
    term2 = (x1 + 3 * x4) * np.exp(1 + np.sin(x3))

    term1 = np.nan_to_num(term1, nan=0.0, posinf=10.0, neginf=-10.0)
    term2 = np.nan_to_num(term2, nan=0.0, posinf=100.0, neginf=-100.0)

    result = term1 + term2

    return result.reshape(-1, 1)


def park_lf(x: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    Park function (Low-Fidelity) - Modified for proper unfavorable scenario

    HF: f(x) = (x1/2)·sqrt((x2 + x3²)x4/x1²) + (x1 + 3·x4)·exp(1 + sin(x3))

    LF: Two modifications for better R² control:
        1. x4 coefficient: 3 → (3 - 1.5·(1-α))
        2. exp argument: exp(1 + sin(x3)) → exp(1 + α·sin(x3))

    α = 1.0: identical to HF (R² = 1.0)
    α = 0.6: moderate difference (R² > 0.9, favorable)
    α = 0.0: large difference (R² < 0.75, unfavorable)
    """
    eps = 1e-8

    x1 = np.maximum(x[:, 0], eps)
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = np.maximum(x[:, 3], eps)

    # Modification 1: x4 coefficient
    x4_coef = 3 - 1.5 * (1 - alpha)

    # term1: same as HF
    inner = (x2 + x3**2) * x4 / (x1**2 + eps)
    term1 = (x1 / 2) * np.sqrt(np.maximum(inner, 0))

    # term2: x4 coefficient + exp argument modified with alpha
    # Modification 2: exp(1 + sin(x3)) → exp(1 + alpha*sin(x3))
    term2 = (x1 + x4_coef * x4) * np.exp(1 + alpha * np.sin(x3))

    term1 = np.nan_to_num(term1, nan=0.0, posinf=10.0, neginf=-10.0)
    term2 = np.nan_to_num(term2, nan=0.0, posinf=100.0, neginf=-100.0)

    result = term1 + term2

    return result.reshape(-1, 1)


# =============================================================================
# INFORMATIVENESS METRICS
# =============================================================================

def compute_r2(f_hf: Callable, f_lf: Callable, dim: int, alpha: float,
               n_samples: int = 1000, seed: int = 42) -> float:
    """
    Compute R² between HF and LF functions (informativeness measure)

    R² = 1: LF perfectly predicts HF (most informative)
    R² = 0: No linear correlation
    """
    np.random.seed(seed)
    x = np.random.uniform(0, 1, (n_samples, dim))

    y_hf = f_hf(x).flatten()
    y_lf = f_lf(x, alpha).flatten()

    reg = LinearRegression()
    reg.fit(y_lf.reshape(-1, 1), y_hf)
    y_pred = reg.predict(y_lf.reshape(-1, 1))

    return r2_score(y_hf, y_pred)


def find_global_minimum(f: Callable, dim: int,
                        n_random: int = 10000, n_local: int = 20) -> Tuple[np.ndarray, float]:
    """Find global minimum using random search + local optimization"""
    from scipy.optimize import minimize

    np.random.seed(42)
    x_random = np.random.uniform(0, 1, (n_random, dim))
    y_random = f(x_random).flatten()
    best_idx = np.argmin(y_random)

    best_x = x_random[best_idx]
    best_y = y_random[best_idx]

    sorted_idx = np.argsort(y_random)[:n_local]

    for idx in sorted_idx:
        result = minimize(
            lambda xx: f(xx.reshape(1, -1))[0, 0],
            x_random[idx],
            method='L-BFGS-B',
            bounds=[(0, 1)] * dim
        )
        if result.fun < best_y:
            best_x = result.x
            best_y = result.fun

    return best_x, best_y


# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================

SCENARIOS = {
    'favorable': {
        'description': 'LF is cheap (ρ=0.1) and informative (R² > 0.9)',
        'alpha_branin': 0.8,  # R² ≈ 0.97
        'alpha_park': 0.6,    # R² ≈ 0.99
        'rho': 0.1,  # LF cost = 0.1, HF cost = 1
    },
    'unfavorable': {
        'description': 'LF is expensive (ρ=0.5) and less informative (R² < 0.75)',
        'alpha_branin': 0.1,  # R² ≈ 0.56
        'alpha_park': 0.0,    # R² ≈ 0.68
        'rho': 0.5,  # LF cost = 0.5, HF cost = 1
    }
}

FUNCTIONS = {
    'Branin-2D': {
        'hf': branin_hf,
        'lf': branin_lf,
        'dim': 2,
        'f_star': 0.397887,
    },
    'Park-4D': {
        'hf': park_hf,
        'lf': park_lf,
        'dim': 4,
        'f_star': None,  # Will be computed
    }
}


def verify_scenarios():
    """Verify R² values match paper expectations"""
    print("=" * 70)
    print("Scenario Verification (R² informativeness)")
    print("=" * 70)

    # Compute Park f*
    _, park_fstar = find_global_minimum(park_hf, 4)
    FUNCTIONS['Park-4D']['f_star'] = park_fstar
    print(f"\nPark-4D f* = {park_fstar:.6f}")

    print("\n" + "-" * 70)
    print("Favorable Scenario (ρ=0.1, R² > 0.9)")
    print("-" * 70)
    alpha_b = SCENARIOS['favorable']['alpha_branin']
    alpha_p = SCENARIOS['favorable']['alpha_park']
    r2_b = compute_r2(branin_hf, branin_lf, 2, alpha_b)
    r2_p = compute_r2(park_hf, park_lf, 4, alpha_p)
    print(f"  Branin-2D (α={alpha_b}): R² = {r2_b:.4f} {'✓' if r2_b > 0.9 else '✗'}")
    print(f"  Park-4D (α={alpha_p}): R² = {r2_p:.4f} {'✓' if r2_p > 0.9 else '✗'}")

    print("\n" + "-" * 70)
    print("Unfavorable Scenario (ρ=0.5, R² < 0.75)")
    print("-" * 70)
    alpha_b = SCENARIOS['unfavorable']['alpha_branin']
    alpha_p = SCENARIOS['unfavorable']['alpha_park']
    r2_b = compute_r2(branin_hf, branin_lf, 2, alpha_b)
    r2_p = compute_r2(park_hf, park_lf, 4, alpha_p)
    print(f"  Branin-2D (α={alpha_b}): R² = {r2_b:.4f} {'✓' if r2_b < 0.75 else '✗'}")
    print(f"  Park-4D (α={alpha_p}): R² = {r2_p:.4f} {'✓' if r2_p < 0.75 else '✗'}")

    # Show R² for various alpha values
    print("\n" + "-" * 70)
    print("R² vs Alpha sweep")
    print("-" * 70)
    print(f"{'Alpha':<8} {'Branin R²':<12} {'Park R²':<12}")
    for alpha in [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0]:
        r2_b = compute_r2(branin_hf, branin_lf, 2, alpha)
        r2_p = compute_r2(park_hf, park_lf, 4, alpha)
        print(f"{alpha:<8.1f} {r2_b:<12.4f} {r2_p:<12.4f}")


if __name__ == "__main__":
    verify_scenarios()
