#!/bin/bash
# Smoke test: run each new baseline on Branin-Fav, seed=42, 1 seed only.
# Order: fastest to slowest.

set -e
cd "$(dirname "$0")"

echo "=== Smoke Test: 4 New Baselines on Branin-Fav (seed=42) ==="

python -c "
import sys, time, numpy as np, torch
from synthetic_functions import branin_hf, branin_lf
from benchmark_parallel import SyntheticBenchmark, run_bo, MFGP
from baselines import SparseMFGP, NARGP, DKLMultiFidelity, run_successive_halving

benchmark = SyntheticBenchmark('Branin-Fav', branin_hf, branin_lf, 2, 0.8, 0.1, 0.397887, 50)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
budget = 50
seed = 42

# 1. Successive Halving
t0 = time.time()
r = run_successive_halving(benchmark, budget, seed)
print(f'Successive Halving: regret={r[\"final_regret\"]:.4f}, n_hf={r[\"n_hf\"]}, n_lf={r[\"n_lf\"]}, time={time.time()-t0:.1f}s')

# 2. NARGP
t0 = time.time()
r = run_bo(benchmark, NARGP, budget, seed, device)
print(f'NARGP:              regret={r[\"final_regret\"]:.4f}, n_hf={r[\"n_hf\"]}, n_lf={r[\"n_lf\"]}, time={time.time()-t0:.1f}s')

# 3. Sparse MFGP
t0 = time.time()
r = run_bo(benchmark, SparseMFGP, budget, seed, device)
print(f'Sparse MFGP:        regret={r[\"final_regret\"]:.4f}, n_hf={r[\"n_hf\"]}, n_lf={r[\"n_lf\"]}, time={time.time()-t0:.1f}s')

# 4. DKL Multi-Fidelity
t0 = time.time()
r = run_bo(benchmark, DKLMultiFidelity, budget, seed, device)
print(f'DKL Multi-Fidelity: regret={r[\"final_regret\"]:.4f}, n_hf={r[\"n_hf\"]}, n_lf={r[\"n_lf\"]}, time={time.time()-t0:.1f}s')

print('=== All smoke tests passed ===')
"
