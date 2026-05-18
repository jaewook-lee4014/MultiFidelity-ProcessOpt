#!/usr/bin/env python
"""Smoke test for alpha-weighted adaptive fidelity selection (v2 with safety streak)."""
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import time

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from benchmark_adaptive import (
    SyntheticBenchmark, branin_hf, branin_lf,
    MFGP, Sequential, run_bo_adaptive, GP_MODELS, compute_alpha
)

# ====== Test 1: Branin-Fav (rho=0.1, favorable) ======
print("\n" + "="*60)
print("TEST 1: Branin-Fav (rho=0.1, favorable)")
print("="*60)
benchmark = SyntheticBenchmark('Branin-Fav', branin_hf, branin_lf, 2, 0.8, 0.1, 0.397887, 50)
print(f"n_candidates={benchmark.n_candidates}, rho={benchmark.cost_ratio}")

for model_name, model_class in [('MFGP', MFGP), ('Sequential', Sequential)]:
    print(f"\n--- {model_name} ---")
    t0 = time.time()
    result = run_bo_adaptive(benchmark, model_class, budget=50, seed=42, device=device)
    elapsed = time.time() - t0
    
    decisions = result['fidelity_decisions']
    n_hf_chosen = sum(1 for d in decisions if d['fidelity_chosen'] == 1)
    n_lf_chosen = sum(1 for d in decisions if d['fidelity_chosen'] == 0)
    
    print(f"  Final regret: {result['final_regret']:.6f}")
    print(f"  Best y: {result['best_y']:.6f}")
    print(f"  n_hf={result['n_hf']}, n_lf={result['n_lf']}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Fidelity choices: HF={n_hf_chosen}, LF={n_lf_chosen}")
    
    if n_hf_chosen == 0:
        print(f"  FAIL: All LF, no HF (degenerate)")
    elif n_lf_chosen == 0:
        print(f"  WARN: All HF, no LF")
    else:
        hf_ratio = n_hf_chosen / (n_hf_chosen + n_lf_chosen)
        print(f"  OK: Mixed (HF ratio: {hf_ratio:.1%})")
    
    # Print alpha evolution
    alphas = [d['alpha'] for d in decisions if not np.isnan(d.get('alpha', float('nan')))]
    if alphas:
        print(f"  Alpha: start={alphas[0]:.4f}, end={alphas[-1]:.4f}")
    
    # Print first 10 decisions
    print(f"  First 10 decisions:")
    for d in decisions[:10]:
        fid = "HF" if d['fidelity_chosen'] == 1 else "LF"
        clfc = d.get('consecutive_lf_count', '?')
        print(f"    step={d['step']:3d} {fid} cost={d['cost']:.2f} "
              f"score={d['score_max']:.4f} alpha={d['alpha']:.4f} clfc={clfc}")
    
    if len(decisions) > 15:
        print(f"  Last 5 decisions:")
        for d in decisions[-5:]:
            fid = "HF" if d['fidelity_chosen'] == 1 else "LF"
            clfc = d.get('consecutive_lf_count', '?')
            print(f"    step={d['step']:3d} {fid} cost={d['cost']:.2f} "
                  f"score={d['score_max']:.4f} alpha={d['alpha']:.4f} clfc={clfc}")

# ====== Test 2: Branin-Unfav (rho=0.5, unfavorable) ======
print("\n" + "="*60)
print("TEST 2: Branin-Unfav (rho=0.5, unfavorable)")
print("="*60)
benchmark_unfav = SyntheticBenchmark('Branin-Unfav', branin_hf, branin_lf, 2, 0.1, 0.5, 0.397887, 50)
print(f"n_candidates={benchmark_unfav.n_candidates}, rho={benchmark_unfav.cost_ratio}")

for model_name, model_class in [('MFGP', MFGP), ('Sequential', Sequential)]:
    print(f"\n--- {model_name} ---")
    t0 = time.time()
    result = run_bo_adaptive(benchmark_unfav, model_class, budget=50, seed=42, device=device)
    elapsed = time.time() - t0
    
    decisions = result['fidelity_decisions']
    n_hf_chosen = sum(1 for d in decisions if d['fidelity_chosen'] == 1)
    n_lf_chosen = sum(1 for d in decisions if d['fidelity_chosen'] == 0)
    
    print(f"  Final regret: {result['final_regret']:.6f}")
    print(f"  Best y: {result['best_y']:.6f}")
    print(f"  n_hf={result['n_hf']}, n_lf={result['n_lf']}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Fidelity choices: HF={n_hf_chosen}, LF={n_lf_chosen}")
    
    if n_hf_chosen == 0:
        print(f"  FAIL: All LF (degenerate)")
    elif n_lf_chosen == 0:
        print(f"  OK for unfavorable: All HF (expected)")
    else:
        hf_ratio = n_hf_chosen / (n_hf_chosen + n_lf_chosen)
        print(f"  Mixed (HF ratio: {hf_ratio:.1%})")
    
    alphas = [d['alpha'] for d in decisions if not np.isnan(d.get('alpha', float('nan')))]
    if alphas:
        print(f"  Alpha: start={alphas[0]:.4f}, end={alphas[-1]:.4f}")

print("\n" + "="*60)
print("SMOKE TEST COMPLETE")
print("="*60)
