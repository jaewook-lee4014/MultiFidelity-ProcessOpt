#!/usr/bin/env python
"""
Analyze convergence speed - iterations to reach specific regret thresholds
"""

import json
import numpy as np
from pathlib import Path

# Load results
results_dir = Path("results_parallel_20251218_205819")
with open(results_dir / "results.json", "r") as f:
    results = json.load(f)

# Define thresholds for each function
thresholds = {
    "Branin-2D": [1.0, 0.5, 0.1, 0.05, 0.01, 0.001],  # f* = 0.3979
    "Park-4D": [0.5, 0.1, 0.01, 0.001, 1e-6, 1e-9]     # f* = 0
}

models = ['GP', 'DNGO', 'BNN', 'MC-Dropout', 'Deep Ensemble', 'SNGP']

print("=" * 80)
print("CONVERGENCE ANALYSIS: Iterations to reach regret threshold")
print("=" * 80)

for func_name, func_thresholds in thresholds.items():
    print(f"\n{'='*80}")
    print(f"{func_name}")
    print("=" * 80)

    for threshold in func_thresholds:
        print(f"\n--- Regret ≤ {threshold:.0e} ---")
        convergence = []

        for model in models:
            regrets = results[func_name][model]['regrets_mean']

            # Find first iteration where regret <= threshold
            iter_to_reach = None
            for i, r in enumerate(regrets):
                if r <= threshold:
                    iter_to_reach = i + 1  # 1-indexed
                    break

            if iter_to_reach is not None:
                convergence.append((model, iter_to_reach))
            else:
                convergence.append((model, ">50"))

        # Sort by convergence speed
        def sort_key(x):
            return x[1] if isinstance(x[1], int) else 999

        convergence.sort(key=sort_key)

        # Print ranking
        for rank, (model, iters) in enumerate(convergence, 1):
            if isinstance(iters, int):
                print(f"  {rank}. {model:15s}: {iters:3d} iterations")
            else:
                print(f"  {rank}. {model:15s}: {iters} (not reached)")

# Create summary table
print("\n" + "=" * 80)
print("SUMMARY TABLE: Iterations to reach threshold (- = not reached)")
print("=" * 80)

for func_name, func_thresholds in thresholds.items():
    print(f"\n{func_name}:")

    # Header
    header = f"{'Model':15s}"
    for t in func_thresholds:
        header += f" | {t:.0e}"
    print(header)
    print("-" * len(header))

    for model in models:
        regrets = results[func_name][model]['regrets_mean']
        row = f"{model:15s}"

        for threshold in func_thresholds:
            iter_to_reach = None
            for i, r in enumerate(regrets):
                if r <= threshold:
                    iter_to_reach = i + 1
                    break

            if iter_to_reach is not None:
                row += f" | {iter_to_reach:5d}"
            else:
                row += f" |     -"

        print(row)

# Rankings by average iterations across thresholds (lower is better)
print("\n" + "=" * 80)
print("OVERALL RANKING (Average iterations across all reachable thresholds)")
print("=" * 80)

overall_scores = {model: [] for model in models}

for func_name, func_thresholds in thresholds.items():
    for model in models:
        regrets = results[func_name][model]['regrets_mean']

        for threshold in func_thresholds:
            for i, r in enumerate(regrets):
                if r <= threshold:
                    overall_scores[model].append(i + 1)
                    break

print(f"\n{'Model':15s} | {'Avg Iters':>10s} | {'# Thresholds Reached':>20s}")
print("-" * 50)

ranking = []
for model in models:
    scores = overall_scores[model]
    if scores:
        avg = np.mean(scores)
        ranking.append((model, avg, len(scores)))
    else:
        ranking.append((model, 999, 0))

ranking.sort(key=lambda x: (x[1], -x[2]))

for rank, (model, avg, n_reached) in enumerate(ranking, 1):
    if avg < 999:
        print(f"{rank}. {model:15s} | {avg:10.1f} | {n_reached:>20d}/12")
    else:
        print(f"{rank}. {model:15s} | {'N/A':>10s} | {n_reached:>20d}/12")
