# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-Fidelity Bayesian Optimization research with two application domains:
- **Perovskites**: Optimizing ABX3 perovskite solar cell compositions for target bandgap (1.34 eV) using DFT calculations at two fidelity levels (GGA=cost 1, HSE06=cost 8)
- **Process**: Minimizing selling price ($/kg) of glacial acetic acid separation using BioSTEAM process simulation (shortcut vs rigorous MESH distillation)

Each domain has its own `CLAUDE.md` with domain-specific details—refer to `Perovskites/CLAUDE.md` and `Process/CLAUDE.md`.

## Setup and Commands

### Perovskites (primary active development)

```bash
# Install
cd Perovskites/2.Transfer_learning/Pure_TL_BO
pip install -r requirements.txt

# Single optimization run
python main.py --mode single --cost-budget 50 --verbose

# Multiple runs for statistics
python main.py --mode multiple --num-runs 100 --cost-budget 50

# 12-model benchmark comparison
cd ../mf_benchmark
pip install -r requirements.txt
python benchmark_parallel.py --n-seeds 20 --n-workers 48
```

### Process

```bash
pip install biosteam scipy scikit-optimize torch
# Primarily notebook-driven — see Process/4. Modelling/ and Process/7. Optimisation/
```

### HPC (King's College London CREATE, SLURM)

```bash
# Template at Perovskites/2.Transfer_learning/Pure_TL_BO/templates/submit_template.sh
# Targets NVIDIA GH200 nodes, CUDA 12.4+
sbatch submit_all_models.sh
```

### Testing

No formal test framework. Validation is through Jupyter notebooks and ad-hoc `test_*.py` scripts. No CI/CD.

## Architecture

### Perovskites Pipeline

```
Low-fidelity DFT data → Pretrain DNN feature extractor → Fine-tune on HF data → BLR for UQ → EI acquisition → Next experiment
```

**Core module**: `Perovskites/2.Transfer_learning/Pure_TL_BO/`
- `models.py` — DNN architectures
- `optimization.py` — BO loop with Expected Improvement
- `data_utils.py` — Perovskite data loading (from `0.Data/lookup_table.pkl`)
- `config.py` — Central parameters (epochs, budget, target value)
- `main.py` — CLI entry point (single/multiple run modes)

**Benchmark framework**: `Perovskites/2.Transfer_learning/mf_benchmark/`
- 12 transfer learning variants: MFGP, Sequential TL, Progressive Unfreezing, Curriculum Learning, DNGO (Joint/Gradient), Knowledge Distillation, Domain Adaptation (MMD), Soft Parameter Sharing, Pseudo-Labeling, Adapters, Two-Stage Joint

### Process Pipeline

```
ShortCutDesign(X) → heuristic params → RigorousDesign(X, heuristics) → MSP
```

**Core classes** in `Process/1. Code/`:
- `ShortCutDesign.py` — Low-fidelity (Fenske-Underwood-Gilliland shortcuts)
- `RigorousDesign.py` — High-fidelity (MESH distillation)
- Interfaces differ: `ShortCutDesign.func(X)` returns scalar; `RigorousDesign.func(X, heuristic_parameters)` returns tuple `(capex, opex, purity, time, msp)`
- Failed simulations return penalty value 100 (not inf)
- 8D design vector: extraction stages, 3×(Lr, Hr) recovery pairs, HX temperature

## Key Dependencies

- **Perovskites**: PyTorch, BoTorch, GPyTorch, RDKit, scikit-optimize, optuna
- **Process**: BioSTEAM, scipy, scikit-optimize, PyTorch

## Repository Conventions

- Paths with spaces: Process subdirectories use spaces (e.g., `Process/1. Code/`)
- Data files: `Perovskites/0.Data/lookup_table.pkl` is the central perovskite database
- BioSTEAM requires `bst.nbtutorial()` before use (called in class constructors)
- Always test with `ShortCutDesign` before running expensive `RigorousDesign`
