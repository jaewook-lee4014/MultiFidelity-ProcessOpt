# MultiFidelity-ProcessOpt

Research codebase for **Multi-Fidelity Bayesian Optimization (MFBO)** applied to two distinct scientific domains:

| Domain | Goal | Low-Fidelity (cost) | High-Fidelity (cost) |
|--------|------|---------------------|----------------------|
| **Perovskites** | Find ABX₃ compositions with bandgap ≈ 1.34 eV | GGA DFT (1) | HSE06 DFT (8) |
| **Process**     | Minimize MSP ($/kg) of glacial acetic acid separation | Fenske–Underwood–Gilliland shortcut | MESH rigorous distillation |

Both domains share the same algorithmic core — a deep neural network feature extractor pre-trained on cheap data, fine-tuned on a small high-fidelity set, with Bayesian Linear Regression (BLR) providing uncertainty for an Expected Improvement acquisition.

---

## Repository layout

```
.
├── Perovskites/                            # Primary active development
│   ├── 0.Data/                             # lookup_table.pkl + descriptor JSONs
│   ├── 1.Atlas_cod/                        # Atlas-framework MFBO notebooks
│   ├── 2.Transfer_learning/
│   │   ├── Pure_TL_BO/                     # Core TL+BO library (CLI: main.py)
│   │   ├── mf_benchmark/                   # 12-model parallel benchmark + adaptive fidelity
│   │   ├── synthetic_benchmark/            # Branin/Park/FreeSolv/COFs/Polar test functions
│   │   └── ...
│   ├── results/                            # Aggregated experiment outputs
│   ├── DNGO_Implementation_Guide.md
│   ├── README.md                           # Domain-specific Perovskite README (Korean)
│   └── CLAUDE.md
│
├── Process/                                # Acetic-acid separation (BioSTEAM)
│   ├── 1. Code/                            # ShortCutDesign.py, RigorousDesign.py, OneColumn.py
│   ├── 1-1. Code_py/                       # Standalone optimization scripts
│   ├── 3. Data/
│   ├── 4. Modelling/                       # baseline / transfer_learning / OnlyHiFi notebooks
│   ├── 5. Visualisation/                   # PFD diagrams, learning curves
│   ├── 6. TestCode/                        # BNN, conditional diffusion, MF-GP experiments
│   ├── 7. Optimisation/                    # GPR_bo, DNN_bo, DNGO_bo notebooks
│   ├── 8. Validation/
│   └── CLAUDE.md
│
├── BASELINE_SPEC.md                        # Baseline method specifications (untracked)
├── CLAUDE.md                               # Top-level guidance for Claude Code
├── requirements.txt                        # Minimal root deps (biosteam, scipy)
└── .gitignore
```

Domain-specific guidance lives in `Perovskites/CLAUDE.md` and `Process/CLAUDE.md`.

---

## Quick start

### Perovskites — single MFBO run

```bash
cd Perovskites/2.Transfer_learning/Pure_TL_BO
pip install -r requirements.txt

# Single run (cost budget 50 units)
python main.py --mode single --cost-budget 50 --verbose

# 100 replicates for statistics
python main.py --mode multiple --num-runs 100 --cost-budget 50
```

### Perovskites — 12-model benchmark

```bash
cd Perovskites/2.Transfer_learning/mf_benchmark
pip install -r requirements.txt
python benchmark_parallel.py --n-seeds 20 --n-workers 48
```

Variants compared: MFGP, Sequential TL, Progressive Unfreezing, Curriculum Learning, DNGO (Joint / Gradient), Knowledge Distillation, Domain Adaptation (MMD), Soft Parameter Sharing, Pseudo-Labeling, Adapters, Two-Stage Joint.

### Perovskites — adaptive fidelity selection

`mf_benchmark/benchmark_adaptive.py` implements **α-weighted cost-weighted EI**:

```
score_lf = α · EI_lf / ρ        score_hf = EI_hf / 1.0
α = |corr(y_lf, y_hf)|  clipped to [0.1, 1.0]
```

with a safety cap of `max(3, ⌊2/ρ⌋)` consecutive LF picks before forcing an HF query.

### Process — BioSTEAM simulation

```bash
pip install biosteam scipy scikit-optimize torch
```

Always call `ShortCutDesign` before the expensive `RigorousDesign`; the latter requires heuristic parameters derived from the shortcut run:

```python
shortcut = ShortCutDesign()
heuristics = shortcut.shortcut_results(X)
rigorous  = RigorousDesign()
capex, opex, purity, time, msp = rigorous.func(X, heuristic_parameters=heuristics)
```

8-D design vector: `[n_stages, Lr1, Hr1, Lr2, Hr2, T_hex, Lr3, Hr3]`.
Failed simulations return penalty value `100` (not `inf`). Operating hours: 7920 hr/yr.

### HPC (KCL CREATE, SLURM)

```bash
# Template: Perovskites/2.Transfer_learning/Pure_TL_BO/templates/submit_template.sh
# Targets: NVIDIA GH200 / A100 80G / H200 / B200
sbatch submit_all_models.sh
```

Partitions in use: `interruptible_gpu` (A100 80G, requires `--constraint=a100_80g`) and `tier1_gpu` (H200 / B200).

---

## Key dependencies

| Component | Stack |
|-----------|-------|
| Perovskites | PyTorch 2.9+, BoTorch 0.16.1, GPyTorch, RDKit, scikit-optimize, optuna |
| Process     | BioSTEAM, scipy, scikit-optimize, PyTorch |

> **BoTorch 0.16.1 note:** `qMultiFidelityExpectedImprovement` is **not** available. Use `qMultiFidelityKnowledgeGradient` or a manual cost-weighted EI.

---

## Repository conventions

- `Process/` subdirectories contain **spaces** — quote paths in shell commands.
- `Perovskites/0.Data/lookup_table.pkl` is the canonical perovskite database.
- BioSTEAM requires `bst.nbtutorial()` once per session (called inside the class constructors).
- No formal test framework; validation lives in `test_*.py` scripts and Jupyter notebooks. No CI/CD.

---

## Repository sync status (as of 2026-05-18)

**Remote:** `git@github.com:jaewook-lee4014/MultiFidelity-ProcessOpt.git`
**Branch:** `main` → `origin/main`

| Indicator | Value |
|-----------|-------|
| Commits ahead of `origin/main` | **0** |
| Commits behind `origin/main` | **0** |
| HEAD | `1ec574f Add LF-BLR benchmark results (20260123)` |
| Last push | 2026-01-23 |

The branch **tip is in sync**, but the working tree has accumulated substantial unpushed work since the last commit:

| Drift | Count | Notes |
|-------|-------|-------|
| Modified tracked files | **7** | ~1,099 insertions / 502 deletions across `mf_benchmark/benchmark_parallel.py`, four `synthetic_benchmark/*.py` files, `Process/5. Visualisation/learning_curve.ipynb`, `Process/CLAUDE.md` |
| Untracked items | **91** | New scripts + result directories |
| New top-level docs | 2 | `BASELINE_SPEC.md`, root-level `CLAUDE.md` |
| New code (untracked) | ~30 `.py` files | adaptive-fidelity benchmark, MFGP variants, random baselines, plotting + submit scripts, UQ tests |
| Result directories (untracked) | ~12 | `benchmark_adaptive_*`, `mfgp_variants_*`, `new_baselines_*`, `blr_placement_*`, `compare_mf_models_*`, etc. — totaling ≈ **120 MB** (largest: `benchmark_adaptive_phase2/` at 46 MB) |
| New methodology docs | 2 | `mf_benchmark/METHODOLOGY.md`, `mf_benchmark/latex_regret_tables.md` |
| New Process PFD assets | 11 files | SVG / PNG / PDF flowsheets in `Process/5. Visualisation/` |

**Bottom line:** roughly four months of development (Jan → May 2026) — including the adaptive-fidelity work, MFGP variant sweep, new baselines, and Process visualization — is local-only. Before pushing, decide which of the result directories (~120 MB) belong in git versus `.gitignore`; the current `.gitignore` only excludes `/Process/3. Data`.

---

## License & contact

Research code, no license file present. For questions or issues, please open a GitHub issue.
