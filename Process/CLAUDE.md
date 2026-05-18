# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-fidelity Bayesian Optimization for glacial acetic acid separation from fermentation broth. Uses BioSTEAM process simulation with two fidelity levels: rigorous MESH distillation (expensive, accurate) vs. Fenske-Underwood-Gilliland shortcut methods (cheap, approximate).

**Objective**: Minimize MSP (Minimum Selling Price, $/kg) subject to product purity >= 98 wt% acetic acid.

## Dependencies

```bash
pip install biosteam scipy  # from requirements.txt in repo root
pip install scikit-optimize torch  # optional: GP optimization, neural network surrogates
```

## Core Simulation Classes

Both classes live in `1. Code/` and share most of the same interface, but with important differences:

### Interface Differences (NOT identical)

| Method | `ShortCutDesign` | `RigorousDesign` |
|--------|-------------------|-------------------|
| `func(X)` | Returns scalar (MSP value) | Returns tuple: `(capex, opex, purity, time, msp)` |
| `func(X, heuristic_parameters)` | Not supported | Requires heuristic dict from ShortCutDesign |
| Setup method | `_set(X)` | `_SetRigorous(X, heuristic_parameters)` |
| Extractor | `MixerSettler` (single-stage; `n_stages` param unused) | `MultiStageMixerSettlers` (multi-stage) |
| Distillation | `ShortcutColumn` for all columns | `MESHDistillation` for ED/ED2, `ShortcutColumn` for RD |
| `shortcut_results(X)` | Returns dict with heuristic params + costs | Not available |

### Shared Interface

```python
model._bounds()              # 8D bounds list
model._integrality()         # Boolean list (only n_stages is integer)
model.simulate()             # Run BioSTEAM simulation (increments nEval)
model.MSP()                  # Minimum Selling Price with constraint penalty
model.capex() / model.opex() # Equipment costs (MMUSD)
model.acetic_acid_constraint()  # 0 if purity >= 98%, positive violation otherwise
model.natural_units(X)       # Convert normalized [0,1] inputs to physical units
model.history                # List of [nEval, objective, purity, elapsed_time]
```

### 8D Design Vector

```python
X = [n_stages,       # Extraction stages (10-50, integer)
     Lr1, Hr1,       # Extract distiller recoveries (0-0.9999)
     Lr2, Hr2,       # Acid purification recoveries (0-0.9999)
     T_hex,          # Heat exchanger temperature (273-350 K)
     Lr3, Hr3]       # Raffinate distiller recoveries (0-0.9999)
```

### Heuristic Parameter Bridge

RigorousDesign needs heuristic parameters derived from ShortCutDesign to converge:

```python
shortcut = ShortCutDesign()
heuristics = shortcut.shortcut_results(X)  # Runs simulation, returns dict
# Dict keys: SplitRatio, boilup_1/2/3, N_stages_1/2/3, feed_stage_1/2/3, shortcut_time

rigorous = RigorousDesign()
result = rigorous.func(X, heuristic_parameters=heuristics)
```

### Error Handling

Failed simulations return penalty value of 100 (not np.inf). Both classes track `error_num` and also cap MSP > 100 to 50 with an error counter increment.

## Process Flow

Extraction (EtAc solvent, 1.5:1 ratio) -> Extract distiller (Water/AceticAcid separation) -> Acid purification (EtAc/AceticAcid separation) -> Product. Distillates mix and recycle through HX -> Settler -> Splitter back to extractor. Raffinate goes through its own distiller for solvent recovery.

Operating hours: 330 days/year (7920 hr/yr).

## Repository Layout

- `1. Code/` — Core simulation classes (RigorousDesign.py, ShortCutDesign.py, OneColumn.py)
- `4. Modelling/` — Multi-fidelity modeling notebooks (baseline, transfer_learning, OnlyHiFi)
- `7. Optimisation/` — BO notebooks: GPR_bo.ipynb, DNN_bo.ipynb, DNGO_bo.ipynb
- `6. TestCode/` — Experimental: BNN, conditional diffusion, MF-GP
- `1-1. Code_py/` — Standalone Python optimization scripts
- `../Perovskites/` — Separate transfer learning BO project (not Process-specific)

## Development Notes

- No test framework; validation is notebook-based
- No CI/CD pipeline
- BioSTEAM requires `bst.nbtutorial()` call before use (done in class constructors)
- Both classes call `bst.settings.set_thermo(['Water', 'AceticAcid', 'EthylAcetate'])` on every `_set`/`_SetRigorous` call
- Always test with ShortCutDesign before running expensive RigorousDesign evaluations
