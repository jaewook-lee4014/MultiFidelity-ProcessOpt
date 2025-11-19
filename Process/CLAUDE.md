# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Multi-Fidelity Process Optimization** project for chemical separation processes using Bayesian Optimization. The project compares high-fidelity rigorous simulations with low-fidelity shortcut methods to optimize process parameters while managing computational costs.

**Application Domain**: Glacial acetic acid separation from fermentation broth using liquid-liquid extraction and distillation columns.

**Core Framework**: BioSTEAM (Biorefinery Simulation and Techno-Economic Analysis Modules)

## Repository Structure

```
Process/
├── 1. Code/                      # Core simulation classes (PRIMARY)
│   ├── RigorousDesign.py        # High-fidelity MESH distillation simulator
│   ├── ShortCutDesign.py        # Low-fidelity shortcut design simulator
│   └── OneColumn.py             # Single column baseline model
├── 4. Modelling/                # Multi-fidelity modeling experiments
├── 7. Optimisation/             # Optimization algorithms (GP, DNN-based BO)
├── 6. TestCode/                 # Experimental implementations (BNN, GP, diffusion)
├── 8. Validation/               # Result validation notebooks
├── 5. Visualisation/            # Learning curves and plots
├── 0. HPC_Control/              # HPC resource monitoring
├── 2. BackupCode/               # Legacy code archive
└── 1-1. Code_py/                # Standalone optimization examples
```

## Core Architecture

### Dual-Fidelity Simulation Framework

The project uses two simulation classes with **identical interfaces** but different computational costs:

1. **`RigorousDesign`** (High-Fidelity)
   - Uses MESH (non-equilibrium) distillation equations
   - Simulation time: ~minutes per evaluation
   - High accuracy, computationally expensive
   - Location: `1. Code/RigorousDesign.py`

2. **`ShortCutDesign`** (Low-Fidelity)
   - Uses Fenske-Underwood-Gilliland shortcut methods
   - Simulation time: ~seconds per evaluation (10-100x faster)
   - Lower accuracy, computationally cheap
   - Location: `1. Code/ShortCutDesign.py`

### Key Class Methods (Both Classes)

```python
# Initialization
model = RigorousDesign(verbose=False)  # or ShortCutDesign(verbose=False)

# Optimization interface
objective_value = model.func(X)  # X is 8D design vector

# Access methods
bounds = model._bounds()          # Get variable bounds
ints = model._integrality()       # Which variables are integers
model.simulate()                  # Run simulation
msp = model.MSP()                # Get Minimum Selling Price (objective)
capex = model.capex()            # Get capital costs
opex = model.opex()              # Get operating costs
cost = model.cost()              # Get total cost

# Constraint checking
constraint = model.acetic_acid_constraint()  # Returns constraint value

# History tracking
model.history  # List of [iteration, objective, purity, time]
model.nEval    # Number of evaluations counter
```

### Design Variables (8D Vector)

```python
X = [
    n_stages,    # Number of extraction stages (10-50, integer)
    Lr1, Hr1,    # Light/heavy key recoveries for extract distillation (0-0.9999)
    Lr2, Hr2,    # Light/heavy key recoveries for acid purification (0-0.9999)
    T_hex,       # Heat exchanger temperature (273-350 K)
    Lr3, Hr3,    # Light/heavy key recoveries for raffinate distillation (0-0.9999)
]
```

### Process Flow

1. **Extraction**: Liquid-liquid extraction with ethyl acetate solvent
2. **Extract Distillation**: Separate water from acetic acid
3. **Acid Purification**: MESH distillation for high purity product
4. **Solvent Recovery**: Heat exchanger + settler + distillation
5. **Raffinate Treatment**: Final distillation to recover product

**Objective**: Minimize MSP (Minimum Selling Price, $/kg)
**Constraint**: Product purity ≥ 98 wt% acetic acid

## Dependencies

Install required packages:

```bash
pip install biosteam scipy numpy matplotlib
```

Optional packages for advanced optimization:
```bash
pip install scikit-optimize torch
```

**Core Dependencies**:
- `biosteam` - Process simulation framework
- `scipy` - Optimization algorithms (differential_evolution, Latin Hypercube sampling)
- `numpy` - Numerical computing
- `matplotlib` - Visualization

**Optional**:
- `scikit-optimize` - Gaussian Process optimization (used in `7. Optimisation/`)
- `torch` - PyTorch for neural network surrogates (BNN, DNN)

## Common Development Tasks

### Running Simulations

```python
# Import simulation classes
import sys
sys.path.append('/Users/k23070952/MultiFidelity-ProcessOpt/Process/1. Code')
from RigorousDesign import RigorousDesign
from ShortCutDesign import ShortCutDesign

# Initialize and run
rigorous = RigorousDesign(verbose=True)
X = [12, 0.95, 0.95, 0.999, 0.999, 310, 0.99, 0.99]
objective = rigorous.func(X)
```

### Running Notebooks

Primary optimization notebooks:
```bash
# Main Gaussian Process + Bayesian Optimization
jupyter notebook "7. Optimisation/GPR_bo.ipynb"

# DNN-based Bayesian Optimization
jupyter notebook "7. Optimisation/DNN_bo.ipynb"

# Multi-fidelity modeling
jupyter notebook "4. Modelling/baseline.ipynb"
jupyter notebook "4. Modelling/transfer_learning.ipynb"
```

### Typical Optimization Workflow

```python
from scipy.optimize import differential_evolution

# 1. Initialize model
model = ShortCutDesign(verbose=True)

# 2. Get bounds and integrality constraints
bounds = model._bounds()
ints = model._integrality()

# 3. Run optimization
result = differential_evolution(
    model.func,
    bounds=bounds,
    integrality=ints,
    maxiter=100,
    workers=1
)

print(f"Best MSP: {result.fun}")
print(f"Best X: {result.x}")
```

## Multi-Fidelity Bayesian Optimization Pattern

The repository implements a common pattern across multiple notebooks:

1. **Generate low-fidelity samples** (fast exploration)
   ```python
   shortcut = ShortCutDesign()
   Y_lofi = [shortcut.func(X) for X in samples]
   ```

2. **Build surrogate model** (GP, BNN, or DNN)
   ```python
   surrogate = train_model(X_samples, Y_lofi)
   ```

3. **Use acquisition function** (Expected Improvement)
   ```python
   candidates = acquisition_function(surrogate, bounds)
   ```

4. **Evaluate with high-fidelity** (accurate but expensive)
   ```python
   rigorous = RigorousDesign()
   Y_hifi = [rigorous.func(X) for X in candidates]
   ```

5. **Update and repeat**

## Important Implementation Details

### Constraint Handling

Both classes implement constraint violations as penalties:
```python
# If product purity < 98%, add penalty to objective
penalty = 10 * (constraint_violation ** 2)
```

### Heuristic Parameters

`RigorousDesign` requires heuristic parameters from `ShortCutDesign`:
```python
# Get heuristics from shortcut design
shortcut = ShortCutDesign()
shortcut.func(X)  # Run simulation
heuristics = shortcut.shortcut_results()

# Use heuristics to initialize rigorous design
rigorous = RigorousDesign()
rigorous._SetRigorous(X, heuristics)
```

Heuristic parameters include:
- `boilup_1/2/3`: Distillation column boilup ratios
- `N_stages_1/2/3`: Number of theoretical stages
- `feed_stage_1/2/3`: Optimal feed stage locations
- `SplitRatio`: Settler split fraction

### Error Handling

Both classes track simulation errors:
```python
model.error_num  # Count of failed simulations
```

Failed simulations return high penalty values (typically 1e10).

### Performance Tracking

```python
model.history  # Format: [iteration, objective, purity, elapsed_time]
model.nEval    # Total function evaluations
```

## Key Notebooks

- **`7. Optimisation/GPR_bo.ipynb`** (1.5+ MB): Main Gaussian Process Regression + BO implementation
- **`4. Modelling/baseline.ipynb`** (1.8 MB): Reference high-fidelity optimization results
- **`4. Modelling/OnlyHiFi.ipynb`** (1.5 MB): Pure rigorous design optimization
- **`4. Modelling/transfer_learning.ipynb`**: Transfer learning between fidelities
- **`check_results.ipynb`**: Result analysis and verification

## Working with BioSTEAM

BioSTEAM requires careful setup:

```python
import biosteam as bst

# Enable notebook mode (light-mode diagrams)
bst.nbtutorial()

# Define chemicals for the process
bst.settings.set_thermo(['Water', 'AceticAcid', 'EthylAcetate'])

# Create streams
feed = bst.Stream(ID='feed', AceticAcid=1000, Water=9000, units='kg/hr')

# Create system
with bst.System('system_name') as sys:
    # Define unit operations here
    pass

# Simulate
sys.simulate()
```

## Git Workflow Notes

- Modified files in `../Perovskites/` are part of a separate Transfer Learning BO project (not Process-specific)
- Main branch is `main` (no upstream branch configured)
- Recent commits focus on DNGO/BNN implementations and visualization features

## Architecture Patterns

1. **Modular Design**: Easy swapping between ShortCut and Rigorous classes
2. **History Tracking**: All optimizations save iteration history for analysis
3. **Constraint-Aware**: Penalty methods enforce product purity constraints
4. **Cost-Conscious**: Explicit tracking of computational budget
5. **Dual-Fidelity**: Strategic use of cheap approximations vs. expensive accurate simulations

## Notes for Future Development

- Both `RigorousDesign` and `ShortCutDesign` classes should maintain identical interfaces
- Always use `func(X)` as the optimization interface
- Test with `ShortCutDesign` first before running expensive `RigorousDesign` evaluations
- Consider computational budget when designing optimization strategies
- Use `verbose=True` during development for detailed output
- Heuristic parameters from shortcut methods are critical for rigorous simulation convergence
