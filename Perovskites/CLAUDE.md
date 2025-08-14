# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Multi-Fidelity Bayesian Optimization (MFBO) research project for optimizing perovskite solar cell materials. The project combines transfer learning with deep neural networks and Bayesian optimization to efficiently find optimal perovskite compositions while minimizing computational costs.

## Key Commands

### Installation and Setup
```bash
# Install dependencies for Transfer Learning + BO module
cd 2.Transfer_learning/Pure_TL_BO
pip install -r requirements.txt
```

### Running Experiments

```bash
# Single optimization run with visualization
python main.py --mode single --cost-budget 50 --verbose

# Multiple runs for statistical analysis
python main.py --mode multiple --num-runs 100 --cost-budget 50

# With hyperparameter Bayesian optimization
python main.py --mode single --use-hyperparameter-bo --pretrain-bo-trials 5 --finetune-bo-trials 5

# Run with custom configuration
python main.py --mode single --cost-budget 100 --num-initial 5 --verbose
```

### Testing
```bash
# Run tests (if test files exist)
pytest

# Run interactive tests
jupyter notebook test_tl_bo.ipynb
```

## Architecture

### Core Components

1. **Data Layer** (`0.Data/`)
   - `lookup_table.pkl`: Pre-computed perovskite properties database
   - Molecular descriptors: `organics.json`, `cations.json`, `anions.json`
   - Contains both high-fidelity (HSE06) and low-fidelity (GGA) bandgap calculations

2. **Multi-Fidelity Bayesian Optimization** (`1.Atlas_cod/`)
   - `MFBO.ipynb`: Main implementation using Atlas framework
   - Implements cost-aware optimization with intelligent fidelity selection
   - Uses 8:1 ratio of low:high fidelity evaluations

3. **Transfer Learning Module** (`2.Transfer_learning/Pure_TL_BO/`)
   - `models.py`: Neural network architectures for feature extraction
   - `optimization.py`: Bayesian optimization logic with Expected Improvement
   - `hyperparameter_optimization.py`: Automated neural architecture search
   - `data_utils.py`: Data loading and preprocessing for perovskites
   - `config.py`: Central configuration management

### Key Design Patterns

- **Two-Stage Learning**: Pretrain on cheap data → Fine-tune on expensive data
- **Uncertainty Quantification**: Bayesian Linear Regression for confidence estimates
- **Cost-Aware Optimization**: Budget constraints guide fidelity selection
- **Modular Architecture**: Clear separation between data, models, and optimization

### Data Flow

1. Load perovskite compositions and properties from lookup table
2. Apply transfer learning: extract features using DNN pretrained on low-fidelity data
3. Fine-tune model on limited high-fidelity samples
4. Use Bayesian optimization to suggest next experiments
5. Track costs and performance metrics throughout optimization

## Important Implementation Details

### Perovskite Representation
- Compositions: ABX3 structure (A=organic/cation, B=metal, X=halide)
- Feature encoding: Molecular descriptors concatenated into fixed-length vectors
- Target property: Bandgap optimization (typically targeting 1.34 eV)

### Cost Model
- Low-fidelity (GGA): Cost = 1 unit
- High-fidelity (HSE06): Cost = 8 units
- Budget typically 50-100 units per optimization run

### Optimization Strategy
- Initial random sampling (default 5 points)
- Expected Improvement acquisition function
- Multi-fidelity scheduling based on uncertainty and cost
- Early termination when target is achieved

### Key Parameters in `config.py`
- `PRETRAIN_EPOCHS`: Training epochs for low-fidelity data
- `FINETUNE_EPOCHS`: Fine-tuning epochs on high-fidelity data
- `NUM_INITIAL`: Initial random samples before optimization
- `COST_BUDGET`: Total computational budget
- `TARGET_VALUE`: Target bandgap value (1.34 eV)

## Research Context

This codebase implements methods from multi-fidelity optimization research, combining:
- Transfer learning for feature extraction from cheap simulations
- Bayesian optimization for sample-efficient exploration
- Cost-aware decision making for computational efficiency

The project demonstrates significant computational savings (8-10x) compared to traditional high-fidelity-only optimization while maintaining similar final performance.