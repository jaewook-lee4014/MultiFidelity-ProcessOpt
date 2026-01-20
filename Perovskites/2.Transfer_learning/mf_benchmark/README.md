# Multi-Fidelity Bayesian Optimization Benchmark

12개 Multi-Fidelity 모델을 비교하는 벤치마크 코드입니다.

## Models (12)

| Category | Model | Acquisition |
|----------|-------|-------------|
| **GP-based** | MFGP (BoTorch) | EI (Expected Improvement) |
| **Sequential Transfer** | Sequential | argmin(mean) |
| | Progressive | argmin(mean) |
| | Curriculum | argmin(mean) |
| | Two-Stage Joint | argmin(mean) |
| **DNGO Variants** | DNGO-Joint | argmin(mean) |
| | DNGO-Gradient | argmin(mean) |
| **Transfer Learning** | Knowledge Distillation | argmin(mean) |
| | Domain Adaptation (MMD) | argmin(mean) |
| | Soft Parameter Sharing | argmin(mean) |
| | Pseudo-Labeling | argmin(mean) |
| | Adapter | argmin(mean) |

## Benchmarks (6)

### Synthetic
- **Branin-Fav**: 2D, favorable scenario (high R^2, low cost ratio)
- **Branin-Unfav**: 2D, unfavorable scenario (low R^2, high cost ratio)
- **Park-4D**: 4D function

### Chemistry
- **COFs**: Covalent Organic Frameworks
- **FreeSolv**: Solvation free energy
- **Polarizability**: Molecular polarizability

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For GPU support (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Basic Run (3 seeds)
```bash
python benchmark.py
```

### Custom Seeds
```bash
python benchmark.py --n-seeds 5 --base-seed 0
```

### Run Script (HPC)
```bash
chmod +x run.sh
./run.sh
```

## Output

Results are saved to `benchmark_no_blr_YYYYMMDD_HHMMSS/`:
- `results.csv`: All results (benchmark, model, seed, final_regret, etc.)
- `{benchmark}_comparison.png`: Comparison plots per benchmark

## File Structure

```
mf_benchmark/
├── benchmark.py           # Main benchmark script
├── synthetic_functions.py # Branin, Park functions
├── requirements.txt       # Dependencies
├── README.md             # This file
├── run.sh                # HPC run script
└── data/
    ├── cofs.csv          # COFs dataset
    ├── freesolv.csv      # FreeSolv dataset
    └── polarizability.csv # Polarizability dataset
```

## Reference

Based on: "Best Practices for Multi-Fidelity Bayesian Optimization" (Nature Computational Science)
