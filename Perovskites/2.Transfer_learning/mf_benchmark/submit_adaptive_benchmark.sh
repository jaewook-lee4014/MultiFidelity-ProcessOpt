#!/bin/bash
#SBATCH --job-name=adapt_p2
#SBATCH --partition=interruptible_gpu
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm_adaptive_p2_%A_%a.out
#SBATCH --error=slurm_adaptive_p2_%A_%a.err

echo "========================================"
echo "Adaptive Benchmark Phase 2 - Array task ${SLURM_ARRAY_TASK_ID}"
echo "========================================"
echo "Start: $(date)"
echo "Node: $(hostname)"
nvidia-smi 2>/dev/null || echo "No GPU info available"

export PYTHONUNBUFFERED=1
export PATH="/scratch/users/k23070952/vllm_env310/bin:$PATH"

cd /scratch/prj/eng_waste_to_protein/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/mf_benchmark

# Shared output directory for all array tasks
OUTPUT_DIR="benchmark_adaptive_phase2"

# Split 7 benchmarks across 4 array tasks
case ${SLURM_ARRAY_TASK_ID} in
    0) BENCHMARKS="Branin-Fav,Branin-Unfav" ;;
    1) BENCHMARKS="Park-Fav,Park-Unfav" ;;
    2) BENCHMARKS="COFs,FreeSolv" ;;
    3) BENCHMARKS="Polarizability" ;;
esac

echo "Benchmarks: ${BENCHMARKS}"
echo "Output dir: ${OUTPUT_DIR}"

python -u benchmark_adaptive.py \
    --n-seeds 20 \
    --n-workers 48 \
    --phase 2 \
    --benchmarks "${BENCHMARKS}" \
    --output-dir "${OUTPUT_DIR}"

echo "End: $(date)"
echo "========================================"
