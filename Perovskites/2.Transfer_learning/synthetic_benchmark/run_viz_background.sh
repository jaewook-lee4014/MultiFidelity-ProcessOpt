#!/bin/bash
# Run visualize_mf_bo_steps.py in background (survives session disconnect)

cd /cephfs/volumes/hpc_data_prj/eng_waste_to_protein/ae035a41-20d2-44f3-aa46-14424ab0f6bf/repositories/MultiFidelity-ProcessOpt/Perovskites/2.Transfer_learning/synthetic_benchmark

source /software/spackages_v0_21_prod/apps/linux-ubuntu22.04-zen2/gcc-13.2.0/anaconda3-2022.10-5wy43yh5crcsmws4afls5thwoskzarhe/etc/profile.d/conda.sh
conda activate pytorch_env

PYTHONUNBUFFERED=1 python visualize_mf_bo_steps.py --seed 1004 --budget 50 --save-interval 1 > viz_run_1004.log 2>&1
