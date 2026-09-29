#!/usr/bin/env bash
# Prepare the three HW models (GPU recommended). Idempotent per stage: skip
# a stage by commenting it out. On Windows run the same python commands in
# PowerShell (all launchers pass --num-workers 0; see docs/HW_DESIGN_CONTRACT.md).
set -euo pipefail
cd "$(dirname "$0")/../.."

# 1) Verified pipeline: dense FP32 (D1) + branch-aware Ks=32,Kb=16 (D3 init).
python -m funcodekan.experiments.hw_prepare --model branch_source --num-workers 0

# 2) D2: LSQ W4A4 QAT from the dense checkpoint (~20 epochs, cosine LR).
python -m funcodekan.experiments.hw_prepare --model lsq_w4a4 --num-workers 0

# 3) D3: LSQ A4 + W4 codebook fake-quant finetune (indices frozen).
python -m funcodekan.experiments.hw_prepare --model funccode_w4a4 --num-workers 0
