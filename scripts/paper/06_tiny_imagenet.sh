#!/usr/bin/env bash
# Phase 06 (E6) — Tiny ImageNet, reduced grid (Tier 2). Long run.
# Usage: bash scripts/paper/06_tiny_imagenet.sh [main|spline_instability]
set -e
MODE=${1:-main}
mkdir -p logs/paper

if [[ "${MODE}" == "main" ]]; then
  python -m funcodekan.experiments.all_kan_datasets \
    --dataset tiny_imagenet --seed 42 \
    --out-dir runs_paper/tiny_imagenet --run-name tiny_main_seed42 \
    --variants gram fast mlp --methods function branch \
    --clusters-list 32 --bits-list 8 4 2 \
    --width 128 --epochs 60 --finetune-epochs 40 \
    --batch-size 512 --test-batch-size 1024 \
    2>&1 | tee logs/paper/06_tiny_main.log

elif [[ "${MODE}" == "spline_instability" ]]; then
  # C4 negative-result documentation: spline family on the hardest task.
  python -m funcodekan.experiments.all_kan_datasets \
    --dataset tiny_imagenet --seed 42 \
    --out-dir runs_paper/tiny_imagenet --run-name tiny_spline_seed42 \
    --variants spline --methods function branch \
    --clusters-list 32 --bits-list 4 \
    --width 128 --epochs 60 --finetune-epochs 40 \
    --batch-size 512 --test-batch-size 1024 \
    2>&1 | tee logs/paper/06_tiny_spline.log
else
  echo "Modes: main | spline_instability"; exit 1
fi
