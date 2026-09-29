#!/usr/bin/env bash
# Phase 01 (E1, E10) — MNIST + Fashion-MNIST main grids, 3 seeds.
# Usage: bash scripts/paper/01_mnist_family.sh [main|width_sweep]
set -e
MODE=${1:-main}
mkdir -p logs/paper

if [[ "${MODE}" == "main" ]]; then
  # E1a MNIST, seeds 42/123/2026 — existing verified multiseed protocol
  bash scripts/mnist/run_multiseed_all_kan.sh main_all_bits \
    2>&1 | tee logs/paper/01_mnist_multiseed.log

  # E1b Fashion-MNIST, same protocol via the dataset-generic driver
  for SEED in 42 123 2026; do
    python -m funcodekan.experiments.all_kan_datasets \
      --dataset fashion_mnist --seed ${SEED} \
      --out-dir runs_paper/fashion_mnist \
      --run-name fashion_main_seed${SEED} \
      --variants spline fast gram mlp --methods function branch \
      --clusters-list 16 32 --bits-list 8 6 4 3 2 \
      --width 64 --epochs 10 --finetune-epochs 20 \
      2>&1 | tee logs/paper/01_fashion_seed${SEED}.log
  done

elif [[ "${MODE}" == "width_sweep" ]]; then
  # E10 (Tier 3) — width scaling on MNIST, spline + gram, seed 42
  for W in 32 64 128 256; do
    python -m funcodekan.experiments.all_kan_datasets \
      --dataset mnist --seed 42 \
      --out-dir runs_paper/width_sweep --run-name mnist_w${W} \
      --variants spline gram --methods function branch \
      --clusters-list 16 32 --bits-list 4 \
      --width ${W} --epochs 10 --finetune-epochs 20 \
      2>&1 | tee logs/paper/01_width_${W}.log
  done
else
  echo "Modes: main | width_sweep"; exit 1
fi
