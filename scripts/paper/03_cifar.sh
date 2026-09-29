#!/usr/bin/env bash
# Phase 03 (E3, E9) — CIFAR-10 3 seeds + CIFAR-100.
# Usage: bash scripts/paper/03_cifar.sh [main|extra_seeds]
set -e
MODE=${1:-main}
mkdir -p logs/paper

if [[ "${MODE}" == "main" ]]; then
  bash scripts/cifar/run_cifar_experiments.sh cifar10_multiseed \
    2>&1 | tee logs/paper/03_cifar10_multiseed.log
  bash scripts/cifar/run_cifar_experiments.sh cifar100_main \
    2>&1 | tee logs/paper/03_cifar100_seed42.log

elif [[ "${MODE}" == "extra_seeds" ]]; then
  # E9 — upgrade CIFAR-100 to 3 seeds (mirrors cifar100_main hyperparameters)
  for SEED in 123 2026; do
    python -m funcodekan.experiments.all_kan_cifar \
      --dataset cifar100 --seed ${SEED} \
      --out-dir runs_cifar --run-name cifar100_main_seed${SEED} \
      --variants spline fast gram mlp --methods function branch \
      --clusters-list 16 32 --bits-list 4 \
      --epochs 80 --finetune-epochs 40 --width 128 \
      --batch-size 512 --test-batch-size 1024 \
      2>&1 | tee logs/paper/03_cifar100_seed${SEED}.log
  done
else
  echo "Modes: main | extra_seeds"; exit 1
fi
