#!/usr/bin/env bash
# Phase 02 (E2) — tabular/synthetic suite, 5 seeds each (cheap runs).
# dry_bean and mushroom need network access (UCI) on first use.
set -e
mkdir -p logs/paper
DATASETS=${DATASETS:-"moons circle_in_circle wine dry_bean mushroom"}
for DS in ${DATASETS}; do
  for SEED in 42 123 2026 7 1337; do
    python -m funcodekan.experiments.all_kan_datasets \
      --dataset ${DS} --seed ${SEED} \
      --out-dir runs_paper/tabular --run-name ${DS}_seed${SEED} \
      --variants spline fast gram mlp --methods function branch \
      --clusters-list 8 16 32 --bits-list 8 6 4 3 2 \
      --width 32 --epochs 40 --finetune-epochs 30 \
      --batch-size 256 --test-batch-size 1024 \
      2>&1 | tee logs/paper/02_${DS}_seed${SEED}.log
  done
done
