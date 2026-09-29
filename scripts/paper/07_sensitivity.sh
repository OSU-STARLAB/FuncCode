#!/usr/bin/env bash
# Phase 07 (E7) — sensitivity studies on MNIST SplineKAN, seed 42.
set -e
mkdir -p logs/paper
COMMON="--out-dir runs_paper/sensitivity --seed 42 --epochs 10 --finetune-epochs 20 --bits-list 4"

# (a) K sweep, function-space
for K in 4 8 16 32 64 128; do
  python -m funcodekan.experiments.mnist ${COMMON} \
    --run-name sens_k${K} --cluster-method function --clusters ${K} \
    2>&1 | tee logs/paper/07_k${K}.log
done

# (b) signature length (function-samples)
for S in 32 128 512; do
  python -m funcodekan.experiments.mnist ${COMMON} \
    --run-name sens_samples${S} --cluster-method function --clusters 32 \
    --function-samples ${S} \
    2>&1 | tee logs/paper/07_samples${S}.log
done

# (c) sampling domain: fixed grid vs activation statistics
for D in grid activation; do
  python -m funcodekan.experiments.mnist ${COMMON} \
    --run-name sens_domain_${D} --cluster-method function --clusters 32 \
    --function-domain ${D} \
    2>&1 | tee logs/paper/07_domain_${D}.log
done
