#!/usr/bin/env bash
# Phase 08 (E8) — Section-4 analysis: function- vs coefficient-space
# redundancy across datasets and basis families.
set -e
mkdir -p logs/paper
for DS in mnist fashion_mnist wine cifar10; do
  for V in spline fast gram; do
    python tools/analyze_edge_redundancy.py \
      --dataset ${DS} --variant ${V} --epochs 5 --width 64 \
      --ks 2 4 8 16 32 64 128 \
      --out-dir runs_paper/redundancy \
      2>&1 | tee logs/paper/08_${DS}_${V}.log
  done
done
echo "Spectrum / inertia / effective-rank CSVs in runs_paper/redundancy/"
