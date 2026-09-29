#!/usr/bin/env bash
# Phase 00 — timed smoke of every experiment path. Use the printed wall-times
# to calibrate the wall-clock estimates in REPRODUCE.md for your own GPU.
set -e
mkdir -p logs/paper
t() { local name=$1; shift; local s=$SECONDS
      "$@" > "logs/paper/smoke_${name}.log" 2>&1
      echo "smoke ${name}: $((SECONDS-s))s"; }

t unit_tests        python -m pytest tests/ -q
t mnist_all_kan     bash scripts/mnist/run_all_kan.sh smoke
t mnist_ablation    bash scripts/mnist/run_spline_baseline.sh smoke
t variant_ablation  bash scripts/mnist/run_variant_ablations.sh smoke
t tabular           bash scripts/datasets/run_all_kan_datasets.sh smoke
t cifar10           bash scripts/cifar/run_cifar_experiments.sh cifar10_smoke
t cifar100          bash scripts/cifar/run_cifar_experiments.sh cifar100_smoke
t redundancy        python tools/analyze_edge_redundancy.py --dataset moons \
                      --variant spline --epochs 1 --width 16 --samples 32 \
                      --ks 2 4 8 --out-dir runs_paper/redundancy_smoke
echo "All smokes passed. Calibrate the REPRODUCE.md time estimates with the times above."
