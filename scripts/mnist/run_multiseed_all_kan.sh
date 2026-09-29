#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-smoke}

SEEDS=(42 123 2026)

ROOT="runs_multiseed"

run_one () {
  local seed=$1
  local run_name=$2
  shift 2

  echo "======================================================================"
  echo "Running seed ${seed}, run_name=${run_name}"
  echo "======================================================================"

  python -m funcodekan.experiments.all_kan_mnist \
    --seed "${seed}" \
    --out-dir "${ROOT}" \
    --run-name "${run_name}" \
    "$@"
}

if [[ "${MODE}" == "smoke" ]]; then
  for seed in "${SEEDS[@]}"; do
    run_one "${seed}" "smoke_seed${seed}" \
      --variants spline fast gram mlp \
      --methods function branch \
      --clusters-list 8 \
      --bits-list 4 \
      --epochs 1 \
      --finetune-epochs 1 \
      --width 32 \
      --batch-size 512 \
      --test-batch-size 1024
  done

elif [[ "${MODE}" == "main" ]]; then
  for seed in "${SEEDS[@]}"; do
    run_one "${seed}" "all_kan_mnist_main_seed${seed}" \
      --variants spline fast gram mlp \
      --methods function branch \
      --clusters-list 16 32 \
      --bits-list 4 \
      --epochs 10 \
      --finetune-epochs 20 \
      --width 64 \
      --batch-size 1024 \
      --test-batch-size 2048
  done

elif [[ "${MODE}" == "main_all_bits" ]]; then
  for seed in "${SEEDS[@]}"; do
    run_one "${seed}" "all_kan_mnist_allbits_seed${seed}" \
      --variants spline fast gram mlp \
      --methods function branch \
      --clusters-list 16 32 \
      --bits-list 8 6 4 3 2 \
      --epochs 10 \
      --finetune-epochs 20 \
      --width 64 \
      --batch-size 1024 \
      --test-batch-size 2048
  done

elif [[ "${MODE}" == "spline_main" ]]; then
  for seed in "${SEEDS[@]}"; do
    run_one "${seed}" "spline_main_seed${seed}" \
      --variants spline \
      --methods coefficient function branch \
      --clusters-list 16 32 \
      --bits-list 4 \
      --epochs 10 \
      --finetune-epochs 20 \
      --width 64 \
      --batch-size 1024 \
      --test-batch-size 2048
  done

else
  echo "Unknown mode: ${MODE}"
  echo "Available modes: smoke, main, main_all_bits, spline_main"
  exit 1
fi

echo "======================================================================"
echo "Multi-seed runs finished."
echo "Now summarize, for example:"
echo "python tools/summarize_multiseed_all_kan.py --root runs_multiseed --out runs_multiseed/multiseed_summary.csv"
echo "======================================================================"
