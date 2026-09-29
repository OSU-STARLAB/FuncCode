#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-cifar10_smoke}
ROOT="runs_cifar"

run_one () {
  local dataset=$1
  local seed=$2
  local run_name=$3
  shift 3

  python -m funcodekan.experiments.all_kan_cifar \
    --dataset "${dataset}" \
    --seed "${seed}" \
    --out-dir "${ROOT}" \
    --run-name "${run_name}" \
    "$@"
}

if [[ "${MODE}" == "cifar10_smoke" ]]; then
  run_one cifar10 42 "cifar10_smoke_seed42" \
    --variants spline fast gram mlp \
    --methods function branch \
    --clusters-list 8 \
    --bits-list 4 \
    --epochs 1 \
    --finetune-epochs 1 \
    --width 32 \
    --subset-train 5000 \
    --subset-test 2000 \
    --batch-size 512 \
    --test-batch-size 1024

elif [[ "${MODE}" == "cifar10_main" ]]; then
  run_one cifar10 42 "cifar10_main_seed42" \
    --variants spline fast gram mlp \
    --methods function branch \
    --clusters-list 16 32 \
    --bits-list 4 \
    --epochs 50 \
    --finetune-epochs 30 \
    --width 128 \
    --batch-size 512 \
    --test-batch-size 1024

elif [[ "${MODE}" == "cifar10_multiseed" ]]; then
  for seed in 42 123 2026; do
    run_one cifar10 "${seed}" "cifar10_main_seed${seed}" \
      --variants spline fast gram mlp \
      --methods function branch \
      --clusters-list 16 32 \
      --bits-list 4 \
      --epochs 50 \
      --finetune-epochs 30 \
      --width 128 \
      --batch-size 512 \
      --test-batch-size 1024
  done

elif [[ "${MODE}" == "cifar10_fast_debug" ]]; then
  run_one cifar10 42 "cifar10_fast_debug_seed42" \
    --variants fast mlp \
    --methods function branch \
    --clusters-list 16 \
    --bits-list 4 \
    --epochs 10 \
    --finetune-epochs 10 \
    --width 64 \
    --subset-train 10000 \
    --subset-test 5000 \
    --batch-size 512 \
    --test-batch-size 1024

elif [[ "${MODE}" == "cifar100_smoke" ]]; then
  run_one cifar100 42 "cifar100_smoke_seed42" \
    --variants spline fast gram mlp \
    --methods function branch \
    --clusters-list 8 \
    --bits-list 4 \
    --epochs 1 \
    --finetune-epochs 1 \
    --width 32 \
    --subset-train 5000 \
    --subset-test 2000 \
    --batch-size 512 \
    --test-batch-size 1024

elif [[ "${MODE}" == "cifar100_main" ]]; then
  run_one cifar100 42 "cifar100_main_seed42" \
    --variants spline fast gram mlp \
    --methods function branch \
    --clusters-list 16 32 \
    --bits-list 4 \
    --epochs 80 \
    --finetune-epochs 40 \
    --width 128 \
    --batch-size 512 \
    --test-batch-size 1024

else
  echo "Unknown mode: ${MODE}"
  echo "Available: cifar10_smoke, cifar10_main, cifar10_multiseed, cifar10_fast_debug, cifar100_smoke, cifar100_main"
  exit 1
fi

echo "Finished. Summarize with:"
echo "python tools/summarize_cifar_runs.py --root runs_cifar --out runs_cifar/cifar_summary.csv"
