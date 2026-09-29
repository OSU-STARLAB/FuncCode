#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-smoke}

COMMON="--epochs 10 --finetune-epochs 20 --width 64 --clusters-list 16 32 --bits-list 8 6 4 3 2 --batch-size 1024 --test-batch-size 2048"

if [[ "${MODE}" == "smoke" ]]; then
  python -m funcodekan.experiments.all_kan_mnist \
    --run-name smoke_all_kan \
    --variants spline fast gram mlp \
    --methods function branch \
    --clusters-list 8 \
    --bits-list 8 4 \
    --epochs 1 \
    --finetune-epochs 1 \
    --width 32 \
    --batch-size 512 \
    --test-batch-size 1024

elif [[ "${MODE}" == "main" ]]; then
  python -m funcodekan.experiments.all_kan_mnist \
    --run-name all_kan_mnist_main \
    --variants spline fast gram mlp \
    --methods function branch \
    ${COMMON}

elif [[ "${MODE}" == "spline" ]]; then
  python -m funcodekan.experiments.all_kan_mnist \
    --run-name all_kan_spline \
    --variants spline \
    --methods coefficient function branch \
    ${COMMON}

elif [[ "${MODE}" == "fast" ]]; then
  python -m funcodekan.experiments.all_kan_mnist \
    --run-name all_kan_fast \
    --variants fast \
    --methods coefficient function branch \
    ${COMMON}

elif [[ "${MODE}" == "gram" ]]; then
  python -m funcodekan.experiments.all_kan_mnist \
    --run-name all_kan_gram \
    --variants gram \
    --methods coefficient function branch \
    ${COMMON}

elif [[ "${MODE}" == "mlp" ]]; then
  python -m funcodekan.experiments.all_kan_mnist \
    --run-name all_kan_mlp \
    --variants mlp \
    --methods none \
    ${COMMON}

else
  echo "Unknown mode: ${MODE}"
  echo "Available: smoke, main, spline, fast, gram, mlp"
  exit 1
fi
