#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-main}

if [[ "${MODE}" == "smoke" ]]; then
  python -m funcodekan.experiments.mnist \
    --run-name smoke_mnist_splinekan \
    --epochs 1 \
    --finetune-epochs 1 \
    --width 32 \
    --clusters 8 \
    --bits-list 8 4 \
    --batch-size 512 \
    --test-batch-size 1024
elif [[ "${MODE}" == "main" ]]; then
  python -m funcodekan.experiments.mnist \
    --run-name main_mnist_splinekan \
    --epochs 10 \
    --finetune-epochs 3 \
    --width 64 \
    --clusters 16 \
    --bits-list 8 6 4 3 2 \
    --batch-size 1024 \
    --test-batch-size 2048
else
  echo "Unknown mode: ${MODE}"
  echo "Use: bash scripts/mnist/run_spline_baseline.sh smoke"
  echo " or: bash scripts/mnist/run_spline_baseline.sh main"
  exit 1
fi
