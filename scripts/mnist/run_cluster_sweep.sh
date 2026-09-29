#!/usr/bin/env bash
set -euo pipefail

for K in 4 8 16 32; do
  python -m funcodekan.experiments.mnist \
    --run-name sweep_mnist_k${K} \
    --epochs 10 \
    --finetune-epochs 3 \
    --width 64 \
    --clusters ${K} \
    --bits-list 8 6 4 3 2 \
    --batch-size 1024 \
    --test-batch-size 2048
done
