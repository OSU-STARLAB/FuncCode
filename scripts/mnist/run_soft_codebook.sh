#!/usr/bin/env bash
set -e

# Differentiable soft-to-hard codebook learning experiments.
# Usage:
#   bash scripts/mnist/run_soft_codebook.sh k32
#   bash scripts/mnist/run_soft_codebook.sh k16
#   bash scripts/mnist/run_soft_codebook.sh sweep

MODE=${1:-k32}

if [ "$MODE" = "k16" ]; then
  python -m funcodekan.experiments.soft_codebook \
    --run-name soft_branch_index_k16 \
    --epochs 10 --soft-epochs 10 --hard-finetune-epochs 20 \
    --width 64 --spline-clusters 16 \
    --spline-method function --function-domain grid --function-samples 128 \
    --temp-start 2.0 --temp-end 0.25 \
    --distill-alpha 0.5 --entropy-lambda 1e-3 --balance-lambda 1e-3 \
    --bits-list 8 6 4 3 2 \
    --batch-size 1024 --test-batch-size 2048
elif [ "$MODE" = "k32" ]; then
  python -m funcodekan.experiments.soft_codebook \
    --run-name soft_branch_index_k32 \
    --epochs 10 --soft-epochs 10 --hard-finetune-epochs 20 \
    --width 64 --spline-clusters 32 \
    --spline-method function --function-domain grid --function-samples 128 \
    --temp-start 2.0 --temp-end 0.25 \
    --distill-alpha 0.5 --entropy-lambda 1e-3 --balance-lambda 1e-3 \
    --bits-list 8 6 4 3 2 \
    --batch-size 1024 --test-batch-size 2048
elif [ "$MODE" = "sweep" ]; then
  bash "$0" k16
  bash "$0" k32
else
  echo "Unknown mode: $MODE"
  echo "Choose one of: k16, k32, sweep"
  exit 1
fi
