#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-smoke}
COMMON="--epochs 10 --finetune-epochs 20 --width 64 --bits-list 8 6 4 3 2 --batch-size 1024 --test-batch-size 2048"

if [[ "${MODE}" == "smoke" ]]; then
  python -m funcodekan.experiments.mnist \
    --run-name smoke_branch_index_k8 \
    --cluster-method branch_index \
    --branch-spline-method function \
    --branch-spline-clusters 8 \
    --branch-function-samples 64 \
    --branch-function-domain grid \
    --epochs 1 \
    --finetune-epochs 1 \
    --width 32 \
    --clusters 8 \
    --bits-list 8 4 \
    --batch-size 512 \
    --test-batch-size 1024

elif [[ "${MODE}" == "compare_k16" ]]; then
  python -m funcodekan.experiments.mnist --run-name func_k16_ref --cluster-method function --clusters 16 --function-samples 128 --function-domain grid --include-base-in-function ${COMMON}
  python -m funcodekan.experiments.mnist --run-name branch_k16_s16_b8_ref --cluster-method branch --clusters 16 --branch-spline-method function --branch-spline-clusters 16 --branch-base-clusters 8 --branch-function-samples 128 --branch-function-domain grid ${COMMON}
  python -m funcodekan.experiments.mnist --run-name branch_index_k16 --cluster-method branch_index --clusters 16 --branch-spline-method function --branch-spline-clusters 16 --branch-function-samples 128 --branch-function-domain grid ${COMMON}

elif [[ "${MODE}" == "compare_k32" ]]; then
  python -m funcodekan.experiments.mnist --run-name func_k32_ref --cluster-method function --clusters 32 --function-samples 128 --function-domain grid --include-base-in-function ${COMMON}
  python -m funcodekan.experiments.mnist --run-name branch_k32_s32_b16_ref --cluster-method branch --clusters 32 --branch-spline-method function --branch-spline-clusters 32 --branch-base-clusters 16 --branch-function-samples 128 --branch-function-domain grid ${COMMON}
  python -m funcodekan.experiments.mnist --run-name branch_index_k32 --cluster-method branch_index --clusters 32 --branch-spline-method function --branch-spline-clusters 32 --branch-function-samples 128 --branch-function-domain grid ${COMMON}

elif [[ "${MODE}" == "compare_k64" ]]; then
  python -m funcodekan.experiments.mnist --run-name func_k64_ref --cluster-method function --clusters 64 --function-samples 128 --function-domain grid --include-base-in-function ${COMMON}
  python -m funcodekan.experiments.mnist --run-name branch_k64_s64_b16_ref --cluster-method branch --clusters 64 --branch-spline-method function --branch-spline-clusters 64 --branch-base-clusters 16 --branch-function-samples 128 --branch-function-domain grid ${COMMON}
  python -m funcodekan.experiments.mnist --run-name branch_index_k64 --cluster-method branch_index --clusters 64 --branch-spline-method function --branch-spline-clusters 64 --branch-function-samples 128 --branch-function-domain grid ${COMMON}

elif [[ "${MODE}" == "all" ]]; then
  bash "$0" compare_k16
  bash "$0" compare_k32
  bash "$0" compare_k64
else
  echo "Unknown mode: ${MODE}"
  echo "Available: smoke, compare_k16, compare_k32, compare_k64, all"
  exit 1
fi
