#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-smoke}
COMMON="--epochs 10 --finetune-epochs 20 --width 64 --bits-list 8 6 4 3 2 --batch-size 1024 --test-batch-size 2048"

if [[ "${MODE}" == "smoke" ]]; then
  python -m funcodekan.experiments.mnist \
    --run-name smoke_func_k8 \
    --cluster-method function \
    --epochs 1 \
    --finetune-epochs 1 \
    --width 32 \
    --clusters 8 \
    --bits-list 8 4 \
    --function-samples 64 \
    --function-domain grid \
    --include-base-in-function \
    --batch-size 512 \
    --test-batch-size 1024

elif [[ "${MODE}" == "compare_k16" ]]; then
  python -m funcodekan.experiments.mnist --run-name coeff_k16 --cluster-method coefficient --clusters 16 ${COMMON}
  python -m funcodekan.experiments.mnist --run-name func_k16 --cluster-method function --clusters 16 --function-samples 128 --function-domain grid --include-base-in-function ${COMMON}

elif [[ "${MODE}" == "compare_k32" ]]; then
  python -m funcodekan.experiments.mnist --run-name coeff_k32 --cluster-method coefficient --clusters 32 ${COMMON}
  python -m funcodekan.experiments.mnist --run-name func_k32 --cluster-method function --clusters 32 --function-samples 128 --function-domain grid --include-base-in-function ${COMMON}

elif [[ "${MODE}" == "compare_k64" ]]; then
  python -m funcodekan.experiments.mnist --run-name coeff_k64 --cluster-method coefficient --clusters 64 ${COMMON}
  python -m funcodekan.experiments.mnist --run-name func_k64 --cluster-method function --clusters 64 --function-samples 128 --function-domain grid --include-base-in-function ${COMMON}

elif [[ "${MODE}" == "compare_k128" ]]; then
  python -m funcodekan.experiments.mnist --run-name coeff_k128 --cluster-method coefficient --clusters 128 ${COMMON}
  python -m funcodekan.experiments.mnist --run-name func_k128 --cluster-method function --clusters 128 --function-samples 128 --function-domain grid --include-base-in-function ${COMMON}

elif [[ "${MODE}" == "all" ]]; then
  bash "$0" compare_k16
  bash "$0" compare_k32
  bash "$0" compare_k64
  bash "$0" compare_k128
else
  echo "Unknown mode: ${MODE}"
  exit 1
fi
