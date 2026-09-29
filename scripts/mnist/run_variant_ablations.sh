#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-smoke}
ROOT="runs_variant_ablations"

run_one () {
  local seed=$1
  local run_name=$2
  shift 2

  echo "======================================================================"
  echo "Variant ablation run: seed=${seed}, run_name=${run_name}"
  echo "======================================================================"

  python -m funcodekan.experiments.variant_ablation \
    --seed "${seed}" \
    --out-dir "${ROOT}" \
    --run-name "${run_name}" \
    "$@"
}

if [[ "${MODE}" == "smoke" ]]; then
  run_one 42 "smoke_variant_ablations_seed42" \
    --variants spline fast gram \
    --methods srb index soft \
    --clusters-list 8 \
    --residual-fractions 0.25 \
    --bits-list 4 \
    --epochs 1 \
    --finetune-epochs 1 \
    --soft-epochs 1 \
    --width 32 \
    --batch-size 512 \
    --test-batch-size 1024

elif [[ "${MODE}" == "main" ]]; then
  run_one 42 "variant_ablations_main_seed42" \
    --variants spline fast gram \
    --methods srb index soft \
    --clusters-list 16 32 \
    --residual-fractions 0.10 0.25 0.50 \
    --bits-list 4 \
    --epochs 10 \
    --finetune-epochs 20 \
    --soft-epochs 20 \
    --width 64 \
    --batch-size 1024 \
    --test-batch-size 2048

elif [[ "${MODE}" == "multiseed" ]]; then
  for seed in 42 123 2026; do
    run_one "${seed}" "variant_ablations_main_seed${seed}" \
      --variants spline fast gram \
      --methods srb index soft \
      --clusters-list 16 32 \
      --residual-fractions 0.10 0.25 0.50 \
      --bits-list 4 \
      --epochs 10 \
      --finetune-epochs 20 \
      --soft-epochs 20 \
      --width 64 \
      --batch-size 1024 \
      --test-batch-size 2048
  done

elif [[ "${MODE}" == "fast_debug" ]]; then
  run_one 42 "fast_ablation_debug_seed42" \
    --variants fast \
    --methods srb index soft \
    --clusters-list 16 \
    --residual-fractions 0.25 0.50 \
    --bits-list 4 \
    --epochs 5 \
    --finetune-epochs 10 \
    --soft-epochs 10 \
    --width 64 \
    --batch-size 1024 \
    --test-batch-size 2048

else
  echo "Unknown mode: ${MODE}"
  echo "Available: smoke, main, multiseed, fast_debug"
  exit 1
fi

echo "======================================================================"
echo "Finished. Summarize with:"
echo "python tools/summarize_variant_ablations.py --root runs_variant_ablations --out runs_variant_ablations/variant_ablation_summary.csv"
echo "======================================================================"
