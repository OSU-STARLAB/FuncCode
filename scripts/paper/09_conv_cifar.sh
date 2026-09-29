#!/usr/bin/env bash
# Phase 09 -- convolutional KAGN backbones on CIFAR-10 / CIFAR-100.
#
# Replaces the flattened-image CIFAR protocol (phase 03) for the main table.
# Designed for a multi-GPU host: whole configs are dispatched to free GPU slots
# by a queue. Set GPUS and SLOTS_PER_GPU for your machine.
#
# Usage:
#   bash scripts/paper/09_conv_cifar.sh smoke     # ~10 min, wiring check
#   bash scripts/paper/09_conv_cifar.sh tier1     # ~14 h, the main table
#   bash scripts/paper/09_conv_cifar.sh tier2     # ~27 h, seeds 123 + 2026
#   bash scripts/paper/09_conv_cifar.sh tier3     # ~15 h, ablations
#   bash scripts/paper/09_conv_cifar.sh tables    # CPU, rerun anytime
#   bash scripts/paper/09_conv_cifar.sh status    # poll a running queue
set -euo pipefail

MODE=${1:-tier1}
GPUS=${GPUS:-"0 1"}
# Two jobs per card: the 8-layer net peaks at ~1.2 GiB at batch 128 and is
# latency-bound, so 2 slots/GPU measured 1.35x the throughput of 1 (3 gave no
# further gain). 4 concurrent jobs x 3 dataloader workers fits 16 cores.
SLOTS_PER_GPU=${SLOTS_PER_GPU:-2}
OUT_DIR=${OUT_DIR:-runs_conv_cifar}
# Smoke uses a toy preset; keep it out of OUT_DIR or summarize_conv_cifar.py
# rglobs its summary.csv into the paper tables.
SMOKE_DIR=${SMOKE_DIR:-runs_conv_smoke}
DATA_ROOT=${DATA_ROOT:-./data}
PLAN_DIR=scripts/paper/plans
QUEUE_DIR=${OUT_DIR}/_queue

mkdir -p logs/paper "${PLAN_DIR}" "${OUT_DIR}" "${SMOKE_DIR}"

case "${MODE}" in
  smoke)
    echo "== smoke: 2 epochs, tiny preset, both datasets =="
    for DS in cifar10 cifar100; do
      python -m funcodekan.experiments.conv_cifar \
        --dataset "${DS}" --preset kagn_tiny_smoke --smoke \
        --data-root "${DATA_ROOT}" --out-dir "${SMOKE_DIR}/_smoke" \
        --methods function branch --baselines uniform lsq pq prune \
        2>&1 | tee "logs/paper/09_smoke_${DS}.log"
    done
    echo "== smoke OK =="
    ;;

  tier1|tier2|tier3)
    python tools/make_conv_plan.py --tier "${MODE}" \
      --out "${PLAN_DIR}/cifar_${MODE}.json" \
      --out-dir "${OUT_DIR}" --data-root "${DATA_ROOT}"
    python tools/gpu_queue.py --plan "${PLAN_DIR}/cifar_${MODE}.json" \
      --queue-dir "${QUEUE_DIR}" --gpus ${GPUS} --slots-per-gpu "${SLOTS_PER_GPU}" \
      2>&1 | tee -a "logs/paper/09_queue_${MODE}.log"
    bash "$0" tables
    ;;

  tables)
    python tools/summarize_conv_cifar.py --root "${OUT_DIR}" \
      --out-dir "${OUT_DIR}/tables" 2>&1 | tee logs/paper/09_tables.log
    echo "== tables in ${OUT_DIR}/tables =="
    ;;

  status)
    python tools/gpu_queue.py --plan "${PLAN_DIR}/cifar_${2:-tier1}.json" \
      --queue-dir "${QUEUE_DIR}" --status-only
    ;;

  *)
    echo "Modes: smoke | tier1 | tier2 | tier3 | tables | status"; exit 1
    ;;
esac
