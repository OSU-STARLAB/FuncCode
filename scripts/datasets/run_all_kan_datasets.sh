#!/usr/bin/env bash
# FuncCode compression experiments on the extended dataset registry.
#
# Usage:
#   bash scripts/datasets/run_all_kan_datasets.sh smoke          # tiny run, moons
#   bash scripts/datasets/run_all_kan_datasets.sh tabular_main   # moons, circles, wine, dry_bean, mushroom
#   bash scripts/datasets/run_all_kan_datasets.sh fashion_main   # Fashion-MNIST, full protocol
#   bash scripts/datasets/run_all_kan_datasets.sh tiny_imagenet  # Tiny ImageNet (long)
#   bash scripts/datasets/run_all_kan_datasets.sh list           # show available dataset keys
set -e
MODE=${1:-smoke}
OUT=runs_datasets

if [[ "${MODE}" == "list" ]]; then
  python -c "from funcodekan.data.bundles import available_classification_datasets as f; print('\n'.join(f()))"
  exit 0
fi

if [[ "${MODE}" == "smoke" ]]; then
  python -m funcodekan.experiments.all_kan_datasets \
    --dataset moons --out-dir ${OUT} --run-name smoke_moons \
    --variants spline fast gram mlp --methods function branch \
    --clusters-list 8 --bits-list 4 \
    --width 16 --epochs 2 --finetune-epochs 2 --function-samples 32 \
    --batch-size 256 --test-batch-size 512

elif [[ "${MODE}" == "tabular_main" ]]; then
  # Small tabular/synthetic sets: widths kept modest; full method protocol.
  for DS in moons circle_in_circle wine dry_bean mushroom; do
    python -m funcodekan.experiments.all_kan_datasets \
      --dataset ${DS} --out-dir ${OUT} \
      --variants spline fast gram mlp --methods function branch \
      --clusters-list 16 32 --bits-list 8 6 4 3 2 \
      --width 32 --epochs 40 --finetune-epochs 30 \
      --batch-size 256 --test-batch-size 1024
  done

elif [[ "${MODE}" == "fashion_main" ]]; then
  # Same protocol as the verified MNIST main experiment.
  python -m funcodekan.experiments.all_kan_datasets \
    --dataset fashion_mnist --out-dir ${OUT} \
    --variants spline fast gram mlp --methods function branch \
    --clusters-list 16 32 --bits-list 8 6 4 3 2 \
    --width 64 --epochs 10 --finetune-epochs 20

elif [[ "${MODE}" == "tiny_imagenet" ]]; then
  # 64x64x3 flattened (12288-dim). Long run; consider fewer variants first.
  python -m funcodekan.experiments.all_kan_datasets \
    --dataset tiny_imagenet --out-dir ${OUT} \
    --variants spline fast gram mlp --methods function branch \
    --clusters-list 16 32 --bits-list 8 6 4 3 2 \
    --width 128 --epochs 60 --finetune-epochs 40 \
    --batch-size 512 --test-batch-size 1024

else
  echo "Unknown mode: ${MODE}"
  echo "Modes: smoke | tabular_main | fashion_main | tiny_imagenet | list"
  exit 1
fi
