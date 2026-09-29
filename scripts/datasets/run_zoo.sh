#!/usr/bin/env bash
# Train external zoo models (requires the 'kans' package; see docs/ZOO.md).
#
# Usage:
#   bash scripts/datasets/run_zoo.sh mnist_mlps    # MLP KANs on MNIST (spline/fast/gram/pykan)
#   bash scripts/datasets/run_zoo.sh tabular       # KANELE tabular models
#   bash scripts/datasets/run_zoo.sh traffic       # VIKIN/PDR-KAN traffic models
#   bash scripts/datasets/run_zoo.sh conv_cifar10  # conv/KAGN CIFAR-10 models
set -e
MODE=${1:-mnist_mlps}

if [[ "${MODE}" == "mnist_mlps" ]]; then
  for ARCH in kan_mlp_mnist kan_mlp_mnist_fastkan kan_mlp_mnist_gram kan_mlp_mnist_pykan; do
    python -m funcodekan.experiments.zoo_train --dataset mnist --arch ${ARCH} --epochs 20
  done
elif [[ "${MODE}" == "tabular" ]]; then
  python -m funcodekan.experiments.zoo_train --dataset wine --arch kan_wine --epochs 200 --batch-size 64
  python -m funcodekan.experiments.zoo_train --dataset dry_bean --arch kan_dry_bean --epochs 100
elif [[ "${MODE}" == "traffic" ]]; then
  for ARCH in kan_traffic_3layer kan_traffic_3layer_g4 kan_traffic_2layer mlp_traffic_3layer mlp_traffic_4layer; do
    python -m funcodekan.experiments.zoo_train --dataset traffic_california --arch ${ARCH} --sensor 0 --epochs 50
  done
elif [[ "${MODE}" == "conv_cifar10" ]]; then
  for ARCH in kan_convnet_cifar10 kagn_simple_cifar10 kan_resnet_cifar10; do
    python -m funcodekan.experiments.zoo_train --dataset cifar10 --arch ${ARCH} --epochs 120 --batch-size 128
  done
else
  echo "Modes: mnist_mlps | tabular | traffic | conv_cifar10"; exit 1
fi
