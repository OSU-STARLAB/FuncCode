# Extended datasets and the external model zoo

## Dataset registry (`funcodekan/data/registry.py`)

Auto-fetching loaders (dataset definitions carried over unchanged from the
upstream KAN-datasets collection):

| Key | Modality | Classes | Needs network |
|---|---|---|---|
| `moons`, `circle_in_circle` | synthetic 2-D | 2 | no (sklearn) |
| `wine` | tabular | 3 | no (sklearn) |
| `dry_bean` | tabular | 7 | yes (ucimlrepo/UCI) |
| `mushroom` | tabular | 2 | yes (UCI) |
| `mnist`, `fashion_mnist`, `cifar10`, `cifar100` | image | 10/10/10/100 | first use |
| `tiny_imagenet` | image 64x64 | 200 | first use (cs231n) |
| `traffic_california` | time-series regression | — | first use (LSTNet) |

`imagenet` and several others are registered with manual-download guidance
(they raise `NotAvailable` with instructions). Smoke-test what your machine
can reach with `python -m funcodekan.data.registry` (optionally pass keys).

Extra pip deps for the auto-fetch paths: `ucimlrepo requests openpyxl`
(install via `pip install -e ".[datasets]"`).

## Bundles (`funcodekan/data/bundles.py`)

`get_dataset_bundle(key, ...)` converts any classification dataset into the
standard bundle every experiment driver consumes. Guarantees:

- `mnist` / `cifar10` / `cifar100` (flattened) delegate to the original
  paper-verified loaders — results stay comparable with REPRODUCE.md.
- Tabular data is standardized with train-split statistics only and split
  train/val/test stratified with a fixed seed (deterministic).
- `flatten=False` returns [C, H, W] tensors for convolutional zoo models.
- `get_traffic_regression_loaders(...)` implements the VIKIN / PDR-KAN
  protocol (72h → 96h windows, chronological 7:2:1, per-sensor max scaling).

## Running FuncCode compression on the new datasets

`funcodekan/experiments/all_kan_datasets.py` is a 15-line delta of the
verified `all_kan_mnist.py` (data loading only — diff it to confirm). Same
protocol, stages, storage accounting, and summary format, so all existing
tools (`summarize_all_kan_mnist.py`, LaTeX table makers, HW estimator) work
on its outputs unchanged.

```bash
bash scripts/datasets/run_all_kan_datasets.sh list
bash scripts/datasets/run_all_kan_datasets.sh smoke          # moons, minutes
bash scripts/datasets/run_all_kan_datasets.sh tabular_main   # 5 tabular sets
bash scripts/datasets/run_all_kan_datasets.sh fashion_main   # MNIST protocol
bash scripts/datasets/run_all_kan_datasets.sh tiny_imagenet
```

## External model zoo (`funcodekan/models/zoo.py`)

Carried over from the upstream SparseKAN model definitions with the model code
unchanged; only the import of the external **`kans` package** is guarded.

**The `kans` package is required and was not part of the upload.** It must
provide: `kans.efficient_kan` (KANLinear, KAN), `kans.fastkan`
(FastKANLayer), `kans.kagn_kagn_conv` (GRAMLayer, KAGN, KAGNConv2DLayer),
`kans.pykan.KANLayer`, `kans.kan_convolution.KANConvFast`, and
`kans.regularization`. Vendor your `kans/` folder at the repository root (or
pip-install it). Until then `funcodekan.models.zoo` imports fine with
`KANS_AVAILABLE = False`, and instantiating any zoo model raises an
informative error. The two HuggingFace transfer helpers additionally need
the original repo's `util.load_hf_weights_into_model`.

Train zoo models with `funcodekan/experiments/zoo_train.py` (classification
accuracy, or MSE for `traffic_california`):

```bash
python -m funcodekan.experiments.zoo_train --dataset mnist --arch kan_mlp_mnist
bash scripts/datasets/run_zoo.sh mnist_mlps | tabular | traffic | conv_cifar10
```

Arch names are the `create_model` dispatcher keys (e.g. `kan_mlp_mnist`,
`kan_mlp_mnist_fastkan`, `kagn_simple_cifar10`, `kan_resnet_cifar100`,
`kagn_v2`, `kan_wine`, `kan_traffic_3layer`).

## Zoo models vs FuncCode compression

FuncCode's codebook compression operates on fully connected KAN layers via
the basis/base interface (`DirectKANVariant`). Correspondences:

| Zoo model | FuncCode equivalent |
|---|---|
| `KAN_MLP_MNIST` (efficient-kan 784-64-10) | `DirectKANVariant("spline", 784, 64, 10)` |
| `KAN_MLP_MNIST_FASTKAN` | `DirectKANVariant("fast", ...)` |
| `KAN_MLP_MNIST_GRAM` | `DirectKANVariant("gram", ...)` |
| `KAN_Wine` [13,4,3], `KAN_DryBean` [16,2,7] | spline variant with matching dims* |

So "FuncCode on dataset X" experiments should use
`funcodekan.experiments.all_kan_datasets` (trains FuncCode's own variants and
compresses them); the zoo trainer provides the external reference numbers.
*The current `DirectKANVariant` uses one hidden width; multi-width stacks and
direct compression of pretrained zoo checkpoints (weight import from
efficient-kan layers) are natural next extensions. Convolutional zoo models
(ConvNet/ResNet/VGG-KAGN) are supported for training/evaluation only —
compressing conv KAN layers is future work.
