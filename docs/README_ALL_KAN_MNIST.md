# All-KAN MNIST Evaluation Patch

This patch adds a unified MNIST evaluation pipeline for multiple KAN variants:

```text
spline  : EfficientKAN-style B-spline basis
fast    : FastKAN-style RBF basis
gram    : GRAM/KAGN-style polynomial basis
mlp     : standard MLP reference baseline
```

## New files

```text
src/all_kan_variants.py
src/all_kan_compression.py
src/experiment_all_kan_mnist.py
tools/summarize_all_kan_mnist.py
tools/make_all_kan_latex_table.py
run_all_kan_mnist.sh
```

## What the experiment does

For each selected variant:

```text
1. Train dense FP32 model from scratch on MNIST.
2. Save dense checkpoint.
3. Compress dense model using:
   - coefficient-space clustering
   - function-space clustering
   - branch-aware codebooks
4. Fine-tune compressed codebooks.
5. Apply hardware-aware W8/W6/W4/W3/W2 codebook quantization.
6. Save summary.csv.
```

For the MLP baseline, the script trains dense FP32 and evaluates simple uniform PTQ over linear weights.

## Smoke test

```bash
bash run_all_kan_mnist.sh smoke
```

## Main run

```bash
bash run_all_kan_mnist.sh main
```

## Variant-specific runs

```bash
bash run_all_kan_mnist.sh spline
bash run_all_kan_mnist.sh fast
bash run_all_kan_mnist.sh gram
bash run_all_kan_mnist.sh mlp
```

## Recommended first full experiment

```bash
python -m src.experiment_all_kan_mnist \
  --variants spline fast gram mlp \
  --methods function branch \
  --clusters-list 16 32 \
  --bits-list 8 6 4 3 2 \
  --epochs 10 \
  --finetune-epochs 20 \
  --width 64 \
  --batch-size 1024 \
  --test-batch-size 2048 \
  --run-name all_kan_mnist_main
```

## Summarize

```bash
python tools/summarize_all_kan_mnist.py \
  --root runs/all_kan_mnist_main \
  --out runs/all_kan_mnist_main/combined_summary.csv
```

## LaTeX table

```bash
python tools/make_all_kan_latex_table.py \
  --csv runs/all_kan_mnist_main/combined_summary.csv \
  --out-tex runs/all_kan_mnist_main/all_kan_results.tex
```

## Important notes

This patch uses a unified direct-weight implementation for all KAN variants.
Every KAN layer stores an edge tensor:

```text
weight shape = [out_features, in_features, basis_dim + 1]
```

The last channel is treated as the base branch.
The previous channels are treated as the basis/spline branch.

This gives a fair shared interface for compression:

```text
shared codebook:
    clusters full edge vectors

branch-aware codebook:
    clusters basis branch and base branch separately
```
