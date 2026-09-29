# Multi-Seed Statistics Patch for FuncCode-KAN

This patch adds multi-seed experiment scripts and summary tools.

## Progress Tracker

```text
[done] Single-seed All-KAN MNIST evaluation
[done] Cross-KAN W4 summary table
[done] Paper draft with citations and figure placeholders
[now] Add multi-seed statistics
[next] Run 3-seed smoke validation
[next] Run 3-seed main MNIST experiment
[next] Generate mean/std paper table
[next] Update paper Results section with mean ± std
```

## New files

```text
scripts/run_multiseed_all_kan_mnist.sh
tools/summarize_multiseed_all_kan.py
tools/make_multiseed_latex_table.py
tools/select_best_multiseed_methods.py
```

## Recommended first smoke test

```bash
bash scripts/run_multiseed_all_kan_mnist.sh smoke
```

This runs seeds:

```text
42, 123, 2026
```

with tiny settings:

```text
epochs = 1
finetune_epochs = 1
clusters = 8
bits = W4
variants = spline, fast, gram, mlp
```

## Main 3-seed experiment

```bash
bash scripts/run_multiseed_all_kan_mnist.sh main
```

This runs:

```text
seeds = 42, 123, 2026
variants = spline, fast, gram, mlp
methods = function, branch
clusters = 16, 32
bits = W4
epochs = 10
finetune_epochs = 20
```

## Longer main experiment with all bitwidths

```bash
bash scripts/run_multiseed_all_kan_mnist.sh main_all_bits
```

This runs:

```text
bits = W8, W6, W4, W3, W2
```

## Summarize all seeds

After running multi-seed experiments:

```bash
python tools/summarize_multiseed_all_kan.py \
  --root runs_multiseed/all_kan_mnist_main \
  --out runs_multiseed/all_kan_mnist_main/multiseed_summary.csv
```

## Select best method per variant

```bash
python tools/select_best_multiseed_methods.py \
  --csv runs_multiseed/all_kan_mnist_main/multiseed_summary.csv \
  --out runs_multiseed/all_kan_mnist_main/best_methods.csv
```

## Generate LaTeX table

```bash
python tools/make_multiseed_latex_table.py \
  --csv runs_multiseed/all_kan_mnist_main/multiseed_summary.csv \
  --out-tex runs_multiseed/all_kan_mnist_main/multiseed_table.tex \
  --caption "Multi-seed MNIST results across KAN variants under W4 codebook quantization." \
  --label "tab:multiseed_mnist"
```

## Paper reporting format

Use:

```text
mean ± std over 3 seeds
```

Example:

```latex
SplineKAN branch $K_s=32,K_b=16$ & $95.46 \pm 0.12$ & $56.47$ & $31.64\times$
```

## Important note

Storage and compression are deterministic for a fixed architecture and compression setting, but they are still summarized with mean/std for consistency. Accuracy is the main statistic.
