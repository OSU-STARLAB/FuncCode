# Variant Ablation Patch: SRB / Index-Efficient / Soft-to-Hard for SplineKAN, FastKAN, and GRAM

This patch extends the ablation suite to all KAN variants supported by the current All-KAN MNIST codebase.

## Progress Tracker

```text
[done] Single-seed All-KAN MNIST evaluation
[done] Multi-seed All-KAN statistics
[done] Branch-aware main method validated across Spline/Fast/GRAM
[done] Variant ablation patch created
[now] Run smoke ablation test
[next] Run main variant ablation experiment
[next] Summarize ablation results across KAN variants
[next] Generate LaTeX ablation table
[next] Update paper ablation section
```

## New files

```text
src/variant_ablation_models.py
src/experiment_variant_ablation_mnist.py
tools/summarize_variant_ablations.py
tools/make_variant_ablation_latex_table.py
scripts/run_variant_ablations_mnist.sh
```

## What this adds

For each KAN variant:

```text
spline
fast
gram
```

it evaluates:

```text
srb          : Sparse Residual Branch Codebooks
index        : index-efficient branch sharing
soft         : differentiable soft-to-hard branch-index learning
```

## Dependency assumption

This patch assumes you already installed the previous All-KAN MNIST patch, because it imports:

```text
src.all_kan_variants
src.all_kan_compression
src.data
src.train_utils
```

## Smoke test

```bash
bash scripts/run_variant_ablations_mnist.sh smoke
```

This runs tiny settings:

```text
variants = spline fast gram
methods = srb index soft
clusters = 8
residual fraction = 0.25
epochs = 1
finetune epochs = 1
```

## Main single-seed experiment

```bash
bash scripts/run_variant_ablations_mnist.sh main
```

This runs:

```text
variants = spline fast gram
methods = srb index soft
clusters = 16 32
residual fractions = 0.10 0.25 0.50
bits = W4
epochs = 10
finetune epochs = 20
```

## Multi-seed version

```bash
bash scripts/run_variant_ablations_mnist.sh multiseed
```

This runs seeds:

```text
42 123 2026
```

## Summarize

```bash
python tools/summarize_variant_ablations.py \
  --root runs_variant_ablations \
  --out runs_variant_ablations/variant_ablation_summary.csv
```

## Make LaTeX table

```bash
python tools/make_variant_ablation_latex_table.py \
  --csv runs_variant_ablations/variant_ablation_summary.csv \
  --out-tex runs_variant_ablations/variant_ablation_table.tex \
  --caption "Extended ablation results across KAN variants under W4 codebook quantization." \
  --label "tab:variant_ablations"
```

## Paper interpretation target

The main question:

```text
Do SRB, index-efficient, and soft-to-hard ablation trends hold across basis families?
```

Expected possible outcomes:

```text
1. If index-efficient fails for all variants:
   Strong evidence that independent base assignment is generally necessary.

2. If SRB recovers accuracy for all variants:
   Strong evidence that sparse base residuals are a useful robustness extension.

3. If soft-to-hard improves but remains below branch-aware:
   Strong evidence that the bottleneck is representational, not only optimization.

4. If FastKAN behaves differently:
   Evidence that RBF KANs need basis-specific compression/quantization.
```
