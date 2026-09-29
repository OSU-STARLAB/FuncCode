# Differentiable Soft-to-Hard Codebook Learning

This patch adds the next research module after SRB codebooks.

## Research idea

Static K-means fixes each edge's codebook index before fine-tuning. The new module makes the index assignment differentiable during training:

```text
soft assignment = softmax(edge_logits / temperature)
edge function   = weighted sum of codebook entries
```

During training, both the codebook vectors and the assignment logits are optimized. During deployment, the assignments are hardened with `argmax`, so the final compressed model is still hardware-friendly:

```text
spline codebook + one compact spline index per edge + conditional base codebook
```

This is meant to improve the weak point we observed in the index-efficient branch variant: it has excellent storage, but poor accuracy after static hard assignment.

## Files added

```text
src/soft_codebook.py              # Soft-to-hard model and builder
src/experiment_soft_codebook.py   # Full MNIST experiment script
run_soft_codebook.sh              # Easy commands
README_SOFT_CODEBOOK.md           # This file
```

## How to run

From the project root:

```bash
pip install -r requirements.txt
bash run_soft_codebook.sh k32
```

For the smaller 16-cluster model:

```bash
bash run_soft_codebook.sh k16
```

For both:

```bash
bash run_soft_codebook.sh sweep
```

## Output

Each run writes:

```text
runs/soft_branch_index_k32/config.json
runs/soft_branch_index_k32/dense.pt
runs/soft_branch_index_k32/hard_finetuned.pt
runs/soft_branch_index_k32/hwq_w8.pt
runs/soft_branch_index_k32/hwq_w6.pt
runs/soft_branch_index_k32/hwq_w4.pt
runs/soft_branch_index_k32/hwq_w3.pt
runs/soft_branch_index_k32/hwq_w2.pt
runs/soft_branch_index_k32/summary.csv
```

## Important rows in `summary.csv`

Look especially at:

```text
hard_argmax_before_soft_training
hard_argmax_after_soft_training
hard_finetuned_fp32_codebook
hard_finetuned_hwq_w4
hard_finetuned_hwq_w2
```

These tell us whether differentiable assignment learning improves hard deployable accuracy before and after codebook fine-tuning.

## Recommended first comparison

Compare this run:

```bash
bash run_soft_codebook.sh k32
```

against your previous static index-efficient branch run:

```text
runs/branch_index_k32_ref/summary.csv
```

The main research question is:

```text
Can learned soft-to-hard assignments recover the accuracy lost by static index-efficient branch clustering while preserving the same storage format?
```

## Suggested ablations

After the first run, try:

```bash
python -m src.experiment_soft_codebook --run-name soft_k32_no_distill --spline-clusters 32 --distill-alpha 0.0 --soft-epochs 10 --hard-finetune-epochs 20
python -m src.experiment_soft_codebook --run-name soft_k32_more_entropy --spline-clusters 32 --entropy-lambda 5e-3 --soft-epochs 10 --hard-finetune-epochs 20
python -m src.experiment_soft_codebook --run-name soft_k32_low_temp --spline-clusters 32 --temp-start 1.0 --temp-end 0.10 --soft-epochs 10 --hard-finetune-epochs 20
```

## Progress tracker

- [done] Dense SplineKAN training from scratch
- [done] Coefficient-space clustering baseline
- [done] Function-space clustering
- [done] Branch-aware base/spline codebooks
- [done] Storage breakdown
- [done] Index-efficient branch-aware variant
- [done] Sparse residual branch-aware variant
- [new] Differentiable soft-to-hard codebook learning
- [next] FPGA/BRAM bandwidth estimator
