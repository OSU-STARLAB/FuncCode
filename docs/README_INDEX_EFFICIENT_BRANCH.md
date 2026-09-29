# Index-Efficient Branch-Aware Codebook Patch

This patch adds an index-efficient branch-aware KAN compression mode.

## Motivation

The previous branch-aware model stores two assignment streams per edge:

```text
spline_id per edge
base_id per edge
```

This greatly improves accuracy, but the storage breakdown showed that the second `base_id`
stream is the main reason compression decreases.

## New idea

The index-efficient branch-aware model stores only one assignment stream:

```text
spline_id per edge
```

Then it reconstructs both branches from that same ID:

```text
spline_weight = spline_codebook[spline_id]
base_weight   = conditional_base_codebook[spline_id]
```

So the compressed edge becomes:

```text
edge_weight = [
    spline_codebook[spline_id],
    conditional_base_codebook[spline_id]
]
```

This preserves branch-separated parameters but removes the second base-index stream.

## New cluster method

```bash
--cluster-method branch_index
```

## New experiment script

```bash
bash run_index_efficient_branch.sh smoke
bash run_index_efficient_branch.sh compare_k16
bash run_index_efficient_branch.sh compare_k32
bash run_index_efficient_branch.sh compare_k64
bash run_index_efficient_branch.sh all
```

## Most important comparison

```text
shared function-space K=16
vs.
full branch-aware K=16/B8
vs.
index-efficient branch-aware K=16
```

Expected tradeoff:

```text
shared function-space:
    highest compression, lower accuracy

full branch-aware:
    highest accuracy, lower compression

index-efficient branch-aware:
    middle ground; aims to recover much of branch-aware accuracy
    while keeping storage close to shared function-space
```

## Manual run

```bash
python -m src.experiment_mnist \
  --run-name branch_index_k16 \
  --cluster-method branch_index \
  --branch-spline-method function \
  --branch-spline-clusters 16 \
  --branch-function-samples 128 \
  --branch-function-domain grid \
  --epochs 10 \
  --finetune-epochs 20 \
  --width 64 \
  --clusters 16 \
  --bits-list 8 6 4 3 2 \
  --batch-size 1024 \
  --test-batch-size 2048
```

## Files overwritten

```text
src/models.py
src/clustering.py
src/storage.py
src/quantization.py
src/experiment_mnist.py
run_index_efficient_branch.sh
```
