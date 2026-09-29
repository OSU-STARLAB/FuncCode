# Storage Breakdown Patch

This patch adds detailed storage accounting for shared and branch-aware compressed KANs.

## What it adds

Every `summary.csv` row now includes:

```text
storage_total_bits
storage_total_kib
shared_codebook_bits
shared_index_bits
spline_codebook_bits
spline_index_bits
base_codebook_bits
base_index_bits
scale_bits
metadata_bits
```

For shared codebook models:

```text
shared_codebook_bits + shared_index_bits
```

For branch-aware models:

```text
spline_codebook_bits + spline_index_bits
+ base_codebook_bits + base_index_bits
```

For HWQ rows, the codebook bits are low-bit quantized and scale bits are included.

## Files to copy

Copy/overwrite these files into your current codebase:

```text
src/storage.py
src/quantization.py
src/experiment_mnist.py
tools/summarize_storage_breakdown.py
```

## Run

After installing the patch, rerun one branch-aware experiment:

```bash
bash run_branch_aware.sh compare_k16
```

Then inspect:

```bash
python tools/summarize_storage_breakdown.py runs/branch_k16_s16_b8/summary.csv
```

## Recommended paper table

Use these columns:

```text
method
test_acc
storage_total_kib
compression_vs_dense
spline_index_kib
base_index_kib
spline_codebook_kib
base_codebook_kib
scale_kib
```

This will clearly show that branch-aware improves accuracy because it spends additional bits on a separate base index stream.
