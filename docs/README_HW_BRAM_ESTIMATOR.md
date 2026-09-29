# FPGA / BRAM Bandwidth Estimator Patch

This patch adds an analytical FPGA-oriented memory estimator for the compressed KAN experiments.

It is designed for paper-level hardware interpretation, not exact post-synthesis reporting.

## What it estimates

From a `summary.csv`, it reports:

```text
total storage KiB
compression ratio
BRAM18 estimate
BRAM36 estimate
codebook KiB
index KiB
scale KiB
index fraction
codebook fraction
estimated dense FP32 traffic per inference
estimated compressed traffic per inference
traffic reduction
```

For branch-aware and SRB variants, it also reports branch-specific storage:

```text
spline index KiB
base index KiB
residual index KiB
conditional base codebook KiB
residual codebook KiB
```

## Why this is useful

Our storage breakdown showed that compressed KAN models are index-dominated. This estimator turns that into a hardware-facing message:

```text
The main FPGA memory bottleneck is not codebook storage.
The bottleneck is assignment/index traffic.
```

This supports the paper argument for branch-aware compression and future index-compression designs.

## Install

From your project root:

```bash
unzip HW_BRAM_Estimator_patch.zip
```

## Run on one experiment

```bash
python tools/estimate_fpga_bram.py \
  --summary runs/branch_k32_s32_b16_ref/summary.csv \
  --out runs/branch_k32_s32_b16_ref/hw_estimate.csv
```

## Run on multiple experiments

```bash
python tools/compare_hw_estimates.py \
  --summaries \
    runs/func_k16_ref/summary.csv \
    runs/branch_k16_s16_b8_ref/summary.csv \
    runs/branch_index_k16_ref/summary.csv \
    runs/srb_k16_r25_b8/summary.csv \
  --out runs/hw_compare_k16.csv
```

## Recommended paper commands

K=16:

```bash
python tools/compare_hw_estimates.py \
  --summaries \
    runs/func_k16_ref/summary.csv \
    runs/branch_k16_s16_b8_ref/summary.csv \
    runs/branch_index_k16_ref/summary.csv \
    runs/srb_k16_r10_b8/summary.csv \
    runs/srb_k16_r25_b8/summary.csv \
    runs/srb_k16_r50_b8/summary.csv \
  --out runs/hw_compare_k16.csv
```

K=32:

```bash
python tools/compare_hw_estimates.py \
  --summaries \
    runs/func_k32_ref/summary.csv \
    runs/branch_k32_s32_b16_ref/summary.csv \
    runs/srb_k32_r10_b8/summary.csv \
    runs/srb_k32_r25_b8/summary.csv \
    runs/srb_k32_r50_b8/summary.csv \
  --out runs/hw_compare_k32.csv
```

## Important interpretation

This estimator assumes that compressed codebooks and index streams are stored on-chip or fetched through a packed memory format.

It does not model:
- spline-basis arithmetic cost,
- LUT/DSP use,
- routing,
- initiation interval,
- clock frequency,
- actual Vivado/Vitis HLS scheduling.

Those should be stated as limitations.

The estimator is still useful because it gives a consistent hardware-memory view across compression methods.
