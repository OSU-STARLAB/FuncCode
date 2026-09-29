# Model and protocol details — conv-KAGN CIFAR backbones

Reference for the paper's setup section. Everything here is read from
`funcodekan/models/conv_kagn.py`, `funcodekan/data/cifar_conv.py`,
`funcodekan/utils/conv_training.py` and
`funcodekan/compression/conv_compression.py` as run.

The measured results these settings produce are reported in the paper and its
appendix; `REPRODUCE.md` gives the commands that regenerate them.

---

## 1. Summary

Both reported CIFAR backbones are the **same 8-layer KAGN convolutional
architecture**, differing only in classifier width:

| | CIFAR-10 | CIFAR-100 |
|---|---|---|
| preset | `kagn_simple_cifar10_8_layer_v2` | `kagn_simple_cifar100_8_layer_v2` |
| class | `EightSimpleConvKAGN` | `EightSimpleConvKAGN` |
| channels | `[64,128,256,256,384,384,512,256]` | `[64,128,256,256,384,384,512,256]` |
| groups | 1 | 1 |
| degree (conv / head) | 3 / 3 | 3 / 3 |
| classes | 10 | 100 |
| **parameters** | **30,623,552** | **30,738,752** |
| **KAN edges** | **6,123,712** | **6,146,752** |
| coefficients per edge `C` | 5 | 5 |
| FP32 storage | 116.84 MiB | 117.28 MiB |
| bits/edge (FP32) | 160.049 | 160.049 |
| dense accuracy (3 seeds) | **91.43 ± 0.21** | **62.83 ± 0.96** |

Using one architecture for both datasets is deliberate: it puts them at
comparable edge counts (6.12M vs 6.15M), so the bits/edge axis and the
99.7%-index-bits result apply identically to both rows of the table.

> **Disclosure.** The originally planned CIFAR-10 backbone was a 4-layer
> `SimpleConvKAGN([32,64,128,256], groups=4)`. It reaches only **64.26%** against
> a ≥91% target — its dropout (0.25/0.5) is sized for the 61× larger 8-layer net
> and drives the small model into severe underfitting — and was replaced by the
> 8-layer backbone. §5 documents it and the reason; the substitution is stated in
> the paper's appendix.

---

## 2. The KAGN layer

### 2.1 Edge function

Every compressible layer exposes an edge matrix `[E, C]` with `C = degree + 2`.
Each row is one Kolmogorov–Arnold edge function:

```
phi_e(x) = sum_{d=0..degree} w[e,d] · P_d(tanh x)  +  w[e,-1] · silu(x)
```

* `P_d` — Gram (shifted-Legendre) polynomials, recurrence
  `P_i = (x·P_{i-1}·(2i−1) − P_{i-2}·(i−1)) / i`, with `P_0 = 1`, `P_1 = x`
* `tanh` maps activations into `[-1,1]`, the domain the recurrence assumes
* slot `[-1]` is the **base branch** (SiLU), matching the
  `[..., :-1] = basis, [..., -1] = base` convention in
  `funcodekan/models/variants.py`, so branch-aware clustering splits identically
  to the MLP-KAN track

With `degree = 3`: `C = 3 + 2 = 5` coefficients per edge. **This is the constant
the storage argument turns on** — scalar quantisation pays per coefficient,
FuncCode pays one index per edge.

### 2.2 What counts as an edge

| layer type | edge | count |
|---|---|---|
| `KAGNConv2d` | `(out_ch, in_ch/groups, k_h, k_w)` | `out · (in/groups) · k_h · k_w` |
| `KAGNLinear` | `(out_feature, in_feature)` | `out · in` |

Parameters are stored in **conv-ready layout** (`poly_weight`, `base_weight`) so
the dense forward pass needs no reshape; the `[E, C]` edge view is materialised
only for clustering.

### 2.3 Forward pass

`KAGNConv2d` — two parallel convolutions summed, then norm and activation:

```
y = SiLU( BatchNorm2d( conv2d(P(tanh x), W_poly) + conv2d(SiLU(x), W_base) ) )
```

Dropout2d is applied to the **input** when `dropout > 0`. The polynomial branch
expands channels by `degree+1 = 4` before the convolution
(`[B,C,H,W] → [B,C·4,H,W]`), which is why activation memory is ~4× a plain conv.

`KAGNLinear` (classifier head) — LayerNorm → dropout → same two-branch form,
with no output activation when `final=True`:

```
y = F.linear(P(tanh x̂), W_poly) + F.linear(SiLU(x̂), W_base),  x̂ = LayerNorm(x)
```

All convolutions are 3×3, `padding=1`, `dilation=1`.

---

## 3. Architecture: `EightSimpleConvKAGN`

Strides `(1, 2, 2, 1, 1, 2, 1, 1)`, then global average pool and a KAGN head.

### 3.1 Feature-map flow (32×32 input)

| block | op | in | out |
|---|---|---|---|
| `layers.0` | KAGNConv2d s1 | 3 × 32 × 32 | 64 × 32 × 32 |
| `layers.1` | KAGNConv2d s2 | 64 × 32 × 32 | 128 × 16 × 16 |
| `layers.2` | KAGNConv2d s2 | 128 × 16 × 16 | 256 × 8 × 8 |
| `layers.3` | KAGNConv2d s1 | 256 × 8 × 8 | 256 × 8 × 8 |
| `layers.4` | KAGNConv2d s1 | 256 × 8 × 8 | 384 × 8 × 8 |
| `layers.5` | KAGNConv2d s2 | 384 × 8 × 8 | 384 × 4 × 4 |
| `layers.6` | KAGNConv2d s1 | 384 × 4 × 4 | 512 × 4 × 4 |
| `layers.7` | KAGNConv2d s1 | 512 × 4 × 4 | 256 × 4 × 4 |
| `layers.8` | AdaptiveAvgPool2d | 256 × 4 × 4 | 256 × 1 × 1 |
| `output` | KAGNLinear (final) | 256 | `num_classes` |

### 3.2 Edge counts per layer

Identical for both datasets except the head.

| layer | in | out | edges | share |
|---|---|---|---|---|
| `layers.0` | 3 | 64 | 1,728 | 0.03% |
| `layers.1` | 64 | 128 | 73,728 | 1.2% |
| `layers.2` | 128 | 256 | 294,912 | 4.8% |
| `layers.3` | 256 | 256 | 589,824 | 9.6% |
| `layers.4` | 256 | 384 | 884,736 | 14.4% |
| `layers.5` | 384 | 384 | 1,327,104 | 21.6% |
| `layers.6` | 384 | 512 | 1,769,472 | 28.9% |
| `layers.7` | 512 | 256 | 1,179,648 | 19.3% |
| `output` | 256 | 10 / 100 | 2,560 / 25,600 | 0.04% / 0.42% |
| **total** | | | **6,123,712 / 6,146,752** | |

Two consequences used in the analysis:

* **The scale is 60× the MLP-KAN models** (thousands of edges), which is why the
  explicit `[E, S]` signature matrix is infeasible — at `E = 1.77M`, `S = 128`
  one layer alone is ~900 MB — and why the whitened 5-D formulation is required.
* **The first conv and head together are 0.4% of all edges** yet leaving them
  uncompressed is worth **+2.59 pp** (skip-first/skip-head ablation).

### 3.3 Parameter split (CIFAR-100)

| component | count |
|---|---|
| edge coefficients (`poly_weight` + `base_weight`) | 30,733,760 |
| norm / affine parameters | 4,992 |
| BatchNorm running statistics (buffers) | 4,480 |
| **total parameters** | **30,738,752** |

Edge coefficients are 99.98% of parameters. The remainder is small but **always
counted** in every storage figure (§6).

---

## 4. Regularisation

| setting | value |
|---|---|
| `dropout` (Dropout2d, conv inputs) | 0.25 |
| `dropout_linear` (head) | 0.5 |
| first layer dropout | 0.0 (always disabled) |
| norm | BatchNorm2d (conv, affine) / LayerNorm (head) |
| activation | SiLU |

Both presets share these values. **They are correct for this 30.6M-parameter
model and catastrophically wrong for the 502K-parameter 4-layer variant** — see
§5.

---

## 5. The 4-layer backbone (documented, not used)

`SimpleConvKAGN([32,64,128,256], groups=4)`, strides `(1,2,2,1)`.

| | |
|---|---|
| parameters | 502,432 |
| edges | 100,192 |
| FP32 storage | 1.92 MiB |
| **dense accuracy** | **64.26%** (target ≥91%) |

| layer | in/group | out | groups | edges |
|---|---|---|---|---|
| `layers.0` | 3 | 32 | 1 | 864 |
| `layers.1` | 8 | 64 | 4 | 4,608 |
| `layers.2` | 16 | 128 | 4 | 18,432 |
| `layers.3` | 32 | 256 | 4 | 73,728 |
| `output` | 256 | 10 | 1 | 2,560 |

**Why it fails.** A 200-epoch controlled sweep (one variable per arm):

| arm | params | dropout | groups | test | train loss | peak epoch | regime |
|---|---|---|---|---|---|---|---|
| stock | 502K | 0.25/0.5 | 4 | 64.26 | 1.2389 | 128/200 | **underfit** |
| no dropout | 502K | 0/0 | 4 | 83.92 | 0.7471 | 172/200 | converged |
| no dropout, groups=1 | 1.95M | 0/0 | 1 | 86.23 | 0.5945 | 57/200 | **overfit** |
| 8-layer (adopted) | 30.6M | 0.25/0.5 | 1 | 90.85 | 0.6267 | 146/200 | converged |

The shared `dropout = 0.25 / 0.5` is sized for a model 61× larger. On the small
net it forces underfitting — training loss stalls at 1.24 against a
label-smoothing floor of ~0.55, and validation peaks at epoch 128 then declines
for 72 epochs. Removing dropout recovers **+19.9 pp at identical parameter
count** but still tops out at 83.92%. **No 4-layer variant reaches 91%.**

Note `funcodekan/utils/conv_training.py` states the recipe takes "the 4-layer net
from the low 80s to ~92% on CIFAR-10". That figure matches the **8-layer** net
(90.85–91.57); the comment appears attached to the wrong architecture.

---

## 6. Storage accounting

One accountant for every method (`conv_compression.storage_breakdown`,
`conv_baselines.baseline_storage_bits`), so all rows are commensurable.

**FP32 reference** (`dense_storage_bits`):

```
edge coefficients:  E · C · 32 bits
norm/affine params: numel · 32 bits
BN running stats:   numel · 32 bits          (running_mean, running_var)
```

**Compressed methods** charge, at `other_bits = 16` for non-edge tensors:

| method | edge cost per edge | plus |
|---|---|---|
| FuncCode `function` | `ceil(log2 K)` | codebook `K·C` at `codebook_bits` |
| FuncCode `branch` | `ceil(log2 K) + ceil(log2 K_b)` | two codebooks |
| uniform / LSQ | `C · bits` | 2 scales per out-channel, 16 bits |
| product quant. (`m` subvectors) | `m · ceil(log2 K)` | `m` codebooks at FP32 |
| prune + W4 | `C·(1−s)·4` | 1-bit mask per coefficient, scales |
| iso-storage dense | `C · 32` on a narrowed net | — |

`ceil(log2 K)` via `index_bits_for`. Normalisation parameters and folded BN
statistics are counted for **every** method, including baselines;
`test_storage_accounting_loses_nothing` asserts nothing is dropped.

**Measured**: index bits are **99.7%** of compressed storage at this scale, so
codebook precision is nearly free — the entire w8→w2 sweep spans **<0.002
bits/edge** (codebook-precision sweep).

Product quantisation is charged **more conservatively than FuncCode**: its
codebooks are counted at FP32 while FuncCode's are quantised to `codebook_bits`.

---

## 7. Data pipeline

`funcodekan/data/cifar_conv.py`. NCHW throughout — no flattening.

| | CIFAR-10 | CIFAR-100 |
|---|---|---|
| mean | (0.4914, 0.4822, 0.4465) | (0.5071, 0.4865, 0.4409) |
| std | (0.2470, 0.2435, 0.2616) | (0.2673, 0.2564, 0.2762) |

**Train augmentation** — `RandomCrop(32, padding=4, padding_mode="reflect")` →
`RandomHorizontalFlip` → `ToTensor` → `Normalize` → `Cutout(8)`.
Cutout is a single 8×8 square zeroed *after* normalisation, centre sampled
uniformly and clipped at borders (DeVries & Taylor).

**Eval** — `ToTensor` → `Normalize` only.

**Splits** — 45,000 train / 5,000 val (held out from train, seeded) / 10,000
test. Model selection is on val; test is reported once.

This replaces the flattened protocol in `funcodekan/data/cifar.py`, whose
`FlattenNormalize` does `x.view(-1)` with no channel normalisation.

---

## 8. Training protocol

Identical for both datasets except mixup.

| setting | value |
|---|---|
| optimiser | AdamW, lr 1e-3, weight decay 5e-5 |
| no-decay group | all 1-D tensors, `norm`, `codebook` |
| schedule | cosine, 5 warmup epochs, `min_lr_ratio` 0.01 |
| epochs | 200 (dense), 30 (fine-tune, every compressed arm) |
| fine-tune lr | 2e-4, warmup 2 |
| batch size | 128 (train) / 512 (eval) |
| label smoothing | 0.1 |
| **mixup α** | **0.0 (CIFAR-10) / 0.2 (CIFAR-100)** |
| EMA decay | 0.999 |
| grad clip | 1.0 (global norm) |
| precision | fp16 autocast + `GradScaler` (portable to GPUs without bf16/TF32) |
| cudnn | `benchmark=True` |
| seeds | 42, 123, 2026 |

**EMA** — a shadow copy updated every step; val is evaluated on both the live
and EMA weights, and the better is selected. If EMA wins on test it is adopted
as the deployed model.

**Equal-budget rule** — every compressed arm starts from the *same* dense
checkpoint and receives the *same* 30-epoch fine-tune. The single exception is
uniform PTQ, which is inference-only by design and receives **no** fine-tuning;
its results are a PTQ-fragility measurement, not a scalar-quantisation baseline.

**Reproducibility** — a fixed seed is not sufficient. Run-to-run variation at
fixed seed is **0.34 pp** (CIFAR-10 dense, seed 42: 90.85 vs 91.19) from
`cudnn.benchmark` autotuning and non-deterministic atomics. Exact reproduction
needs `setup_backend(deterministic=True)` and `--no-amp`.

---

## 9. Compression configuration

| | |
|---|---|
| clustering metric | whitened function-space (default), `signature`, `coefficient` |
| function samples `S` | 128 grid points |
| fit samples | 300,000 edges max (subsampled for k-means) |
| `K` swept | 8, 16, 32, 64, 256 |
| `K_b` (branch) | `K/2` |
| codebook bits | 8 (**recommended**), 4 (unstable — see below) |

**Whitening.** Function-space distance is
`d(a,b)² = (w_a − w_b)ᵀ G (w_a − w_b)` with `G = BᵀB` for basis matrix `B[s,c]`.
Cholesky `G = LLᵀ` and `z = Lᵀw` give `d(a,b)² = ‖z_a − z_b‖²`, so Euclidean
k-means on the **5-dimensional** whitened coefficients is exactly function-space
k-means. Verified to 3.4e-07 with identical partitions
(`tests/test_conv_kagn.py`); 26× less memory than the explicit `[E,S]` path.

Note the metric ablation: whitening halves `fn_rel_err` as designed but
**loses 1.37 pp** to naive coefficient-space clustering. The identity is a
computational contribution, not an accuracy one.

**Codebook precision.** Report **w8**. w4 is *unstable* — four runs of the
identical K=32 config spanned 9.99 pp (47.46 → 57.45) where w8 spanned 1.35 pp.
`fn_rel_err` cannot predict this, being measured before codebook quantisation.

---

## 10. Reproduction

```bash
conda activate funcodekan
pytest tests/ -q                       # 33 passed

python -m funcodekan.experiments.conv_cifar \
  --dataset cifar10 --preset kagn_simple_cifar10_8_layer_v2 --seed 42 \
  --epochs 200 --finetune-epochs 30 --batch-size 128 --lr 1e-3 --mixup 0.0 \
  --data-root ./data --out-dir runs_conv_cifar --stages dense

python -m funcodekan.experiments.conv_cifar \
  --dataset cifar100 --preset kagn_simple_cifar100_8_layer_v2 --seed 42 \
  --epochs 200 --finetune-epochs 30 --batch-size 128 --lr 1e-3 --mixup 0.2 \
  --data-root ./data --out-dir runs_conv_cifar --stages dense
```

Measured on a 32 GB data-centre GPU, batch 128, fp16: **~1.2 GiB peak** and
**39 ms/step** for the 8-layer net (351 steps/epoch). Two jobs per GPU gives
1.35× throughput; three gives no further gain.
