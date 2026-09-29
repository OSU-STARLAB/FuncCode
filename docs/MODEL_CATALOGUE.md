# FuncCode-KAN — complete model and dataset catalogue

Every model family and dataset exercised by the FuncCode experiments, across
both tracks. Companion to `MODEL_DETAILS.md`, which covers the conv-CIFAR
backbones in depth.

Two tracks share the compression machinery and the edge convention:

| track | models | datasets | status |
|---|---|---|---|
| **MLP-KAN** (original) | `DirectKANVariant` × 3 bases + MLP baseline | MNIST, Fashion-MNIST, CIFAR-10/100, Tiny ImageNet, 5 tabular | as published |
| **conv-KAGN** | `EightSimpleConvKAGN`, `SimpleConvKAGN` | CIFAR-10, CIFAR-100 | reported CIFAR rows |

> **Numbers below are computed from the code**, not copied from logs. Parameter
> and edge counts were verified by instantiating each model. Accuracies are
> quoted only where they were measured directly; MLP-KAN accuracies come from
> the fully connected protocol and are not re-verified in this document.

---

## 1. The shared edge convention

Every compressible layer exposes an edge matrix `[E, C]`. One row is one
Kolmogorov–Arnold edge function, and the **last slot is always the base branch**:

```
phi_e(x) = sum_{d} w[e,d] · B_d(x)   +   w[e,-1] · silu(x)
                   \___basis___/          \__base branch__/
```

`C = basis_dim + 1`. Branch-aware clustering splits at that boundary identically
in both tracks — this is what lets one compression implementation serve MLP and
conv models.

| family | basis `B_d` | basis_dim | **C** |
|---|---|---|---|
| `spline` | B-spline, `grid_size=5`, `spline_order=3` | 8 | **9** |
| `fast` | RBF, `num_grids=8` on `[-2,2]` | 8 | **9** |
| `gram` | Gram/shifted-Legendre over `tanh x`, `degree=3` | 4 | **5** |
| `KAGNConv2d` / `KAGNLinear` (conv track) | Gram over `tanh x`, `degree=3` | 4 | **5** |
| `mlp` (baseline) | — no edge functions — | — | — |

**`C` is the constant the storage argument turns on.** Scalar quantisation pays
per coefficient (`C · bits`), FuncCode pays one index per edge
(`ceil(log2 K)`). The advantage is therefore ~1.8× larger for spline/fast
(`C=9`) than for gram/KAGN (`C=5`) — the conv track runs at the *least*
favourable `C` in the codebase.

**Edge counts.** MLP-KAN: `E = out · in` per layer. Conv: `E = out · (in/groups) · k_h · k_w`.

---

## 2. Track A — MLP-KAN (`DirectKANVariant`)

Two-layer architecture `[input_dim → hidden_width → num_classes]`, weights held
as an explicit `[out, in, C]` parameter per layer. Images are **flattened**;
there is no convolution and (in `funcodekan/data/cifar.py`) no channel
normalisation.

### 2.1 Model sizes, as run by the paper scripts

| dataset | input | classes | width | variant | C | edges | params | FP32 |
|---|---|---|---|---|---|---|---|---|
| **MNIST** | 784 | 10 | 64 | spline | 9 | 50,816 | 457,344 | 1.74 MiB |
| | | | | fast | 9 | 50,816 | 459,040 | 1.74 MiB |
| | | | | gram | 5 | 50,816 | 254,228 | 0.97 MiB |
| | | | | mlp | — | — | 50,890 | — |
| **Fashion-MNIST** | 784 | 10 | 64 | spline / fast / gram | 9/9/5 | 50,816 | 457,344 / 459,040 / 254,228 | 1.74 / 1.74 / 0.97 MiB |
| **CIFAR-10** (flat) | 3072 | 10 | 128 | spline | 9 | 394,496 | 3,550,464 | 13.54 MiB |
| | | | | fast | 9 | 394,496 | 3,556,864 | 13.54 MiB |
| | | | | gram | 5 | 394,496 | 1,972,756 | 7.52 MiB |
| | | | | mlp | — | — | 394,634 | — |
| **CIFAR-100** (flat) | 3072 | 100 | 128 | spline | 9 | 406,016 | 3,654,144 | 13.94 MiB |
| | | | | fast | 9 | 406,016 | 3,660,544 | 13.94 MiB |
| | | | | gram | 5 | 406,016 | 2,030,536 | 7.74 MiB |
| **Tiny ImageNet** (flat) | 12,288 | 200 | 128 | spline | 9 | 1,598,464 | 14,386,176 | 54.88 MiB |
| | | | | fast | 9 | 1,598,464 | 14,411,008 | 54.88 MiB |
| | | | | gram | 5 | 1,598,464 | 7,992,976 | 30.49 MiB |
| **moons** | 2 | 2 | 32 | spline / fast / gram | 9/9/5 | 128 | 1,152 / 1,220 / 708 | 4.5 / 4.5 / 2.5 KiB |
| **circle_in_circle** | 2 | 2 | 32 | spline / fast / gram | 9/9/5 | 128 | 1,152 / 1,220 / 708 | 4.5 / 4.5 / 2.5 KiB |
| **wine** | 13 | 3 | 32 | spline / fast / gram | 9/9/5 | 512 | 4,608 / 4,698 / 2,630 | 18.0 / 18.0 / 10.0 KiB |
| **dry_bean** | 16 | 7 | 32 | spline / fast / gram | 9/9/5 | 736 | 6,624 / 6,720 / 3,758 | 25.9 / 25.9 / 14.4 KiB |
| **mushroom** | ~117 † | 2 | 32 | spline / fast / gram | 9/9/5 | ~3,800 † | — | — |

† `load_mushroom` one-hot-encodes 22 categorical features via
`pd.get_dummies`, so the input dimension is the number of dummy columns
(~117 for `agaricus-lepiota`), not 22. Exact value depends on the fetched file;
it could not be resolved offline here. The other tabular dims are exact.

**Scale contrast with the conv track.** The largest MLP-KAN model reaches 1.6M
edges (Tiny ImageNet); the conv backbones reach **6.1M**. The tabular models are
128–3,800 edges. This ~48,000× span across the study is why the explicit
`[E, S]` signature matrix works for MLP-KANs and is infeasible at conv scale
(§4).

### 2.2 Layer details

**`SplineBasisLayer`** — B-spline basis on a uniform grid over `[-1,1]`,
`grid_size=5`, `spline_order=3`, built by the Cox–de Boor recurrence with an
`eps=1e-8` guard. Grid registered as a buffer (not learned).

**`FastRBFLayer`** — Gaussian RBFs, `num_grids=8` centres linearly spaced on
`[-2,2]`, denominator `(max−min)/(n−1)`. Optional `LayerNorm` on the input
(enabled when `in_features > 1`).

**`GramPolynomialLayer`** — the same Gram recurrence as the conv track, over
`tanh`-normalised inputs, `degree=3`, with `LayerNorm` on the **output**.

**`MLPBaseline`** — plain `Linear → SiLU → Linear`. Reference point only; it has
no edge functions and is not compressible by FuncCode.

### 2.3 Training (`scripts/paper/01`–`03`, `06`)

| suite | datasets | seeds | epochs | fine-tune | width | K | codebook bits |
|---|---|---|---|---|---|---|---|
| MNIST | mnist | 42/123/2026 | (multiseed protocol) | — | 64 | 16, 32 | 8,6,4,3,2 |
| Fashion-MNIST | fashion_mnist | 42/123/2026 | 10 | 20 | 64 | 16, 32 | 8,6,4,3,2 |
| Tabular | moons, circle_in_circle, wine, dry_bean, mushroom | 42/123/2026/7/1337 | 40 | 30 | 32 | 8,16,32 | 8,6,4,3,2 |
| CIFAR (flat) | cifar10, cifar100 | — | — | — | 128 | — | — |
| Tiny ImageNet | tiny_imagenet | — | — | — | 128 | — | — |
| Width sweep | mnist | 42 | 10 | 20 | **32/64/128/256** | 16, 32 | 4 |

Methods swept: `function`, `branch`. Variants: `spline fast gram mlp` (Tiny
ImageNet splits this into `gram fast mlp` plus a separate `spline` run — see
`06_tiny_imagenet.sh spline_instability`).

### 2.4 Why the CIFAR rows were replaced

`all_kan_cifar.py` builds a `DirectKANVariant` on a flattened 3072-vector at
hidden width 128 — a 2-layer MLP-KAN. `funcodekan/data/cifar.py`'s
`FlattenNormalize` does `x.view(-1)` with no channel normalisation. Reported
dense accuracy: **CIFAR-10 46.82%, CIFAR-100 14.21%**.

Those are architecture artefacts, not compression results, which is what the
convolutional track exists to fix (§3).

---

## 3. Track B — conv-KAGN

Full detail in `MODEL_DETAILS.md`. Summary:

| | CIFAR-10 | CIFAR-100 | CIFAR-10 (documented, **failed**) |
|---|---|---|---|
| preset | `kagn_simple_cifar10_8_layer_v2` | `kagn_simple_cifar100_8_layer_v2` | `kagn_simple_cifar10` |
| class | `EightSimpleConvKAGN` | `EightSimpleConvKAGN` | `SimpleConvKAGN` |
| channels | `[64,128,256,256,384,384,512,256]` | same | `[32,64,128,256]` |
| groups | 1 | 1 | 4 |
| params | 30,623,552 | 30,738,752 | 502,432 |
| edges | 6,123,712 | 6,146,752 | 100,192 |
| FP32 | 116.84 MiB | 117.28 MiB | 1.92 MiB |
| **dense acc (3 seeds)** | **91.43 ± 0.21** | **62.83 ± 0.96** | **64.26** (target ≥91) |
| vs flattened MLP-KAN | 46.82 → **91.43** | 14.21 → **62.83** | — |

Presets also defined but **not used** in the reported results:
`kagn_simple_cifar100` (4-layer at 100 classes) and `kagn_tiny_smoke`
(`[8,16,16,32]`, wiring checks only).

The 4-layer CIFAR-10 backbone fails because both presets share
`dropout=0.25/0.5`, sized for the 30.6M-parameter model; on the 502K model it
forces underfitting. `MODEL_DETAILS.md` §5 has the controlled sweep.

---

## 4. Compression models (both tracks)

These are not architectures but *weight providers* substituted into a trained
model.

| class | file | role |
|---|---|---|
| `DenseEdgeWeights` | `models/conv_kagn.py` | FP32 reference, conv-ready layout |
| `SharedCodebookEdgeWeights` | `models/conv_kagn.py` | FuncCode `function`: one index per edge into a `[K, C]` codebook |
| `BranchCodebookEdgeWeights` | `models/conv_kagn.py` | FuncCode `branch`: separate codebooks for polynomial (`K`) and base (`K_b`) slots |
| `SharedCodebookKAN` / `BranchCodebookKAN` | `models/variants.py` | MLP-KAN equivalents |
| `ClusteredSplineKAN`, `BranchAwareClusteredSplineKAN`, `IndexEfficientBranchSplineKAN`, `SparseResidualBranchSplineKAN` | `models/spline.py` | spline-specific clustered variants |
| `SoftIndexBranchSplineKAN` | `models/soft_codebook.py` | differentiable soft assignment, annealed to hard |
| `IndexEfficientBranchKAN`, `SparseResidualBranchKAN`, `SoftToHardBranchIndexKAN` | `models/ablations.py` | variant-ablation arms |

**The scale problem.** Function-space clustering nominally materialises an
`[E, S]` signature matrix. At `E = 1.77M` (conv layer 6) and `S = 128` that is
~900 MB for one layer. The whitening identity — Cholesky `G = LLᵀ`, `z = Lᵀw`,
so Euclidean k-means on the **`C`-dimensional** whitened coefficients is exactly
function-space k-means — reduces this to 5-D and makes the conv track feasible
(26× less memory, ~140 s for 6.15M edges). Verified to 3.4e-07 with identical
partitions in `tests/test_conv_kagn.py`.

Caveat from the metric ablation: the whitening halves `fn_rel_err` as designed
but **loses 1.37 pp** to naive coefficient-space clustering. It is a
computational contribution, not an accuracy one.

---

## 5. Baselines (conv track)

| method | cost per edge | reaches |
|---|---|---|
| uniform PTQ | `C · bits` | 10.04 b/e at W2 — **no fine-tuning, by design** |
| LSQ QAT | `C · bits` | 10.04 b/e at W2 |
| product quantisation | `m · ceil(log2 K)` | **4.03 b/e at m=2, K=4** |
| magnitude prune + W4 | `C·(1−s)·4` + 1-bit mask | 6.04 b/e at 95% |
| iso-storage dense | `C · 32` on a narrowed net | any, by construction |

Product quantisation at `m=2, K=4` is the strongest baseline found and beats
FuncCode at every budget on both datasets. Reaching it requires `K=4`; the
default `--pq-clusters` grid in `make_conv_plan.py` starts at 16, so `K=4` must
be requested explicitly.

---

## 6. Dataset registry

`funcodekan/data/registry.py` registers 15 datasets. Those actually exercised:

| key | modality | classes | notes |
|---|---|---|---|
| `mnist` | image 28×28×1 | 10 | torchvision |
| `fashion_mnist` | image 28×28×1 | 10 | torchvision |
| `cifar10` | image 32×32×3 | 10 | both tracks |
| `cifar100` | image 32×32×3 | 100 | both tracks |
| `tiny_imagenet` | image 64×64×3 | 200 | manual fetch |
| `moons` | synthetic 2-D | 2 | sklearn |
| `circle_in_circle` | synthetic 2-D | 2 | sklearn |
| `wine` | tabular, 13 feat | 3 | sklearn |
| `dry_bean` | tabular, 16 feat | 7 | UCI (network) |
| `mushroom` | tabular, 22 categorical → one-hot | 2 | UCI (network) |

Registered but **not used** in the reported experiments: `imagenet` (license,
manual), `reuters`, `botswana`, `knot_theory`, `jsc_openml`, and
`traffic_california` (regression, sliding-window).

**Splits.** Tabular: stratified 70/10/20 train/val/test, standardised using
train statistics only. Images (conv track): 45,000/5,000/10,000, model selection
on val, test reported once.

**Conv-track CIFAR normalisation** (`data/cifar_conv.py`, NCHW):

| | mean | std |
|---|---|---|
| CIFAR-10 | (0.4914, 0.4822, 0.4465) | (0.2470, 0.2435, 0.2616) |
| CIFAR-100 | (0.5071, 0.4865, 0.4409) | (0.2673, 0.2564, 0.2762) |

Augmentation: `RandomCrop(32, padding=4, reflect)` → `RandomHorizontalFlip` →
`ToTensor` → `Normalize` → `Cutout(8)`.

---

## 7. `models/zoo.py` — present but not used in reported results

`zoo.py` (~1,650 lines) declares a large library of KAN architectures. It imports
cleanly, but **none of these produce numbers in the reported experiments** —
`zoo_train.py` is driven by `scripts/datasets/run_zoo.sh`, is optional, and
requires an external `kans` package at the repo root.

Families declared: `KAN_MLP_*` (MNIST/CIFAR/TinyImageNet, FastKAN/Gram/PyKAN
variants), `KANLeNet_MNIST`, `KANConvNet_*`, `KANResNet_*` (+ `KANBasicBlock`),
`VGGKAGN` / `VGGKAGNCifar`, `testConvKAGN`, and ImageNet-pretrained KAGN loaders
for Tiny ImageNet transfer.

**Important for the paper:** `zoo.py` also declares its own `SimpleConvKAGN` and
`EightSimpleConvKAGN`, which are *not* the classes used in this work. The
conv track reimplements them in `models/conv_kagn.py` because the zoo versions
depend on the external `kans` package and store parameters in a layout FuncCode's
edge clustering cannot see. **Same architecture, different implementation** —
cite `conv_kagn.py`.

---

## 8. Summary table for the paper

| track | model | dataset | edges | C | dense acc |
|---|---|---|---|---|---|
| MLP-KAN | `DirectKANVariant` spline/fast | MNIST | 50,816 | 9 | (existing protocol) |
| MLP-KAN | `DirectKANVariant` gram | MNIST | 50,816 | 5 | (existing protocol) |
| MLP-KAN | `DirectKANVariant` | Fashion-MNIST | 50,816 | 9 / 5 | (existing protocol) |
| MLP-KAN | `DirectKANVariant` | 5 tabular | 128–3,800 | 9 / 5 | (existing protocol) |
| MLP-KAN | `DirectKANVariant` | Tiny ImageNet | 1,598,464 | 9 / 5 | (existing protocol) |
| MLP-KAN | `DirectKANVariant` | CIFAR-10 (flat) | 394,496 | 9 / 5 | **46.82** ← replaced |
| MLP-KAN | `DirectKANVariant` | CIFAR-100 (flat) | 406,016 | 9 / 5 | **14.21** ← replaced |
| **conv-KAGN** | `EightSimpleConvKAGN` | **CIFAR-10** | **6,123,712** | **5** | **91.43 ± 0.21** |
| **conv-KAGN** | `EightSimpleConvKAGN` | **CIFAR-100** | **6,146,752** | **5** | **62.83 ± 0.96** |
| conv-KAGN | `SimpleConvKAGN` | CIFAR-10 | 100,192 | 5 | 64.26 (not used) |

Three things to disclose: the **CIFAR-10 backbone substitution**, **CIFAR-100's
3.17 pp dense shortfall**, and that **conv-CIFAR is single-architecture,
single-polynomial-family** (KAGN degree 3).
