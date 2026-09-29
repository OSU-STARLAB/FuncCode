# Convolutional KAGN track (CIFAR-10 / CIFAR-100)

The fully connected CIFAR protocol (`funcodekan/experiments/all_kan_cifar.py`)
builds a `DirectKANVariant` on a flattened 3072-dim input at hidden width 128 —
a two-layer MLP-KAN with no convolutional structure and, per
`funcodekan/data/cifar.py`, no channel normalisation. Its dense reference tops
out near 47% on CIFAR-10 and 14% on CIFAR-100, which makes any compression
measured against it hard to interpret.

This track replaces the CIFAR rows with convolutional KAGN backbones so the
dense reference is competitive, keeping the FuncCode compression machinery
unchanged. Entry point: `funcodekan/experiments/conv_cifar.py`.

Full architecture and hyperparameter reference: `MODEL_DETAILS.md`.

---

## 1. What an "edge" is in a conv KAN

For an MLP-KAN, an edge is an `(out, in)` pair carrying a `coeff_dim`-vector.
For a conv KAGN layer, an edge is an `(out_ch, in_ch, kh, kw)` tuple carrying
the same `degree + 2 = 5` coefficients: `degree + 1` Gram-polynomial
coefficients over `tanh(x)`, plus one base-branch (SiLU) coefficient in the last
slot. The convention matches `funcodekan/models/variants.py` exactly, so
branch-aware clustering splits at the same boundary in both tracks.

The consequence is scale. `EightSimpleConvKAGN([64,128,256,256,384,384,512,256])`
has **6.15M edges** against a few thousand for the MLP models:

| layer | edges |
|---|---|
| 3→64 | 1,728 |
| 64→128 | 73,728 |
| 128→256 | 294,912 |
| 256→256 | 589,824 |
| 256→384 | 884,736 |
| 384→384 | 1,327,104 |
| 384→512 | 1,769,472 |
| 512→256 | 1,179,648 |
| head | 25,600 |

## 2. The Gram-whitening identity

The fully connected clustering path materialises an `[E, S]` signature matrix.
At `E = 1.77M` and `S = 128` that is ~900 MB for a single layer, so it does not
run at conv scale.

Function-space distance between two edges is

```
d(a,b)² = ∫ (φ_a(x) − φ_b(x))² dx = (w_a − w_b)ᵀ G (w_a − w_b),   G = BᵀB
```

where `B[s,c]` is the basis evaluated on the sampling grid. Cholesky-factor
`G = LLᵀ` and set `z = Lᵀw`. Then `d(a,b)² = ‖z_a − z_b‖²`, so Euclidean
k-means on the **5-dimensional** whitened coefficients is exactly function-space
k-means, and the whitened centroid maps back to the coefficient mean of its
members.

Verified in `tests/test_conv_kagn.py`: metric agreement to 3.4e-07 and identical
partitions versus the explicit-signature path. 26× less memory; the full
6.15M-edge model clusters in ~140 s on 4 CPU threads at 2.75 GB peak.

## 3. One default deliberately changed

`compression/cross_variant.py` z-scores each signature before clustering, then
averages *raw* coefficients to form centroids. The metric is therefore
shape-only while the centroid is scale-aware, so a cluster spanning two orders
of magnitude in edge magnitude gets a centroid that fits none of its members.
That is survivable at MLP width 64 and harmful for conv layers.

The default in this track is unnormalised true L² (`--metric whiten`). The
legacy behaviour is available via `--metric signature --signature-normalize`
and is reported as an ablation arm.

## 4. The comparison axis

`C = 5` coefficients per edge. Scalar quantisation pays **per coefficient**;
FuncCode pays **per edge** (one `ceil(log2 K)` index).

| method | bits/edge | vs FP32 |
|---|---|---|
| dense FP32 | 160 | 1.0× |
| uniform W8 | 40 | 4.0× |
| uniform W4 | 20 | 8.0× |
| uniform W2 (scalar floor) | 10 | 16.0× |
| FuncCode K=256 | 8 | 20.0× |
| FuncCode K=32 | 5 | 32.0× |
| FuncCode K=16 | 4 | 40.0× |

A 256-entry codebook costs fewer bits per edge than W2, and below 10 bits/edge
no scalar scheme has bits left to spend. On the reported model, storage comes
out **99.7% index bits** — codebooks amortise to nothing at this scale, which is
a conv-specific measurement the fully connected experiments cannot show.

The comparison axis is therefore **accuracy at matched bits/edge**
(`table_iso_storage.tex`), not compression ratio against each family's own dense
model.

## 5. Budget parity

Every arm starts from the *same* dense checkpoint and receives an *identical*
fine-tuning budget (30 epochs, lr 2e-4). The one exception is uniform PTQ, which
is inference-only by design and receives no fine-tuning; it is reported as a
post-training-quantisation fragility measurement, not as a tuned scalar
baseline.

Storage is charged by a single accountant
(`conv_compression.storage_breakdown`) that always counts normalisation
parameters and folded BatchNorm statistics for every method;
`test_storage_accounting_loses_nothing` asserts nothing is dropped. Product
quantisation is charged more conservatively than FuncCode: its codebooks are
counted at FP32 while FuncCode's are quantised to `--codebook-bits`.

The iso-storage dense baseline — a narrower network trained from scratch at the
same bit budget — is included as the natural control: if the budget is N bits,
is it better spent on a codebook or on a smaller dense net?

## 6. Codebook precision

Report **w8**. The w4 setting is unstable at this scale: four runs of an
identical `K=32` CIFAR-100 configuration spanned 9.99 pp, where w8 spanned
1.35 pp, with identical pre-quantisation reconstruction error in every case.
Because the entire w8→w2 sweep moves storage by under 0.002 bits/edge, there is
no storage argument for taking the risk.

## 7. Known limitations

- Two backbones and one polynomial family (KAGN, degree 3).
- The FPGA/BRAM analysis covers the fully connected models and has not been
  redone at conv scale.
- The whitening identity holds for a fixed sampling grid, so a different input
  distribution induces a slightly different metric.
- Run-to-run variation at a fixed seed is 0.34 pp (cudnn autotuning,
  non-deterministic atomics). Exact reproduction needs
  `setup_backend(deterministic=True)` and `--no-amp`.

## 8. Running it

```bash
bash scripts/paper/09_conv_cifar.sh smoke    # ~10 min, wiring check
bash scripts/paper/09_conv_cifar.sh tier1    # the main table
bash scripts/paper/09_conv_cifar.sh tables   # CPU, rerun anytime
```

See `REPRODUCE.md` for the full step-by-step protocol, and the top-level
`README.md` for individual-experiment commands.
