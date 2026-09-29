# HW design contract — frozen decisions

The normative specification of the integer datapath shared by the emulator
(`funcodekan/hw/fixed_point.py`) and the HLS designs (`hw/hls/`). Every
constant referenced here lives in exactly two places, `funcodekan/hw/hw_spec.py`
(Python) and `hw/hls/common/hw_config.h` (HLS), and the two must never drift.

`docs/HW_FAIRNESS.md` carries the comparison contract (what is identical across
designs, what is allowed to differ, and the threats to validity). This document
is the engineering source of truth for the arithmetic itself.

---

## Frozen design decisions (implemented identically in emulator and HLS)

`docs/HW_FAIRNESS.md` carries the full fairness contract; this section is the
engineering source of truth. Constants live in
`funcodekan/hw/hw_spec.py` (Python) and `hw/hls/common/hw_config.h` (HLS)
and must never drift.

### Quantization contract (D2 and D3, identical)

Fake-quant points in `QuantSplineKAN` — quantize exactly what the HW
quantizes, nothing more:

1. **(a) layer-1 input**: signed INT4 LSQ activation quantizer (per-tensor
   step `s_a0`). MNIST inputs are mean/std-standardized (0.1307/0.3081), so
   they are signed — symmetric signed range −8..7.
2. **(b) spline coefficients** per layer: signed INT4 LSQ weight quantizer,
   per-tensor step `s_ws{l}` (range −8..7).
3. **(c) base weights** per layer: signed INT4 LSQ weight quantizer,
   per-tensor step `s_wb{l}` (range −8..7).
4. **(d) inter-layer activation** (between layer 1 and layer 2): signed INT4
   LSQ activation quantizer, per-tensor step `s_a1`.

Logits are NOT quantized (INT32 fixed-point output, see below). Per-tensor
scales only (HW simplicity). D3 uses the same quantizer config on codebook
ENTRIES (spline codebook [32,8] → one per-tensor scale; base codebook [16]
→ one per-tensor scale) so D2 vs D3 isolates codebook sharing only.
D3 indices are FROZEN throughout finetune.

### LSQ

`LsqQuantizer(bits=4, signed=True)`: learnable step `s`, STE rounding,
gradient scale g = 1/sqrt(N·Qp), init `s = 2·mean(|x|)/sqrt(Qp)` on first
batch. Qn=−8, Qp=7.

### Basis/SiLU LUTs (D2 and D3 — the A4 insight)

With A4 activations, a layer input takes ≤16 values `q·s_a`, q∈{−8..7}.
Per layer we therefore precompute, indexed by `u = q + 8` (offset code):

- `LUT_B[16][8]`: the 8 Cox–de Boor B-spline basis values at x = q·s_a,
  stored **int16 in Q2.14** (value_int = round_half_up(v·2^14); bases are a
  partition of unity in [0,1], so 2^14 always fits int16 at max precision),
- `LUT_S[16]`: SiLU(q·s_a), stored **int16 in Q6.10** (covers (−32, 32);
  export asserts, which requires `s_a < 32/7 ≈ 4.57`. Chosen after M1: the
  trained D2 hidden step is ≈3.06, so SiLU reaches ≈21 — Q5.11's ±16 was
  too tight; Q6.10 keeps int16 with ample headroom).

Per-branch formats cost nothing: the requant multipliers are per-branch
anyway. D1 computes Cox–de Boor / SiLU in FP32 (per-precision-class basis
algorithm, applied to no design's disadvantage — documented in
HW_FAIRNESS.md).

B-spline convention replicated from `funcodekan/models/spline.py`
(`SplineLayer.b_splines`): 12 knots `arange(-3,9)·0.4 − 1.0` ∈ [−2.2, 2.2],
half-open order-0 indicators `[t_i, t_{i+1})`, Cox–de Boor with `eps=1e-8`
in denominators, inputs outside the knot span get all-zero bases (spline
branch contributes 0; SiLU base branch still fires). The LUT builder uses
this exact float32 computation before Q4.12 rounding.

### Integer datapath (D2 and D3)

Per layer l, per output o (all integer):

```
acc_s[o] = sum_i sum_k  qsw[o,i,k] · LUT_B[u_i][k]      (int32)
acc_b[o] = sum_i        qbw[o,i]   · LUT_S[u_i]          (int32)
```

`qsw`/`qbw` are the INT4 weight codes (D2: dense arrays; D3:
`spline_codebook_q[sid[o,i]][k]` / `base_codebook_q[bid[o,i]]` — the ONLY
D2↔D3 difference). Worst-case magnitudes: spline ≈ 8·2^14·784 ≈ 2^27
(Σ_k basis = 1), base ≈ 8·(32·2^10)·784 ≈ 2^31-safe margins checked by the
emulator with hard runtime assertions on the real data.

**Requantization** (hidden layer): real value is
`y = acc_s·(s_ws·2^-14) + acc_b·(s_wb·2^-10)`; the next-layer code is
`q' = clamp(round(y / s_a1), −8, 7)`. Fixed-point form (frozen):

```
M_s = round(s_ws·2^-14 / s_a1 · 2^SHIFT),
M_b = round(s_wb·2^-10 / s_a1 · 2^SHIFT)                      (int32)
SHIFT = largest shift such that max(M_s, M_b) ≤ 2^31 − 1
t   = acc_s·M_s + acc_b·M_b                                   (int64)
q'  = clamp( (t + (1 << (SHIFT−1))) >> SHIFT, −8, 7 )         (round-half-up)
```

**Logits** (output layer): same scheme, no clamp; multipliers target the
frozen logit scale `s_logit = 2^-16` (LOGIT_FRAC_BITS = 16), i.e.
`logit_int32 ≈ logit_real · 2^16`. L2 compares these INT32 logits bit-exactly.
(16 fractional bits chosen so logit-grid ties sit far below real decision
margins; MNIST logits ±50 still fit int32 with 2^15 headroom.)

**Rounding mode everywhere**: round-half-up (`floor(x + 0.5)`), including
input quantization `q = clamp(floor(x/s_a0 + 0.5), −8, 7)`, LUT entry
rounding, and requant. (Not torch's round-half-to-even; the eval-time
reference model uses the HW rounding.)

### D1 (FP32) numeric reference

FP32 dense weights from the verified `dense.pt`; float Cox–de Boor + SiLU;
float accumulate; FP32 logits. Emulator = float32 numpy path; L1 tolerance
≤1e-4 vs torch logits, C-sim tolerance 1e-3, argmax-exact.

### Storage accounting

All storage claims computed by `funcodekan.analysis.storage`
(`compressed_storage_breakdown`, `required_index_bits = ceil(log2 K)`,
`bits_to_kib`) — never re-derived by hand. D3: spline indices 5 b, base
indices 4 b, codebooks at W4, scales FP32 (documented count: one per-tensor
scale per codebook per layer in the HW contract; the paper table reports
the breakdown from the analysis functions plus the actual exported scale
words).

---

---

## Frozen design decisions — GRAM integer datapath (G2/G3)

Everything from the spline contract carries over unchanged (INT4 ranges,
per-tensor LSQ, round-half-up, basis LUT int16 **Q2.14** — gram basis
values lie in [−1.2, 1.2] —, SiLU-of-input LUT int16 Q6.10, int32
accumulators, int32-multiplier requant). New, frozen additions:

1. **Pre-LN combine**: `v[o] = (acc_s·M_s + acc_b·M_b + 2^(s−1)) >> s`
   targeting fixed scale **2^-12** (`PRE_LN_FRAC = 12`), int32, no clamp
   (export asserts range).
2. **Integer LayerNorm** (deterministic, exact centered sums):
   `S = Σv` (int64); `c[o] = N·v[o] − S` (exact); `Q = Σc²` (int64;
   export asserts < 2^62); `R = floor(Q/N) + E` with
   `E = round(N²·2^24·1e-5)` (the layer's eps, precomputed integer);
   `r = isqrt64(R)` (**bit-by-bit restoring integer square root, 32
   iterations** — identical algorithm in torch eval-reference, numpy
   emulator, HLS); normalized `n[o] = (2·c[o]·2^12 + r) div (2r)` (floor
   division = round-half-up), Q4.12.
3. **LN affine**: `G[o] = round(γ[o]·2^12)` (int16 **Q4.12** — trained
   gammas reach ~2.0, beyond Q2.14; export asserts |γ|<8),
   `B[o] = round(β[o]·2^24)` (int32); `y = (n·G + B + 2^11) >> 12` →
   Q.12 int32.
4. **Output SiLU** via a model-independent **257-entry interpolated
   integer table**: domain [−8, 8) step 1/16, entries int16 Q6.10,
   linear interpolation `s = T[i] + ((T[i+1]−T[i])·frac + 128) >> 8`
   (frac = 8 bits). The QAT wrapper trains through this exact
   piecewise-linear SiLU (and through the basis/SiLU input LUT rounding),
   so train == deploy.
5. **Hidden requant**: single multiplier from Q6.10 to the next A4 step;
   **logits**: Q6.10 → 2^-16 is an exact `<< 6`.
6. **Eval-reference rule**: the torch wrapper's eval path uses the
   integer LN + integer SiLU table verbatim (int64 torch ops); float LN
   is used only inside the training forward (differentiability). L1
   compares eval path ↔ emulator.
7. **D3g codebook quantization contract**: per-tensor LSQ on codebook
   entries, indices frozen — IDENTICAL to spline D3 (the verified
   pipeline's per-row PTQ stage remains the software story; the HW
   contract is per-tensor for the same reason as spline: one multiplier
   per branch).
8. G1 (`gram_fp32`) computes tanh/recurrence/LN/SiLU in FP32 (same
   per-precision-class rule as D1; TB tolerance 1e-3, argmax exact).

LN parameter storage (148 FP32 values dense / fixed-point G,B,E words in
the quantized designs) is counted explicitly in all three designs'
parameter-memory column.

## KAGN-Conv architecture (K-series, frozen when trained)

`GramConv2D` (new, additive): per pixel the SAME gram math —
`out = SiLU(BatchNorm2d(base_conv(SiLU(x)) + poly_conv(basis(x))))`,
poly weight `[out_ch, in_ch·4, 3, 3]`, base weight `[out_ch, in_ch, 3, 3]`
(degree 3). MNIST arch (sized for all-on-chip FP32): GramConv2D 1→16 s2 →
14×14; GramConv2D 16→32 s2 → 7×7; global average pool; gram FC head
16→10... (final channel widths frozen after the training run). BatchNorm
folds into per-channel requant constants at inference (no runtime norm in
the conv layers); the FC head reuses the integer-LN datapath. Conv
"edges" = (out_ch, in_ch, kh, kw) positions, each a 4-vector + base
scalar → branch codebooks Ks=32/Kb=16 + 9-bit index streams, mirroring
the FC storage model.
