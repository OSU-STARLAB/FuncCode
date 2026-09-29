# HW_FAIRNESS.md — the three-way comparison contract

> **2026-07 extension:** the same contract now covers three KAN families —
> Spline (D1–D3), GRAM FC (G1–G3) and KAGN-Conv (K1–K3). Everything in
> this document applies per family; the family-specific addendum at the
> end lists the GRAM/KAGN-specific frozen pieces (integer LayerNorm /
> InstanceNorm, interpolated SiLU table) and the K-series provenance
> caveat. Comparisons are made WITHIN a family (the three designs share
> everything but precision/memory format); cross-family columns are
> reported side by side without claiming identical workloads.

This document freezes what is IDENTICAL across the three HLS designs, what
is ALLOWED to differ (the variables under study), the quantization
contract, the fixed-point formats, and the known threats to validity.
It is part of the paper deliverable. Constants live in
`funcodekan/hw/hw_spec.py` ⇔ `hw/hls/common/hw_config.h` and must never
drift; the emulator (`funcodekan/hw/fixed_point.py`) is the normative
specification of the integer datapath.

## Designs

| ID | Name | Weights | Activations |
|---|---|---|---|
| D1 | `fp32` | FP32 dense ROMs | FP32 |
| D2 | `lsq_w4a4` | INT4 dense ROMs (LSQ QAT) | INT4 (LSQ) |
| D3 | `funccode_w4a4` | INT4 codebooks (Ks=32, Kb=16) + index streams (5 b + 4 b) | INT4 (LSQ) |

## Identical across all three designs

- **FPGA part**: `xczu9eg-ffvb1156-2-e` (ZCU102-class). NOT the ZU7EV of
  the original plan — D1's 1.744 MiB of FP32 constant ROMs needs 397
  BRAM36 (> ZU7EV's 312) and UltraScale+ URAM cannot hold initialized
  ROMs; adjusted for ALL designs identically (HW_DESIGN_CONTRACT.md, M0 note).
- **Clock target**: 150 MHz (6.67 ns), same `create_clock` in one shared
  Tcl driver (`hw/tcl/run_hls.tcl`).
- **I/O interface**: plain-array `ap_memory` ports, one image per
  invocation (batch = 1), `ap_ctrl_hs` control. Element type follows the
  design's activation format (that format IS a variable under study).
- **Layer/loop architecture**: identical `INIT → IN(i) → [basis for x_i] →
  OUT(o) {8-wide MAC + base MAC} → REQ` nest in all three tops; the OUT
  loop is `PIPELINE II=1` and the 8-term spline MAC fully `UNROLL`ed in
  all three. All trip counts compile-time constants (784/64/10).
- **Accumulator organization**: per-output accumulator register arrays
  (`ARRAY_PARTITION complete`), separate spline/base accumulators.
- **Testbench**: one shared `hw/tb/tb_main.cpp` for all designs (only the
  golden data types switch via `DESIGN_FP32`).
- **Golden vector set**: the SAME 1000 stratified MNIST test images (first
  100 per class in test order, deterministic; indices pinned with SHA256
  in `hw/golden/*/manifest.json`). Cosim subset: the first **50** of those
  1000 (`-DTB_N=50`), same for all designs.
- **Toolchain**: Vivado HLS 2019.1, one solution `sol1` per design, same
  script for csim/csynth/cosim/export.

## The ONLY intended differences

1. **Numeric precision** (D1 vs D2): FP32 datapath + runtime Cox–de Boor
   basis/SiLU vs INT4/INT32 datapath + 16-entry basis/SiLU LUTs.
   The LUT trick is legitimate per-precision-class engineering: with A4
   activations a layer input takes ≤16 values, so the LUTs are exact — an
   FP32 design has no such option. The LUT trick is applied to BOTH D2 and
   D3 identically.
2. **Weight-memory format** (D2 vs D3): dense INT4 ROM reads vs
   index-stream → codebook lookup. D2's weight ROMs are nibble-packed POD
   `uint32` words — one word carries the 8 spline coefficients of an edge
   (or 8 consecutive outputs' base weights, zero-padded to the next
   multiple of 8) at EXACTLY 4 bits/weight, so the BRAM bit count equals
   the analytical W4 model. (Motivation: both the csim gcc and the HLS
   2019 synthesis front end fail on ~450k `ap_int<4>` initializers; the
   packed encoding is value-identical, and csim re-verified bit-exact
   1000/1000 after the change.) D3 reads `ap_uint<5>`/`ap_uint<4>` index
   ROMs (POD in csim) and tiny INT4 codebooks. The two top.cpp files are
   line-for-line identical except the fetch lines and the ROM pragmas for
   their respective arrays.

## Quantization contract (D2 and D3, identical)

Fake-quant is applied at exactly four points — what the HW quantizes,
nothing more, nothing less:

| Point | Quantizer | Range | Scale |
|---|---|---|---|
| (a) layer-1 input | LSQ A4 signed | −8..7 | per-tensor `s_a0` |
| (b) spline coefficients / spline codebook | LSQ W4 signed | −8..7 | per-tensor per layer `s_ws{l}` |
| (c) base weights / base codebook | LSQ W4 signed | −8..7 | per-tensor per layer `s_wb{l}` |
| (d) inter-layer activation | LSQ A4 signed | −8..7 | per-tensor `s_a1` |

- Logits are NOT quantized: INT32 fixed-point output at scale 2^-16.
- Per-tensor scales only (HW simplicity). Per-channel was not needed:
  both designs cleared the 95.2% gate without it.
- D3 quantizes CODEBOOK ENTRIES with the same LSQ quantizer class and
  per-tensor granularity as D2's dense weights, and its cluster indices
  are FROZEN buffers throughout finetune — D2 vs D3 therefore isolates
  *codebook sharing*, not quantizer tricks.
- LSQ: STE rounding, step gradient scale g = 1/sqrt(N·Qp), step init
  2·mean|x|/sqrt(Qp). Rounding is round-half-up (floor(x+0.5)) in
  training AND eval AND HW — never torch's half-to-even.
- QAT trains THROUGH the HW LUT rounding (basis/SiLU values are rounded
  to the LUT grids in the QAT forward with STE), so the trained model is
  the deployed model.

## Frozen fixed-point formats

| Item | Format | Rationale |
|---|---|---|
| Basis LUT (16×8/layer) | int16 Q2.14 | bases are a partition of unity in [0,1]; 2^14 always fits int16 |
| SiLU LUT (16/layer) | int16 Q6.10 | covers (−32,32); trained hidden step ≈3.06 ⇒ SiLU ≈21; asserts `s_a < 32/7` |
| Accumulators | int32 | worst-case ≈2^28 with real data; emulator hard-asserts |
| Requant | per-branch int32 multiplier + shared shift | `M = round(ratio·2^shift)`, largest shift with M ≤ 2^31−1; int64 product; `(t + 2^(shift−1)) >> shift` (round-half-up); clamp −8..7 (hidden) |
| Logits | int32, scale 2^-16 | grid ties (1.5e-5) far below decision margins; ±50 range ⇒ 2^15 headroom |

The LUTs are built by `funcodekan.hw.fixed_point.build_layer_luts` using
the verified `SplineLayer.b_splines` itself in float32 (same values the
QAT forward saw), then rounded once — a single source of LUT truth for
QAT, emulator, and the exported `params.h`.

## Verification status (see runs_hw/verification_log.json)

| Rung | D1 | D2 | D3 |
|---|---|---|---|
| L1 (torch ↔ emulator, 10k) | max logit diff ≤1e-4, argmax 10000/10000 ✓ | argmax **10000/10000** ✓ | argmax **10000/10000** ✓ |
| L2 (emulator ↔ csim, 1000) | float ≤1e-3, argmax 1000/1000 ✓ | **bit-exact 1000/1000** ✓ | **bit-exact 1000/1000** ✓ |
| L3 (csim ↔ cosim, 50) | **PASS 50/50** ✓ (≤1e-3/argmax; 8.9 h XSIM; measured RTL latency 983,593 cycles) | **bit-exact 50/50** ✓ | **bit-exact 50/50** ✓ |
| L4 (≤0.3 pp vs FP32 dense) | — | 96.22 vs 96.28 = 0.06 pp ✓ | 95.22 vs 96.28 = 1.06 pp ✗ (analysis below) |

Report stages available: post-synthesis (HLS csynth) on xczu9eg for all
three; post-implementation (Vivado OOC place+route) on the licensed
xczu7ev for **D2 and D3 only** — D1's 421 BRAM36 of FP32 ROMs exceed the
ZU7EV, and this installation has no RTL-synthesis license for the ZU9EG
(WebPACK). D2 147 BRAM18 vs D3 38 BRAM18 post-route (3.9× ≈ the
analytical 4× parameter-memory drop); both meet 150 MHz.

Accuracies (full 10k test set, emulator = deployed arithmetic):
dense FP32 **96.28%**, D2 **96.22%**, D3 **95.22%** (M1/M2 gates ≥95.2% ✓).

### L4 analysis for D3 (documented, not silently relaxed)

The 0.3 pp near-lossless budget assumed a ~95.5% dense reference (the
paper's number); this environment's seed-42 dense run reached **96.28%**
(+0.78 pp vs paper). D3's gap decomposes as: branch-aware clustering
(Ks=32, Kb=16, FP32 codebooks, finetuned) −0.23 pp → 96.05%; W4 codebook
PTQ (the verified pipeline's own stage) −0.50 pp → 95.55%; adding A4
activations + per-tensor W4 LSQ finetune −0.33 pp → 95.22%. The dominant
loss is in the verified W4 compression itself, which this project must
not modify (frozen indices, fixed Ks/Kb are the FuncCode claim). Longer
finetune (50 epochs) did not help (95.16%). Relative to the paper's
95.5% dense reference, D3's 95.22% is within 0.3 pp; relative to this
run's stronger dense it is not. Both numbers are reported; the paper
table reports measured accuracy per design, so no claim depends on
hiding this.

## Threats to validity

- **Cosim subset**: RTL cosim covers 50/1000 vectors (wall-time bound;
  same subset all designs). C-sim covers all 1000 bit-exactly, and the
  RTL is generated from the same source csim compiled.
- **csim POD storage**: the csim compiler (gcc 6.2) cannot compile >100k
  class-type initializers, so large ROMs are `signed/unsigned char` in
  csim and true `ap_int<4>`/`ap_uint<5>` under `__SYNTHESIS__`. Values are
  range-identical; cosim (L3) re-checks the synthesized narrow-ROM RTL
  against the same golden vectors.
- **D1 float accumulation order** differs between numpy emulator, torch,
  and RTL (sequential per-input accumulation); covered by the 1e-3
  tolerance and argmax equality, standard for FP32 comparisons.
- **Achieved II differs for D1** although the II=1 directive is identical:
  the floating-point accumulator update (`acc += ...`) has multi-cycle
  latency, so the scheduler serializes D1's inner loop (csynth: ~1.06M
  cycles vs ~51k for D2/D3). This is an inherent property of the numeric
  format under study, not a directive difference; both the directives and
  the achieved IIs are reported.
- **Latency**: memory/resources are the headline claim; latency is
  reported with the above caveat.
- **Post-synthesis vs post-implementation**: both are reported where
  available, always labelled, never mixed within a comparison
  (`tools/make_hw_comparison_table.py` emits separate tables).
- **Accuracy reference**: the FP32 dense reference in THIS environment
  (96.28%) is 0.78 pp above the paper's (95.50%); see the L4 analysis.

---

# Family addendum: GRAM FC (G1–G3) and KAGN-Conv (K1–K3)

## What carries over unchanged

Part/clock/toolchain/testbench/golden-vector protocol, INT4 ranges,
per-tensor LSQ contract, round-half-up, basis LUT Q2.14 + input-SiLU LUT
Q6.10 (the A4 16-entry LUT trick — GRAM's tanh-polynomial basis is
per-value just like the spline basis), INT32 accumulators, int32-multiplier
requant, identical-loop-skeleton rule between each family's LSQ and
FuncCode designs (fetch lines only), csim(1000)/cosim(50) verification.

## GRAM-specific frozen pieces (docs/HW_DESIGN_CONTRACT.md has the formulas)

- Output `SiLU(LayerNorm(·))` on every layer → **deterministic integer
  LayerNorm** (exact centered sums, bit-by-bit isqrt64, floor-division
  rounding) + **257-entry interpolated integer SiLU table** — one frozen
  implementation trio (torch eval-reference / numpy emulator / HLS
  `int_layernorm.h`), validated bit-exact by the G2/G3 csim runs
  (1000/1000) and G2 cosim (50/50).
- LN affine: gamma int16 **Q4.12** (trained gammas reach ~2.0), beta int32
  Q.24. QAT trains THROUGH the table-SiLU and LUT roundings; float LN is
  used only in the training forward (documented train/eval gap ~1e-3,
  eval = deployed arithmetic = the reported accuracy).

## KAGN-Conv provenance caveat and specifics

- The external `kans` package is absent and the verified pipeline has NO
  conv compression — the K-series model (`funcodekan/hw/kagn_conv.py`),
  its branch clustering, and all K claims are **new additive research
  code**, verified by the same bit-exact ladder but NOT carrying the
  paper-verified-pipeline pedigree. Architecture: GramConv2D 1→16→32
  (3×3, stride 2) + GAP + gram FC head; 25,360 params (99.1 KiB FP32).
- Norm choice: **InstanceNorm2d(affine=True)** (the zoo VGG-KAGN option).
  BatchNorm was tried first and was structurally unstable on this small
  net (train 95%+ / eval collapse via compounding running-stats drift —
  measured). Instance norm has no train/eval
  gap and reuses the integer-LayerNorm kernel per channel over spatial
  positions (N=196/49, exact centered sums stay inside int64).
- Zero padding contributes NOTHING (the float model zero-pads the
  basis/SiLU maps; integer designs skip out-of-range taps).
- Pooling: integer sum of the 7×7 codes; 1/49 folds into the head-input
  requant multiplier.
- Honest finding: branch compression costs more here (dense 95.43% →
  branch-FT 91.15% at Ks=32/Kb=16, vs −0.3/−0.7 pp for the FC families) —
  reported as-is; Ks/Kb kept at the family-consistent (32,16).
