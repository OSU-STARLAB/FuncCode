# HW_MANUAL_GRAM_KAGN.md — GRAM & KAGN-Conv: Training → Export → HLS

The operator's manual for the two extension families, at the same depth as
`HW_MANUAL.md` (the Efficient-KAN/spline manual — read its sections 0, 3
and 5 first; everything there about the toolchain, the Tcl driver, the
verification ladder and the report pipeline applies here unchanged).
Companion documents: `HW_DESIGN_CONTRACT.md` (frozen numeric contracts),
`HW_FAIRNESS.md` (family addendum), .

The flow per family is identical in shape to the spline flow:

```
1. TRAIN                      2. EXPORT (runs L1+L4)       3. VIVADO HLS
hw_prepare_gram / _kagn  -->  gram_export / kagn_export -->  run_hls.tcl
runs_hw/<family>_source/      hw/golden/<design>/           csim -> synth ->
runs_hw/g2_*, g3_*, k2_*,       params.h, golden.h,         cosim -> ZU7EV
k3_*  (model.pt, scales)        .npy, manifest.json         impl_eval.tcl
                                                            -> 13_reports.sh
```

The six designs and their directory names (used consistently for
`hw/hls/<design>/`, `hw/golden/<design>/`, `hw/proj/<design>*`,
`hw/results/<design>/`, and as the `run_hls.tcl` design argument):

| ID | Design dir | Weights | Activations |
|---|---|---|---|
| G1 | `gram_fp32` | FP32 dense [64·784·5 + 10·64·5] | FP32 |
| G2 | `gram_lsq_w4a4` | INT4 dense (LSQ QAT), nibble-packed | INT4 |
| G3 | `gram_funccode_w4a4` | INT4 codebooks Ks=32/Kb=16 + 5b/4b ids | INT4 |
| K1 | `kagnconv_fp32` | FP32 conv 1→16→32 + gram FC head | FP32 |
| K2 | `kagnconv_lsq_w4a4` | INT4 dense (QAT **from scratch**) | INT4 |
| K3 | `kagnconv_funccode_w4a4` | INT4 codebooks **Ks=64**/Kb=16 + 6b/4b ids | INT4 |

---

## 0. What is different from the spline family (read this first)

Two datapath elements are new, and two protocol rules are different:

1. **GRAM layers end in `SiLU(LayerNorm(·))`** — on every layer, including
   the 10-way output. The quantized designs therefore contain a
   **deterministic integer LayerNorm** (exact centered sums, bit-by-bit
   `isqrt64`, floor-division rounding; γ int16 Q4.12, β int32 Q.24, the
   eps folded to `round(N²·2^24·1e-5)`) followed by a **257-entry
   interpolated integer SiLU table** (domain [−8,8) step 1/16, entries
   Q6.10, 8-bit-fraction lerp). One frozen implementation trio:
   `funcodekan/hw/gram_spec.py` (torch eval-reference) ⇔
   `funcodekan/hw/gram_fixed_point.py` (numpy emulator) ⇔
   `hw/hls/common/int_layernorm.h` (HLS). Never change one without the
   others. The GRAM basis itself is per-value
   (`P0=1, P1=tanh(x), Pd = z·P(d−1) − 0.1·P(d−2)`, basis_dim=4), so the
   A4 16-entry LUT trick works exactly as for splines — just with 4 MAC
   lanes instead of 8, and the basis LUT stays Q2.14 (gram basis values
   lie in [−1.2, 1.2]).
2. **KAGN-Conv is new research code**, not verified-pipeline pedigree: the
   external `kans` package is absent and the verified pipeline has no
   conv compression. `funcodekan/hw/kagn_conv.py` implements `GramConv2D`
   (the conv realization of the same gram layer math) with
   **InstanceNorm2d(affine=True)** — BatchNorm collapses train/eval on
   this small net (measured) while instance norm
   has zero train/eval gap and reuses the SAME integer-LN kernel per
   channel over spatial positions (N = 196 for conv1, 49 for conv2).
   Its claims rest on the same bit-exact ladder, and that provenance
   difference is stated in HW_FAIRNESS.md.
3. **K2 must be trained `--from-scratch`.** W4 PTQ of the FP32-trained
   dense conv model collapses to 23.4% (≈25% of the conv weights round to
   zero and this 25k-parameter GAP-head net has no redundancy to absorb
   it —). QAT from random init lets weights and
   LSQ steps co-adapt to the W4 grid.
4. **K3 clusters K2's W4-native weights** (`--cluster-from k2`) at
   **Ks=64** (32 gave 87.79%, 64 gives 89.78%; 40 epochs saturates) —
   per-family K selection is the paper's own protocol; the index stream
   grows 5→6 bits per poly edge (`kidxrom_t`/`kidx_t` types).

Eval-mode of every quantized wrapper IS the deployed integer arithmetic —
validation/test numbers during QAT are already deployed-accuracy, and the
KAGN wrappers match the emulator EXACTLY (0.0 logit difference).

---

## 1. Train the models

Everything (both families, all six models, in dependency order):

```bash
bash scripts/hw/20_train_gram_kagn.sh
```

### 1a. GRAM source run (dense + verified branch compression)

```bash
python -m funcodekan.experiments.hw_prepare_gram --model gram_source
```

Delegates to the VERIFIED `all_kan_mnist` driver (variants=gram,
methods=branch, clusters 32 → Kb auto = 16, seed 42, 10 dense + 20
finetune epochs, W8→W2 PTQ sweep). Writes `runs_hw/gram_hw_source/gram/`:

| File | Contents | Feeds |
|---|---|---|
| `dense.pt` | `{"model": state_dict}` of `DirectKANVariant("gram",784,64,10,degree=3)` | G1 weights, G2 QAT init |
| `branch_k32/clustered_finetuned.pt` | `BranchCodebookKAN` state_dict (codebook params `spline_codebooks.{i}` + frozen id buffers `spline_cluster_ids_{i}` + `layers.{i}.norm.*` LN params) | G3 init |
| `summary.csv` | stages `dense_fp32` … `clustered_hwq_w4` | dense reference accuracy |

Reference: dense **96.68%**; branch (32,16) W4 PTQ 96.36% at 17.6×.

### 1b. G2 — LSQ W4A4 QAT

```bash
python -m funcodekan.experiments.hw_prepare_gram --model gram_lsq_w4a4
# options: --epochs 20 (default) --lr 5e-4 --dense-ckpt <path>
```

`QuantGramKAN` (funcodekan/hw/gram_qat.py) wraps the untouched dense
model: A4 LSQ at layer inputs + inter-layer, per-tensor W4 LSQ on basis
coefficients and base weights, Q2.14/Q6.10 LUT rounding via STE, Q.12
grid on the pre-LN value, float LN in the training forward, and the
table-SiLU (float twin) — while **eval mode runs the integer LN + integer
table SiLU** (the deployed arithmetic). Output
`runs_hw/g2_gram_lsq_w4a4/`: `model.pt`, `scales.json`, `qat_history.csv`,
`summary.csv`. Reference: **97.23%** (0.55 pp ABOVE dense — the table-SiLU
and LUT rounding act as regularizers here).

### 1c. G3 — FuncCode W4A4 codebook finetune

```bash
python -m funcodekan.experiments.hw_prepare_gram --model gram_funccode_w4a4
# options: --epochs 25 (default) --lr 5e-4 --clustered-ckpt <path>
```

`rebuild_gram_branch` reconstructs the verified `BranchCodebookKAN` from
the state_dict, then `QuantGramBranchKAN` adds the SAME quantizers as G2 —
the W4 LSQ acts on the CODEBOOKS (per-tensor, identical contract to
spline D3), indices stay frozen buffers. Output
`runs_hw/g3_gram_funccode_w4a4/`. Reference: **96.09%** (Δ0.59 pp vs
dense; the 1 pp family gate passes, the 0.3 pp L4 is recorded as fail —
same honest pattern as spline D3).

### 1d. KAGN source run (dense conv + reference clustering)

```bash
python -m funcodekan.experiments.hw_prepare_kagn --model kagnconv_source
# options: --channels 16 32  --dense-epochs 20  --finetune-epochs 15
```

Trains `KagnConvNet` (GramConv2D 1→16 s2 → 14×14, 16→32 s2 → 7×7, global
average pool, gram FC head 32→10; 25,360 params = 99.1 KiB FP32; data via
`get_dataset_bundle("mnist", flatten=False)` — same normalization as the
verified loader) with cosine LR, then branch-clusters and finetunes as a
REFERENCE point. Writes `runs_hw/kagnconv_hw_source/{dense.pt,
clustered_finetuned.pt, summary.csv}`. Reference: dense **95.43%**.
NOTE: the source clustering (from FP32-dense weights) is kept for the
record, but K3 does NOT use it — see 1f.

### 1e. K2 — W4A4 QAT from scratch

```bash
python -m funcodekan.experiments.hw_prepare_kagn --model kagnconv_lsq_w4a4 --from-scratch
# --epochs 30 (default when --from-scratch)  --qat-lr 5e-4
```

`QuantKagnConvNet`: A4 at input / after conv1 / after conv2 / pooled head
input (4 quantizers), per-tensor W4 on each stage's poly + base weights
(6 quantizers), LUT-rounded basis/SiLU maps, float InstanceNorm +
table-SiLU in training — integer instance norm + integer SiLU in eval.
Zero padding contributes nothing (the float model zero-pads the
basis/SiLU MAPS). Output `runs_hw/k2_kagnconv_lsq_w4a4/`.
Reference: **92.74%** (Δ2.69 pp — the intrinsic W4A4 cost on this small
conv net; do NOT initialize from `dense.pt`, that path starts at 23% and
caps at ~91%).

### 1f. K3 — FuncCode W4A4, clustered from K2

```bash
python -m funcodekan.experiments.hw_prepare_kagn --model kagnconv_funccode_w4a4 --cluster-from k2 --ks 64 --kb 16
# --epochs 25 (default; 40 saturates at the same accuracy)
```

Loads K2's trained net (the `net.*` keys of its `model.pt`), clusters
each stage's edges — conv "edges" are (out_ch, in_ch, ky, kx) kernel
positions carrying 4 poly coefficients + 1 base scalar, clustered by
function-space signatures on the grid domain exactly like FC edges —
freezes the indices, and finetunes codebooks + LSQ steps.
Output `runs_hw/k3_kagnconv_funccode_w4a4/`. Reference: **89.78%**
(K2→K3 = 2.96 pp; the conv family's larger sharing cost is a reported
finding, not tuned away).

---

## 2. Export golden vectors + parameter headers (runs the L1/L4 gates)

```bash
bash scripts/hw/21_export_gram_kagn.sh        # all six designs
# per design:
python -m funcodekan.hw.gram_export --design g1|g2|g3
python -m funcodekan.hw.kagn_export --design k1|k2|k3
```

Each invocation: rebuilds the wrapper from its checkpoint → builds the
integer model (`gram_fixed_point.py` / `kagn_fixed_point.py`) → runs
**L1 on the full 10k test set** (argmax must match 10,000/10,000 for the
quantized designs; the command exits non-zero on failure — nothing
downstream may run) → picks the SAME deterministic 1000 stratified golden
vectors (first 100 per class) → writes `hw/golden/<design>/` with SHA256
manifest. Re-export whenever a checkpoint or any constant in
`hw_spec.py`/`gram_spec.py` changes.

### What's inside params.h

Common to all quantized designs: the 257-entry `silu_tab` (generated,
identical everywhere), per-layer 16-entry `l{n}_lut_b[16][4]` (Q2.14
gram basis at the layer's dequant points) and `l{n}_lut_s[16]` (Q6.10
SiLU), requant multipliers/shift to the Q.12 pre-norm grid, and the
activation requant (`act_mult`/`act_shift`).

- **G2**: `l{n}_basis_pk[out*in]` — **uint16** word per edge, 4 nibbles =
  the 4 INT4 basis coefficients (exactly 4 b/coefficient);
  `l{n}_base_pk[⌈out/8⌉*in]` uint32 (8 consecutive outputs' base nibbles);
  LN constants `l{n}_gamma_q[out]` (lut_t Q4.12), `l{n}_beta_q[out]`
  (int Q.24), `l{n}_eps_int`.
- **G3**: `l{n}_basis_codebook_q[32][4]` + `l{n}_base_codebook_q[16]`
  (w4_t) and index streams `l{n}_basis_ids` (`sidxrom_t`: ap_uint<5> in
  synthesis, uchar in csim) / `l{n}_base_ids` (`bidxrom_t`), plus the
  same LN constants.
- **G1**: `l{n}_weight[out*in][5]` float (4 basis + 1 base per edge),
  `l{n}_gamma`/`l{n}_beta` float.
- **K2**: per conv stage `c1_/c2_` blocks (LUTs, `mult_poly`/`mult_base`/
  `shift` — per-tensor now, instance norm needs no BN fold —, `gamma_q`/
  `beta_q`/`eps_int` of the stage's position count 196/49, act requant),
  weights `c{n}_poly_pk` (uint16/edge) + `c{n}_base_pk` (uint32 groups of
  8 output channels, column index i*9+ky*3+kx); `pool_mult`/`pool_shift`
  (the 1/49 folded with the head step); and the `hd_*` head block
  (gram-FC layout, HD_BASE_GROUPS etc.).
- **K3**: same blocks with `c{n}_poly_codebook_q[64][4]`,
  `c{n}_base_codebook_q[16]`, `c{n}_poly_ids` (**`kidxrom_t`** —
  ap_uint<6> in synthesis, for Ks=64) and `c{n}_base_ids`; head
  `hd_basis_codebook_q`/`hd_basis_ids` likewise.
- **K1**: `c{n}_weight[out*in*9][5]` float, `c{n}_gamma`/`c{n}_beta`
  (instance-norm affine), head floats.

Golden vectors: quantized designs get INT4 input codes (`in_code_t`,
KAGN images flattened row-major to 784) and INT32 logits at 2^-16;
fp32 designs get float32 inputs/logits (TB tolerance 1e-3).

---

## 3. Run Vivado HLS

Same driver and modes as the spline family — only the design names change:

```bash
vivado_hls -f hw/tcl/run_hls.tcl -tclargs <design> csim          # L2: all 1000 vectors
vivado_hls -f hw/tcl/run_hls.tcl -tclargs <design> synth         # csynth only
vivado_hls -f hw/tcl/run_hls.tcl -tclargs <design> impl 50       # csynth + RTL cosim (L3)
vivado_hls -f hw/tcl/run_hls.tcl -tclargs <design> synth 50 xczu7ev-ffvc1156-2-e   # ZU7EV csynth (project <design>_alt)
vivado -mode batch -source hw/tcl/impl_eval.tcl -tclargs <design> [xczu7ev-ffvc1156-2-e]
```

or everything (csim → cosim → ZU7EV route → archive) per design:

```bash
bash scripts/hw/22_run_hls_gram_kagn.sh                   # all six
bash scripts/hw/22_run_hls_gram_kagn.sh gram_lsq_w4a4     # just one
```

**Pass criteria are unchanged**: quantized designs must print
`vectors_with_logit_mismatch=0` (bit-exact INT32 logits) in csim on all
1000, fp32 designs pass at 1e-3/argmax-exact. **Only the line
`*** C/RTL co-simulation finished: PASS ***` is the RTL cosim verdict** —
cosim prints the C-phase `TB_RESULT` BEFORE the RTL simulation runs
(don't be fooled).

**Cosim time budgeting** (measured):

| Design | cycles/vector | 50-vector wall time |
|---|---|---|
| gram_lsq / gram_funccode | ~51.5k | ~25–40 min |
| gram_fp32 | ~290k | ~2.5 h |
| kagnconv_lsq / _funccode | ~294k | ~1.5–2.5 h |
| kagnconv_fp32 | ~1.87M | **~20+ h** — start last / overnight; spot-check with `impl 3` first |

**ZU7EV post-implementation covers ALL SIX designs** — unlike spline,
both families' FP32 parameter sets (992.5 / 99.1 KiB) fit the licensed
part. `impl_eval.tcl` handles the fp32 designs' floating-point-operator
IP automatically (it uses an on-disk project and sources the HLS-emitted
`*_ip.tcl` scripts — problem #21; the in-memory flow fails with
"module kan_top_ap_faddfsub... not found").

Reports land where the spline flow puts them
(`hw/results/<design>/{syn, sim, syn_alt, impl, impl_alt}`), and the SAME
one command regenerates all tables across all nine designs:

```bash
bash scripts/hw/13_reports.sh
```

---

## 4. Reading the results (known-good reference, seed 42)

Post-synthesis (xczu9eg, 150 MHz target):

| Design | Acc % | Params KiB | BRAM18 | DSP | Latency |
|---|---|---|---|---|---|
| gram_fp32 | 96.68 | 992.5 | 469 | 62 | 1,934 µs |
| gram_lsq_w4a4 | 97.23 | 124.1 | 76 | 52 | 343.6 µs |
| gram_funccode_w4a4 | 96.09 | 56.3 | 43 | 51 | 343.6 µs |
| kagnconv_fp32 | 95.43 | 99.1 | 96 | 109 | 12,525 µs (clock est. 11.4 ns — misses 150 MHz; reported honestly) |
| kagnconv_lsq_w4a4 | 92.74 | 12.4 | 41 | 84 | 1,963 µs |
| kagnconv_funccode_w4a4 | 89.78 | 7.5 | 36 | 82 | 1,963 µs |

ZU7EV post-implementation (timing met at ~5.9–6.3 ns except as noted):
gram 608 / 90 / 46 BRAM18; kagn 38 / 32 BRAM18 (K2/K3). Identical
latency within each family's quantized pair — the fairness signature.
Note the GRAM quantized designs' LUT/FF (~32k) is dominated by the
integer LN (two isqrt64 blocks) + table-SiLU logic — expected, honest.

Verification expectations: L1 argmax 10,000/10,000 (quantized), csim
bit-exact 1000/1000, cosim `finished: PASS` on 50 vectors; for KAGN the
torch-eval ↔ emulator match is EXACT (0.0 logit difference) — if a test
shows nonzero, a constant has drifted between `gram_spec.py` /
`kagn_fixed_point.py` / `int_layernorm.h`.

---

## 5. Changing things safely

- **Retrain**: rerun the affected step of section 1, then the design's
  export (section 2), then csim → cosim → route, then `13_reports.sh`.
  K3 depends on K2's checkpoint — retraining K2 invalidates K3.
- **Numeric-contract constants** (`gram_spec.py`: PRE_LN_FRAC, NORM_FRAC,
  GAMMA_FRAC, SILU_TAB_*, LOGIT_SHIFT; `hw_spec.py` as before): change in
  the Python spec AND `hw/hls/common/int_layernorm.h` (and hw_config.h
  types) together, retrain (QAT trains through the tables), re-export,
  re-verify. History: GAMMA_FRAC moved Q2.14→Q4.12 when a trained γ hit
  2.005 — the export assert catches this class of drift.
- **Ks/Kb for K3**: `--ks/--kb` flags; Ks>32 for conv poly ids requires
  the 6-bit `kidxrom_t` path (already wired); Ks>64 would need a 7-bit
  type + the same three-file update. The exporter infers Ks/Kb from the
  checkpoint shapes.
- **Conv architecture** (`--channels`): changes `CHANNELS` in
  kagn_export.py, the dims macros, and the HLS tops' latency — the tops
  are dimension-macro-driven, but re-verify the int32/int64 overflow
  asserts (emulator raises on violation).
- The lockstep rule holds per family: `gram_lsq_w4a4/top.cpp` ⇔
  `gram_funccode_w4a4/top.cpp` and `kagnconv_lsq_w4a4/top.cpp` ⇔
  `kagnconv_funccode_w4a4/top.cpp` differ ONLY in weight-fetch lines and
  ROM pragmas.

## 6. Troubleshooting (extension-specific; spline manual §7 still applies)

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError ... circular import (funcodekan.compression.clustering)` | pre-existing `compression`↔`models` cycle when compression is imported first | `import funcodekan.models` BEFORE any compression import (the new drivers carry the guard) |
| `NotImplementedError: "host_softmax" not implemented for 'Long'` | integer eval logits fed to the verified evaluate's cross-entropy | wrappers return logits as EXACT floats at real scale (int·2⁻¹⁶ < 2²⁴ is lossless) — pattern to copy for new families |
| `AssertionError: gamma exceeds int16 Q4.12` at export | a trained LN/instance-norm γ beyond ±8 | widen GAMMA_FRAC in the three implementations together, retrain not needed (γ-quant is eval-only), re-export + re-verify L1 |
| Dense conv trains to 95% but eval collapses erratically | BatchNorm running-stats pathology compounding across layers | use the shipped InstanceNorm `GramConv2D`; don't reintroduce BN |
| K2-from-dense starts at ~23% and caps ~91% | FP32-dense conv weights are W4-hostile (≈25% round to zero) | `--from-scratch` (and cluster K3 from K2) |
| RTL cosim burns sim-time far beyond `cycles × 6.67 ns × N` with no vector progress | `tanhf` FPO core deadlock (fp32 designs) | tanh is computed via the expf identity in `gram_basis_fp32.h`; if you add new float cores, spot-check cosim with `impl 3` before a 50-vector run |
| `module 'kan_top_ap_f...' not found` in Vivado route | float-operator IP not generated | `impl_eval.tcl` already sources the `*_ip.tcl` scripts in an on-disk project — don't revert it to in-memory mode |
| A cosim "passed" suspiciously fast / early TB_RESULT | that's the C-capture phase, not the RTL | only `*** C/RTL co-simulation finished: PASS ***` counts |
| `OverflowError: int LayerNorm variance exceeds 2^62` | pre-norm values too large for the exact centered-sum trick at that N | lower PRE_LN_FRAC for that norm site (three-file change) or reduce N per norm group; current formats hold for N ≤ ~500 at the observed value ranges |
| Emulator↔wrapper mismatch in KAGN tests (expected 0.0) | constant drift between gram_spec / kagn_fixed_point / int_layernorm.h | diff the frozen constants; the L1 gate will also fail — do not proceed to HLS |
