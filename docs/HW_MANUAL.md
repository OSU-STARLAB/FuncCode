# HW_MANUAL.md — Training → Export → HLS, step by step

A practical operator's manual for the FuncCode-KAN hardware flow. It walks
the complete path: train the three models, export golden vectors and
parameter headers, run Vivado HLS (csim / csynth / cosim), get
post-implementation numbers, and regenerate the paper tables. Companion
documents: `docs/HW_DESIGN_CONTRACT.md` (frozen numeric contract + milestone
log) and `docs/HW_FAIRNESS.md` (what is identical across designs and why).

The flow at a glance:

```
                 verified pipeline                funcodekan/hw
┌──────────────┐  dense.pt   ┌─────────────────┐  model.pt  ┌──────────────┐
│ 1. TRAIN     │────────────>│ 2. QAT/FINETUNE │───────────>│ 3. EXPORT    │
│ hw_prepare   │ clustered_  │ hw_prepare      │            │ hw.export    │
│ branch_source│ finetuned.pt│ lsq_w4a4 /      │            │ (runs L1+L4) │
└──────────────┘             │ funccode_w4a4   │            └──────┬───────┘
                             └─────────────────┘                   │ params.h
                                                                   │ golden.h
                                                                   v manifest
┌──────────────┐  reports    ┌─────────────────┐  RTL       ┌──────────────┐
│ 6. TABLES    │<────────────│ 5. IMPL (P&R)   │<───────────│ 4. VIVADO HLS│
│ 13_reports.sh│ hw/results/ │ impl_eval.tcl   │            │ run_hls.tcl  │
└──────────────┘             └─────────────────┘            │ csim→synth→  │
                                                            │ cosim (L2,L3)│
                                                            └──────────────┘
```

The three designs:

| ID | HLS dir (`hw/hls/`, `hw/golden/`) | Weights | Activations |
|---|---|---|---|
| D1 | `fp32` | FP32 dense ROMs | FP32 |
| D2 | `lsq_w4a4` | INT4 dense (LSQ QAT), nibble-packed ROMs | INT4 |
| D3 | `funccode_w4a4` | INT4 codebooks (Ks=32,Kb=16) + 5b/4b index ROMs | INT4 |

---

## 0. Prerequisites

- Python ≥3.10 with the repo installed: `pip install -e ".[dev]"`
  (torch, torchvision, numpy, pandas, scikit-learn, tqdm, pytest).
  A CUDA GPU makes training ~10× faster; everything also runs on CPU.
- Xilinx **Vivado HLS 2019.1** (`vivado_hls`) and **Vivado** (`vivado`).
  On Windows they are not on PATH by default:
  ```powershell
  $env:PATH += ";C:\Xilinx\Vivado\2019.1\bin"
  ```
- Sanity check before starting:
  ```bash
  python -c "import funcodekan"
  pytest tests/ -q          # all tests must pass before you touch anything
  vivado_hls -version       # expect 2019.x
  ```
- **Windows only:** every training/eval command below must run with
  `--num-workers 0` (the verified MNIST loader uses a lambda transform
  that cannot be pickled to spawn-based DataLoader workers). The
  `hw_prepare` driver already defaults to 0.

All commands run from the repository root. Outputs land in `runs_hw/`
(checkpoints, scales, logs, verification log), `hw/golden/` (exports),
`hw/proj/` (HLS projects, disposable), `hw/results/` (archived reports).

---

## 1. Train the three models

One command runs everything (GPU recommended, ~1.5 h total):

```bash
bash scripts/hw/10_train_models.sh
```

Or step by step, which is what the script does:

### 1a. The source run (dense FP32 + branch-aware clustered model)

```bash
python -m funcodekan.experiments.hw_prepare --model branch_source
```

This delegates to the **verified** pipeline
(`funcodekan.experiments.mnist`) with the paper's branch configuration
(seed 42, 10 dense epochs, branch clustering Ks=32/Kb=16 with
function-space spline clustering, 20 finetune epochs, W8→W2 PTQ sweep).
It writes `runs_hw/branch_k32_s32_b16/`:

| File | Contents | Used by |
|---|---|---|
| `dense.pt` | `{"model": state_dict}` of `DenseSplineKAN(784,64,10,5,3)` | D1 weights, D2 QAT init |
| `clustered_finetuned.pt` | state_dict + `clustered_export` (codebooks + frozen cluster indices) | D3 init |
| `summary.csv` | per-stage accuracy + bit-exact storage | dense reference accuracy for L4 |

Check before proceeding: the `dense_fp32` row in `summary.csv` should be
~95.5–96.5% and the `clustered_hwq_w4` row must show
`compression_vs_dense = 31.64` **exactly** (it is a closed-form number; if
it differs, a flag was wrong).

### 1b. D2 — LSQ W4A4 quantization-aware training

```bash
python -m funcodekan.experiments.hw_prepare --model lsq_w4a4
# options: --epochs 20 (default) --lr 5e-4 --dense-ckpt <path> --run-name <name>
```

What happens: `QuantSplineKAN` (`funcodekan/hw/qat.py`) wraps the
untouched dense model and fake-quantizes exactly four points — layer-1
input (A4), spline coefficients (W4), base weights (W4), inter-layer
activation (A4) — all per-tensor LSQ with learnable steps, **and** rounds
basis/SiLU values to the hardware LUT grids during training, so the
trained network is the deployed network. 20 epochs cosine-LR AdamW,
best-val checkpointing.

Output `runs_hw/d2_lsq_w4a4/`: `model.pt` (wrapper state_dict),
`scales.json` (all LSQ steps), `qat_history.csv`, `summary.csv`.
**Acceptance: test accuracy ≥ 95.2%** (printed at the end and appended to
`runs_hw/verification_log.json`).

### 1c. D3 — FuncCode W4A4 activation-quant finetune

```bash
python -m funcodekan.experiments.hw_prepare --model funccode_w4a4
# options: --epochs 25 --lr 1e-3 --clustered-ckpt <path> --run-name <name>
```

What happens: the branch model is rebuilt from `clustered_export`
(`rebuild_branch_model` in `funcodekan/hw/act_quant.py`), then
`QuantBranchKAN` adds the SAME LSQ quantizers as D2 — but the W4
quantizers act on the **codebooks** (spline `[32,8]`, base `[16]`), and
the cluster **indices stay frozen** (they are buffers; only codebook
entries and LSQ steps train). This is the point of the comparison: D2 vs
D3 isolates codebook sharing, not quantizer tricks.

Practical tuning note from this repo's runs: 10 epochs @5e-4 gave 95.07%,
**25 epochs @1e-3 gave 95.20%** (kept), 50 epochs gave 95.16% — more is
not better; the model saturates. If you retrain and miss 95.2%, try the
25-epoch/1e-3 setting first.

Output `runs_hw/d3_funccode_w4a4/`: same file set as D2, plus the storage
breakdown recomputed with `funcodekan.analysis.storage` (never re-derive
storage by hand — the exactness is a paper claim).

---

## 2. Export golden vectors + parameter headers (and gate on L1)

```bash
bash scripts/hw/11_export_golden.sh          # all three designs
# or per design:
python -m funcodekan.hw.export --design d1   # fp32
python -m funcodekan.hw.export --design d2   # lsq_w4a4
python -m funcodekan.hw.export --design d3   # funccode_w4a4
```

Each invocation does four things, in order:

1. **Builds the integer model** from the checkpoint via
   `funcodekan/hw/fixed_point.py` — INT4 weight codes, the two 16-entry
   LUTs per layer (basis Q2.14, SiLU Q6.10, built through the verified
   `SplineLayer.b_splines` itself so there is a single source of LUT
   truth), and the requant multipliers/shifts. For D1 it builds the FP32
   numpy twin instead.
2. **Runs L1 on the full 10k test set**: emulator argmax must equal the
   PyTorch fake-quant model on 10,000/10,000 images (D1: logits ≤1e-4).
   Also L4 (accuracy vs the dense reference). Both are appended to
   `runs_hw/verification_log.json`. **The command exits non-zero if L1
   fails** — nothing downstream should be run in that case.
3. **Selects the 1000 golden vectors**: the first 100 test images of each
   class in test-set order (deterministic, no RNG), quantizes the inputs
   to INT4 codes (D1: float32 pixels), and computes the golden outputs
   with the emulator (INT32 logits at scale 2^-16; D1: float32 logits).
4. **Writes `hw/golden/<design>/`**:

| File | Contents |
|---|---|
| `params.h` | weights/codebooks/indices/LUTs/requant constants as C arrays, with `HW_STATIC_ASSERT` size checks |
| `golden.h` | `golden_inputs`, `golden_outputs`, `golden_labels` (+ `GOLDEN_N/IN/OUT` macros) for the testbench |
| `golden_*.npy` | the same data for Python-side tests |
| `manifest.json` | shapes, scales, requant constants, SHA256 of every artifact, the L1/L4 results, emulator accuracy |

**Regenerate exports whenever a checkpoint or anything in
`funcodekan/hw/hw_spec.py` changes.** The manifest SHA256s are the
mechanism for knowing what the HLS was verified against — if
`params.h` on disk doesn't hash to the manifest value, re-export.

### What's inside params.h (per design)

- Common per layer: `l{n}_lut_b[16][8]`, `l{n}_lut_s[16]` (int16 LUTs),
  `l{n}_mult_spline`, `l{n}_mult_base`, `l{n}_shift` (requant), dims
  macros `L{n}_IN/OUT`.
- **D1**: `l{n}_weight[out*in][9]` float (spline coeffs + base weight),
  `l{n}_grid[12]` (knot vector).
- **D2**: `l{n}_spline_pk[out*in]` — one uint32 word per edge holding the
  8 INT4 spline coefficients (nibble k at bits `4k+3:4k`), and
  `l{n}_base_pk[ceil(out/8)*in]` — 8 consecutive outputs' base weights
  per word. Decode with `w4_unpack(word, nib)` from `hw_config.h`.
  (Packed POD instead of `ap_int<4>` arrays because both 2019.1 compilers
  fail on ~450k class-type initializers; exactly 4 bits/weight either way.)
- **D3**: `l{n}_spline_codebook_q[Ks][8]`, `l{n}_base_codebook_q[Kb]`
  (INT4, tiny) and the index streams `l{n}_spline_ids[out*in]`
  (`sidxrom_t` = `ap_uint<5>` in synthesis, `unsigned char` in csim) and
  `l{n}_base_ids[out*in]` (4-bit equivalent).

---

## 3. Run Vivado HLS

Everything goes through one Tcl driver:

```bash
vivado_hls -f hw/tcl/run_hls.tcl -tclargs <design> <mode> [cosim_n] [part]
#  design: fp32 | lsq_w4a4 | funccode_w4a4
#  mode:   csim | synth | impl
```

or all at once (csim → csynth+cosim → Vivado P&R → archive):

```bash
bash scripts/hw/12_run_hls.sh                 # all three designs
bash scripts/hw/12_run_hls.sh lsq_w4a4        # just one
```

### 3a. `csim` — C simulation over all 1000 golden vectors (rung L2)

```bash
vivado_hls -f hw/tcl/run_hls.tcl -tclargs lsq_w4a4 csim
```

Creates project `hw/proj/lsq_w4a4_csim/`, compiles the design top +
shared testbench (`hw/tb/tb_main.cpp`) with the design's `golden.h`, runs
all 1000 vectors. The testbench prints one summary line:

```
TB_RESULT n=1000 vectors_with_logit_mismatch=0 argmax_agree=1000 accuracy_pct=96.80
TB_PASS
```

For D2/D3 the comparison is **bit-exact INT32 logits** (any nonzero
`vectors_with_logit_mismatch` fails the run and csim exits non-zero);
for D1 it is float ≤1e-3 with exact argmax. If this fails, do NOT proceed
— see Troubleshooting.

Expected wall time: ~1–3 min per design (most of it compiling the big
generated headers).

### 3b. `synth` — csynth only (fast iteration)

```bash
vivado_hls -f hw/tcl/run_hls.tcl -tclargs funccode_w4a4 synth
```

Creates/resets `hw/proj/funccode_w4a4/`, runs `csynth_design` at
150 MHz on `xczu9eg-ffvb1156-2-e`. Read the results at
`hw/proj/<design>/sol1/syn/report/kan_top_csynth.rpt`:

- **Latency (cycles)** near the top — D2/D3 should read 50,983 fixed;
  D1 is a range (825k–1.06M) because the float adders serialize the loop.
- **Utilization Estimates → Summary** — BRAM_18K / DSP48E / FF / LUT.
- **Per-loop II**: open `layer0_csynth.rpt` / `layer1_csynth.rpt`; the
  `* Loop:` table shows `IN0_OUT0 ... achieved II=1` for D2/D3.

Expected csynth wall time: fp32 ≈1 min, lsq_w4a4 ≈4 min,
funccode_w4a4 ≈3 min.

### 3c. `impl` — csynth + RTL co-simulation (rung L3)

```bash
vivado_hls -f hw/tcl/run_hls.tcl -tclargs lsq_w4a4 impl 50
```

Same as `synth`, then `cosim_design -rtl verilog` with the testbench
compiled at `-DTB_N=50` (the first 50 golden vectors — same subset for
every design; the count is the third argument). The cosim verdict is the
same `TB_RESULT` line plus `*** C/RTL co-simulation finished: PASS ***`,
and the measured RTL latency lands in
`hw/proj/<design>/sol1/sim/report/kan_top_cosim.rpt`.

**Budget cosim time by cycle count**: XSIM runs roughly 0.5–1 ms of
simulated time per minute on a desktop. D2/D3 (51k cycles/vector ×50 ≈
17 ms) finish in ~20–40 min. D1 (~1M cycles/vector ×50 ≈ 330 ms) takes
**~9 hours** — start it last, let it run overnight, or spot-check with
`-tclargs fp32 impl 3` first (3 vectors ≈ 30 min).

Do NOT run two invocations against the same `hw/proj/<design>` project
concurrently. To run a quick job while a long one holds the project,
pass the default part as the 4th argument — that switches to a separate
`<design>_alt` project (this is also how the ZU7EV runs work, below).

### 3d. Post-implementation numbers — `impl_eval.tcl` (NOT export_design)

`export_design` is unusable in Vivado HLS 2019.1 on any machine whose
clock is past 2021 (Y2K22 IP-packager bug → `ERROR: [IMPL 213-28] Failed
to generate IP`). Instead, run plain Vivado out-of-context synthesis +
place + route on the csynth-generated Verilog:

```bash
vivado -mode batch -source hw/tcl/impl_eval.tcl -tclargs lsq_w4a4
```

Requires csynth to have run first (it reads
`hw/proj/<design>/sol1/syn/verilog/`). Writes
`hw/results/<design>/impl/{utilization_route.rpt, timing_route.rpt}` —
true post-route LUT/FF/BRAM/DSP and WNS at the 6.67 ns clock.

**Licensing reality on this class of install (WebPACK):** `xczu9eg` has
no RTL-synthesis license, so the ZU9EG post-impl step will fail with a
license error. The licensed ZU7EV is used for a **secondary D2-vs-D3
post-impl comparison** (D1's FP32 ROMs don't fit that part):

```bash
# csynth for the alternate part goes to hw/proj/<design>_alt
vivado_hls -f hw/tcl/run_hls.tcl -tclargs lsq_w4a4 synth 50 xczu7ev-ffvc1156-2-e
vivado -mode batch -source hw/tcl/impl_eval.tcl -tclargs lsq_w4a4 xczu7ev-ffvc1156-2-e
# reports land in hw/results/lsq_w4a4/impl_alt/
```

`scripts/hw/12_run_hls.sh` runs this automatically for D2 and D3.

### 3e. Archive

`12_run_hls.sh` copies every report into `hw/results/<design>/`
(`syn/`, `sim/`, `syn_alt/`, `impl/`, `impl_alt/`, csim log). The table
tools read ONLY from `hw/results/` so that every published number traces
to an archived file. If you ran stages by hand, re-run the archive block
of `12_run_hls.sh` (or copy `hw/proj/<design>/sol1/{syn,sim}/report/*`
yourself) before generating tables.

---

## 4. Generate the comparison tables

```bash
bash scripts/hw/13_reports.sh
```

which is exactly:

```bash
python tools/parse_hls_reports.py  --results-root hw/results \
    --out runs_hw/tables/hw_report_summary.csv
python tools/make_hw_comparison_table.py \
    --csv runs_hw/tables/hw_report_summary.csv --runs-dir runs_hw \
    --out-tex runs_hw/tables/hw_comparison.tex --out-md docs/HW_COMPARISON.md
```

`parse_hls_reports.py` emits one CSV row per (design, stage):
`post_synthesis` (from `syn/csynth.xml`, plus per-loop IIs from the layer
rpt files), `post_implementation` (ZU9EG route reports, if licensed) and
`post_implementation_zu7ev` (the secondary D2/D3 run). Derived columns:
`latency_us` (worst-case cycles × target period) and `images_per_s`.

`make_hw_comparison_table.py` merges in the accuracy column (taken from
the golden `manifest.json` L1 results — i.e., the accuracy of the
arithmetic actually deployed) and the analytical parameter-storage column,
and emits one LaTeX table + one markdown table **per stage** — stages are
never mixed in one table (fairness rule). Every table footer lists the
source report path per row.

---

## 5. The verification ladder — what to check when

| Rung | Compares | Where it runs | Pass bar |
|---|---|---|---|
| L1 | PyTorch fake-quant ↔ Python emulator, 10k images | `python -m funcodekan.hw.export` (step 2) | argmax 10000/10000 (D1: logits ≤1e-4) |
| L2 | emulator ↔ HLS C-sim, 1000 golden vectors | `run_hls.tcl <d> csim` (step 3a) | bit-exact INT32 logits (D1: ≤1e-3) |
| L3 | C-sim ↔ RTL cosim, ≥50 vectors | `run_hls.tcl <d> impl 50` (step 3c) | same testbench criterion |
| L4 | emulator accuracy vs FP32 dense | automatic during export | Δ ≤ 0.3 pp |

Never advance with a broken lower rung. Every result appends to
`runs_hw/verification_log.json`; read it with:

```bash
python -c "import json; [print(e.get('milestone'), e.get('rung') or e.get('check'), e.get('design'), 'passed=', e.get('passed')) for e in json.load(open('runs_hw/verification_log.json'))]"
```

The CI-fast versions of these checks live in `tests/test_hw_lsq.py`,
`tests/test_hw_emulator.py`, `tests/test_hw_export.py` (synthetic tiny
models, no downloads): `pytest tests/ -q`.

---

## 6. Changing things safely

- **Retrain with different seeds/epochs**: rerun step 1 (the affected
  model), then step 2 (export re-runs L1), then steps 3a→3c per design,
  then step 4. Nothing else.
- **Changing any numeric-contract constant** (`funcodekan/hw/hw_spec.py`:
  LUT formats, LOGIT_FRAC_BITS, rounding, requant scheme) invalidates
  trained checkpoints (QAT trains through the LUT rounding), all golden
  exports and all HLS runs. Change it in `hw_spec.py` AND
  `hw/hls/common/hw_config.h` together, document it in
  HW_DESIGN_CONTRACT/HW_FAIRNESS, retrain, re-export, re-verify. The emulator
  (`fixed_point.py`) is the normative spec — the HLS must match it, never
  the other way around.
- **Editing a top.cpp**: D2 and D3 tops are deliberately line-for-line
  identical except the weight-fetch lines and ROM pragmas. Keep them in
  lockstep or the fairness claim dies. After any edit: csim (L2) before
  synth, cosim (L3) before believing resource numbers.
- **New FPGA part**: change `FPGA_PART` in `hw_spec.py`, the `part`
  variables in `hw/tcl/run_hls.tcl` and `hw/tcl/impl_eval.tcl`, and
  document per the §3 escape clause (same part for ALL designs).

---

## 7. Troubleshooting (all failure modes seen on this project)

| Symptom | Cause | Fix |
|---|---|---|
| `EOFError`/pickle error at first training batch (Windows) | DataLoader workers can't pickle the MNIST lambda transform | pass `--num-workers 0` |
| csim: gcc `internal compiler error: in gt_ggc_m_S` | >~100k `ap_int` initializers in a generated header | already engineered around: weights are packed POD uint32, ids are POD in csim (`*rom_t` typedefs in `hw_config.h`); if you add new big class-typed arrays, don't |
| csynth hangs forever at "Analyzing design file" (0% CPU) | same root cause in the synthesis front end (~450k class initializers) | same fix as above |
| csynth: `error: C++ requires a type specifier` at `static_assert` | synthesis front end is pre-C++11 | use the `HW_STATIC_ASSERT` macro (no-op under `__SYNTHESIS__`) — the export already does |
| `ERROR: [IMPL 213-28] Failed to generate IP` at export_design | Y2K22 date bug in the 2019.1 IP packager | don't use export_design; use `hw/tcl/impl_eval.tcl` |
| `ERROR: [Common 17-345] A valid license was not found ... 'xczu9eg'` | WebPACK install; ZU9EG needs a paid license | post-impl on the licensed ZU7EV (D2/D3 only), post-synth estimates for the 3-way table |
| `ERROR: unknown design '-f'` from run_hls.tcl | `$argv` contains `-f <script>` before user args in 2019.1 | already handled in the script; don't strip the guard |
| `OverflowError: silu Q6.10 LUT exceeds int16` at export | a trained activation step > 32/7 ≈ 4.57 (SiLU value out of LUT range) | inspect `scales.json`; retrain with lower LR/more epochs, or (spec change!) widen the SiLU LUT format in `hw_spec.py` + `hw_config.h` and re-run everything |
| `error: unable to find numeric literal operator 'operator""f'` | a float printed without a decimal point (e.g. `-1f`) in a hand-made header | exporter already guards this (`_fmt_f32`); pattern to copy if you emit floats elsewhere |
| cosim takes forever | wall time ∝ cycles × vectors; D1 is ~1M cycles/vector | subset via the `cosim_n` argument (document N); spot-check with 3 before committing to 50 |
| csim/cosim pass but table numbers look stale | tools read `hw/results/`, not `hw/proj/` | re-archive (rerun `12_run_hls.sh` or copy reports), then `13_reports.sh` |
| L1 argmax mismatches on a handful of images | fake-quant model and emulator disagree at a rounding boundary — usually a drifted constant between `hw_spec.py` and a stale checkpoint/export | re-export; if it persists, diff `manifest.json` scales vs `runs_hw/<run>/scales.json`; as designed, LOGIT_FRAC_BITS=16 makes genuine ties astronomically rare |

## 7b. The GRAM and KAGN-Conv families (2026-07 extension)

> **Full-depth manual for these two families: `docs/HW_MANUAL_GRAM_KAGN.md`**
> (per-step walkthroughs, params.h contents, cosim time budgets, the K2
> from-scratch / K3 cluster-from-K2 protocol rationale, and the
> extension-specific troubleshooting table). The summary below is the
> quick-reference version.

The same train → export → HLS → tables path exists for two more families;
everything in sections 1–6 applies with these substitutions:

| Step | GRAM (G1/G2/G3) | KAGN-Conv (K1/K2/K3) |
|---|---|---|
| Train | `bash scripts/hw/20_train_gram_kagn.sh` (or `python -m funcodekan.experiments.hw_prepare_gram --model gram_source\|gram_lsq_w4a4\|gram_funccode_w4a4`) | same script (or `hw_prepare_kagn --model kagnconv_source\|kagnconv_lsq_w4a4 --from-scratch\|kagnconv_funccode_w4a4 --cluster-from k2`) |
| Export + L1 | `bash scripts/hw/21_export_gram_kagn.sh` (`python -m funcodekan.hw.gram_export --design g1\|g2\|g3`) | same script (`kagn_export --design k1\|k2\|k3`) |
| HLS | `bash scripts/hw/22_run_hls_gram_kagn.sh` — csim(1000) + cosim(50) on ZU9EG **plus ZU7EV post-impl for ALL THREE designs** (their FP32 params fit that part, unlike spline) | same script |

Family-specific protocol notes (why the commands differ from spline):

- **GRAM** layers end in `SiLU(LayerNorm(·))` → the quantized designs use
  the deterministic **integer LayerNorm** + **interpolated integer SiLU
  table** (`gram_spec.py` ⇔ `int_layernorm.h`); QAT trains through the
  table. Eval mode of the wrappers IS the deployed arithmetic — that is
  the accuracy you report.
- **KAGN-Conv is new research code** (the `kans` package is absent; conv
  compression is not in the verified pipeline): `GramConv2D` uses
  **InstanceNorm2d** (BatchNorm collapses train/eval on this small net —
  measured), zero padding contributes nothing, pooling is an integer sum
  folded into the head requant, the FC head reuses the gram integer-LN
  path. **K2 must be trained `--from-scratch`** (W4 PTQ of the FP32 dense
  collapses to ~23%: a quarter of the conv weights round to zero) and
  **K3 clusters K2's W4-native weights** (`--cluster-from k2`, Ks=64).
- New troubleshooting entries: (a) `tanhf` FPO core **deadlocks Verilog
  cosim** (sim time overruns the latency bound with no vector progress) —
  the fp32 designs compute tanh via the `expf` identity
  (`gram_basis_fp32.h`); (b) plain-Vivado impl of fp32 designs needs the
  FPO IP: `impl_eval.tcl` uses an on-disk project and sources the
  HLS-emitted `*_ip.tcl` scripts; (c) cosim prints the C-phase
  `TB_RESULT` BEFORE the RTL runs — only
  `*** C/RTL co-simulation finished: PASS ***` is the RTL verdict.

## 8. Known-good reference results (seed 42, this repo)

For sanity-checking a reproduction, the expected end state:

- Accuracy (full 10k, emulator arithmetic): dense **96.28%**,
  D2 **96.22%**, D3 **95.22%**; L1 argmax 10000/10000 for D2/D3.
- csim: D2/D3 bit-exact 1000/1000; cosim: D1 50/50 (≤1e-3),
  D2/D3 bit-exact 50/50.
- Post-synthesis (xczu9eg, 150 MHz): D1 842 BRAM18 / 48 DSP / 1.06M
  cycles; D2 108 BRAM18 / 24 DSP / 50,983 cycles; D3 **32 BRAM18** /
  23 DSP / 50,983 cycles.
- Post-implementation (xczu7ev): D2 147 BRAM18 vs D3 **38 BRAM18**
  (3.9×), both meeting 150 MHz timing.
- Storage (analytical, must match to the bit): D1 1786.5 KiB,
  D2 223.3 KiB, D3 56.5 KiB (31.64× at the W4 PTQ stage of the source
  run).
- GRAM family (seed 42): dense 96.68%, G2 97.23%, G3 96.09%; post-synth
  ZU9EG BRAM18 483 / 76 / 43; quantized latency 343.6 µs (identical G2/G3);
  ZU7EV post-impl G2 90 vs G3 46 BRAM18, timing met.
- KAGN-Conv family (seed 42): dense 95.43%, K2 92.74% (from-scratch QAT),
  K3 89.78% (Ks=64, clustered from K2); csim bit-exact 1000/1000 for
  K2/K3; the larger quantization/compression cost of the conv family is a
  reported finding, not tuned away.
