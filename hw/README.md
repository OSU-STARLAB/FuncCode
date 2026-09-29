# Hardware track — FPGA accelerators for FuncCode-KAN

This directory holds the FPGA half of the paper: nine Vivado HLS accelerators
covering three model families × three weight formats, the exported weights and
golden vectors that drive them, the archived synthesis and place-and-route
reports behind the appendix hardware tables, and the build scripts that
regenerate all of it.

The claim being tested is narrow and mechanical. Within each family the two
quantized designs differ **only in how a weight is fetched** — one reads an INT4
weight from a ROM, the other reads a small index and looks the value up in a
codebook. Everything else (part, clock, directives, dataflow, activation
precision, testbench) is identical and frozen. Any resource difference between
them is therefore caused by the memory format and nothing else, and the
identical cycle counts are the signature that the comparison is fair.

---

## Contents

1. [The nine designs](#1-the-nine-designs)
2. [Directory layout](#2-directory-layout)
3. [What you can check without an FPGA toolchain](#3-what-you-can-check-without-an-fpga-toolchain)
4. [Prerequisites for the full flow](#4-prerequisites-for-the-full-flow)
5. [The full flow, end to end](#5-the-full-flow-end-to-end)
6. [The verification ladder](#6-the-verification-ladder)
7. [The frozen contract](#7-the-frozen-contract)
8. [Reading the results](#8-reading-the-results)
9. [What ships and what is regenerated](#9-what-ships-and-what-is-regenerated)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. The nine designs

| ID | Design dir | Model | Weights | Acts |
|---|---|---|---|---|
| D1 | `fp32` | Spline KAN 784→64→10 (G=5, k=3) | FP32 dense | FP32 |
| D2 | `lsq_w4a4` | same | INT4 dense (LSQ QAT) | INT4 |
| D3 | `funccode_w4a4` | same | INT4 codebooks Ks=32/Kb=16 + 5b/4b indices | INT4 |
| G1 | `gram_fp32` | GRAM KAN 784→64→10 (degree 3) | FP32 dense | FP32 |
| G2 | `gram_lsq_w4a4` | same | INT4 dense (LSQ QAT) | INT4 |
| G3 | `gram_funccode_w4a4` | same | INT4 codebooks Ks=32/Kb=16 + 5b/4b indices | INT4 |
| K1 | `kagnconv_fp32` | GramConv2D 1→16→32 (3×3, s2) + GAP + GRAM head | FP32 dense | FP32 |
| K2 | `kagnconv_lsq_w4a4` | same | INT4 dense (QAT from scratch) | INT4 |
| K3 | `kagnconv_funccode_w4a4` | same | INT4 codebooks Ks=64/Kb=16 + 6b/4b indices | INT4 |

Edge and parameter counts: spline 50,816 edges × 9 = 457,344 parameters;
GRAM 50,816 × 5 = 254,080; KAGN-conv 5,072 kernel-position edges × 5 = 25,360.
Seed 42 throughout, MNIST.

Within each family, D2/D3, G2/G3 and K2/K3 have **identical worst-case cycle
counts** and II=1 fused main loops. That is the fairness signature: the only
thing that moves is BRAM.

---

## 2. Directory layout

```text
hw/
├── hls/                      HLS C++ sources (the designs themselves)
│   ├── common/               shared, frozen numeric contract
│   │   ├── hw_config.h         ap_int typedefs, part, clock, LUT formats
│   │   ├── bspline_fp32.h      Cox–de Boor basis, FP32 reference path
│   │   ├── gram_basis_fp32.h   Gram polynomial basis, FP32 reference path
│   │   ├── int_layernorm.h     integer LayerNorm datapath
│   │   └── requant.h           INT32 → INT4 requantization
│   └── <design>/top.cpp      one top-level per design (9 of them)
├── tb/tb_main.cpp            testbench, shared by every design
├── tcl/
│   ├── run_hls.tcl           csim / csynth / cosim driver
│   └── impl_eval.tcl         out-of-context Vivado synth + P&R
├── golden/<design>/
│   ├── params.h              exported weights / codebooks / indices / LUTs
│   ├── golden.h              1000 stratified test vectors, as C arrays
│   ├── golden_{inputs,outputs,labels,indices}.npy   the same, as arrays
│   └── manifest.json         accuracies, shapes, SHA256 pins
├── results/<design>/
│   ├── syn/                  post-synthesis reports, xczu9eg
│   ├── syn_alt/              post-synthesis reports, xczu7ev
│   ├── sim/                  C/RTL co-simulation reports
│   ├── impl_alt/             post-place-and-route utilization + timing
│   └── kan_top_csim.log      C simulation log
└── verification_log.json     append-only L1–L4 record, incl. superseded runs
```

The Python side lives outside this directory:

| path | role |
|---|---|
| `funcodekan/hw/` | export code, fixed-point emulator, QAT/LSQ, verification |
| `funcodekan/experiments/hw_prepare*.py` | model preparation drivers |
| `scripts/hw/` | the numbered pipeline (train → export → HLS → reports) |
| `tools/parse_hls_reports.py` | reports → tidy CSV |
| `tools/make_hw_comparison_table.py` | CSV → LaTeX + Markdown tables |
| `configs/hw/` | exact settings of every model-preparation run |

Key modules in `funcodekan/hw/`:

| module | role |
|---|---|
| `hw_spec.py`, `gram_spec.py` | the frozen constants — part, clock, bit widths, LUT formats |
| `fixed_point.py`, `gram_fixed_point.py`, `kagn_fixed_point.py` | the **normative** integer datapath (numpy emulator) |
| `qat.py`, `gram_qat.py`, `kagn_qat.py`, `lsq.py`, `act_quant.py` | fake-quant wrappers and LSQ training |
| `export.py`, `gram_export.py`, `kagn_export.py` | emit `params.h`, `golden.h`, `.npy`, `manifest.json` |
| `verify.py`, `veriflog.py` | the verification ladder and its append-only log |
| `kagn_conv.py` | `GramConv2D`, the conv KAGN layer of the K-series |

`fixed_point.py` is the specification, not a convenience: the HLS C++ is
written to match it, and L1/L2/L3 all assert agreement against it.

---

## 3. What you can check without an FPGA toolchain

Most of the hardware claims are checkable with nothing but Python. The
emulator is the reference implementation of the deployed arithmetic, so it can
be run and tested directly:

```bash
pytest tests/test_hw_emulator.py tests/test_hw_export.py \
       tests/test_hw_lsq.py tests/test_hw_gram.py tests/test_hw_kagn.py -q
```

These cover the integer datapath, the requantization scheme, LSQ gradients,
the export format, and round-tripping of the emitted headers — no Vivado, no
GPU, no downloads.

The archived reports are plain text and can be read directly:

```bash
python tools/parse_hls_reports.py --results-root hw/results \
  --out runs_hw/tables/hw_report_summary.csv
python tools/make_hw_comparison_table.py \
  --csv runs_hw/tables/hw_report_summary.csv --runs-dir runs_hw \
  --out-tex runs_hw/tables/hw_comparison.tex --out-md docs/HW_COMPARISON.md
# or simply:
bash scripts/hw/13_reports.sh
```

This regenerates the appendix hardware tables from the shipped `hw/results/`
reports, so the numbers in the paper can be traced back to raw tool output
without rebuilding anything.

The deployed accuracies are pinned in `hw/golden/<design>/manifest.json`, and
the full L1–L4 record — including attempts that were superseded — is in
`hw/verification_log.json`.

---

## 4. Prerequisites for the full flow

| requirement | notes |
|---|---|
| **Vivado HLS 2019.1** | The designs are written against this version. `vivado_hls` must be on `PATH`. |
| **Vivado 2019.1** | For post-implementation numbers, via `impl_eval.tcl`. |
| xczu9eg-ffvb1156-2-e | Primary part (ZCU102-class, 912 BRAM36) — chosen so even the largest FP32 baseline fits on chip. |
| xczu7ev-ffvc1156-2-e | Secondary part for post-implementation. **D1 does not fit**: its 1.744 MiB of FP32 constant ROMs need 397 BRAM36 and the part has 312. |
| Clock target | 6.67 ns (150 MHz), identical for every design. |
| GPU | Only for retraining. Not needed to build or simulate. |

Post-implementation numbers come from `impl_eval.tcl` running Vivado
out-of-context on the csynth-generated Verilog, rather than from
`export_design`, because the 2019.1 IP packager hits a date-overflow bug on
modern systems (`ERROR: [IMPL 213-28] Failed to generate IP`). It is the same
netlist.

**Post-synthesis and post-implementation numbers are never mixed inside a
single comparison.**

---

## 5. The full flow, end to end

Everything below is already done and shipped; run it only to reproduce from
scratch. Steps 1–2 need a GPU and rewrite `hw/golden/`; step 3 needs Vivado.

```bash
# 1. Prepare models (GPU). Spline family D1/D2/D3:
bash scripts/hw/10_train_models.sh
#    GRAM (G-series) and KAGN-Conv (K-series):
bash scripts/hw/20_train_gram_kagn.sh

# 2. Export weights + golden vectors, and run the L1/L4 gates.
#    Fails hard if any L1 rung fails.
bash scripts/hw/11_export_golden.sh        # d1 d2 d3
bash scripts/hw/21_export_gram_kagn.sh     # g1..g3, k1..k3

# 3. HLS: csim (L2) -> csynth + cosim (L3) -> Vivado P&R; archive reports.
bash scripts/hw/12_run_hls.sh              # fp32 lsq_w4a4 funccode_w4a4
bash scripts/hw/22_run_hls_gram_kagn.sh    # gram_* and kagnconv_*

# 4. Reports -> CSV -> LaTeX + Markdown tables (CPU, seconds, rerun anytime).
bash scripts/hw/13_reports.sh
```

Individual pieces:

```bash
# one model
python -m funcodekan.experiments.hw_prepare --model lsq_w4a4
python -m funcodekan.experiments.hw_prepare_gram --model gram_funccode_w4a4
python -m funcodekan.experiments.hw_prepare_kagn --model kagnconv_lsq_w4a4 --from-scratch

# one export
python -m funcodekan.hw.export      --design d3
python -m funcodekan.hw.gram_export --design g3
python -m funcodekan.hw.kagn_export --design k3

# one design through HLS
vivado_hls -f hw/tcl/run_hls.tcl -tclargs funccode_w4a4 csim
vivado_hls -f hw/tcl/run_hls.tcl -tclargs funccode_w4a4 impl 50
vivado      -mode batch -source hw/tcl/impl_eval.tcl -tclargs funccode_w4a4
```

`run_hls.tcl` modes: `csim` (all 1000 golden vectors, the L2 gate), `synth`
(csynth only, quick iteration), `impl` (csynth + cosim on N vectors, the L3
gate). An optional fourth argument overrides the part.

Three training protocols differ by necessity, and the reason is recorded rather
than hidden:

- **D3/G3** cluster from the family's dense checkpoint, then fine-tune codebooks
  with indices frozen.
- **K2** trains W4A4 **from scratch**: post-hoc W4 on the FP32-trained conv
  weights collapses to 23.4%, because roughly a quarter of the tiny conv weights
  round to zero and the small GAP-head network has no redundancy to absorb it.
- **K3** clusters from the **K2** (W4-native) weights, not from FP32 dense —
  clustering FP32 dense gives 66.4%.

`docs/HW_FAIRNESS.md` states which of these are held identical and which are
the variables under study.

---

## 6. The verification ladder

Every design climbs four rungs, and each result is appended to
`hw/verification_log.json`.

| rung | what is compared | scope | gate |
|---|---|---|---|
| **L1** | PyTorch model ↔ numpy fixed-point emulator | all 10,000 MNIST test images | argmax exact for quantized designs; ≤1e-4/1e-3 for FP32 |
| **L2** | HLS C simulation ↔ emulator INT32 logits | 1,000 golden vectors | bit-exact for quantized designs |
| **L3** | RTL co-simulation ↔ C simulation | first 50 golden vectors | bit-exact for quantized designs |
| **L4** | deployed accuracy vs the family's dense FP32 | — | reported per design |

All nine designs completed every rung: the six quantized designs are
**bit-exact** at L2 and L3 and match argmax 10000/10000 at L1; the three FP32
baselines pass within float tolerance with exact argmax.

Golden vectors are the first 100 test images of each class — 1,000 stratified,
deterministic, SHA256-pinned in each `manifest.json`. Co-simulation uses the
first 50 of exactly that set, identical for every design.

L4 is where the honest costs appear, and they are reported rather than gated
away: D3 is 1.06 pp below its dense reference, G3 0.59 pp, K3 5.65 pp. The
analysis of each is in `docs/HW_FAIRNESS.md`.

**The accuracy convention matters.** Unless labelled otherwise, a quantized
design's accuracy is its **deployed-arithmetic** accuracy on the full 10k test
set — the L1-verified emulator path recorded in `manifest.json` — not the
training-time floating-point proxy. Hardware accuracies are never mixed with
software accuracies in a single comparison.

---

## 7. The frozen contract

Two documents govern the comparison, and the code is written to them:

- **`docs/HW_DESIGN_CONTRACT.md`** — the numeric specification: quantization
  points, INT4 ranges, LSQ, basis and SiLU LUT formats, the integer datapath,
  requantization, storage accounting, and the GRAM/KAGN datapath extensions.
- **`docs/HW_FAIRNESS.md`** — the comparison contract: what is identical across
  designs, what is allowed to differ, and the threats to validity.

Every constant exists in exactly two places, `funcodekan/hw/hw_spec.py` and
`hw/hls/common/hw_config.h`, and the two must never drift. If you change one,
change the other, then retrain, re-export and re-verify. The emulator is the
normative specification of the datapath; the HLS follows it, not the reverse.

Storage numbers are computed by `funcodekan.analysis.storage`
(`compressed_storage_breakdown`, `required_index_bits = ceil(log2 K)`,
`bits_to_kib`) and are never re-derived by hand.

Operator-level how-to per family: `docs/HW_MANUAL.md` (spline) and
`docs/HW_MANUAL_GRAM_KAGN.md` (GRAM and KAGN-Conv).

---

## 8. Reading the results

`docs/HW_RESULTS.md` is the complete compendium — every model and hardware
number for all nine designs, with provenance for each. The shape of the result:

| | spline | GRAM | KAGN-conv |
|---|---|---|---|
| weight storage, LSQ → FuncCode | 3.95× | 2.20× | 1.65× |
| BRAM18, LSQ → FuncCode (syn / route) | 3.4× / 3.87× | 1.77× / 1.96× | 1.14× / 1.19× |
| latency, LSQ → FuncCode | **1.00×** | **1.00×** | **1.00×** |
| accuracy cost, LSQ → FuncCode | 1.00 pp | 1.14 pp | 2.96 pp |
| composite BRAM, FP32 → FuncCode (syn) | **26.3×** | 10.9× | 2.7× |

The latency row is the one to read first: identical to the cycle in all three
families, which is what makes the BRAM saving attributable to the weight format
and to nothing else.

The index-stream share explains the trend across families — 98.9% (spline),
99.1% (GRAM), 82.3% (KAGN-conv). Once codebooks are quantized to four bits they
occupy a few hundred bytes and essentially all remaining weight storage is
pointers. The conv family is the informative exception: a conv KAN already
shares its edge functions across spatial positions, so it has far fewer edges
per parameter and the codebooks no longer amortize away. The gain shrinking
from 3.95× to 1.65× is the expected consequence, not an anomaly.

---

## 9. What ships and what is regenerated

**Shipped:** HLS sources, testbench, tcl drivers, exported `params.h` and
`golden.h` plus `.npy` and manifests for all nine designs, archived synthesis
and place-and-route reports, the verification log, the per-run `summary.csv`
files that `13_reports.sh` reads for its accuracy and storage columns, and the
model-preparation configs. The designs can therefore be built, simulated and
verified, and the tables regenerated, without retraining anything.

**Not shipped:** `hw/proj/` — the Vivado HLS working directories (generated
RTL, solution databases, simulation artifacts). These are build output,
recreated by `run_hls.tcl`. Training checkpoints and logs under `runs_hw/` are
also omitted; `configs/hw/` holds the settings and
`scripts/hw/10_train_models.sh` / `20_train_gram_kagn.sh` regenerate them.

Note that `scripts/hw/11_export_golden.sh` and `21_export_gram_kagn.sh`
**overwrite** `hw/golden/`. Only run them if you have retrained, or if you
intend to replace the shipped exports.

---

## 10. Troubleshooting

| symptom | cause / fix |
|---|---|
| `vivado_hls: command not found` | Add the 2019.1 `bin` directory to `PATH`. The scripts call `vivado_hls` unqualified. |
| `ERROR: [IMPL 213-28] Failed to generate IP` | The 2019.1 IP-packager date bug. Expected — this is exactly why post-implementation goes through `impl_eval.tcl` instead of `export_design`. |
| D1 fails to place on xczu7ev | Expected and documented: 397 BRAM36 of FP32 ROMs against 312 available. D1's post-implementation cell is left empty rather than substituting a synthesis estimate. |
| csim compiler ICE on large arrays | Already handled: csim uses POD storage for the large generated ROMs while synthesis uses true narrow `ap_int` types. Values are identical — see the comment in `hw/hls/common/hw_config.h`. |
| C simulation mismatches the emulator | The emulator is normative. Check that `hw_spec.py` and `hw_config.h` still agree; drift between them is the usual cause. |
| Golden vectors do not match `manifest.json` | The manifest pins SHA256 per artefact. A mismatch means `hw/golden/` was regenerated from a different checkpoint. |
| A GRAM FP32 cosim hangs | A `tanhf` deadlock in this toolchain version; fixed in the current `gram_basis_fp32.h`. |

For the pipeline as a whole see the top-level `README.md` and `REPRODUCE.md`;
for the software-side compression results, `docs/CONV_CIFAR.md` and
`docs/MODEL_DETAILS.md`.
