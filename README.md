# FuncCode-KAN

**Codebook-based functional compression and hardware-aware quantization for
Kolmogorov–Arnold Networks (KANs).**

Anonymous code release accompanying the paper submission. This repository
contains everything needed to reproduce every experiment in the paper and its
appendix.

FuncCode-KAN trains a dense KAN normally, represents its learned edge functions
by coefficient vectors or sampled function signatures, builds shared or
branch-aware codebooks over them, fine-tunes the codebooks, quantizes them to
low precision, and reports bit-exact packed storage plus analytical FPGA BRAM
cost.

**The core idea in one line:** a KAN edge carries `C` coefficients. Scalar
quantization pays *per coefficient* (`C·b` bits/edge); FuncCode pays *per edge*
(one `ceil(log2 K)` index). Below `C·2` bits/edge, scalar schemes have no bits
left to spend and FuncCode still does.

---

## Table of contents

1. [Installation](#1-installation)
2. [Verify the install](#2-verify-the-install-2-minutes)
3. [The three experimental tracks](#3-the-three-experimental-tracks)
4. [Repository layout](#4-repository-layout)
5. [Running individual experiments](#5-running-individual-experiments)
6. [Reproducing the complete paper](#6-reproducing-the-complete-paper)
7. [Understanding the outputs](#7-understanding-the-outputs)
8. [Using the package as a library](#8-using-the-package-as-a-library)
9. [Extending to new KAN variants](#9-extending-to-new-kan-variants)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Installation

```bash
conda create -n funcodekan python=3.10 -y
conda activate funcodekan
pip install -e ".[dev]"
```

Python ≥ 3.10 is required; 3.10 and 3.12 are both tested. Optional extras:

| extra | installs | needed for |
|---|---|---|
| `dev` | pytest, safetensors, matplotlib | tests, export tools, Pareto plots |
| `datasets` | ucimlrepo, requests, openpyxl | the UCI tabular datasets (Section 5.4) |

```bash
pip install -e ".[dev,datasets]"     # everything
```

A CUDA GPU is used automatically when available; every code path also runs on
CPU (slowly). **Run all commands from the repository root.** Output folders
(`runs*/`) are created on demand and datasets download to `./data/`
automatically.

---

## 2. Verify the install (2 minutes)

```bash
pytest tests/ -q
```

Expected: all tests pass. These use synthetic data only — no downloads, no GPU.
They cover the HWQ bit-packing library, the export format, every compression
path end-to-end, the conv-KAGN whitening identity, and the FPGA track's
fixed-point integer datapath.

| test file | what it checks |
|---|---|
| `test_bitpacking.py` | bit-exact packing/unpacking at 2–8 bits |
| `test_export_toy.py` | compressed-state export format + metadata round-trip |
| `test_pipeline_smoke.py` | all five SplineKAN cluster methods, cross-variant compression for spline/fast/gram, SRB / index-efficient / soft-to-hard, W4 quantization, storage accounting |
| `test_conv_kagn.py` | conv-KAGN layers, the Gram-whitening identity vs. explicit signatures, storage accountant completeness |
| `test_datasets_integration.py` | dataset bundles, generic driver on offline data, graceful zoo degradation |
| `test_hw_emulator.py` | the fixed-point integer datapath — the normative spec the HLS is written against |
| `test_hw_export.py` | `params.h` / `golden.h` emission and round-trip |
| `test_hw_lsq.py` | LSQ step-size learning and gradients |
| `test_hw_gram.py`, `test_hw_kagn.py` | GRAM and KAGN-Conv integer datapaths |

The five `test_hw_*` files verify the hardware arithmetic without needing
Vivado, a GPU, or any download.

Then run the 1-epoch end-to-end smoke tests (downloads MNIST / CIFAR on first
use):

```bash
make smoke-mnist        # ~3 min
bash scripts/paper/09_conv_cifar.sh smoke     # ~10 min, conv track
```

---

## 3. The three experimental tracks

The paper reports three tracks that **share one compression implementation**.
Tracks A and B expose each layer as an edge matrix `[E, C]`, where one row is
one Kolmogorov–Arnold edge function and the last slot is always the base
branch; Track C takes the result to silicon:

```
phi_e(x) = sum_d w[e,d] · B_d(x)   +   w[e,-1] · silu(x)
                 \___basis___/          \__base branch__/
```

| | **Track A — fully connected KAN** | **Track B — convolutional KAGN** |
|---|---|---|
| driver | `experiments/all_kan_mnist.py`, `all_kan_cifar.py`, `all_kan_datasets.py` | `experiments/conv_cifar.py` |
| models | SplineKAN, FastKAN, GRAM/KAGN, MLP baseline | `SimpleConvKAGN` (4-layer), `EightSimpleConvKAGN` (8-layer) |
| datasets | MNIST, Fashion-MNIST, CIFAR-10/100 (flattened), Tiny ImageNet, 5 tabular | CIFAR-10, CIFAR-100 |
| edges | `(out, in)` pairs, ~10³–10⁶ | `(out_ch, in_ch, kh, kw)` tuples, ~6.1×10⁶ |
| clustering | explicit `[E, S]` signature matrix | whitened 5-D coefficients (identity, 26× less memory) |
| reported codebook precision | **W4** (`clustered_hwq_w4`) | **w8** (w4 is unstable at conv scale) |
| detail doc | `docs/README_ALL_KAN_MNIST.md` | `docs/CONV_CIFAR.md`, `docs/MODEL_DETAILS.md` |

Track B exists because the flattened-image CIFAR protocol in Track A reaches
only ~47% (CIFAR-10) / ~14% (CIFAR-100) dense, which makes compression measured
against it hard to interpret. Track B's dense references are 91.4% and 62.8%.

### Track C — FPGA accelerators

Nine Vivado HLS designs, three model families × three weight formats
(FP32 dense / INT4 dense via LSQ QAT / INT4 FuncCode codebooks), synthesized
and placed-and-routed on ZynqMP parts at a 6.67 ns target.

| | detail |
|---|---|
| code | `funcodekan/hw/` (export + fixed-point emulator + QAT), `hw/` (HLS C++, tcl, golden vectors, reports) |
| scripts | `scripts/hw/10..13` (spline family), `scripts/hw/20..22` (GRAM and KAGN-Conv) |
| designs | D1/D2/D3 spline, G1/G2/G3 GRAM, K1/K2/K3 KAGN-Conv |
| headline | within a family the INT4 and FuncCode designs have **identical cycle counts**; only BRAM moves — 3.4× on spline post-synthesis, 3.87× post-route |
| verification | four-rung ladder: torch ↔ emulator (10k images) → C simulation (1000 vectors) → RTL co-simulation (50) → deployed accuracy |
| detail doc | **`hw/README.md`** — build, verify, and read the results |

Track C ships buildable: the exported weights, golden vectors and archived
synthesis reports are all included, so the tables can be regenerated and the
designs simulated without retraining. Most of it is checkable with Python
alone — see `hw/README.md` §3.

### Compression methods

| method | flag / API |
|---|---|
| Coefficient-space codebooks | `coefficient` |
| Function-space codebooks | `function` |
| Branch-aware base/spline codebooks | `branch` |
| Index-efficient branch sharing | `branch_index` / `index` |
| Sparse Residual Branch codebooks (SRB) | `branch_residual` / `srb` |
| Soft-to-hard branch-index learning | `soft` |
| Hardware-aware codebook quantization | `--bits-list 8 6 4 3 2` (Track A) / `--codebook-bits` (Track B) |

### Baselines (Track B)

| baseline | flag | cost per edge |
|---|---|---|
| uniform PTQ (inference-only by design) | `uniform` | `C·b` |
| LSQ QAT | `lsq` | `C·b` |
| product quantization | `pq` | `m·ceil(log2 K)` |
| magnitude prune + W4 | `prune` | `C·(1−s)·4` + 1-bit mask |
| iso-storage dense (narrower net, same budget) | `iso` | `C·32` on a narrowed net |

---

## 4. Repository layout

```text
.
├── funcodekan/                  Installable Python package
│   ├── data/                    mnist.py, cifar.py (flattened), cifar_conv.py (NCHW),
│   │                            registry.py (~20 datasets), bundles.py (adapter)
│   ├── models/                  spline.py       Dense + all compressed SplineKAN models
│   │                            variants.py     SplineKAN / FastKAN / GRAM / MLP
│   │                            ablations.py    SRB, index-efficient, soft-to-hard
│   │                            soft_codebook.py  Differentiable soft-index SplineKAN
│   │                            conv_kagn.py    Conv KAGN layers, backbones, presets
│   │                            zoo.py          Optional external model zoo (needs `kans`)
│   ├── compression/             clustering.py       Coefficient/function/branch KMeans
│   │                            function_space.py   Edge-function sampling
│   │                            cross_variant.py    Variant-agnostic compression + storage
│   │                            conv_compression.py Whitened conv clustering + storage
│   │                            conv_baselines.py   uniform / LSQ / PQ / prune / iso
│   ├── quantization/            hwq_pipeline.py     Codebook quantization + HWQ export
│   ├── analysis/                storage.py          Bit-exact storage accounting
│   ├── hwq/                     Standalone HWQ library (bit-packing, export, CLI)
│   ├── hw/                      FPGA track: hw_spec.py / gram_spec.py (frozen constants),
│   │                            fixed_point.py (normative integer emulator), qat.py,
│   │                            lsq.py, act_quant.py, export.py / gram_export.py /
│   │                            kagn_export.py, kagn_conv.py, verify.py
│   ├── utils/                   training.py, conv_training.py, inspect_checkpoint.py
│   └── experiments/             Runnable drivers (see §5), incl. hw_prepare*.py
├── hw/                          FPGA artefacts — see hw/README.md
│   ├── hls/                     HLS C++ sources: common/ + one top.cpp per design
│   ├── tb/  tcl/                Testbench and Vivado HLS / Vivado drivers
│   ├── golden/<design>/         params.h, golden.h, .npy vectors, manifest.json
│   ├── results/<design>/        Archived synthesis, co-simulation and P&R reports
│   └── verification_log.json    Append-only L1–L4 record
├── scripts/
│   ├── mnist/                   MNIST experiment & ablation launchers
│   ├── cifar/                   Flattened CIFAR launcher
│   ├── datasets/                Extended datasets + optional model zoo
│   ├── hw/                      FPGA pipeline: 10..13 (spline), 20..22 (GRAM/KAGN)
│   └── paper/                   Numbered phases 00–09 + run_all.sh + plans/
├── tools/                       Summarizers, LaTeX table generators, FPGA/BRAM estimator,
│                                GPU job queue, HLS report parser, Pareto analysis
├── tests/                       Unit + synthetic end-to-end tests (incl. HW emulator)
├── docs/                        Per-method notes, model catalogue, protocol details,
│                                HW_DESIGN_CONTRACT / HW_FAIRNESS / HW_MANUAL* / HW_RESULTS
├── configs/conv_cifar/          The exact config.json of every reported conv run
├── configs/hw/                  The exact config.json of every HW model-preparation run
├── REPRODUCE.md                 Complete end-to-end reproduction protocol
└── pyproject.toml  Makefile  requirements.txt  LICENSE
```

---

## 5. Running individual experiments

Every experiment is a Python module with a full CLI. **Every hyperparameter is a
flag** — `python -m <module> --help` lists them all. The shell launchers below
are thin wrappers that fill in the paper's settings.

### 5.1 Track A — cross-variant MNIST (main fully connected table)

Dense + function-space + branch-aware compression for all four model families,
swept W8→W2:

```bash
bash scripts/mnist/run_all_kan.sh main
# modes: smoke | main | spline | fast | gram | mlp
```

Equivalent direct call (this is what `main` runs):

```bash
python -m funcodekan.experiments.all_kan_mnist \
  --run-name all_kan_mnist_main --out-dir runs \
  --variants spline fast gram mlp --methods function branch \
  --clusters-list 16 32 --bits-list 8 6 4 3 2 \
  --width 64 --epochs 10 --finetune-epochs 20 \
  --lr 1e-3 --finetune-lr 5e-4 --seed 42 \
  --batch-size 1024 --test-batch-size 2048
```

(`--out-dir`, `--lr`, `--finetune-lr`, `--seed` and the batch sizes above are
the argparse defaults; they are spelled out here so the settings are explicit.)

Summarize and build the LaTeX table:

```bash
python tools/summarize_all_kan_mnist.py \
  --root runs/all_kan_mnist_main \
  --out runs/all_kan_mnist_main/combined_summary_w4.csv
python tools/make_all_kan_latex_table.py \
  --csv runs/all_kan_mnist_main/combined_summary_w4.csv \
  --out-tex runs/all_kan_mnist_main/all_kan_main.tex
```

### 5.2 Track A — multi-seed MNIST

```bash
bash scripts/mnist/run_multiseed_all_kan.sh main_all_bits
# modes: smoke | main | main_all_bits | spline_main       (seeds 42, 123, 2026)

python tools/summarize_multiseed_all_kan.py \
  --root runs_multiseed --out runs_multiseed/all_kan_mnist_multiseed_summary.csv
python tools/select_best_multiseed_methods.py \
  --csv runs_multiseed/all_kan_mnist_multiseed_summary.csv \
  --out runs_multiseed/best_multiseed_methods.csv
python tools/make_multiseed_latex_table.py \
  --csv runs_multiseed/all_kan_mnist_multiseed_summary.csv \
  --out-tex runs_multiseed/multiseed_all_methods.tex
python tools/make_multiseed_latex_table.py \
  --csv runs_multiseed/best_multiseed_methods.csv \
  --out-tex runs_multiseed/multiseed_best_methods.tex
```

### 5.3 Track A — SplineKAN ablation suite

Each launcher writes one run folder per configuration under `runs/`, each with a
`summary.csv` giving accuracy and bit-exact storage at every bit width.

```bash
bash scripts/mnist/run_function_space.sh all     # coefficient vs function space, K ∈ {16,32,64,128}
bash scripts/mnist/run_branch_aware.sh all       # branch-aware (Ks, Kb), K ∈ {16,32,64}
bash scripts/mnist/run_index_efficient.sh all    # index-efficient vs references, K ∈ {16,32,64}
bash scripts/mnist/run_srb_codebooks.sh all      # SRB, ρ ∈ {0.10,0.25,0.50}, K ∈ {16,32}
bash scripts/mnist/run_soft_codebook.sh sweep    # soft-to-hard branch-index; also: k16 | k32
bash scripts/mnist/run_spline_baseline.sh main   # single reference run;      also: smoke
bash scripts/mnist/run_cluster_sweep.sh          # K ∈ {4,8,16,32}, no mode argument
```

The first four also accept per-K modes (`smoke`, `compare_k16`, `compare_k32`,
`compare_k64`, `compare_k128` where applicable) if you only need one point.

### 5.4 Track A — cross-variant ablations (SRB / index / soft on spline, fast, gram)

```bash
bash scripts/mnist/run_variant_ablations.sh main
# modes: smoke | main | multiseed | fast_debug

python tools/summarize_variant_ablations.py \
  --root runs_variant_ablations --out runs_variant_ablations/variant_ablation_summary.csv
python tools/make_variant_ablation_latex_table.py \
  --csv runs_variant_ablations/variant_ablation_summary.csv \
  --out-tex runs_variant_ablations/variant_ablation_table.tex
```

### 5.5 Track A — extended datasets (Fashion-MNIST, tabular, Tiny ImageNet)

`all_kan_datasets` is a data-loading-only delta of the verified `all_kan_mnist`
driver, so every summarizer and table generator works on its outputs unchanged.

```bash
bash scripts/datasets/run_all_kan_datasets.sh list           # show dataset keys
bash scripts/datasets/run_all_kan_datasets.sh smoke
bash scripts/datasets/run_all_kan_datasets.sh tabular_main   # moons, circles, wine, dry_bean, mushroom
bash scripts/datasets/run_all_kan_datasets.sh fashion_main
bash scripts/datasets/run_all_kan_datasets.sh tiny_imagenet  # long

python -m funcodekan.experiments.all_kan_datasets --dataset fashion_mnist --seed 42
```

`dry_bean` and `mushroom` fetch from UCI on first use and need `pip install
-e ".[datasets]"` plus network access.

### 5.6 Track A — flattened CIFAR (legacy protocol, superseded by Track B)

Retained because the paper reports it as the motivation for Track B.

```bash
bash scripts/cifar/run_cifar_experiments.sh cifar10_main        # seed 42
bash scripts/cifar/run_cifar_experiments.sh cifar10_multiseed   # seeds 42, 123, 2026
bash scripts/cifar/run_cifar_experiments.sh cifar100_main       # seed 42
# also: cifar10_smoke | cifar100_smoke | cifar10_fast_debug
```

> **Never summarize a runs folder that mixes smoke and main runs** — the
> summarizers `rglob` for `summary.csv`. Copy the paper seeds into a clean
> folder first; see `REPRODUCE.md` §4.

### 5.7 Track B — convolutional KAGN CIFAR (the reported CIFAR rows)

The driver runs three stages in order: train the dense backbone (or reuse a
cached checkpoint), compress + fine-tune + quantize each `(method, K)` arm, then
apply each baseline with an equal fine-tuning budget.

**Wiring check (~10 min, tiny preset, 2 epochs):**

```bash
bash scripts/paper/09_conv_cifar.sh smoke
```

**Step 1 — train the dense backbone** (this is the expensive part; every arm
reuses it):

```bash
python -m funcodekan.experiments.conv_cifar \
  --dataset cifar10 --preset kagn_simple_cifar10_8_layer_v2 --seed 42 \
  --epochs 200 --batch-size 128 --lr 1e-3 --mixup 0.0 \
  --data-root ./data --out-dir runs_conv_cifar --run-name cifar10_s42_dense \
  --stages dense

python -m funcodekan.experiments.conv_cifar \
  --dataset cifar100 --preset kagn_simple_cifar100_8_layer_v2 --seed 42 \
  --epochs 200 --batch-size 128 --lr 1e-3 --mixup 0.2 \
  --data-root ./data --out-dir runs_conv_cifar --run-name cifar100_s42_dense \
  --stages dense
```

**Step 2 — one FuncCode arm** (repeat per method × K; each is independent and
can run in parallel):

```bash
python -m funcodekan.experiments.conv_cifar \
  --dataset cifar10 --preset kagn_simple_cifar10_8_layer_v2 --seed 42 \
  --epochs 200 --finetune-epochs 30 --batch-size 128 --lr 1e-3 --mixup 0.0 \
  --data-root ./data --out-dir runs_conv_cifar --run-name fc_function_K32_cifar10_s42 \
  --stages funccode --methods function --clusters-list 32 --codebook-bits 8 4 \
  --dense-ckpt runs_conv_cifar/cifar10_s42_dense/dense.pt --require-ckpt
```

**Step 3 — one baseline arm:**

```bash
python -m funcodekan.experiments.conv_cifar \
  --dataset cifar10 --preset kagn_simple_cifar10_8_layer_v2 --seed 42 \
  --epochs 200 --finetune-epochs 30 --batch-size 128 --lr 1e-3 --mixup 0.0 \
  --data-root ./data --out-dir runs_conv_cifar --run-name bl_pq_cifar10_s42 \
  --stages baselines --baselines pq --pq-subvectors 2 \
  --dense-ckpt runs_conv_cifar/cifar10_s42_dense/dense.pt --require-ckpt
```

**Step 4 — tables (CPU, rerun any time):**

```bash
bash scripts/paper/09_conv_cifar.sh tables
# -> runs_conv_cifar/tables/{conv_cifar_all.csv, conv_cifar_agg.csv,
#                            table_main_conv.tex, table_iso_storage.tex, pareto_conv.csv}
```

The exact configuration of every reported conv run is committed under
`configs/conv_cifar/<run_name>.json` — each file is the `argparse` namespace the
driver was invoked with, so any single run can be reproduced verbatim.

**Automated, multi-GPU:** `scripts/paper/09_conv_cifar.sh tier1|tier2|tier3`
generates a job plan with `tools/make_conv_plan.py` and dispatches it with
`tools/gpu_queue.py`, which respects dependencies (dense before arms), packs
several jobs per GPU, and resumes after interruption:

```bash
GPUS="0 1" SLOTS_PER_GPU=2 bash scripts/paper/09_conv_cifar.sh tier1
bash scripts/paper/09_conv_cifar.sh status tier1     # poll a running queue
```

### 5.8 Track C — FPGA accelerators

Full instructions, including prerequisites and the verification ladder, are in
**[`hw/README.md`](hw/README.md)**. The short version:

```bash
# (a) Verify the hardware arithmetic — no Vivado, no GPU, no downloads
pytest tests/test_hw_emulator.py tests/test_hw_export.py tests/test_hw_lsq.py \
       tests/test_hw_gram.py tests/test_hw_kagn.py -q

# (b) Regenerate the appendix hardware tables from the shipped raw reports
bash scripts/hw/13_reports.sh
# -> runs_hw/tables/{hw_report_summary.csv, hw_comparison.tex}, docs/HW_COMPARISON.md

# (c) Build and simulate a design (needs Vivado HLS 2019.1 on PATH)
vivado_hls -f hw/tcl/run_hls.tcl -tclargs funccode_w4a4 csim     # L2: 1000 vectors
vivado_hls -f hw/tcl/run_hls.tcl -tclargs funccode_w4a4 impl 50  # L3: csynth + cosim
vivado      -mode batch -source hw/tcl/impl_eval.tcl -tclargs funccode_w4a4

# (d) Rebuild everything from scratch (GPU for the training stages)
bash scripts/hw/10_train_models.sh      # D1/D2/D3 spline family
bash scripts/hw/11_export_golden.sh     # export weights + golden vectors, L1/L4 gates
bash scripts/hw/12_run_hls.sh           # HLS for all three, archive reports
bash scripts/hw/20_train_gram_kagn.sh   # G-series and K-series
bash scripts/hw/21_export_gram_kagn.sh
bash scripts/hw/22_run_hls_gram_kagn.sh
```

Steps (a) and (b) need nothing but Python and reproduce the paper's hardware
tables from raw tool output. Step (d) **overwrites `hw/golden/`** — only run it
if you intend to replace the shipped exports.

Individual model preparation and export:

```bash
python -m funcodekan.experiments.hw_prepare --model lsq_w4a4
python -m funcodekan.hw.export --design d3          # d1|d2|d3
python -m funcodekan.hw.gram_export --design g3     # g1|g2|g3
python -m funcodekan.hw.kagn_export --design k3     # k1|k2|k3
```

### 5.9 Hardware / storage / BRAM analysis (analytical)

Analytical packed-storage, traffic, and BRAM18/BRAM36 estimates. These are
closed-form from model shape, `K`, and bit width — they contain no randomness
and must reproduce **exactly**.

```bash
python tools/summarize_storage_breakdown.py runs/<run_name>/summary.csv
python tools/estimate_fpga_bram.py --summary runs/<run_name>/summary.csv \
  --out runs/<run_name>/hw_estimate.csv
python tools/print_hw_interpretation.py --hw-csv runs/<run_name>/hw_estimate.csv

python tools/compare_hw_estimates.py \
  --summaries runs/func_k16_ref/summary.csv \
              runs/branch_k16_s16_b8_ref/summary.csv \
              runs/branch_index_k16_ref/summary.csv \
              runs/srb_k16_r10_b8/summary.csv \
              runs/srb_k16_r25_b8/summary.csv \
              runs/srb_k16_r50_b8/summary.csv \
  --out runs/hw_compare_k16.csv

python tools/make_paper_hw_table.py --hw-csv runs/hw_compare_k16.csv \
  --out-tex runs/hw_compare_k16.tex \
  --caption "Analytical FPGA-memory estimates for K=16 compressed KAN variants." \
  --label "tab:hw_k16"
```

The referenced run names are produced by `run_index_efficient.sh all` and
`run_srb_codebooks.sh all` (§5.3). Details: `docs/README_HW_BRAM_ESTIMATOR.md`,
`docs/README_STORAGE_BREAKDOWN.md`.

### 5.10 Analysis tools

```bash
# Edge-redundancy spectra, effective rank, inertia-vs-K
python tools/analyze_edge_redundancy.py --dataset mnist --variant spline \
  --epochs 5 --width 64 --ks 2 4 8 16 32 64 128 --out-dir runs_paper/redundancy

# Accuracy-vs-storage Pareto frontiers (CSV + optional plot)
python tools/make_pareto_data.py --help

# Inspect any saved checkpoint
python -m funcodekan.utils.inspect_checkpoint <path.pt>
```

---

## 6. Reproducing the complete paper

**See [`REPRODUCE.md`](REPRODUCE.md)** for the full ordered protocol: hardware
assumptions, wall-clock budgets, phase-by-phase commands, the mapping from each
run to each table in the paper, and the tolerances within which each number
should reproduce.

The short version:

```bash
# Track A — all fully connected results, phases 00-08
bash scripts/paper/run_all.sh tier1     # core results
bash scripts/paper/run_all.sh tier2     # + Tiny ImageNet, sensitivity, redundancy
bash scripts/paper/run_all.sh all       # everything

# Track B — the reported CIFAR rows
bash scripts/paper/09_conv_cifar.sh tier1     # main table
bash scripts/paper/09_conv_cifar.sh tier2     # seeds 123 + 2026
bash scripts/paper/09_conv_cifar.sh tier3     # ablations
bash scripts/paper/09_conv_cifar.sh tables
```

`run_all.sh` continues past a failed phase, logs everything to `logs/paper/`,
and prints a PASS/FAIL summary at the end. Run it under `tmux`/`screen`.

---

## 7. Understanding the outputs

The two tracks write slightly different CSV schemas, because Track A sweeps
*stages* of one pipeline while Track B compares *methods* at matched storage.

### Track A run folder

| file | contents |
|---|---|
| `<variant>/summary.csv` | one row per (method, K, stage) for that model family |
| `combined_summary.csv` | all variants concatenated |
| `<variant>/dense.pt` | the trained dense checkpoint |

Columns: `variant`, `method`, `clusters`, `base_clusters`, `stage`,
`codebook_bits`, `test_acc`, `storage_kib`, `compression_vs_dense`, plus the
full storage breakdown (`codebook_bits_total`, `index_bits_total`,
`scale_bits_total`, …).

`stage` is the one to filter on — **always compare like with like**:

| `stage` value | meaning |
|---|---|
| `dense_fp32` | uncompressed reference for that variant |
| `clustered_before_finetune_fp32_codebook` | after clustering, before fine-tuning |
| `clustered_finetuned_fp32_codebook` | after codebook fine-tuning, FP32 codebooks |
| `clustered_hwq_w<b>` | codebooks quantized to `<b>` bits — **`w4` is the reported point** |
| `dense_uniform_ptq_w<b>` | scalar PTQ reference at `<b>` bits |

### Track B run folder

| file | contents |
|---|---|
| `config.json` | the complete resolved CLI namespace — the run's exact settings |
| `summary.csv` | one row per (method, operating point) |
| `dense.pt` | the trained dense backbone (`--stages dense` runs only) |

Columns: `dataset`, `preset`, `seed`, `method`, `config`, `family`,
`bits_per_edge`, `test_acc`, `storage_kib`, `compression`, `n_edges`,
`clusters`, `base_clusters`, `codebook_bits`, `ft_val`, `cluster_seconds`,
`fn_rel_err`, and the storage breakdown.

| column | meaning |
|---|---|
| `method` | `dense`, `funccode_function`, `funccode_branch`, `uniform_ptq`, `lsq_qat`, `product_quant`, `prune_w4`, `iso_dense` |
| `config` | the operating point within that method, e.g. `K16_w4`, `m2_K16`, `s0.9` |
| `bits_per_edge` | **the cross-method comparison axis** — storage ÷ edge count |
| `compression` | vs. the same backbone's dense FP32 model |
| `fn_rel_err` | function-space reconstruction error, measured *before* codebook quantization |

The reported operating point is `clustered_hwq_w4` for Track A and the w8
codebook stage for Track B.

---

## 8. Using the package as a library

```python
import torch
from funcodekan.models import DenseSplineKAN
from funcodekan.compression import build_clustered_from_dense

dense = DenseSplineKAN(input_dim=784, hidden_width=64, output_dim=10,
                       grid_size=5, spline_order=3)

compressed = build_clustered_from_dense(
    dense, 784, 64, 10, 5, 3,
    num_clusters=32, seed=42, cluster_method="branch",
    branch_spline_clusters=32, branch_base_clusters=16,
)
y = compressed(torch.randn(8, 784))
```

The reusable hardware-quantization primitives — uniform symmetric/asymmetric
codebook quantizers, bit-packing, compressed-state export with metadata,
fake-quant fine-tuning, and a CLI — live in `funcodekan/hwq/` and have no
dependency on the experiment drivers.

---

## 9. Extending to new KAN variants

The cross-variant pipeline is variant-agnostic as long as a layer exposes a
**basis branch** and a **base branch** as per-edge coefficient tensors:

1. **`funcodekan/models/variants.py`** — implement a `<Name>BasisLayer`
   following `SplineBasisLayer` / `FastKANBasisLayer` / `GramBasisLayer` (set
   `variant_name`, expose basis coefficients of shape `[out, in, basis_dim]`
   and base weights `[out, in]`), and register it in the `DirectKANVariant`
   constructor (`elif variant == "cheby": ...`).
2. **Add the name to the `choices` lists** in
   `funcodekan/experiments/all_kan_mnist.py`, `all_kan_cifar.py`, and
   `variant_ablation.py`.
3. **Nothing else changes** — `compression/cross_variant.py`,
   `models/ablations.py`, quantization, storage accounting, summarizers, and
   table generators all operate on the generic basis/base interface.
4. Validate:

```bash
python -m funcodekan.experiments.all_kan_mnist \
  --variants cheby --methods function branch --clusters-list 8 --bits-list 4 \
  --epochs 1 --finetune-epochs 1 --width 32
```

plus a synthetic forward test in `tests/test_pipeline_smoke.py`.

---

## 10. Troubleshooting

| symptom | cause / fix |
|---|---|
| Summarizer output has unexpected rows | It `rglob`s for `summary.csv`. A smoke run is sitting under the same root — use a clean folder (`REPRODUCE.md` §4). |
| Storage numbers differ from the paper | These are deterministic. A difference means a flag mismatch (`--width`, `--clusters-list`, `--bits-list`, `--codebook-bits`), not nondeterminism. Compare against `configs/`. |
| Accuracy differs by ~0.3 pp at a fixed seed | Expected. `cudnn.benchmark` autotuning and non-deterministic atomics. For exact reproduction use `setup_backend(deterministic=True)` and `--no-amp`. |
| `ImportError: kans` | Only the optional model zoo (`zoo_train.py`, `scripts/datasets/run_zoo.sh`) needs it. No reported result depends on it. |
| UCI dataset download fails | `dry_bean` / `mushroom` need `pip install -e ".[datasets]"` and network access. |
| Conv run OOMs | Lower `--batch-size` (128 default, ~1.2 GiB peak at that setting) or `--num-workers`. |
| Conv w4 results vary wildly | Known and reported: w4 is unstable at conv scale (see `docs/CONV_CIFAR.md` §6). All conv results are reported at w8. |

---

## Reproducibility notes

- Seeds: 42, 123, 2026 for multi-seed protocols; 42 for single-seed. All seeds
  are CLI flags.
- Storage and compression columns are analytical and must match **exactly**.
  Accuracy columns match in trend and approximately in value; method *ordering*
  within a variant should match exactly.
- FPGA/BRAM numbers are analytical estimates, not post-synthesis measurements.
- SplineKAN instability under compression on CIFAR is an expected, reported
  negative result.

## License

MIT License (see `LICENSE`).
