# REPRODUCE.md — reproducing the paper end to end

This is the complete, ordered protocol. `README.md` §5 covers running any single
experiment on its own; this document runs all of them in the right order and
maps each output to the table or figure it produces.

Run everything **from the repository root**.

---

## Contents

- [0. What must match, and how closely](#0-what-must-match-and-how-closely)
- [1. Environment](#1-environment)
- [2. Phase 0 — sanity (15 min)](#2-phase-0--sanity-15-min)
- [3. Track A — fully connected results](#3-track-a--fully-connected-results)
- [4. The clean-folder rule](#4-the-clean-folder-rule)
- [5. Track B — convolutional CIFAR results](#5-track-b--convolutional-cifar-results)
- [5b. Track C — FPGA accelerators](#5b-track-c--fpga-accelerators)
- [6. Tables and figures](#6-tables-and-figures)
- [7. Run-to-table map](#7-run-to-table-map)
- [8. Time and storage budget](#8-time-and-storage-budget)
- [9. Restarting, resuming, and partial reproduction](#9-restarting-resuming-and-partial-reproduction)

---

## 0. What must match, and how closely

Two different standards apply, and keeping them separate matters.

**Storage and compression columns must match EXACTLY.** `storage_kib`,
`bits_per_edge`, `compression_vs_dense`, index/codebook bit counts, and all
FPGA/BRAM estimates are closed-form analytical computations from model shape,
`K`, and bit width. They contain no randomness and no floating-point
accumulation over data. If any storage number differs from the paper by even one
bit, a flag is wrong — most likely `--width`, `--clusters-list`, `--bits-list`,
or `--codebook-bits`. Check against the committed `configs/` before suspecting
the code.

**Accuracy columns match in trend, approximately in value.** Same seeds and same
code give near-identical numbers, but GPU nondeterminism and PyTorch version
differences cause drift:

| result | expected tolerance |
|---|---|
| MNIST (Track A) | ±0.1–0.3 pp |
| CIFAR-10 flattened, multi-seed mean (Track A) | ±1–2 pp |
| CIFAR-100 flattened, single seed (Track A) | ±2–3 pp |
| Conv CIFAR, fixed seed (Track B) | ±0.34 pp run-to-run at the *same* seed |
| Method **ordering** within a variant | must match exactly |

The ±0.34 pp figure for Track B is measured, not estimated (CIFAR-10 dense,
seed 42: 90.85 vs 91.19 across repeats). Its source is `cudnn.benchmark`
autotuning and non-deterministic atomics. For bit-identical reruns use
`setup_backend(deterministic=True)` and pass `--no-amp`; expect roughly a 2×
slowdown.

Record your `torch.__version__` and GPU model alongside any results you compare.

---

## 1. Environment

```bash
conda create -n funcodekan python=3.10 -y
conda activate funcodekan
pip install -e ".[dev,datasets]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Reference environment for the reported numbers:** Python 3.10, PyTorch 2.3.1
with CUDA 12.1, two 32 GB data-centre GPUs, 16 CPU cores. Track A runs
comfortably on a single GPU. Track B assumes at least one 32 GB GPU; peak
memory is ~1.2 GiB per job at `--batch-size 128`, so the constraint is time,
not memory.

Datasets download automatically to `./data/` on first use. `dry_bean` and
`mushroom` are fetched from UCI and need the `datasets` extra plus network
access.

---

## 2. Phase 0 — sanity (15 min)

```bash
pytest tests/ -q                          # synthetic, no downloads, no GPU
bash scripts/paper/00_smoke.sh            # 1-epoch pass through every path
bash scripts/paper/09_conv_cifar.sh smoke # conv track wiring, ~10 min
```

`00_smoke.sh` prints a wall-time per path. Use those numbers to scale the
estimates in §8 to your own hardware before committing to a long run.

Smoke outputs land in `runs*/smoke_*` and `runs_conv_smoke/`. **Do not
summarize them** — see §4.

---

## 3. Track A — fully connected results

### Option A — automated

```bash
tmux new -s paper
bash scripts/paper/run_all.sh tier1     # core results
bash scripts/paper/run_all.sh tier2     # + Tiny ImageNet, sensitivity, redundancy
bash scripts/paper/run_all.sh all       # both
# detach: Ctrl+b d     reattach: tmux attach -t paper
```

`run_all.sh` continues past a failed phase, logs each to `logs/paper/`, and
prints a PASS/FAIL summary at the end.

`tier1` runs phases 00, 01, 02, 03, 04, 05. `tier2` adds 08, 07, 06,
`03 extra_seeds`, and refreshes 05.

### Option B — phase by phase

Run in this order. Phases 01–04 are independent of each other once 00 passes,
so they can be run in any order or in parallel on separate GPUs; phase 05 must
come last because it consumes their outputs.

```bash
# Phase 00 — smoke + timing calibration                              (~15 min)
bash scripts/paper/00_smoke.sh

# Phase 01 — MNIST + Fashion-MNIST, seeds 42/123/2026                (~4-6 h)
bash scripts/paper/01_mnist_family.sh main
bash scripts/paper/01_mnist_family.sh width_sweep    # optional, appendix

# Phase 02 — 5 tabular/synthetic datasets, 5 seeds each              (~2-4 h)
bash scripts/paper/02_tabular.sh
#   override the set: DATASETS="moons wine" bash scripts/paper/02_tabular.sh

# Phase 03 — flattened CIFAR-10 (3 seeds) + CIFAR-100 (seed 42)      (~12-18 h)
bash scripts/paper/03_cifar.sh main
bash scripts/paper/03_cifar.sh extra_seeds           # tier 2

# Phase 04 — SplineKAN ablation suite + cross-family ablations       (~6-10 h)
bash scripts/paper/04_ablations.sh

# Phase 06 — Tiny ImageNet, reduced grid                             (long, tier 2)
bash scripts/paper/06_tiny_imagenet.sh main
bash scripts/paper/06_tiny_imagenet.sh spline_instability

# Phase 07 — sensitivity: K sweep, signature length, domain          (~2-3 h)
bash scripts/paper/07_sensitivity.sh

# Phase 08 — edge-redundancy analysis across datasets and families   (~2-3 h)
bash scripts/paper/08_redundancy.sh

# Phase 05 — CPU post-processing: summaries, HW/BRAM, LaTeX tables   (minutes)
bash scripts/paper/05_hardware_and_tables.sh
```

Phase 05 is safe to rerun at any point; it checks for each input and silently
skips whatever is not there yet. Run it early to see partial tables.

---

## 4. The clean-folder rule

Every summarizer walks its `--root` recursively looking for `summary.csv`. A
1-epoch smoke run sitting under the same root will be pulled into the paper
tables with no warning.

**Rule: never summarize a folder that mixes smoke and main runs.** Copy the
paper seeds into a clean folder first:

```bash
mkdir -p runs_cifar_clean/cifar10_multiseed runs_cifar_clean/cifar100_main
cp -r runs_cifar/cifar10_main_seed42 \
      runs_cifar/cifar10_main_seed123 \
      runs_cifar/cifar10_main_seed2026 runs_cifar_clean/cifar10_multiseed/
cp -r runs_cifar/cifar100_main_seed42 runs_cifar_clean/cifar100_main/

python tools/summarize_cifar_runs.py \
  --root runs_cifar_clean/cifar10_multiseed \
  --out runs_cifar_clean/cifar10_multiseed_summary_clean.csv \
  --stage clustered_hwq_w4
python tools/make_cifar_latex_table.py \
  --csv runs_cifar_clean/cifar10_multiseed_summary_clean.csv \
  --out-tex runs_cifar_clean/cifar10_multiseed_table_clean.tex \
  --dataset cifar10 --stage clustered_hwq_w4 --best-only \
  --caption "Multi-seed CIFAR-10 compressed KAN results under W4 codebook quantization." \
  --label "tab:cifar10_multiseed"
```

`05_hardware_and_tables.sh` already applies this pattern for CIFAR. The conv
track avoids the problem structurally: smoke output goes to `runs_conv_smoke/`,
never to `runs_conv_cifar/`.

---

## 5. Track B — convolutional CIFAR results

These are the CIFAR rows reported in the paper. Full protocol rationale:
`docs/CONV_CIFAR.md`; architecture and hyperparameter reference:
`docs/MODEL_DETAILS.md`.

### 5.1 Automated (recommended)

```bash
GPUS="0 1" SLOTS_PER_GPU=2 bash scripts/paper/09_conv_cifar.sh tier1
GPUS="0 1" SLOTS_PER_GPU=2 bash scripts/paper/09_conv_cifar.sh tier2
GPUS="0 1" SLOTS_PER_GPU=2 bash scripts/paper/09_conv_cifar.sh tier3
bash scripts/paper/09_conv_cifar.sh tables
```

| tier | contents |
|---|---|
| `tier1` | both datasets, seed 42, all methods and baselines — the main table |
| `tier2` | seeds 123 and 2026, for the error bars |
| `tier3` | ablations: skip-first/skip-head, clustering metric, codebook-bit sweep |

Each tier generates a job plan with `tools/make_conv_plan.py` and dispatches it
with `tools/gpu_queue.py`. The queue respects dependencies (the dense backbone
is priority 0 and unblocks every arm for that seed), packs `SLOTS_PER_GPU` jobs
per card, orders longest-first within a priority tier, and records completion so
an interrupted run resumes rather than restarting.

Poll a running queue from another shell:

```bash
bash scripts/paper/09_conv_cifar.sh status tier1
```

Set `GPUS` and `SLOTS_PER_GPU` for your machine. Two slots per card measured
1.35× the throughput of one on the reference hardware; three gave no further
gain. On a single GPU: `GPUS="0" SLOTS_PER_GPU=2`.

### 5.2 Manual, one arm at a time

**Step 1 — dense backbones.** Everything else reuses these checkpoints, so run
them first. One per (dataset, seed).

```bash
python -m funcodekan.experiments.conv_cifar \
  --dataset cifar10 --preset kagn_simple_cifar10_8_layer_v2 --seed 42 \
  --epochs 200 --batch-size 128 --lr 1e-3 --mixup 0.0 \
  --finetune-epochs 30 --num-workers 3 \
  --data-root ./data --out-dir runs_conv_cifar --run-name cifar10_8l_s42_dense \
  --stages dense

python -m funcodekan.experiments.conv_cifar \
  --dataset cifar100 --preset kagn_simple_cifar100_8_layer_v2 --seed 42 \
  --epochs 200 --batch-size 128 --lr 1e-3 --mixup 0.2 \
  --finetune-epochs 30 --num-workers 3 \
  --data-root ./data --out-dir runs_conv_cifar --run-name cifar100_s42_dense \
  --stages dense
```

Repeat with `--seed 123` and `--seed 2026` for the three-seed error bars.

Expected dense accuracy: **91.43 ± 0.21** (CIFAR-10), **62.83 ± 0.96**
(CIFAR-100), three seeds. If dense lands far below this, stop and fix the
backbone before running compression arms — compressing a weak baseline is what
this track exists to avoid.

**Step 2 — FuncCode arms.** One job per `(method, K)`; all are independent given
the dense checkpoint.

```bash
for K in 8 16 32 64 256; do
  for M in function branch; do
    python -m funcodekan.experiments.conv_cifar \
      --dataset cifar10 --preset kagn_simple_cifar10_8_layer_v2 --seed 42 \
      --epochs 200 --finetune-epochs 30 --batch-size 128 --lr 1e-3 --mixup 0.0 \
      --num-workers 3 --data-root ./data --out-dir runs_conv_cifar \
      --run-name "fc_${M}_K${K}_cifar10_8l_s42" \
      --stages funccode --methods "${M}" --clusters-list "${K}" \
      --codebook-bits 8 4 \
      --dense-ckpt runs_conv_cifar/cifar10_8l_s42_dense/dense.pt --require-ckpt
  done
done
```

**Step 3 — baselines.** Same dense checkpoint, same 30-epoch fine-tune budget.
Uniform PTQ is the one exception: it is inference-only by design and receives no
fine-tuning, and is reported as a PTQ-fragility measurement rather than a tuned
scalar baseline.

```bash
# uniform PTQ (cheap, no training)
python -m funcodekan.experiments.conv_cifar ... \
  --stages baselines --baselines uniform --uniform-bits 8 4 3 2

# LSQ QAT
python -m funcodekan.experiments.conv_cifar ... \
  --stages baselines --baselines lsq --lsq-bits 4 2

# product quantization — note K=4 is the strongest setting and is NOT the default
python -m funcodekan.experiments.conv_cifar ... \
  --stages baselines --baselines pq --pq-subvectors 2 --clusters-list 4

# magnitude prune + W4
python -m funcodekan.experiments.conv_cifar ... \
  --stages baselines --baselines prune --prune-sparsity 0.9 0.95

# iso-storage dense (trains a narrower net from scratch at the same bit budget)
python -m funcodekan.experiments.conv_cifar ... \
  --stages baselines --baselines iso --clusters-list 16 32 256
```

**Step 4 — ablations (tier 3).** Single seed, CIFAR-100, `K=32`, w8:

```bash
# skip the first conv and the classifier head (0.4% of edges)
... --stages funccode --methods function --clusters-list 32 --skip-first --skip-head

# clustering metric comparison
... --metric whiten                             # default: whitened function space
... --metric coefficient                        # raw coefficient space
... --metric signature --signature-normalize    # legacy z-scored signature

# codebook precision sweep
... --codebook-bits 8 6 4 3 2
```

**Step 5 — tables.**

```bash
bash scripts/paper/09_conv_cifar.sh tables
# or directly:
python tools/summarize_conv_cifar.py --root runs_conv_cifar --out-dir runs_conv_cifar/tables
```

Produces in `runs_conv_cifar/tables/`:

| file | contents |
|---|---|
| `conv_cifar_all.csv` | every row from every run |
| `conv_cifar_agg.csv` | seed-aggregated mean ± std |
| `table_main_conv.tex` | the CIFAR half of the main table |
| `table_iso_storage.tex` | accuracy at matched bits/edge — the headline comparison |
| `pareto_conv.csv` | accuracy-vs-storage frontier data |

### 5.3 Reproducing one specific reported run

Every reported conv run's exact argument namespace is committed:

```bash
ls configs/conv_cifar/                      # 105 run configurations
cat configs/conv_cifar/bl_iso_K16_cifar100_s42.json
```

Each JSON is the resolved `argparse` namespace, so any single run can be
reconstructed flag-for-flag. Note `dense_ckpt` inside: the run depends on the
matching `*_dense` run having completed first.

---

## 5b. Track C — FPGA accelerators

Nine Vivado HLS designs behind the appendix hardware tables. Full protocol:
**`hw/README.md`**. Unlike Tracks A and B this ships *already built* — the
exported weights, golden vectors and archived synthesis reports are all
included — so the two verification steps below need no GPU and no FPGA tools.

**Step 1 — verify the arithmetic (no Vivado, no GPU, seconds):**

```bash
pytest tests/test_hw_emulator.py tests/test_hw_export.py tests/test_hw_lsq.py \
       tests/test_hw_gram.py tests/test_hw_kagn.py -q
```

The numpy fixed-point emulator in `funcodekan/hw/fixed_point.py` is the
normative specification of the deployed integer datapath; the HLS C++ is
written to match it, and every rung of the verification ladder asserts
agreement against it.

**Step 2 — regenerate the hardware tables from raw tool output (CPU, seconds):**

```bash
bash scripts/hw/13_reports.sh
# -> runs_hw/tables/hw_report_summary.csv
#    runs_hw/tables/hw_comparison.tex
#    docs/HW_COMPARISON.md
```

This parses the shipped `hw/results/<design>/` reports — the actual Vivado
csynth XML, co-simulation reports and post-route utilization/timing files — so
every number in the appendix hardware tables can be traced to tool output
without rebuilding anything.

**Step 3 — rebuild the designs (needs Vivado HLS 2019.1 on `PATH`):**

```bash
bash scripts/hw/12_run_hls.sh            # fp32, lsq_w4a4, funccode_w4a4
bash scripts/hw/22_run_hls_gram_kagn.sh  # gram_* and kagnconv_*
```

Each design runs csim over all 1000 golden vectors (rung L2), then csynth plus
co-simulation over the first 50 (rung L3), then out-of-context Vivado
place-and-route for the post-implementation numbers, archiving every report
back into `hw/results/`.

**Step 4 — retrain and re-export from scratch (GPU; overwrites `hw/golden/`):**

```bash
bash scripts/hw/10_train_models.sh       # D1/D2/D3
bash scripts/hw/11_export_golden.sh      # export + L1/L4 gates, fails hard on L1
bash scripts/hw/20_train_gram_kagn.sh    # G-series and K-series
bash scripts/hw/21_export_gram_kagn.sh
```

Only run step 4 if you intend to replace the shipped exports. Exact settings
for every preparation run are in `configs/hw/`.

**What must match.** Resource and cycle numbers are tool output and reproduce
exactly for a given Vivado version and part; a different Vivado version will
shift LUT/FF counts somewhat but must preserve the orderings and the identical
cycle counts within each family. Deployed accuracies are pinned in
`hw/golden/<design>/manifest.json` and are bit-exact by construction: the six
quantized designs match argmax 10000/10000 at L1 and are bit-exact at L2 and
L3. Post-synthesis and post-implementation numbers are never mixed within a
comparison.

---

## 6. Tables and figures

```bash
bash scripts/paper/05_hardware_and_tables.sh    # Track A -> runs_paper/tables/
bash scripts/paper/09_conv_cifar.sh tables      # Track B -> runs_conv_cifar/tables/
bash scripts/hw/13_reports.sh                   # Track C -> runs_hw/tables/
```

All three are CPU-only, take minutes, skip missing inputs, and are safe to
rerun.

Using the generated tables in the paper source:

```latex
\input{runs_paper/tables/multiseed_best_methods}      % MNIST main
\input{runs_paper/tables/multiseed_all_methods}       % MNIST appendix
\input{runs_paper/tables/variant_ablation_table}      % cross-variant ablations
\input{runs_paper/tables/hw_compare_k16}              % hardware section
\input{runs_paper/tables/hw_compare_k32}
\input{runs_conv_cifar/tables/table_main_conv}        % CIFAR main rows
\input{runs_conv_cifar/tables/table_iso_storage}      % matched-bits comparison
```

---

## 7. Run-to-table map

| paper artefact | produced by | output |
|---|---|---|
| MNIST main table | phase 01 → phase 05 | `runs_paper/tables/multiseed_best_methods.tex` |
| MNIST full appendix table | phase 01 → phase 05 | `runs_paper/tables/multiseed_all_methods.tex` |
| Fashion-MNIST | phase 01 | `runs_paper/fashion_mnist/combined_summary_w4.csv` |
| Tabular suite | phase 02 | `runs_paper/tabular/combined_summary_w4.csv` |
| CIFAR rows (reported) | phase 09 tier1+tier2 | `runs_conv_cifar/tables/table_main_conv.tex` |
| Matched-bits/edge comparison | phase 09 | `runs_conv_cifar/tables/table_iso_storage.tex` |
| Flattened-CIFAR rows (legacy) | phase 03 → phase 05 | `runs_paper/tables/cifar10_table.tex`, `cifar100_table.tex` |
| Cross-variant ablations | phase 04 → phase 05 | `runs_paper/tables/variant_ablation_table.tex` |
| SplineKAN ablation suite | phase 04 | per-run `runs/<name>/summary.csv` |
| FPGA / BRAM **estimates** | phase 04 → phase 05 | `runs_paper/tables/hw_compare_k16.tex`, `hw_compare_k32.tex` |
| FPGA **measured** resources (all 9 designs) | `scripts/hw/13_reports.sh` | `runs_hw/tables/hw_comparison.tex`, `hw_report_summary.csv` |
| Post-synthesis / post-route reports | shipped | `hw/results/<design>/{syn,syn_alt,impl_alt}/` |
| Verification ladder L1–L4 | shipped | `hw/verification_log.json` |
| Deployed accuracies (9 designs) | shipped | `hw/golden/<design>/manifest.json` |
| Storage breakdown | `tools/summarize_storage_breakdown.py` | stdout / CSV |
| Edge-redundancy analysis | phase 08 | `runs_paper/redundancy/` |
| Accuracy-vs-storage Pareto | phase 05 | `runs_paper/tables/pareto_*.csv` |
| Conv ablations | phase 09 tier3 | `runs_conv_cifar/tables/conv_cifar_all.csv` |
| Tiny ImageNet | phase 06 | `runs_paper/tiny_imagenet/` |
| Sensitivity sweeps | phase 07 | `runs_paper/sensitivity/` |

---

## 8. Time and storage budget

Wall-clock on the reference hardware (two 32 GB data-centre GPUs, 16 CPU cores).
Scale by the times `00_smoke.sh` prints on your machine.

**Track A** — single GPU:

| phase | wall-clock |
|---|---|
| 00 smoke | 15 min |
| 01 MNIST family, 3 seeds | 4–6 h |
| 02 tabular, 5 datasets × 5 seeds | 2–4 h |
| 03 flattened CIFAR | 12–18 h |
| 04 ablation suite | 6–10 h |
| 05 tables | minutes (CPU) |
| 06 Tiny ImageNet | long (tier 2) |
| 07 sensitivity | 2–3 h |
| 08 redundancy | 2–3 h |
| **tier1 total** | **≈ 1.5–2 GPU-days** |

**Track B** — two GPUs, 2 slots each:

| tier | jobs | wall-clock |
|---|---|---|
| smoke | 2 | ~10 min |
| tier1 | ~43 | ~14 h |
| tier2 | — | ~27 h |
| tier3 | — | ~15 h |

The dominant single cost is a dense 8-layer backbone: ~4 GPU-hours at 200
epochs (~39 ms/step, 351 steps/epoch). This is why the queue makes dense runs
priority 0 and why every compression arm reuses the checkpoint instead of
retraining.

**Disk.** Budget ~1 GB for datasets and, if you keep dense checkpoints, ~120 MB
per conv backbone (one per dataset × seed). Full conv run outputs including
checkpoints reach several hundred GB; pass `--out-dir` per tier and delete
`dense.pt` files once their arms have completed if space is tight.

---

## 9. Restarting, resuming, and partial reproduction

- **The queue resumes.** Re-running `09_conv_cifar.sh tier1` skips jobs already
  recorded complete in `runs_conv_cifar/_queue`. Delete that directory to force
  a full rerun.
- **Dense checkpoints are reusable.** Point `--dense-ckpt` at an existing
  `dense.pt` and pass `--require-ckpt` to fail loudly rather than silently
  retraining.
- **Table generation is idempotent.** Phases 05 and `09 ... tables` can be run
  as often as you like, including against partial results.
- **To reproduce only the headline claim**, the minimum path is: phase 00, the
  two dense conv backbones at seed 42, the FuncCode `function` arms at
  `K ∈ {8,16,32,64,256}`, the `pq`/`uniform`/`iso` baselines, then
  `09 ... tables`. That gives `table_main_conv.tex` and `table_iso_storage.tex`
  at a single seed, without the error bars.
- **To check the analytical claims with no training at all**, run `pytest
  tests/ -q`: the storage accountant, the bit-packing, the export format and the
  Gram-whitening identity are all verified on synthetic models.
