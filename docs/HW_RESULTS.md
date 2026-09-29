# HW_RESULTS.md — Complete results for all nine accelerators

The single reference document for every model and hardware result of the
FuncCode-KAN accelerator project (three families × three designs).
Every number is validated against the repository artifact named in each
section; the two comparison tables regenerate from the archived reports
with `bash scripts/hw/13_reports.sh`.

Seed 42 throughout. Accuracy convention: unless labeled otherwise, the
canonical accuracy of a quantized design is the **deployed-arithmetic
accuracy on the full 10k MNIST test set** (the L1-verified emulator/eval
path, from `hw/golden/<design>/manifest.json`) — not the training-time
floating-point proxy.

---

## 1. The nine designs

| ID | Design dir | Model | Weights | Acts |
|---|---|---|---|---|
| D1 | `fp32` | Spline KAN 784→64→10 (G=5, k=3) | FP32 dense | FP32 |
| D2 | `lsq_w4a4` | same | INT4 dense (LSQ QAT) | INT4 |
| D3 | `funccode_w4a4` | same | INT4 codebooks Ks=32/Kb=16 + 5b/4b ids | INT4 |
| G1 | `gram_fp32` | GRAM KAN 784→64→10 (degree 3) | FP32 dense | FP32 |
| G2 | `gram_lsq_w4a4` | same | INT4 dense (LSQ QAT) | INT4 |
| G3 | `gram_funccode_w4a4` | same | INT4 codebooks Ks=32/Kb=16 + 5b/4b ids | INT4 |
| K1 | `kagnconv_fp32` | GramConv2D 1→16→32 (3×3,s2) + GAP + GRAM head | FP32 dense | FP32 |
| K2 | `kagnconv_lsq_w4a4` | same | INT4 dense (QAT from scratch) | INT4 |
| K3 | `kagnconv_funccode_w4a4` | same | INT4 codebooks Ks=64/Kb=16 + 6b/4b ids | INT4 |

Edge/parameter counts: spline 50,816 edges × 9 = 457,344 params;
GRAM 50,816 × 5 = 254,080; KAGN-conv 5,072 kernel-position edges × 5 =
25,360.

---

## 2. Model accuracy — every training stage

### 2.1 Source runs (verified pipeline for D/G; K1 reference run)
<!-- source: runs_hw/branch_k32_s32_b16/summary.csv,
     runs_hw/gram_hw_source/gram/summary.csv,
     runs_hw/kagnconv_hw_source/summary.csv -->

| Stage | Spline acc % | GRAM acc % | KAGN-conv acc % |
|---|---|---|---|
| dense FP32 | **96.28** | **96.68** | **95.43** |
| branch-clustered, before finetune | 94.94 | 86.73 | 10.32 |
| branch-clustered, finetuned (FP32 codebooks) | 96.05 | 96.30 | 91.15 |
| codebook PTQ W8 | 96.05 | 96.30 | — |
| codebook PTQ W6 | 96.03 | 96.36 | — |
| codebook PTQ **W4** | **95.55** | **96.36** | — |
| codebook PTQ W3 | 95.92 | 96.02 | — |
| codebook PTQ W2 | 94.23 | 90.51 | — |

Notes: spline branch (32,16) W4 = **31.64×** compression exactly (the
paper-pinned number). The KAGN source clustering of FP32-trained conv
weights is near-random before finetune (10.32%) — an early symptom of
the W4-hostility of those weights (§2.3); the K-series therefore uses
its own protocol and this reference run feeds only K1.

### 2.2 Deployed W4A4 models (canonical accuracies)
<!-- source: hw/golden/<design>/manifest.json (L1) -->

| Design | Deployed acc % | Δ vs family dense (pp) | Training epochs / protocol |
|---|---|---|---|
| D2 | 96.22 | 0.06 | 20-epoch LSQ QAT from dense ckpt |
| D3 | 95.22 | 1.06 | 25-epoch codebook+step finetune, indices frozen |
| G2 | **97.23** | **−0.55 (above dense)** | 20-epoch LSQ QAT from dense ckpt |
| G3 | 96.09 | 0.59 | 25-epoch codebook finetune, indices frozen |
| K2 | 92.74 | 2.69 | **30-epoch QAT from scratch** |
| K3 | 89.78 | 5.65 | 40-epoch codebook finetune, clustered **from K2**, Ks=64 |

FP32 baselines deploy at their dense accuracies (D1 96.28, G1 96.68,
K1 95.43; emulator ≤1e-4 / ≤1e-3 of the torch reference with exact
argmax on 10k).

### 2.3 Protocol-exploration results (recorded, superseded)
<!-- source: runs_hw/verification_log.json; run dirs retained -->

| Attempt | Result | Disposition |
|---|---|---|
| D3, 10-epoch finetune | 95.07% | superseded by 25-epoch (95.20 train-eval / 95.22 deployed) |
| D3, 50-epoch finetune | 95.16% | saturation; 25-epoch kept |
| KAGN W4 PTQ on FP32 dense (no QAT) | **23.4%** (~25% of conv weights round to 0) | motivated K2 from-scratch |
| K2 QAT initialized from dense | 91.22% | superseded by from-scratch (92.74) |
| K3 clustered from FP32 dense | 66.44% | superseded by cluster-from-K2 |
| K3 from K2, Ks=32, 25 ep | 87.79% | superseded by Ks=64 |
| K3 from K2, Ks=64, 25 ep | 89.60% | superseded by 40 ep (89.78; saturated) |

---

## 3. Storage and compression (bit-exact analytical model)
<!-- source: runs_hw/<run>/summary.csv, conventions of
     funcodekan.analysis.storage -->

| Design | Weight storage | Compression vs family dense | Index-stream share |
|---|---|---|---|
| D1 | 1,786.500 KiB | 1.00× | — |
| D2 | 223.336 KiB | 8.00× | — |
| D3 | 56.469 KiB | **31.64×** | **98.9%** (55.83 KiB of ids) |
| G1 | 992.500 KiB | 1.00× | — |
| G2 | 124.086 KiB | 8.00× | — |
| G3 | 56.344 KiB | **17.62×** | **99.1%** |
| K1 | 99.063 KiB | 1.00× | — |
| K2 | 12.422 KiB | 7.98× | — |
| K3 | 7.527 KiB | **13.16×** | **82.3%** (6.19 KiB ids + 0.40 codebooks + 0.94 scales) |

FuncCode-vs-LSQ weight-storage ratios: D 3.95×, G 2.20×, K 1.65×.

---

## 4. Hardware — post-synthesis (Vivado HLS 2019.1, `xczu9eg-ffvb1156-2-e`, 6.67 ns target)
<!-- source: hw/results/<design>/syn/csynth.xml via
     runs_hw/tables/hw_report_summary.csv -->

| Design | Est. clk (ns) | Cycles (worst) | LUT | FF | DSP | BRAM18 | Latency (µs) | Img/s |
|---|---|---|---|---|---|---|---|---|
| D1 | 5.78 | 1,062,457 | 15,928 | 15,328 | 48 | 842 | 7,086.6 | 141 |
| D2 | 6.50 | 50,983 | 3,374 | 5,457 | 24 | 108 | 340.1 | 2,941 |
| D3 | 6.50 | 50,983 | 3,197 | 5,523 | 23 | **32** | 340.1 | 2,941 |
| G1 | 5.78 | 289,938 | 17,935 | 15,846 | 62 | 469 | 1,933.9 | 517 |
| G2 | 6.60 | 51,512 | 34,571 | 42,863 | 52 | 76 | 343.6 | 2,910 |
| G3 | 6.60 | 51,512 | 34,381 | 42,893 | 51 | **43** | 343.6 | 2,910 |
| K1 | 11.39† | 1,877,859 | 37,810 | 28,421 | 109 | 96 | 12,525.3 | 80 |
| K2 | 6.70‡ | 294,316 | 61,532 | 71,169 | 84 | 41 | 1,963.1 | 509 |
| K3 | 6.70‡ | 294,316 | 61,008 | 70,873 | 82 | **36** | 1,963.1 | 509 |

† csynth estimation pessimism for the floating-point cores — post-route
timing is MET at 5.30 ns (§5), so the target-period latency is valid.
‡ estimate marginally above target; post-route met with ≥0.7 ns slack.
Latency µs = worst-case cycles × 6.67 ns; batch = 1. Within each family
the LSQ and FuncCode designs have IDENTICAL cycle counts and II=1 fused
main loops (synthesis loop reports) — the fairness signature.

## 5. Hardware — post-implementation (Vivado place-and-route, `xczu7ev-ffvc1156-2-e`, 6.67 ns constraint)
<!-- source: hw/results/<design>/impl_alt/{utilization,timing}_route.rpt;
     achieved period = constraint − WNS -->

| Design | Achieved clk (ns) | LUT | FF | DSP | BRAM18 | Timing |
|---|---|---|---|---|---|---|
| D1 | — | — | — | — | — | n/a: 397 BRAM36 of FP32 ROMs exceed this part |
| D2 | 6.56 | 3,546 | 4,935 | 32 | 147 | met |
| D3 | 6.20 | 3,554 | 4,936 | 31 | **38** | met |
| G1 | 6.22 | 15,084 | 13,393 | 62 | 608 | met |
| G2 | 6.27 | 32,019 | 39,345 | 64 | 90 | met |
| G3 | 6.20 | 31,934 | 39,346 | 59 | **46** | met |
| K1 | **5.30** | 26,870 | 22,968 | 109 | 118 | met (+1.371 ns slack, 0/45,480 endpoints failing) |
| K2 | 5.93 | 52,634 | 64,956 | 120 | 38 | met |
| K3 | 5.89 | 52,966 | 64,892 | 120 | **32** | met |

D1's absence is a licensing/part reality (its ROMs don't fit the
licensed ZU7EV and the ZU9EG has no RTL-synthesis license here);
post-synthesis and post-implementation numbers are never mixed in any
comparison.

## 6. Derived stage-by-stage gains

| Metric | Spline | GRAM | KAGN-conv |
|---|---|---|---|
| FP32→LSQ: weight storage | 8.0× | 8.0× | 8.0× |
| FP32→LSQ: BRAM18 (syn) | 7.8× (842→108) | 6.2× (469→76) | 2.3× (96→41) |
| FP32→LSQ: latency | 20.8× | 5.6× | 6.4× |
| FP32→LSQ: accuracy cost | 0.06 pp | −0.55 pp (gain) | 2.69 pp |
| LSQ→FuncCode: weight storage | 3.95× | 2.20× | 1.65× |
| LSQ→FuncCode: BRAM18 (syn / route) | 3.4× / 3.87× | 1.77× / 1.96× | 1.14× / 1.19× |
| LSQ→FuncCode: latency | 1.00× (identical) | 1.00× | 1.00× |
| LSQ→FuncCode: accuracy cost | 1.00 pp | 1.14 pp | 2.96 pp |
| FP32→FuncCode composite BRAM (syn) | **26.3×** | 10.9× | 2.7× |

## 7. Verification ladder — final status (all nine designs complete)
<!-- source: runs_hw/verification_log.json; cosim reports under
     hw/results/<design>/sim/ -->

| Design | L1 (torch↔emulator, 10k) | L2 (csim, 1000) | L3 (RTL cosim, 50) | L4 (≤0.3 pp vs dense) |
|---|---|---|---|---|
| D1 | ≤1e-4, argmax 10000/10000 | ≤1e-3, 1000/1000 | PASS 50/50 (measured RTL 983,593 cyc) | — |
| D2 | argmax **10000/10000** | **bit-exact** 1000/1000 | **bit-exact** 50/50 | ✓ (0.06) |
| D3 | argmax **10000/10000** | **bit-exact** 1000/1000 | **bit-exact** 50/50 | ✗ (1.06; analysis in HW_FAIRNESS) |
| G1 | ≤1e-4, 10000/10000 | ≤1e-3, 1000/1000 | PASS 50/50 (after tanhf-deadlock fix) | — |
| G2 | argmax **10000/10000** | **bit-exact** 1000/1000 | **bit-exact** 50/50 | ✓ (−0.55) |
| G3 | argmax **10000/10000** | **bit-exact** 1000/1000 | **bit-exact** 50/50 | ✗ (0.59; within the 1 pp family gate) |
| K1 | ≤1e-3, 10000/10000 | ≤1e-3, 1000/1000 | PASS 50/50 (zero violations; 625.3 ms sim) | — |
| K2 | argmax **10000/10000** (emulator match EXACT, 0.0) | **bit-exact** 1000/1000 | **bit-exact** 50/50 | ✗ (2.69; conv W4A4 cost, reported) |
| K3 | argmax **10000/10000** (EXACT) | **bit-exact** 1000/1000 | **bit-exact** 50/50 | ✗ (5.65; conv sharing cost, reported) |

Golden subsets: 1000 stratified test images (first 100/class,
deterministic, SHA256-pinned); cosim = the first 50 of those, identical
for every design.

## 8. Regeneration & provenance

```bash
bash scripts/hw/13_reports.sh      # regenerates the comparison tables from hw/results/
pytest tests/ -q                   # full suite, including the HW emulator/export tests
```

What ships with this release, and where each number comes from:

| claim | shipped artefact |
|---|---|
| Deployed accuracies (2.2) | `hw/golden/<design>/manifest.json` (SHA256-pinned) |
| Post-synthesis resources (4) | `hw/results/<design>/syn/csynth.xml`, `syn_alt/` |
| Post-implementation resources (5) | `hw/results/<design>/impl_alt/{utilization,timing}_route.rpt` |
| C/RTL co-simulation (7) | `hw/results/<design>/sim/kan_top_cosim.rpt`, `kan_top_csim.log` |
| Verification ladder L1-L4 (7) | `hw/verification_log.json` (append-only, includes superseded attempts) |
| Accuracy + storage stage tables (2.1, 3) | `runs_hw/<run>/summary.csv` |
| Exact run settings | `configs/hw/<run>.json` |

The `runs_hw/<run>/summary.csv` files ship because
`tools/make_hw_comparison_table.py` reads them for the accuracy and
parameter-storage columns; the training runs that produced them (checkpoints,
logs) are not redistributed but are regenerated by
`scripts/hw/10_train_models.sh` and `scripts/hw/20_train_gram_kagn.sh`.

Companion documents: `HW_FAIRNESS.md` (what is identical and what differs per
design, plus the L4 analyses), `HW_DESIGN_CONTRACT.md` (the frozen integer
datapath), `HW_MANUAL.md` / `HW_MANUAL_GRAM_KAGN.md` (how to reproduce every
number here), and `hw/README.md` (build and verification quickstart).
