# FuncCode-KAN hardware comparison (auto-generated)

Regenerate with `bash scripts/hw/13_reports.sh`.

### Hardware comparison (post-synthesis, xczu9eg)

| ID | Design | Acc (%) | Params (KiB) | LUT | FF | DSP | BRAM18 | Latency (us) | Img/s |
|---|---|---|---|---|---|---|---|---|---|
| D1 | Spline FP32 dense | 96.28 | 1786.5 | 15928 | 15328 | 48 | 842 | 7086.6 | 141 |
| D2 | Spline LSQ W4A4 | 96.22 | 223.3 | 3374 | 5457 | 24 | 108 | 340.1 | 2941 |
| D3 | Spline FuncCode W4A4 | 95.22 | 56.5 | 3197 | 5523 | 23 | 32 | 340.1 | 2941 |
| G1 | GRAM FP32 dense | 96.68 | 992.5 | 17935 | 15846 | 62 | 469 | 1933.9 | 517 |
| G2 | GRAM LSQ W4A4 | 97.23 | 124.1 | 34571 | 42863 | 52 | 76 | 343.6 | 2910 |
| G3 | GRAM FuncCode W4A4 | 96.09 | 56.3 | 34381 | 42893 | 51 | 43 | 343.6 | 2910 |
| K1 | KAGN-Conv FP32 dense | 95.43 | 99.1 | 37810 | 28421 | 109 | 96 | 12525.3 | 80 |
| K2 | KAGN-Conv LSQ W4A4 | 92.74 | 12.4 | 61532 | 71169 | 84 | 41 | 1963.1 | 509 |
| K3 | KAGN-Conv FuncCode W4A4 | 89.78 | 7.5 | 61008 | 70873 | 82 | 36 | 1963.1 | 509 |

Sources:
- D1: `hw\results\fp32\syn\csynth.xml`
- D2: `hw\results\lsq_w4a4\syn\csynth.xml`
- D3: `hw\results\funccode_w4a4\syn\csynth.xml`
- G1: `hw\results\gram_fp32\syn\csynth.xml`
- G2: `hw\results\gram_lsq_w4a4\syn\csynth.xml`
- G3: `hw\results\gram_funccode_w4a4\syn\csynth.xml`
- K1: `hw\results\kagnconv_fp32\syn\csynth.xml`
- K2: `hw\results\kagnconv_lsq_w4a4\syn\csynth.xml`
- K3: `hw\results\kagnconv_funccode_w4a4\syn\csynth.xml`

### Hardware comparison (post-implementation, xczu7ev; all designs whose parameters fit this licensed part -- the spline FP32 baseline (D1) does not (HW_FAIRNESS.md))

| ID | Design | Acc (%) | Params (KiB) | LUT | FF | DSP | BRAM18 | Latency (us) | Img/s |
|---|---|---|---|---|---|---|---|---|---|
| D2 | Spline LSQ W4A4 | 96.22 | 223.3 | 3546 | 4935 | 32 | 147 | 340.1 | 2941 |
| D3 | Spline FuncCode W4A4 | 95.22 | 56.5 | 3554 | 4936 | 31 | 38 | 340.1 | 2941 |
| G1 | GRAM FP32 dense | 96.68 | 992.5 | 15084 | 13393 | 62 | 608 | 1933.9 | 517 |
| G2 | GRAM LSQ W4A4 | 97.23 | 124.1 | 32019 | 39345 | 64 | 90 | 343.6 | 2910 |
| G3 | GRAM FuncCode W4A4 | 96.09 | 56.3 | 31934 | 39346 | 59 | 46 | 343.6 | 2910 |
| K1 | KAGN-Conv FP32 dense | 95.43 | 99.1 | 26870 | 22968 | 109 | 118 | 12525.3 | 80 |
| K2 | KAGN-Conv LSQ W4A4 | 92.74 | 12.4 | 52634 | 64956 | 120 | 38 | 1963.1 | 509 |
| K3 | KAGN-Conv FuncCode W4A4 | 89.78 | 7.5 | 52966 | 64892 | 120 | 32 | 1963.1 | 509 |

Sources:
- D2: `hw\results\lsq_w4a4\impl_alt\utilization_route.rpt`
- D3: `hw\results\funccode_w4a4\impl_alt\utilization_route.rpt`
- G1: `hw\results\gram_fp32\impl_alt\utilization_route.rpt`
- G2: `hw\results\gram_lsq_w4a4\impl_alt\utilization_route.rpt`
- G3: `hw\results\gram_funccode_w4a4\impl_alt\utilization_route.rpt`
- K1: `hw\results\kagnconv_fp32\impl_alt\utilization_route.rpt`
- K2: `hw\results\kagnconv_lsq_w4a4\impl_alt\utilization_route.rpt`
- K3: `hw\results\kagnconv_funccode_w4a4\impl_alt\utilization_route.rpt`
