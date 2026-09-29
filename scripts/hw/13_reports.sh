#!/usr/bin/env bash
# Parse archived HLS reports -> tidy CSV -> LaTeX + markdown comparison
# tables. One command regenerates everything from raw report files.
set -euo pipefail
cd "$(dirname "$0")/../.."

python tools/parse_hls_reports.py \
  --results-root hw/results \
  --out runs_hw/tables/hw_report_summary.csv

python tools/make_hw_comparison_table.py \
  --csv runs_hw/tables/hw_report_summary.csv \
  --runs-dir runs_hw \
  --out-tex runs_hw/tables/hw_comparison.tex \
  --out-md docs/HW_COMPARISON.md

echo "Tables written to runs_hw/tables/ and docs/HW_COMPARISON.md"
