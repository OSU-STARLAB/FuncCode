#!/usr/bin/env bash
# Phase 05 (E5) — CPU post-processing: summaries, HW/BRAM analysis, LaTeX
# tables, Pareto CSVs. Safe to rerun any time; skips missing inputs.
set -e
mkdir -p logs/paper runs_paper/tables

have() { [[ -e "$1" ]]; }

# --- MNIST multiseed tables (T1) ---
if have runs_multiseed; then
  python tools/summarize_multiseed_all_kan.py --root runs_multiseed \
    --out runs_multiseed/all_kan_mnist_multiseed_summary.csv
  python tools/select_best_multiseed_methods.py \
    --csv runs_multiseed/all_kan_mnist_multiseed_summary.csv \
    --out runs_multiseed/best_multiseed_methods.csv
  python tools/make_multiseed_latex_table.py \
    --csv runs_multiseed/all_kan_mnist_multiseed_summary.csv \
    --out-tex runs_paper/tables/multiseed_all_methods.tex
  python tools/make_multiseed_latex_table.py \
    --csv runs_multiseed/best_multiseed_methods.csv \
    --out-tex runs_paper/tables/multiseed_best_methods.tex
fi

# --- Fashion-MNIST + tabular + width sweep summaries ---
for ROOT in runs_paper/fashion_mnist runs_paper/tabular runs_paper/width_sweep; do
  if have ${ROOT}; then
    python tools/summarize_all_kan_mnist.py --root ${ROOT} \
      --out ${ROOT}/combined_summary_w4.csv || true
  fi
done

# --- Variant-ablation table (T2 companion) ---
if have runs_variant_ablations; then
  python tools/summarize_variant_ablations.py --root runs_variant_ablations \
    --out runs_variant_ablations/variant_ablation_summary.csv
  python tools/make_variant_ablation_latex_table.py \
    --csv runs_variant_ablations/variant_ablation_summary.csv \
    --out-tex runs_paper/tables/variant_ablation_table.tex
fi

# --- HW / BRAM analysis from Phase-04 reference runs (T3) ---
if have runs/func_k16_ref/summary.csv; then
  python tools/compare_hw_estimates.py --summaries \
      runs/func_k16_ref/summary.csv \
      runs/branch_k16_s16_b8_ref/summary.csv \
      runs/branch_index_k16_ref/summary.csv \
      runs/srb_k16_r10_b8/summary.csv \
      runs/srb_k16_r25_b8/summary.csv \
      runs/srb_k16_r50_b8/summary.csv \
    --out runs_paper/tables/hw_compare_k16.csv
  python tools/make_paper_hw_table.py --hw-csv runs_paper/tables/hw_compare_k16.csv \
    --out-tex runs_paper/tables/hw_compare_k16.tex \
    --caption "Analytical FPGA-memory estimates for K=16 compressed KAN variants." \
    --label "tab:hw_k16"
fi
if have runs/func_k32_ref/summary.csv; then
  python tools/compare_hw_estimates.py --summaries \
      runs/func_k32_ref/summary.csv \
      runs/branch_k32_s32_b16_ref/summary.csv \
      runs/srb_k32_r10_b8/summary.csv \
      runs/srb_k32_r25_b8/summary.csv \
      runs/srb_k32_r50_b8/summary.csv \
    --out runs_paper/tables/hw_compare_k32.csv
  python tools/make_paper_hw_table.py --hw-csv runs_paper/tables/hw_compare_k32.csv \
    --out-tex runs_paper/tables/hw_compare_k32.tex \
    --caption "Analytical FPGA-memory estimates for K=32 compressed KAN variants." \
    --label "tab:hw_k32"
fi

# --- CIFAR clean tables (T1) ---
if have runs_cifar/cifar10_main_seed42; then
  mkdir -p runs_cifar_clean/cifar10_multiseed runs_cifar_clean/cifar100
  cp -rn runs_cifar/cifar10_main_seed* runs_cifar_clean/cifar10_multiseed/ 2>/dev/null || true
  cp -rn runs_cifar/cifar100_main_seed* runs_cifar_clean/cifar100/ 2>/dev/null || true
  python tools/summarize_cifar_runs.py --root runs_cifar_clean/cifar10_multiseed \
    --out runs_cifar_clean/cifar10_summary.csv --stage clustered_hwq_w4
  python tools/make_cifar_latex_table.py --csv runs_cifar_clean/cifar10_summary.csv \
    --out-tex runs_paper/tables/cifar10_table.tex --dataset cifar10 \
    --stage clustered_hwq_w4 --best-only \
    --caption "Multi-seed CIFAR-10 compressed KAN results under W4 codebook quantization." \
    --label "tab:cifar10"
  python tools/summarize_cifar_runs.py --root runs_cifar_clean/cifar100 \
    --out runs_cifar_clean/cifar100_summary.csv --stage clustered_hwq_w4
  python tools/make_cifar_latex_table.py --csv runs_cifar_clean/cifar100_summary.csv \
    --out-tex runs_paper/tables/cifar100_table.tex --dataset cifar100 \
    --stage clustered_hwq_w4 --best-only \
    --caption "CIFAR-100 compressed KAN results under W4 codebook quantization." \
    --label "tab:cifar100"
fi

# --- Pareto data for F3 ---
for ROOT in runs_multiseed runs_paper/fashion_mnist runs_paper/tabular runs_cifar; do
  if have ${ROOT}; then
    NAME=$(basename ${ROOT})
    python tools/make_pareto_data.py --roots ${ROOT} \
      --out runs_paper/tables/pareto_${NAME}.csv || true
  fi
done

echo "Tables and figure data in runs_paper/tables/"
