#!/usr/bin/env bash
# Master runner. Usage: bash scripts/paper/run_all.sh [tier1|tier2|all]
# Continues past failed phases; prints a pass/fail summary at the end.
TIER=${1:-tier1}
mkdir -p logs/paper
declare -a RESULTS

phase() { local name=$1; shift
  echo "==================== ${name} ===================="
  local s=$SECONDS
  if "$@"; then RESULTS+=("PASS  ${name}  $(( (SECONDS-s)/60 ))min")
  else          RESULTS+=("FAIL  ${name}  $(( (SECONDS-s)/60 ))min"); fi }

phase 00_smoke              bash scripts/paper/00_smoke.sh
phase 01_mnist_family       bash scripts/paper/01_mnist_family.sh main
phase 02_tabular            bash scripts/paper/02_tabular.sh
phase 03_cifar              bash scripts/paper/03_cifar.sh main
phase 04_ablations          bash scripts/paper/04_ablations.sh
phase 05_tables             bash scripts/paper/05_hardware_and_tables.sh

if [[ "${TIER}" == "tier2" || "${TIER}" == "all" ]]; then
  phase 08_redundancy       bash scripts/paper/08_redundancy.sh
  phase 07_sensitivity      bash scripts/paper/07_sensitivity.sh
  phase 06_tiny_imagenet    bash scripts/paper/06_tiny_imagenet.sh main
  phase 06_tiny_spline      bash scripts/paper/06_tiny_imagenet.sh spline_instability
  phase 03_cifar_extra      bash scripts/paper/03_cifar.sh extra_seeds
  phase 05_tables_refresh   bash scripts/paper/05_hardware_and_tables.sh
fi

echo; echo "================ SUMMARY ================"
printf '%s\n' "${RESULTS[@]}"
