#!/usr/bin/env bash
# Phase 04 (E4) — full SplineKAN ablation suite + cross-family ablations.
set -e
mkdir -p logs/paper
bash scripts/mnist/run_function_space.sh all    2>&1 | tee logs/paper/04_funcspace.log   # E4a
bash scripts/mnist/run_branch_aware.sh all      2>&1 | tee logs/paper/04_branch.log      # E4b
bash scripts/mnist/run_index_efficient.sh all   2>&1 | tee logs/paper/04_idx.log         # E4c
bash scripts/mnist/run_srb_codebooks.sh all     2>&1 | tee logs/paper/04_srb.log         # E4c
bash scripts/mnist/run_soft_codebook.sh sweep   2>&1 | tee logs/paper/04_soft.log        # E4d
bash scripts/mnist/run_variant_ablations.sh multiseed 2>&1 | tee logs/paper/04_variant_abl.log
