#!/usr/bin/env bash
# GRAM (G-series) + KAGN-Conv (K-series) model preparation. Mirrors
# 10_train_models.sh. Windows note: --num-workers 0 is the default in
# these drivers.
set -euo pipefail
cd "$(dirname "$0")/../.."

# GRAM: verified all_kan_mnist source run (dense + branch Ks=32/Kb=16)
python -m funcodekan.experiments.hw_prepare_gram --model gram_source
# G2: LSQ W4A4 QAT (20 epochs) ; G3: codebook finetune (25 epochs)
python -m funcodekan.experiments.hw_prepare_gram --model gram_lsq_w4a4
python -m funcodekan.experiments.hw_prepare_gram --model gram_funccode_w4a4

# KAGN-Conv (NEW additive research code; see HW_FAIRNESS addendum).
# K2 trains W4A4 from scratch (post-hoc W4 on the FP32 dense collapses —
# measured); K3 clusters the K2-trained (W4-native) weights.
python -m funcodekan.experiments.hw_prepare_kagn --model kagnconv_source
python -m funcodekan.experiments.hw_prepare_kagn --model kagnconv_lsq_w4a4 --from-scratch
python -m funcodekan.experiments.hw_prepare_kagn --model kagnconv_funccode_w4a4 --cluster-from k2
