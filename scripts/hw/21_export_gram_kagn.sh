#!/usr/bin/env bash
# Golden exports + L1/L4 gates for the GRAM and KAGN-Conv designs.
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m funcodekan.hw.gram_export --design g1
python -m funcodekan.hw.gram_export --design g2
python -m funcodekan.hw.gram_export --design g3

python -m funcodekan.hw.kagn_export --design k1
python -m funcodekan.hw.kagn_export --design k2
python -m funcodekan.hw.kagn_export --design k3
