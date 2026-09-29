#!/usr/bin/env bash
# Export golden vectors + params headers for all three designs and run the
# L1 (+L4) verification gates. Fails hard if any L1 rung fails.
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m funcodekan.hw.export --design d1
python -m funcodekan.hw.export --design d2
python -m funcodekan.hw.export --design d3

echo "Golden exports complete; see hw/golden/*/manifest.json and"
echo "runs_hw/verification_log.json for the L1/L4 records."
