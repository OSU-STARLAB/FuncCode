#!/usr/bin/env bash
# Vivado HLS flow for the GRAM and KAGN-Conv designs: csim(1000) ->
# csynth + cosim(50) on xczu9eg -> ZU7EV csynth + post-implementation.
# Unlike the spline family, the GRAM/KAGN FP32 designs FIT the licensed
# ZU7EV (992.5 / 99.1 KiB of params), so all three designs of each family
# get post-implementation numbers.
set -euo pipefail
cd "$(dirname "$0")/../.."

DESIGNS=("$@")
if [ ${#DESIGNS[@]} -eq 0 ]; then
  DESIGNS=(gram_fp32 gram_lsq_w4a4 gram_funccode_w4a4
           kagnconv_fp32 kagnconv_lsq_w4a4 kagnconv_funccode_w4a4)
fi
PART7=xczu7ev-ffvc1156-2-e

for d in "${DESIGNS[@]}"; do
  vivado_hls -f hw/tcl/run_hls.tcl -tclargs "$d" csim
  vivado_hls -f hw/tcl/run_hls.tcl -tclargs "$d" impl 50
  vivado_hls -f hw/tcl/run_hls.tcl -tclargs "$d" synth 50 "$PART7"
  vivado -mode batch -source hw/tcl/impl_eval.tcl -tclargs "$d" "$PART7"

  mkdir -p "hw/results/$d/syn" "hw/results/$d/sim" "hw/results/$d/syn_alt"
  cp -f hw/proj/"$d"/sol1/syn/report/*     "hw/results/$d/syn/"     2>/dev/null || true
  cp -f hw/proj/"$d"/sol1/sim/report/*     "hw/results/$d/sim/"     2>/dev/null || true
  cp -f hw/proj/"$d"_alt/sol1/syn/report/* "hw/results/$d/syn_alt/" 2>/dev/null || true
  cp -f hw/proj/"$d"_csim/sol1/csim/report/*.log "hw/results/$d/"   2>/dev/null || true
done

echo "GRAM/KAGN HLS runs complete. Reports in hw/results/."
