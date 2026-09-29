#!/usr/bin/env bash
# Vivado HLS flow for all three designs: csim (1000 vectors, L2) then
# csynth + cosim (50 vectors, L3) + export_design -flow impl. Archives every
# report under hw/results/<design>/.
#
# Requires vivado_hls (2019.x) on PATH; on this project's Windows host:
#   $env:PATH += ";C:\Xilinx\Vivado\2019.1\bin"
# Usage: bash scripts/hw/12_run_hls.sh [design ...]   (default: all three)
set -euo pipefail
cd "$(dirname "$0")/../.."

DESIGNS=("$@")
if [ ${#DESIGNS[@]} -eq 0 ]; then
  DESIGNS=(fp32 lsq_w4a4 funccode_w4a4)
fi
COSIM_N=50

for d in "${DESIGNS[@]}"; do
  echo "=== $d : csim (1000 golden vectors) ==="
  vivado_hls -f hw/tcl/run_hls.tcl -tclargs "$d" csim

  echo "=== $d : csynth + cosim ($COSIM_N vectors) ==="
  vivado_hls -f hw/tcl/run_hls.tcl -tclargs "$d" impl "$COSIM_N"

  echo "=== $d : Vivado out-of-context synth+P&R (post-impl numbers) ==="
  vivado -mode batch -source hw/tcl/impl_eval.tcl -tclargs "$d"

  if [ "$d" != "fp32" ]; then
    echo "=== $d : SECONDARY ZU7EV csynth + post-impl (D2/D3 only) ==="
    # D1's FP32 ROMs exceed the ZU7EV's BRAM; the licensed-part post-impl
    # comparison is therefore D2-vs-D3 only (docs/HW_FAIRNESS.md).
    vivado_hls -f hw/tcl/run_hls.tcl -tclargs "$d" synth 50 xczu7ev-ffvc1156-2-e
    vivado -mode batch -source hw/tcl/impl_eval.tcl -tclargs "$d" xczu7ev-ffvc1156-2-e
  fi

  echo "=== $d : archiving reports ==="
  mkdir -p "hw/results/$d/syn" "hw/results/$d/sim" "hw/results/$d/syn_alt"
  cp -f hw/proj/"$d"/sol1/syn/report/*.rpt "hw/results/$d/syn/" 2>/dev/null || true
  cp -f hw/proj/"$d"/sol1/syn/report/*.xml "hw/results/$d/syn/" 2>/dev/null || true
  cp -f hw/proj/"$d"/sol1/sim/report/*.rpt "hw/results/$d/sim/" 2>/dev/null || true
  cp -f hw/proj/"$d"_alt/sol1/syn/report/*.xml "hw/results/$d/syn_alt/" 2>/dev/null || true
  cp -f hw/proj/"$d"_alt/sol1/syn/report/*.rpt "hw/results/$d/syn_alt/" 2>/dev/null || true
  cp -f hw/proj/"$d"_csim/sol1/csim/report/*.log "hw/results/$d/" 2>/dev/null || true
  # impl_eval.tcl writes hw/results/$d/impl{,_alt} directly
done

echo "HLS runs complete. Reports in hw/results/."
