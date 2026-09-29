# Post-implementation evaluation WITHOUT export_design (the Vivado HLS
# 2019.1 IP packager hits the Y2K22 date-overflow bug on modern systems:
# "ERROR: [IMPL 213-28] Failed to generate IP"). Instead, run Vivado
# synthesis + place + route out-of-context directly on the csynth-generated
# Verilog — same netlist, honest post-implementation numbers.
#
#   vivado -mode batch -source hw/tcl/impl_eval.tcl -tclargs <design>
#
# Emits hw/results/<design>/impl/{utilization_route.rpt, timing_route.rpt}.

set design  [lindex $argv 0]
set altpart [lindex $argv 1]   ;# optional: licensed part for the secondary
                               ;# D2-vs-D3 post-impl run (e.g. xczu7ev...)
set _designs {fp32 lsq_w4a4 funccode_w4a4
              gram_fp32 gram_lsq_w4a4 gram_funccode_w4a4
              kagnconv_fp32 kagnconv_lsq_w4a4 kagnconv_funccode_w4a4}
if {[lsearch $_designs $design] < 0} {
    puts "ERROR: unknown design '$design'"
    exit 1
}

set script_dir [file dirname [file normalize [info script]]]
set root       [file normalize "$script_dir/../.."]
set part       "xczu9eg-ffvb1156-2-e"   ;# mirrors funcodekan/hw/hw_spec.py
set vdir       "$root/hw/proj/$design/sol1/syn/verilog"
set outdir     "$root/hw/results/$design/impl"
if {$altpart != ""} {
    set part $altpart
    set vdir "$root/hw/proj/${design}_alt/sol1/syn/verilog"
    set outdir "$root/hw/results/$design/impl_alt"
}
set period     6.67

file mkdir $outdir
# ROM init .dat files are referenced relative to the netlist directory
cd $vdir

# On-disk throwaway project: fp32 designs ship *_ip.tcl scripts that
# create the floating-point-operator IP cores via create_ip, which needs a
# managed project (in-memory mode fails with "module ... not found").
set projdir "$root/hw/proj/_impl_eval_tmp"
file delete -force $projdir
create_project -force impl_eval_tmp $projdir -part $part
read_verilog [glob "$vdir/*.v"]
foreach ipscript [glob -nocomplain "$vdir/*_ip.tcl"] {
    source $ipscript
}
synth_design -top kan_top -mode out_of_context
create_clock -period $period -name ap_clk [get_ports ap_clk]
opt_design
place_design
route_design
report_utilization    -file "$outdir/utilization_route.rpt"
report_timing_summary -file "$outdir/timing_route.rpt"
close_project
file delete -force $projdir
puts "IMPL_EVAL_DONE $design"
exit 0
