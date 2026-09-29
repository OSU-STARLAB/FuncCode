# Vivado HLS 2019.1 driver for the three FuncCode-KAN designs.
#
#   vivado_hls -f hw/tcl/run_hls.tcl -tclargs <design> <mode> [cosim_n]
#
#   design: fp32 | lsq_w4a4 | funccode_w4a4
#   mode:
#     csim   - C simulation of all GOLDEN_N (=1000) vectors  (rung L2 gate)
#     synth  - csynth only (quick iteration)
#     impl   - csynth + cosim on [cosim_n] vectors (default 50, rung L3).
#              Post-implementation numbers come from hw/tcl/impl_eval.tcl
#              (plain Vivado on the generated Verilog) because
#              export_design's IP packager hits the Y2K22 date bug in
#              2019.1 ("Failed to generate IP").
#
# Same part / clock / directives for every design (docs/HW_FAIRNESS.md).
# One solution per design named sol1; reports archived by
# scripts/hw/12_run_hls.sh into hw/results/.

# vivado_hls 2019.1 passes "-f <script> <tclargs...>" in $argv
set _userargs $::argv
if {[lindex $_userargs 0] == "-f"} {
    set _userargs [lrange $_userargs 2 end]
}
set design  [lindex $_userargs 0]
set mode    [lindex $_userargs 1]
set cosim_n [lindex $_userargs 2]
set altpart [lindex $_userargs 3]   ;# optional part override (e.g. the
                                    ;# licensed xczu7ev for the secondary
                                    ;# D2-vs-D3 post-impl comparison)
if {$cosim_n == ""} { set cosim_n 50 }

set _designs {fp32 lsq_w4a4 funccode_w4a4
              gram_fp32 gram_lsq_w4a4 gram_funccode_w4a4
              kagnconv_fp32 kagnconv_lsq_w4a4 kagnconv_funccode_w4a4}
if {[lsearch $_designs $design] < 0} {
    puts "ERROR: unknown design '$design'"
    exit 1
}
if {[lsearch {csim synth impl} $mode] < 0} {
    puts "ERROR: unknown mode '$mode'"
    exit 1
}

set script_dir [file dirname [file normalize [info script]]]
set root       [file normalize "$script_dir/../.."]

set part    "xczu9eg-ffvb1156-2-e"  ;# mirrors funcodekan/hw/hw_spec.py
set period  6.67
set projsuffix ""
if {$altpart != ""} {
    set part $altpart
    set projsuffix "_alt"
}                    ;# 150 MHz

set common  "$root/hw/hls/common"
set golden  "$root/hw/golden/$design"
set src     "$root/hw/hls/$design/top.cpp"
set tb      "$root/hw/tb/tb_main.cpp"
set cflags  "-I$common -I$golden"
if {[string match "*fp32" $design]} {
    set tbflags "$cflags -DDESIGN_FP32"
} else {
    set tbflags "$cflags"
}

if {![file exists "$golden/params.h"]} {
    puts "ERROR: $golden/params.h missing - run scripts/hw/11_export_golden.sh first"
    exit 1
}

file mkdir "$root/hw/proj"
cd "$root/hw/proj"

if {$mode == "csim"} {
    open_project -reset "${design}_csim"
    set_top kan_top
    add_files $src -cflags $cflags
    add_files -tb $tb -cflags $tbflags
    open_solution "sol1"
    set_part $part
    create_clock -period $period
    csim_design -clean
} else {
    open_project -reset "${design}${projsuffix}"
    set_top kan_top
    add_files $src -cflags $cflags
    add_files -tb $tb -cflags "$tbflags -DTB_N=$cosim_n"
    open_solution "sol1"
    set_part $part
    create_clock -period $period
    csynth_design
    if {$mode == "impl"} {
        cosim_design -rtl verilog
    }
}

exit 0
