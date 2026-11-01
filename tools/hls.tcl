# hls.tcl — Vitis HLS flow of the CNN accelerator (cnn_hw)
#
# Creates the project in build/hls/cnn_proj (xc7z020clg400-1, 100 MHz,
# VELA-NFR-001) and runs the steps listed in $VELA_HLS_STEPS:
#   csim   C-simulation with hls/testbench.cpp (bit-exact vs the TN004 model)
#   synth  C synthesis: latency and resource report
#   cosim  C/RTL co-simulation with the same testbench (VELA-NFR-007)
#   export IP packaging for Vivado (Phase 3), build/hls/cnn_hw_ip.zip
# The out-of-context implementation is a separate Vivado run
# (tools/hls_impl.tcl, `make hls-impl`): Vitis would otherwise stay in memory
# while it runs Vivado (export_design -flow impl).
#
# Environment:
#   VELA_HLS_STEPS  default "csim synth"
#   VELA_HLS_N      testbench images (default 20; keep small for cosim)
#
# Run from build/ (Vitis writes logs in the current directory), see
# `make hls-csim`, `make hls-synth`, `make hls-cosim`. Expects
# build/hls/tb_*.bin (tools/gen_hls_tb_data.py) and hls/generated/cnn_weights.h.

set root  [file normalize [file join [file dirname [info script]] ..]]
set hls   [file join $root hls]
set data  [file join $root build hls]

set steps [expr {[info exists ::env(VELA_HLS_STEPS)] ? $::env(VELA_HLS_STEPS) : "csim synth"}]
set n     [expr {[info exists ::env(VELA_HLS_N)] ? $::env(VELA_HLS_N) : 20}]

set cflags "-I$hls -I[file join $hls generated]"

open_project -reset [file join $data cnn_proj]
set_top cnn_hw
add_files [file join $hls cnn_hw.cpp] -cflags $cflags
add_files -tb [file join $hls testbench.cpp] -cflags $cflags

open_solution -reset sol1 -flow_target vivado
set_part xc7z020clg400-1
create_clock -period 10 -name default

if {[lsearch $steps csim] >= 0} {
    csim_design -argv "$data $n"
}
if {[lsearch $steps synth] >= 0 || [lsearch $steps cosim] >= 0 || [lsearch $steps export] >= 0} {
    csynth_design
}
if {[lsearch $steps cosim] >= 0} {
    cosim_design -argv "$data $n" -rtl verilog
}
if {[lsearch $steps export] >= 0} {
    export_design -format ip_catalog -output [file join $data cnn_hw_ip.zip]
}
exit
