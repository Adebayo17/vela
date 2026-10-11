# synth.tcl — Out-of-context synthesis + place & route of the VELA RTL modules
#
# Vivado non-project mode, one module at a time, default parameters
# (IMG_WIDTH = 96, IMG_HEIGHT = 64). For each module, writes to
# <out_dir>/<module>/ :
#   utilization.rpt   report_utilization after routing
#   timing.rpt        report_timing_summary after routing
#   summary.txt       "<module> <WNS> <WHS> <WHS_IO>" (ns)
#
# Usage (from the repository root, see `make synth`):
#   vivado -mode batch -nojournal -nolog -source tools/synth.tcl \
#          -tclargs <out_dir> [module ...]
#
# Timing model: single 100 MHz clock (VELA-NFR-001). Inputs and outputs get
# a 0 ns delay relative to clk, i.e. they are assumed to come from / go to
# registers outside the module: port-to-register logic is timed too.
#
# Hold: WHS is measured between internal registers only. Port-to-register
# hold (WHS_IO) is not meaningful out of context: with a 0 ns input delay the
# tool assumes the input changes exactly on the clock edge, while the clock
# reaches the registers later through its routing. It is reported for
# information and checked in the full design (Phase 3).

set part   xc7z020clg400-1
set period 10.000

set root    [file normalize [file join [file dirname [info script]] ..]]
set out_dir [file normalize [lindex $argv 0]]
set only    [lrange $argv 1 end]

# top module -> source files in rtl/
set modules {
    grayscale        {grayscale.v}
    threshold        {threshold.v}
    window3x3        {window3x3.v}
    gaussian_filter  {gaussian_filter.v window3x3.v}
    sobel_filter     {sobel_filter.v window3x3.v}
    median_filter    {median_filter.v window3x3.v}
    spi_master       {spi_master.v}
    oled_controller  {oled_controller.v spi_master.v}
}

# Slack of the first path of a list, or NA when there is no such path
# (e.g. grayscale has no register-to-register path)
proc slack {paths} {
    if {[llength $paths] == 0} { return NA }
    return [get_property SLACK [lindex $paths 0]]
}

foreach {top srcs} $modules {
    if {[llength $only] && [lsearch -exact $only $top] < 0} {
        continue
    }
    puts "\n==================== $top ===================="
    set dir [file join $out_dir $top]
    file mkdir $dir

    close_project -quiet
    create_project -in_memory -part $part
    foreach f $srcs {
        read_verilog [file join $root rtl $f]
    }

    synth_design -top $top -part $part -mode out_of_context

    create_clock -period $period -name clk [get_ports clk]
    set_input_delay  -clock clk 0 [get_ports -filter {DIRECTION == IN && NAME != clk}]
    set_output_delay -clock clk 0 [get_ports -filter {DIRECTION == OUT}]

    opt_design
    place_design
    route_design

    report_utilization    -file [file join $dir utilization.rpt]
    report_timing_summary -file [file join $dir timing.rpt]

    set wns    [slack [get_timing_paths -setup -max_paths 1]]
    set regs   [all_registers]
    set whs    [slack [get_timing_paths -hold -max_paths 1 -from $regs -to $regs]]
    set whs_io [slack [get_timing_paths -hold -max_paths 1]]
    set fh [open [file join $dir summary.txt] w]
    puts $fh "$top $wns $whs $whs_io"
    close $fh
    puts "SUMMARY $top WNS=$wns WHS=$whs WHS_IO=$whs_io"
}
