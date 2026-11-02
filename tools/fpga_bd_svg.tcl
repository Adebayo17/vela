# fpga_bd_svg.tcl — Export the block design schematic to docs/diagrams/vela_bd.svg
#
# write_bd_layout only works with the GUI: the Makefile runs this script in
# Vivado GUI mode on a virtual display (xvfb-run), after make fpga-bd.
#   vivado -mode gui -source ../tools/fpga_bd_svg.tcl -tclargs <out_dir>

set root    [file normalize [file join [file dirname [info script]] ..]]
set out_dir [file normalize [lindex $argv 0]]

open_project [file join $out_dir vela vela.xpr]
open_bd_design [get_files vela_bd.bd]
set svg [file join $root docs diagrams vela_bd.svg]
write_bd_layout -force -format svg $svg

# Explicit white background (Vivado leaves it transparent: unreadable with a
# dark GitHub theme), inserted right after the <desc> element
set fh [open $svg r]; set text [read $fh]; close $fh
set text [regsub {</desc>} $text {</desc>
<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>}]
set fh [open $svg w]; puts -nonewline $fh $text; close $fh
puts "STAGE svg done"
exit
