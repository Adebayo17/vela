# fpga.tcl — VELA Phase 3: Vivado project, block design, bitstream and XSA
#
# Builds the system of VELA-TN-007 from scratch (nothing is kept in git but
# this script, the RTL and the constraints):
#
#   Zynq PS (Arty Z7-20 preset, FCLK_CLK0 = 100 MHz) ── M_AXI_GP0 ── SmartConnect
#       ├── cnn_hw          (HLS IP, TN005)               0x4000_0000  64 KB
#       ├── axi_gpio_io     (BTN/SW in, LED out)          0x4120_0000  64 KB
#       ├── axi_gpio_ctrl   (vela_pl ctrl / status)       0x4121_0000  64 KB
#       ├── axi_bram_ctrl_src  ── bram_src  (port B: vela_pl)  0x4200_0000  16 KB
#       └── axi_bram_ctrl_disp ── bram_disp (port B: vela_pl)  0x4202_0000  32 KB
#   vela_pl (RTL module reference) ── Pmod OLEDrgb on JA
#
# Usage (from build/, see the Makefile targets fpga-bd and fpga):
#   vivado -mode batch -source ../tools/fpga.tcl -tclargs <out_dir> bd
#   vivado -mode batch -source ../tools/fpga.tcl -tclargs <out_dir> all
#     bd  : project + block design + validation + docs/diagrams/vela_bd.svg
#     all : bd + synthesis + implementation + bitstream + XSA + reports
#
# The CNN IP must be extracted in build/ip/cnn_hw (make fpga-ip).

set root    [file normalize [file join [file dirname [info script]] ..]]
set out_dir [file normalize [lindex $argv 0]]
set stage   [lindex $argv 1]
set part    xc7z020clg400-1
set board   digilentinc.com:arty-z7-20:part0:1.1
set proj    vela
set bd_name vela_bd

# Board files: the Digilent boards installed from the Vivado board store are
# not searched in batch mode; VELA_BOARD_REPO overrides the default location
if {[info exists ::env(VELA_BOARD_REPO)]} {
    set board_repo $::env(VELA_BOARD_REPO)
} else {
    set board_repo [file normalize \
        ~/.Xilinx/Vivado/[version -short]/xhub/board_store/xilinx_board_store/XilinxBoardStore/Vivado/[version -short]/boards]
}
set_param board.repoPaths [list $board_repo]

file mkdir $out_dir
create_project $proj [file join $out_dir $proj] -part $part -force
set_property board_part $board [current_project]
set_property target_language Verilog [current_project]
set_property ip_repo_paths [file join $root build ip] [current_project]
update_ip_catalog

# RTL of the programmable logic (vela_pl and everything it instantiates)
foreach f {vela_pl filter_engine display_reader oled_controller spi_master grayscale
           threshold sobel_filter gaussian_filter median_filter window3x3} {
    add_files -norecurse [file join $root rtl $f.v]
}
add_files -fileset constrs_1 -norecurse [file join $root constraints vela_arty_z7_20.xdc]
update_compile_order -fileset sources_1

# ─── Block design ────────────────────────────────────────────────────────────

create_bd_design $bd_name

# Processing system: board preset (DDR, MIO, SD0, UART0), GP0 master, 100 MHz
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} $ps
set_property -dict [list \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_EN_RST0_PORT {1} \
] $ps

set rst  [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_100M]
set axi  [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {5}] $axi

set cnn  [create_bd_cell -type ip -vlnv xilinx.com:hls:cnn_hw:1.0 cnn_hw]

set gio  [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_io]
set_property -dict [list \
    CONFIG.C_IS_DUAL {1} \
    CONFIG.C_GPIO_WIDTH {6}  CONFIG.C_ALL_INPUTS {1} \
    CONFIG.C_GPIO2_WIDTH {4} CONFIG.C_ALL_OUTPUTS_2 {1} \
] $gio

set gctl [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_ctrl]
set_property -dict [list \
    CONFIG.C_IS_DUAL {1} \
    CONFIG.C_GPIO_WIDTH {8}  CONFIG.C_ALL_OUTPUTS {1} \
    CONFIG.C_GPIO2_WIDTH {8} CONFIG.C_ALL_INPUTS_2 {1} \
] $gctl

# Image memories: AXI BRAM controller on port A (ARM), vela_pl on port B
foreach {name} {src disp} {
    set ctl [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 axi_bram_ctrl_$name]
    set_property -dict [list CONFIG.PROTOCOL {AXI4LITE} CONFIG.SINGLE_PORT_BRAM {1} \
                             CONFIG.DATA_WIDTH {32}] $ctl
    set mem [create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 bram_$name]
    set_property -dict [list CONFIG.Memory_Type {True_Dual_Port_RAM}] $mem
}

# RTL of the PL (VELA-TN-007 §6), default parameters (real OLED delays)
set pl [create_bd_cell -type module -reference vela_pl vela_pl_0]

# BTN + SW concatenated on axi_gpio_io channel 1: [3:0] BTN, [5:4] SW (TN007 §7)
set cat [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 btn_sw_concat]
set_property -dict [list CONFIG.NUM_PORTS {2} CONFIG.IN0_WIDTH {4} CONFIG.IN1_WIDTH {2}] $cat

# ─── Connections ─────────────────────────────────────────────────────────────

set clk [get_bd_pins ps7/FCLK_CLK0]
connect_bd_net $clk [get_bd_pins ps7/M_AXI_GP0_ACLK] [get_bd_pins rst_ps7_100M/slowest_sync_clk] \
    [get_bd_pins axi_smc/aclk] [get_bd_pins cnn_hw/ap_clk] \
    [get_bd_pins axi_gpio_io/s_axi_aclk] [get_bd_pins axi_gpio_ctrl/s_axi_aclk] \
    [get_bd_pins axi_bram_ctrl_src/s_axi_aclk] [get_bd_pins axi_bram_ctrl_disp/s_axi_aclk] \
    [get_bd_pins vela_pl_0/clk]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins rst_ps7_100M/ext_reset_in]
connect_bd_net [get_bd_pins rst_ps7_100M/interconnect_aresetn] [get_bd_pins axi_smc/aresetn]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins cnn_hw/ap_rst_n] \
    [get_bd_pins axi_gpio_io/s_axi_aresetn] [get_bd_pins axi_gpio_ctrl/s_axi_aresetn] \
    [get_bd_pins axi_bram_ctrl_src/s_axi_aresetn] [get_bd_pins axi_bram_ctrl_disp/s_axi_aresetn] \
    [get_bd_pins vela_pl_0/rst_n]

connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0] [get_bd_intf_pins axi_smc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc/M00_AXI] [get_bd_intf_pins cnn_hw/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins axi_smc/M01_AXI] [get_bd_intf_pins axi_gpio_io/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc/M02_AXI] [get_bd_intf_pins axi_gpio_ctrl/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc/M03_AXI] [get_bd_intf_pins axi_bram_ctrl_src/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc/M04_AXI] [get_bd_intf_pins axi_bram_ctrl_disp/S_AXI]

foreach name {src disp} {
    connect_bd_intf_net [get_bd_intf_pins axi_bram_ctrl_$name/BRAM_PORTA] \
                        [get_bd_intf_pins bram_$name/BRAM_PORTA]
}
connect_bd_intf_net [get_bd_intf_pins vela_pl_0/BRAM_SRC]  [get_bd_intf_pins bram_src/BRAM_PORTB]
connect_bd_intf_net [get_bd_intf_pins vela_pl_0/BRAM_DISP] [get_bd_intf_pins bram_disp/BRAM_PORTB]

connect_bd_net [get_bd_pins axi_gpio_ctrl/gpio_io_o]  [get_bd_pins vela_pl_0/ctrl]
connect_bd_net [get_bd_pins vela_pl_0/status]         [get_bd_pins axi_gpio_ctrl/gpio2_io_i]

# External ports (pins in constraints/vela_arty_z7_20.xdc)
create_bd_port -dir I -from 3 -to 0 btn
create_bd_port -dir I -from 1 -to 0 sw
create_bd_port -dir O -from 3 -to 0 led
connect_bd_net [get_bd_ports btn] [get_bd_pins btn_sw_concat/In0]
connect_bd_net [get_bd_ports sw]  [get_bd_pins btn_sw_concat/In1]
connect_bd_net [get_bd_pins btn_sw_concat/dout] [get_bd_pins axi_gpio_io/gpio_io_i]
connect_bd_net [get_bd_pins axi_gpio_io/gpio2_io_o] [get_bd_ports led]
foreach p {oled_cs oled_sclk oled_mosi oled_dc oled_res oled_vccen oled_pmoden} {
    create_bd_port -dir O $p
    connect_bd_net [get_bd_pins vela_pl_0/$p] [get_bd_ports $p]
}

# ─── Address map (VELA-TN-007 §4) ────────────────────────────────────────────

set space [get_bd_addr_spaces ps7/Data]
assign_bd_address -target_address_space $space -offset 0x40000000 -range 64K \
    [get_bd_addr_segs cnn_hw/s_axi_control/Reg]
assign_bd_address -target_address_space $space -offset 0x41200000 -range 64K \
    [get_bd_addr_segs axi_gpio_io/S_AXI/Reg]
assign_bd_address -target_address_space $space -offset 0x41210000 -range 64K \
    [get_bd_addr_segs axi_gpio_ctrl/S_AXI/Reg]
assign_bd_address -target_address_space $space -offset 0x42000000 -range 16K \
    [get_bd_addr_segs axi_bram_ctrl_src/S_AXI/Mem0]
assign_bd_address -target_address_space $space -offset 0x42020000 -range 32K \
    [get_bd_addr_segs axi_bram_ctrl_disp/S_AXI/Mem0]

# Port A in READ_FIRST: display_reader (port B) reading a word the ARM writes
# gets the old or the new value, never a corrupted one (TN007 §5.1)
foreach name {src disp} {
    set_property -dict [list CONFIG.Operating_Mode_A {READ_FIRST}] [get_bd_cells bram_$name]
}

regenerate_bd_layout
validate_bd_design
save_bd_design

# Facts checked against the specification
set fclk [get_property CONFIG.PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ $ps]
puts "CHECK FCLK_CLK0 = $fclk MHz"
foreach name {src disp} {
    set mem [get_bd_cells bram_$name]
    puts "CHECK bram_$name depth=[get_property CONFIG.Write_Depth_A $mem]\
 mode_a=[get_property CONFIG.Operating_Mode_A $mem]\
 mode_b=[get_property CONFIG.Operating_Mode_B $mem]\
 reg_b=[get_property CONFIG.Register_PortB_Output_of_Memory_Primitives $mem]"
}
foreach seg [get_bd_addr_segs -of_objects $space] {
    puts "CHECK address [get_property NAME $seg] [get_property OFFSET $seg] [get_property RANGE $seg]"
}
puts "CHECK UART0=[get_property CONFIG.PCW_UART0_PERIPHERAL_ENABLE $ps]\
 SD0=[get_property CONFIG.PCW_SD0_PERIPHERAL_ENABLE $ps]"

# The block design schematic (docs/diagrams/vela_bd.svg) needs the GUI:
# tools/fpga_bd_svg.tcl, run by make fpga-bd on a virtual display

# Top-level wrapper; IPs synthesized with the design (one Vivado process:
# lower memory peak than one out-of-context run per IP)
set bd_file [get_files $bd_name.bd]
set_property synth_checkpoint_mode None $bd_file
generate_target all $bd_file
set wrapper [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper
set_property top ${bd_name}_wrapper [current_fileset]
update_compile_order -fileset sources_1

if {$stage ne "all"} {
    puts "STAGE bd done"
    exit 0
}

# ─── Synthesis, implementation, bitstream, XSA ───────────────────────────────

launch_runs synth_1 -jobs 1
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} { error "synthesis failed" }

launch_runs impl_1 -to_step write_bitstream -jobs 1
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "implementation failed" }

open_run impl_1
set rpt [file join $out_dir reports]
file mkdir $rpt
report_timing_summary -file [file join $rpt timing_summary.rpt]
report_utilization    -file [file join $rpt utilization.rpt]
report_utilization -hierarchical -hierarchical_depth 2 -file [file join $rpt utilization_hier.rpt]

set wns [get_property SLACK [lindex [get_timing_paths -max_paths 1 -setup] 0]]
set whs [get_property SLACK [lindex [get_timing_paths -max_paths 1 -hold] 0]]
puts "SUMMARY WNS=$wns WHS=$whs"

write_hw_platform -fixed -include_bit -force [file join $out_dir vela.xsa]
file copy -force [glob [file join $out_dir $proj $proj.runs impl_1 *.bit]] \
    [file join $out_dir vela.bit]

# VELA-NFR-011: complete design at 100 MHz, WNS >= 0 and WHS >= 0
if {$wns < 0 || $whs < 0} {
    puts "CHECK VELA-NFR-011 FAIL (WNS=$wns WHS=$whs)"
    exit 1
}
puts "CHECK VELA-NFR-011 PASS (WNS=$wns WHS=$whs)"
puts "STAGE all done"
