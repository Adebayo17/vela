## vela_arty_z7_20.xdc — VELA Phase 3 pin constraints (Digilent Arty Z7-20)
##
## Pins from the Digilent board file (arty-z7-20 1.1, part0_pins.xml); the
## OLED pins were verified on this board by the earlier Sobel project
## (VELA-TN-007 §8). The PL clock is FCLK_CLK0 (100 MHz), constrained by the
## processing_system7 IP itself.

## ── Pmod OLEDrgb on connector JA ────────────────────────────────────────────
set_property -dict { PACKAGE_PIN Y18 IOSTANDARD LVCMOS33 } [get_ports oled_cs]      ;# JA1
set_property -dict { PACKAGE_PIN Y19 IOSTANDARD LVCMOS33 } [get_ports oled_mosi]    ;# JA2
set_property -dict { PACKAGE_PIN Y17 IOSTANDARD LVCMOS33 } [get_ports oled_sclk]    ;# JA4
set_property -dict { PACKAGE_PIN U18 IOSTANDARD LVCMOS33 } [get_ports oled_dc]      ;# JA7
set_property -dict { PACKAGE_PIN U19 IOSTANDARD LVCMOS33 } [get_ports oled_res]     ;# JA8
set_property -dict { PACKAGE_PIN W18 IOSTANDARD LVCMOS33 } [get_ports oled_vccen]   ;# JA9
set_property -dict { PACKAGE_PIN W19 IOSTANDARD LVCMOS33 } [get_ports oled_pmoden]  ;# JA10

## ── Buttons, switches, LEDs (read / written by the ARM through axi_gpio_io) ──
set_property -dict { PACKAGE_PIN D19 IOSTANDARD LVCMOS33 } [get_ports {btn[0]}]
set_property -dict { PACKAGE_PIN D20 IOSTANDARD LVCMOS33 } [get_ports {btn[1]}]
set_property -dict { PACKAGE_PIN L20 IOSTANDARD LVCMOS33 } [get_ports {btn[2]}]
set_property -dict { PACKAGE_PIN L19 IOSTANDARD LVCMOS33 } [get_ports {btn[3]}]
set_property -dict { PACKAGE_PIN M20 IOSTANDARD LVCMOS33 } [get_ports {sw[0]}]
set_property -dict { PACKAGE_PIN M19 IOSTANDARD LVCMOS33 } [get_ports {sw[1]}]
set_property -dict { PACKAGE_PIN R14 IOSTANDARD LVCMOS33 } [get_ports {led[0]}]
set_property -dict { PACKAGE_PIN P14 IOSTANDARD LVCMOS33 } [get_ports {led[1]}]
set_property -dict { PACKAGE_PIN N16 IOSTANDARD LVCMOS33 } [get_ports {led[2]}]
set_property -dict { PACKAGE_PIN M14 IOSTANDARD LVCMOS33 } [get_ports {led[3]}]

## ── Timing exceptions ───────────────────────────────────────────────────────
## Slow or asynchronous I/O: the OLED signals change at most every 8 cycles
## (6.25 MHz SPI) and the SSD1331 samples them with its own timing; buttons and
## switches are asynchronous, sampled and debounced by the software (10 ms);
## LEDs are static. No input / output delay is meaningful for them.
set_false_path -to   [get_ports {oled_cs oled_mosi oled_sclk oled_dc oled_res oled_vccen oled_pmoden}]
set_false_path -from [get_ports {btn[*] sw[*]}]
set_false_path -to   [get_ports {led[*]}]
