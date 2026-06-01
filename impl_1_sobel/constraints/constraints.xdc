## Clock Signal
set_property -dict { PACKAGE_PIN H16    IOSTANDARD LVCMOS33 } [get_ports { clk }]; #IO_L13P_T2_MRCC_35 Sch=SYSCLK
create_clock -add -name sys_clk_pin -period 8.00 -waveform {0 4} [get_ports { clk }];#set


## Reset Button
set_property -dict { PACKAGE_PIN D19    IOSTANDARD LVCMOS33 } [get_ports { btn_rst }]; #IO_L4P_T0_35 Sch=BTN0

## Sobel Switch
set_property -dict { PACKAGE_PIN M20    IOSTANDARD LVCMOS33 } [get_ports { sw_filter }];  #IO_L7N_T1_AD2N_35 Sch=SW0
set_property -dict { PACKAGE_PIN M19    IOSTANDARD LVCMOS33 } [get_ports { sw_pattern }]; #IO_L7P_T1_AD2P_35 Sch=SW1

## Pmod Header JA
# Pin 1 (Top Left) -> CS (Chip Select)
set_property -dict { PACKAGE_PIN Y18   IOSTANDARD LVCMOS33 } [get_ports { oled_cs }]; 

# Pin 2 -> MOSI (SDIN)
set_property -dict { PACKAGE_PIN Y19   IOSTANDARD LVCMOS33 } [get_ports { oled_sdin }]; 

# Pin 3 -> NC (Non Connecté - Used for MISO normally)
# (Nothing to do here)

# Pin 4 -> SCLK (Serial Clock)
set_property -dict { PACKAGE_PIN Y17   IOSTANDARD LVCMOS33 } [get_ports { oled_sclk }]; 

# Pin 7 (Bas Gauche) -> DC (Data/Command)
set_property -dict { PACKAGE_PIN U18   IOSTANDARD LVCMOS33 } [get_ports { oled_dc }]; 

# Pin 8 -> RES (Reset Screen)
set_property -dict { PACKAGE_PIN U19   IOSTANDARD LVCMOS33 } [get_ports { oled_res }]; 

# Pin 9 -> VCCEN (Voltage Control)
set_property -dict { PACKAGE_PIN W18   IOSTANDARD LVCMOS33 } [get_ports { oled_vccen }]; 

# Pin 10 -> PMODEN (Power Module Enable)
set_property -dict { PACKAGE_PIN W19   IOSTANDARD LVCMOS33 } [get_ports { oled_pmoden }];

