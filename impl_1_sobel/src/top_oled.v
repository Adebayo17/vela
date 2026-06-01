`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Design Name: top_oled
// Module Name: top_oled
// Project Name: Sobel Edge Detction
// Target Devices: Arty Z7-20
// Description: 
//////////////////////////////////////////////////////////////////////////////////


module top_oled(
    input  wire clk,        
    input  wire btn_rst,    // Reset Button (BTN0)
    input  wire sw_filter,  // SW0 - Activate Sobel
    input  wire sw_pattern, // SW1 - Pattern selection (0:Checkerboard, 1:Benin Flag)
    
    // Pmod OLEDrgb signals (JA connector)
    output wire oled_cs,    // Chip Select (Actif Low)
    output wire oled_sdin,  // MOSI (Data)
    output wire oled_sclk,  // Clock SPI
    output wire oled_dc,    // Data/Command
    output wire oled_res,   // Reset Screen
    output wire oled_vccen, // Power Enable
    output wire oled_pmoden // Power Enable
);

    assign oled_cs = 1'b0; 

    // --- Intermediate AXI Signals ---
    wire [15:0] pattern_data, sobel_data, mux_data;
    wire        pattern_valid, sobel_valid, mux_valid;
    wire        pattern_ready, sobel_ready, mux_ready;
    wire        pattern_sof;
    wire        pattern_eol;

    // 1. Pattern Generator
    pattern_selector u_pattern (
        .clk            (clk),
        .rst            (btn_rst),
        .pattern_sel    ({1'b0, sw_pattern}),  // 0=Checkerboard, 1=Benin Flag
        .m_axis_tdata   (pattern_data),
        .m_axis_tvalid  (pattern_valid),
        .m_axis_tready  (pattern_ready),
        .sof            (pattern_sof),
        .eol            (pattern_eol)
    );

    // 2. Sobel Filter (Processing)
    sobel_filter u_sobel (
        .clk            (clk), 
        .rst            (btn_rst),
        .s_axis_tdata   (pattern_data),
        .s_axis_tvalid  (pattern_valid),
        .s_axis_tready  (sobel_ready), // Ready from the filter to TPG
        .m_axis_tdata   (sobel_data),
        .m_axis_tvalid  (sobel_valid),
        .m_axis_tready  (mux_ready)
    );

    // 3. Multiplexer 
    assign mux_data   = sw_filter ? sobel_data  : pattern_data;
    assign mux_valid  = sw_filter ? sobel_valid : pattern_valid;
    
    wire axis_oled_ready; // Controller feedback
    assign mux_ready  = axis_oled_ready;
    assign pattern_ready = sw_filter ? sobel_ready : axis_oled_ready;


    // 4. OLED Controlled (Destination)    
    oled_controller u_ctrl (
        .clk            (clk), 
        .rst            (btn_rst),
        .s_axis_tdata   (mux_data),
        .s_axis_tvalid  (mux_valid),
        .s_axis_tready  (axis_oled_ready),
        .oled_spi_sclk  (oled_sclk), 
        .oled_spi_mosi  (oled_sdin),
        .oled_dc        (oled_dc), 
        .oled_res       (oled_res),
        .oled_vccen     (oled_vccen), 
        .oled_pmoden    (oled_pmoden)
    );

endmodule
