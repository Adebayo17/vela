// vela_pl.v — Top of the VELA programmable logic (Phase 3 block design)
//
// Added to the Vivado block design as a module reference (VELA-TN-007 §6.1):
//
//   ctrl (axi_gpio_ctrl ch1) ──► filter_engine ──► status (axi_gpio_ctrl ch2)
//   source BRAM port B ────────► filter_engine ──┐
//                                                ├─► display BRAM port B
//   oled_controller ◄── display_reader ◄─────────┘   (multiplexer below)
//         │
//         └──► Pmod OLEDrgb on JA (oled_cs tied low: single slave)
//
// Display port B multiplexer (VELA-TN-007 §6.4): filter_engine drives the
// port (writes) while busy, display_reader (reads) otherwise. The engine
// only writes the image zone (columns 0..SRC_WIDTH-1); the text zone
// (columns SRC_WIDTH..DISP_WIDTH-1) is written only by the ARM through port
// A (VELA-TN-007 §5.1).
//
// ctrl:   [0] start (rising edge), [3:1] mode, [7:4] reserved
// status: [0] busy, [1] done, [7:2] 0
//
// The BRAM ports carry X_INTERFACE_INFO attributes so that Vivado connects
// them as BRAM_PORT interfaces to the Block Memory Generators (attributes
// only: no Xilinx primitive, simulators ignore them).
//
// Verified by: verif/tests/test_rtl_vela_pl.py
//
// Part of VELA — Vision Edge Learning Accelerator
// Phase 3: FPGA integration

`timescale 1ns / 1ps

module vela_pl #(
    parameter       CLK_FREQ          = 100_000_000,       // FCLK_CLK0 (VELA-NFR-001)
    parameter       SPI_FREQ          = 6_250_000,
    parameter       POWER_UP_CYCLES   = CLK_FREQ / 50,     // oled_controller delays
    parameter       RESET_CYCLES      = CLK_FREQ / 10_000,
    parameter       VCC_ON_CYCLES     = CLK_FREQ / 40,
    parameter       DISPLAY_ON_CYCLES = CLK_FREQ / 10,
    parameter       SRC_WIDTH         = 64,                // image zone
    parameter       SRC_HEIGHT        = 64,
    parameter       DISP_WIDTH        = 96,                // OLED
    parameter       DISP_HEIGHT       = 64,
    parameter [7:0] THRESHOLD         = 8'd128,
    parameter       MAG_SHIFT         = 2
)(
    input  wire        clk,
    input  wire        rst_n,              // synchronous, active low
    input  wire [7:0]  ctrl,
    output wire [7:0]  status,

    // Source BRAM, port B (read only)
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_SRC CLK" *)
    (* X_INTERFACE_PARAMETER = "MASTER_TYPE BRAM_CTRL, MEM_SIZE 16384, MEM_WIDTH 32, MEM_ECC NONE, READ_LATENCY 1" *)
    output wire        src_clk,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_SRC RST" *)
    output wire        src_rst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_SRC EN" *)
    output wire        src_en,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_SRC WE" *)
    output wire [3:0]  src_we,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_SRC ADDR" *)
    output wire [31:0] src_addr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_SRC DIN" *)
    output wire [31:0] src_din,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_SRC DOUT" *)
    input  wire [31:0] src_dout,

    // Display BRAM, port B (engine writes / reader reads)
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_DISP CLK" *)
    (* X_INTERFACE_PARAMETER = "MASTER_TYPE BRAM_CTRL, MEM_SIZE 32768, MEM_WIDTH 32, MEM_ECC NONE, READ_LATENCY 1" *)
    output wire        disp_clk,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_DISP RST" *)
    output wire        disp_rst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_DISP EN" *)
    output wire        disp_en,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_DISP WE" *)
    output wire [3:0]  disp_we,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_DISP ADDR" *)
    output wire [31:0] disp_addr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_DISP DIN" *)
    output wire [31:0] disp_din,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_DISP DOUT" *)
    input  wire [31:0] disp_dout,

    // Pmod OLEDrgb (connector JA)
    output wire        oled_cs,
    output wire        oled_sclk,
    output wire        oled_mosi,
    output wire        oled_dc,
    output wire        oled_res,
    output wire        oled_vccen,
    output wire        oled_pmoden
);

    // ── Filter engine ──────────────────────────────────────────────────────

    wire        eng_busy, eng_done, eng_we;
    wire [31:0] eng_addr, eng_din;

    filter_engine #(
        .SRC_WIDTH  (SRC_WIDTH),
        .SRC_HEIGHT (SRC_HEIGHT),
        .DISP_WIDTH (DISP_WIDTH),
        .THRESHOLD  (THRESHOLD),
        .MAG_SHIFT  (MAG_SHIFT)
    ) u_filter_engine (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (ctrl[0]),
        .mode       (ctrl[3:1]),
        .busy       (eng_busy),
        .done       (eng_done),
        .src_addr   (src_addr),
        .src_en     (src_en),
        .src_dout   (src_dout),
        .dst_addr   (eng_addr),
        .dst_we     (eng_we),
        .dst_din    (eng_din)
    );

    assign status  = {6'd0, eng_done, eng_busy};

    assign src_clk = clk;
    assign src_rst = 1'b0;
    assign src_we  = 4'd0;
    assign src_din = 32'd0;

    // ── Display reader ─────────────────────────────────────────────────────

    wire        rd_en, grant;
    wire [31:0] rd_addr;
    wire [15:0] pix_tdata;
    wire        pix_tvalid, pix_tready;

    assign grant = ~eng_busy;

    display_reader #(
        .N_PIXELS       (DISP_WIDTH * DISP_HEIGHT)
    ) u_display_reader (
        .clk            (clk),
        .rst_n          (rst_n),
        .grant          (grant),
        .mem_addr       (rd_addr),
        .mem_en         (rd_en),
        .mem_dout       (disp_dout),
        .m_axis_tdata   (pix_tdata),
        .m_axis_tvalid  (pix_tvalid),
        .m_axis_tready  (pix_tready)
    );

    // ── Display port B multiplexer: engine while busy, reader otherwise ────

    assign disp_clk  = clk;
    assign disp_rst  = 1'b0;
    assign disp_en   = eng_busy ? eng_we : rd_en;
    assign disp_we   = (eng_busy & eng_we) ? 4'hF : 4'h0;
    assign disp_addr = eng_busy ? eng_addr : rd_addr;
    assign disp_din  = eng_din;

    // ── OLED display path ──────────────────────────────────────────────────

    oled_controller #(
        .CLK_FREQ          (CLK_FREQ),
        .SPI_FREQ          (SPI_FREQ),
        .POWER_UP_CYCLES   (POWER_UP_CYCLES),
        .RESET_CYCLES      (RESET_CYCLES),
        .VCC_ON_CYCLES     (VCC_ON_CYCLES),
        .DISPLAY_ON_CYCLES (DISPLAY_ON_CYCLES)
    ) u_oled_controller (
        .clk            (clk),
        .rst_n          (rst_n),
        .s_axis_tdata   (pix_tdata),
        .s_axis_tvalid  (pix_tvalid),
        .s_axis_tready  (pix_tready),
        .oled_spi_sclk  (oled_sclk),
        .oled_spi_mosi  (oled_mosi),
        .oled_dc        (oled_dc),
        .oled_res       (oled_res),
        .oled_vccen     (oled_vccen),
        .oled_pmoden    (oled_pmoden)
    );

    assign oled_cs = 1'b0;

    wire unused = &{1'b0, ctrl[7:4]};

endmodule
