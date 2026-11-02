// display_reader.v — Display memory -> AXI-Stream pixels for oled_controller
//
// Reads the display memory words 0, 1, ..., N_PIXELS-1, 0, 1, ... for ever
// and sends the RGB565 pixel of each word (bits 15:0) over AXI-Stream
// (VELA-TN-007 §6.3):
//
//   - nothing is read before the first m_axis_tready: the first pixel sent
//     is word 0 read at that time, matching the SSD1331 RAM pointer set to
//     (0, 0) by the init sequence, so pixel n always lands at
//     (n mod 96, n / 96) on the 96 x 64 display;
//   - one-word buffer: m_axis_tvalid / m_axis_tdata are held until
//     m_axis_tready (AXI-Stream rule), no pixel is lost or duplicated;
//   - a read is issued only when `grant` is high (the display port of
//     vela_pl belongs to filter_engine while it runs).
//
// Memory: BRAM controller conventions (byte address = word index * 4), read
// latency 1 cycle. A read issued just before `grant` falls still completes:
// filter_engine writes at the earliest two cycles after taking the port.
//
// Throughput: one pixel every 3 cycles at most; oled_controller takes one
// every ~272 cycles (2 bytes x 17 SCLK half periods of 8 cycles).
//
// Verified by: verif/tests/test_rtl_display_reader.py
//
// Part of VELA — Vision Edge Learning Accelerator
// Phase 3: FPGA integration

`timescale 1ns / 1ps

module display_reader #(
    parameter N_PIXELS = 6144              // 96 x 64
)(
    input  wire        clk,
    input  wire        rst_n,              // synchronous, active low
    input  wire        grant,              // display port available
    // Display memory, read only
    output wire [31:0] mem_addr,           // byte address
    output wire        mem_en,
    input  wire [31:0] mem_dout,           // 0x0000 + RGB565, 1 cycle after mem_en
    // AXI-Stream master (pixels to oled_controller)
    output reg  [15:0] m_axis_tdata,
    output reg         m_axis_tvalid,
    input  wire        m_axis_tready
);

    localparam [15:0] LAST = N_PIXELS - 1;

    reg         started;        // oled_controller has asked for a pixel once
    reg         rd_pending;     // read issued last cycle, data on mem_dout
    reg  [15:0] idx;            // next word to read

    assign mem_en   = started & grant & ~m_axis_tvalid & ~rd_pending;
    assign mem_addr = {14'd0, idx, 2'b00};

    always @(posedge clk) begin
        if (!rst_n) begin
            started       <= 1'b0;
            rd_pending    <= 1'b0;
            idx           <= 16'd0;
            m_axis_tdata  <= 16'd0;
            m_axis_tvalid <= 1'b0;
        end else begin
            if (m_axis_tready)
                started <= 1'b1;

            rd_pending <= mem_en;
            if (mem_en)
                idx <= (idx == LAST) ? 16'd0 : idx + 16'd1;

            // Buffer: filled one cycle after the read, emptied by the handshake
            // (both cannot happen together: a read needs an empty buffer)
            if (rd_pending) begin
                m_axis_tdata  <= mem_dout[15:0];
                m_axis_tvalid <= 1'b1;
            end else if (m_axis_tvalid && m_axis_tready) begin
                m_axis_tvalid <= 1'b0;
            end
        end
    end

    wire unused = &{1'b0, mem_dout[31:16]};

endmodule
