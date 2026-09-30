// gaussian_filter.v — Gaussian blur (3x3)
// Kernel: [1 2 1; 2 4 2; 1 2 1] / 16
//
// Requires 2-line buffer for 3x3 sliding window.
//
// Part of VELA — Vision Edge Learning Accelerator
// Phase 2a: Verilog RTL image filters

module gaussian_filter #(
    parameter IMG_WIDTH = 96
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [7:0]  pixel_in,
    output reg         valid_out,
    output reg  [7:0]  pixel_out
);

    // TODO: Phase 2a implementation
    // - Line buffers
    // - 3x3 window extraction
    // - Weighted sum with shift division (>> 4)

endmodule
