// sobel_filter.v — Sobel edge detection (3x3)
// Gx = [-1 0 1; -2 0 2; -1 0 1], Gy = [-1 -2 -1; 0 0 0; 1 2 1]
// Magnitude = |Gx| + |Gy| (approximation)
//
// Requires 2-line buffer for 3x3 sliding window.
//
// Part of VELA — Vision Edge Learning Accelerator
// Phase 2a: Verilog RTL image filters

module sobel_filter #(
    parameter IMG_WIDTH = 96
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [7:0]  pixel_in,   // grayscale input
    output reg         valid_out,
    output reg  [7:0]  pixel_out   // edge magnitude
);

    // TODO: Phase 2a implementation
    // - Line buffers (BRAM or shift registers)
    // - 3x3 window extraction
    // - Gx, Gy convolution
    // - Magnitude approximation

endmodule
