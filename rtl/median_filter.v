// median_filter.v — Median filter (3x3)
// Sorts 9 pixel values, outputs the middle one.
// Uses a sorting network for deterministic latency.
//
// Requires 2-line buffer for 3x3 sliding window.
//
// Part of VELA — Vision Edge Learning Accelerator
// Phase 2a: Verilog RTL image filters

module median_filter #(
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
    // - 9-input sorting network (compare-and-swap)
    // - Output median (element [4] after sort)

endmodule
