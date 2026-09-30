// grayscale.v — RGB to Grayscale conversion
// Luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
// Fixed-point approximation: Y = (77*R + 150*G + 29*B) >> 8
//
// Part of VELA — Vision Edge Learning Accelerator
// Phase 2a: Verilog RTL image filters

module grayscale (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [7:0]  r_in,
    input  wire [7:0]  g_in,
    input  wire [7:0]  b_in,
    output reg         valid_out,
    output reg  [7:0]  y_out
);

    // TODO: Phase 2a implementation

endmodule
