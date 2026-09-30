// threshold.v — Binary threshold filter
// Output = (pixel > threshold) ? 255 : 0
//
// Part of VELA — Vision Edge Learning Accelerator
// Phase 2a: Verilog RTL image filters

module threshold #(
    parameter [7:0] THRESHOLD = 8'd128
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [7:0]  pixel_in,
    output reg         valid_out,
    output reg  [7:0]  pixel_out
);

    // TODO: Phase 2a implementation

endmodule
