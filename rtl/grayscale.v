// grayscale.v — RGB to Grayscale conversion
// Luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
// Fixed-point approximation: Y = (77*R + 150*G + 29*B + 128) >> 8
// (+128 = round to nearest: max error vs float model < 1 LSB)
//
// Latency: 1 clock cycle (registered output).
// Max sum = 256*255 + 128 = 65408 -> fits in 16 bits, Y max = 255 (no saturation needed).
//
// Bit-exact reference model: verif/models/filters_hw.py::grayscale_u8
//
// Part of VELA — Vision Edge Learning Accelerator
// Phase 2a: Verilog RTL image filters

`timescale 1ns / 1ps

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

    localparam [15:0] COEF_R = 16'd77;
    localparam [15:0] COEF_G = 16'd150;
    localparam [15:0] COEF_B = 16'd29;
    localparam [15:0] ROUND  = 16'd128;

    // Constant multiplications: synthesized as shift-and-add (no DSP needed)
    wire [15:0] sum = COEF_R * {8'd0, r_in} + COEF_G * {8'd0, g_in}
                    + COEF_B * {8'd0, b_in} + ROUND;

    // sum[7:0] is the fractional part, discarded after rounding
    /* verilator lint_off UNUSEDSIGNAL */
    wire [7:0] unused_frac = sum[7:0];
    /* verilator lint_on UNUSEDSIGNAL */

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            y_out     <= 8'd0;
        end else begin
            valid_out <= valid_in;
            if (valid_in)
                y_out <= sum[15:8];
        end
    end

endmodule
