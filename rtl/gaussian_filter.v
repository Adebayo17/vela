// gaussian_filter.v — Gaussian blur (3x3)
// Kernel: [1 2 1; 2 4 2; 1 2 1] / 16
// Y = (sum(kernel * window) + 8) >> 4   (+8 = round to nearest: error <= 0.5 LSB)
//
// Window, line buffers and zero padding: window3x3.v
// Max sum = 16*255 + 8 = 4088 -> 12 bits, Y max = 255 (no saturation needed).
// Multiplications by 1/2/4 are wire shifts: no DSP, only adders.
//
// Latency: window3x3 latency + 1 clock cycle.
// ready_in is low during the end-of-frame flush (IMG_WIDTH+1 cycles).
//
// Bit-exact reference model: verif/models/filters_hw.py::gaussian_u8
//
// Part of VELA — Vision Edge Learning Accelerator
// Phase 2a: Verilog RTL image filters

module gaussian_filter #(
    parameter IMG_WIDTH  = 96,
    parameter IMG_HEIGHT = 64
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    output wire        ready_in,
    input  wire [7:0]  pixel_in,
    output reg         valid_out,
    output reg  [7:0]  pixel_out
);

    // ── 3x3 window ─────────────────────────────────────────────────────────

    wire        win_valid;
    wire [71:0] win;

    window3x3 #(
        .IMG_WIDTH  (IMG_WIDTH),
        .IMG_HEIGHT (IMG_HEIGHT)
    ) u_window (
        .clk        (clk),
        .rst_n      (rst_n),
        .valid_in   (valid_in),
        .ready_in   (ready_in),
        .pixel_in   (pixel_in),
        .valid_out  (win_valid),
        .window_out (win)
    );

    // w00 = MSB (top-left) ... w22 = LSB (bottom-right), zero-extended to 12 bits
    wire [11:0] w00 = {4'd0, win[71:64]}, w01 = {4'd0, win[63:56]}, w02 = {4'd0, win[55:48]};
    wire [11:0] w10 = {4'd0, win[47:40]}, w11 = {4'd0, win[39:32]}, w12 = {4'd0, win[31:24]};
    wire [11:0] w20 = {4'd0, win[23:16]}, w21 = {4'd0, win[15:8]},  w22 = {4'd0, win[7:0]};

    // ── Weighted sum: corners x1, edges x2, center x4 ──────────────────────

    wire [11:0] corners = w00 + w02 + w20 + w22;
    wire [11:0] edges   = w01 + w10 + w12 + w21;
    wire [11:0] sum     = corners + (edges << 1) + (w11 << 2) + 12'd8;

    // sum[3:0] is the fractional part, discarded after rounding
    /* verilator lint_off UNUSEDSIGNAL */
    wire [3:0] unused_frac = sum[3:0];
    /* verilator lint_on UNUSEDSIGNAL */

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            pixel_out <= 8'd0;
        end else begin
            valid_out <= win_valid;
            if (win_valid)
                pixel_out <= sum[11:4];
        end
    end

endmodule
