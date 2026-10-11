// sobel_filter.v — Sobel edge detection (3x3)
// Gx = [-1 0 1; -2 0 2; -1 0 1], Gy = [-1 -2 -1; 0 0 0; 1 2 1]
// Magnitude = |Gx| + |Gy| (approximation of sqrt(Gx^2 + Gy^2))
// Y = min(Magnitude >> MAG_SHIFT, 255)
//
// Dynamic range: |Gx|, |Gy| <= 1020 (11-bit signed), |Gx| + |Gy| <= 1530
// (11-bit unsigned). The 8-bit output saturates at 255; MAG_SHIFT divides
// the magnitude by 2^MAG_SHIFT first if the result looks too saturated.
// With MAG_SHIFT = 0, Y == clip(golden model, 0, 255) exactly.
//
// Window, line buffers and zero padding: window3x3.v
// Multiplications by 2 are wire shifts: no DSP, only adders.
//
// Latency: window3x3 latency + 2 clock cycles (gradients are registered:
// the single-cycle version missed 100 MHz by 0.1 ns after place & route).
// ready_in is low during the end-of-frame flush (IMG_WIDTH+1 cycles).
//
// Bit-exact reference model: verif/models/filters_hw.py::sobel_u8
//
// Part of VELA — Vision Edge Learning Accelerator
// Phase 2a: Verilog RTL image filters

module sobel_filter #(
    parameter IMG_WIDTH  = 96,
    parameter IMG_HEIGHT = 64,
    parameter MAG_SHIFT  = 0
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    output wire        ready_in,
    input  wire [7:0]  pixel_in,   // grayscale input
    output reg         valid_out,
    output reg  [7:0]  pixel_out   // edge magnitude
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

    // w00 = MSB (top-left) ... w22 = LSB (bottom-right), zero-extended to 12 bits.
    // The center pixel (win[39:32]) has weight 0 in both kernels.
    wire signed [11:0] w00 = {4'd0, win[71:64]}, w01 = {4'd0, win[63:56]}, w02 = {4'd0, win[55:48]};
    wire signed [11:0] w10 = {4'd0, win[47:40]},                           w12 = {4'd0, win[31:24]};
    wire signed [11:0] w20 = {4'd0, win[23:16]}, w21 = {4'd0, win[15:8]},  w22 = {4'd0, win[7:0]};

    /* verilator lint_off UNUSEDSIGNAL */
    wire [7:0] unused_center = win[39:32];
    /* verilator lint_on UNUSEDSIGNAL */

    // ── Gradients (signed, |G| <= 1020 fits in 12-bit signed) ──────────────

    wire signed [11:0] gx_c = (w02 + (w12 <<< 1) + w22) - (w00 + (w10 <<< 1) + w20);
    wire signed [11:0] gy_c = (w20 + (w21 <<< 1) + w22) - (w00 + (w01 <<< 1) + w02);

    // Pipeline register: gradients | absolute value, sum, saturation
    reg signed [11:0] gx, gy;
    reg               g_valid;

    always @(posedge clk) begin
        if (!rst_n)
            g_valid <= 1'b0;
        else
            g_valid <= win_valid;
        gx <= gx_c;
        gy <= gy_c;
    end

    // |G| <= 1020: the negated value always fits in the 11 LSBs (sign bit unused)
    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [11:0] neg_gx = -gx;
    wire signed [11:0] neg_gy = -gy;
    /* verilator lint_on UNUSEDSIGNAL */
    wire [10:0] abs_gx = gx[11] ? neg_gx[10:0] : gx[10:0];
    wire [10:0] abs_gy = gy[11] ? neg_gy[10:0] : gy[10:0];

    // ── Magnitude, scaling, saturation ─────────────────────────────────────

    wire [10:0] mag        = abs_gx + abs_gy;        // <= 1530 < 2048
    wire [10:0] mag_scaled = mag >> MAG_SHIFT;
    wire [7:0]  mag_sat    = (mag_scaled > 11'd255) ? 8'd255 : mag_scaled[7:0];

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            pixel_out <= 8'd0;
        end else begin
            valid_out <= g_valid;
            if (g_valid)
                pixel_out <= mag_sat;
        end
    end

endmodule
