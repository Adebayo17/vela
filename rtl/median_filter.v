// median_filter.v — Median filter (3x3)
// Y = median of the 9 window pixels (zero padding at the borders).
//
// Sorting network: "opt_med9" median selection, 19 compare-and-swap (CAS)
// operations on 9 levels, split into 3 pipeline stages of 3 levels each so
// that each stage stays short at 100 MHz. CAS(a, b): p[a] = min, p[b] = max.
// The median ends in p[4]. Exhaustively verified in Python (all 9! orderings
// of distinct values + all 512 0/1 inputs): verif/models/filters_hw.py.
//
// Only the values still needed are registered between stages:
// 9 after stage 1, 3 (p2, p4, p6) after stage 2.
//
// Window, line buffers and zero padding: window3x3.v
// Latency: window3x3 latency + 3 clock cycles.
// ready_in is low during the end-of-frame flush (IMG_WIDTH+1 cycles).
//
// Bit-exact reference model: verif/models/filters_hw.py::median_u8
// (network: MEDIAN9_STAGES, same CAS list as below)
//
// Part of VELA — Vision Edge Learning Accelerator
// Phase 2a: Verilog RTL image filters

`timescale 1ns / 1ps

module median_filter #(
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

    // Compare-and-swap on two regs: a = min, b = max (blocking, combinational).
    // tmp: swap variable, one per stage so that each is driven by a single block.
    `define CAS(a, b, tmp) if (a > b) begin tmp = a; a = b; b = tmp; end

    // ── Stage 1: levels 1-3 (sort each row) ───────────────────────────────

    reg [7:0] p0, p1, p2, p3, p4, p5, p6, p7, p8, t;
    reg [7:0] s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7, s1_8;
    reg       s1_valid;

    always @* begin
        {p0, p1, p2, p3, p4, p5, p6, p7, p8} = win;   // p0 = w00 (MSB) ... p8 = w22
        t = 8'd0;
        `CAS(p1, p2, t) `CAS(p4, p5, t) `CAS(p7, p8, t)
        `CAS(p0, p1, t) `CAS(p3, p4, t) `CAS(p6, p7, t)
        `CAS(p1, p2, t) `CAS(p4, p5, t) `CAS(p7, p8, t)
    end

    always @(posedge clk) begin
        if (!rst_n)
            s1_valid <= 1'b0;
        else
            s1_valid <= win_valid;
        {s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7, s1_8}
            <= {p0, p1, p2, p3, p4, p5, p6, p7, p8};
    end

    // ── Stage 2: levels 4-6 ────────────────────────────────────────────────

    reg [7:0] q0, q1, q2, q3, q4, q5, q6, q7, q8, u;
    reg [7:0] s2_2, s2_4, s2_6;
    reg       s2_valid;

    always @* begin
        {q0, q1, q2, q3, q4, q5, q6, q7, q8}
            = {s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7, s1_8};
        u = 8'd0;
        `CAS(q0, q3, u) `CAS(q5, q8, u) `CAS(q4, q7, u)
        `CAS(q3, q6, u) `CAS(q1, q4, u) `CAS(q2, q5, u)
        `CAS(q4, q7, u)
    end

    always @(posedge clk) begin
        if (!rst_n)
            s2_valid <= 1'b0;
        else
            s2_valid <= s1_valid;
        {s2_2, s2_4, s2_6} <= {q2, q4, q6};
    end

    // ── Stage 3: levels 7-9, median in r4 ─────────────────────────────────

    reg [7:0] r2, r4, r6, v;

    always @* begin
        {r2, r4, r6} = {s2_2, s2_4, s2_6};
        v = 8'd0;
        `CAS(r4, r2, v) `CAS(r6, r4, v) `CAS(r4, r2, v)
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            pixel_out <= 8'd0;
        end else begin
            valid_out <= s2_valid;
            if (s2_valid)
                pixel_out <= r4;
        end
    end

    `undef CAS

endmodule
