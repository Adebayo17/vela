// window3x3.v — 3x3 sliding window generator with zero padding
//
// Shared front-end of the 3x3 filters (sobel, gaussian, median).
//
// Input : one frame of IMG_WIDTH x IMG_HEIGHT pixels, raster order,
//         accepted when valid_in && ready_in.
// Output: one 3x3 window per input pixel, same raster order, centered on
//         that pixel. Neighbors outside the image are 0 (same as np.pad
//         with constant 0 in the golden model) -> output size = input size.
//
// window_out = {w00, w01, w02, w10, w11, w12, w20, w21, w22}  (w00 = MSB)
//              row-major, w00 = top-left, w11 = center.
//
// How it works:
//   - Two line buffers (lb1 = previous line, lb2 = line before) give, for
//     each new pixel k, the pixels k-W and k-2W of the same column.
//   - A 3x3 register array shifts left on every step; its bottom-right is
//     pixel k, so its center is pixel k-W-1.
//   - The center lags the input by W+1 pixels: after the last pixel of the
//     frame, W+1 zero "flush" steps are injected to output the last line.
//     ready_in is low during the flush.
//   - Neighbors outside the image (row -1, row H, column -1, column W) are
//     masked to 0 using the center position.
//
// Latency: W+1 accepted pixels + 2 clock cycles.
// Throughput: 1 pixel/cycle, plus W+1 flush cycles per frame.
//
// Bit-exact reference model: verif/models/filters_hw.py::windows_3x3_u8
//
// Part of VELA — Vision Edge Learning Accelerator
// Phase 2a: Verilog RTL image filters

module window3x3 #(
    parameter IMG_WIDTH  = 96,
    parameter IMG_HEIGHT = 64
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    output wire        ready_in,
    input  wire [7:0]  pixel_in,
    output reg         valid_out,
    output reg  [71:0] window_out
);

    localparam NPIX = IMG_WIDTH * IMG_HEIGHT;
    localparam CW   = $clog2(IMG_WIDTH);             // column counter width
    localparam RW   = $clog2(IMG_HEIGHT);            // row counter width
    localparam KW   = $clog2(NPIX + IMG_WIDTH + 1);  // step counter width

    // Values always fit: widths are computed with $clog2 above
    /* verilator lint_off WIDTHTRUNC */
    localparam [CW-1:0] LAST_COL    = IMG_WIDTH - 1;
    localparam [RW-1:0] LAST_ROW    = IMG_HEIGHT - 1;
    localparam [KW-1:0] K_FIRST_OUT = IMG_WIDTH + 1;     // first step with a valid center
    localparam [KW-1:0] K_LAST_PIX  = NPIX - 1;          // last real pixel
    localparam [KW-1:0] K_LAST      = NPIX + IMG_WIDTH;  // last flush step
    /* verilator lint_on WIDTHTRUNC */

    // ── Step control ────────────────────────────────────────────────────────

    reg           flushing;
    reg  [KW-1:0] k;          // step index in the frame (pixels + flush)
    reg  [CW-1:0] ptr;        // line buffer address = k mod W

    assign ready_in = !flushing;

    wire       step   = flushing || valid_in;
    wire [7:0] new_px = flushing ? 8'd0 : pixel_in;

    // ── Line buffers (no reset: maps to distributed RAM / BRAM) ────────────

    reg  [7:0] lb1 [0:IMG_WIDTH-1];   // pixel k-W
    reg  [7:0] lb2 [0:IMG_WIDTH-1];   // pixel k-2W

    wire [7:0] lb1_rd = lb1[ptr];
    wire [7:0] lb2_rd = lb2[ptr];

    always @(posedge clk) begin
        if (step) begin
            lb1[ptr] <= new_px;
            lb2[ptr] <= lb1_rd;
        end
    end

    // ── 3x3 window registers + center position ─────────────────────────────

    reg  [7:0] t0, t1, t2;   // top row
    reg  [7:0] m0, m1, m2;   // middle row (m1 = center)
    reg  [7:0] b0, b1, b2;   // bottom row

    reg  [RW-1:0] c_row, cur_row;   // next center / center of current window
    reg  [CW-1:0] c_col, cur_col;
    reg           win_valid;

    always @(posedge clk) begin
        if (step) begin
            {t0, t1, t2} <= {t1, t2, lb2_rd};
            {m0, m1, m2} <= {m1, m2, lb1_rd};
            {b0, b1, b2} <= {b1, b2, new_px};
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            flushing  <= 1'b0;
            k         <= {KW{1'b0}};
            ptr       <= {CW{1'b0}};
            c_row     <= {RW{1'b0}};
            c_col     <= {CW{1'b0}};
            cur_row   <= {RW{1'b0}};
            cur_col   <= {CW{1'b0}};
            win_valid <= 1'b0;
        end else begin
            win_valid <= 1'b0;

            if (step) begin
                ptr <= (ptr == LAST_COL) ? {CW{1'b0}} : ptr + 1'b1;

                // Center = pixel k-W-1: valid once the window covers row 0, col 0
                if (k >= K_FIRST_OUT) begin
                    win_valid <= 1'b1;
                    cur_row   <= c_row;
                    cur_col   <= c_col;
                    if (c_col == LAST_COL) begin
                        c_col <= {CW{1'b0}};
                        c_row <= c_row + 1'b1;
                    end else begin
                        c_col <= c_col + 1'b1;
                    end
                end

                if (k == K_LAST_PIX)
                    flushing <= 1'b1;

                if (k == K_LAST) begin   // end of frame: ready for the next one
                    flushing <= 1'b0;
                    k        <= {KW{1'b0}};
                    ptr      <= {CW{1'b0}};
                    c_row    <= {RW{1'b0}};
                    c_col    <= {CW{1'b0}};
                end else begin
                    k <= k + 1'b1;
                end
            end
        end
    end

    // ── Border masking (zero padding) ──────────────────────────────────────

    wire top_ok   = (cur_row != {RW{1'b0}});
    wire bot_ok   = (cur_row != LAST_ROW);
    wire left_ok  = (cur_col != {CW{1'b0}});
    wire right_ok = (cur_col != LAST_COL);

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out  <= 1'b0;
            window_out <= 72'd0;
        end else begin
            valid_out <= win_valid;
            if (win_valid)
                window_out <= {
                    (top_ok && left_ok)  ? t0 : 8'd0,
                     top_ok              ? t1 : 8'd0,
                    (top_ok && right_ok) ? t2 : 8'd0,
                     left_ok             ? m0 : 8'd0,
                                           m1,
                     right_ok            ? m2 : 8'd0,
                    (bot_ok && left_ok)  ? b0 : 8'd0,
                     bot_ok              ? b1 : 8'd0,
                    (bot_ok && right_ok) ? b2 : 8'd0
                };
        end
    end

endmodule
