// filter_engine.v — One image from the source memory to the display memory
//
// On a rising edge of `start` (while idle), reads the SRC_WIDTH x SRC_HEIGHT
// source image (one RGB888 pixel per 32-bit word, 0x00RRGGBB, raster order),
// sends it through the filter path selected by `mode` and writes the RGB565
// result into the IMAGE ZONE of the display memory (VELA-TN-007 §5.1, §6.2):
//
//   mode 0 copy      RGB                               -> RGB565
//   mode 1 grayscale RGB -> grayscale                  -> gray RGB565
//   mode 2 threshold RGB -> grayscale -> threshold     -> gray RGB565
//   mode 3 gaussian  R, G, B -> gaussian_filter x 3    -> RGB565
//   mode 4 sobel     RGB -> grayscale -> sobel_filter  -> gray RGB565
//   mode 5 median    R, G, B -> median_filter x 3      -> RGB565
//   mode 6, 7        copy
//
// Display memory word of output pixel (x, y): y * DISP_WIDTH + x, x < SRC_WIDTH.
// The engine never writes columns SRC_WIDTH..DISP_WIDTH-1 (text zone, owned
// by the ARM): the address generator cannot produce them.
//
// Memories: BRAM controller conventions (byte address = word index * 4),
// read latency 1 cycle. The source word is registered once (src_q) before
// the filters: the BRAM output is not registered (latency 1), and its
// clock-to-out plus the grayscale adders missed 100 MHz by 0.74 ns in the
// complete design. The source is read one word per cycle without
// stopping: the 3x3 filters drop ready_in only during their end-of-frame
// flush, after the last pixel (VELA-TN-003 §6.4), and their last output comes
// after that flush. Only the selected path receives pixels, so when done
// rises every filter is ready: a new start can be accepted at once (ready_in
// is therefore not used).
//
// Status: busy rises on the accepted start edge and falls one cycle after the
// last write has been presented to the memory (the display port multiplexer
// of vela_pl uses busy to select the writer). done rises with the fall of
// busy and stays set until the next start.
//
// Duration: SRC_WIDTH * SRC_HEIGHT cycles + pipeline latency
// (64 x 64: 4 096 + at most SRC_WIDTH + 9 cycles, ~41 us at 100 MHz).
//
// Bit-exact reference model: verif/models/pipeline_hw.py::filter_frame
// Verified by: verif/tests/test_rtl_filter_engine.py
//
// Part of VELA — Vision Edge Learning Accelerator
// Phase 3: FPGA integration

`timescale 1ns / 1ps

module filter_engine #(
    parameter       SRC_WIDTH  = 64,       // image zone width (pixels)
    parameter       SRC_HEIGHT = 64,       // image height (pixels)
    parameter       DISP_WIDTH = 96,       // display memory row length (pixels)
    parameter [7:0] THRESHOLD  = 8'd128,   // threshold.v parameter
    parameter       MAG_SHIFT  = 2         // sobel_filter.v parameter
)(
    input  wire        clk,
    input  wire        rst_n,              // synchronous, active low
    // Control (axi_gpio_ctrl, VELA-TN-007 §7)
    input  wire        start,              // rising edge starts one image
    input  wire [2:0]  mode,               // sampled with the start edge
    output wire        busy,
    output reg         done,
    // Source memory, read only
    output wire [31:0] src_addr,           // byte address
    output wire        src_en,
    input  wire [31:0] src_dout,           // 0x00RRGGBB, 1 cycle after src_en
    // Display memory, write only (image zone)
    output reg  [31:0] dst_addr,           // byte address
    output reg         dst_we,             // write strobe (whole word)
    output reg  [31:0] dst_din             // 0x0000 + RGB565
);

    localparam [15:0] N_PIXELS = SRC_WIDTH * SRC_HEIGHT;
    localparam [15:0] LAST_X   = SRC_WIDTH - 1;
    localparam [15:0] ROW_STEP = DISP_WIDTH;

    localparam [2:0] M_COPY   = 3'd0,
                     M_GRAY   = 3'd1,
                     M_THRESH = 3'd2,
                     M_GAUSS  = 3'd3,
                     M_SOBEL  = 3'd4,
                     M_MEDIAN = 3'd5;

    // ── Control state ───────────────────────────────────────────────────────

    reg         start_d1;       // previous value of start (edge detection)
    reg         running;        // reading / writing the image
    reg         finish;         // last write presented this cycle
    reg  [2:0]  mode_r;         // path of the current image
    reg  [15:0] rd_idx;         // next source word to read
    reg         rd_valid;       // src_dout holds a pixel this cycle
    reg  [23:0] src_q;          // registered source pixel (RGB888)
    reg         pix_valid;      // src_q holds a pixel this cycle
    reg  [15:0] wr_idx;         // output pixels written so far
    reg  [15:0] wr_x;           // column of the next output pixel
    reg  [15:0] row_base;       // display word index of the current row

    wire start_rise = start & ~start_d1;

    assign busy     = running;
    assign src_en   = running & (rd_idx < N_PIXELS);
    assign src_addr = {14'd0, rd_idx, 2'b00};

    wire [7:0] r = src_q[23:16];
    wire [7:0] g = src_q[15:8];
    wire [7:0] b = src_q[7:0];

    // ── Filter paths: only the selected one receives pixels ────────────────

    wire gray_path = (mode_r == M_GRAY) | (mode_r == M_THRESH) | (mode_r == M_SOBEL);
    wire gray_vin  = pix_valid & gray_path;
    wire gauss_vin = pix_valid & (mode_r == M_GAUSS);
    wire med_vin   = pix_valid & (mode_r == M_MEDIAN);

    wire        gray_vout;
    wire [7:0]  gray_y;

    grayscale u_grayscale (
        .clk        (clk),
        .rst_n      (rst_n),
        .valid_in   (gray_vin),
        .r_in       (r),
        .g_in       (g),
        .b_in       (b),
        .valid_out  (gray_vout),
        .y_out      (gray_y)
    );

    wire        thr_vin   = gray_vout & (mode_r == M_THRESH);
    wire        sobel_vin = gray_vout & (mode_r == M_SOBEL);
    wire        thr_vout, sobel_vout, sobel_ready;
    wire [7:0]  thr_y, sobel_y;

    threshold #(
        .THRESHOLD  (THRESHOLD)
    ) u_threshold (
        .clk        (clk),
        .rst_n      (rst_n),
        .valid_in   (thr_vin),
        .pixel_in   (gray_y),
        .valid_out  (thr_vout),
        .pixel_out  (thr_y)
    );

    sobel_filter #(
        .IMG_WIDTH  (SRC_WIDTH),
        .IMG_HEIGHT (SRC_HEIGHT),
        .MAG_SHIFT  (MAG_SHIFT)
    ) u_sobel (
        .clk        (clk),
        .rst_n      (rst_n),
        .valid_in   (sobel_vin),
        .ready_in   (sobel_ready),
        .pixel_in   (gray_y),
        .valid_out  (sobel_vout),
        .pixel_out  (sobel_y)
    );

    // Gaussian and median in colour: one instance per channel, same valid_in,
    // so the three run in lockstep; channel R drives the write control.
    wire [2:0]  gauss_ready, gauss_vout, med_ready, med_vout;
    wire [7:0]  gauss_px [0:2];
    wire [7:0]  med_px   [0:2];
    wire [7:0]  chan     [0:2];

    assign chan[0] = r;
    assign chan[1] = g;
    assign chan[2] = b;

    genvar c;
    generate
        for (c = 0; c < 3; c = c + 1) begin : g_channel
            gaussian_filter #(
                .IMG_WIDTH  (SRC_WIDTH),
                .IMG_HEIGHT (SRC_HEIGHT)
            ) u_gaussian (
                .clk        (clk),
                .rst_n      (rst_n),
                .valid_in   (gauss_vin),
                .ready_in   (gauss_ready[c]),
                .pixel_in   (chan[c]),
                .valid_out  (gauss_vout[c]),
                .pixel_out  (gauss_px[c])
            );

            median_filter #(
                .IMG_WIDTH  (SRC_WIDTH),
                .IMG_HEIGHT (SRC_HEIGHT)
            ) u_median (
                .clk        (clk),
                .rst_n      (rst_n),
                .valid_in   (med_vin),
                .ready_in   (med_ready[c]),
                .pixel_in   (chan[c]),
                .valid_out  (med_vout[c]),
                .pixel_out  (med_px[c])
            );
        end
    endgenerate

    // ── Output selection and RGB565 conversion (VELA-TN-003 §11) ───────────

    reg         out_valid;
    reg  [15:0] out_px;

    always @(*) begin
        out_valid = 1'b0;
        out_px    = 16'd0;
        case (mode_r)
            M_GRAY: begin
                out_valid = gray_vout;
                out_px    = {gray_y[7:3], gray_y[7:2], gray_y[7:3]};
            end
            M_THRESH: begin
                out_valid = thr_vout;
                out_px    = {thr_y[7:3], thr_y[7:2], thr_y[7:3]};
            end
            M_GAUSS: begin
                out_valid = gauss_vout[0];
                out_px    = {gauss_px[0][7:3], gauss_px[1][7:2], gauss_px[2][7:3]};
            end
            M_SOBEL: begin
                out_valid = sobel_vout;
                out_px    = {sobel_y[7:3], sobel_y[7:2], sobel_y[7:3]};
            end
            M_MEDIAN: begin
                out_valid = med_vout[0];
                out_px    = {med_px[0][7:3], med_px[1][7:2], med_px[2][7:3]};
            end
            default: begin                                      // copy (0, 6, 7)
                out_valid = pix_valid;
                out_px    = {r[7:3], g[7:2], b[7:3]};
            end
        endcase
    end

    // ── Sequencing ─────────────────────────────────────────────────────────

    always @(posedge clk) begin
        if (!rst_n) begin
            start_d1  <= 1'b0;
            running   <= 1'b0;
            finish    <= 1'b0;
            done      <= 1'b0;
            mode_r    <= M_COPY;
            rd_idx    <= 16'd0;
            rd_valid  <= 1'b0;
            src_q     <= 24'd0;
            pix_valid <= 1'b0;
            wr_idx    <= 16'd0;
            wr_x      <= 16'd0;
            row_base  <= 16'd0;
            dst_addr  <= 32'd0;
            dst_we    <= 1'b0;
            dst_din   <= 32'd0;
        end else begin
            start_d1 <= start;
            dst_we   <= 1'b0;
            rd_valid  <= src_en;
            pix_valid <= rd_valid;
            if (rd_valid)
                src_q <= src_dout[23:0];

            // Start: mode sampled now (6-7 select no filter path and fall in
            // the copy case); the first read is issued next cycle, once
            // mode_r holds the new path
            if (start_rise && !busy) begin
                running   <= 1'b1;
                done      <= 1'b0;
                mode_r    <= mode;
                rd_idx    <= 16'd0;
                wr_idx    <= 16'd0;
                wr_x      <= 16'd0;
                row_base  <= 16'd0;
            end

            if (src_en)
                rd_idx <= rd_idx + 16'd1;

            // Write the k-th result at row_base + wr_x (image zone only)
            if (running && out_valid && !finish) begin
                dst_we   <= 1'b1;
                dst_din  <= {16'd0, out_px};
                dst_addr <= {14'd0, row_base + wr_x, 2'b00};
                if (wr_x == LAST_X) begin
                    wr_x     <= 16'd0;
                    row_base <= row_base + ROW_STEP;
                end else begin
                    wr_x <= wr_x + 16'd1;
                end
                wr_idx <= wr_idx + 16'd1;
                if (wr_idx == N_PIXELS - 16'd1)
                    finish <= 1'b1;
            end

            // Last write is presented this cycle: release the port next edge
            if (finish) begin
                finish  <= 1'b0;
                running <= 1'b0;
                done    <= 1'b1;
            end
        end
    end

    // Outputs of channels G and B only duplicate the R control; ready_in is
    // always high when a start is accepted (see header); byte 3 of the source
    // word is unused (0x00RRGGBB); RGB565 drops the 2 LSBs of a grey value
    // (G keeps 6 bits, R and B 5 bits)
    wire unused = &{1'b0, gauss_vout[2:1], med_vout[2:1], sobel_ready, gauss_ready,
                    med_ready, src_dout[31:24], thr_y[1:0], sobel_y[1:0]};

endmodule
