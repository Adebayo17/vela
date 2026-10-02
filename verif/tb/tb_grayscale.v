// tb_grayscale.v — File-driven testbench for grayscale.v
//
// Driven by verif/tests/test_rtl_grayscale.py :
//   +IN=<file>    input pixels, one 24-bit hex word per line (RRGGBB)
//   +OUT=<file>   output luminance, one 8-bit hex word per line
//   +N=<count>    number of pixels
//   +SEED=<int>   seed for random valid_in gaps (0 = no gaps)
//
// Part of VELA — Vision Edge Learning Accelerator
// Phase 2a: Verilog RTL image filters

`timescale 1ns / 1ps

module tb_grayscale;

    localparam MAX_PIXELS = 65536;

    reg         clk = 1'b0;
    reg         rst_n = 1'b0;
    reg         valid_in = 1'b0;
    reg  [7:0]  r_in = 8'd0, g_in = 8'd0, b_in = 8'd0;
    wire        valid_out;
    wire [7:0]  y_out;

    reg  [23:0] pixels [0:MAX_PIXELS-1];
    reg  [1023:0] in_file, out_file;
    integer     n, seed, fd, sent, received, cycles;

    grayscale dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (valid_in),
        .r_in      (r_in),
        .g_in      (g_in),
        .b_in      (b_in),
        .valid_out (valid_out),
        .y_out     (y_out)
    );

    always #5 clk = ~clk;   // 100 MHz

    // Capture every valid output
    always @(posedge clk) begin
        if (rst_n && valid_out) begin
            $fwrite(fd, "%02x\n", y_out);
            received = received + 1;
        end
    end

    initial begin
        if (!$value$plusargs("IN=%s", in_file) || !$value$plusargs("OUT=%s", out_file)
            || !$value$plusargs("N=%d", n)) begin
            $display("ERROR: missing +IN, +OUT or +N plusarg");
            $finish;
        end
        if (!$value$plusargs("SEED=%d", seed)) seed = 0;

        $readmemh(in_file, pixels, 0, n - 1);
        fd = $fopen(out_file, "w");
        sent = 0;
        received = 0;

        repeat (4) @(posedge clk);
        rst_n <= 1'b1;

        while (sent < n) begin
            @(posedge clk);
            // Random gaps in valid_in (~25%) to exercise the handshake
            if (seed != 0 && ($random(seed) & 3) == 0) begin
                valid_in <= 1'b0;
            end else begin
                valid_in <= 1'b1;
                {r_in, g_in, b_in} <= pixels[sent];
                sent = sent + 1;
            end
        end
        @(posedge clk);
        valid_in <= 1'b0;

        // Drain pipeline
        cycles = 0;
        while (received < n && cycles < 100) begin
            @(posedge clk);
            cycles = cycles + 1;
        end

        $fclose(fd);
        if (received != n)
            $display("ERROR: expected %0d outputs, got %0d", n, received);
        else
            $display("DONE: %0d pixels", received);
        $finish;
    end

endmodule
