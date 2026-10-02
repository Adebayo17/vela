// tb_window3x3.v — File-driven testbench for window3x3.v
//
// Driven by verif/tests/test_rtl_window3x3.py :
//   +IN=<file>    input pixels, one 8-bit hex word per line (one or more frames)
//   +OUT=<file>   output windows, one 72-bit hex word per line (w00 first)
//   +N=<count>    number of pixels (multiple of IMG_WIDTH*IMG_HEIGHT)
//   +SEED=<int>   seed for random valid_in gaps (0 = no gaps)
//
// Handshake: a pixel is consumed when valid_in && ready_in at a rising edge.
// Inputs are driven on the falling edge to avoid races with the DUT.
//
// Part of VELA — Vision Edge Learning Accelerator
// Phase 2a: Verilog RTL image filters

`timescale 1ns / 1ps

module tb_window3x3 #(
    parameter IMG_WIDTH  = 8,    // overridden with iverilog -P
    parameter IMG_HEIGHT = 6
);

    localparam MAX_PIXELS = 65536;

    reg         clk = 1'b0;
    reg         rst_n = 1'b0;
    reg         valid_in = 1'b0;
    wire        ready_in;
    reg  [7:0]  pixel_in = 8'd0;
    wire        valid_out;
    wire [71:0] window_out;

    reg  [7:0]  pixels [0:MAX_PIXELS-1];
    reg  [1023:0] in_file, out_file;
    integer     n, seed, fd, sent, received, cycles;

    window3x3 #(
        .IMG_WIDTH  (IMG_WIDTH),
        .IMG_HEIGHT (IMG_HEIGHT)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .valid_in   (valid_in),
        .ready_in   (ready_in),
        .pixel_in   (pixel_in),
        .valid_out  (valid_out),
        .window_out (window_out)
    );

    always #5 clk = ~clk;   // 100 MHz

    // Count consumed pixels and capture every valid output
    always @(posedge clk) begin
        if (rst_n && valid_in && ready_in)
            sent = sent + 1;
        if (rst_n && valid_out) begin
            $fwrite(fd, "%018x\n", window_out);
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
        @(negedge clk);
        rst_n = 1'b1;

        // Drive on falling edges; hold the same pixel until it is consumed
        while (sent < n) begin
            if (seed != 0 && ($random(seed) & 3) == 0) begin
                valid_in = 1'b0;
            end else begin
                valid_in = 1'b1;
                pixel_in = pixels[sent];
            end
            @(negedge clk);
        end
        valid_in = 1'b0;

        // Drain: flush (W+1 steps) + pipeline
        cycles = 0;
        while (received < n && cycles < IMG_WIDTH + 100) begin
            @(posedge clk);
            cycles = cycles + 1;
        end
        repeat (5) @(posedge clk);   // catch any extra (unexpected) output

        $fclose(fd);
        if (received != n)
            $display("ERROR: expected %0d outputs, got %0d", n, received);
        else
            $display("DONE: %0d windows", received);
        $finish;
    end

endmodule
