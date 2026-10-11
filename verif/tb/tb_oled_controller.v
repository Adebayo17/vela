// tb_oled_controller.v — File-driven testbench for oled_controller.v
//
// Driven by verif/tests/test_rtl_oled_controller.py :
//   +IN=<file>      RGB565 pixels, one 16-bit hex word per line
//   +OUT=<file>     48-bit hex words, in time order:
//                     0 <dc> <byte> <cycle>     SPI byte (spi_monitor.v)
//                     1 <sig> <value> <cycle>   control signal change
//                   sig: 1 = oled_pmoden, 2 = oled_vccen, 3 = oled_res,
//                        4 = s_axis_tready (first rise only)
//   +N=<count>      number of pixels
//   +NBYTES=<int>   expected number of SPI bytes (init + 2 per pixel)
//   +HALF=<int>     expected SCLK half period in clk cycles (from the spec)
//   +MAXCYC=<int>   simulation limit in clock cycles
//   +SEED=<int>     seed for random s_axis_tvalid gaps (0 = always valid)
//   +VCD=<file>     optional: dump all signals to a VCD file (GTKWave)
//
// AXI-Stream: a pixel is consumed when s_axis_tvalid && s_axis_tready at a
// rising edge. Inputs are driven on the falling edge.
//
// Part of VELA — Vision Edge Learning Accelerator (verification only)

`timescale 1ns / 1ps

module tb_oled_controller #(
    parameter CLK_FREQ        = 100_000_000,   // overridden with iverilog -P
    parameter SPI_FREQ        = 6_250_000,
    parameter POWER_UP_CYCLES = 0,             // 0 = keep the DUT defaults (from CLK_FREQ)
    parameter RESET_CYCLES    = 0
);

    localparam MAX_PIXELS = 65536;

    reg         clk = 1'b0;
    reg         rst_n = 1'b0;
    reg  [15:0] s_axis_tdata = 16'd0;
    reg         s_axis_tvalid = 1'b0;
    wire        s_axis_tready;
    wire        oled_spi_sclk, oled_spi_mosi, oled_dc, oled_res, oled_vccen, oled_pmoden;

    reg  [15:0] pixels [0:MAX_PIXELS-1];
    reg  [1023:0] in_file, out_file, vcd_file;
    integer     n, nbytes_exp, maxcyc, seed, fd, sent;
    reg  [31:0] cycle = 0, half = 0;
    wire [31:0] nbytes, errors;
    reg         pmoden_q = 1'b0, vccen_q = 1'b0, res_q = 1'b1, tready_seen = 1'b0;

    // Short delays for functional tests, or the DUT defaults (real timing)
    `define OLED_PORTS                      \
        .clk           (clk),               \
        .rst_n         (rst_n),             \
        .s_axis_tdata  (s_axis_tdata),      \
        .s_axis_tvalid (s_axis_tvalid),     \
        .s_axis_tready (s_axis_tready),     \
        .oled_spi_sclk (oled_spi_sclk),     \
        .oled_spi_mosi (oled_spi_mosi),     \
        .oled_dc       (oled_dc),           \
        .oled_res      (oled_res),          \
        .oled_vccen    (oled_vccen),        \
        .oled_pmoden   (oled_pmoden)

    generate
        if (POWER_UP_CYCLES > 0) begin : g_short
            oled_controller #(
                .CLK_FREQ        (CLK_FREQ),
                .SPI_FREQ        (SPI_FREQ),
                .POWER_UP_CYCLES (POWER_UP_CYCLES),
                .RESET_CYCLES    (RESET_CYCLES)
            ) dut (`OLED_PORTS);
        end else begin : g_default
            oled_controller #(
                .CLK_FREQ        (CLK_FREQ),
                .SPI_FREQ        (SPI_FREQ)
            ) dut (`OLED_PORTS);
        end
    endgenerate

    spi_monitor mon (
        .clk    (clk),
        .rst_n  (rst_n),
        .sclk   (oled_spi_sclk),
        .mosi   (oled_spi_mosi),
        .dc     (oled_dc),
        .cycle  (cycle),
        .half   (half),
        .fd     (fd),
        .nbytes (nbytes),
        .errors (errors)
    );

    always #5 clk = ~clk;   // 100 MHz

    // Cycle counter, consumed pixels, control signal changes
    always @(posedge clk) begin
        cycle <= cycle + 1;
        if (rst_n) begin
            if (s_axis_tvalid && s_axis_tready)
                sent = sent + 1;
            if (oled_pmoden !== pmoden_q) $fwrite(fd, "11%02x%08x\n", oled_pmoden, cycle);
            if (oled_vccen  !== vccen_q)  $fwrite(fd, "12%02x%08x\n", oled_vccen,  cycle);
            if (oled_res    !== res_q)    $fwrite(fd, "13%02x%08x\n", oled_res,    cycle);
            if (s_axis_tready && !tready_seen) begin
                $fwrite(fd, "14%02x%08x\n", 8'd1, cycle);
                tready_seen = 1'b1;
            end
            pmoden_q = oled_pmoden;
            vccen_q  = oled_vccen;
            res_q    = oled_res;
        end
    end

    initial begin
        if (!$value$plusargs("IN=%s", in_file) || !$value$plusargs("OUT=%s", out_file)
            || !$value$plusargs("N=%d", n) || !$value$plusargs("NBYTES=%d", nbytes_exp)
            || !$value$plusargs("HALF=%d", half) || !$value$plusargs("MAXCYC=%d", maxcyc)) begin
            $display("ERROR: missing +IN, +OUT, +N, +NBYTES, +HALF or +MAXCYC plusarg");
            $finish;
        end
        if (!$value$plusargs("SEED=%d", seed)) seed = 0;

        // Optional waveform dump for GTKWave: +VCD=<file>
        if ($value$plusargs("VCD=%s", vcd_file)) begin
            $dumpfile(vcd_file);
            $dumpvars(0, tb_oled_controller);
        end

        $readmemh(in_file, pixels, 0, n - 1);
        fd = $fopen(out_file, "w");
        sent = 0;

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        // Drive on falling edges; hold the same pixel until it is consumed
        while (sent < n && cycle < maxcyc) begin
            if (seed != 0 && ($random(seed) & 3) == 0) begin
                s_axis_tvalid = 1'b0;
            end else begin
                s_axis_tvalid = 1'b1;
                s_axis_tdata  = pixels[sent];
            end
            @(negedge clk);
        end
        s_axis_tvalid = 1'b0;

        // Wait for every expected SPI byte, then a little more to catch extras
        while (nbytes < nbytes_exp && cycle < maxcyc)
            @(posedge clk);
        repeat (400) @(posedge clk);

        $fclose(fd);
        if (errors != 0)
            $display("ERROR: %0d SPI protocol violation(s)", errors);
        else if (cycle >= maxcyc)
            $display("ERROR: timeout after %0d cycles (%0d/%0d SPI bytes)", cycle, nbytes, nbytes_exp);
        else if (nbytes != nbytes_exp)
            $display("ERROR: expected %0d SPI bytes, got %0d", nbytes_exp, nbytes);
        else
            $display("DONE: %0d pixels, %0d SPI bytes", sent, nbytes);
        $finish;
    end

endmodule
