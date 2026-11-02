// tb_spi_master.v — File-driven testbench for spi_master.v
//
// Driven by verif/tests/test_rtl_spi_master.py :
//   +IN=<file>    bytes to send, one 8-bit hex word per line
//   +OUT=<file>   decoded SPI bytes (spi_monitor format, 48-bit hex words)
//   +N=<count>    number of bytes
//   +HALF=<int>   expected SCLK half period in clk cycles (from the spec)
//   +SEED=<int>   0 = back-to-back bytes; else random idle gaps between
//                 bytes and spurious `start` pulses while busy (must be ignored)
//   +VCD=<file>   optional: dump all signals to a VCD file (GTKWave)
//
// Checks done here: exactly one done_tick per byte, busy low at the end.
// Protocol checks (mode 0, half period): spi_monitor.v.
//
// Part of VELA — Vision Edge Learning Accelerator (verification only)

`timescale 1ns / 1ps

module tb_spi_master #(
    parameter CLK_FREQ = 100_000_000,   // overridden with iverilog -P
    parameter SPI_FREQ = 6_250_000
);

    localparam MAX_BYTES = 65536;

    reg         clk = 1'b0;
    reg         rst_n = 1'b0;
    reg         start = 1'b0;
    reg  [7:0]  data_in = 8'd0;
    wire        busy, done_tick, sclk, mosi;

    reg  [7:0]  bytes [0:MAX_BYTES-1];
    reg  [1023:0] in_file, out_file, vcd_file;
    integer     n, seed, fd, sent, dones, gap, timeout;
    reg  [31:0] cycle = 0, half = 0;
    wire [31:0] nbytes, errors;

    spi_master #(
        .CLK_FREQ (CLK_FREQ),
        .SPI_FREQ (SPI_FREQ)
    ) dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (start),
        .data_in   (data_in),
        .busy      (busy),
        .done_tick (done_tick),
        .sclk      (sclk),
        .mosi      (mosi)
    );

    spi_monitor mon (
        .clk    (clk),
        .rst_n  (rst_n),
        .sclk   (sclk),
        .mosi   (mosi),
        .dc     (1'b0),
        .cycle  (cycle),
        .half   (half),
        .fd     (fd),
        .nbytes (nbytes),
        .errors (errors)
    );

    always #5 clk = ~clk;   // 100 MHz

    always @(posedge clk) begin
        cycle <= cycle + 1;
        if (rst_n && done_tick)
            dones = dones + 1;
    end

    initial begin
        if (!$value$plusargs("IN=%s", in_file) || !$value$plusargs("OUT=%s", out_file)
            || !$value$plusargs("N=%d", n) || !$value$plusargs("HALF=%d", half)) begin
            $display("ERROR: missing +IN, +OUT, +N or +HALF plusarg");
            $finish;
        end
        if (!$value$plusargs("SEED=%d", seed)) seed = 0;

        // Optional waveform dump for GTKWave: +VCD=<file>
        if ($value$plusargs("VCD=%s", vcd_file)) begin
            $dumpfile(vcd_file);
            $dumpvars(0, tb_spi_master);
        end

        $readmemh(in_file, bytes, 0, n - 1);
        fd = $fopen(out_file, "w");
        sent = 0;
        dones = 0;

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        // Drive on falling edges: one start pulse per byte, when not busy
        while (sent < n) begin
            @(negedge clk);
            if (!busy && !start) begin
                if (seed != 0) begin
                    gap = $random(seed) & 7;            // 0..7 idle cycles
                    repeat (gap) @(negedge clk);
                end
                start = 1'b1;
                data_in = bytes[sent];
                sent = sent + 1;
                @(negedge clk);
                start = 1'b0;
            end else if (busy && seed != 0 && ($random(seed) & 15) == 0) begin
                start = 1'b1;                           // spurious: must be ignored
                data_in = 8'hA5;
                @(negedge clk);
                start = 1'b0;
            end
        end

        // Wait for the last byte
        timeout = 0;
        while ((busy || nbytes < n) && timeout < 100000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        repeat (50) @(posedge clk);   // catch any extra byte

        $fclose(fd);
        if (errors != 0)
            $display("ERROR: %0d SPI protocol violation(s)", errors);
        else if (nbytes != n)
            $display("ERROR: expected %0d SPI bytes, got %0d", n, nbytes);
        else if (dones != n)
            $display("ERROR: expected %0d done_tick pulses, got %0d", n, dones);
        else if (busy)
            $display("ERROR: busy still high at the end");
        else
            $display("DONE: %0d bytes", nbytes);
        $finish;
    end

endmodule
