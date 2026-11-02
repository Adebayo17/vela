// tb_display_reader.v — File-driven testbench for display_reader
//
// Display memory model: N_PIXELS 32-bit words loaded from +IN, read latency 1
// (Block Memory Generator port B). The upper half-words are random in the
// test data: only bits 15:0 may reach the stream.
//
// Plusargs:
//   +IN, +OUT, +N        memory content, output file, N_PIXELS
//   +COUNT=<n>           pixels to collect (several frames)
//   +SEED=<int>          0: tready always high, grant always high;
//                        else random tready and random grant
//   +TREADY_AT=<cycle>   first cycle with tready allowed (default 20)
//
// Output (+OUT): the COUNT pixels accepted on the stream (tvalid && tready),
// 16-bit hex, in order.
//
// Checks (each prints "ERROR: ..."):
//   - no memory read before the first tready, nor while grant is low;
//   - AXI-Stream: tdata stable and tvalid held while tvalid && !tready.
//
// Part of VELA — Vision Edge Learning Accelerator (verification only)

`timescale 1ns / 1ps

module tb_display_reader #(
    parameter N_PIXELS = 20       // overridden with iverilog -P
);

    reg         clk = 1'b0;
    reg         rst_n = 1'b0;
    reg         grant = 1'b0;
    reg         tready = 1'b0;
    wire [31:0] mem_addr;
    wire        mem_en;
    reg  [31:0] mem_dout = 32'd0;
    wire [15:0] tdata;
    wire        tvalid;

    reg  [31:0] mem [0:N_PIXELS-1];
    reg  [1023:0] in_file, out_file, vcd_file;
    integer     n, count, seed, tready_at, fd, received, cycle;
    reg         tready_seen, stall;
    reg  [15:0] tdata_q;

    display_reader #(
        .N_PIXELS       (N_PIXELS)
    ) dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .grant          (grant),
        .mem_addr       (mem_addr),
        .mem_en         (mem_en),
        .mem_dout       (mem_dout),
        .m_axis_tdata   (tdata),
        .m_axis_tvalid  (tvalid),
        .m_axis_tready  (tready)
    );

    always #5 clk = ~clk;   // 100 MHz

    // Memory: read latency 1, access checks
    always @(posedge clk) begin
        if (rst_n && mem_en) begin
            if (!tready_seen) $display("ERROR: read before the first tready (cycle %0d)", cycle);
            if (!grant)       $display("ERROR: read while grant is low (cycle %0d)", cycle);
            if (mem_addr[1:0] != 2'b00 || (mem_addr >> 2) >= N_PIXELS)
                $display("ERROR: bad address %h", mem_addr);
            else
                mem_dout <= mem[mem_addr >> 2];
        end
    end

    // Stream: capture handshakes, check the AXI-Stream rules
    always @(posedge clk) begin
        cycle = cycle + 1;
        if (rst_n) begin
            if (tready) tready_seen = 1'b1;
            if (stall && !tvalid)      $display("ERROR: tvalid dropped before tready (cycle %0d)", cycle);
            if (stall && tdata !== tdata_q) $display("ERROR: tdata changed before tready (cycle %0d)", cycle);
            if (tvalid && tready) begin
                $fwrite(fd, "%04x\n", tdata);
                received = received + 1;
            end
            stall   = tvalid && !tready;
            tdata_q = tdata;
        end
    end

    initial begin
        if (!$value$plusargs("IN=%s", in_file) || !$value$plusargs("OUT=%s", out_file)
            || !$value$plusargs("N=%d", n) || !$value$plusargs("COUNT=%d", count)) begin
            $display("ERROR: missing +IN, +OUT, +N or +COUNT plusarg");
            $finish;
        end
        if (n != N_PIXELS) begin
            $display("ERROR: +N=%0d, expected %0d", n, N_PIXELS);
            $finish;
        end
        if (!$value$plusargs("SEED=%d", seed)) seed = 0;
        if (!$value$plusargs("TREADY_AT=%d", tready_at)) tready_at = 20;

        if ($value$plusargs("VCD=%s", vcd_file)) begin
            $dumpfile(vcd_file);
            $dumpvars(0, tb_display_reader);
        end

        $readmemh(in_file, mem, 0, n - 1);
        fd = $fopen(out_file, "w");
        received = 0;
        cycle = 0;
        tready_seen = 1'b0;
        stall = 1'b0;

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        // Drive tready and grant on falling edges
        while (received < count && cycle < 100 * count + 1000) begin
            if (seed == 0) begin
                grant  = 1'b1;
                tready = (cycle >= tready_at);
            end else begin
                grant  = ($random(seed) & 3) != 0;                        // low 25 %
                tready = (cycle >= tready_at) && (($random(seed) & 1) != 0); // low 50 %
            end
            @(negedge clk);
        end
        tready = 1'b0;
        repeat (5) @(posedge clk);

        $fclose(fd);
        if (received < count)
            $display("ERROR: expected %0d pixels, got %0d", count, received);
        else
            $display("DONE: %0d pixels", received);
        $finish;
    end

endmodule
