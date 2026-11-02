// tb_filter_engine.v — File-driven testbench for filter_engine
//
// Models the two memories as the Block Memory Generators of the block design
// (port B, read latency 1, byte addresses):
//   - source: SRC_WIDTH x SRC_HEIGHT words 0x00RRGGBB, loaded from +IN;
//   - display: DISP_WIDTH x SRC_HEIGHT words, pre-filled with the marker
//     MARK | index, so that any word the engine does not write is visible.
//
// Runs (all without reset in between):
//   1. start with +MODE, wait for done;
//      with +BUSY_START=<m>: a second start edge with mode m is sent while the
//      engine is busy (must be ignored);
//   2. optional: +MODE2=<m>, started +GAP2 cycles after done (restart).
//
// Output (+OUT): the whole display memory (DISP_WIDTH x SRC_HEIGHT words,
// 32-bit hex), then one line with the start -> done cycle count of the last
// run.
//
// Checks (each prints "ERROR: ..."):
//   - every write is word aligned, inside the memory and in the image zone
//     (column < SRC_WIDTH): the text zone is never written;
//   - writes happen only while busy;
//   - in the last run each image-zone word is written exactly once;
//   - busy rises after the start edge, done only after busy falls.
//
// Part of VELA — Vision Edge Learning Accelerator (verification only)

`timescale 1ns / 1ps

module tb_filter_engine #(
    parameter SRC_WIDTH  = 8,     // overridden with iverilog -P
    parameter SRC_HEIGHT = 6,
    parameter DISP_WIDTH = 12
);

    localparam N_SRC  = SRC_WIDTH * SRC_HEIGHT;
    localparam N_DISP = DISP_WIDTH * SRC_HEIGHT;
    localparam [31:0] MARK = 32'hA5A5_0000;

    reg         clk = 1'b0;
    reg         rst_n = 1'b0;
    reg         start = 1'b0;
    reg  [2:0]  mode = 3'd0;
    wire        busy, done;
    wire [31:0] src_addr, dst_addr, dst_din;
    wire        src_en, dst_we;
    reg  [31:0] src_dout = 32'd0;

    reg  [31:0] src_mem [0:N_SRC-1];
    reg  [31:0] dst_mem [0:N_DISP-1];
    integer     writes  [0:N_DISP-1];

    reg  [1023:0] in_file, out_file, vcd_file;
    integer     n, seed, fd, i, cycles, timeout, mode1, mode2, gap2, busy_start;
    reg         busy_seen;

    filter_engine #(
        .SRC_WIDTH  (SRC_WIDTH),
        .SRC_HEIGHT (SRC_HEIGHT),
        .DISP_WIDTH (DISP_WIDTH)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (start),
        .mode       (mode),
        .busy       (busy),
        .done       (done),
        .src_addr   (src_addr),
        .src_en     (src_en),
        .src_dout   (src_dout),
        .dst_addr   (dst_addr),
        .dst_we     (dst_we),
        .dst_din    (dst_din)
    );

    always #5 clk = ~clk;   // 100 MHz

    // Source memory: read latency 1
    always @(posedge clk) begin
        if (src_en) begin
            if (src_addr[1:0] != 2'b00 || (src_addr >> 2) >= N_SRC)
                $display("ERROR: bad source address %h", src_addr);
            else
                src_dout <= src_mem[src_addr >> 2];
        end
    end

    // Display memory: write checks
    always @(posedge clk) begin
        if (rst_n && dst_we) begin
            if (!busy)
                $display("ERROR: write while not busy (address %h)", dst_addr);
            if (dst_addr[1:0] != 2'b00 || (dst_addr >> 2) >= N_DISP) begin
                $display("ERROR: bad display address %h", dst_addr);
            end else if (((dst_addr >> 2) % DISP_WIDTH) >= SRC_WIDTH) begin
                $display("ERROR: write in the text zone, word %0d", dst_addr >> 2);
            end else begin
                dst_mem[dst_addr >> 2] <= dst_din;
                writes[dst_addr >> 2] = writes[dst_addr >> 2] + 1;
            end
        end
    end

    // One run: start edge with mode m, wait for done; returns the cycle count
    task run(input integer m);
        begin
            for (i = 0; i < N_DISP; i = i + 1)
                writes[i] = 0;
            @(negedge clk);
            mode  = m;
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
            cycles = 1;
            busy_seen = 1'b0;
            timeout = 4 * N_SRC + 20 * SRC_WIDTH + 200;
            while (!done && cycles < timeout) begin
                if (busy) busy_seen = 1'b1;
                if (done && busy) $display("ERROR: done and busy together");
                // Second start edge while busy: must be ignored
                if (busy_start >= 0 && cycles == 4) begin
                    mode  = busy_start;
                    start = 1'b1;
                end
                if (busy_start >= 0 && cycles == 5) start = 1'b0;
                @(negedge clk);
                cycles = cycles + 1;
            end
            if (!done)      $display("ERROR: done never rose (mode %0d)", m);
            if (!busy_seen) $display("ERROR: busy never rose (mode %0d)", m);
            if (busy)       $display("ERROR: busy still high with done");
        end
    endtask

    initial begin
        if (!$value$plusargs("IN=%s", in_file) || !$value$plusargs("OUT=%s", out_file)
            || !$value$plusargs("N=%d", n) || !$value$plusargs("MODE=%d", mode1)) begin
            $display("ERROR: missing +IN, +OUT, +N or +MODE plusarg");
            $finish;
        end
        if (n != N_SRC) begin
            $display("ERROR: +N=%0d, expected %0d", n, N_SRC);
            $finish;
        end
        if (!$value$plusargs("SEED=%d", seed)) seed = 0;
        if (!$value$plusargs("MODE2=%d", mode2)) mode2 = -1;
        if (!$value$plusargs("GAP2=%d", gap2)) gap2 = 1;
        if (!$value$plusargs("BUSY_START=%d", busy_start)) busy_start = -1;

        if ($value$plusargs("VCD=%s", vcd_file)) begin
            $dumpfile(vcd_file);
            $dumpvars(0, tb_filter_engine);
        end

        $readmemh(in_file, src_mem, 0, n - 1);
        for (i = 0; i < N_DISP; i = i + 1)
            dst_mem[i] = MARK | i;

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        repeat (3) @(negedge clk);

        run(mode1);
        if (mode2 >= 0) begin
            busy_start = -1;
            repeat (gap2) @(negedge clk);
            run(mode2);
        end

        // Each image-zone word written exactly once in the last run
        for (i = 0; i < N_DISP; i = i + 1)
            if ((i % DISP_WIDTH) < SRC_WIDTH && writes[i] != 1)
                $display("ERROR: word %0d written %0d times", i, writes[i]);

        fd = $fopen(out_file, "w");
        for (i = 0; i < N_DISP; i = i + 1)
            $fwrite(fd, "%08x\n", dst_mem[i]);
        $fwrite(fd, "%08x\n", cycles);
        $fclose(fd);
        $display("DONE: %0d cycles", cycles);
        $finish;
    end

endmodule
