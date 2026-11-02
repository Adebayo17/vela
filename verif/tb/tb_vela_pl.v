// tb_vela_pl.v — System testbench for vela_pl (filter engine + display path)
//
// Models the two Block Memory Generators on port B (read latency 1, byte
// addresses, READ_FIRST): source (SRC_WIDTH x SRC_HEIGHT words, from +IN) and
// display (DISP_WIDTH x SRC_HEIGHT words). The display memory starts with:
//   - image zone (x < SRC_WIDTH): 0x0000FFFF ("old" image, white);
//   - text zone  (x >= SRC_WIDTH): 0x0000_0800 + word index, as if the ARM
//     had written it through port A.
// At cycle +START_AT, ctrl = {mode, start}: the engine copies / filters the
// source into the image zone while display_reader streams the display memory
// to oled_controller (short delays). spi_monitor decodes the SPI bus.
//
// Output (+OUT, 48-bit hex lines, decoded by verif/models/oled_hw.py):
//   0 <dc> <byte> <cycle>   SPI byte (spi_monitor)
//   1 5 01 <cycle>          rising edge of status[1] (done)
//
// Checks: oled_cs always 0; status[7:2] always 0.
//
// Part of VELA — Vision Edge Learning Accelerator (verification only)

`timescale 1ns / 1ps

module tb_vela_pl #(
    parameter SRC_WIDTH  = 8,     // overridden with iverilog -P
    parameter SRC_HEIGHT = 6,
    parameter DISP_WIDTH = 12
);

    localparam N_SRC  = SRC_WIDTH * SRC_HEIGHT;
    localparam N_DISP = DISP_WIDTH * SRC_HEIGHT;

    reg         clk = 1'b0;
    reg         rst_n = 1'b0;
    reg  [7:0]  ctrl = 8'd0;
    wire [7:0]  status;
    wire        src_clk, src_rst, src_en, disp_clk, disp_rst, disp_en;
    wire [3:0]  src_we, disp_we;
    wire [31:0] src_addr, src_din, disp_addr, disp_din;
    reg  [31:0] src_dout = 32'd0, disp_dout = 32'd0;
    wire        oled_cs, oled_sclk, oled_mosi, oled_dc, oled_res, oled_vccen, oled_pmoden;

    reg  [31:0] src_mem  [0:N_SRC-1];
    reg  [31:0] disp_mem [0:N_DISP-1];
    reg  [1023:0] in_file, out_file, vcd_file;
    integer     n, i, mode, start_at, nbytes_exp, maxcyc, fd;
    reg  [31:0] cycle = 0, half = 0;
    wire [31:0] nbytes, errors;
    reg         done_q = 1'b0;

    vela_pl #(
        .CLK_FREQ          (100_000_000),
        .SPI_FREQ          (6_250_000),
        .POWER_UP_CYCLES   (200),
        .RESET_CYCLES      (50),
        .VCC_ON_CYCLES     (300),
        .DISPLAY_ON_CYCLES (400),
        .SRC_WIDTH         (SRC_WIDTH),
        .SRC_HEIGHT        (SRC_HEIGHT),
        .DISP_WIDTH        (DISP_WIDTH),
        .DISP_HEIGHT       (SRC_HEIGHT)
    ) dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .ctrl        (ctrl),
        .status      (status),
        .src_clk     (src_clk),
        .src_rst     (src_rst),
        .src_en      (src_en),
        .src_we      (src_we),
        .src_addr    (src_addr),
        .src_din     (src_din),
        .src_dout    (src_dout),
        .disp_clk    (disp_clk),
        .disp_rst    (disp_rst),
        .disp_en     (disp_en),
        .disp_we     (disp_we),
        .disp_addr   (disp_addr),
        .disp_din    (disp_din),
        .disp_dout   (disp_dout),
        .oled_cs     (oled_cs),
        .oled_sclk   (oled_sclk),
        .oled_mosi   (oled_mosi),
        .oled_dc     (oled_dc),
        .oled_res    (oled_res),
        .oled_vccen  (oled_vccen),
        .oled_pmoden (oled_pmoden)
    );

    spi_monitor mon (
        .clk    (clk),
        .rst_n  (rst_n),
        .sclk   (oled_sclk),
        .mosi   (oled_mosi),
        .dc     (oled_dc),
        .cycle  (cycle),
        .half   (half),
        .fd     (fd),
        .nbytes (nbytes),
        .errors (errors)
    );

    always #5 clk = ~clk;   // 100 MHz

    // Memories, port B: read latency 1, READ_FIRST (old data on a write)
    always @(posedge clk) begin
        if (src_en) begin
            if (src_we != 4'd0) $display("ERROR: write on the source port");
            if ((src_addr >> 2) >= N_SRC) $display("ERROR: bad source address %h", src_addr);
            else src_dout <= src_mem[src_addr >> 2];
        end
        if (disp_en) begin
            if ((disp_addr >> 2) >= N_DISP) begin
                $display("ERROR: bad display address %h", disp_addr);
            end else begin
                disp_dout <= disp_mem[disp_addr >> 2];
                if (disp_we == 4'hF)
                    disp_mem[disp_addr >> 2] <= disp_din;
                else if (disp_we != 4'h0)
                    $display("ERROR: partial write enable %b", disp_we);
            end
        end
    end

    always @(posedge clk) begin
        cycle <= cycle + 1;
        if (rst_n) begin
            if (oled_cs !== 1'b0)          $display("ERROR: oled_cs not tied low");
            if (status[7:2] !== 6'd0)      $display("ERROR: status[7:2] = %b", status[7:2]);
            if (status[1] && !done_q)      $fwrite(fd, "1501%08x\n", cycle);
            done_q <= status[1];
        end
    end

    initial begin
        if (!$value$plusargs("IN=%s", in_file) || !$value$plusargs("OUT=%s", out_file)
            || !$value$plusargs("N=%d", n) || !$value$plusargs("MODE=%d", mode)
            || !$value$plusargs("START_AT=%d", start_at) || !$value$plusargs("NBYTES=%d", nbytes_exp)
            || !$value$plusargs("HALF=%d", half) || !$value$plusargs("MAXCYC=%d", maxcyc)) begin
            $display("ERROR: missing plusarg (+IN +OUT +N +MODE +START_AT +NBYTES +HALF +MAXCYC)");
            $finish;
        end
        if (n != N_SRC) begin
            $display("ERROR: +N=%0d, expected %0d", n, N_SRC);
            $finish;
        end

        if ($value$plusargs("VCD=%s", vcd_file)) begin
            $dumpfile(vcd_file);
            $dumpvars(0, tb_vela_pl);
        end

        $readmemh(in_file, src_mem, 0, n - 1);
        for (i = 0; i < N_DISP; i = i + 1)
            disp_mem[i] = ((i % DISP_WIDTH) < SRC_WIDTH) ? 32'h0000_FFFF : (32'h0000_0800 + i);
        fd = $fopen(out_file, "w");

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        // Start pulse (2 cycles, like a GPIO write), mode kept afterwards
        while (cycle < start_at) @(negedge clk);
        ctrl = {4'd0, mode[2:0], 1'b1};
        repeat (2) @(negedge clk);
        ctrl = {4'd0, mode[2:0], 1'b0};

        while (nbytes < nbytes_exp && cycle < maxcyc) @(posedge clk);
        repeat (2) @(posedge clk);

        $fclose(fd);
        if (nbytes < nbytes_exp)
            $display("ERROR: %0d SPI bytes, expected %0d", nbytes, nbytes_exp);
        else if (errors != 0)
            $display("ERROR: %0d SPI protocol errors", errors);
        else
            $display("DONE: %0d bytes", nbytes);
        $finish;
    end

endmodule
