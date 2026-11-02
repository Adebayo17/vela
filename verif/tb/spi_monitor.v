// spi_monitor.v — Passive SPI (mode 0) monitor for testbenches
//
// Watches sclk / mosi / dc as the slave (SSD1331) would and:
//   - rebuilds each byte (MSB first, sampled on the rising edge of sclk) and
//     writes it to `fd` as one 48-bit hex line:
//         0 <dc:4> <byte:8> <cycle:32>      cycle = first rising edge of the byte
//   - checks mode 0: mosi must not change while sclk is high;
//   - checks that dc is constant during a byte;
//   - checks every sclk half period inside a byte against `half` (in clk
//     cycles), given by the test from the specified SCLK frequency.
// Each violation prints "ERROR: ..." and increments `errors`.
//
// sclk is a register clocked by clk in the DUT, so sampling it on clk
// rising edges sees every transition exactly.
//
// Part of VELA — Vision Edge Learning Accelerator (verification only)

`timescale 1ns / 1ps

module spi_monitor (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        sclk,
    input  wire        mosi,
    input  wire        dc,
    input  wire [31:0] cycle,      // testbench cycle counter (timestamps)
    input  wire [31:0] half,       // expected sclk half period, in clk cycles
    input  wire [31:0] fd,         // output file descriptor
    output reg  [31:0] nbytes,     // bytes decoded so far
    output reg  [31:0] errors      // protocol violations so far
);

    reg        sclk_q, mosi_q, dc0;
    reg  [3:0] nbits;
    reg  [7:0] shifter;
    reg [31:0] since_edge, byte_cycle;

    initial begin
        sclk_q = 1'b0; mosi_q = 1'b0; dc0 = 1'b0;
        nbits = 4'd0; shifter = 8'd0; since_edge = 0; byte_cycle = 0;
        nbytes = 0; errors = 0;
    end

    always @(posedge clk) begin
        if (rst_n) begin
            since_edge = since_edge + 1;

            // Mode 0: data must be stable while sclk is high
            if (sclk_q && sclk && mosi !== mosi_q) begin
                errors = errors + 1;
                $display("ERROR: MOSI changed while SCLK high (cycle %0d)", cycle);
            end

            if (sclk !== sclk_q) begin
                // Half period check, except for the first rising edge of a byte
                if (!(sclk && nbits == 0) && since_edge != half) begin
                    errors = errors + 1;
                    $display("ERROR: SCLK half period %0d cycles, expected %0d (cycle %0d)",
                             since_edge, half, cycle);
                end
                since_edge = 0;

                if (sclk) begin                         // rising edge: sample
                    if (nbits == 0) begin
                        dc0 = dc;
                        byte_cycle = cycle;
                    end else if (dc !== dc0) begin
                        errors = errors + 1;
                        $display("ERROR: DC changed during a byte (cycle %0d)", cycle);
                    end
                    shifter = {shifter[6:0], mosi};
                    nbits = nbits + 1;
                    if (nbits == 8) begin
                        $fwrite(fd, "0%01x%02x%08x\n", dc0, shifter, byte_cycle);
                        nbytes = nbytes + 1;
                        nbits = 0;
                    end
                end
            end
        end
        sclk_q = sclk;
        mosi_q = mosi;
    end

endmodule
