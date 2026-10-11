// spi_master.v — SPI master (mode 0, MSB first) for the Pmod OLEDrgb
//
// Sends one byte per `start` pulse:
//   - mode 0: SCLK idle low, MOSI set up before the rising edge (sampled by
//     the slave), changed after the falling edge;
//   - SCLK frequency = CLK_FREQ / (2 * DIVIDER), DIVIDER = CLK_FREQ / SPI_FREQ / 2
//     (100 MHz / 6.25 MHz -> DIVIDER = 8, 16 clock cycles per SCLK period);
//   - `busy` is high from the cycle after `start` until the end of the byte;
//     `done_tick` pulses for one cycle at the end.
// `start` is ignored while `busy` is high.
//
// SSD1331 serial clock limit: 150 ns min cycle (~6.67 MHz): 6.25 MHz is valid.
//
// Verified by: verif/tests/test_rtl_spi_master.py
//
// Part of VELA — Vision Edge Learning Accelerator

`timescale 1ns / 1ps

module spi_master #(
    parameter CLK_FREQ = 100_000_000,   // system clock (VELA-NFR-001)
    parameter SPI_FREQ = 6_250_000      // SPI target
)(
    input  wire        clk,
    input  wire        rst_n,           // synchronous, active low

    // Control interface
    input  wire        start,           // pulse: send data_in
    input  wire [7:0]  data_in,         // byte to send
    output reg         busy,            // high during the transfer
    output reg         done_tick,       // 1-cycle pulse at the end

    // Physical interface (to Pmod)
    output reg         sclk,
    output reg         mosi
);

    // ── SCLK generator: one enable pulse per half SCLK period ──────────────

    localparam DIVIDER = (CLK_FREQ / SPI_FREQ) / 2;
    localparam CW      = (DIVIDER > 1) ? $clog2(DIVIDER) : 1;

    /* verilator lint_off WIDTHTRUNC */
    localparam [CW-1:0] DIV_LAST = DIVIDER - 1;   // fits: CW = $clog2(DIVIDER)
    /* verilator lint_on WIDTHTRUNC */

    reg [CW-1:0] clk_cnt;
    reg          spi_clk_en;

    always @(posedge clk) begin
        if (!rst_n) begin
            clk_cnt    <= {CW{1'b0}};
            spi_clk_en <= 1'b0;
        end else if (busy) begin
            if (clk_cnt == DIV_LAST) begin
                clk_cnt    <= {CW{1'b0}};
                spi_clk_en <= 1'b1;
            end else begin
                clk_cnt    <= clk_cnt + 1'b1;
                spi_clk_en <= 1'b0;
            end
        end else begin
            clk_cnt    <= {CW{1'b0}};
            spi_clk_en <= 1'b0;
        end
    end

    // ── Bit sequencer: 8 bits x 2 SCLK phases = 16 steps, step 16 = end ────

    reg [4:0] bit_cnt;
    reg [6:0] shift_reg;    // remaining bits; the MSB goes to mosi at start

    always @(posedge clk) begin
        if (!rst_n) begin
            busy      <= 1'b0;
            done_tick <= 1'b0;
            mosi      <= 1'b0;
            sclk      <= 1'b0;          // mode 0: idle low
            bit_cnt   <= 5'd0;
            shift_reg <= 7'd0;
        end else begin
            done_tick <= 1'b0;

            if (!busy && start) begin
                busy      <= 1'b1;
                shift_reg <= data_in[6:0];
                bit_cnt   <= 5'd0;
                mosi      <= data_in[7];    // MSB first, set up before 1st rising edge
                sclk      <= 1'b0;
            end else if (busy && spi_clk_en) begin
                if (bit_cnt == 5'd16) begin
                    busy      <= 1'b0;      // end of transfer
                    done_tick <= 1'b1;
                    sclk      <= 1'b0;
                end else begin
                    sclk <= ~sclk;
                    // falling edge (sclk was high): present the next bit
                    if (sclk && bit_cnt < 5'd15) begin
                        mosi      <= shift_reg[6];
                        shift_reg <= {shift_reg[5:0], 1'b0};
                    end
                    bit_cnt <= bit_cnt + 1'b1;
                end
            end
        end
    end

endmodule
