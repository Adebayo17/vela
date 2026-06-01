`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: spi_master
// Description: SPI Master to communicate with the Pmod OLEDrgb module
//////////////////////////////////////////////////////////////////////////////////


module spi_master#(
    parameter CLK_FREQ  = 125_000_000, // System Clock
    parameter SPI_FREQ  = 6_250_000    // SPI target (6.25 MHz = 125/20)
)(
    input   wire        clk,
    input   wire        rst,
    
    // Control Interface (FPGA)
    input   wire        start,          // Pulse to start sending
    input   wire [7:0]  data_in,        // Byte to send
    output  reg         busy,           // '1' during sending
    output  reg         done_tick,      // Pulse at the end of sending
    
    // Physical Interface (to Pmod)
    output  reg         sclk,
    output  reg         mosi
    );
    
    // --- 1. SPI Clock Generator ---
    // We want to generate SCLK. We count the half-periods.
    localparam DIVIDER = (CLK_FREQ / SPI_FREQ) / 2;
    
    reg [$clog2(DIVIDER)-1:0]   clk_cnt;
    reg                         spi_clk_en; // Enable pulse to change the SPI state
    
    always @(posedge clk) begin 
        if (rst) begin
            clk_cnt     <= 0;
            spi_clk_en  <= 0;
        end else begin 
            if (busy) begin 
                if (clk_cnt == DIVIDER - 1) begin 
                    clk_cnt     <= 0;
                    spi_clk_en  <= 1;
                end else begin
                    clk_cnt     <= clk_cnt + 1;
                    spi_clk_en  <= 0;
                end 
            end else begin 
                clk_cnt     <= 0;
                spi_clk_en  <= 0;
            end 
        end 
    end 
    
    // --- 2. SPI FSM States ---
    // We want to manage 8 bits. Each bit has 2 phases (LOW, HIGH on SCLK)
    // So 16 states on SCLK
    
    reg [4:0] bit_cnt;  // Count 0...16 (16=end)
    reg [7:0] shift_reg;
    
    always @(posedge clk) begin 
        if (rst) begin 
            busy        <= 0;
            done_tick   <= 0;
            mosi        <= 0;
            sclk        <= 0;   // Mode 0: IDLE Low
            bit_cnt     <= 0;
            shift_reg   <= 0;
        end else begin 
            done_tick   <= 0;   // Default
            
            if (!busy && start) begin 
                // Starting
                busy        <= 1;
                shift_reg   <= data_in;
                bit_cnt     <= 0;
                mosi        <= data_in[7];  // MSB First
                sclk        <= 0;           // 1st Edge is Rising  
            end else if (busy && spi_clk_en) begin
                // SCLK and MOSI toggle logic
                // bit_cnt 0 -> SCLK goes up   (Capture)
                // bit_cnt 1 -> SCLK goes down (Change Data)
                // ...
                
                if (bit_cnt == 16) begin 
                    // End of Transmission
                    busy        <= 0;
                    done_tick   <= 1;
                    sclk        <= 0;
                end else begin 
                    // SPI clock toggle
                    sclk <= ~sclk;
                    
                    // If Falling Edge, we change data
                    if (sclk == 1'b1) begin // If was HIGH, it goes down
                        if (bit_cnt < 15) begin 
                            mosi        <= shift_reg[6];    // Extract next bit
                            shift_reg   <= {shift_reg[6:0], 1'b0};
                        end 
                    end
                    
                    bit_cnt <= bit_cnt + 1; 
                end 
            end 
        end 
    end 
endmodule









