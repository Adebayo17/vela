// oled_controller.v — Pmod OLEDrgb (SSD1331, 96x64) controller
//
// Power-up and init sequence, then streams RGB565 pixels received over
// AXI-Stream to the display through spi_master (2 bytes per pixel, high
// byte first, oled_dc = 1).
//
//   1. power on (pmoden, vccen), wait POWER_UP_CYCLES  (20 ms)
//   2. oled_res low for RESET_CYCLES                   (100 us), then high
//   3. wait RESET_CYCLES, send the 31 init bytes of init_rom (oled_dc = 0)
//   4. s_axis_tready = 1: accept a pixel, send its 2 bytes, repeat
//
// A pixel is accepted on a rising edge where s_axis_tvalid && s_axis_tready.
// Throughput is limited by SPI: 2 bytes x 17 half SCLK periods per pixel.
//
// Delays are derived from CLK_FREQ; POWER_UP_CYCLES and RESET_CYCLES can be
// overridden (tests use short values to keep simulations fast).
//
// Verified by: verif/tests/test_rtl_oled_controller.py
//
// Part of VELA — Vision Edge Learning Accelerator

`timescale 1ns / 1ps

module oled_controller #(
    parameter CLK_FREQ        = 100_000_000,       // system clock (VELA-NFR-001)
    parameter SPI_FREQ        = 6_250_000,         // SPI clock
    parameter POWER_UP_CYCLES = CLK_FREQ / 50,     // 20 ms
    parameter RESET_CYCLES    = CLK_FREQ / 10_000  // 100 us
)(
    input   wire        clk,
    input   wire        rst_n,              // synchronous, active low
    
    // AXI-Stream Slave Interface (Pixels Input)
    input   wire [15:0] s_axis_tdata,       // Pixel RGB565 (16-bit)
    input   wire        s_axis_tvalid,      // "Here is a pixel"
    output  reg         s_axis_tready,      // "I'm ready"
    
    // Physical signals to Pmod OLEDrgb
    output  wire        oled_spi_sclk,
    output  wire        oled_spi_mosi,
    output  reg         oled_dc,            // 0=Cmd, 1=Data
    output  reg         oled_res,           // Reset (Active Low)
    output  reg         oled_vccen,         // Power Enable
    output  reg         oled_pmoden         // Power Enable
    );
    
    // --- 1. FSM States ---
    localparam S_POWER_UP   = 0;
    localparam S_RESET      = 1;
    localparam S_INIT       = 2;
    localparam S_WAIT_DATA  = 3;    // Wait AXI data
    localparam S_SEND_HIGH  = 4;    // Send HIGH BYTE 
    localparam S_WAIT_HIGH  = 5;    // Wait for SPI to finish
    localparam S_SEND_LOW   = 6;    // Send LOW BYTE
    localparam S_WAIT_LOW   = 7;    // Wait for SPI to finish
    
    
    localparam DW = $clog2(POWER_UP_CYCLES + 1);   // delay counter width

    /* verilator lint_off WIDTHTRUNC */
    localparam [DW-1:0] POWER_UP_DLY = POWER_UP_CYCLES;   // fit: DW from $clog2
    localparam [DW-1:0] RESET_DLY    = RESET_CYCLES;
    /* verilator lint_on WIDTHTRUNC */

    reg [2:0]    state;
    reg [15:0]   pixel_buffer;      // Store pixel to send
    reg [DW-1:0] delay_cnt;
    
    // --- 2. ROM Initialization (SSD1331 commands) ---
    // Standard sequence for OLED RGB 96*64
    localparam ROM_SIZE = 31;
    reg [7:0] init_rom [0:ROM_SIZE-1];
    reg [4:0] rom_index;
    initial begin 
        init_rom[0]  = 8'hAE; // Display OFF
        init_rom[1]  = 8'hA0; // Set Remap
        init_rom[2]  = 8'h72; // RGB Color (Format 65k couleurs)
        init_rom[3]  = 8'hA1; // Set Start Line
        init_rom[4]  = 8'h00;
        init_rom[5]  = 8'hA2; // Set Display Offset
        init_rom[6]  = 8'h00;
        init_rom[7]  = 8'hA4; // Normal Display
        init_rom[8]  = 8'hA8; // Set Multiplex Ratio
        init_rom[9]  = 8'h3F; // 64 MUX
        init_rom[10] = 8'hAD; // Master Config
        init_rom[11] = 8'h8E; // External VCC
        init_rom[12] = 8'hB0; // Power Save Mode
        init_rom[13] = 8'h0B; // Disable
        init_rom[14] = 8'hB1; // Phase Period
        init_rom[15] = 8'h31; 
        init_rom[16] = 8'hB3; // Clock Divider
        init_rom[17] = 8'hF0; 
        init_rom[18] = 8'h8A; // Pre-charge A
        init_rom[19] = 8'h64;
        init_rom[20] = 8'h8B; // Pre-charge B
        init_rom[21] = 8'h78;
        init_rom[22] = 8'h8C; // Pre-charge C
        init_rom[23] = 8'h64;
        
        // --- NOUVEAUX REGLAGES GEOMETRIE ---
        init_rom[24] = 8'h15; // Set Column Address
        init_rom[25] = 8'h00; // Start = 0
        init_rom[26] = 8'h5F; // End = 95 (0x5F)
        
        init_rom[27] = 8'h75; // Set Row Address
        init_rom[28] = 8'h00; // Start = 0
        init_rom[29] = 8'h3F; // End = 63 (0x3F)
        
        init_rom[30] = 8'hAF; // Display ON (Last command!)
    end 
    
    // --- 3. SPI MASTER Instance ---
    reg        spi_start;
    reg [7:0]  spi_data;
    wire       spi_busy;
    wire       spi_done;

    spi_master #(
        .CLK_FREQ(CLK_FREQ),
        .SPI_FREQ(SPI_FREQ)
    ) u_spi (
        .clk(clk),
        .rst_n(rst_n),
        .start(spi_start),
        .data_in(spi_data),
        .busy(spi_busy),
        .done_tick(spi_done),
        .sclk(oled_spi_sclk),
        .mosi(oled_spi_mosi)
    );
    
    // --- 4. FSM Logic ---
    always @(posedge clk) begin
        if (!rst_n) begin
            state           <= S_POWER_UP;
            oled_vccen      <= 0;
            oled_pmoden     <= 0;
            oled_res        <= 1;       // Reset is Active-Low, so 1 = inactif
            oled_dc         <= 0;       // Default: Mode Commande 
            rom_index       <= 0;
            delay_cnt       <= POWER_UP_DLY;    // 20 ms wait at power up
            spi_start       <= 0;
            s_axis_tready   <= 0;      // Upper Byte 
        end else begin
            // Timer management 
            if (delay_cnt != {DW{1'b0}}) delay_cnt <= delay_cnt - 1'b1;
            
            // SPI Pulse management (1 cycle)
            spi_start <= 0; 

            case (state)
                // --- Init Sequence ---
                // Step A : Turn on the power supplies
                S_POWER_UP: begin
                    oled_vccen  <= 1;
                    oled_pmoden <= 1;
                    if (delay_cnt == {DW{1'b0}}) begin
                        state <= S_RESET;
                        delay_cnt <= RESET_DLY; // reset pulse (100 us)
                        oled_res <= 0;      // Activate Reset
                    end
                end

                // Step B : Hardware Reset
                S_RESET: begin
                    if (delay_cnt == {DW{1'b0}}) begin
                        oled_res <= 1;      // Release Reset
                        state <= S_INIT;
                        delay_cnt <= RESET_DLY; // post-reset wait (100 us)
                    end
                end

                // Step C : Send Init Sequence
                S_INIT: begin
                    // We wait until the post-reset delay is over AND the SPI is free
                    if (delay_cnt == {DW{1'b0}} && !spi_busy && !spi_start) begin
                        if (rom_index < ROM_SIZE) begin
                            oled_dc   <= 0;                     // It's a Command
                            spi_data  <= init_rom[rom_index];
                            spi_start <= 1;                     // Start sending
                            rom_index <= rom_index + 1;
                        end else begin
                            s_axis_tready <= 1;
                            state <= S_WAIT_DATA;               // Init terminated!
                        end
                    end
                end

                // --- Actif Mode : AXI Stream ---
                S_WAIT_DATA: begin
                    s_axis_tready <= 1;
                    oled_dc   <= 1; // Mode Data
                    
                    if (s_axis_tvalid) begin
                        pixel_buffer    <= s_axis_tdata;
                        s_axis_tready   <= 0;
                        state           <= S_SEND_HIGH; 
                    end 
                end
                
                S_SEND_HIGH: begin
                    if (!spi_busy) begin 
                        spi_data    <= pixel_buffer[15:8];  // Byte 1 (RRRRRGGG)
                        spi_start   <= 1;
                        state       <= S_WAIT_HIGH;
                    end 
                end
                
                S_WAIT_HIGH: begin
                    // We wait for SPI to finish sending the 1st byte
                    if (spi_done) begin
                        state <= S_SEND_LOW;
                    end
                end

                S_SEND_LOW: begin
                    if (!spi_busy) begin
                        spi_data    <= pixel_buffer[7:0]; // Byte 2 (GGGBBBBB)
                        spi_start   <= 1;
                        state       <= S_WAIT_LOW;
                    end
                end

                S_WAIT_LOW: begin
                    // On attend la fin de l'octet bas
                    if (spi_done) begin
                        s_axis_tready <= 1;
                        state <= S_WAIT_DATA; // Pixel finished, next!
                    end
                end
            endcase
        end
    end 
endmodule
