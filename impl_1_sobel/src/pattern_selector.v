`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: pattern_selector
// Description: Select Pattern to display 
// Pattern sel : 
// - 00: Checkerboard RB
// - 01: Benin Flag
//////////////////////////////////////////////////////////////////////////////////


module pattern_selector(
    input   wire        clk,
    input   wire        rst,
    input   wire [1:0]  pattern_sel,        // 00: Checkerboard RB, 01: Benin Flag, ...
    // AXI-Stream Master Interface (Output to OLED)
    output  reg  [15:0] m_axis_tdata,       // Pixel RGB565 (16-bit)
    output  reg         m_axis_tvalid,      // "I have a Pixel ready"
    input   wire        m_axis_tready,      // "Is the Screen Ready?"
    
    // Synchronization signals
    output  reg         sof,                // Start of Frame
    output  reg         eol                 // End of Line
    );
    
    // Parameters
    localparam WIDTH    = 96;
    localparam HEIGHT   = 64;
    localparam H_TIER   = 32;   // WIDTH/3 for flag
    localparam V_TIER   = 32;   // HEIGHT/2 for flag
    
    // Position counter
    reg [6:0] x_cnt;    // 0...95
    reg [5:0] y_cnt;    // 0...63
    
    // Pattern signals
    wire [15:0] checkerboard_data;
    wire [15:0] bflag_data;
    
    // Pattern Generation
    pattern_checkerboard u_checker (
        .x(x_cnt),
        .y(y_cnt),
        .pixel_data(checkerboard_data)
    );
    
    pattern_benin_flag u_bflag(
        .x(x_cnt),
        .y(y_cnt),
        .tiers_x(H_TIER),
        .half_y(V_TIER),
        .pixel_data(bflag_data)
    );
    
    // Main Process
    always @(posedge clk) begin
        if (rst) begin
            x_cnt           <= 0;
            y_cnt           <= 0;
            m_axis_tvalid   <= 0;
            m_axis_tdata    <= 16'h0000;
            sof             <= 0;
            eol             <= 0;
        end else begin 
            // default
            sof <= 0;
            eol <= 0;
            
            // AXI-Stream management
            if (m_axis_tready || !m_axis_tvalid) begin
                m_axis_tvalid <= 1'b1;
                
                // Pattern selection
                case (pattern_sel)
                    2'b00: m_axis_tdata <= checkerboard_data;
                    2'b01: m_axis_tdata <= bflag_data;
                    default: m_axis_tdata <= 16'h0000;
                endcase
                
                // Counters management
                if (x_cnt == WIDTH - 1) begin
                    x_cnt <= 0;
                    eol <= 1'b1;
                    
                    if (y_cnt == HEIGHT - 1) begin
                        y_cnt <= 0;
                        sof <= 1'b1;
                    end else begin
                        y_cnt <= y_cnt + 1;
                    end
                end else begin
                    x_cnt <= x_cnt + 1;
                end
            end
            
            // If no one is ready, we keep the data
            if (!m_axis_tready) begin
                m_axis_tvalid <= 1'b1;  // Keep valid data
            end
        end 
    end 
endmodule
