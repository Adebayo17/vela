`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: pattern_checkerboard
// Description: Red/Blue Checkerboard (8x8 pixel squares)
//////////////////////////////////////////////////////////////////////////////////


module pattern_checkerboard(
    input   wire [6:0] x,
    input   wire [5:0] y,
    output  reg [15:0] pixel_data
    );
    
    // RGB565 colors contants
    localparam RED_565   = 16'hF800;  // Pure RED
    localparam BLUE_565  = 16'h001F;  // Pure BLUE
    localparam WHITE_565 = 16'hFFFF;
    localparam BLACK_565 = 16'h0000;
    
    // Cell size (power of 2 for optimization)
    localparam CASE_SIZE = 8;  // 8x8 pixels
    
    always @(*) begin
        // Creating the checkerboard using the most significant bits
        // x[3] corresponds to x/8, y[3] corresponds to y/8
        if (x[CASE_SIZE-5] ^ y[CASE_SIZE-5])  // x[3] ^ y[3] for CASE_SIZE=8
            pixel_data = RED_565;
        else
            pixel_data = BLUE_565;
    end
endmodule
