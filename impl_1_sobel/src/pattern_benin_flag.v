`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: pattern_benin_flag
// Description: Flag of Benin
// - Vertical green stripe on the left (1/3) 
// - Horisontal yellow stripe at the top right (2/3) 
// - Horizontal red stripe at the bottom right (2/3)
//////////////////////////////////////////////////////////////////////////////////


module pattern_benin_flag(
    input   wire [6:0] x,
    input   wire [5:0] y,
    input   wire [6:0] tiers_x,     // 1/3 of WIDTH
    input   wire [5:0] half_y,      // 1/2 of HEIGHT
    output  reg  [15:0] pixel_data
    );
    
    // Official colors of the Beninese flag in RGB565
    // Green: #008751 -> R=0, G=135, B=81
    localparam GREEN_565  = {5'h00, 6'h21, 5'h11};  // Approximately 16'h0431
    // Yellow: #FCD116 -> R=252, G=209, B=22  
    localparam YELLOW_565 = {5'h1F, 6'h34, 5'h02};  // Approximately 16'hFD02  
    // Red: #E8112D -> R=232, G=17, B=45
    localparam RED_565    = {5'h1D, 6'h04, 5'h05};  // Approximately 16'hE905
    
    always @(*) begin
        if (x < tiers_x && x >= 0) begin
            // Vertical green stripe (1/3 left)
            pixel_data = GREEN_565;
        end else begin
            // Right part (2/3)
            if (y < half_y && y >= 0) begin
                // Upper Half : Yellow
                pixel_data = YELLOW_565;
            end else begin
                // Bottom Half : Red
                pixel_data = RED_565;
            end
        end
    end
endmodule
