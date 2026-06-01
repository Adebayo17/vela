`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: sobel_filter
// Description: Sobel filter with border management 
//////////////////////////////////////////////////////////////////////////////////


module sobel_filter #(
    parameter THRESHOLD = 80
)(
    input   wire        clk,
    input   wire        rst,
    // AXI Slave (Inputs from TPG)
    input   wire [15:0] s_axis_tdata,
    input   wire        s_axis_tvalid,
    output  wire        s_axis_tready,
    // AXI Master (Outputs to OLED)
    output  reg  [15:0] m_axis_tdata,
    output  reg         m_axis_tvalid,
    input   wire        m_axis_tready
    );
    
    localparam WIDTH    = 96;
    localparam HEIGHT   = 64;
    
    // Borders counters (Gérés en interne)
    reg [6:0] col_cnt;
    reg [5:0] row_cnt;
    
    // Conversion RGB565 -> Gray
    wire [7:0] r8 = {s_axis_tdata[15:11], s_axis_tdata[13:11]};
    wire [7:0] g8 = {s_axis_tdata[10:5],  s_axis_tdata[9:8]};
    wire [7:0] b8 = {s_axis_tdata[4:0],   s_axis_tdata[2:0]};
    wire [10:0] gray_sum = (r8 * 11'd2) + (g8 * 11'd5) + b8;
    wire [7:0] gray = gray_sum[10:3];
    
    // Line Buffers
    reg [7:0] line_buf_1 [0:WIDTH-1];
    reg [7:0] line_buf_2 [0:WIDTH-1];
    
    // Matrix 3x3
    reg [7:0] p11, p12, p13;
    reg [7:0] p21, p22, p23;
    reg [7:0] p31, p32, p33;
    
    // Pipeline Registers
    reg signed [10:0] gx, gy;
    reg               valid_pipeline;
    reg               is_border_stage1, is_border_stage2;
    
    // Freeze Logique
    wire pipeline_en = m_axis_tready || !m_axis_tvalid;
    assign s_axis_tready = pipeline_en;
    
    // Intermediate combinatorial calculations to secure the signed/unsigned
    wire [10:0] col_right  = p13 + (p23 << 1) + p33;
    wire [10:0] col_left   = p11 + (p21 << 1) + p31;
    wire [10:0] row_bottom = p31 + (p32 << 1) + p33;
    wire [10:0] row_top    = p11 + (p12 << 1) + p13;
    
    // Calculating the magnitude
    wire [10:0] abs_gx = gx[10] ? -gx : gx;
    wire [10:0] abs_gy = gy[10] ? -gy : gy;
    wire [11:0] magnitude = abs_gx + abs_gy; // 12-bits to avoid overflow
    
    always @(posedge clk) begin
        if (rst) begin
            col_cnt             <= 7'd0;
            row_cnt             <= 6'd0;
            m_axis_tvalid       <= 1'b0;
            m_axis_tdata        <= 16'h0000;
            valid_pipeline      <= 1'b0;
            {p11,p12,p13}       <= 0;
            {p21,p22,p23}       <= 0;
            {p31,p32,p33}       <= 0;
            {gx,gy}             <= 0;
            is_border_stage1    <= 1'b0;
            is_border_stage2    <= 1'b0;
            
        end else if (pipeline_en) begin
            
            // ----------------------------------------------------
            // STAGE 3 : Output
            // ----------------------------------------------------
            // Always Send the valid data.
            m_axis_tvalid <= valid_pipeline; 
            
            if (valid_pipeline) begin
                if (is_border_stage2) begin
                    m_axis_tdata <= 16'h0000; // Force to Black if it's an edge
                end else begin
                    m_axis_tdata <= (magnitude > THRESHOLD) ? 16'hFFFF : 16'h0000;
                end
            end
            
            // ----------------------------------------------------
            // STAGE 2 : Sobel Gradient Calculation
            // ----------------------------------------------------
            valid_pipeline   <= s_axis_tvalid;
            is_border_stage2 <= is_border_stage1;

            if (s_axis_tvalid) begin
                gx <= $signed({1'b0, col_right})  - $signed({1'b0, col_left});
                gy <= $signed({1'b0, row_bottom}) - $signed({1'b0, row_top});
            end
            
            // ----------------------------------------------------
            // STAGE 1 : Shift buffers & Position calculation
            // ----------------------------------------------------
            if (s_axis_tvalid) begin
                line_buf_2[col_cnt] <= line_buf_1[col_cnt];
                line_buf_1[col_cnt] <= gray;
                
                p11 <= p12; p12 <= p13; p13 <= line_buf_2[col_cnt];
                p21 <= p22; p22 <= p23; p23 <= line_buf_1[col_cnt];
                p31 <= p32; p32 <= p33; p33 <= gray;

                // We pre-calculate if we are on an edge to force it to be black later
                if (col_cnt < 2 || col_cnt == WIDTH - 1 || row_cnt < 2 || row_cnt == HEIGHT - 1)
                    is_border_stage1 <= 1'b1;
                else
                    is_border_stage1 <= 1'b0;

                // Internal counters (autonomous and synchronous)
                if (col_cnt == WIDTH - 1) begin
                    col_cnt <= 7'd0;
                    if (row_cnt == HEIGHT - 1) row_cnt <= 6'd0;
                    else row_cnt <= row_cnt + 1;
                end else begin
                    col_cnt <= col_cnt + 1;
                end
            end else begin
                is_border_stage1 <= 1'b0;
            end
        end
    end

endmodule




