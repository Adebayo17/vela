// threshold.v — Binary threshold filter
// Output = (pixel > threshold) ? 255 : 0
//
// Latency: 1 clock cycle (registered output).
//
// Bit-exact reference model: verif/models/filters_hw.py::threshold_u8
//
// Part of VELA — Vision Edge Learning Accelerator
// Phase 2a: Verilog RTL image filters

module threshold #(
    parameter [7:0] THRESHOLD = 8'd128
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [7:0]  pixel_in,
    output reg         valid_out,
    output reg  [7:0]  pixel_out
);

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            pixel_out <= 8'd0;
        end else begin
            valid_out <= valid_in;
            if (valid_in)
                pixel_out <= (pixel_in > THRESHOLD) ? 8'd255 : 8'd0;
        end
    end

endmodule
