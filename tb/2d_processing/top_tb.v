`timescale 1ns / 1ps

module top_test();
localparam PIXW = 8;
localparam Output_Width = 32;
localparam K_size = 3;
localparam LB_size = 20;

reg clk_50M, rst;
reg master_start;
reg [(K_size * K_size * PIXW)-1 : 0] weights;
reg [(K_size * K_size * Output_Width)-1 : 0] bias;
reg WVALID1, WVALID2, WVALID3, WVALID4;
reg [PIXW-1:0] inp_pix1, inp_pix2, inp_pix3, inp_pix4;
wire [Output_Width-1 : 0] out;

// Storage array to capture outputs
reg [Output_Width-1:0] out_capture [0:LB_size-1];
integer capture_idx;

Pipelining #(
    .PIXW(PIXW),
    .Output_Width(Output_Width),
    .K_size(K_size)
) uut (
    .clk_50M(clk_50M),
    .rst(rst),
    .master_start(master_start),
    .WVALID1(WVALID1), .WVALID2(WVALID2),
    .WVALID3(WVALID3), .WVALID4(WVALID4),
    .weights(weights), .bias(bias),
    .inp_pix1(inp_pix1), .inp_pix2(inp_pix2),
    .inp_pix3(inp_pix3), .inp_pix4(inp_pix4),
    .out(out)
);

// Clock
always #10 clk_50M = ~clk_50M;

// Capture output every time vec_valid goes high
// We monitor out changing as proxy since vec_done comes 1 cycle after vec_valid
always @(posedge clk_50M) begin
    if(out !== 0 && out !== out_capture[capture_idx]) begin
        out_capture[capture_idx] <= out;
        capture_idx <= capture_idx + 1;
    end
end

integer i;
initial begin
    // -------------------------------------
    // Init everything
    // -------------------------------------
    clk_50M     = 1'b0;
    rst         = 1'b0;
    master_start= 1'b0;
    WVALID1     = 1'b0;
    WVALID2     = 1'b0;
    WVALID3     = 1'b0;
    WVALID4     = 1'b0;
    inp_pix1    = 0;
    inp_pix2    = 0;
    inp_pix3    = 0;
    inp_pix4    = 0;
    capture_idx = 0;

    // -------------------------------------
    // Set weights - all 1s (identity-like)
    // kernel: 3x3 = 9 weights, each PIXW bits
    // flat order: w[0] is top-left, w[8] is bottom-right
    // -------------------------------------
    weights = {
        8'd1, 8'd1, 8'd1,
        8'd1, 8'd1, 8'd1,
        8'd1, 8'd1, 8'd1
    };

    // -------------------------------------
    // Set bias - all 0s for simplicity
    // 9 bias values each Output_Width bits
    // -------------------------------------
    bias = 0;

    // -------------------------------------
    // Release reset after 2 cycles
    // -------------------------------------
    repeat(2) @(posedge clk_50M);
    rst = 1'b1;

    // -------------------------------------
    // Assert master_start
    // -------------------------------------
    @(posedge clk_50M);
    #1; // Simulate AXI register update delay
    master_start = 1'b1;
    @(posedge clk_50M);
    #1;
    // -------------------------------------
    // Feed pixels - 20 pixels per row
    // Row 1 (LB1): pixels 1..20
    // Row 2 (LB2): pixels 21..40
    // Row 3 (LB3): pixels 41..60
    // Row 4 (LB4): pixels 61..80
    // All 4 rows sent simultaneously each cycle
    // -------------------------------------
    WVALID1 = 1'b1;
    WVALID2 = 1'b1;
    WVALID3 = 1'b1;
    WVALID4 = 1'b1;

    for(i = 0; i < LB_size; i = i + 1) begin
        inp_pix1 = i + 1;         // row 0: 1,2,3...20
        inp_pix2 = i + 21;        // row 1: 21,22...40
        inp_pix3 = i + 41;        // row 2: 41,42...60
        inp_pix4 = i + 61;        // row 3: 61,62...80
        @(posedge clk_50M);
        #1;
    end

    // -------------------------------------
    // Done sending, deassert WVALID
    // -------------------------------------
    WVALID1 = 1'b0;
    WVALID2 = 1'b0;
    WVALID3 = 1'b0;
    WVALID4 = 1'b0;

    // -------------------------------------
    // Wait for all convolution outputs
    // LB_size - K_size = 17 valid strides
    // Each takes 2 cycles (vec engine delay)
    // So wait 17 * 2 + some margin
    // -------------------------------------
    repeat(50) @(posedge clk_50M);

    // -------------------------------------
    // Print captured results
    // -------------------------------------
    $display("--- Convolution Outputs ---");
    for(i = 0; i < capture_idx; i = i + 1) begin
        $display("Output[%0d] = %0d", i, out_capture[i]);
    end

    $finish;
end

endmodule
