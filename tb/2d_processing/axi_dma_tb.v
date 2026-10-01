`timescale 1ns / 1ps

module AXI_DMA_test();
localparam PIXW         = 8;
localparam Output_Width = 32;
localparam K_size       = 3;
localparam LB_size      = 20;

reg clk_50M, rst, master_start;
reg [(K_size*K_size*PIXW)-1:0]         weights;
reg [(K_size*K_size*Output_Width)-1:0] bias;
reg [PIXW-1:0] s_axis_tdata;
reg s_axis_tvalid, s_axis_tlast;
wire s_axis_tready;
wire [Output_Width-1:0] out;

// INCREASED SIZE: 20 slots is too small for multiple rows of convolutions!
reg [Output_Width-1:0] out_capture [0:199]; 
integer capture_idx;
integer i;

Pipelining #(.PIXW(PIXW), .Output_Width(Output_Width),
             .K_size(K_size), .Lb_size(LB_size)) uut
    (.clk_50M(clk_50M), .rst(rst), .master_start(master_start),
     .s_axis_tdata(s_axis_tdata), .s_axis_tvalid(s_axis_tvalid),
     .s_axis_tlast(s_axis_tlast), .s_axis_tready(s_axis_tready),
     .weights(weights), .bias(bias), .out(out));

always #10 clk_50M = ~clk_50M;

// capture outputs
always @(posedge clk_50M) begin
    #1;
    if(out !== 0 && out !== out_capture[capture_idx]) begin
        out_capture[capture_idx] <= out;
        capture_idx <= capture_idx + 1;
    end
end

initial begin
    clk_50M       = 0;
    rst           = 0;
    master_start  = 0;
    s_axis_tdata  = 0;
    s_axis_tvalid = 0;
    s_axis_tlast  = 0;
    capture_idx   = 0;

    weights = {9{8'd1}};
    bias    = 0;
    
    // Safely initialize the new larger capture array
    for(i = 0; i < 200; i = i + 1) begin
        out_capture[i] = 0;
    end

    // release reset
    repeat(2) @(posedge clk_50M); #1;
    rst = 1;

    // assert master_start
    @(posedge clk_50M); #1;
    master_start = 1;

    @(posedge clk_50M); #1;

    // SEND 6 ROWS SEQUENTIALLY (120 pixels)
    // Rows 1-4 trigger the START state. 
    // Rows 5-6 will force the state machine deep into the BTW state!
    for(i = 0; i < LB_size * 6; i = i + 1) begin
        // wait for tready with #1 after clock edge
        @(posedge clk_50M); #1;
        while(!s_axis_tready) begin
            @(posedge clk_50M); #1;
        end
        
        s_axis_tdata  = i + 1;
        s_axis_tvalid = 1'b1;
        // Fire tlast ONLY on the 120th pixel
        s_axis_tlast  = (i == (LB_size * 6) - 1) ? 1'b1 : 1'b0; 
    end

    // deassert after last pixel
    @(posedge clk_50M); #1;
    s_axis_tvalid = 1'b0;
    s_axis_tlast  = 1'b0;
    s_axis_tdata  = 0;

    // Increased wait time: Give the pipeline enough time to drain the final rows
    repeat(200) @(posedge clk_50M); #1;

    $display("--- Convolution Outputs ---");
    for(i = 0; i < capture_idx; i = i + 1)
        $display("Output[%0d] = %0d", i, out_capture[i]);

    $finish;
end

endmodule
