module conv_layer_top #(parameter PIXW=8, K_size=3, Lb_size=256, Output_Width=32, NUM_ENGINES = 6)
   (input clk_50M, input rst, input master_start,
    input [(6*PIXW)-1:0] s_axis_tdata,   // 48-bit, 6 channels
    input s_axis_tvalid, input s_axis_tlast,
    output s_axis_tready,
    input [(K_size*K_size*PIXW*6)-1:0] weights_flat,  // from AXI-Lite regs
    input [(Output_Width*6)-1:0] bias_flat,
    input [Output_Width -1 : 0] bias,
    input  [1:0]  mode,  
    input  [8:0]             active_width,
    input [4:0] shift_amount,   
    output signed [Output_Width-1:0] accum_out,  //output of mode 0, 1
    output [(NUM_ENGINES*PIXW)-1:0] quantized_pixel_out,
    output valid_out
     );
    wire signed [Output_Width-1:0] out1, out2, out3, out4, out5, out6;
    
    
    wire [NUM_ENGINES-1 : 0] vec_valid_tree;
    wire [(K_size*K_size*PIXW)-1:0] ifMAP_flat1, ifMAP_flat2, ifMAP_flat3, ifMAP_flat4, ifMAP_flat5, ifMAP_flat6;

    top_w_axic_rdc #(.PIXW(PIXW), .K_size(K_size), .LB_size(Lb_size)) buf_stage (
        .clk_50M(clk_50M), .rst(rst), .master_start(master_start), .active_width(active_width),
        .s_axis_tdata(s_axis_tdata), .s_axis_tvalid(s_axis_tvalid), .s_axis_tlast(s_axis_tlast),
        .s_axis_tready(s_axis_tready), .vec_valid_tree(vec_valid_tree),
        .ifMAP_flat1(ifMAP_flat1), .ifMAP_flat2(ifMAP_flat2), .ifMAP_flat3(ifMAP_flat3),
        .ifMAP_flat4(ifMAP_flat4), .ifMAP_flat5(ifMAP_flat5), .ifMAP_flat6(ifMAP_flat6)
    );

    wire done1, done2, done3, done4, done5, done6;
    wire done_all;
    assign done_all = done1; // all 6 finish together (lockstep) - pick one

localparam integer W_SLICE = K_size * K_size * PIXW;
localparam integer B_SLICE = Output_Width;

Vector_Engine #(.PIXW(PIXW),.Output_Width(Output_Width),.Kernel_size(K_size)) VE1(
    .clk_50M(clk_50M), .rst(rst), .en(vec_valid_tree[0]),
    .ifMAP_flat(ifMAP_flat1),
    .weights(weights_flat[1*W_SLICE-1 : 0*W_SLICE]),
    .bias(bias_flat[1*B_SLICE-1 : 0*B_SLICE]),
    .out(out1), .done(done1));

Vector_Engine #(.PIXW(PIXW),.Output_Width(Output_Width),.Kernel_size(K_size)) VE2(
    .clk_50M(clk_50M), .rst(rst), .en(vec_valid_tree[1]),
    .ifMAP_flat(ifMAP_flat2),
    .weights(weights_flat[2*W_SLICE-1 : 1*W_SLICE]),
    .bias(bias_flat[2*B_SLICE-1 : 1*B_SLICE]),
    .out(out2), .done(done2));

Vector_Engine #(.PIXW(PIXW),.Output_Width(Output_Width),.Kernel_size(K_size)) VE3(
    .clk_50M(clk_50M), .rst(rst), .en(vec_valid_tree[2]),
    .ifMAP_flat(ifMAP_flat3),
    .weights(weights_flat[3*W_SLICE-1 : 2*W_SLICE]),
    .bias(bias_flat[3*B_SLICE-1 : 2*B_SLICE]),
    .out(out3), .done(done3));

Vector_Engine #(.PIXW(PIXW),.Output_Width(Output_Width),.Kernel_size(K_size)) VE4(
    .clk_50M(clk_50M), .rst(rst), .en(vec_valid_tree[3]),
    .ifMAP_flat(ifMAP_flat4),
    .weights(weights_flat[4*W_SLICE-1 : 3*W_SLICE]),
    .bias(bias_flat[4*B_SLICE-1 : 3*B_SLICE]),
    .out(out4), .done(done4));

Vector_Engine #(.PIXW(PIXW),.Output_Width(Output_Width),.Kernel_size(K_size)) VE5(
    .clk_50M(clk_50M), .rst(rst), .en(vec_valid_tree[4]),
    .ifMAP_flat(ifMAP_flat5),
    .weights(weights_flat[5*W_SLICE-1 : 4*W_SLICE]),
    .bias(bias_flat[5*B_SLICE-1 : 4*B_SLICE]),
    .out(out5), .done(done5));

Vector_Engine #(.PIXW(PIXW),.Output_Width(Output_Width),.Kernel_size(K_size)) VE6(
    .clk_50M(clk_50M), .rst(rst), .en(vec_valid_tree[5]),
    .ifMAP_flat(ifMAP_flat6),
    .weights(weights_flat[6*W_SLICE-1 : 5*W_SLICE]),
    .bias(bias_flat[6*B_SLICE-1 : 5*B_SLICE]),
    .out(out6), .done(done6));
    
    
Channel_Accumulator_6to1 #(.Output_Width(Output_Width), .PIXW(PIXW), .NUM_ENGINES(NUM_ENGINES)) CA6T1(
    .clk(clk_50M), .rst_n(rst), .valid_in(done_all), .in_ch0(out1), .in_ch1(out2), .in_ch2(out3),  .in_ch3(out4),
    .in_ch4(out5), .in_ch5(out6), .bias(bias), .mode(mode), .shift_amount(shift_amount), .accum_out(accum_out), 
    .quantized_pixel_out(quantized_pixel_out), .valid_out(valid_out) );
endmodule
