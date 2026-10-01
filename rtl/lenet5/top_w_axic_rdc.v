module top_w_axic_rdc#(parameter PIXW = 8, 
                       parameter K_size = 3, //size of kernel is 3x3
                       parameter LB_size = 256,
                       parameter NUM_ENGINES = 6)(
                       input clk_50M,
                       input rst,
                       input master_start,
                       input  [(6*PIXW)-1:0]  s_axis_tdata,
                       input  [8:0]             active_width,
                       input          s_axis_tvalid,  // replaces WVALID1..4
                       input          s_axis_tlast,   // end of row signal from DMA
                       output         s_axis_tready,   // backpressure to DMA
                       output [NUM_ENGINES-1:0]  vec_valid_tree,  //goes to Vector engine as enable signal
                       output [(K_size * K_size * PIXW)-1 : 0] ifMAP_flat1,
                       output [(K_size * K_size * PIXW)-1 : 0] ifMAP_flat2,
                       output [(K_size * K_size * PIXW)-1 : 0] ifMAP_flat3,
                       output [(K_size * K_size * PIXW)-1 : 0] ifMAP_flat4,
                       output [(K_size * K_size * PIXW)-1 : 0] ifMAP_flat5,
                       output [(K_size * K_size * PIXW)-1 : 0] ifMAP_flat6
    );
        
wire WREADY1, WREADY2, TEMPREADY;
assign s_axis_tready = ( WREADY1 || WREADY2 || TEMPREADY);

// Slice the 48-bit shared bus into 6 independent 8-bit channel lanes.
// Channel 1 = bits [PIXW-1:0], channel 2 = next PIXW bits, etc.
// ============================================================
wire [PIXW-1:0] s_axis_tdata1 = s_axis_tdata[1*PIXW-1 : 0*PIXW];
wire [PIXW-1:0] s_axis_tdata2 = s_axis_tdata[2*PIXW-1 : 1*PIXW];
wire [PIXW-1:0] s_axis_tdata3 = s_axis_tdata[3*PIXW-1 : 2*PIXW];
wire [PIXW-1:0] s_axis_tdata4 = s_axis_tdata[4*PIXW-1 : 3*PIXW];
wire [PIXW-1:0] s_axis_tdata5 = s_axis_tdata[5*PIXW-1 : 4*PIXW];
wire [PIXW-1:0] s_axis_tdata6 = s_axis_tdata[6*PIXW-1 : 5*PIXW];


//--------AXI Controller------
wire rd_start, write_en1, write_en2, temp_write_en, recycle_en1, recycle_en2; 
wire [NUM_ENGINES-1:0] lb_state_tree;
wire all_strides_done;
wire ptr_max1, ptr_max2, tempfull; 
wire  retired_pixel_valid;
//wire [7:0] pointer1, pointer2, pointer3, pointer4;

AXIController #(.PIXW(PIXW),. K_size(K_size), . LB_size(LB_size)) axc(.clk_50M(clk_50M), .rst(rst),
                  .s_axis_tvalid(s_axis_tvalid), .s_axis_tlast(s_axis_tlast), 
                  .all_strides_done(all_strides_done), .master_start(master_start),
//                  .pointer1(pointer1), .pointer2(pointer2),.pointer3(pointer3),.pointer4(pointer4),
                  .ptr_max1(ptr_max1), .ptr_max2(ptr_max2), .tempfull(tempfull), .retired_pixel_valid( retired_pixel_valid),
                  .rd_start(rd_start), 
                  .WREADY1(WREADY1), .WREADY2(WREADY2),.TEMPREADY(TEMPREADY),
                  .lb_state_tree(lb_state_tree),
                  .write_en1(write_en1),.write_en2(write_en2),
                  .temp_write_en(temp_write_en), .recycle_en1(recycle_en1), .recycle_en2(recycle_en2));
                  
                  
//---------------PositionTracker------------
wire  [7:0] base_pos;
PositionTracker #(. K_size(K_size), . LB_size(LB_size)) PT (.clk_50M(clk_50M), .rst(rst),.rd_start(rd_start),
                  .master_start(master_start), .base_pos(base_pos), .active_width(active_width),
                   .all_strides_done(all_strides_done), .vec_valid_tree(vec_valid_tree));
                   
                   
//--------------------top_pipelined--------------
top_pipelined  #(.PIXW(PIXW),. K_size(K_size), .LB_size(LB_size)) TP1 ( .clk_50M(clk_50M), .rst(rst),
                   .s_axis_tdata(s_axis_tdata1), .write_en1(write_en1),.write_en2(write_en2),
                  .temp_write_en(temp_write_en), .lb_state(lb_state_tree[0]), .base_pos(base_pos), .active_width(active_width), .master_start(master_start),
                   .recycle_en1(recycle_en1), .recycle_en2(recycle_en2), .retired_pixel_valid( retired_pixel_valid), 
                   .ptr_max1(ptr_max1), .ptr_max2(ptr_max2), .tempfull(tempfull),.ifMAP_flat(ifMAP_flat1));
                   
top_pipelined  #(.PIXW(PIXW),. K_size(K_size), .LB_size(LB_size)) TP2 ( .clk_50M(clk_50M), .rst(rst),
                   .s_axis_tdata(s_axis_tdata2), .write_en1(write_en1),.write_en2(write_en2),
                  .temp_write_en(temp_write_en), .lb_state(lb_state_tree[1]), .base_pos(base_pos),.active_width(active_width), .master_start(master_start),
                   .recycle_en1(recycle_en1), .recycle_en2(recycle_en2), .ifMAP_flat(ifMAP_flat2));
                   
top_pipelined  #(.PIXW(PIXW),. K_size(K_size), .LB_size(LB_size)) TP3 ( .clk_50M(clk_50M), .rst(rst),
                   .s_axis_tdata(s_axis_tdata3), .write_en1(write_en1),.write_en2(write_en2),
                  .temp_write_en(temp_write_en), .lb_state(lb_state_tree[2]), .base_pos(base_pos),.active_width(active_width), .master_start(master_start),
                   .recycle_en1(recycle_en1), .recycle_en2(recycle_en2), .ifMAP_flat(ifMAP_flat3));
                   
top_pipelined  #(.PIXW(PIXW),. K_size(K_size), .LB_size(LB_size)) TP4 ( .clk_50M(clk_50M), .rst(rst),
                   .s_axis_tdata(s_axis_tdata4), .write_en1(write_en1),.write_en2(write_en2),
                  .temp_write_en(temp_write_en), .lb_state(lb_state_tree[3]), .base_pos(base_pos),.active_width(active_width), .master_start(master_start),
                   .recycle_en1(recycle_en1), .recycle_en2(recycle_en2), .ifMAP_flat(ifMAP_flat4));
                   
top_pipelined  #(.PIXW(PIXW),. K_size(K_size), .LB_size(LB_size)) TP5 ( .clk_50M(clk_50M), .rst(rst),
                   .s_axis_tdata(s_axis_tdata5), .write_en1(write_en1),.write_en2(write_en2),
                  .temp_write_en(temp_write_en), .lb_state(lb_state_tree[4]), .base_pos(base_pos),.active_width(active_width), .master_start(master_start),
                  .recycle_en1(recycle_en1), .recycle_en2(recycle_en2), .ifMAP_flat(ifMAP_flat5));
                   
top_pipelined  #(.PIXW(PIXW),. K_size(K_size), .LB_size(LB_size)) TP6 ( .clk_50M(clk_50M), .rst(rst),
                   .s_axis_tdata(s_axis_tdata6), .write_en1(write_en1),.write_en2(write_en2),
                  .temp_write_en(temp_write_en), .lb_state(lb_state_tree[5]), .base_pos(base_pos),.active_width(active_width), .master_start(master_start),
                  .recycle_en1(recycle_en1), .recycle_en2(recycle_en2), .ifMAP_flat(ifMAP_flat6));                                                        
                   
endmodule
