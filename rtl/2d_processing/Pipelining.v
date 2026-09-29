//After master start we need 1 clk - cycle delay to start latching data correctly
//Pipelines everything together
module Pipelining#(parameter PIXW = 8, 
                   parameter Output_Width = 32,
                   parameter K_size = 3, //size of kernel is 3x3
                   parameter Lb_size = 256
                   )(
                   input clk_50M,
                   input rst,
                   input master_start,
                   input  [PIXW-1:0]  s_axis_tdata,   
                   input          s_axis_tvalid,  // replaces WVALID1..4
                   input          s_axis_tlast,   // end of row signal from DMA
                   input [(K_size * K_size * PIXW)-1 : 0] weights,
                   input [(K_size * K_size * Output_Width)-1 : 0] bias,
                   output         s_axis_tready,  // backpressure to DMA
                   output [Output_Width-1 : 0] out,
                   output vec_done_out
    );
//----------------------------------
//----------AXI DMA to RTL bridge---
wire WREADY1, WREADY2, WREADY3, WREADY4;
assign s_axis_tready = ( WREADY1 || WREADY2 || WREADY3 || WREADY4 );
//--------------------------
//----------Line Buffers--------
wire [(3*PIXW)-1:0] out_pix1, out_pix2, out_pix3, out_pix4;
wire [7:0] pointer1, pointer2, pointer3, pointer4;
wire [7:0] base_pos;

Line_Buffer LB1 (.clk_50M(clk_50M), .rst(rst), .inp_pix(s_axis_tdata), 
                .write_en(write_en1), .base_pos(base_pos), .out_pix(out_pix1),
                .pointer_out(pointer1));
                
Line_Buffer LB2 (.clk_50M(clk_50M), .rst(rst), .inp_pix(s_axis_tdata), 
                .write_en(write_en2), .base_pos(base_pos), .out_pix(out_pix2),
                .pointer_out(pointer2));
                
Line_Buffer LB3 (.clk_50M(clk_50M), .rst(rst), .inp_pix(s_axis_tdata), 
                .write_en(write_en3), .base_pos(base_pos), .out_pix(out_pix3),
                .pointer_out(pointer3));               
                  
Line_Buffer LB4 (.clk_50M(clk_50M), .rst(rst), .inp_pix(s_axis_tdata), 
                .write_en(write_en4), .base_pos(base_pos), .out_pix(out_pix4),
                .pointer_out(pointer4));     

//------------------------------
//------AXI Controller---------
wire rd_start, write_en1, write_en2, write_en3, write_en4; 
wire [1:0] lb_state;
wire all_strides_done;
AXIController #(.PIXW(PIXW),. K_size(K_size), . LB_size(Lb_size)) axc(.clk_50M(clk_50M), .rst(rst),
                  .s_axis_tvalid(s_axis_tvalid), .s_axis_tlast(s_axis_tlast), 
                  .all_strides_done(all_strides_done), .master_start(master_start),
                  .pointer1(pointer1), .pointer2(pointer2),.pointer3(pointer3),.pointer4(pointer4),
                  .rd_start(rd_start), 
                  .WREADY1(WREADY1), .WREADY2(WREADY2),.WREADY3(WREADY3),.WREADY4(WREADY4),
                  .lb_state(lb_state),
                  .write_en1(write_en1),.write_en2(write_en2),
                  .write_en3(write_en3),.write_en4(write_en4));
                             
//--------------------------------
//-----------Read Controller------
wire vec_done;
assign vec_done_out = vec_done;
wire vec_valid;
wire [(K_size * K_size * PIXW)-1:0] ifMAP_flat;

ReadController #(.PIXW(PIXW),. K_size(K_size), . LB_size(Lb_size)) rdc (.clk_50M(clk_50M), .rst(rst), .rd_start(rd_start), .lb_state(lb_state),
                   .out_pix1(out_pix1), .out_pix2(out_pix2), .master_start(master_start),
                   .out_pix3(out_pix3),.out_pix4(out_pix4),
                   .all_strides_done(all_strides_done), .base_pos(base_pos),
                   .ifMAP_flat(ifMAP_flat), .vec_valid(vec_valid));
                   
//--------------------------------------    
//--------Vector Engine----------------
Vector_Engine   #(.PIXW(PIXW),.Output_Width (Output_Width), .Kernel_size(K_size)) VE
                 (.clk_50M(clk_50M), .rst(rst), .en(vec_valid), .ifMAP_flat(ifMAP_flat),
                .weights(weights), .bias(bias), .out(out), .done(vec_done));
//-----------------------
endmodule
