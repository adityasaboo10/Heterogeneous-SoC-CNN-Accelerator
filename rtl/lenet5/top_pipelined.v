module top_pipelined#(parameter PIXW = 8, 
                      parameter LB_size = 256,
                   parameter K_size = 3 //size of kernel is 3x3
                   )(
    input clk_50M,
    input rst,
    input  [PIXW-1:0]  s_axis_tdata,
    input write_en1, write_en2, write_en3, write_en4,
    input [1:0] lb_state,
    input [7:0] base_pos,
    input  [8:0] active_width,
//    input rd_start,
    input master_start,
    output ptr_max1, ptr_max2, ptr_max3, ptr_max4,
//    output [7:0] pointer1, pointer2, pointer3, pointer4,
//    output all_strides_done,
    output [(K_size * K_size * PIXW)-1:0] ifMAP_flat
//    output vec_valid //goes to vector engine
    );
    
wire [(3*PIXW)-1:0] out_pix1, out_pix2, out_pix3, out_pix4;       

//----------------Line Buffers----------------------    
Line_Buffer #(.PIXW(PIXW), .LB_size(LB_size)) LB1 (.clk_50M(clk_50M), .rst(rst), .inp_pix(s_axis_tdata), 
                .write_en(write_en1),.active_width(active_width), .base_pos(base_pos), .out_pix(out_pix1),
                .ptr_max(ptr_max1));
                
Line_Buffer #(.PIXW(PIXW), .LB_size(LB_size)) LB2 (.clk_50M(clk_50M), .rst(rst), .inp_pix(s_axis_tdata), 
                .write_en(write_en2),.active_width(active_width), .base_pos(base_pos), .out_pix(out_pix2),
                .ptr_max(ptr_max2));
                
Line_Buffer #(.PIXW(PIXW), .LB_size(LB_size)) LB3 (.clk_50M(clk_50M), .rst(rst), .inp_pix(s_axis_tdata), 
                .write_en(write_en3),.active_width(active_width), .base_pos(base_pos), .out_pix(out_pix3),
                .ptr_max(ptr_max3));               
                  
Line_Buffer #(.PIXW(PIXW), .LB_size(LB_size)) LB4 (.clk_50M(clk_50M), .rst(rst), .inp_pix(s_axis_tdata), 
                .write_en(write_en4),.active_width(active_width), .base_pos(base_pos), .out_pix(out_pix4),
                .ptr_max(ptr_max4));  
                
             
////-----------------ReadController----------------------------                
//ReadController #(.PIXW(PIXW),. K_size(K_size), . LB_size(Lb_size)) rdc (.clk_50M(clk_50M), .rst(rst), .rd_start(rd_start), .lb_state(lb_state),
//                   .out_pix1(out_pix1), .out_pix2(out_pix2), .master_start(master_start),
//                   .out_pix3(out_pix3),.out_pix4(out_pix4),
//                   .all_strides_done(all_strides_done), .base_pos(base_pos),
//                   .ifMAP_flat(ifMAP_flat), .vec_valid(vec_valid));
                   
                   
//--------------IFMAP mux--------------------------------------
 IfmapMux #(.PIXW(PIXW),. K_size(K_size))ifm (.clk_50M(clk_50M), .rst(rst),.master_start(master_start),
                                              .lb_state(lb_state), .out_pix1(out_pix1), .out_pix2(out_pix2),
                                              .out_pix3(out_pix3),.out_pix4(out_pix4),
                                              .ifMAP_flat(ifMAP_flat)
                                            );
endmodule
