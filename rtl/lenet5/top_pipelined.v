module top_pipelined#(parameter PIXW = 8, 
                      parameter LB_size = 256,
                      parameter K_size = 3 //size of kernel is 3x3
                   )(
    input clk_50M,
    input rst,
    input  [PIXW-1:0]  s_axis_tdata,
    input write_en1, write_en2, temp_write_en,
    input lb_state,
    input [7:0] base_pos,
    input  [8:0] active_width,
//    input rd_start,
    input master_start,
    input recycle_en1, recycle_en2,
    output ptr_max1, ptr_max2, tempfull,
//    output [7:0] pointer1, pointer2, pointer3, pointer4,
//    output all_strides_done,
    output [(K_size * K_size * PIXW)-1:0] ifMAP_flat,
    output  retired_pixel_valid
    );
    
wire [(3*PIXW)-1:0] out_pix1, out_pix2, out_pix_temp;   
wire [PIXW-1:0] retired_pix;   

//----------------Line Buffers----------------------    
Line_Buffer #(.PIXW(PIXW), .LB_size(LB_size)) LB1 (.clk_50M(clk_50M), .rst(rst), .inp_pix(s_axis_tdata),.retired_pix(retired_pix), 
                .write_en(write_en1),.recycle_en(recycle_en1), .active_width(active_width), .base_pos(base_pos), .out_pix(out_pix1),
                .ptr_max(ptr_max1));
                
Line_Buffer #(.PIXW(PIXW), .LB_size(LB_size)) LB2 (.clk_50M(clk_50M), .rst(rst), .inp_pix(s_axis_tdata), .retired_pix(retired_pix),
                .write_en(write_en2),.recycle_en(recycle_en2), .active_width(active_width), .base_pos(base_pos), .out_pix(out_pix2),
                .ptr_max(ptr_max2));
                
temp_buf # (.PIXW(PIXW), .LB_size(LB_size), .K_size(K_size)) tempbuf (.clk_50M(clk_50M), .rst(rst), .master_start(master_start),
              .inp_pix(s_axis_tdata),.temp_write_en(temp_write_en),.out_pix_temp(out_pix_temp),. retired_pixel_valid( retired_pixel_valid),
              .retired_pix(retired_pix),.tempfull(tempfull));
                   
//--------------IFMAP mux--------------------------------------
 IfmapMux #(.PIXW(PIXW),. K_size(K_size))ifm (.clk_50M(clk_50M), .rst(rst),.master_start(master_start),
                                              .lb_state(lb_state), .out_pix1(out_pix1), .out_pix2(out_pix2),
                                              .out_pix_temp(out_pix_temp), .ifMAP_flat(ifMAP_flat));
endmodule
