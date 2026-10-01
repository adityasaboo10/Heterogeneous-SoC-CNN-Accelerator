module Line_Buffer #(parameter PIXW = 8, LB_size = 256)
   (input clk_50M,
    input rst,
    input [PIXW-1:0] inp_pix,
    input [PIXW-1:0] retired_pix,
    input write_en,
    input recycle_en,
    input  [8:0]             active_width,
    input [7:0] base_pos,     // caller still passes the "left" edge of the window
    output [(3*PIXW)-1:0] out_pix,
//    output [7:0] pointer_out,
    output ptr_max //flag to indicate if the ptr is at max position
    );
    
wire lb_write_enable;
assign lb_write_enable = write_en | recycle_en;

wire [PIXW-1:0] lb_input;
assign lb_input = recycle_en ? retired_pix : inp_pix;
//    assign pointer_out = pointer;
(* max_fanout = 3 *) (* equivalent_register_removal = "no" *)
    reg ptr_max_reg;
    
    assign ptr_max = ptr_max_reg;
    
    (* ram_style = "block" *) reg [PIXW-1:0] LB [LB_size-1:0];  //physical memory is the same
    reg [7:0] pointer;

    // Only ONE read address needed: the leading edge of the 3-wide window
    wire [7:0] read_addr = base_pos;

    reg [PIXW-1:0] rd_data;   // BRAM's own registered output (1 cycle latency)
    reg [PIXW-1:0] tap1, tap2; // shift-register taps to reconstruct history

    always @(posedge clk_50M or negedge rst) begin
        if (!rst) begin 
            pointer <= 0;
            ptr_max_reg <= 1'b0;
        end
        else if (lb_write_enable)begin 
            if(pointer == active_width - 2) ptr_max_reg <= 1'b1; // see below
            else if(pointer == active_width -1) ptr_max_reg <= 1'b0;
            pointer <= (pointer == active_width -1) ? 0 : pointer + 1;
        end
    end
    /*
    ex : for 28 pixels ( 0 to 27)
    pixel 26        pointer=26     ptr_max=0
     clock edge
                  ptr_max ? 1
                  pointer ? 27

pixel 27        pointer=27     ptr_max=1  ? FINAL PIXEL
    */
    always @(posedge clk_50M) begin
        if (lb_write_enable) LB[pointer] <= lb_input;   // single write port
        rd_data <= LB[read_addr];               // single read port, reads the "newest" pixel of the window
        tap1 <= rd_data;   // 1 cycle older -> becomes "middle" pixel
        tap2 <= tap1;      // 2 cycles older -> becomes "oldest/base" pixel
    end

    assign out_pix = {rd_data, tap1, tap2}; // {newest, middle, oldest} == {out2, out1, out0}
endmodule
