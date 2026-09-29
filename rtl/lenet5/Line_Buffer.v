module Line_Buffer #(parameter PIXW = 8, LB_size = 256)
   (input clk_50M,
    input rst,
    input [PIXW-1:0] inp_pix,
    input write_en,
    input [7:0] base_pos,     // caller still passes the "left" edge of the window
    output [(3*PIXW)-1:0] out_pix,
//    output [7:0] pointer_out,
    output ptr_max //flag to indicate if the ptr is at max position
    );

//    assign pointer_out = pointer;
(* max_fanout = 3 *) (* equivalent_register_removal = "no" *)
    reg ptr_max_reg;
    
    assign ptr_max = ptr_max_reg;
    
    (* ram_style = "block" *) reg [PIXW-1:0] LB [LB_size-1:0];
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
        else if (write_en)begin 
            if(pointer == LB_size - 2) ptr_max_reg <= 1'b1;
            else if(pointer == LB_size-1) ptr_max_reg <= 1'b0;
            pointer <= (pointer == LB_size-1) ? 0 : pointer + 1;
        end
    end

    always @(posedge clk_50M) begin
        if (write_en) LB[pointer] <= inp_pix;   // single write port
        rd_data <= LB[read_addr];               // single read port, reads the "newest" pixel of the window
        tap1 <= rd_data;   // 1 cycle older -> becomes "middle" pixel
        tap2 <= tap1;      // 2 cycles older -> becomes "oldest/base" pixel
    end

    assign out_pix = {rd_data, tap1, tap2}; // {newest, middle, oldest} == {out2, out1, out0}
endmodule
