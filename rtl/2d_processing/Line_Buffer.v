//LINE_Buffer must be reset before using again 

module Line_Buffer #(parameter PIXW = 8, LB_size = 256)
   (input clk_50M,
    input rst,  //active low
    input [PIXW-1 :0] inp_pix, 
    input write_en,  //comes from AXIcontroller to tell when to write
    input [7:0] base_pos,  //from central controller 
    output wire [(3 * PIXW)-1:0] out_pix,
    output [7:0] pointer_out
    );
    
assign pointer_out = pointer;

(* ram_style = "block" *) reg [PIXW-1 : 0] LB [LB_size-1 : 0];
reg [7:0] pointer;  // 5 bits is enough to point to 20 LB cells
integer i;

//Write Logic 
always @(posedge clk_50M or negedge rst) begin
    if(!rst)begin
        pointer <= 0;
        for(i = 0  ; i < LB_size ; i = i+1) begin
            LB[i] <= 0; 
        end
    end
    else if(write_en) begin
        LB[pointer] <= inp_pix;
        if (pointer == LB_size - 1)
            pointer <= 0;
        else
            pointer <= pointer + 1;
    end
end

// Safe Read Logic (Restored from previous step)
wire [7:0] pos1 = (base_pos + 1 >= LB_size) ? (base_pos + 1 - LB_size) : (base_pos + 1);
wire [7:0] pos2 = (base_pos + 2 >= LB_size) ? (base_pos + 2 - LB_size) : (base_pos + 2);

assign out_pix = { LB[pos2], LB[pos1], LB[base_pos] };


endmodule   
