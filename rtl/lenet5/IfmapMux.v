module IfmapMux #(parameter PIXW=8, parameter K_size=3)(
    input clk_50M,
    input rst,
    input master_start,
    input [1:0] lb_state,
    input [(3*PIXW)-1:0] out_pix1, out_pix2, out_pix3, out_pix4,
    output reg [(K_size*K_size*PIXW)-1:0] ifMAP_flat);
    
    
    localparam buf1 = 2'd0;
    localparam buf2 = 2'd1;
    localparam buf3 = 2'd2;
    localparam buf4 = 2'd3;

    reg [(K_size*K_size*PIXW)-1:0] ifMAP_flat_comb;
    always @(*) begin
    ifMAP_flat_comb = 0;
    if (rst && master_start) begin
        case (lb_state)
            buf1: ifMAP_flat_comb = {out_pix4, out_pix3, out_pix2};
            buf2: ifMAP_flat_comb = {out_pix1, out_pix4, out_pix3};
            buf3: ifMAP_flat_comb = {out_pix2, out_pix1, out_pix4};
            buf4: ifMAP_flat_comb = {out_pix3, out_pix2, out_pix1};
        endcase
    end
end
// registered output. Breaks the combinational chain into two cycles.
    always @(posedge clk_50M or negedge rst) begin
        if (!rst)
            ifMAP_flat <= 0;
        else
            ifMAP_flat <= ifMAP_flat_comb;
    end
endmodule
