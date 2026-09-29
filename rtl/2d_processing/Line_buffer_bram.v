module Line_Buffer #(parameter PIXW = 8, LB_size = 256)
   (input clk_50M,
    input rst, // active low
    input [PIXW-1 :0] inp_pix, 
    input write_en, 
    input [7:0] base_pos, 
    output [(3 * PIXW)-1:0] out_pix,
    output [7:0] pointer_out
    );
    
    assign pointer_out = pointer;

    // 1. Manually Replicate the Memory Array (3 Arrays for 3 Read Ports)
    (* ram_style = "block" *) reg [PIXW-1 : 0] LB_0 [LB_size-1 : 0];
    (* ram_style = "block" *) reg [PIXW-1 : 0] LB_1 [LB_size-1 : 0];
    (* ram_style = "block" *) reg [PIXW-1 : 0] LB_2 [LB_size-1 : 0];

    reg [7:0] pointer; 
    
    // Output Registers
    reg [PIXW-1 : 0] out0, out1, out2;

    wire [7:0] pos1 = (base_pos + 1 >= LB_size) ? (base_pos + 1 - LB_size) : (base_pos + 1);
    wire [7:0] pos2 = (base_pos + 2 >= LB_size) ? (base_pos + 2 - LB_size) : (base_pos + 2);

    // --------------------------------------------------------
    // BLOCK 1: Pointer Control (Allowed to have Asynchronous Reset)
    // --------------------------------------------------------
    always @(posedge clk_50M or negedge rst) begin
        if (!rst) begin
            pointer <= 0;
        end else if (write_en) begin
            if (pointer == LB_size - 1)
                pointer <= 0;
            else
                pointer <= pointer + 1;
        end
    end

    // --------------------------------------------------------
    // BLOCK 2: Pure BRAM Inference (NO RESET ALLOWED HERE!)
    // --------------------------------------------------------
    always @(posedge clk_50M) begin
        // Simultaneous Writes to all 3 blocks
        if (write_en) begin
            LB_0[pointer] <= inp_pix;
            LB_1[pointer] <= inp_pix;
            LB_2[pointer] <= inp_pix;
        end
        
        // Independent Synchronous Reads
        out0 <= LB_0[base_pos];
        out1 <= LB_1[pos1];
        out2 <= LB_2[pos2];
    end

    // Concatenate outputs
    assign out_pix = {out2, out1, out0};

endmodule
