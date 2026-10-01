`timescale 1ns / 1ps
(* use_dsp = "yes" *)

module PE #(parameter PIXW = 8, 
            parameter Output_Width = 32)(
            input clk_50M, 
            input rst,
            input [PIXW-1 :0] a, //Unsigned Pixel
            input [PIXW-1 :0] b, // Signed Weight
            input [Output_Width-1 :0] c,
            input en,
            output reg done,
            output reg [Output_Width-1 :0] out
    );
// --- THE 9-BIT PADDING TRICK ---
    wire signed [PIXW   :0] a_signed = {1'b0, a}; // Force to positive
    wire signed [PIXW-1:0] b_signed = b;         // Read as signed
    
// Natively handle the signed multiplication
    wire signed [Output_Width-1:0] mult_result = a_signed * b_signed;
    
always @(posedge clk_50M or negedge rst) begin
    if(!rst || !en) begin  //en should be 1 to start computing
        out <= 0;
        done <= 1'b0;
    end
    else begin
        out <= mult_result + $signed(c);
        done <= 1'b1;
    end
end
endmodule
