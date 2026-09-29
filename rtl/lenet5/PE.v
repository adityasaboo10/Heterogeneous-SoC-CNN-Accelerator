`timescale 1ns / 1ps
(* use_dsp = "yes" *)

module PE #(parameter PIXW = 8, 
            parameter Output_Width = 32)(
            input clk_50M, 
            input rst, // Note: Evaluated synchronously now!
            input [PIXW-1 :0] a, //Unsigned Pixel
            input [PIXW-1 :0] b, // Signed Weight
            input en,
            output reg done,
            output reg [Output_Width-1 :0] out
    );

    // --- PIPELINE STAGE 1: DSP Input Registers (AREG / BREG / CREG) ---
    // Registering the inputs allows Vivado to pack these into the DSP slice.
    reg signed [PIXW   :0] a_reg; 
    reg signed [PIXW-1 :0] b_reg;          
    reg en_reg;

    always @(posedge clk_50M) begin
        if (!rst) begin
            a_reg  <= 0;
            b_reg  <= 0;
            en_reg <= 1'b0;
        end else begin
            a_reg  <= {1'b0, a}; // Force to positive
            b_reg  <= b;         // Read as signed
            en_reg <= en;        // Delay the enable to match the data!
        end
    end

    // --- MAC OPERATION ---
    wire signed [Output_Width-1:0] mult_result = a_reg * b_reg;
    
    // --- PIPELINE STAGE 2: DSP Output Register (PREG) ---
    // Note: No negedge rst here! 
    always @(posedge clk_50M) begin
        if(!rst) begin  
            out  <= {Output_Width{1'b0}};
            done <= 1'b0;
        end
        else if(en_reg) begin // Trigger on the pipelined enable
            out  <= mult_result;
            done <= 1'b1;
        end
        else begin
            done <= 1'b0;
        end
    end
endmodule
