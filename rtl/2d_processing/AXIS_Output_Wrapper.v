


module AXIS_Output_Wrapper #(parameter Width = 32, parameter TOTAL_PIXELS = 62720 ) //245 X 256
    (
    input clk_50M,
    input rst,
    //From Vector Engine
    input [Width-1:0] ve_out,
    input ve_done,
    // Outputs to the AXI DMA (S_AXIS_S2MM)
    output [Width-1:0] m_axis_tdata,
    output reg m_axis_tvalid,
    input  m_axis_tready,
    output reg m_axis_tlast
    );
    
assign m_axis_tdata = ve_out;
reg [31:0] pixel_count;

always @(posedge clk_50M or negedge rst) begin
        if(!rst) begin
            m_axis_tvalid <= 1'b0;
            m_axis_tlast  <= 1'b0;
            pixel_count   <= 0;
        end
        else begin
            m_axis_tvalid <= ve_done;
            if (ve_done && m_axis_tready) begin
                if (pixel_count == TOTAL_PIXELS - 1) begin
                    m_axis_tlast <= 1'b1;
                    pixel_count <= 0; // Reset for the next frame
                end else begin
                    m_axis_tlast <= 1'b0;
                    pixel_count <= pixel_count + 1;
                end
            end else begin
                m_axis_tlast <= 1'b0;
            end
        end 
end
endmodule
