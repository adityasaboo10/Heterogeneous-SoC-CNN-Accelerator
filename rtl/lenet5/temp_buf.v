module temp_buf #(parameter PIXW = 8, LB_size = 256, K_size= 3)(
    input clk_50M,
    input rst,
    input master_start,
    input [PIXW-1:0] inp_pix,
    input temp_write_en,
    output [(3*PIXW)-1:0] out_pix_temp,
    output retired_pixel_valid,
    output [PIXW-1:0] retired_pix,
    output reg tempfull
    );
    
    (* ram_style = "block" *) reg [PIXW-1:0] tempBuf [K_size-1:0];  //physical memory is the same
    //for 3 cycle delay
    reg [PIXW-1:0] rd_data;   // BRAM's own registered output (1 cycle latency)
    reg [PIXW-1:0] tap1, tap2; // shift-register taps to reconstruct history
    
    reg [1:0] cycle_counter;
        
    assign out_pix_temp = {rd_data, tap1, tap2}; // {newest, middle, oldest} == {out2, out1, out0}
    assign retired_pix = tempBuf[0];
    assign retired_pixel_valid = tempfull && temp_write_en;
    
    always @(posedge clk_50M) begin 
        if(!rst || !master_start) begin
            rd_data <= 0;
            tap1 <= 0;
            tap2 <= 0;
        end
        else begin
            rd_data <= tempBuf[0];
            tap1 <= rd_data;
            tap2 <= tap1; 
        end
    end
    
    always @(posedge clk_50M or negedge rst) begin
        if(!rst || !master_start) begin
            cycle_counter <= 0;
            tempfull <= 1'b0;
        end
        else if(temp_write_en) begin
            tempBuf[2] <= inp_pix;
            tempBuf[1] <= tempBuf[2];
            tempBuf[0] <= tempBuf[1];
            
            tempfull <= (cycle_counter == K_size-1);    
                             
            if(temp_write_en && cycle_counter < K_size-1) cycle_counter <= cycle_counter + 1;
            else cycle_counter <= cycle_counter ;
        end
    end
endmodule
