module ReadController #(parameter PIXW = 8, parameter K_size = 3, parameter LB_size = 256)
                 (input clk_50M,
                  input rst,
                  input rd_start,
                  input [1:0] lb_state,
                  input [(3*PIXW)-1:0] out_pix1, out_pix2, out_pix3, out_pix4,
                  input master_start,
                  output reg all_strides_done,
                  output reg [7:0] base_pos,
                  output reg [(K_size * K_size * PIXW)-1:0] ifMAP_flat,
                  output vec_valid
                  );

localparam buf1 = 2'd0;
localparam buf2 = 2'd1;
localparam buf3 = 2'd2;
localparam buf4 = 2'd3;

reg [7:0] rd_counter;
reg reading_active;

assign vec_valid = (rst && master_start) ? reading_active : 1'b0;

// Sequential block: reading_active and stride counters
always @(posedge clk_50M or negedge rst) begin
    if (!rst) begin
        base_pos         <= 0;
        all_strides_done <= 0;
        rd_counter       <= 0;
        reading_active   <= 0;
    end
    else if (!master_start) begin
        base_pos         <= 0;
        all_strides_done <= 0;
        rd_counter       <= 0;
        reading_active   <= 0;
    end
    else begin
        all_strides_done <= 1'b0;

        // reading_active latch: set on rd_start, clear when strides done
        if (rd_start)
            reading_active <= 1'b1;
        else if (all_strides_done)
            reading_active <= 1'b0;

        // stride counter and base_pos
        if (reading_active) begin
            if (rd_counter < LB_size - K_size) begin
                base_pos   <= base_pos + 1;
                rd_counter <= rd_counter + 1;
            end
            else begin
                all_strides_done <= 1'b1;
                rd_counter       <= 0;
                base_pos         <= 0;
            end
        end
    end
end

// Combinational block: ifMAP_flat mux only
always @(*) begin
    ifMAP_flat = 0;
    if (rst && master_start) begin
        case (lb_state)
            buf1: ifMAP_flat = {out_pix4, out_pix3, out_pix2};
            buf2: ifMAP_flat = {out_pix1, out_pix4, out_pix3};
            buf3: ifMAP_flat = {out_pix2, out_pix1, out_pix4};
            buf4: ifMAP_flat = {out_pix3, out_pix2, out_pix1};
        endcase
    end
end

endmodule
