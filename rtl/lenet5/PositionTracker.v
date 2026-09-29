`timescale 1ns / 1ps
//between rd_start and a valid pixel generation there is delay of 4 clk cycles
module PositionTracker #(
    parameter K_size      = 3, 
    parameter LB_size     = 256, 
    parameter NUM_ENGINES = 6
)(
    input                    clk_50M, 
    input                    rst, 
    input                    rd_start, 
    input                    master_start, 
    output reg [7:0]         base_pos, 
    output reg               all_strides_done, 
    output [NUM_ENGINES-1:0] vec_valid_tree
);     
    
    reg [7:0] rd_counter;
    reg       reading_active;
    
    // 3-stage pipeline to absorb Line_Buffer latency (Cycles 1, 2, 3)
    reg [2:0] valid_pipeline;

    // 4th stage tree replication register for physical fanout isolation (Cycle 4)
    (* max_fanout = 4 *) (* equivalent_register_removal = "no" *)
    reg [NUM_ENGINES-1:0] vec_valid_delayed_tree;

    assign vec_valid_tree = vec_valid_delayed_tree;  
    integer e;

    always @(posedge clk_50M or negedge rst) begin
        if (!rst) begin
            base_pos               <= 8'd0;
            all_strides_done       <= 1'b0;
            rd_counter             <= 8'd0;
            reading_active         <= 1'b0;
            valid_pipeline         <= 3'b000;
            vec_valid_delayed_tree <= {NUM_ENGINES{1'b0}};
        end
        else if (!master_start) begin
            base_pos               <= 8'd0;
            all_strides_done       <= 1'b0;
            rd_counter             <= 8'd0;
            reading_active         <= 1'b0;
            valid_pipeline         <= 3'b000;
            vec_valid_delayed_tree <= {NUM_ENGINES{1'b0}};
        end
        else begin
            // 1. Shift valid pipeline across clock cycles //3 cycle delay
            valid_pipeline <= {valid_pipeline[1:0], reading_active};

            // 2. Register tree from valid_pipeline[2] (4th total cycle delay)
            for (e = 0; e < NUM_ENGINES; e = e + 1) begin
                vec_valid_delayed_tree[e] <= valid_pipeline[2];
            end

            all_strides_done <= 1'b0;

            // 3. Reading active latch
            if (rd_start)
                reading_active <= 1'b1;
            else if (all_strides_done)
                reading_active <= 1'b0;

            // 4. Stride counter and base_pos generation
            if (reading_active) begin
                if (rd_counter < (LB_size - K_size)) begin
                    base_pos   <= base_pos + 1'b1;
                    rd_counter <= rd_counter + 1'b1;
                end
                else begin
                    all_strides_done <= 1'b1;
                    rd_counter       <= 8'd0;
                    base_pos         <= 8'd0;
                end
            end
        end
    end

endmodule
