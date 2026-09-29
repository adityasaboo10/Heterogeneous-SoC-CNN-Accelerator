//In the start all 4 LBs fill one by one
//After that each row fills with new set of pixels
/*
 STALL LOGIC (buf1..buf4 in AXIController):
 LB full but math not done -> row_full<=1 -> WREADY drops -> write_en
 drops -> pointer FREEZES (doesn't reset; only resets on next write_en).
 else-if waits for all_strides_done (1-cycle pulse, safe same-clk) ->
 row_full<=0, advance to next buf. row_full can only be 1 when this
 else-if runs (WREADY gates the if-branch by !row_full), so the clear
 is always real, never stuck, never a no-op.
*/
    module AXIController #(parameter PIXW = 8,parameter K_size = 3, parameter LB_size = 256, NUM_ENGINES = 6)
                         (input clk_50M,
                          input rst,
                          input s_axis_tvalid,
                          input s_axis_tlast,
                          input all_strides_done,
                          input master_start,
//                          input [7:0] pointer1, pointer2, pointer3, pointer4, //comes from LB
                          input ptr_max1, ptr_max2, ptr_max3, ptr_max4,
                          output reg rd_start, 
                          output WREADY1, WREADY2, WREADY3, WREADY4,
                          output [(NUM_ENGINES * 2)-1:0] lb_state_tree,
                          output wire write_en1, write_en2, write_en3, write_en4
                          );
                          
                          
    // Internally, replace the single lb_state register with:
    (* max_fanout = 4 *) (* equivalent_register_removal = "no" *)
    reg [1:0] lb_state_replicated [0:NUM_ENGINES-1];
    //---------------
    // Disable FSM extraction so Vivado obeys the fanout limit, 
    // forcing it to replicate these control states locally.
    
    (* fsm_extraction = "none" *)
    (* max_fanout = 4 *) 
    (* equivalent_register_removal = "no" *)
    reg start_State;
    localparam START = 1'b0;
    localparam BTW = 1'b1;
    //----------------------
    
    (* fsm_extraction = "none" *)
    (* max_fanout = 4 *) 
    (* equivalent_register_removal = "no" *)
    reg [1:0] pointer_state ;
    localparam P1 = 2'd0;
    localparam P2 = 2'd1;
    localparam P3 = 2'd2;
    localparam P4 = 2'd3;
    //-----BTW state param----
    localparam buf1 = 2'd0;
    localparam buf2 = 2'd1;
    localparam buf3 = 2'd2;
    localparam buf4 = 2'd3;
    //----------------------
    (* max_fanout = 4 *) (* equivalent_register_removal = "no" *) reg [1:0] current_row;
    (* max_fanout = 4 *) (* equivalent_register_removal = "no" *)
    reg check;   //accounts for TLAST signal (at the last data pixel check is set to 0 to stop input
    (* max_fanout = 4 *) (* equivalent_register_removal = "no" *) reg row_full;
    reg math_done;
    //---------------------
    // write_en fires when stream is valid and that LB is ready
    assign write_en1 = s_axis_tvalid && WREADY1 && check;  //handshake
    assign write_en2 = s_axis_tvalid && WREADY2 && check;
    assign write_en3 = s_axis_tvalid && WREADY3 && check;
    assign write_en4 = s_axis_tvalid && WREADY4 && check;
    
    assign WREADY1 = (current_row == 0 && master_start && rst && !row_full) ? 1 : 0 ;
    assign WREADY2 = (current_row == 1 && master_start && rst && !row_full) ? 1 : 0 ;
    assign WREADY3 = (current_row == 2 && master_start && rst && !row_full) ? 1 : 0 ;
    assign WREADY4 = (current_row == 3 && master_start && rst && !row_full) ? 1 : 0 ;
    
    integer i;
    
    assign lb_state_tree = {lb_state_replicated[0], lb_state_replicated[1], lb_state_replicated[2],
                         lb_state_replicated[3], lb_state_replicated[4], lb_state_replicated[5]};
                         
    always @(posedge clk_50M or negedge rst) begin
        if(!rst || !master_start) begin
            rd_start    <= 1'b0;
            for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                lb_state_replicated[i] <= buf4;
            end 
            start_State <= START;
            current_row <= 2'd0;
            check       <= 1'b0;
            pointer_state <= P1;
            row_full    <= 1'b0;
            math_done   <= 1'b0;
        end 
        else begin
            if(all_strides_done) math_done <= 1'b1;  //math_done acts as a latch
            else if(rd_start) math_done <= 1'b0;

            case(start_State) 
                START : begin
                    check <= 1'b1;
                    rd_start <= 1'b0;
                    if(s_axis_tvalid) begin
                        case(pointer_state) 
                            P1 : begin
                                if(ptr_max1 && WREADY1) begin
                                    current_row <= 2'd1;
                                    pointer_state <= P2;
                                end
                                else begin
                                    current_row <= 2'd0;
                                    pointer_state <= P1;
                                end
                            end
                            P2 : begin
                             if(ptr_max2 && WREADY2) begin
                                    current_row <= 2'd2;
                                    pointer_state <= P3;
                                end
                                else begin
                                    current_row <= 2'd1;
                                    pointer_state <= P2;
                                end
                            end
                            P3 : begin
                             if(ptr_max3 && WREADY3) begin
                                    current_row <= 2'd3;
                                    pointer_state <= P4;
                                    rd_start <= 1'b1;   //start computation as we have valid data
                                end
                                else begin
                                    current_row <= 2'd2;
                                    pointer_state <= P3;
                                end
                            end 
                            P4 : begin
                                if(ptr_max4 && WREADY4) begin
                                    start_State <= BTW;
                                   for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                                        lb_state_replicated[i] <= buf1;
                                   end
                                    current_row <= 2'd0;
                                    rd_start <= 1'b1;  //after 4th row is filled we again have valid data
                                end
                                else begin
                                    current_row <= 2'd3;
                                    pointer_state <= P4;
                                end
                            end
                        endcase
                    end
                end
                BTW : begin
                    rd_start <= 1'b0;
                    if(s_axis_tvalid && s_axis_tlast && (WREADY1 || WREADY2 || WREADY3 || WREADY4)) check <= 1'b0;
                    
                    case(lb_state_replicated[0])
//--------------------------------------
                        buf1 : begin
                            if(ptr_max1 && s_axis_tvalid && WREADY1) begin
                                if(math_done || all_strides_done) begin  //after buffer is filled, if math is also done move on
                                    current_row <= 2'd1;
                                    for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                                        lb_state_replicated[i] <= buf2;
                                    end
                                    rd_start <= 1'b1;
                                end
                                else begin  //math is slower than buf
                                    row_full <= 1'b1;  //sets WREADY as low
                                end
                            end
                            else if(row_full && all_strides_done) begin
                            // We were paused, math just finished, resume!
                                row_full <= 1'b0;
                                current_row <= 2'd1;
                                for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                                        lb_state_replicated[i] <= buf2;
                                end
                                rd_start <= 1'b1;
                            end
                        end    
//--------------------------------------------
                        buf2 : begin
                            if(ptr_max2 && s_axis_tvalid && WREADY2) begin
                                if(math_done || all_strides_done) begin  //after buffer is filled, if math is also done move on
                                    current_row <= 2'd2;
                                    for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                                        lb_state_replicated[i] <= buf3;
                                    end
                                    rd_start <= 1'b1;
                                end
                                else begin  //math is slower than buf
                                    row_full <= 1'b1;
                                end
                            end
                            else if(row_full && all_strides_done) begin
                            // We were paused, math just finished, resume!
                                row_full <= 1'b0;
                                current_row <= 2'd2;
                                for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                                    lb_state_replicated[i] <= buf3;
                                end
                                rd_start <= 1'b1;
                            end
                        end
//--------------------------------------------
                        buf3 : begin
                            if(ptr_max3 && s_axis_tvalid && WREADY3) begin
                                if(math_done || all_strides_done) begin  //after buffer is filled, if math is also done move on
                                    current_row <= 2'd3;
                                    for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                                        lb_state_replicated[i] <= buf4;
                                    end
                                    rd_start <= 1'b1;
                                end
                                else begin  //math is slower than buf
                                    row_full <= 1'b1;
                                end
                            end
                            else if(row_full && all_strides_done) begin
                            // We were paused, math just finished, resume!
                                row_full <= 1'b0;
                                current_row <= 2'd3;
                                for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                                    lb_state_replicated[i] <= buf4;
                                end
                                rd_start <= 1'b1;
                            end
                        end
//--------------------------------------------
                        buf4 : begin
                            if(ptr_max4 && s_axis_tvalid && WREADY4) begin
                                if(math_done || all_strides_done) begin  //after buffer is filled, if math is also done move on
                                    current_row <= 2'd0;
                                    for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                                        lb_state_replicated[i] <= buf1;
                                    end
                                    rd_start <= 1'b1;
                                end
                                else begin  //math is slower than buf
                                    row_full <= 1'b1;
                                end
                            end
                            else if(row_full && all_strides_done) begin
                            // We were paused, math just finished, resume!
                                row_full <= 1'b0;
                                current_row <= 2'd0;
                                for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                                    lb_state_replicated[i] <= buf1;
                                end
                                rd_start <= 1'b1;
                            end
                        end                                                
                    endcase
                end
            endcase
        end
    end
    endmodule
