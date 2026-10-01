//In the start all 2 LBs fill one by one
//After that each pixels goes to te,p buffer and moves internally
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
                          input ptr_max1, ptr_max2, tempfull,
                          input retired_pixel_valid,
                          output rd_start, 
                          output WREADY1, WREADY2, TEMPREADY,
                          output [NUM_ENGINES-1:0] lb_state_tree, //inidicates which oldest row
                          output wire write_en1, write_en2, temp_write_en,
                          output recycle_en1, recycle_en2
                          );
                          
// Internally, replace the single lb_state register with:
(* max_fanout = 4 *) (* equivalent_register_removal = "no" *)
    reg lb_state_replicated [0:NUM_ENGINES-1]; 
    reg waiting_for_temp;  //0 means computation started, 1 means buf doesnt have its first 3 pixel window
//---------------
    // Disable FSM extraction so Vivado obeys the fanout limit, 
    // forcing it to replicate these control states locally.
    
    (* fsm_extraction = "none" *)
    (* max_fanout = 4 *) 
    (* equivalent_register_removal = "no" *)
    reg start_State;
    localparam START = 1'b0;
    localparam BTW = 1'b1;

    //-----BTW state param----
    localparam buf1 = 1'd0;
    localparam buf2 = 1'd1;
    
    (* max_fanout = 4 *) (* equivalent_register_removal = "no" *) reg current_row;
    
    localparam R1 = 1'd0;
    localparam R2 = 1'd1;
    (* max_fanout = 4 *) (* equivalent_register_removal = "no" *)
    reg check;   //accounts for TLAST signal (at the last data pixel check is set to 0 to stop input
    (* max_fanout = 4 *) (* equivalent_register_removal = "no" *) reg row_full;
    reg math_done;
    
    //---------------------
    // write_en fires when stream is valid and that LB is ready
    assign write_en1 = s_axis_tvalid && WREADY1 && check;  //handshake
    assign write_en2 = s_axis_tvalid && WREADY2 && check;
    assign temp_write_en = s_axis_tvalid && TEMPREADY && check;
    
    assign WREADY1 = (start_State == START && current_row == R1 && master_start && rst && !row_full && check) ? 1 : 0 ;
    assign WREADY2 = (start_State == START && current_row == R2 && master_start && rst && !row_full && check) ? 1 : 0 ;
    assign TEMPREADY = ((start_State == BTW) && master_start && rst && !row_full && check) ? 1 : 0 ;
    
    assign rd_start = (start_State == BTW) && waiting_for_temp && tempfull && temp_write_en;
    
    assign recycle_en1 = (start_State == BTW) && !row_full && (lb_state_replicated[0] == buf1) && retired_pixel_valid;
    assign recycle_en2 = (start_State == BTW) && !row_full && (lb_state_replicated[0] == buf2) && retired_pixel_valid;
    
    integer i;
    
    assign lb_state_tree = {lb_state_replicated[0], lb_state_replicated[1], lb_state_replicated[2],
                         lb_state_replicated[3], lb_state_replicated[4], lb_state_replicated[5]};
                         
    always @(posedge clk_50M or negedge rst) begin
        if(!rst || !master_start) begin
            waiting_for_temp <= 1'b0;
            for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                lb_state_replicated[i] <= buf1;
            end 
            start_State <= START;
            current_row <= R1;
            check       <= 1'b0;
//            pointer_state <= P1;
            row_full    <= 1'b0;
            math_done   <= 1'b0;
        end 
        else begin
            if(rd_start) math_done <= 1'b0;
            else if(all_strides_done) math_done <= 1'b1;  //math_done acts as a latch
            
            case(start_State) 
                START : begin
                    check <= 1'b1;
                    if(s_axis_tvalid) begin
                        case(current_row)
                        R1 : begin
                            if(ptr_max1 && WREADY1) begin
                                    current_row <= R2;
                            end
                            else begin
                                current_row <= R1;
                            end
                        end
                        R2 : begin
                            if(ptr_max2 && WREADY2) begin
                                    start_State <= BTW;
                                    waiting_for_temp <= 1'b1;
                                    for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                                        lb_state_replicated[i] <= buf1;
                                    end
                            end
                            else begin
                                current_row <= R2;
                            end
                        end 
                        endcase
                    end
                end
                //------------------------ 
                BTW : begin
                     // all incoming AXI pixels go into tempbuf
                     if (temp_write_en && s_axis_tlast) begin
                        check <= 1'b0;
                     end
                    //----------------------
                    // waiting for new 3 pixels in the buf
                    if (waiting_for_temp) begin
                        if (tempfull && temp_write_en) begin
                            // First three pixels of this row now exist.
                             waiting_for_temp <= 1'b0;
                        end
                    end
                     // --------current row is now running ---------
                     else begin
                        case (lb_state_replicated[0])
                            buf1 : begin  //row 1 is the oldest row
                                // ---------------------------------------------
                                // Final recycled pixel of this row is being written into LB1 now.
                                if (ptr_max1 && recycle_en1) begin   //data recycling finished
                                    if (math_done || all_strides_done) begin  //is math done too
                                        row_full <= 1'b0;
                                        for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                                            lb_state_replicated[i] <= buf2;  //row 2 has oldest pixel now
                                        end
                                        waiting_for_temp <= 1'b1;
                                    end
                                    else begin //if math not done set row_full and wait for math
                                        row_full <= 1'b1;
                                    end
                                end 
                                // Math has now finally finished.
                                //-----------------------------
                                else if(row_full && all_strides_done) begin
                                    row_full <= 1'b0;
                                    for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                                        lb_state_replicated[i] <= buf2;
                                    end
                                    waiting_for_temp <= 1'b1;
                                end
                            end 
                            buf2 : begin  //row 2 is the oldest row
                                // ---------------------------------------------
                                // Final recycled pixel of this row is being written into LB2 now.
                                if (ptr_max2 && recycle_en2) begin   //data recycling finished
                                    if (math_done || all_strides_done) begin  //is math done too
                                        row_full <= 1'b0;
                                        for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                                            lb_state_replicated[i] <= buf1;  //row 2 has oldest pixel now
                                        end
                                        waiting_for_temp <= 1'b1;
                                    end
                                    else begin //if math not done set row_full and wait for math
                                        row_full <= 1'b1;
                                    end
                                end 
                                // Math has now finally finished.
                                //-----------------------------
                                else if(row_full && all_strides_done) begin
                                    row_full <= 1'b0;
                                    for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                                        lb_state_replicated[i] <= buf1;
                                    end
                                    waiting_for_temp <= 1'b1;
                                end
                            end  
                        endcase
                     end   
                end 
            endcase 
        end
    end
endmodule
