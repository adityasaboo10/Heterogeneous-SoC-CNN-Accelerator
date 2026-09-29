//In the start all 4 LBs fill one by one
    //After that each row fills with new set of pixels
    
    module AXIController #(parameter PIXW = 8,parameter K_size = 3, parameter LB_size = 256)
                         (input clk_50M,
                          input rst,
                          input s_axis_tvalid,
                          input s_axis_tlast,
                          input all_strides_done,
                          input master_start,
                          input [7:0] pointer1, pointer2, pointer3, pointer4,
                          output reg rd_start, 
                          output WREADY1, WREADY2, WREADY3, WREADY4,
                          output reg [1:0] lb_state,
                          output wire write_en1, write_en2, write_en3, write_en4
                          );
    //---------------
    
    reg start_State;
    localparam START = 1'b0;
    localparam BTW = 1'b1;
    //----------------------
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
    reg [1:0] current_row;
    reg check; 
    reg row_full;
    reg math_done;
    //---------------------
    // write_en fires when stream is valid and that LB is ready
    assign write_en1 = s_axis_tvalid && WREADY1 && check;
    assign write_en2 = s_axis_tvalid && WREADY2 && check;
    assign write_en3 = s_axis_tvalid && WREADY3 && check;
    assign write_en4 = s_axis_tvalid && WREADY4 && check;
    
    assign WREADY1 = (current_row == 0 && master_start && rst && !row_full) ? 1 : 0 ;
    assign WREADY2 = (current_row == 1 && master_start && rst && !row_full) ? 1 : 0 ;
    assign WREADY3 = (current_row == 2 && master_start && rst && !row_full) ? 1 : 0 ;
    assign WREADY4 = (current_row == 3 && master_start && rst && !row_full) ? 1 : 0 ;
    
    
    always @(posedge clk_50M or negedge rst) begin
        if(!rst) begin
            rd_start    <= 1'b0;
            lb_state    <= buf4;
            start_State <= START;
            current_row <= 2'd0;
            check       <= 1'b0;
            pointer_state <= P1;
            row_full    <= 1'b0;
            math_done   <= 1'b0;
        end
        else if (!master_start) begin 
            rd_start    <= 1'b0;
            lb_state    <= buf4;
            start_State <= START;
            current_row <= 2'd0;
            check       <= 1'b0;
            pointer_state <= P1;
            row_full    <= 1'b0;
            math_done   <= 1'b0;
        end
        else begin
            if(all_strides_done) math_done <= 1'b1;
            else if(rd_start) math_done <= 1'b0;

            case(start_State) 
                START : begin
                    check <= 1'b1;
                    rd_start <= 1'b0;
                    if(s_axis_tvalid) begin
                        case(pointer_state) 
                            P1 : begin
                                if(pointer1 == LB_size - 1 && WREADY1) begin
                                    current_row <= 2'd1;
                                    pointer_state <= P2;
                                end
                                else begin
                                    current_row <= 2'd0;
                                    pointer_state <= P1;
                                end
                            end
                            P2 : begin
                             if(pointer2 == LB_size - 1 && WREADY2) begin
                                    current_row <= 2'd2;
                                    pointer_state <= P3;
                                end
                                else begin
                                    current_row <= 2'd1;
                                    pointer_state <= P2;
                                end
                            end
                            P3 : begin
                             if(pointer3 == LB_size - 1 && WREADY3) begin
                                    current_row <= 2'd3;
                                    pointer_state <= P4;
                                    rd_start <= 1'b1;
                                end
                                else begin
                                    current_row <= 2'd2;
                                    pointer_state <= P3;
                                end
                            end 
                            P4 : begin
                                if(pointer4 == LB_size - 1 && WREADY4) begin
                                    start_State <= BTW;
                                    lb_state <= buf1;
                                    current_row <= 2'd0;
                                    rd_start <= 1'b1;  
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
                    if(s_axis_tlast) check <= 1'b0;
                    
                    case(lb_state)
//--------------------------------------
                        buf1 : begin
                            if(pointer1 ==  LB_size - 1 && s_axis_tvalid && WREADY1) begin
                                if(math_done || all_strides_done) begin  //after buffer is filled, if math is also done move on
                                    current_row <= 2'd1;
                                    lb_state <= buf2;
                                    rd_start <= 1'b1;
                                end
                                else begin  //math is slower than buf
                                    row_full <= 1'b1;
                                end
                            end
                            else if(row_full && all_strides_done) begin
                            // We were paused, math just finished, resume!
                                row_full <= 1'b0;
                                current_row <= 2'd1;
                                lb_state <= buf2;
                                rd_start <= 1'b1;
                            end
                        end    
//--------------------------------------------
                        buf2 : begin
                            if(pointer2 ==  LB_size - 1 && s_axis_tvalid && WREADY2) begin
                                if(math_done || all_strides_done) begin  //after buffer is filled, if math is also done move on
                                    current_row <= 2'd2;
                                    lb_state <= buf3;
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
                                lb_state <= buf3;
                                rd_start <= 1'b1;
                            end
                        end
//--------------------------------------------
                        buf3 : begin
                            if(pointer3 ==  LB_size - 1 && s_axis_tvalid && WREADY3) begin
                                if(math_done || all_strides_done) begin  //after buffer is filled, if math is also done move on
                                    current_row <= 2'd3;
                                    lb_state <= buf4;
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
                                lb_state <= buf4;
                                rd_start <= 1'b1;
                            end
                        end
//--------------------------------------------
                        buf4 : begin
                            if(pointer4 ==  LB_size - 1 && s_axis_tvalid && WREADY4) begin
                                if(math_done || all_strides_done) begin  //after buffer is filled, if math is also done move on
                                    current_row <= 2'd0;
                                    lb_state <= buf1;
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
                                lb_state <= buf1;
                                rd_start <= 1'b1;
                            end
                        end                                                
                    endcase
                end
            endcase
        end
    end
    endmodule
