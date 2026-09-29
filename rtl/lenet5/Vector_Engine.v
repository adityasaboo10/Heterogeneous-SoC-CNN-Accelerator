(* keep_hierarchy = "yes" *) module Vector_Engine #(parameter PIXW = 8, 
                           parameter Output_Width = 32,
                           parameter Kernel_size = 3
                           )(
                           input clk_50M,
                           input rst,
                           input en, //Engine Level
                           input [(Kernel_size * Kernel_size * PIXW)-1 : 0] ifMAP_flat,
                           input [(Kernel_size * Kernel_size * PIXW)-1 : 0] weights,
                           input [ Output_Width-1 : 0] bias,
                           output reg signed [Output_Width-1 : 0] out,
                           output done
        );
    // Replicate enable register inside each Vector Engine to keep local fanout <= 9
    /*
    The master signal drives en_local inside the Vector Engine. 
    Then, en_local acts as a local distributor, driving the actual logic (MAC units)
    only within its specific engine. This physically isolates the engine's 
    internal timing from the rest of the chip.
    */
//    (* max_fanout = 16 *) (* equivalent_register_removal = "no" *)
//    reg en_local;
    

    wire done_arr [0:(Kernel_size*Kernel_size)-1];
 
    wire signed [Output_Width-1 : 0] mac_outputs [0 : (Kernel_size*Kernel_size)-1];
    genvar row, col;
    
    generate 
    for( row = 0; row < Kernel_size; row=row+1) begin
        for(col = 0 ; col < Kernel_size; col = col+1)begin
        localparam integer idx = (row * Kernel_size) + col;
            PE#(
                .PIXW(PIXW),
                .Output_Width (Output_Width)
            ) u_pe (
                .clk_50M(clk_50M),
                .rst(rst),
                .a(ifMAP_flat[((idx+1)*PIXW)-1: idx*PIXW]),
                .b(weights[((idx+1)*PIXW)-1: idx*PIXW]),
                .en(en),
                .done(done_arr[idx]),
                .out( mac_outputs[idx])
                    );
        end
    end
    endgenerate
    
    // ============================================================
    // PIPELINED ADDER TREE (K=3, N=9) - replaces the old ripple sum
    // Stage1: 4 parallel adds + 1 passthrough  -> 5 values
    // Stage2: 2 parallel adds + 1 passthrough  -> 3 values
    // Stage3: 1 add + 1 passthrough            -> 2 values
    // Stage4: final add -> out
    // "valid" shadows the data through the same 4 stages so `done`
    // stays aligned with `out`.
    // ============================================================
    reg signed [Output_Width-1:0] s1 [0:4];
    reg signed [Output_Width-1:0] s2 [0:2];
    reg signed [Output_Width-1:0] s3 [0:1];
    reg v1, v2, v3, v4;
    reg signed [Output_Width-1:0] bias_d1, bias_d2, bias_d3;

    always @(posedge clk_50M or negedge rst) begin
        if (!rst) begin
            s1[0]<=0; s1[1]<=0; s1[2]<=0; s1[3]<=0; s1[4]<=0; v1<=0;
        end else begin
            s1[0] <= mac_outputs[0] + mac_outputs[1];
            s1[1] <= mac_outputs[2] + mac_outputs[3];
            s1[2] <= mac_outputs[4] + mac_outputs[5];
            s1[3] <= mac_outputs[6] + mac_outputs[7];
            s1[4] <= mac_outputs[8];              // odd one out, passthrough
            bias_d1 <= bias; 
            v1    <= done_arr[0];
        end
    end

    always @(posedge clk_50M or negedge rst) begin
        if (!rst) begin
            s2[0]<=0; s2[1]<=0; s2[2]<=0; v2<=0;
        end else begin
            s2[0] <= s1[0] + s1[1];
            s2[1] <= s1[2] + s1[3];
            s2[2] <= s1[4];
            v2    <= v1;
            bias_d2 <= bias_d1; 
        end
    end

    always @(posedge clk_50M or negedge rst) begin
        if (!rst) begin
            s3[0]<=0; s3[1]<=0; v3<=0;
        end else begin
            s3[0] <= s2[0] + s2[1];
            s3[1] <= s2[2];
            v3    <= v2;
             bias_d3 <= bias_d2; 
        end
    end

    always @(posedge clk_50M or negedge rst) begin
        if (!rst) begin
            out <= 0; v4 <= 0;
        end else begin
            out <= s3[0] + s3[1] + bias_d3;   // final sum
            v4  <= v3;
        end
    end

    assign done = v4;
endmodule
