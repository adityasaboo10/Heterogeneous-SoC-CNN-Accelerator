module Channel_Accumulator_6to1 #(
    parameter Output_Width = 32, 
    parameter NUM_ENGINES  = 6, 
    parameter PIXW         = 8
)(
    input                                       clk,
    input                                       rst_n,
    input                                       valid_in, // done signal from VE
    input signed [Output_Width-1:0]             in_ch0,   // Engine 1 / Bypass
    input signed [Output_Width-1:0]             in_ch1,   // Engine 2
    input signed [Output_Width-1:0]             in_ch2,   // Engine 3
    input signed [Output_Width-1:0]             in_ch3,   // Engine 4
    input signed [Output_Width-1:0]             in_ch4,   // Engine 5
    input signed [Output_Width-1:0]             in_ch5,   // Engine 6
    input signed [Output_Width-1:0]             bias,     // Added in ACCUMULATE mode
    input        [1:0]                          mode,
    input        [4:0]                          shift_amount,
    output reg signed [Output_Width-1:0]  accum_out,
    output reg [(NUM_ENGINES*PIXW)-1:0]         quantized_pixel_out,
    output reg                                  valid_out
);
   
    localparam MODE_ACCUMULATE        = 2'd0; // 6-ch sum + bias + ReLU + quant (Conv2)
    localparam MODE_BYPASS_RAW        = 2'd1; // Passthrough, no ReLU, no quant (debug)
    localparam MODE_BYPASS_QUANT      = 2'd2; // Passthrough + quant only (no ReLU)
    localparam MODE_BYPASS_RELU_QUANT = 2'd3; // Passthrough + ReLU + quant (Conv1)
    
    integer i;

    // =========================================================================
    // STAGE 1: 3 Parallel Adds + 6-Channel Bypass Registers
    // =========================================================================
    reg signed [Output_Width-1:0] sum1_0, sum1_1, sum1_2;
    reg signed [Output_Width-1:0] bypass_d1 [0:NUM_ENGINES-1];
    reg signed [Output_Width-1:0] bias_d1;
    reg [1:0] mode_d1;
    reg [4:0] shift_d1;
    reg       v_s1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum1_0   <= 0; sum1_1 <= 0; sum1_2 <= 0; 
            bias_d1  <= 0; mode_d1 <= 0; shift_d1 <= 0; v_s1 <= 1'b0;
            for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                bypass_d1[i] <= 0; 
            end
        end else begin
            sum1_0 <= in_ch0 + in_ch1;
            sum1_1 <= in_ch2 + in_ch3;
            sum1_2 <= in_ch4 + in_ch5;

            bypass_d1[0] <= in_ch0;
            bypass_d1[1] <= in_ch1;
            bypass_d1[2] <= in_ch2;
            bypass_d1[3] <= in_ch3;
            bypass_d1[4] <= in_ch4;
            bypass_d1[5] <= in_ch5;

            shift_d1 <= shift_amount;
            bias_d1  <= bias;
            mode_d1  <= mode;
            v_s1     <= valid_in;
        end
    end

    // =========================================================================
    // STAGE 2: 1 Add + Pipeline Alignment
    // =========================================================================
    reg signed [Output_Width-1:0] sum2_0, sum2_1;
    reg signed [Output_Width-1:0] bypass_d2 [0:NUM_ENGINES-1];
    reg signed [Output_Width-1:0] bias_d2;
    reg [1:0] mode_d2;
    reg [4:0] shift_d2;
    reg       v_s2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum2_0   <= 0; sum2_1 <= 0; 
            bias_d2  <= 0; mode_d2 <= 0; shift_d2 <= 0; v_s2 <= 1'b0;
            for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                bypass_d2[i] <= 0; 
            end 
        end else begin
            sum2_0 <= sum1_0 + sum1_1;
            sum2_1 <= sum1_2;

            for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                bypass_d2[i] <= bypass_d1[i];
            end

            bias_d2  <= bias_d1;
            mode_d2  <= mode_d1;
            shift_d2 <= shift_d1;
            v_s2     <= v_s1;
        end
    end

    // =========================================================================
    // STAGE 3: Final Sum, Bias, Mode Select, ReLU & Replicated Shift Regs
    // =========================================================================
    wire signed [Output_Width-1:0] accumulate_result = sum2_0 + sum2_1 + bias_d2;
    wire apply_relu_s2 = (mode_d2 == MODE_ACCUMULATE) || (mode_d2 == MODE_BYPASS_RELU_QUANT);

    reg signed [Output_Width-1:0] sel_val [0:NUM_ENGINES-1];
    always @(*) begin
        sel_val[0] = (mode_d2 == MODE_ACCUMULATE) ? accumulate_result : bypass_d2[0];
        for (i = 1; i < NUM_ENGINES; i = i + 1) begin
            sel_val[i] = (mode_d2 == MODE_ACCUMULATE) ? {Output_Width{1'b0}} : bypass_d2[i];
        end
    end

    // Registered Stage 3 signals
    reg signed [Output_Width-1:0] pre_quant_val [0:NUM_ENGINES-1];  //ReLU value
    reg       v_s3;

    // Per-channel replicated registers with constraints to kill high net delay
    (* max_fanout = 4 *) (* equivalent_register_removal = "no" *)
    reg [4:0] shift_tree [0:NUM_ENGINES-1];

    (* max_fanout = 4 *) (* equivalent_register_removal = "no" *)
    reg [1:0] mode_tree  [0:NUM_ENGINES-1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_s3 <= 1'b0;
            for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                pre_quant_val[i] <= {Output_Width{1'b0}};
                shift_tree[i]    <= 5'd0;
                mode_tree[i]     <= 2'd0;
            end
        end else begin
            v_s3 <= v_s2;

            for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                if (apply_relu_s2 && (sel_val[i] < 0))
                    pre_quant_val[i] <= {Output_Width{1'b0}};
                else
                    pre_quant_val[i] <= sel_val[i];

                // Local physical copies of shift and mode per channel
                shift_tree[i] <= shift_d2;
                mode_tree[i]  <= mode_d2;
            end
        end
    end

   // =========================================================================
    // STAGE 4: Clocked Variable Right-Shifter
    // =========================================================================
    reg signed [Output_Width-1:0] shifted_reg [0:NUM_ENGINES-1];
    reg signed [Output_Width-1:0] accum_d4;
    (* max_fanout = 4 *) (* equivalent_register_removal = "no" *)
    reg [1:0] mode_d4 [0:NUM_ENGINES-1];
    reg v_s4;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accum_d4 <= {Output_Width{1'b0}};
            v_s4     <= 1'b0;
            for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                shifted_reg[i] <= {Output_Width{1'b0}};
                mode_d4[i]     <= 2'd0;
            end
        end else begin
            accum_d4 <= pre_quant_val[0];
            v_s4     <= v_s3;

            for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                shifted_reg[i] <= pre_quant_val[i] >>> shift_tree[i];
                mode_d4[i]     <= mode_tree[i];
            end
        end
    end

    // =========================================================================
    // STAGE 5: Clocked Clamping to [0, 255] & Output Registers
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accum_out           <= {Output_Width{1'b0}};
            quantized_pixel_out <= {(NUM_ENGINES*PIXW){1'b0}};
            valid_out           <= 1'b0;
        end else begin
            valid_out <= v_s4;

            if (v_s4) begin
                accum_out <= accum_d4;

                for (i = 0; i < NUM_ENGINES; i = i + 1) begin
                    if (mode_d4[i] == MODE_BYPASS_RAW) begin
                        quantized_pixel_out[i*PIXW +: PIXW] <= {PIXW{1'b0}};
                    end else if (shifted_reg[i] > 32'sd255) begin
                        quantized_pixel_out[i*PIXW +: PIXW] <= 8'd255;
                    end else if (shifted_reg[i] < 0) begin
                        quantized_pixel_out[i*PIXW +: PIXW] <= 8'd0;
                    end else begin
                        quantized_pixel_out[i*PIXW +: PIXW] <= shifted_reg[i][7:0];
                    end
                end
            end else begin
                accum_out           <= {Output_Width{1'b0}};
                quantized_pixel_out <= {(NUM_ENGINES*PIXW){1'b0}};
            end
        end
    end
endmodule
