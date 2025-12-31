//==============================================================================
// Parameterized Fully Connected Layer for TDNN
// Supports variable input/output dimensions with pipelined MAC
//==============================================================================
// Features:
// - Configurable input/output dimensions
// - Pipelined multiply-accumulate for timing closure
// - Q1.15 weights, Q8.8 activations
// - Optional bias addition
// - LeakyReLU or Tanh activation selection
//==============================================================================

module fc_layer #(
    parameter INPUT_DIM = 18,
    parameter OUTPUT_DIM = 32,
    parameter DATA_WIDTH = 16,
    parameter WEIGHT_WIDTH = 16,
    parameter ACC_WIDTH = 32,
    parameter ACTIVATION = "LEAKYRELU",  // "LEAKYRELU", "TANH", "NONE"
    parameter WEIGHT_ADDR_WIDTH = 10,
    parameter BIAS_ADDR_WIDTH = 6
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Input vector (streamed)
    input  wire signed [DATA_WIDTH-1:0] in_data,
    input  wire                         in_valid,
    input  wire                         in_last,      // Last element of input vector
    
    // Output vector (streamed)
    output reg  signed [DATA_WIDTH-1:0] out_data,
    output reg                          out_valid,
    output reg                          out_last,     // Last element of output vector
    
    // Weight memory interface
    output wire [WEIGHT_ADDR_WIDTH-1:0] weight_addr,
    input  wire signed [WEIGHT_WIDTH-1:0] weight_data,
    
    // Bias memory interface
    output wire [BIAS_ADDR_WIDTH-1:0]   bias_addr,
    input  wire signed [DATA_WIDTH-1:0] bias_data,
    
    // Control
    input  wire                         layer_start,
    output reg                          layer_done
);

    //==========================================================================
    // Local Parameters
    //==========================================================================
    localparam INPUT_CNT_WIDTH = $clog2(INPUT_DIM + 1);
    localparam OUTPUT_CNT_WIDTH = $clog2(OUTPUT_DIM + 1);
    localparam SHIFT_NORM = 15;  // Weight Q1.15 normalization

    //==========================================================================
    // State Machine
    //==========================================================================
    localparam IDLE = 3'd0;
    localparam LOAD_INPUT = 3'd1;
    localparam COMPUTE = 3'd2;
    localparam ACTIVATE = 3'd3;
    localparam OUTPUT = 3'd4;
    
    reg [2:0] state, next_state;

    //==========================================================================
    // Counters and Control
    //==========================================================================
    reg [INPUT_CNT_WIDTH-1:0] in_cnt;
    reg [OUTPUT_CNT_WIDTH-1:0] out_cnt;
    reg [INPUT_CNT_WIDTH-1:0] mac_cnt;
    
    //==========================================================================
    // Input Buffer
    //==========================================================================
    reg signed [DATA_WIDTH-1:0] input_buf [0:INPUT_DIM-1];
    
    //==========================================================================
    // MAC Pipeline
    //==========================================================================
    reg signed [ACC_WIDTH-1:0] accumulator;
    reg signed [DATA_WIDTH-1:0] mult_a;
    reg signed [WEIGHT_WIDTH-1:0] mult_b;
    reg signed [ACC_WIDTH-1:0] mult_result;
    reg mac_valid_d1, mac_valid_d2;
    
    //==========================================================================
    // Weight Address Generation
    //==========================================================================
    reg [WEIGHT_ADDR_WIDTH-1:0] weight_addr_reg;
    assign weight_addr = weight_addr_reg;
    assign bias_addr = out_cnt[BIAS_ADDR_WIDTH-1:0];

    //==========================================================================
    // State Machine Logic
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end
    
    always @(*) begin
        next_state = state;
        case (state)
            IDLE: begin
                if (layer_start)
                    next_state = LOAD_INPUT;
            end
            
            LOAD_INPUT: begin
                if (in_valid && in_last)
                    next_state = COMPUTE;
            end
            
            COMPUTE: begin
                if (mac_cnt == INPUT_DIM - 1 && mac_valid_d2)
                    next_state = ACTIVATE;
            end
            
            ACTIVATE: begin
                next_state = OUTPUT;
            end
            
            OUTPUT: begin
                if (out_cnt == OUTPUT_DIM - 1)
                    next_state = IDLE;
                else
                    next_state = COMPUTE;
            end
            
            default: next_state = IDLE;
        endcase
    end

    //==========================================================================
    // Input Loading
    //==========================================================================
    integer i;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_cnt <= 0;
            for (i = 0; i < INPUT_DIM; i = i + 1)
                input_buf[i] <= 0;
        end else begin
            if (state == IDLE) begin
                in_cnt <= 0;
            end else if (state == LOAD_INPUT && in_valid) begin
                input_buf[in_cnt] <= in_data;
                in_cnt <= in_cnt + 1;
            end
        end
    end

    //==========================================================================
    // MAC Computation
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mac_cnt <= 0;
            accumulator <= 0;
            mult_a <= 0;
            mult_b <= 0;
            mult_result <= 0;
            mac_valid_d1 <= 1'b0;
            mac_valid_d2 <= 1'b0;
            out_cnt <= 0;
            weight_addr_reg <= 0;
        end else begin
            // Pipeline valid signals
            mac_valid_d2 <= mac_valid_d1;
            
            case (state)
                IDLE: begin
                    mac_cnt <= 0;
                    out_cnt <= 0;
                    accumulator <= 0;
                    weight_addr_reg <= 0;
                end
                
                COMPUTE: begin
                    // Stage 1: Read operands
                    mult_a <= input_buf[mac_cnt];
                    mult_b <= weight_data;
                    mac_valid_d1 <= 1'b1;
                    
                    // Update weight address for next cycle
                    weight_addr_reg <= out_cnt * INPUT_DIM + mac_cnt + 1;
                    
                    // Stage 2: Multiply
                    mult_result <= mult_a * mult_b;
                    
                    // Stage 3: Accumulate
                    if (mac_valid_d2) begin
                        if (mac_cnt == 0) begin
                            accumulator <= mult_result;
                        end else begin
                            accumulator <= accumulator + mult_result;
                        end
                    end
                    
                    // Update MAC counter
                    if (mac_cnt < INPUT_DIM - 1) begin
                        mac_cnt <= mac_cnt + 1;
                    end
                end
                
                ACTIVATE: begin
                    mac_valid_d1 <= 1'b0;
                    mac_cnt <= 0;
                    // Weight address for next output neuron
                    weight_addr_reg <= (out_cnt + 1) * INPUT_DIM;
                end
                
                OUTPUT: begin
                    out_cnt <= out_cnt + 1;
                    accumulator <= 0;
                end
            endcase
        end
    end

    //==========================================================================
    // Activation and Output
    //==========================================================================
    reg signed [DATA_WIDTH-1:0] normalized;
    reg signed [DATA_WIDTH-1:0] with_bias;
    reg signed [DATA_WIDTH-1:0] activated;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            normalized <= 0;
            with_bias <= 0;
            activated <= 0;
            out_data <= 0;
            out_valid <= 1'b0;
            out_last <= 1'b0;
            layer_done <= 1'b0;
        end else begin
            layer_done <= 1'b0;
            out_valid <= 1'b0;
            out_last <= 1'b0;
            
            if (state == ACTIVATE) begin
                // Normalize Q1.15 * Q8.8 = Q9.23 -> Q8.8
                normalized <= accumulator >>> SHIFT_NORM;
                
                // Add bias
                with_bias <= normalized + bias_data;
                
                // Apply activation
                case (ACTIVATION)
                    "LEAKYRELU": begin
                        // LeakyReLU: max(x, 0.25*x)
                        if (with_bias < 0)
                            activated <= with_bias >>> 2;  // 0.25 = >>2
                        else
                            activated <= with_bias;
                    end
                    
                    "TANH": begin
                        // Tanh: use LUT (connected externally)
                        // For now, pass through - actual tanh via tanh_lut module
                        activated <= with_bias;
                    end
                    
                    default: begin  // "NONE"
                        activated <= with_bias;
                    end
                endcase
            end
            
            if (state == OUTPUT) begin
                out_data <= activated;
                out_valid <= 1'b1;
                
                if (out_cnt == OUTPUT_DIM - 1) begin
                    out_last <= 1'b1;
                    layer_done <= 1'b1;
                end
            end
        end
    end

endmodule
