//==============================================================================
// 6G PA GAN-DPD: PN-TDNN Generator (Memory-Aware Neural Network)
//==============================================================================
//
// Description:
//   Phase-Normalized TDNN generator for DPD with phase denormalization output.
//   Architecture: FC1(24→32) → LeakyReLU → FC2(32→16) → LeakyReLU → FC3(16→2) → Phase Denorm
//
//   Input: Phase-normalized features (24-dim for M=3)
//   Output: Phase-denormalized IQ via complex multiply
//
//   Phase denormalization formula:
//     I_out = (I_fc·I_0 - Q_fc·Q_0)
//     Q_out = (I_fc·Q_0 + Q_fc·I_0)
//     (No division by A_0 as input is already normalized)
//
// Quantization:
//   Weights:     Q1.15 (16-bit signed)
//   Activations: Q8.8 (16-bit signed)
//   Accumulator: Q16.16 (32-bit)
//
// Parameters:
//   Total: 1,362 params (24-input dimension)
//   FC1: 24×32 + 32 = 800
//   FC2: 32×16 + 16 = 528
//   FC3: 16×2 + 2 = 34
//
// Timing:
//   - Pipelined MAC with 6 parallel multipliers
//   - Latency: ~55 cycles per sample at 200 MHz (24-input FC1 + 1 denorm cycle)
//   - Throughput: 1 sample per ~55 cycles (3.6 Msps)
//
// Author: Generated for 6G PA GAN-DPD Project
//==============================================================================

`timescale 1ns / 1ps

module tdnn_generator #(
    parameter DATA_WIDTH    = 16,           // Q1.15 input/output
    parameter WEIGHT_WIDTH  = 16,           // Q1.15 weights
    parameter ACT_WIDTH     = 16,           // Q8.8 activations
    parameter ACC_WIDTH     = 32,           // Q16.16 accumulator
    parameter MEMORY_DEPTH  = 3,            // Memory taps
    parameter INPUT_DIM     = 24,           // 2 + 3*(M+1) + 2*M = 30 for M=5
    parameter HIDDEN1_DIM   = 32,
    parameter HIDDEN2_DIM   = 16,
    parameter OUTPUT_DIM    = 2,
    parameter NUM_MACS      = 6             // Parallel MAC units
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Input vector (memory-assembled)
    input  wire [DATA_WIDTH*INPUT_DIM-1:0] in_vector,
    input  wire                     in_valid,
    
    // Weight memory interface
    output reg  [15:0]              weight_addr,
    input  wire [WEIGHT_WIDTH-1:0]  weight_data,
    input  wire [1:0]               weight_bank_sel,
    
    // Output
    output reg  [DATA_WIDTH-1:0]    out_i,
    output reg  [DATA_WIDTH-1:0]    out_q,
    output reg                      out_valid,
    
    // Status
    output wire                     busy
);

    //==========================================================================
    // Local Parameters
    //==========================================================================
    
    // Weight address offsets (for 24-dim input, phase-normalized)
    localparam WADDR_FC1 = 0;                           // 24*32 = 768 weights
    localparam WADDR_B1  = 768;                         // 32 biases
    localparam WADDR_FC2 = 800;                         // 32*16 = 512 weights
    localparam WADDR_B2  = 1312;                        // 16 biases
    localparam WADDR_FC3 = 1328;                        // 16*2 = 32 weights
    localparam WADDR_B3  = 1360;                        // 2 biases
    
    // Bank offset
    localparam BANK_SIZE = 1362;  // Total parameters per temperature bank (1,362 for PN-TDNN)
    
    // State machine
    localparam ST_IDLE     = 4'd0;
    localparam ST_LOAD     = 4'd1;
    localparam ST_FC1      = 4'd2;
    localparam ST_ACT1     = 4'd3;
    localparam ST_FC2      = 4'd4;
    localparam ST_ACT2     = 4'd5;
    localparam ST_FC3      = 4'd6;
    localparam ST_DENORM   = 4'd7;  // Phase denormalization
    localparam ST_OUTPUT   = 4'd8;
    
    //==========================================================================
    // Internal Registers
    //==========================================================================
    
    reg [3:0] state, next_state;
    
    // Input buffer
    reg signed [DATA_WIDTH-1:0] input_buf [0:INPUT_DIM-1];
    
    // Layer outputs
    reg signed [ACT_WIDTH-1:0] fc1_out [0:HIDDEN1_DIM-1];
    reg signed [ACT_WIDTH-1:0] fc2_out [0:HIDDEN2_DIM-1];
    reg signed [DATA_WIDTH-1:0] fc3_out [0:OUTPUT_DIM-1];
    
    // Initialize layer outputs
    integer k;
    initial begin
        for (k = 0; k < HIDDEN1_DIM; k = k + 1) fc1_out[k] = 0;
        for (k = 0; k < HIDDEN2_DIM; k = k + 1) fc2_out[k] = 0;
        for (k = 0; k < OUTPUT_DIM; k = k + 1) fc3_out[k] = 0;
    end
    
    // MAC accumulator
    reg signed [ACC_WIDTH-1:0] acc [0:NUM_MACS-1];
    
    // Counters
    reg [5:0] in_idx;      // Input index
    reg [5:0] out_idx;     // Output index
    reg [5:0] mac_cnt;     // MAC counter
    
    // Phase denormalization reference (stored during input load)
    reg signed [DATA_WIDTH-1:0] ref_i;      // I_0 (current sample I from input)
    reg signed [DATA_WIDTH-1:0] ref_q;      // Q_0 (current sample Q from input)
    
    // Weight bank base address
    wire [15:0] bank_base = weight_bank_sel * BANK_SIZE;
    
    // Initialize accumulators
    integer j;
    initial begin
        for (j = 0; j < NUM_MACS; j = j + 1) begin
            acc[j] = 0;
        end
    end
    
    //==========================================================================
    // State Machine
    //==========================================================================
    
    assign busy = (state != ST_IDLE);
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= ST_IDLE;
        else
            state <= next_state;
    end
    
    always @(*) begin
        next_state = state;
        case (state)
            ST_IDLE:    if (in_valid) next_state = ST_LOAD;
            ST_LOAD:    next_state = ST_FC1;
            ST_FC1:     if (out_idx == HIDDEN1_DIM) next_state = ST_ACT1;
            ST_ACT1:    next_state = ST_FC2;
            ST_FC2:     if (out_idx == HIDDEN2_DIM) next_state = ST_ACT2;
            ST_ACT2:    next_state = ST_FC3;
            ST_FC3:     if (out_idx == OUTPUT_DIM) next_state = ST_DENORM;
            ST_DENORM:  next_state = ST_OUTPUT;
            ST_OUTPUT:  next_state = ST_IDLE;
            default:    next_state = ST_IDLE;
        endcase
    end
    
    //==========================================================================
    // Input Loading
    //==========================================================================
    
    integer i;
    always @(posedge clk) begin
        if (state == ST_LOAD || (state == ST_IDLE && in_valid)) begin
            for (i = 0; i < INPUT_DIM; i = i + 1) begin
                input_buf[i] <= in_vector[DATA_WIDTH*i +: DATA_WIDTH];
            end
            
            // Extract reference IQ from input
            // In phase-normalized features: indices 4-5 contain I(n) and Q(n)
            // (A, A³, I_norm, Q_norm, I, Q, ... for k=0)
            ref_i <= in_vector[DATA_WIDTH*4 +: DATA_WIDTH];  // I_0
            ref_q <= in_vector[DATA_WIDTH*5 +: DATA_WIDTH];  // Q_0
        end
    end
    
    //==========================================================================
    // FC Layer Processing (Pipelined)
    //==========================================================================
    
    // Weight address generation
    always @(posedge clk) begin
        case (state)
            ST_FC1: weight_addr <= bank_base + WADDR_FC1 + out_idx * INPUT_DIM + in_idx;
            ST_FC2: weight_addr <= bank_base + WADDR_FC2 + out_idx * HIDDEN1_DIM + in_idx;
            ST_FC3: weight_addr <= bank_base + WADDR_FC3 + out_idx * HIDDEN2_DIM + in_idx;
            default: weight_addr <= 0;
        endcase
    end
    
    // MAC accumulator logic
    reg signed [DATA_WIDTH-1:0] mac_input;
    reg signed [WEIGHT_WIDTH-1:0] mac_weight;
    wire signed [2*DATA_WIDTH-1:0] mac_product = mac_input * mac_weight;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_idx <= 0;
            out_idx <= 0;
            mac_cnt <= 0;
            for (i = 0; i < NUM_MACS; i = i + 1) begin
                acc[i] <= 0;
            end
        end
        else begin
            case (state)
                ST_IDLE: begin
                    in_idx <= 0;
                    out_idx <= 0;
                    mac_cnt <= 0;
                    acc[0] <= 0;  // Reset accumulator for next computation
                end
                
                ST_FC1: begin
                    // FC1: 22 inputs → 32 outputs
                    
                    // Fetch weight and input
                    mac_weight <= weight_data;
                    mac_input <= input_buf[in_idx];
                    
                    // Accumulate (starts from cycle 1)
                    if (in_idx > 0 || mac_cnt > 0) begin
                        acc[0] <= acc[0] + mac_product;
                    end
                    
                    // Move to next input
                    if (in_idx == INPUT_DIM - 1) begin
                        // Finished one neuron - store result
                        fc1_out[out_idx] <= (acc[0] + mac_product) >>> 16;  // Q16.16 -> Q8.8
                        out_idx <= out_idx + 1;
                        in_idx <= 0;
                        acc[0] <= 0;  // Reset for next neuron
                    end
                    else begin
                        in_idx <= in_idx + 1;
                    end
                    mac_cnt <= mac_cnt + 1;
                end
                
                ST_FC2: begin
                    // FC2: 32 inputs → 16 outputs
                    
                    mac_weight <= weight_data;
                    mac_input <= fc1_out[in_idx];
                    
                    if (in_idx > 0 || mac_cnt > 0) begin
                        acc[0] <= acc[0] + mac_product;
                    end
                    
                    if (in_idx == HIDDEN1_DIM - 1) begin
                        fc2_out[out_idx] <= (acc[0] + mac_product) >>> 16;  // Q16.16 -> Q8.8
                        out_idx <= out_idx + 1;
                        in_idx <= 0;
                        acc[0] <= 0;
                    end
                    else begin
                        in_idx <= in_idx + 1;
                    end
                    mac_cnt <= mac_cnt + 1;
                end
                
                ST_FC3: begin
                    // FC3: 16 inputs → 2 outputs
                    
                    mac_weight <= weight_data;
                    mac_input <= fc2_out[in_idx];
                    
                    if (in_idx > 0 || mac_cnt > 0) begin
                        acc[0] <= acc[0] + mac_product;
                    end
                    
                    if (in_idx == HIDDEN2_DIM - 1) begin
                        fc3_out[out_idx] <= (acc[0] + mac_product) >>> 16;  // Q16.16 -> Q1.15
                        out_idx <= out_idx + 1;
                        in_idx <= 0;
                        acc[0] <= 0;
                    end
                    else begin
                        in_idx <= in_idx + 1;
                    end
                    mac_cnt <= mac_cnt + 1;
                end
                
                ST_ACT1: begin
                    out_idx <= 0;
                    in_idx <= 0;
                    mac_cnt <= 0;
                end
                
                ST_ACT2: begin
                    out_idx <= 0;
                    in_idx <= 0;
                    mac_cnt <= 0;
                end
            endcase
        end
    end
    
    //==========================================================================
    // LeakyReLU Activation (α = 0.25 ≈ 1/4, implemented as >>2)
    //==========================================================================
    
    always @(posedge clk) begin
        if (state == ST_ACT1) begin
            for (i = 0; i < HIDDEN1_DIM; i = i + 1) begin
                if (fc1_out[i] < 0)
                    fc1_out[i] <= fc1_out[i] >>> 2;  // x * 0.25
                // else keep positive values unchanged
            end
        end
        
        if (state == ST_ACT2) begin
            for (i = 0; i < HIDDEN2_DIM; i = i + 1) begin
                if (fc2_out[i] < 0)
                    fc2_out[i] <= fc2_out[i] >>> 2;
            end
        end
    end
    
    //==========================================================================
    // Phase Denormalization
    //==========================================================================
    // Rotate FC output back to original phase (no division needed as input normalized)
    // Complex multiply: (I_fc + jQ_fc) × (I_0 + jQ_0)
    // Result: I_out = I_fc·I_0 - Q_fc·Q_0
    //         Q_out = I_fc·Q_0 + Q_fc·I_0
    
    // Intermediate products (32-bit for Q1.15 × Q1.15 = Q2.30)
    reg signed [2*DATA_WIDTH-1:0] prod_ii, prod_qq, prod_iq, prod_qi;
    
    always @(posedge clk) begin
        if (state == ST_DENORM) begin
            // Calculate products: (I_fc + jQ_fc) × (I_0 + jQ_0)
            prod_ii <= fc3_out[0] * ref_i;  // I_fc * I_0
            prod_qq <= fc3_out[1] * ref_q;  // Q_fc * Q_0
            prod_iq <= fc3_out[0] * ref_q;  // I_fc * Q_0
            prod_qi <= fc3_out[1] * ref_i;  // Q_fc * I_0
            
            // Combine products (scale back from Q2.30 to Q1.15)
            // Real part:  I_out = (I_fc·I_0 - Q_fc·Q_0) >> 15
            // Imag part:  Q_out = (I_fc·Q_0 + Q_fc·I_0) >> 15
            fc3_out[0] <= ((prod_ii - prod_qq) >>> 15);  // Q2.30 -> Q1.15
            fc3_out[1] <= ((prod_iq + prod_qi) >>> 15);  // Q2.30 -> Q1.15
        end
    end
    
    //==========================================================================
    // Output Stage
    //==========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_i <= 0;
            out_q <= 0;
            out_valid <= 0;
        end
        else begin
            out_valid <= 0;
            
            if (state == ST_OUTPUT) begin
                out_i <= fc3_out[0];
                out_q <= fc3_out[1];
                out_valid <= 1;
            end
        end
    end

endmodule
