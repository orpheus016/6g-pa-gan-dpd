//==============================================================================
// 6G PA GAN-DPD: A-SPSA Engine (Annealing SPSA Weight Update)
//==============================================================================
//
// Description:
//   Simultaneous Perturbation Stochastic Approximation engine for online
//   weight adaptation. Uses annealing schedule for learning rate.
//
// Algorithm:
//   For each iteration k:
//   1. Generate Δk ∈ {-1, +1}^n using LFSR (Bernoulli)
//   2. Compute J(w + ck·Δk) and J(w - ck·Δk) using PA feedback
//   3. Gradient estimate: gk = [J(w+) - J(w-)] / (2·ck·Δk)
//   4. Weight update: w ← w - ak·gk
//
// Annealing (shift-register based):
//   ak = a0 >> (k / anneal_period)
//   ck = c0 >> (k / anneal_period)
//
// Author: Generated for 6G PA GAN-DPD Project
//==============================================================================

`timescale 1ns / 1ps

module aspsa_engine #(
    parameter WEIGHT_WIDTH = 16,            // Q1.15 weights
    parameter NUM_WEIGHTS  = 1170,          // Total weights
    parameter LR_WIDTH     = 16,            // Q0.16 learning rate
    parameter ANNEAL_PERIOD = 1000,         // Iterations per anneal step
    parameter MAX_ANNEAL_STEPS = 8,         // Maximum anneal steps
    parameter LFSR_SEED    = 16'hACE1       // LFSR initial seed
)(
    input  wire                     clk,            // 1 MHz clock
    input  wire                     rst_n,
    
    // Control
    input  wire                     enable,
    input  wire                     anneal_reset,   // Reset annealing (temp change)
    
    // Error input (from error_metric module)
    input  wire signed [15:0]       error_metric,   // Q8.8 error (e.g., EVM)
    input  wire                     error_valid,
    
    // Weight memory interface
    output reg  [15:0]              weight_addr,
    output reg  [WEIGHT_WIDTH-1:0]  weight_data,
    output reg                      weight_we,
    
    // Shadow memory handshake
    output reg                      update_req,
    input  wire                     update_ack,
    
    // Status
    output reg  [15:0]              iteration,
    output reg  [LR_WIDTH-1:0]      learning_rate,
    output wire                     busy
);

    //==========================================================================
    // Local Parameters
    //==========================================================================
    
    // Initial learning rate (Q0.16): 0.01 ≈ 655
    localparam [LR_WIDTH-1:0] LR_INITIAL = 16'd655;
    
    // Initial perturbation (Q0.16): 0.001 ≈ 65
    localparam [LR_WIDTH-1:0] PERT_INITIAL = 16'd65;
    
    // State machine
    localparam ST_IDLE        = 4'd0;
    localparam ST_PERTURB_POS = 4'd1;   // Apply +ck*Δk
    localparam ST_WAIT_POS    = 4'd2;   // Wait for J(w+)
    localparam ST_PERTURB_NEG = 4'd3;   // Apply -ck*Δk
    localparam ST_WAIT_NEG    = 4'd4;   // Wait for J(w-)
    localparam ST_GRADIENT    = 4'd5;   // Compute gradient
    localparam ST_UPDATE      = 4'd6;   // Update weights
    localparam ST_SYNC        = 4'd7;   // Request shadow memory sync
    localparam ST_WAIT_SYNC   = 4'd8;   // Wait for sync ack
    localparam ST_ANNEAL      = 4'd9;   // Check annealing schedule
    
    //==========================================================================
    // Internal Registers
    //==========================================================================
    
    reg [3:0] state, next_state;
    
    // LFSR for Bernoulli perturbation generation
    reg [15:0] lfsr;
    
    // Weight storage (current weights)
    reg signed [WEIGHT_WIDTH-1:0] weights [0:NUM_WEIGHTS-1];
    
    // Perturbation vector (±1 for each weight)
    reg delta [0:NUM_WEIGHTS-1];
    
    // Error measurements
    reg signed [15:0] error_pos;       // J(w + ck*Δk)
    reg signed [15:0] error_neg;       // J(w - ck*Δk)
    
    // Gradient estimate
    reg signed [23:0] gradient;
    
    // Counters
    reg [15:0] weight_idx;
    reg [15:0] anneal_cnt;
    reg [3:0]  anneal_step;
    
    // Learning rate and perturbation (annealed)
    reg [LR_WIDTH-1:0] pert_size;
    
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
            ST_IDLE:        if (enable && error_valid) next_state = ST_PERTURB_POS;
            ST_PERTURB_POS: if (weight_idx == NUM_WEIGHTS) next_state = ST_WAIT_POS;
            ST_WAIT_POS:    if (error_valid) next_state = ST_PERTURB_NEG;
            ST_PERTURB_NEG: if (weight_idx == NUM_WEIGHTS) next_state = ST_WAIT_NEG;
            ST_WAIT_NEG:    if (error_valid) next_state = ST_GRADIENT;
            ST_GRADIENT:    next_state = ST_UPDATE;
            ST_UPDATE:      if (weight_idx == NUM_WEIGHTS) next_state = ST_SYNC;
            ST_SYNC:        next_state = ST_WAIT_SYNC;
            ST_WAIT_SYNC:   if (update_ack) next_state = ST_ANNEAL;
            ST_ANNEAL:      next_state = ST_IDLE;
            default:        next_state = ST_IDLE;
        endcase
    end
    
    //==========================================================================
    // LFSR for Bernoulli Perturbation
    //==========================================================================
    
    // 16-bit LFSR with taps at positions 15, 14, 12, 3 (maximal length)
    wire lfsr_feedback = lfsr[15] ^ lfsr[14] ^ lfsr[12] ^ lfsr[3];
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            lfsr <= LFSR_SEED;
        else if (state == ST_PERTURB_POS || state == ST_PERTURB_NEG)
            lfsr <= {lfsr[14:0], lfsr_feedback};
    end
    
    //==========================================================================
    // Annealing Schedule
    //==========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || anneal_reset) begin
            learning_rate <= LR_INITIAL;
            pert_size <= PERT_INITIAL;
            anneal_cnt <= 0;
            anneal_step <= 0;
            iteration <= 0;
        end
        else if (state == ST_ANNEAL) begin
            iteration <= iteration + 1;
            anneal_cnt <= anneal_cnt + 1;
            
            // Check if time to anneal
            if (anneal_cnt >= ANNEAL_PERIOD && anneal_step < MAX_ANNEAL_STEPS) begin
                anneal_cnt <= 0;
                anneal_step <= anneal_step + 1;
                // Shift-based decay (divide by 2)
                learning_rate <= learning_rate >> 1;
                pert_size <= pert_size >> 1;
            end
        end
    end
    
    //==========================================================================
    // Weight Index Counter
    //==========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_idx <= 0;
        end
        else begin
            case (state)
                ST_IDLE: weight_idx <= 0;
                ST_PERTURB_POS, ST_PERTURB_NEG, ST_UPDATE:
                    if (weight_idx < NUM_WEIGHTS)
                        weight_idx <= weight_idx + 1;
                ST_WAIT_POS, ST_WAIT_NEG:
                    weight_idx <= 0;
                default: ;
            endcase
        end
    end
    
    //==========================================================================
    // Perturbation Generation and Weight Modification
    //==========================================================================
    
    integer i;
    
    always @(posedge clk) begin
        if (state == ST_PERTURB_POS && weight_idx < NUM_WEIGHTS) begin
            // Generate Δk from LFSR (LSB → ±1)
            delta[weight_idx] <= lfsr[0];
            
            // Apply positive perturbation: w + ck*Δk
            if (lfsr[0])
                weights[weight_idx] <= weights[weight_idx] + pert_size;
            else
                weights[weight_idx] <= weights[weight_idx] - pert_size;
        end
        
        if (state == ST_PERTURB_NEG && weight_idx < NUM_WEIGHTS) begin
            // Apply negative perturbation: w - 2*ck*Δk (to get w - ck*Δk from w + ck*Δk)
            if (delta[weight_idx])
                weights[weight_idx] <= weights[weight_idx] - (pert_size << 1);
            else
                weights[weight_idx] <= weights[weight_idx] + (pert_size << 1);
        end
    end
    
    //==========================================================================
    // Error Capture
    //==========================================================================
    
    always @(posedge clk) begin
        if (state == ST_WAIT_POS && error_valid)
            error_pos <= error_metric;
        if (state == ST_WAIT_NEG && error_valid)
            error_neg <= error_metric;
    end
    
    //==========================================================================
    // Gradient Estimation and Weight Update
    //==========================================================================
    
    // Gradient: g = (J+ - J-) / (2*ck)
    // Simplified: sign(J+ - J-) * Δk
    wire error_diff_sign = (error_pos > error_neg);
    
    always @(posedge clk) begin
        if (state == ST_UPDATE && weight_idx < NUM_WEIGHTS) begin
            // Update: w ← w + ck*Δk - ak * sign(J+-J-) * Δk
            // First, restore original weight (undo perturbation)
            if (delta[weight_idx])
                weights[weight_idx] <= weights[weight_idx] + pert_size;  // was at w - ck*Δk
            else
                weights[weight_idx] <= weights[weight_idx] - pert_size;
                
            // Then apply gradient update
            if (error_diff_sign) begin
                // J+ > J-, gradient positive, decrease weight in direction of Δk
                if (delta[weight_idx])
                    weights[weight_idx] <= weights[weight_idx] - learning_rate;
                else
                    weights[weight_idx] <= weights[weight_idx] + learning_rate;
            end
            else begin
                // J+ < J-, gradient negative, increase weight in direction of Δk
                if (delta[weight_idx])
                    weights[weight_idx] <= weights[weight_idx] + learning_rate;
                else
                    weights[weight_idx] <= weights[weight_idx] - learning_rate;
            end
        end
    end
    
    //==========================================================================
    // Weight Memory Interface
    //==========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_addr <= 0;
            weight_data <= 0;
            weight_we <= 0;
        end
        else begin
            weight_we <= 0;
            
            if (state == ST_UPDATE && weight_idx < NUM_WEIGHTS) begin
                weight_addr <= weight_idx;
                weight_data <= weights[weight_idx];
                weight_we <= 1;
            end
        end
    end
    
    //==========================================================================
    // Shadow Memory Sync Request
    //==========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            update_req <= 0;
        end
        else begin
            if (state == ST_SYNC)
                update_req <= 1;
            else if (update_ack)
                update_req <= 0;
        end
    end

endmodule
