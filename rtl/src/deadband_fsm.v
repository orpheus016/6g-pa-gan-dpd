//==============================================================================
// 6G PA GAN-DPD: Deadband FSM for Jitter Prevention
//==============================================================================
//
// Description:
//   4-state FSM preventing SPSA jitter in steady-state by gating updates
//   based on EVM level. Implements hysteresis to prevent chattering.
//
// States:
//   IDLE:   EVM < -45 dB → SPSA disabled (a_k = 0)
//   TRACK:  -45 ≤ EVM < -35 dB → SPSA enabled, normal gain (a_k = 1×)
//   PANIC:  EVM ≥ -35 dB → SPSA enabled, high gain (a_k = 4×)
//   BYPASS: Overflow detected → All adaptation disabled
//
// Hysteresis: 5 dB band prevents rapid state oscillation
//
// Author: Generated for 6G PA GAN-DPD Project
//==============================================================================

`timescale 1ns / 1ps

module deadband_fsm (
    input  wire        clk,
    input  wire        rst_n,
    
    // Error metric input (Q8.8 format, in dB)
    input  wire signed [15:0] evm_db,
    
    // Safety override
    input  wire        overflow_flag,    // From safety_monitor
    input  wire        arm_reset,         // From ARM processor
    
    // State outputs
    output reg  [1:0]  state,            // 0=IDLE, 1=TRACK, 2=PANIC, 3=BYPASS
    output reg         spsa_enable,
    output reg  [1:0]  gain_mult         // 0=off, 1=1×, 2=4×
);

    //==========================================================================
    // State Encoding
    //==========================================================================
    
    localparam STATE_IDLE   = 2'd0;
    localparam STATE_TRACK  = 2'd1;
    localparam STATE_PANIC  = 2'd2;
    localparam STATE_BYPASS = 2'd3;
    
    //==========================================================================
    // Threshold Definitions (Q8.8 format, representing dB)
    //==========================================================================
    
    // -45 dB in Q8.8: -45 × 256 = -11520 = 0xD300 (as signed 16-bit)
    localparam signed [15:0] THRESH_IDLE_ENTER  = 16'hD300;    // -45 dB
    localparam signed [15:0] THRESH_IDLE_EXIT   = 16'hD100;    // -44 dB (hysteresis)
    
    // -40 dB in Q8.8: -40 × 256 = -10240 = 0xD800
    localparam signed [15:0] THRESH_TRACK_EXIT  = 16'hD800;    // -40 dB
    localparam signed [15:0] THRESH_TRACK_ENTER = 16'hDA00;    // -41 dB (hysteresis)
    
    // -35 dB in Q8.8: -35 × 256 = -8960 = 0xDD00
    localparam signed [15:0] THRESH_PANIC_ENTER = 16'hDD00;    // -35 dB
    localparam signed [15:0] THRESH_PANIC_EXIT  = 16'hDB00;    // -36 dB (hysteresis)
    
    //==========================================================================
    // Combinatorial Output Logic
    //==========================================================================
    
    always @(*) begin
        case (state)
            STATE_IDLE: begin
                spsa_enable = 1'b0;
                gain_mult = 2'd0;        // 0× (disabled)
            end
            STATE_TRACK: begin
                spsa_enable = 1'b1;
                gain_mult = 2'd1;        // 1× normal gain
            end
            STATE_PANIC: begin
                spsa_enable = 1'b1;
                gain_mult = 2'd2;        // 4× high gain (see aspsa_engine for interpretation)
            end
            STATE_BYPASS: begin
                spsa_enable = 1'b0;
                gain_mult = 2'd0;        // All disabled
            end
            default: begin
                spsa_enable = 1'b0;
                gain_mult = 2'd0;
            end
        endcase
    end
    
    //==========================================================================
    // State Machine
    //==========================================================================
    
    reg [1:0] next_state;
    
    always @(*) begin
        next_state = state;
        
        // Safety override: any overflow forces BYPASS
        if (overflow_flag) begin
            next_state = STATE_BYPASS;
        end
        else begin
            case (state)
                STATE_IDLE: begin
                    // Exit IDLE if EVM exceeds threshold with hysteresis
                    if (evm_db > THRESH_IDLE_EXIT) begin
                        next_state = STATE_TRACK;
                    end
                end
                
                STATE_TRACK: begin
                    // Go to PANIC if EVM worsens significantly
                    if (evm_db > THRESH_PANIC_ENTER) begin
                        next_state = STATE_PANIC;
                    end
                    // Go to IDLE if EVM improves significantly
                    else if (evm_db <= THRESH_IDLE_ENTER) begin
                        next_state = STATE_IDLE;
                    end
                end
                
                STATE_PANIC: begin
                    // Return to TRACK if EVM improves
                    if (evm_db <= THRESH_PANIC_EXIT) begin
                        next_state = STATE_TRACK;
                    end
                end
                
                STATE_BYPASS: begin
                    // Stay in BYPASS until ARM reset
                    if (arm_reset) begin
                        next_state = STATE_IDLE;
                    end
                end
                
                default: next_state = STATE_IDLE;
            endcase
        end
    end
    
    //==========================================================================
    // Sequential Logic
    //==========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
        end
        else begin
            state <= next_state;
        end
    end

endmodule
