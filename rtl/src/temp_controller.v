//==============================================================================
// 6G PA GAN-DPD: Temperature Controller
//==============================================================================
//
// Description:
//   Monitors temperature and controls weight bank selection.
//   Triggers annealing reset on temperature state transitions.
//
// Temperature States:
//   COLD   (< 15°C)  → Bank 0
//   NORMAL (15-40°C) → Bank 1
//   HOT    (> 40°C)  → Bank 2
//
// Author: Generated for 6G PA GAN-DPD Project
//==============================================================================

`timescale 1ns / 1ps

module temp_controller #(
    parameter COLD_THRESHOLD  = 12'd614,    // ~15°C (assuming 12-bit ADC, 0-100°C range)
    parameter HOT_THRESHOLD   = 12'd1638,   // ~40°C
    parameter HYSTERESIS      = 12'd41,     // ~1°C hysteresis
    parameter FILTER_DEPTH    = 8           // Moving average filter depth
)(
    input  wire         clk,                // 1 MHz clock
    input  wire         rst_n,
    
    // Temperature ADC input
    input  wire [11:0]  temp_adc,           // 12-bit temperature ADC
    input  wire         temp_valid,
    
    // Debug/override
    input  wire [1:0]   force_state,        // Force temperature state
    input  wire         force_enable,       // Enable state override
    
    // Outputs
    output reg  [1:0]   temp_state,         // 0=Cold, 1=Normal, 2=Hot
    output reg          temp_changed,       // Pulse on state change
    output reg          anneal_reset        // Reset A-SPSA annealing
);

    //==========================================================================
    // Local Parameters
    //==========================================================================
    
    localparam STATE_COLD   = 2'd0;
    localparam STATE_NORMAL = 2'd1;
    localparam STATE_HOT    = 2'd2;
    
    //==========================================================================
    // Moving Average Filter
    //==========================================================================
    
    reg [11:0] temp_buffer [0:FILTER_DEPTH-1];
    reg [3:0]  buffer_idx;
    reg [15:0] temp_sum;
    wire [11:0] temp_filtered = temp_sum >> 3;  // Divide by 8
    
    integer i;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buffer_idx <= 0;
            temp_sum <= 0;
            for (i = 0; i < FILTER_DEPTH; i = i + 1)
                temp_buffer[i] <= 12'd1024;  // Initialize to ~25°C
        end
        else if (temp_valid) begin
            // Update moving average
            temp_sum <= temp_sum - temp_buffer[buffer_idx] + temp_adc;
            temp_buffer[buffer_idx] <= temp_adc;
            buffer_idx <= (buffer_idx + 1) & (FILTER_DEPTH - 1);
        end
    end
    
    //==========================================================================
    // Temperature State Machine with Hysteresis
    //==========================================================================
    
    reg [1:0] prev_state;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            temp_state <= STATE_NORMAL;
            prev_state <= STATE_NORMAL;
            temp_changed <= 0;
            anneal_reset <= 0;
        end
        else begin
            temp_changed <= 0;
            anneal_reset <= 0;
            prev_state <= temp_state;
            
            if (force_enable) begin
                // Override mode
                temp_state <= force_state;
                if (temp_state != force_state) begin
                    temp_changed <= 1;
                    anneal_reset <= 1;
                end
            end
            else if (temp_valid) begin
                // State transitions with hysteresis
                case (temp_state)
                    STATE_COLD: begin
                        if (temp_filtered > (COLD_THRESHOLD + HYSTERESIS))
                            temp_state <= STATE_NORMAL;
                    end
                    
                    STATE_NORMAL: begin
                        if (temp_filtered < (COLD_THRESHOLD - HYSTERESIS))
                            temp_state <= STATE_COLD;
                        else if (temp_filtered > (HOT_THRESHOLD + HYSTERESIS))
                            temp_state <= STATE_HOT;
                    end
                    
                    STATE_HOT: begin
                        if (temp_filtered < (HOT_THRESHOLD - HYSTERESIS))
                            temp_state <= STATE_NORMAL;
                    end
                    
                    default: temp_state <= STATE_NORMAL;
                endcase
                
                // Detect state change
                if (temp_state != prev_state) begin
                    temp_changed <= 1;
                    anneal_reset <= 1;  // Reset annealing on any state change
                end
            end
        end
    end

endmodule
