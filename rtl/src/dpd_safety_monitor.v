//==============================================================================
// 6G PA GAN-DPD: Safety Monitor (Real-Time Overflow Detection)
//==============================================================================
//
// Description:
//   Monitors DPD output magnitude at full data rate (250 MHz) for overflow.
//   Triggers bypass if output exceeds 87.5% of full scale for >1 cycle.
//
// Algorithm:
//   magnitude = |dpd_i| + |dpd_q|
//   threshold = 28672 (87.5% of 32768)
//   if magnitude > threshold: flag overflow, latch bypass_active
//
// Features:
//   - Debouncing: requires 2 consecutive cycles above threshold
//   - Latched output: once triggered, stays until ARM reset
//   - Overflow counter: tracks total violations
//
// Author: Generated for 6G PA GAN-DPD Project
//==============================================================================

`timescale 1ns / 1ps

module dpd_safety_monitor (
    input  wire        clk_data,         // 250 MHz data clock
    input  wire        rst_n,
    
    // DPD output (I/Q samples from TDNN)
    input  wire signed [15:0] dpd_i,
    input  wire signed [15:0] dpd_q,
    
    // Control
    input  wire        arm_reset,        // Reset latch from ARM
    
    // Status outputs
    output reg         bypass_active,    // Latched overflow flag
    output reg         overflow_alarm,   // Pulsed on overflow detection
    output reg  [15:0] overflow_count    // Statistics counter
);

    //==========================================================================
    // Local Parameters
    //==========================================================================
    
    // 87.5% of full scale (0.875 × 32768 ≈ 28672)
    localparam [15:0] MAGNITUDE_THRESHOLD = 16'd28672;
    
    // Debounce counter: require 2 consecutive violations
    localparam [1:0] DEBOUNCE_LIMIT = 2'd2;
    
    //==========================================================================
    // Internal Registers
    //==========================================================================
    
    reg [15:0] mag_i, mag_q;            // Absolute values
    reg [16:0] magnitude;                // Sum (17 bits to prevent overflow)
    reg        above_threshold;
    reg [1:0]  debounce_cnt;
    
    //==========================================================================
    // Magnitude Calculation
    //==========================================================================
    
    // Absolute value (take only magnitude, not sign)
    always @(*) begin
        mag_i = (dpd_i[15]) ? (~dpd_i + 1'b1) : dpd_i;
        mag_q = (dpd_q[15]) ? (~dpd_q + 1'b1) : dpd_q;
    end
    
    // L1 norm (Manhattan distance)
    always @(*) begin
        magnitude = {1'b0, mag_i} + {1'b0, mag_q};
    end
    
    // Threshold comparison
    always @(*) begin
        above_threshold = (magnitude > {1'b0, MAGNITUDE_THRESHOLD});
    end
    
    //==========================================================================
    // Debouncing and Latching
    //==========================================================================
    
    always @(posedge clk_data or negedge rst_n) begin
        if (!rst_n) begin
            debounce_cnt <= 2'd0;
            bypass_active <= 1'b0;
            overflow_alarm <= 1'b0;
            overflow_count <= 16'd0;
        end
        else begin
            overflow_alarm <= 1'b0;  // Default: no pulse
            
            // Debounce counter
            if (above_threshold) begin
                if (debounce_cnt < DEBOUNCE_LIMIT) begin
                    debounce_cnt <= debounce_cnt + 1;
                end
                else if (debounce_cnt == DEBOUNCE_LIMIT) begin
                    // Threshold crossed: trigger bypass
                    if (!bypass_active) begin
                        bypass_active <= 1'b1;
                        overflow_alarm <= 1'b1;
                        overflow_count <= overflow_count + 1;
                    end
                end
            end
            else begin
                debounce_cnt <= 2'd0;
            end
            
            // Reset latch from ARM
            if (arm_reset) begin
                bypass_active <= 1'b0;
                debounce_cnt <= 2'd0;
            end
        end
    end

endmodule
