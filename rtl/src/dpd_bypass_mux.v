//==============================================================================
// 6G PA GAN-DPD: Bypass Multiplexer
//==============================================================================
//
// Description:
//   Output stage MUX selecting between DPD output and passthrough (ADC input).
//   Used for safety: if DPD diverges, passthrough ADC directly to DAC.
//
// Logic:
//   bypass_active=0: DAC ← DPD output (normal operation)
//   bypass_active=1: DAC ← ADC input (passthrough, DPD disabled)
//
// Author: Generated for 6G PA GAN-DPD Project
//==============================================================================

`timescale 1ns / 1ps

module dpd_bypass_mux (
    input  wire        clk_data,         // 250 MHz data clock
    input  wire        rst_n,
    
    // Input: ADC (from RF input path)
    input  wire signed [15:0] adc_i,
    input  wire signed [15:0] adc_q,
    
    // Input: DPD (from TDNN generator)
    input  wire signed [15:0] dpd_i,
    input  wire signed [15:0] dpd_q,
    
    // Control: bypass flag from safety monitor
    input  wire        bypass_active,
    
    // Output: to DAC
    output reg  signed [15:0] dac_i,
    output reg  signed [15:0] dac_q
);

    //==========================================================================
    // Multiplexer Logic
    //==========================================================================
    
    always @(posedge clk_data or negedge rst_n) begin
        if (!rst_n) begin
            dac_i <= 16'h0000;
            dac_q <= 16'h0000;
        end
        else begin
            if (bypass_active) begin
                // Bypass: pass ADC directly to DAC
                dac_i <= adc_i;
                dac_q <= adc_q;
            end
            else begin
                // Normal: use DPD output
                dac_i <= dpd_i;
                dac_q <= dpd_q;
            end
        end
    end

endmodule
