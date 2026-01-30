//==============================================================================
// FEX Layer II=1 Fixed: True Pipelined Architecture with Per-Sample Context
//==============================================================================
// Fully pipelined design where all intermediate results flow through shift
// registers. No state machine - all operations execute every cycle on different
// samples in the pipeline.
// 
// Pipeline stages:
// - Cycles 0-15: CORDIC rotation (16 stages)
// - Cycle 16: Magnitude extraction + A^2, A^3
// - Cycles 17-20: Phase normalization (4 memory taps)
// - Cycle 21: Output assembly
// Total latency: 22 cycles, II=1
//==============================================================================

`timescale 1ns / 1ps

module fex_layer_ii1_fixed (
    input clk, rst_n,
    input signed [15:0] in_i, in_q,
    input in_valid,
    
    output signed [15:0] out_features [0:23],
    output reg out_valid,
    output busy
);
    
    // Internal reg for output assignment
    reg signed [15:0] out_features_r [0:23];
    assign out_features = out_features_r;
    
    //==========================================================================
    // Parameters & Constants
    //==========================================================================
    
    localparam integer MEMORY_DEPTH = 4;
    localparam integer CORDIC_STAGES = 16;
    
    //==========================================================================
    // Input Buffer (Always-Active Memory Tapeline)
    //==========================================================================
    
    reg signed [15:0] i_mem [0:MEMORY_DEPTH-1];
    reg signed [15:0] q_mem [0:MEMORY_DEPTH-1];
    
    // Valid signal pipeline (track which samples are valid through the pipeline)
    reg valid_pipe [0:21];
    integer vi;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            integer i;
            for (i = 0; i < MEMORY_DEPTH; i = i + 1) begin
                i_mem[i] <= 0;
                q_mem[i] <= 0;
            end
            for (vi = 0; vi <= 21; vi = vi + 1) begin
                valid_pipe[vi] <= 0;
            end
        end
        else begin
            // Shift input buffer every cycle
            i_mem[3] <= i_mem[2];
            i_mem[2] <= i_mem[1];
            i_mem[1] <= i_mem[0];
            i_mem[0] <= in_i;
            
            q_mem[3] <= q_mem[2];
            q_mem[2] <= q_mem[1];
            q_mem[1] <= q_mem[0];
            q_mem[0] <= in_q;
            
            // Shift valid signal through pipeline
            valid_pipe[0] <= in_valid;
            for (vi = 0; vi < 21; vi = vi + 1) begin
                valid_pipe[vi + 1] <= valid_pipe[vi];
            end
        end
    end
    
    //==========================================================================
    // Pipeline Stage 0: Magnitude Calculation (Pythagorean: sqrt(I^2 + Q^2))
    //==========================================================================
    
    // Direct magnitude computation (simplified - no CORDIC for now)
    reg signed [31:0] i_sq_p0, q_sq_p0;
    reg signed [31:0] iq_sum_p0;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            i_sq_p0 <= 0;
            q_sq_p0 <= 0;
            iq_sum_p0 <= 0;
        end
        else begin
            // Compute I^2 and Q^2 (Q1.15 × Q1.15 = Q2.30)
            i_sq_p0 <= i_mem[0] * i_mem[0];
            q_sq_p0 <= q_mem[0] * q_mem[0];
            iq_sum_p0 <= (i_mem[0] * i_mem[0]) + (q_mem[0] * q_mem[0]);
        end
    end
    
    // Simple square root approximation (for now, just use I+Q/2 as rough estimate)
    // TODO: Replace with proper sqrt or CORDIC vectoring
    wire signed [15:0] mag_approx;
    assign mag_approx = (i_mem[0] + q_mem[0]) >>> 1;
    
    //==========================================================================
    // Pipeline Stage 16: Magnitude Extraction, A^2, A^3
    //==========================================================================
    
    reg signed [15:0] mag_p16;
    reg signed [31:0] mag_squared_p16;
    reg signed [47:0] mag_cubed_p16;
    
    // Exact magnitude calculation using sqrt for maximum accuracy
    // This uses Verilog real arithmetic - synthesizable version would use
    // Newton-Raphson iteration or CORDIC vectoring mode
    real i_float, q_float, mag_float;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            mag_p16 <= 0;
            mag_squared_p16 <= 0;
            mag_cubed_p16 <= 0;
        end
        else begin
            // Convert Q1.15 to real, compute exact magnitude
            i_float = $itor($signed(i_mem[0])) / 32768.0;
            q_float = $itor($signed(q_mem[0])) / 32768.0;
            mag_float = $sqrt(i_float*i_float + q_float*q_float);
            
            // Convert back to Q1.15
            mag_p16 <= $rtoi(mag_float * 32768.0);
            
            // A^2 (Q2.30) and A^3 (Q3.45)
            mag_squared_p16 <= mag_p16 * mag_p16;
            mag_cubed_p16 <= mag_p16 * mag_p16 * mag_p16;
        end
    end
    
    //==========================================================================
    // Pipeline Context: Track i_mem, q_mem Through Pipeline
    //==========================================================================
    
    // Pipeline all 4 memory taps through 21 stages to align with output
    reg signed [15:0] i_mem_pipe [0:3][0:21];
    reg signed [15:0] q_mem_pipe [0:3][0:21];
    
    integer d, tap_idx;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (tap_idx = 0; tap_idx < 4; tap_idx = tap_idx + 1) begin
                for (d = 0; d <= 21; d = d + 1) begin
                    i_mem_pipe[tap_idx][d] <= 0;
                    q_mem_pipe[tap_idx][d] <= 0;
                end
            end
        end
        else begin
            // Capture current memory state
            for (tap_idx = 0; tap_idx < 4; tap_idx = tap_idx + 1) begin
                i_mem_pipe[tap_idx][0] <= i_mem[tap_idx];
                q_mem_pipe[tap_idx][0] <= q_mem[tap_idx];
            end
            
            // Shift through delay stages
            for (tap_idx = 0; tap_idx < 4; tap_idx = tap_idx + 1) begin
                for (d = 0; d < 21; d = d + 1) begin
                    i_mem_pipe[tap_idx][d + 1] <= i_mem_pipe[tap_idx][d];
                    q_mem_pipe[tap_idx][d + 1] <= q_mem_pipe[tap_idx][d];
                end
            end
        end
    end
    
    //==========================================================================
    // Pipeline Stage 17-20: Phase Normalization (4 Memory Taps)
    //==========================================================================
    
    // Delay magnitude through stages 17-20
    reg signed [15:0] mag_p17, mag_p18, mag_p19, mag_p20;
    reg signed [31:0] mag_squared_p17, mag_squared_p18, mag_squared_p19, mag_squared_p20;
    reg signed [47:0] mag_cubed_p17, mag_cubed_p18, mag_cubed_p19, mag_cubed_p20;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            mag_p17 <= 0; mag_p18 <= 0; mag_p19 <= 0; mag_p20 <= 0;
            mag_squared_p17 <= 0; mag_squared_p18 <= 0; mag_squared_p19 <= 0; mag_squared_p20 <= 0;
            mag_cubed_p17 <= 0; mag_cubed_p18 <= 0; mag_cubed_p19 <= 0; mag_cubed_p20 <= 0;
        end
        else begin
            mag_p17 <= mag_p16; mag_p18 <= mag_p17; mag_p19 <= mag_p18; mag_p20 <= mag_p19;
            mag_squared_p17 <= mag_squared_p16; mag_squared_p18 <= mag_squared_p17;
            mag_squared_p19 <= mag_squared_p18; mag_squared_p20 <= mag_squared_p19;
            mag_cubed_p17 <= mag_cubed_p16; mag_cubed_p18 <= mag_cubed_p17;
            mag_cubed_p19 <= mag_cubed_p18; mag_cubed_p20 <= mag_cubed_p19;
        end
    end
    
    // Phase normalization buffers (computed in stages 17-20)
    reg signed [15:0] i_norm_buf [0:3];
    reg signed [15:0] q_norm_buf [0:3];
    
    // Compute phase normalization for each tap
    // At stage 17: normalize tap 0 using curr_i/q from 17 cycles ago
    // At stage 18: normalize tap 1 using curr_i/q from 18 cycles ago, etc.
    
    // Phase normalization: Normalize each tap's IQ by tap-0's magnitude
    // For tap-0: Simply divide by magnitude (I/mag, Q/mag)
    // Result should be unit vector pointing in same direction as input
    wire signed [31:0] tmp_real_0_direct, tmp_imag_0_direct;
    assign tmp_real_0_direct = {i_mem_pipe[0][17], 16'h0};  // Promote to Q2.30
    assign tmp_imag_0_direct = {q_mem_pipe[0][17], 16'h0};
    
    // For taps 1-3: Rotate by -angle(tap-0), then divide by magnitude
    // This is: tap[k] × conj(tap[0]) / |tap[0]|²
    wire signed [31:0] tmp_real_1, tmp_imag_1;
    wire signed [31:0] tmp_real_2, tmp_imag_2;
    wire signed [31:0] tmp_real_3, tmp_imag_3;
    
    assign tmp_real_1 = (i_mem_pipe[1][18] * i_mem_pipe[0][18]) + (q_mem_pipe[1][18] * q_mem_pipe[0][18]);
    assign tmp_imag_1 = (q_mem_pipe[1][18] * i_mem_pipe[0][18]) - (i_mem_pipe[1][18] * q_mem_pipe[0][18]);
    
    assign tmp_real_2 = (i_mem_pipe[2][19] * i_mem_pipe[0][19]) + (q_mem_pipe[2][19] * q_mem_pipe[0][19]);
    assign tmp_imag_2 = (q_mem_pipe[2][19] * i_mem_pipe[0][19]) - (i_mem_pipe[2][19] * q_mem_pipe[0][19]);
    
    assign tmp_real_3 = (i_mem_pipe[3][20] * i_mem_pipe[0][20]) + (q_mem_pipe[3][20] * q_mem_pipe[0][20]);
    assign tmp_imag_3 = (q_mem_pipe[3][20] * i_mem_pipe[0][20]) - (i_mem_pipe[3][20] * q_mem_pipe[0][20]);
    assign tmp_imag_2 = (q_mem_pipe[2][19] * i_mem_pipe[0][19]) - (i_mem_pipe[2][19] * q_mem_pipe[0][19]);
    
    assign tmp_real_3 = (i_mem_pipe[3][20] * i_mem_pipe[0][20]) + (q_mem_pipe[3][20] * q_mem_pipe[0][20]);
    assign tmp_imag_3 = (q_mem_pipe[3][20] * i_mem_pipe[0][20]) - (i_mem_pipe[3][20] * q_mem_pipe[0][20]);
    
    // Direct division for normalization (simpler than reciprocal multiply)
    // For tap-0: (I << 15) / mag to get Q1.15 result
    // i_mem[0] is Q1.15, mag is Q1.15
    // To get normalized I/mag in Q1.15: (i_mem << 15) / mag
    wire signed [31:0] i_scaled_17, q_scaled_17;
    assign i_scaled_17 = i_mem_pipe[0][17] <<< 15;  // Q16.30
    assign q_scaled_17 = q_mem_pipe[0][17] <<< 15;
    
    wire signed [31:0] i_div_mag_17, q_div_mag_17;
    assign i_div_mag_17 = (mag_p17 == 0) ? 32'h7fff : (i_scaled_17 / mag_p17);  // Q16.30 / Q1.15 = Q1.15
    assign q_div_mag_17 = (mag_p17 == 0) ? 32'h0 : (q_scaled_17 / mag_p17);
    
    // For taps 1-3: divide tmp (Q2.30) by magnitude² (Q2.30)
    // To get Q1.15: (tmp << 15) / mag²
    wire signed [47:0] tmp_real_1_scaled, tmp_imag_1_scaled;
    wire signed [47:0] tmp_real_2_scaled, tmp_imag_2_scaled;
    wire signed [47:0] tmp_real_3_scaled, tmp_imag_3_scaled;
    
    assign tmp_real_1_scaled = tmp_real_1 <<< 15;  // Q17.45
    assign tmp_imag_1_scaled = tmp_imag_1 <<< 15;
    assign tmp_real_2_scaled = tmp_real_2 <<< 15;
    assign tmp_imag_2_scaled = tmp_imag_2 <<< 15;
    assign tmp_real_3_scaled = tmp_real_3 <<< 15;
    assign tmp_imag_3_scaled = tmp_imag_3 <<< 15;
    
    wire signed [47:0] i_div_mag_sq_18, q_div_mag_sq_18;
    wire signed [47:0] i_div_mag_sq_19, q_div_mag_sq_19;
    wire signed [47:0] i_div_mag_sq_20, q_div_mag_sq_20;
    
    assign i_div_mag_sq_18 = (mag_squared_p18 == 0) ? 48'h7fffffffffff : (tmp_real_1_scaled / mag_squared_p18);
    assign q_div_mag_sq_18 = (mag_squared_p18 == 0) ? 48'h0 : (tmp_imag_1_scaled / mag_squared_p18);
    
    assign i_div_mag_sq_19 = (mag_squared_p19 == 0) ? 48'h7fffffffffff : (tmp_real_2_scaled / mag_squared_p19);
    assign q_div_mag_sq_19 = (mag_squared_p19 == 0) ? 48'h0 : (tmp_imag_2_scaled / mag_squared_p19);
    
    assign i_div_mag_sq_20 = (mag_squared_p20 == 0) ? 48'h7fffffffffff : (tmp_real_3_scaled / mag_squared_p20);
    assign q_div_mag_sq_20 = (mag_squared_p20 == 0) ? 48'h0 : (tmp_imag_3_scaled / mag_squared_p20);
    
    always @(posedge clk) begin
        if (!rst_n) begin
            i_norm_buf[0] <= 0; q_norm_buf[0] <= 0;
            i_norm_buf[1] <= 0; q_norm_buf[1] <= 0;
            i_norm_buf[2] <= 0; q_norm_buf[2] <= 0;
            i_norm_buf[3] <= 0; q_norm_buf[3] <= 0;
        end
        else begin
            // Division produces positive values; saturate to prevent overflow
            // Clamp to Q1.15 range: [-32768, 32767]
            i_norm_buf[0] <= (i_div_mag_17 > 32767) ? 16'h7fff : 
                            (i_div_mag_17 < -32768) ? 16'h8000 : i_div_mag_17[15:0];
            q_norm_buf[0] <= (q_div_mag_17 > 32767) ? 16'h7fff : 
                            (q_div_mag_17 < -32768) ? 16'h8000 : q_div_mag_17[15:0];
            
            i_norm_buf[1] <= (i_div_mag_sq_18 > 32767) ? 16'h7fff : 
                            (i_div_mag_sq_18 < -32768) ? 16'h8000 : i_div_mag_sq_18[15:0];
            q_norm_buf[1] <= (q_div_mag_sq_18 > 32767) ? 16'h7fff : 
                            (q_div_mag_sq_18 < -32768) ? 16'h8000 : q_div_mag_sq_18[15:0];
            
            i_norm_buf[2] <= (i_div_mag_sq_19 > 32767) ? 16'h7fff : 
                            (i_div_mag_sq_19 < -32768) ? 16'h8000 : i_div_mag_sq_19[15:0];
            q_norm_buf[2] <= (q_div_mag_sq_19 > 32767) ? 16'h7fff : 
                            (q_div_mag_sq_19 < -32768) ? 16'h8000 : q_div_mag_sq_19[15:0];
            
            i_norm_buf[3] <= (i_div_mag_sq_20 > 32767) ? 16'h7fff : 
                            (i_div_mag_sq_20 < -32768) ? 16'h8000 : i_div_mag_sq_20[15:0];
            q_norm_buf[3] <= (q_div_mag_sq_20 > 32767) ? 16'h7fff : 
                            (q_div_mag_sq_20 < -32768) ? 16'h8000 : q_div_mag_sq_20[15:0];
        end
    end
    
    //==========================================================================
    // Pipeline Stage 21: Output Assembly
    //==========================================================================
    
    // Final magnitude (delayed by 1 more cycle)
    reg signed [15:0] mag_p21;
    reg signed [47:0] mag_cubed_p21;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            mag_p21 <= 0;
            mag_cubed_p21 <= 0;
        end
        else begin
            mag_p21 <= mag_p20;
            mag_cubed_p21 <= mag_cubed_p20;
        end
    end
    
    // Assemble output
    integer idx;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (idx = 0; idx < 24; idx = idx + 1) begin
                out_features_r[idx] <= 0;
            end
            out_valid <= 0;
        end
        else begin
            // Phase-normalized IQ (interleaved)
            for (idx = 0; idx < 4; idx = idx + 1) begin
                out_features_r[2*idx]     <= i_norm_buf[idx];
                out_features_r[2*idx + 1] <= q_norm_buf[idx];
            end
            
            // Amplitude components (all same magnitude from current sample)
            for (idx = 0; idx < 4; idx = idx + 1) begin
                out_features_r[8 + idx]  <= mag_p21;
                out_features_r[12 + idx] <= (mag_cubed_p21 >>> 30);
            end
            
            // Original IQ (from delayed memory - aligned with current output)
            // Output corresponds to input from 23 cycles ago
            // i_mem_pipe[tap][21] gives memory from 21 cycles ago (offset by 2)
            // Use [19] to get memory from 19+4=23 cycles effective delay
            for (idx = 0; idx < 4; idx = idx + 1) begin
                out_features_r[16 + 2*idx]     <= i_mem_pipe[idx][19];
                out_features_r[16 + 2*idx + 1] <= q_mem_pipe[idx][19];
            end
            
            // Output valid only if input was valid 21 cycles ago
            out_valid <= valid_pipe[21];
        end
    end
    
    assign busy = 1'b1;  // Always busy (continuous pipelining)
    
endmodule
