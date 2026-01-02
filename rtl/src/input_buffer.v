//==============================================================================
// 6G PA GAN-DPD: Input Buffer with Memory Tap Assembly
//==============================================================================
//
// Description:
//   Buffers input IQ samples and assembles memory-aware input vector with
//   nonlinear feature extraction (|x|, |x|², |x|⁴).
//
//   Output: [I(n), Q(n), |x(n)|, |x(n)|², |x(n)|⁴, |x(n-1)|, |x(n-1)|², |x(n-1)|⁴, ...,
//            |x(n-M)|, |x(n-M)|², |x(n-M)|⁴, I(n-1), Q(n-1), ..., I(n-M), Q(n-M)]
//
// Author: Generated for 6G PA GAN-DPD Project
//==============================================================================

`timescale 1ns / 1ps

module input_buffer #(
    parameter DATA_WIDTH   = 16,            // Q1.15 samples
    parameter MEMORY_DEPTH = 5,             // Number of memory taps
    parameter OUTPUT_DIM   = 30             // 2 + 3*(M+1) + 2*M = 30 for M=5
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Input IQ samples
    input  wire [DATA_WIDTH-1:0]    in_i,
    input  wire [DATA_WIDTH-1:0]    in_q,
    input  wire                     in_valid,
    output wire                     in_ready,
    
    // Output: assembled memory vector
    output wire [DATA_WIDTH*OUTPUT_DIM-1:0] out_vector,
    output reg                      out_valid
);

    //==========================================================================
    // Internal Registers
    //==========================================================================
    
    // Shift registers for I, Q, envelope, and nonlinear features
    reg signed [DATA_WIDTH-1:0] i_buffer [0:MEMORY_DEPTH];
    reg signed [DATA_WIDTH-1:0] q_buffer [0:MEMORY_DEPTH];
    reg [DATA_WIDTH-1:0] env_buffer [0:MEMORY_DEPTH];
    reg [31:0] env_sq_buffer [0:MEMORY_DEPTH];    // |x|² (32-bit to avoid overflow)
    reg [31:0] env_4th_buffer [0:MEMORY_DEPTH];   // |x|⁴ (32-bit, truncated from 64-bit)
    
    // Sample counter (need M+1 samples before first valid output)
    reg [3:0] sample_cnt;
    wire buffer_ready = (sample_cnt >= MEMORY_DEPTH);
    
    // Ready when buffer has enough samples
    assign in_ready = 1'b1;  // Always ready (flow-through design)
    
    //==========================================================================
    // Envelope Calculation (|x| = sqrt(I² + Q²))
    //==========================================================================
    
    // Approximate magnitude using max approximation
    // |x| ≈ max(|I|, |Q|) (simplest, error ~30% worst case, 11% RMS)
    // Could upgrade to alpha-max-beta-min: |x| ≈ max + 0.375*min (error <3.5%)
    
    wire [DATA_WIDTH-1:0] abs_i = (in_i[DATA_WIDTH-1]) ? -in_i : in_i;
    wire [DATA_WIDTH-1:0] abs_q = (in_q[DATA_WIDTH-1]) ? -in_q : in_q;
    wire [DATA_WIDTH-1:0] envelope = (abs_i > abs_q) ? abs_i : abs_q;
    
    // Nonlinear features: |x|² and |x|⁴
    // Note: These use DSP blocks for multiply operations
    wire [31:0] envelope_sq = envelope * envelope;           // |x|²
    wire [63:0] envelope_sq_64 = envelope_sq * envelope_sq;  // |x|⁴ (64-bit)
    wire [31:0] envelope_4th = envelope_sq_64[47:16];        // Truncate to 32-bit (Q16.16)
    
    //==========================================================================
    // Shift Register Logic
    //==========================================================================
    
    integer i;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_cnt <= 0;
            out_valid <= 0;
            for (i = 0; i <= MEMORY_DEPTH; i = i + 1) begin
                i_buffer[i] <= 0;
                q_buffer[i] <= 0;
                env_buffer[i] <= 0;
                env_sq_buffer[i] <= 0;
                env_4th_buffer[i] <= 0;
            end
        end
        else begin
            out_valid <= 0;
            
            if (in_valid) begin
                // Shift in new sample
                for (i = MEMORY_DEPTH; i > 0; i = i - 1) begin
                    i_buffer[i] <= i_buffer[i-1];
                    q_buffer[i] <= q_buffer[i-1];
                    env_buffer[i] <= env_buffer[i-1];
                    env_sq_buffer[i] <= env_sq_buffer[i-1];
                    env_4th_buffer[i] <= env_4th_buffer[i-1];
                end
                i_buffer[0] <= in_i;
                q_buffer[0] <= in_q;
                env_buffer[0] <= envelope;
                env_sq_buffer[0] <= envelope_sq;
                env_4th_buffer[0] <= envelope_4th;
                
                // Update sample counter
                if (sample_cnt < MEMORY_DEPTH)
                    sample_cnt <= sample_cnt + 1;
                else
                    out_valid <= 1;  // Buffer full, valid output
            end
        end
    end
    
    //==========================================================================
    // Output Vector Assembly
    //==========================================================================
    
    // Output format: [I(n), Q(n), |x(n)|, |x(n)|², |x(n)|⁴, |x(n-1)|, |x(n-1)|², |x(n-1)|⁴, ...,
    //                 |x(n-M)|, |x(n-M)|², |x(n-M)|⁴, I(n-1), Q(n-1), ..., I(n-M), Q(n-M)]
    // Total: 2 + 3*(M+1) + 2*M = 2 + 18 + 10 = 30 for M=5
    
    // Current IQ: indices 0-1 (16-bit each)
    assign out_vector[DATA_WIDTH*1-1:DATA_WIDTH*0] = i_buffer[0];  // I(n)
    assign out_vector[DATA_WIDTH*2-1:DATA_WIDTH*1] = q_buffer[0];  // Q(n)
    
    // Nonlinear envelope features: indices 2 to 2+3*(M+1)-1 = 2 to 19
    // For each tap g=0 to M: |x(n-g)|, |x(n-g)|², |x(n-g)|⁴
    generate
        genvar g;
        for (g = 0; g <= MEMORY_DEPTH; g = g + 1) begin : gen_nonlinear
            // |x(n-g)| at position 2 + 3*g (16-bit)
            assign out_vector[DATA_WIDTH*(2+3*g+1)-1:DATA_WIDTH*(2+3*g)] = env_buffer[g];
            // |x(n-g)|² at position 2 + 3*g + 1 (16-bit, truncated from 32-bit)
            assign out_vector[DATA_WIDTH*(2+3*g+2)-1:DATA_WIDTH*(2+3*g+1)] = env_sq_buffer[g][31:16];
            // |x(n-g)|⁴ at position 2 + 3*g + 2 (16-bit, truncated from 32-bit)
            assign out_vector[DATA_WIDTH*(2+3*g+3)-1:DATA_WIDTH*(2+3*g+2)] = env_4th_buffer[g][31:16];
        end
    endgenerate
    
    // IQ memory taps: indices 20 to 29
    // Starting position: 2 + 3*(M+1) = 2 + 18 = 20
    generate
        for (g = 1; g <= MEMORY_DEPTH; g = g + 1) begin : gen_iq_mem
            // I(n-g) at position 20 + 2*(g-1)
            assign out_vector[DATA_WIDTH*(20+2*(g-1)+1)-1:DATA_WIDTH*(20+2*(g-1))] = i_buffer[g];
            // Q(n-g) at position 20 + 2*(g-1) + 1 = 21 + 2*(g-1)
            assign out_vector[DATA_WIDTH*(20+2*(g-1)+2)-1:DATA_WIDTH*(20+2*(g-1)+1)] = q_buffer[g];
        end
    endgenerate

endmodule
