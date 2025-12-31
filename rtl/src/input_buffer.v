//==============================================================================
// 6G PA GAN-DPD: Input Buffer with Memory Tap Assembly
//==============================================================================
//
// Description:
//   Buffers input IQ samples and assembles memory-aware input vector.
//   Output: [I(n), Q(n), |x(n)|, |x(n-1)|, ..., |x(n-M)|, I(n-1), Q(n-1), ..., I(n-M), Q(n-M)]
//
// Author: Generated for 6G PA GAN-DPD Project
//==============================================================================

`timescale 1ns / 1ps

module input_buffer #(
    parameter DATA_WIDTH   = 16,            // Q1.15 samples
    parameter MEMORY_DEPTH = 5,             // Number of memory taps
    parameter OUTPUT_DIM   = 18             // 2 + (M+1) + 2*M = 18 for M=5
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
    
    // Shift registers for I, Q, and envelope
    reg signed [DATA_WIDTH-1:0] i_buffer [0:MEMORY_DEPTH];
    reg signed [DATA_WIDTH-1:0] q_buffer [0:MEMORY_DEPTH];
    reg [DATA_WIDTH-1:0] env_buffer [0:MEMORY_DEPTH];
    
    // Sample counter (need M+1 samples before first valid output)
    reg [3:0] sample_cnt;
    wire buffer_ready = (sample_cnt >= MEMORY_DEPTH);
    
    // Ready when buffer has enough samples
    assign in_ready = 1'b1;  // Always ready (flow-through design)
    
    //==========================================================================
    // Envelope Calculation (|x| = sqrt(I² + Q²))
    //==========================================================================
    
    // Approximate magnitude using alpha-max-beta-min algorithm
    // |x| ≈ max(|I|, |Q|) + 0.375 * min(|I|, |Q|)
    // For simplicity, use |x| ≈ max(|I|, |Q|) (slight underestimate)
    
    wire [DATA_WIDTH-1:0] abs_i = (in_i[DATA_WIDTH-1]) ? -in_i : in_i;
    wire [DATA_WIDTH-1:0] abs_q = (in_q[DATA_WIDTH-1]) ? -in_q : in_q;
    wire [DATA_WIDTH-1:0] envelope = (abs_i > abs_q) ? abs_i : abs_q;
    
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
                end
                i_buffer[0] <= in_i;
                q_buffer[0] <= in_q;
                env_buffer[0] <= envelope;
                
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
    
    // Output format: [I(n), Q(n), |x(n)|, ..., |x(n-M)|, I(n-1), Q(n-1), ..., I(n-M), Q(n-M)]
    // Total: 2 + (M+1) + 2*M = 2 + 6 + 10 = 18 for M=5
    
    // Current IQ: indices 0-1
    assign out_vector[DATA_WIDTH*1-1:DATA_WIDTH*0] = i_buffer[0];  // I(n)
    assign out_vector[DATA_WIDTH*2-1:DATA_WIDTH*1] = q_buffer[0];  // Q(n)
    
    // Envelope memory: indices 2 to 2+M = 2 to 7
    generate
        genvar g;
        for (g = 0; g <= MEMORY_DEPTH; g = g + 1) begin : gen_env
            assign out_vector[DATA_WIDTH*(3+g)-1:DATA_WIDTH*(2+g)] = env_buffer[g];
        end
    endgenerate
    
    // IQ memory taps: indices 8 to 17
    generate
        for (g = 1; g <= MEMORY_DEPTH; g = g + 1) begin : gen_iq_mem
            // I(n-g) at position 2 + (M+1) + 2*(g-1) = 8 + 2*(g-1)
            assign out_vector[DATA_WIDTH*(8+2*(g-1)+1)-1:DATA_WIDTH*(8+2*(g-1))] = i_buffer[g];
            // Q(n-g) at position 8 + 2*(g-1) + 1 = 9 + 2*(g-1)
            assign out_vector[DATA_WIDTH*(8+2*(g-1)+2)-1:DATA_WIDTH*(8+2*(g-1)+1)] = q_buffer[g];
        end
    endgenerate

endmodule
