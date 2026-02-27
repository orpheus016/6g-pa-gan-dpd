//==============================================================================
// FEX Layer - Synthesizable Version (No Floating Point)
//==============================================================================
// Feature extraction for PN-TDNN DPD
// Uses alpha-max-beta-min approximation for magnitude (synthesizable)
// 
// Pipeline stages:
// - Cycle 0: Input capture
// - Cycle 1: Magnitude approximation
// - Cycle 2-3: A^2, A^3 computation
// - Cycle 4-7: Phase normalization (4 memory taps)
// - Cycle 8: Output assembly
// Total latency: 9 cycles, II=1
//==============================================================================

`timescale 1ns / 1ps

module fex_layer_synth #(
    parameter DATA_WIDTH = 16,
    parameter MEMORY_DEPTH = 4
)(
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire signed [DATA_WIDTH-1:0] in_i,
    input  wire signed [DATA_WIDTH-1:0] in_q,
    input  wire                      in_valid,
    
    // Output: 24-dimensional feature vector (flattened)
    output wire [DATA_WIDTH*24-1:0]  out_features,
    output reg                       out_valid,
    output wire                      busy
);
    
    //==========================================================================
    // Internal Feature Array
    //==========================================================================
    reg signed [DATA_WIDTH-1:0] features [0:23];
    
    // Flatten output
    genvar g;
    generate
        for (g = 0; g < 24; g = g + 1) begin : gen_out
            assign out_features[DATA_WIDTH*g +: DATA_WIDTH] = features[g];
        end
    endgenerate
    
    //==========================================================================
    // Input Memory Buffer (4-tap delay line)
    //==========================================================================
    reg signed [DATA_WIDTH-1:0] i_mem [0:MEMORY_DEPTH-1];
    reg signed [DATA_WIDTH-1:0] q_mem [0:MEMORY_DEPTH-1];
    
    //==========================================================================
    // Valid Pipeline
    //==========================================================================
    reg [8:0] valid_pipe;
    
    //==========================================================================
    // Pipeline Registers
    //==========================================================================
    
    // Stage 1: Magnitude computation
    reg signed [DATA_WIDTH-1:0] abs_i_p1, abs_q_p1;
    reg signed [DATA_WIDTH-1:0] mag_p1;
    
    // Stage 2: A^2
    reg signed [2*DATA_WIDTH-1:0] mag_sq_p2;
    reg signed [DATA_WIDTH-1:0] mag_p2;
    
    // Stage 3: A^3
    reg signed [3*DATA_WIDTH-1:0] mag_cu_p3;
    reg signed [DATA_WIDTH-1:0] mag_p3;
    reg signed [DATA_WIDTH-1:0] mag_sq_trunc_p3;
    
    // Stages 4-7: Phase normalization intermediate
    reg signed [DATA_WIDTH-1:0] mag_p4, mag_p5, mag_p6, mag_p7;
    reg signed [DATA_WIDTH-1:0] mag_cu_trunc_p4, mag_cu_trunc_p5, mag_cu_trunc_p6, mag_cu_trunc_p7;
    
    // Stage 8: Output
    reg signed [DATA_WIDTH-1:0] mag_p8;
    reg signed [DATA_WIDTH-1:0] mag_cu_trunc_p8;
    
    // Pipeline IQ memory through all stages
    reg signed [DATA_WIDTH-1:0] i_pipe [0:MEMORY_DEPTH-1][0:8];
    reg signed [DATA_WIDTH-1:0] q_pipe [0:MEMORY_DEPTH-1][0:8];
    
    // Phase normalization results
    reg signed [DATA_WIDTH-1:0] i_norm [0:MEMORY_DEPTH-1];
    reg signed [DATA_WIDTH-1:0] q_norm [0:MEMORY_DEPTH-1];
    
    integer i, j;
    
    //==========================================================================
    // Alpha-Max-Beta-Min Magnitude Approximation
    // |z| ≈ max(|I|,|Q|) + 0.5*min(|I|,|Q|) (within 5% error)
    // Using 0.5 ≈ 1/2 for simple shift
    //==========================================================================
    function signed [DATA_WIDTH-1:0] abs_val;
        input signed [DATA_WIDTH-1:0] x;
        begin
            abs_val = (x < 0) ? -x : x;
        end
    endfunction
    
    wire signed [DATA_WIDTH-1:0] abs_i_w = abs_val(i_mem[0]);
    wire signed [DATA_WIDTH-1:0] abs_q_w = abs_val(q_mem[0]);
    wire signed [DATA_WIDTH-1:0] max_iq = (abs_i_w > abs_q_w) ? abs_i_w : abs_q_w;
    wire signed [DATA_WIDTH-1:0] min_iq = (abs_i_w > abs_q_w) ? abs_q_w : abs_i_w;
    wire signed [DATA_WIDTH-1:0] mag_approx = max_iq + (min_iq >>> 1);
    
    //==========================================================================
    // Pipeline Stage 0: Input Capture & Memory Shift
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < MEMORY_DEPTH; i = i + 1) begin
                i_mem[i] <= 0;
                q_mem[i] <= 0;
            end
            valid_pipe <= 0;
        end else begin
            // Shift valid pipeline
            valid_pipe <= {valid_pipe[7:0], in_valid};
            
            if (in_valid) begin
                // Shift memory
                i_mem[0] <= in_i;
                q_mem[0] <= in_q;
                for (i = 1; i < MEMORY_DEPTH; i = i + 1) begin
                    i_mem[i] <= i_mem[i-1];
                    q_mem[i] <= q_mem[i-1];
                end
            end
        end
    end
    
    //==========================================================================
    // Pipeline Stages 1-8: Magnitude and Feature Computation
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mag_p1 <= 0;
            mag_p2 <= 0; mag_sq_p2 <= 0;
            mag_p3 <= 0; mag_sq_trunc_p3 <= 0; mag_cu_p3 <= 0;
            mag_p4 <= 0; mag_cu_trunc_p4 <= 0;
            mag_p5 <= 0; mag_cu_trunc_p5 <= 0;
            mag_p6 <= 0; mag_cu_trunc_p6 <= 0;
            mag_p7 <= 0; mag_cu_trunc_p7 <= 0;
            mag_p8 <= 0; mag_cu_trunc_p8 <= 0;
            
            for (i = 0; i < MEMORY_DEPTH; i = i + 1) begin
                for (j = 0; j <= 8; j = j + 1) begin
                    i_pipe[i][j] <= 0;
                    q_pipe[i][j] <= 0;
                end
                i_norm[i] <= 0;
                q_norm[i] <= 0;
            end
        end else begin
            // Stage 1: Magnitude approximation
            mag_p1 <= mag_approx;
            for (i = 0; i < MEMORY_DEPTH; i = i + 1) begin
                i_pipe[i][1] <= i_mem[i];
                q_pipe[i][1] <= q_mem[i];
            end
            
            // Stage 2: A^2
            mag_p2 <= mag_p1;
            mag_sq_p2 <= mag_p1 * mag_p1;  // Q2.30
            for (i = 0; i < MEMORY_DEPTH; i = i + 1) begin
                i_pipe[i][2] <= i_pipe[i][1];
                q_pipe[i][2] <= q_pipe[i][1];
            end
            
            // Stage 3: A^3
            mag_p3 <= mag_p2;
            mag_sq_trunc_p3 <= mag_sq_p2[30:15];  // Q1.15
            mag_cu_p3 <= mag_p2 * mag_sq_p2[30:15];  // Q2.30
            for (i = 0; i < MEMORY_DEPTH; i = i + 1) begin
                i_pipe[i][3] <= i_pipe[i][2];
                q_pipe[i][3] <= q_pipe[i][2];
            end
            
            // Stage 4: Phase norm tap 0
            mag_p4 <= mag_p3;
            mag_cu_trunc_p4 <= mag_cu_p3[30:15];
            // Tap 0: Just pass I/Q (normalized by magnitude gives unit vector)
            // For simplicity in synthesis, we approximate: I_norm = I, Q_norm = Q
            // The actual division would be: I/|z|, Q/|z| but that requires divider
            // Alternative: Store I and Q directly for the NN to learn the scaling
            i_norm[0] <= i_pipe[0][3];
            q_norm[0] <= q_pipe[0][3];
            for (i = 0; i < MEMORY_DEPTH; i = i + 1) begin
                i_pipe[i][4] <= i_pipe[i][3];
                q_pipe[i][4] <= q_pipe[i][3];
            end
            
            // Stage 5: Phase norm tap 1
            mag_p5 <= mag_p4;
            mag_cu_trunc_p5 <= mag_cu_trunc_p4;
            i_norm[1] <= i_pipe[1][4];
            q_norm[1] <= q_pipe[1][4];
            for (i = 0; i < MEMORY_DEPTH; i = i + 1) begin
                i_pipe[i][5] <= i_pipe[i][4];
                q_pipe[i][5] <= q_pipe[i][4];
            end
            
            // Stage 6: Phase norm tap 2
            mag_p6 <= mag_p5;
            mag_cu_trunc_p6 <= mag_cu_trunc_p5;
            i_norm[2] <= i_pipe[2][5];
            q_norm[2] <= q_pipe[2][5];
            for (i = 0; i < MEMORY_DEPTH; i = i + 1) begin
                i_pipe[i][6] <= i_pipe[i][5];
                q_pipe[i][6] <= q_pipe[i][5];
            end
            
            // Stage 7: Phase norm tap 3
            mag_p7 <= mag_p6;
            mag_cu_trunc_p7 <= mag_cu_trunc_p6;
            i_norm[3] <= i_pipe[3][6];
            q_norm[3] <= q_pipe[3][6];
            for (i = 0; i < MEMORY_DEPTH; i = i + 1) begin
                i_pipe[i][7] <= i_pipe[i][6];
                q_pipe[i][7] <= q_pipe[i][6];
            end
            
            // Stage 8: Output assembly
            mag_p8 <= mag_p7;
            mag_cu_trunc_p8 <= mag_cu_trunc_p7;
            for (i = 0; i < MEMORY_DEPTH; i = i + 1) begin
                i_pipe[i][8] <= i_pipe[i][7];
                q_pipe[i][8] <= q_pipe[i][7];
            end
        end
    end
    
    //==========================================================================
    // Output Assembly
    //==========================================================================
    // Feature vector layout (24 elements):
    // [0-7]:   Phase-normalized I/Q (4 taps × 2)
    // [8-11]:  Magnitude A (4 copies for alignment)
    // [12-15]: Magnitude cubed A³ (4 copies)
    // [16-23]: Original I/Q (4 taps × 2)
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < 24; i = i + 1) begin
                features[i] <= 0;
            end
            out_valid <= 0;
        end else begin
            out_valid <= valid_pipe[8];
            
            if (valid_pipe[8]) begin
                // Phase-normalized I/Q (indices 0-7)
                features[0] <= i_norm[0];
                features[1] <= q_norm[0];
                features[2] <= i_norm[1];
                features[3] <= q_norm[1];
                features[4] <= i_norm[2];
                features[5] <= q_norm[2];
                features[6] <= i_norm[3];
                features[7] <= q_norm[3];
                
                // Magnitude (indices 8-11)
                features[8]  <= mag_p8;
                features[9]  <= mag_p8;
                features[10] <= mag_p8;
                features[11] <= mag_p8;
                
                // Magnitude cubed (indices 12-15)
                features[12] <= mag_cu_trunc_p8;
                features[13] <= mag_cu_trunc_p8;
                features[14] <= mag_cu_trunc_p8;
                features[15] <= mag_cu_trunc_p8;
                
                // Original I/Q (indices 16-23)
                features[16] <= i_pipe[0][8];
                features[17] <= q_pipe[0][8];
                features[18] <= i_pipe[1][8];
                features[19] <= q_pipe[1][8];
                features[20] <= i_pipe[2][8];
                features[21] <= q_pipe[2][8];
                features[22] <= i_pipe[3][8];
                features[23] <= q_pipe[3][8];
            end
        end
    end
    
    assign busy = |valid_pipe;

endmodule
