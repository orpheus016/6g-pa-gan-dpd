//==============================================================================
// 5x Polyphase Interpolator for 6G DPD - SSR (Super Sample Rate) Version
// Single 200MHz clock domain with 5 parallel outputs
//==============================================================================
// Features:
// - 5x interpolation with SSR=5 parallel outputs
// - 12 taps per phase (60 taps total)
// - Fixed-point Q1.15 I/Q processing
// - Single clock domain (200 MHz) - NO CDC
// - Outputs 5 I/Q pairs per clock = 1 GSps aggregate throughput
//==============================================================================

`timescale 1ns / 1ps

module interpolator1_5_ssr #(
    parameter DATA_WIDTH = 16,
    parameter COEF_WIDTH = 16,
    parameter TAPS_PER_PHASE = 12,        
    parameter ACC_WIDTH = 40
)(
    input  wire                         clk,            // Single 200MHz clock
    input  wire                         rst_n,
    
    // Input sample @ 200MHz (Q1.15 complex)
    input  wire signed [DATA_WIDTH-1:0] in_i,
    input  wire signed [DATA_WIDTH-1:0] in_q,
    input  wire                         in_valid,
    
    // Output: 5 parallel samples @ 200MHz (Q1.15 complex)
    // Effective rate: 5 × 200MHz = 1 GSps
    output reg  signed [DATA_WIDTH-1:0] out_i_0,
    output reg  signed [DATA_WIDTH-1:0] out_q_0,
    output reg  signed [DATA_WIDTH-1:0] out_i_1,
    output reg  signed [DATA_WIDTH-1:0] out_q_1,
    output reg  signed [DATA_WIDTH-1:0] out_i_2,
    output reg  signed [DATA_WIDTH-1:0] out_q_2,
    output reg  signed [DATA_WIDTH-1:0] out_i_3,
    output reg  signed [DATA_WIDTH-1:0] out_q_3,
    output reg  signed [DATA_WIDTH-1:0] out_i_4,
    output reg  signed [DATA_WIDTH-1:0] out_q_4,
    output reg                          out_valid
);

    //==========================================================================
    // 1:5 Polyphase Coefficients (Q1.15)
    //==========================================================================
    
    // Arrays to hold coefficients for each phase
    wire signed [COEF_WIDTH-1:0] coef_p0 [0:TAPS_PER_PHASE-1];
    wire signed [COEF_WIDTH-1:0] coef_p1 [0:TAPS_PER_PHASE-1];
    wire signed [COEF_WIDTH-1:0] coef_p2 [0:TAPS_PER_PHASE-1];
    wire signed [COEF_WIDTH-1:0] coef_p3 [0:TAPS_PER_PHASE-1];
    wire signed [COEF_WIDTH-1:0] coef_p4 [0:TAPS_PER_PHASE-1];

    // Phase 0
    assign coef_p0[0]  = 16'hFFF8; // -0.00025
    assign coef_p0[1]  = 16'h003F; // 0.00193
    assign coef_p0[2]  = 16'hFF29; // -0.00657
    assign coef_p0[3]  = 16'h0227; // 0.01682
    assign coef_p0[4]  = 16'hFB0B; // -0.03871
    assign coef_p0[5]  = 16'h0D20; // 0.10253
    assign coef_p0[6]  = 16'h7DD3; // 0.98299
    assign coef_p0[7]  = 16'hF599; // -0.08126
    assign coef_p0[8]  = 16'h0432; // 0.03277
    assign coef_p0[9]  = 16'hFE31; // -0.01412
    assign coef_p0[10] = 16'h00AE; // 0.00530
    assign coef_p0[11] = 16'hFFD1; // -0.00142

    // Phase 1
    assign coef_p1[0]  = 16'hFFDB; // -0.00114
    assign coef_p1[1]  = 16'h00DB; // 0.00669
    assign coef_p1[2]  = 16'hFD4D; // -0.02109
    assign coef_p1[3]  = 16'h06AE; // 0.05219
    assign coef_p1[4]  = 16'hF09E; // -0.12018
    assign coef_p1[5]  = 16'h2D4F; // 0.35398
    assign coef_p1[6]  = 16'h6D1D; // 0.85245
    assign coef_p1[7]  = 16'hE9D4; // -0.17321
    assign coef_p1[8]  = 16'h0950; // 0.07275
    assign coef_p1[9]  = 16'hFC0C; // -0.03089
    assign coef_p1[10] = 16'h016A; // 0.01106
    assign coef_p1[11] = 16'hFFA9; // -0.00264

    // Phase 2
    assign coef_p2[0]  = 16'hFFB7; // -0.00222
    assign coef_p2[1]  = 16'h0160; // 0.01073
    assign coef_p2[2]  = 16'hFBF2; // -0.03168
    assign coef_p2[3]  = 16'h09C2; // 0.07623
    assign coef_p2[4]  = 16'hE94D; // -0.17735
    assign coef_p2[5]  = 16'h4FE9; // 0.62429
    assign coef_p2[6]  = 16'h4FE9; // 0.62429
    assign coef_p2[7]  = 16'hE94D; // -0.17735
    assign coef_p2[8]  = 16'h09C2; // 0.07623
    assign coef_p2[9]  = 16'hFBF2; // -0.03168
    assign coef_p2[10] = 16'h0160; // 0.01073
    assign coef_p2[11] = 16'hFFB7; // -0.00222

    // Phase 3
    assign coef_p3[0]  = 16'hFFA9; // -0.00264
    assign coef_p3[1]  = 16'h016A; // 0.01106
    assign coef_p3[2]  = 16'hFC0C; // -0.03089
    assign coef_p3[3]  = 16'h0950; // 0.07275
    assign coef_p3[4]  = 16'hE9D4; // -0.17321
    assign coef_p3[5]  = 16'h6D1D; // 0.85245
    assign coef_p3[6]  = 16'h2D4F; // 0.35398
    assign coef_p3[7]  = 16'hF09E; // -0.12018
    assign coef_p3[8]  = 16'h06AE; // 0.05219
    assign coef_p3[9]  = 16'hFD4D; // -0.02109
    assign coef_p3[10] = 16'h00DB; // 0.00669
    assign coef_p3[11] = 16'hFFDB; // -0.00114

    // Phase 4
    assign coef_p4[0]  = 16'hFFD1; // -0.00142
    assign coef_p4[1]  = 16'h00AE; // 0.00530
    assign coef_p4[2]  = 16'hFE31; // -0.01412
    assign coef_p4[3]  = 16'h0432; // 0.03277
    assign coef_p4[4]  = 16'hF599; // -0.08126
    assign coef_p4[5]  = 16'h7DD3; // 0.98299
    assign coef_p4[6]  = 16'h0D20; // 0.10253
    assign coef_p4[7]  = 16'hFB0B; // -0.03871
    assign coef_p4[8]  = 16'h0227; // 0.01682
    assign coef_p4[9]  = 16'hFF29; // -0.00657
    assign coef_p4[10] = 16'h003F; // 0.00193
    assign coef_p4[11] = 16'hFFF8; // -0.00025
    
    //==========================================================================
    // Delay Line (Input Samples)
    // Feeds all 5 phases simultaneously
    //==========================================================================
    
    reg signed [DATA_WIDTH-1:0] delay_i [0:TAPS_PER_PHASE-1];
    reg signed [DATA_WIDTH-1:0] delay_q [0:TAPS_PER_PHASE-1];
    integer i;
    
    // Shift register at 200MHz rate
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < TAPS_PER_PHASE; i = i + 1) begin
                delay_i[i] <= 0;
                delay_q[i] <= 0;
            end
        end else if (in_valid) begin
            delay_i[0] <= in_i;
            delay_q[0] <= in_q;
            for (i = 1; i < TAPS_PER_PHASE; i = i + 1) begin
                delay_i[i] <= delay_i[i-1];
                delay_q[i] <= delay_q[i-1];
            end
        end
    end

    //==========================================================================
    // Polyphase Filter Computation - 5 Parallel MAC Engines
    // All computation in single 200MHz domain
    //==========================================================================
    
    // Accumulator registers for each phase
    reg signed [ACC_WIDTH-1:0] acc_i [0:4];
    reg signed [ACC_WIDTH-1:0] acc_q [0:4];
    reg calc_valid;
    reg calc_valid_d1;  // Pipeline delay for output alignment
    
    // Compute all 5 phases in parallel at 200MHz
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < 5; i = i + 1) begin
                acc_i[i] <= 0;
                acc_q[i] <= 0;
            end
            calc_valid <= 0;
        end else if (in_valid) begin
            // Phase 0 Calculation
            acc_i[0] <= coef_p0[0]*delay_i[0] + coef_p0[1]*delay_i[1] + coef_p0[2]*delay_i[2] + 
                        coef_p0[3]*delay_i[3] + coef_p0[4]*delay_i[4] + coef_p0[5]*delay_i[5] +
                        coef_p0[6]*delay_i[6] + coef_p0[7]*delay_i[7] + coef_p0[8]*delay_i[8] +
                        coef_p0[9]*delay_i[9] + coef_p0[10]*delay_i[10] + coef_p0[11]*delay_i[11];
             
            acc_q[0] <= coef_p0[0]*delay_q[0] + coef_p0[1]*delay_q[1] + coef_p0[2]*delay_q[2] + 
                        coef_p0[3]*delay_q[3] + coef_p0[4]*delay_q[4] + coef_p0[5]*delay_q[5] +
                        coef_p0[6]*delay_q[6] + coef_p0[7]*delay_q[7] + coef_p0[8]*delay_q[8] +
                        coef_p0[9]*delay_q[9] + coef_p0[10]*delay_q[10] + coef_p0[11]*delay_q[11];

            // Phase 1 Calculation
            acc_i[1] <= coef_p1[0]*delay_i[0] + coef_p1[1]*delay_i[1] + coef_p1[2]*delay_i[2] + 
                        coef_p1[3]*delay_i[3] + coef_p1[4]*delay_i[4] + coef_p1[5]*delay_i[5] +
                        coef_p1[6]*delay_i[6] + coef_p1[7]*delay_i[7] + coef_p1[8]*delay_i[8] +
                        coef_p1[9]*delay_i[9] + coef_p1[10]*delay_i[10] + coef_p1[11]*delay_i[11];
            acc_q[1] <= coef_p1[0]*delay_q[0] + coef_p1[1]*delay_q[1] + coef_p1[2]*delay_q[2] + 
                        coef_p1[3]*delay_q[3] + coef_p1[4]*delay_q[4] + coef_p1[5]*delay_q[5] +
                        coef_p1[6]*delay_q[6] + coef_p1[7]*delay_q[7] + coef_p1[8]*delay_q[8] +
                        coef_p1[9]*delay_q[9] + coef_p1[10]*delay_q[10] + coef_p1[11]*delay_q[11];

            // Phase 2 Calculation
            acc_i[2] <= coef_p2[0]*delay_i[0] + coef_p2[1]*delay_i[1] + coef_p2[2]*delay_i[2] + 
                        coef_p2[3]*delay_i[3] + coef_p2[4]*delay_i[4] + coef_p2[5]*delay_i[5] +
                        coef_p2[6]*delay_i[6] + coef_p2[7]*delay_i[7] + coef_p2[8]*delay_i[8] +
                        coef_p2[9]*delay_i[9] + coef_p2[10]*delay_i[10] + coef_p2[11]*delay_i[11];
            acc_q[2] <= coef_p2[0]*delay_q[0] + coef_p2[1]*delay_q[1] + coef_p2[2]*delay_q[2] + 
                        coef_p2[3]*delay_q[3] + coef_p2[4]*delay_q[4] + coef_p2[5]*delay_q[5] +
                        coef_p2[6]*delay_q[6] + coef_p2[7]*delay_q[7] + coef_p2[8]*delay_q[8] +
                        coef_p2[9]*delay_q[9] + coef_p2[10]*delay_q[10] + coef_p2[11]*delay_q[11];

            // Phase 3 Calculation
            acc_i[3] <= coef_p3[0]*delay_i[0] + coef_p3[1]*delay_i[1] + coef_p3[2]*delay_i[2] + 
                        coef_p3[3]*delay_i[3] + coef_p3[4]*delay_i[4] + coef_p3[5]*delay_i[5] +
                        coef_p3[6]*delay_i[6] + coef_p3[7]*delay_i[7] + coef_p3[8]*delay_i[8] +
                        coef_p3[9]*delay_i[9] + coef_p3[10]*delay_i[10] + coef_p3[11]*delay_i[11];
            acc_q[3] <= coef_p3[0]*delay_q[0] + coef_p3[1]*delay_q[1] + coef_p3[2]*delay_q[2] + 
                        coef_p3[3]*delay_q[3] + coef_p3[4]*delay_q[4] + coef_p3[5]*delay_q[5] +
                        coef_p3[6]*delay_q[6] + coef_p3[7]*delay_q[7] + coef_p3[8]*delay_q[8] +
                        coef_p3[9]*delay_q[9] + coef_p3[10]*delay_q[10] + coef_p3[11]*delay_q[11];

            // Phase 4 Calculation
            acc_i[4] <= coef_p4[0]*delay_i[0] + coef_p4[1]*delay_i[1] + coef_p4[2]*delay_i[2] + 
                        coef_p4[3]*delay_i[3] + coef_p4[4]*delay_i[4] + coef_p4[5]*delay_i[5] +
                        coef_p4[6]*delay_i[6] + coef_p4[7]*delay_i[7] + coef_p4[8]*delay_i[8] +
                        coef_p4[9]*delay_i[9] + coef_p4[10]*delay_i[10] + coef_p4[11]*delay_i[11];
            acc_q[4] <= coef_p4[0]*delay_q[0] + coef_p4[1]*delay_q[1] + coef_p4[2]*delay_q[2] + 
                        coef_p4[3]*delay_q[3] + coef_p4[4]*delay_q[4] + coef_p4[5]*delay_q[5] +
                        coef_p4[6]*delay_q[6] + coef_p4[7]*delay_q[7] + coef_p4[8]*delay_q[8] +
                        coef_p4[9]*delay_q[9] + coef_p4[10]*delay_q[10] + coef_p4[11]*delay_q[11];

            calc_valid <= 1'b1;
        end else begin
            calc_valid <= 1'b0;
        end
    end

    //==========================================================================
    // Output Stage - 5 Parallel Outputs (SSR=5)
    // Extract Q1.15 from Q8.32 accumulator (bits [30:15])
    //==========================================================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_i_0 <= 0; out_q_0 <= 0;
            out_i_1 <= 0; out_q_1 <= 0;
            out_i_2 <= 0; out_q_2 <= 0;
            out_i_3 <= 0; out_q_3 <= 0;
            out_i_4 <= 0; out_q_4 <= 0;
            out_valid <= 0;
            calc_valid_d1 <= 0;
        end else begin
            calc_valid_d1 <= calc_valid;
            
            if (calc_valid) begin
                // Output all 5 phases in parallel
                // Truncate accumulator from Q8.32 to Q1.15 (take bits [30:15])
                out_i_0 <= acc_i[0][30:15];
                out_q_0 <= acc_q[0][30:15];
                out_i_1 <= acc_i[1][30:15];
                out_q_1 <= acc_q[1][30:15];
                out_i_2 <= acc_i[2][30:15];
                out_q_2 <= acc_q[2][30:15];
                out_i_3 <= acc_i[3][30:15];
                out_q_3 <= acc_q[3][30:15];
                out_i_4 <= acc_i[4][30:15];
                out_q_4 <= acc_q[4][30:15];
                out_valid <= 1'b1;
            end else begin
                out_valid <= 1'b0;
            end
        end
    end

endmodule
