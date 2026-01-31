//==============================================================================
// 5x Polyphase Interpolator for 6G DPD
// Upsamples from 200MHz to 1GHz for PA output (use High-Speed I/O in FPGA)
//==============================================================================
// Features:
// - 5x interpolation
// - 12 taps per phase (60 taps total)
// - Fixed-point Q1.15 I/Q processing
//==============================================================================

module interpolator1_5 #(
    parameter DATA_WIDTH = 16,
    parameter COEF_WIDTH = 16,
    parameter TAPS_PER_PHASE = 12,        
    parameter ACC_WIDTH = 40
)(
    input  wire                         clk_200,        // Input clock (200MHz)
    input  wire                         clk_1k,        // Output clock (1GHz)
    input  wire                         rst_n,
    
    // Input samples @ 200MHz (Q1.15 complex)
    input  wire signed [DATA_WIDTH-1:0] in_i,
    input  wire signed [DATA_WIDTH-1:0] in_q,
    input  wire                         in_valid,
    
    // Output samples @ 1GHz (Q1.15 complex)
    output reg  signed [DATA_WIDTH-1:0] out_i,
    output reg  signed [DATA_WIDTH-1:0] out_q,
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
    assign coef_p0[0] = 16'hFFF8; // -0.00025
    assign coef_p0[1] = 16'h003F; // 0.00193
    assign coef_p0[2] = 16'hFF29; // -0.00657
    assign coef_p0[3] = 16'h0227; // 0.01682
    assign coef_p0[4] = 16'hFB0B; // -0.03871
    assign coef_p0[5] = 16'h0D20; // 0.10253
    assign coef_p0[6] = 16'h7DD3; // 0.98299
    assign coef_p0[7] = 16'hF599; // -0.08126
    assign coef_p0[8] = 16'h0432; // 0.03277
    assign coef_p0[9] = 16'hFE31; // -0.01412
    assign coef_p0[10] = 16'h00AE; // 0.00530
    assign coef_p0[11] = 16'hFFD1; // -0.00142

    // Phase 1
    assign coef_p1[0] = 16'hFFDB; // -0.00114
    assign coef_p1[1] = 16'h00DB; // 0.00669
    assign coef_p1[2] = 16'hFD4D; // -0.02109
    assign coef_p1[3] = 16'h06AE; // 0.05219
    assign coef_p1[4] = 16'hF09E; // -0.12018
    assign coef_p1[5] = 16'h2D4F; // 0.35398
    assign coef_p1[6] = 16'h6D1D; // 0.85245
    assign coef_p1[7] = 16'hE9D4; // -0.17321
    assign coef_p1[8] = 16'h0950; // 0.07275
    assign coef_p1[9] = 16'hFC0C; // -0.03089
    assign coef_p1[10] = 16'h016A; // 0.01106
    assign coef_p1[11] = 16'hFFA9; // -0.00264

    // Phase 2
    assign coef_p2[0] = 16'hFFB7; // -0.00222
    assign coef_p2[1] = 16'h0160; // 0.01073
    assign coef_p2[2] = 16'hFBF2; // -0.03168
    assign coef_p2[3] = 16'h09C2; // 0.07623
    assign coef_p2[4] = 16'hE94D; // -0.17735
    assign coef_p2[5] = 16'h4FE9; // 0.62429
    assign coef_p2[6] = 16'h4FE9; // 0.62429
    assign coef_p2[7] = 16'hE94D; // -0.17735
    assign coef_p2[8] = 16'h09C2; // 0.07623
    assign coef_p2[9] = 16'hFBF2; // -0.03168
    assign coef_p2[10] = 16'h0160; // 0.01073
    assign coef_p2[11] = 16'hFFB7; // -0.00222

    // Phase 3
    assign coef_p3[0] = 16'hFFA9; // -0.00264
    assign coef_p3[1] = 16'h016A; // 0.01106
    assign coef_p3[2] = 16'hFC0C; // -0.03089
    assign coef_p3[3] = 16'h0950; // 0.07275
    assign coef_p3[4] = 16'hE9D4; // -0.17321
    assign coef_p3[5] = 16'h6D1D; // 0.85245
    assign coef_p3[6] = 16'h2D4F; // 0.35398
    assign coef_p3[7] = 16'hF09E; // -0.12018
    assign coef_p3[8] = 16'h06AE; // 0.05219
    assign coef_p3[9] = 16'hFD4D; // -0.02109
    assign coef_p3[10] = 16'h00DB; // 0.00669
    assign coef_p3[11] = 16'hFFDB; // -0.00114

    // Phase 4
    assign coef_p4[0] = 16'hFFD1; // -0.00142
    assign coef_p4[1] = 16'h00AE; // 0.00530
    assign coef_p4[2] = 16'hFE31; // -0.01412
    assign coef_p4[3] = 16'h0432; // 0.03277
    assign coef_p4[4] = 16'hF599; // -0.08126
    assign coef_p4[5] = 16'h7DD3; // 0.98299
    assign coef_p4[6] = 16'h0D20; // 0.10253
    assign coef_p4[7] = 16'hFB0B; // -0.03871
    assign coef_p4[8] = 16'h0227; // 0.01682
    assign coef_p4[9] = 16'hFF29; // -0.00657
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
    always @(posedge clk_200 or negedge rst_n) begin
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
    // Polyphase Filter Computation
    // 5 Parallel MAC Engines
    //==========================================================================
    
    // Accumulator wires
    reg signed [ACC_WIDTH-1:0] acc_i [0:4];
    reg signed [ACC_WIDTH-1:0] acc_q [0:4];
    reg calc_valid;

    integer t; // tap iterator
    
    // Compute all 5 phases at 200MHz
    always @(posedge clk_200 or negedge rst_n) begin
        if (!rst_n) begin
            for (i=0; i<5; i=i+1) begin
                acc_i[i] <= 0;
                acc_q[i] <= 0;
            end
            calc_valid <= 0;
        end else if (in_valid) begin
            // Initialization
            acc_i[0] <= 0; acc_q[0] <= 0;
            acc_i[1] <= 0; acc_q[1] <= 0;
            acc_i[2] <= 0; acc_q[2] <= 0;
            acc_i[3] <= 0; acc_q[3] <= 0;
            acc_i[4] <= 0; acc_q[4] <= 0;

            // Phase 0 Calculation
            acc_i[0] <= coef_p0[0]*delay_i[0] + coef_p0[1]*delay_i[1] + coef_p0[2]*delay_i[2] + 
                         coef_p0[3]*delay_i[3] + coef_p0[4]*delay_i[4] + coef_p0[5]*delay_i[5] +
                         coef_p0[6]*delay_i[6] + coef_p0[7]*delay_i[7] + coef_p0[8]*delay_i[8] +
                         coef_p0[9]*delay_i[9] + coef_p0[10]*delay_i[10] + coef_p0[11]*delay_i[11];
             
            acc_q[0] <= coef_p0[0]*delay_q[0] + coef_p0[1]*delay_q[1] + coef_p0[2]*delay_q[2] + 
                         coef_p0[3]*delay_q[3] + coef_p0[4]*delay_q[4] + coef_p0[5]*delay_q[5] +
                         coef_p0[6]*delay_q[6] + coef_p0[7]*delay_q[7] + coef_p0[8]*delay_q[8] +
                         coef_p0[9]*delay_q[9] + coef_p0[10]*delay_q[10] + coef_p0[11]*delay_q[11];

            // --- Phase 1 Calculation ---
            acc_i[1] <= coef_p1[0]*delay_i[0] + coef_p1[1]*delay_i[1] + coef_p1[2]*delay_i[2] + 
                         coef_p1[3]*delay_i[3] + coef_p1[4]*delay_i[4] + coef_p1[5]*delay_i[5] +
                         coef_p1[6]*delay_i[6] + coef_p1[7]*delay_i[7] + coef_p1[8]*delay_i[8] +
                         coef_p1[9]*delay_i[9] + coef_p1[10]*delay_i[10] + coef_p1[11]*delay_i[11];
            acc_q[1] <= coef_p1[0]*delay_q[0] + coef_p1[1]*delay_q[1] + coef_p1[2]*delay_q[2] + 
                         coef_p1[3]*delay_q[3] + coef_p1[4]*delay_q[4] + coef_p1[5]*delay_q[5] +
                         coef_p1[6]*delay_q[6] + coef_p1[7]*delay_q[7] + coef_p1[8]*delay_q[8] +
                         coef_p1[9]*delay_q[9] + coef_p1[10]*delay_q[10] + coef_p1[11]*delay_q[11];

            // --- Phase 2 Calculation ---
            acc_i[2] <= coef_p2[0]*delay_i[0] + coef_p2[1]*delay_i[1] + coef_p2[2]*delay_i[2] + 
                         coef_p2[3]*delay_i[3] + coef_p2[4]*delay_i[4] + coef_p2[5]*delay_i[5] +
                         coef_p2[6]*delay_i[6] + coef_p2[7]*delay_i[7] + coef_p2[8]*delay_i[8] +
                         coef_p2[9]*delay_i[9] + coef_p2[10]*delay_i[10] + coef_p2[11]*delay_i[11];
            acc_q[2] <= coef_p2[0]*delay_q[0] + coef_p2[1]*delay_q[1] + coef_p2[2]*delay_q[2] + 
                         coef_p2[3]*delay_q[3] + coef_p2[4]*delay_q[4] + coef_p2[5]*delay_q[5] +
                         coef_p2[6]*delay_q[6] + coef_p2[7]*delay_q[7] + coef_p2[8]*delay_q[8] +
                         coef_p2[9]*delay_q[9] + coef_p2[10]*delay_q[10] + coef_p2[11]*delay_q[11];

            // --- Phase 3 Calculation ---
            acc_i[3] <= coef_p3[0]*delay_i[0] + coef_p3[1]*delay_i[1] + coef_p3[2]*delay_i[2] + 
                         coef_p3[3]*delay_i[3] + coef_p3[4]*delay_i[4] + coef_p3[5]*delay_i[5] +
                         coef_p3[6]*delay_i[6] + coef_p3[7]*delay_i[7] + coef_p3[8]*delay_i[8] +
                         coef_p3[9]*delay_i[9] + coef_p3[10]*delay_i[10] + coef_p3[11]*delay_i[11];
            acc_q[3] <= coef_p3[0]*delay_q[0] + coef_p3[1]*delay_q[1] + coef_p3[2]*delay_q[2] + 
                         coef_p3[3]*delay_q[3] + coef_p3[4]*delay_q[4] + coef_p3[5]*delay_q[5] +
                         coef_p3[6]*delay_q[6] + coef_p3[7]*delay_q[7] + coef_p3[8]*delay_q[8] +
                         coef_p3[9]*delay_q[9] + coef_p3[10]*delay_q[10] + coef_p3[11]*delay_q[11];

            // --- Phase 4 Calculation ---
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
    // Clock Domain Crossing: 200MHz -> 1GHz
    //==========================================================================
    
    // Output Buffer (holds 5 samples)
    // We take the top 16 bits of accumulator
    reg signed [DATA_WIDTH-1:0] buf_i [0:4];
    reg signed [DATA_WIDTH-1:0] buf_q [0:4];
    reg buf_loaded;
    
    // Write at 200MHz
    always @(posedge clk_200 or negedge rst_n) begin
        if (!rst_n) begin
            for (i=0; i<5; i=i+1) begin
                buf_i[i] <= 0;
                buf_q[i] <= 0;
            end
            buf_loaded <= 0;
        end else if (calc_valid) begin
            // Capture all 5 phases
            buf_i[0] <= acc_i[0][30:15]; buf_q[0] <= acc_q[0][30:15];
            buf_i[1] <= acc_i[1][30:15]; buf_q[1] <= acc_q[1][30:15];
            buf_i[2] <= acc_i[2][30:15]; buf_q[2] <= acc_q[2][30:15];
            buf_i[3] <= acc_i[3][30:15]; buf_q[3] <= acc_q[3][30:15];
            buf_i[4] <= acc_i[4][30:15]; buf_q[4] <= acc_q[4][30:15];
            buf_loaded <= 1'b1;
        end else begin
            buf_loaded <= 1'b0;
        end
    end

    //==========================================================================
    // Output Serializer @ 1GHz
    //==========================================================================
    
    reg [2:0] phase_sel; // 0 to 4
    reg buf_loaded_sync1, buf_loaded_sync2, buf_loaded_sync3;
    reg output_active;
    
    // Synchronize buf_loaded to 1GHz domain
    always @(posedge clk_1k or negedge rst_n) begin
        if (!rst_n) begin
            buf_loaded_sync1 <= 1'b0;
            buf_loaded_sync2 <= 1'b0;
            buf_loaded_sync3 <= 1'b0;
        end else begin
            buf_loaded_sync1 <= buf_loaded;
            buf_loaded_sync2 <= buf_loaded_sync1;
            buf_loaded_sync3 <= buf_loaded_sync2;
        end
    end

    // Detect rising edge
    wire start_pulse = buf_loaded_sync2 && !buf_loaded_sync3;
    
    // Phase selection and output state machine
    always @(posedge clk_1k or negedge rst_n) begin
        if (!rst_n) begin
            phase_sel <= 1'b0;
            out_i <= 0;
            out_q <= 0;
            out_valid <= 1'b0;
            output_active <= 1'b0;
        end else begin
            if (start_pulse && !output_active) begin
                output_active <= 1'b1;
                phase_sel <= 0; // Reset to phase 0 on a new data block
            end
            
            if (output_active) begin
                out_valid <= 1'b1;
                out_i <= buf_i[phase_sel];
                out_q <= buf_q[phase_sel];
                
                if (phase_sel == 4) begin
                    phase_sel <= 0;
                    if (!buf_loaded_sync2) output_active <= 0;
                end else begin
                    phase_sel <= phase_sel + 1;
                end
            end else begin
                out_valid <= 1'b0;
                out_i <= 1'b0;
                out_q <= 1'b0;
            end
        end
    end

endmodule