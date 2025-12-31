//==============================================================================
// 2x Polyphase Interpolator for 6G DPD
// Upsamples from 200MHz to 400MHz for PA output
//==============================================================================
// Features:
// - 2x interpolation using polyphase FIR structure
// - 23-tap half-band filter for efficient implementation
// - Symmetric coefficients for reduced multipliers
// - Fixed-point Q1.15 I/Q processing
//==============================================================================

module interpolator #(
    parameter DATA_WIDTH = 16,
    parameter COEF_WIDTH = 16,
    parameter NUM_TAPS = 23,         // Half-band filter taps (odd)
    parameter ACC_WIDTH = 40
)(
    input  wire                         clk_200,        // Input clock (200MHz)
    input  wire                         clk_400,        // Output clock (400MHz)
    input  wire                         rst_n,
    
    // Input samples @ 200MHz (Q1.15 complex)
    input  wire signed [DATA_WIDTH-1:0] in_i,
    input  wire signed [DATA_WIDTH-1:0] in_q,
    input  wire                         in_valid,
    
    // Output samples @ 400MHz (Q1.15 complex)
    output reg  signed [DATA_WIDTH-1:0] out_i,
    output reg  signed [DATA_WIDTH-1:0] out_q,
    output reg                          out_valid
);

    //==========================================================================
    // Half-band Filter Coefficients (Q1.15)
    // Designed for 0.4 transition band, >60dB stopband
    // Only non-zero coefficients stored (every other tap is zero)
    //==========================================================================
    
    localparam NUM_COEF = (NUM_TAPS + 1) / 2;  // 12 coefficients
    
    // Symmetric half-band coefficients (Q1.15)
    // h[n] = h[N-1-n], center tap = 0.5
    wire signed [COEF_WIDTH-1:0] coef [0:NUM_COEF-1];
    
    assign coef[0]  = 16'h0052;    // 0.00249
    assign coef[1]  = 16'hFF6C;    // -0.00452
    assign coef[2]  = 16'h00D4;    // 0.00647
    assign coef[3]  = 16'hFE9E;    // -0.01013
    assign coef[4]  = 16'h01B8;    // 0.01331
    assign coef[5]  = 16'hFD4A;    // -0.02094
    assign coef[6]  = 16'h0378;    // 0.02686
    assign coef[7]  = 16'hFA9E;    // -0.04188
    assign coef[8]  = 16'h0694;    // 0.05127
    assign coef[9]  = 16'hF424;    // -0.09192
    assign coef[10] = 16'h1594;    // 0.16846
    assign coef[11] = 16'h4000;    // 0.5 (center tap)

    //==========================================================================
    // Delay Line (Input Samples)
    //==========================================================================
    
    reg signed [DATA_WIDTH-1:0] delay_i [0:NUM_TAPS-1];
    reg signed [DATA_WIDTH-1:0] delay_q [0:NUM_TAPS-1];
    
    integer i;
    
    // Shift register at 200MHz rate
    always @(posedge clk_200 or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < NUM_TAPS; i = i + 1) begin
                delay_i[i] <= 0;
                delay_q[i] <= 0;
            end
        end else if (in_valid) begin
            delay_i[0] <= in_i;
            delay_q[0] <= in_q;
            for (i = 1; i < NUM_TAPS; i = i + 1) begin
                delay_i[i] <= delay_i[i-1];
                delay_q[i] <= delay_q[i-1];
            end
        end
    end

    //==========================================================================
    // Polyphase Filter Computation
    // Phase 0: Even taps (original samples)
    // Phase 1: Odd taps (interpolated samples)
    //==========================================================================
    
    // Accumulator wires
    reg signed [ACC_WIDTH-1:0] acc_i_phase0, acc_q_phase0;
    reg signed [ACC_WIDTH-1:0] acc_i_phase1, acc_q_phase1;
    
    // Phase outputs (registered)
    reg signed [DATA_WIDTH-1:0] phase0_i, phase0_q;
    reg signed [DATA_WIDTH-1:0] phase1_i, phase1_q;
    reg phase_valid;
    
    // Compute both phases at 200MHz
    always @(posedge clk_200 or negedge rst_n) begin
        if (!rst_n) begin
            acc_i_phase0 <= 0;
            acc_q_phase0 <= 0;
            acc_i_phase1 <= 0;
            acc_q_phase1 <= 0;
            phase0_i <= 0;
            phase0_q <= 0;
            phase1_i <= 0;
            phase1_q <= 0;
            phase_valid <= 1'b0;
        end else if (in_valid) begin
            // Phase 0: Pass through with unit gain at center
            // For half-band, phase 0 just outputs delayed center sample
            phase0_i <= delay_i[(NUM_TAPS-1)/2];
            phase0_q <= delay_q[(NUM_TAPS-1)/2];
            
            // Phase 1: Interpolated sample using non-zero coefficients
            // Sum symmetric pairs
            acc_i_phase1 <= 
                coef[0]  * (delay_i[0]  + delay_i[22]) +
                coef[1]  * (delay_i[2]  + delay_i[20]) +
                coef[2]  * (delay_i[4]  + delay_i[18]) +
                coef[3]  * (delay_i[6]  + delay_i[16]) +
                coef[4]  * (delay_i[8]  + delay_i[14]) +
                coef[5]  * (delay_i[10] + delay_i[12]);
            
            acc_q_phase1 <=
                coef[0]  * (delay_q[0]  + delay_q[22]) +
                coef[1]  * (delay_q[2]  + delay_q[20]) +
                coef[2]  * (delay_q[4]  + delay_q[18]) +
                coef[3]  * (delay_q[6]  + delay_q[16]) +
                coef[4]  * (delay_q[8]  + delay_q[14]) +
                coef[5]  * (delay_q[10] + delay_q[12]);
            
            phase_valid <= 1'b1;
        end else begin
            phase_valid <= 1'b0;
        end
    end
    
    // Normalize phase 1 output (Q1.15 * Q1.15 = Q2.30, need Q1.15)
    wire signed [DATA_WIDTH-1:0] phase1_i_norm = acc_i_phase1[30:15];
    wire signed [DATA_WIDTH-1:0] phase1_q_norm = acc_q_phase1[30:15];

    //==========================================================================
    // Clock Domain Crossing: 200MHz -> 400MHz
    //==========================================================================
    
    // FIFO-style double buffer
    reg signed [DATA_WIDTH-1:0] buf_phase0_i, buf_phase0_q;
    reg signed [DATA_WIDTH-1:0] buf_phase1_i, buf_phase1_q;
    reg buf_loaded;
    
    // Write at 200MHz
    always @(posedge clk_200 or negedge rst_n) begin
        if (!rst_n) begin
            buf_phase0_i <= 0;
            buf_phase0_q <= 0;
            buf_phase1_i <= 0;
            buf_phase1_q <= 0;
            buf_loaded <= 1'b0;
        end else if (phase_valid) begin
            buf_phase0_i <= phase0_i;
            buf_phase0_q <= phase0_q;
            buf_phase1_i <= phase1_i_norm;
            buf_phase1_q <= phase1_q_norm;
            buf_loaded <= 1'b1;
        end
    end

    //==========================================================================
    // Output Multiplexer @ 400MHz
    // Alternates between phase0 and phase1 samples
    //==========================================================================
    
    reg phase_sel;
    reg buf_loaded_sync1, buf_loaded_sync2;
    reg output_active;
    
    // Synchronize buf_loaded to 400MHz domain
    always @(posedge clk_400 or negedge rst_n) begin
        if (!rst_n) begin
            buf_loaded_sync1 <= 1'b0;
            buf_loaded_sync2 <= 1'b0;
        end else begin
            buf_loaded_sync1 <= buf_loaded;
            buf_loaded_sync2 <= buf_loaded_sync1;
        end
    end
    
    // Phase selection and output
    always @(posedge clk_400 or negedge rst_n) begin
        if (!rst_n) begin
            phase_sel <= 1'b0;
            out_i <= 0;
            out_q <= 0;
            out_valid <= 1'b0;
            output_active <= 1'b0;
        end else begin
            if (buf_loaded_sync2) begin
                output_active <= 1'b1;
            end
            
            if (output_active) begin
                out_valid <= 1'b1;
                
                if (phase_sel == 1'b0) begin
                    // Output phase 0 (original sample)
                    out_i <= buf_phase0_i;
                    out_q <= buf_phase0_q;
                end else begin
                    // Output phase 1 (interpolated sample)
                    out_i <= buf_phase1_i;
                    out_q <= buf_phase1_q;
                end
                
                phase_sel <= ~phase_sel;
            end else begin
                out_valid <= 1'b0;
            end
        end
    end

endmodule
