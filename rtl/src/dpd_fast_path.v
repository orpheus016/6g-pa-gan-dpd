//==============================================================================
// DPD Fast Path Top Module: Integrated FEX + Interpolator + TDNN
//==============================================================================
// Purpose: Complete DPD processing chain for Vivado timing/power analysis
//
// Architecture:
//   Raw IQ (200 MHz) → FEX (feature extraction) → TDNN (predistortion)
//   Note: Interpolator skeleton instantiated but bypassed pending upgrade
//
// Target: ZCU104 (xczu7ev-ffvc1156-2-e)
// Clock: 200 MHz (safe margin below 280 MHz TDNN Fmax)
//
// Dataflow:
//   1. FEX extracts 24 phase-normalized features @ 200 MSps (II=1, 23-cycle latency)
//   2. TDNN processes features → outputs predistorted IQ (~55 cycle latency)
//   3. Total latency: ~78 cycles @ 200 MHz = 390 ns
//
// Resource Estimates:
//   - DSPs: ~60 (FEX sqrt/mult + TDNN MACs)
//   - BRAMs: ~10 (TDNN weight storage: 1362 params × 3 banks)
//   - LUTs: ~15k (control logic, pipelines, state machines)
//
//==============================================================================

`timescale 1ns / 1ps

module dpd_fast_path #(
    parameter DATA_WIDTH = 16,          // Q1.15 fixed-point
    parameter FEATURE_DIM = 24,         // FEX output dimension (M=3 memory)
    parameter NUM_TEMP_BANKS = 3,       // Temperature compensation banks
    parameter WEIGHT_ADDR_WIDTH = 12    // 4096 words (covers 1362 × 3 = 4086)
)(
    //==========================================================================
    // Clock & Reset
    //==========================================================================
    input  wire                         clk,            // 200 MHz
    input  wire                         rst_n,
    
    //==========================================================================
    // Input: Raw IQ Samples (200 MSps)
    //==========================================================================
    input  wire signed [DATA_WIDTH-1:0] in_i,
    input  wire signed [DATA_WIDTH-1:0] in_q,
    input  wire                         in_valid,
    
    //==========================================================================
    // Weight Memory Interface (External BRAM for TDNN)
    //==========================================================================
    output wire [WEIGHT_ADDR_WIDTH-1:0] weight_addr,
    input  wire [DATA_WIDTH-1:0]        weight_data,
    input  wire [1:0]                   weight_bank_sel,  // 0-2 for temp banks
    
    //==========================================================================
    // Output: Predistorted IQ (Variable rate, ~3.6 MSps)
    //==========================================================================
    output wire signed [DATA_WIDTH-1:0] out_i,
    output wire signed [DATA_WIDTH-1:0] out_q,
    output wire                         out_valid,
    
    //==========================================================================
    // Status & Debug
    //==========================================================================
    output wire                         fex_busy,
    output wire                         tdnn_busy,
    output wire [15:0]                  debug_latency_cnt
);

    //==========================================================================
    // Internal Signals
    //==========================================================================
    
    // FEX → TDNN interface
    wire signed [DATA_WIDTH-1:0] fex_features [0:FEATURE_DIM-1];
    wire fex_valid;
    
    // Flattened feature vector for TDNN
    wire [DATA_WIDTH*FEATURE_DIM-1:0] fex_features_flat;
    genvar g;
    generate
        for (g = 0; g < FEATURE_DIM; g = g + 1) begin : gen_flatten
            assign fex_features_flat[DATA_WIDTH*g +: DATA_WIDTH] = fex_features[g];
        end
    endgenerate
    
    // Latency measurement (debug)
    reg [15:0] latency_counter;
    reg latency_measuring;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            latency_counter <= 0;
            latency_measuring <= 0;
        end
        else begin
            if (in_valid && !latency_measuring) begin
                latency_counter <= 1;
                latency_measuring <= 1'b1;
            end
            else if (latency_measuring && !out_valid) begin
                latency_counter <= latency_counter + 1;
            end
            else if (out_valid && latency_measuring) begin
                latency_measuring <= 1'b0;  // Stop on first output
            end
        end
    end
    
    assign debug_latency_cnt = latency_counter;
    
    //==========================================================================
    // Module 1: FEX Layer (Feature Extraction with Phase Normalization)
    //==========================================================================
    // Latency: 23 cycles @ 200 MHz
    // Throughput: II=1 (200 MSps)
    // Output: 24 features (8 phase-norm IQ + 4 mag + 4 A³ + 8 memory IQ)
    
    fex_layer_ii1_fixed fex_inst (
        .clk(clk),
        .rst_n(rst_n),
        
        // Input: Raw IQ
        .in_i(in_i),
        .in_q(in_q),
        .in_valid(in_valid),
        
        // Output: 24-dimensional features
        .out_features(fex_features),
        .out_valid(fex_valid),
        .busy(fex_busy)
    );
    
    //==========================================================================
    // Module 2: Interpolator Skeleton (Placeholder - Not in Data Path Yet)
    //==========================================================================
    // NOTE: Interpolator skeleton instantiated but outputs not connected
    // to critical path. Will be integrated after upgrade to proper 5x upsampler.
    //
    // Current skeleton outputs 5 parallel samples but TDNN expects serial input.
    // Options for future integration:
    //   A) Time-multiplex 5 samples into TDNN over 5 cycles
    //   B) Replicate TDNN 5x for parallel processing
    //   C) Add serializer to convert parallel→serial
    
    wire signed [DATA_WIDTH-1:0] interp_out_i [0:4];
    wire signed [DATA_WIDTH-1:0] interp_out_q [0:4];
    wire interp_valid;
    
    interpolator_skeleton interp_inst (
        .clk(clk),
        .rst_n(rst_n),
        
        // Input: Raw IQ (same as FEX input for now)
        .in_i(in_i),
        .in_q(in_q),
        .in_valid(in_valid),
        
        // Output: 5 parallel samples (not connected to data path yet)
        .out_i(interp_out_i),
        .out_q(interp_out_q),
        .out_valid(interp_valid)
    );
    
    // TODO: Connect interpolator outputs when upgraded to proper implementation
    // For timing analysis, having it instantiated provides resource/timing data
    
    //==========================================================================
    // Module 3: TDNN Generator (Predistortion Neural Network)
    //==========================================================================
    // Latency: ~55 cycles @ 200 MHz
    // Throughput: 1 sample per ~55 cycles (3.6 MSps)
    // Architecture: FC1(24→32) → ReLU → FC2(32→16) → ReLU → FC3(16→2) → Denorm
    
    tdnn_generator #(
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(DATA_WIDTH),
        .INPUT_DIM(FEATURE_DIM),
        .HIDDEN1_DIM(32),
        .HIDDEN2_DIM(16),
        .OUTPUT_DIM(2),
        .NUM_MACS(6)
    ) tdnn_inst (
        .clk(clk),
        .rst_n(rst_n),
        
        // Input: FEX features (24-dim flattened vector)
        .in_vector(fex_features_flat),
        .in_valid(fex_valid),
        
        // Weight memory interface
        .weight_addr(weight_addr),
        .weight_data(weight_data),
        .weight_bank_sel(weight_bank_sel),
        
        // Output: Predistorted IQ
        .out_i(out_i),
        .out_q(out_q),
        .out_valid(out_valid),
        .busy(tdnn_busy)
    );
    
    //==========================================================================
    // Synthesis Attributes (Vivado Optimization Hints)
    //==========================================================================
    
    // Mark critical paths for timing optimization
    (* KEEP = "TRUE", ASYNC_REG = "TRUE" *) reg timing_marker;
    
    // Register pipelining hints
    (* MAX_FANOUT = 32 *) wire fex_valid_fanout = fex_valid;
    
    //==========================================================================
    // Integration Notes for Future Upgrade
    //==========================================================================
    //
    // Current Architecture: IQ → FEX → TDNN (interpolator bypassed)
    //   - Works at 200 MSps throughout
    //   - Provides accurate timing data for FEX and TDNN
    //   - Interpolator resources visible in synthesis report
    //
    // Future Architecture Option A: IQ → Interpolator → FEX → TDNN
    //   - Interpolate raw IQ: 200 MSps → 1 GSps
    //   - FEX must run at 1 GSps (5x resource increase)
    //   - TDNN processes 1 GSps feature stream
    //   - Challenge: FEX sqrt/div at 1 GHz may not meet timing on ZCU104
    //
    // Future Architecture Option B: IQ → FEX → Interpolator → TDNN
    //   - Extract features at 200 MSps
    //   - Interpolate 24 features: 200 MSps → 1 GSps (24× parallel FIRs)
    //   - TDNN processes 1 GSps features
    //   - Challenge: Interpolating features vs. raw samples (non-standard)
    //
    // Recommended: Option A with FEX pipelining/parallelization at 200 MHz
    //   - Use parallel processing: FEX outputs 5 samples/clock
    //   - Time-multiplex into TDNN or replicate TDNN
    //
    //==========================================================================

endmodule
