//==============================================================================
// DPD Top Module - 1 GSps Super Sample Rate Architecture
//==============================================================================
// Single 200 MHz clock domain with SSR=5 (5 parallel lanes)
// Effective throughput: 5 × 200 MHz = 1 GSps
//
// Architecture:
//   Input IQ (200 MSps) → Polyphase Interpolator → 5 Parallel Lanes
//   Each lane: FEX → PN-TDNN → Output
//   Output: 5 IQ pairs per clock (160-bit bus)
//
// Resource Estimate (ZCU104):
//   - Interpolator: ~120 DSP48 (60 taps × 2 for I/Q)
//   - FEX × 5: ~50 DSP48 (magnitude, powers)
//   - TDNN × 5: ~300 DSP48 (MAC operations)
//   - Weight ROM: ~22 KB BRAM (1362 × 16-bit × 4 banks)
//==============================================================================

`timescale 1ns / 1ps

module dpd_top_ssr #(
    parameter DATA_WIDTH = 16,
    parameter WEIGHT_WIDTH = 16,
    parameter INPUT_DIM = 24,
    parameter NUM_LANES = 5
)(
    input  wire                     clk,            // 200 MHz system clock
    input  wire                     rst_n,          // Active-low reset
    
    // Input: Single IQ sample @ 200 MSps
    input  wire signed [DATA_WIDTH-1:0] in_i,
    input  wire signed [DATA_WIDTH-1:0] in_q,
    input  wire                     in_valid,
    
    // Temperature bank select (for weight ROM)
    input  wire [1:0]               temp_bank_sel,
    
    // Output: 5 parallel IQ samples @ 200 MHz = 1 GSps aggregate
    output wire signed [DATA_WIDTH-1:0] out_i_0,
    output wire signed [DATA_WIDTH-1:0] out_q_0,
    output wire signed [DATA_WIDTH-1:0] out_i_1,
    output wire signed [DATA_WIDTH-1:0] out_q_1,
    output wire signed [DATA_WIDTH-1:0] out_i_2,
    output wire signed [DATA_WIDTH-1:0] out_q_2,
    output wire signed [DATA_WIDTH-1:0] out_i_3,
    output wire signed [DATA_WIDTH-1:0] out_q_3,
    output wire signed [DATA_WIDTH-1:0] out_i_4,
    output wire signed [DATA_WIDTH-1:0] out_q_4,
    output wire                     out_valid,
    
    // Status
    output wire                     busy
);

    //==========================================================================
    // Internal Signals
    //==========================================================================
    
    // Interpolator outputs (5 parallel phases)
    wire signed [DATA_WIDTH-1:0] interp_i [0:4];
    wire signed [DATA_WIDTH-1:0] interp_q [0:4];
    wire interp_valid;
    
    // FEX outputs (5 lanes × 24 features × 16 bits)
    wire [DATA_WIDTH*INPUT_DIM-1:0] fex_features [0:4];
    wire fex_valid [0:4];
    wire fex_busy [0:4];
    
    // TDNN outputs (5 lanes)
    wire signed [DATA_WIDTH-1:0] tdnn_out_i [0:4];
    wire signed [DATA_WIDTH-1:0] tdnn_out_q [0:4];
    wire tdnn_out_valid [0:4];
    wire tdnn_busy [0:4];
    
    // Weight ROM interface
    wire [15:0] weight_addr [0:4];
    wire [WEIGHT_WIDTH-1:0] weight_data [0:4];
    
    //==========================================================================
    // Polyphase Interpolator (1:5 upsampling)
    // Input: 1 sample @ 200 MHz
    // Output: 5 parallel samples @ 200 MHz (= 1 GSps effective)
    //==========================================================================
    
    interpolator1_5_ssr #(
        .DATA_WIDTH(DATA_WIDTH),
        .COEF_WIDTH(16),
        .TAPS_PER_PHASE(12),
        .ACC_WIDTH(40)
    ) u_interpolator (
        .clk(clk),
        .rst_n(rst_n),
        .in_i(in_i),
        .in_q(in_q),
        .in_valid(in_valid),
        .out_i_0(interp_i[0]),
        .out_q_0(interp_q[0]),
        .out_i_1(interp_i[1]),
        .out_q_1(interp_q[1]),
        .out_i_2(interp_i[2]),
        .out_q_2(interp_q[2]),
        .out_i_3(interp_i[3]),
        .out_q_3(interp_q[3]),
        .out_i_4(interp_i[4]),
        .out_q_4(interp_q[4]),
        .out_valid(interp_valid)
    );
    
    //==========================================================================
    // Weight ROM (shared across all 5 TDNN lanes)
    //==========================================================================
    
    weight_rom #(
        .DATA_WIDTH(WEIGHT_WIDTH),
        .ADDR_WIDTH(16),
        .NUM_BANKS(4),
        .BANK_SIZE(1362)
    ) u_weight_rom (
        .clk(clk),
        .rst_n(rst_n),
        .addr_0(weight_addr[0]),
        .addr_1(weight_addr[1]),
        .addr_2(weight_addr[2]),
        .addr_3(weight_addr[3]),
        .addr_4(weight_addr[4]),
        .bank_sel(temp_bank_sel),
        .data_0(weight_data[0]),
        .data_1(weight_data[1]),
        .data_2(weight_data[2]),
        .data_3(weight_data[3]),
        .data_4(weight_data[4])
    );
    
    //==========================================================================
    // Generate 5 Parallel Processing Lanes
    //==========================================================================
    
    genvar lane;
    generate
        for (lane = 0; lane < NUM_LANES; lane = lane + 1) begin : gen_lane
            
            //------------------------------------------------------------------
            // Feature Extraction (FEX)
            //------------------------------------------------------------------
            fex_layer_synth #(
                .DATA_WIDTH(DATA_WIDTH),
                .MEMORY_DEPTH(4)
            ) u_fex (
                .clk(clk),
                .rst_n(rst_n),
                .in_i(interp_i[lane]),
                .in_q(interp_q[lane]),
                .in_valid(interp_valid),
                .out_features(fex_features[lane]),
                .out_valid(fex_valid[lane]),
                .busy(fex_busy[lane])
            );
            
            //------------------------------------------------------------------
            // PN-TDNN Neural Network
            //------------------------------------------------------------------
            tdnn_generator #(
                .DATA_WIDTH(DATA_WIDTH),
                .WEIGHT_WIDTH(WEIGHT_WIDTH),
                .ACT_WIDTH(16),
                .ACC_WIDTH(32),
                .MEMORY_DEPTH(3),
                .INPUT_DIM(INPUT_DIM),
                .HIDDEN1_DIM(32),
                .HIDDEN2_DIM(16),
                .OUTPUT_DIM(2),
                .NUM_MACS(6)
            ) u_tdnn (
                .clk(clk),
                .rst_n(rst_n),
                .in_vector(fex_features[lane]),
                .in_valid(fex_valid[lane]),
                .weight_addr(weight_addr[lane]),
                .weight_data(weight_data[lane]),
                .weight_bank_sel(temp_bank_sel),
                .out_i(tdnn_out_i[lane]),
                .out_q(tdnn_out_q[lane]),
                .out_valid(tdnn_out_valid[lane]),
                .busy(tdnn_busy[lane])
            );
            
        end
    endgenerate
    
    //==========================================================================
    // Output Assignment
    //==========================================================================
    
    assign out_i_0 = tdnn_out_i[0];
    assign out_q_0 = tdnn_out_q[0];
    assign out_i_1 = tdnn_out_i[1];
    assign out_q_1 = tdnn_out_q[1];
    assign out_i_2 = tdnn_out_i[2];
    assign out_q_2 = tdnn_out_q[2];
    assign out_i_3 = tdnn_out_i[3];
    assign out_q_3 = tdnn_out_q[3];
    assign out_i_4 = tdnn_out_i[4];
    assign out_q_4 = tdnn_out_q[4];
    
    // Output valid when all lanes are valid
    // Note: Due to pipelining, lanes may have different valid timing
    // For now, use lane 0 as reference (all lanes have same latency)
    assign out_valid = tdnn_out_valid[0];
    
    // Busy when any lane is processing
    assign busy = tdnn_busy[0] | tdnn_busy[1] | tdnn_busy[2] | 
                  tdnn_busy[3] | tdnn_busy[4];

endmodule
