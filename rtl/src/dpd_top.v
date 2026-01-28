//==============================================================================
// 6G PA GAN-DPD: Top-Level DPD System
//==============================================================================
//
// Description:
//   Top-level module integrating TDNN generator, A-SPSA adaptation,
//   temperature control, and CDC for weight updates.
//
// Clock Domains:
//   clk_nn (200 MHz)   - NN inference, I/O processing
//   clk_spsa (1 MHz)   - A-SPSA weight updates
//
// Interfaces:
//   AXI-Stream for ADC input and DAC output
//   AXI-Lite for control/status registers
//
// Author: Generated for 6G PA GAN-DPD Project
//==============================================================================

`timescale 1ns / 1ps

module dpd_top #(
    // Data width parameters
    parameter DATA_WIDTH      = 16,         // Q1.15 IQ samples
    parameter WEIGHT_WIDTH    = 16,         // Q1.15 weights
    parameter ACT_WIDTH       = 16,         // Q8.8 activations
    parameter ACC_WIDTH       = 32,         // Q16.16 accumulator
    
    // Network architecture
    parameter MEMORY_DEPTH    = 5,          // Memory taps
    parameter INPUT_DIM       = 30,         // 2 + 3*(M+1) + 2*M = 30 for M=5
    parameter HIDDEN1_DIM     = 32,
    parameter HIDDEN2_DIM     = 16,
    parameter OUTPUT_DIM      = 2,
    
    // Total parameters per bank (for 30-dim input with nonlinear features)
    parameter TOTAL_WEIGHTS   = 1554,       // FC1(30×32=960) + B1(32) + FC2(512) + B2(16) + FC3(32) + B3(2)
    
    // Temperature thresholds (raw ADC values)
    parameter TEMP_COLD_THRESH  = 12'd614,  // ~15°C
    parameter TEMP_HOT_THRESH   = 12'd1638  // ~40°C
)(
    // System clocks and reset
    input  wire                     clk_nn,         // 200 MHz NN clock
    input  wire                     clk_spsa,       // 1 MHz SPSA clock
    input  wire                     rst_n,          // Active-low reset
    
    // ADC Input Interface (AXI-Stream)
    input  wire [DATA_WIDTH-1:0]    s_axis_adc_i,   // ADC I channel
    input  wire [DATA_WIDTH-1:0]    s_axis_adc_q,   // ADC Q channel
    input  wire                     s_axis_adc_valid,
    output wire                     s_axis_adc_ready,
    
    // DAC Output Interface (AXI-Stream)
    output wire [DATA_WIDTH-1:0]    m_axis_dac_i,   // DAC I channel
    output wire [DATA_WIDTH-1:0]    m_axis_dac_q,   // DAC Q channel
    output wire                     m_axis_dac_valid,
    input  wire                     m_axis_dac_ready,
    
    // PA Feedback Interface (for A-SPSA)
    input  wire [DATA_WIDTH-1:0]    s_axis_fb_i,    // Feedback I channel
    input  wire [DATA_WIDTH-1:0]    s_axis_fb_q,    // Feedback Q channel
    input  wire                     s_axis_fb_valid,
    
    // Temperature sensor input
    input  wire [11:0]              temp_adc,       // 12-bit temperature ADC
    input  wire                     temp_valid,
    
    // Control/Status
    input  wire                     dpd_enable,     // Enable DPD processing
    input  wire                     spsa_enable,    // Enable A-SPSA adaptation
    input  wire [1:0]               force_temp_state, // Override temp state (debug)
    input  wire                     force_temp_en,    // Enable temp override
    output wire [1:0]               curr_temp_state,  // Current temperature state
    output wire                     temp_change_flag, // Temperature changed
    output wire [15:0]              curr_spsa_iter,   // Current SPSA iteration
    output wire [15:0]              curr_lr,          // Current learning rate
    output wire                     dpd_busy,
    output wire                     spsa_busy
);

    //==========================================================================
    // Internal Signals
    //==========================================================================
    
    // Temperature controller outputs
    wire [1:0]  temp_state;
    wire        temp_changed;
    wire        anneal_reset;
    
    // Weight bank selection
    wire [1:0]  weight_bank_sel;
    
    // TDNN generator interface
    wire [DATA_WIDTH-1:0] gen_out_i;
    wire [DATA_WIDTH-1:0] gen_out_q;
    wire                  gen_out_valid;
    wire                  gen_busy;
    
    // A-SPSA engine interface
    wire        spsa_weight_update_req;
    wire        spsa_weight_update_ack;
    wire [15:0] spsa_weight_addr;
    wire [WEIGHT_WIDTH-1:0] spsa_weight_data;
    wire        spsa_weight_we;
    wire [15:0] spsa_iteration;
    wire [15:0] spsa_learning_rate;
    wire [3:0]  spsa_fsm_state;
    wire        spsa_busy_int;
    
    // Deadband FSM outputs
    wire [1:0]  deadband_state;
    wire        deadband_spsa_enable;
    wire [1:0]  deadband_gain_mult;
    
    // Safety monitor outputs
    wire        safety_bypass_active;
    wire        safety_overflow_alarm;
    wire [15:0] safety_overflow_count;
    
    // DPD output signals
    wire signed [DATA_WIDTH-1:0] dpd_out_i;
    wire signed [DATA_WIDTH-1:0] dpd_out_q;
    
    // Shadow memory CDC signals
    wire        shadow_swap_req;
    wire        shadow_swap_ack;
    wire        shadow_busy;
    
    // Weight memory interface (NN side - 200 MHz)
    wire [15:0] nn_weight_addr;
    wire [WEIGHT_WIDTH-1:0] nn_weight_data;
    
    // Input buffer outputs
    wire [DATA_WIDTH*INPUT_DIM-1:0] input_vector;
    wire                            input_vector_valid;
    
    // Error metric
    wire signed [DATA_WIDTH-1:0] error_evm;
    wire        error_valid;
    
    //==========================================================================
    // Temperature Controller (1 MHz domain)
    //==========================================================================
    temp_controller #(
        .COLD_THRESHOLD(TEMP_COLD_THRESH),
        .HOT_THRESHOLD(TEMP_HOT_THRESH)
    ) u_temp_ctrl (
        .clk(clk_spsa),
        .rst_n(rst_n),
        
        // Temperature input
        .temp_adc(temp_adc),
        .temp_valid(temp_valid),
        
        // Override (debug)
        .force_state(force_temp_state),
        .force_enable(force_temp_en),
        
        // Outputs
        .temp_state(temp_state),
        .temp_changed(temp_changed),
        .anneal_reset(anneal_reset)
    );
    
    // Weight bank selection
    assign weight_bank_sel = temp_state;
    
    //==========================================================================
    // Deadband FSM (1 MHz domain - jitter prevention)
    //==========================================================================
    deadband_fsm u_deadband_fsm (
        .clk(clk_spsa),
        .rst_n(rst_n),
        
        // Error metric in Q8.8 format (from error_metric)
        .evm_db(error_evm),
        
        // Safety override from overflow detector
        .overflow_flag(safety_overflow_alarm),
        
        // ARM reset (when safety condition clears)
        .arm_reset(1'b0),  // Connect to ARM interrupt handler if available
        
        // State and control outputs
        .state(deadband_state),
        .spsa_enable(deadband_spsa_enable),
        .gain_mult(deadband_gain_mult)
    );
    
    //==========================================================================
    // Safety Monitor (250 MHz / 200 MHz domain - real-time overflow detection)
    //==========================================================================
    dpd_safety_monitor u_safety_monitor (
        .clk_data(clk_nn),
        .rst_n(rst_n),
        
        // DPD output samples
        .dpd_i(dpd_out_i),
        .dpd_q(dpd_out_q),
        
        // Control
        .arm_reset(1'b0),  // Connect to ARM interrupt handler if available
        
        // Status
        .bypass_active(safety_bypass_active),
        .overflow_alarm(safety_overflow_alarm),
        .overflow_count(safety_overflow_count)
    );
    
    //==========================================================================
    // Input Buffer with Memory Taps (200 MHz domain)
    //==========================================================================
    input_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .MEMORY_DEPTH(MEMORY_DEPTH),
        .OUTPUT_DIM(INPUT_DIM)
    ) u_input_buffer (
        .clk(clk_nn),
        .rst_n(rst_n),
        
        // Input IQ
        .in_i(s_axis_adc_i),
        .in_q(s_axis_adc_q),
        .in_valid(s_axis_adc_valid & dpd_enable),
        .in_ready(s_axis_adc_ready),
        
        // Output vector [I(n), Q(n), |x|, memory taps...]
        .out_vector(input_vector),
        .out_valid(input_vector_valid)
    );
    
    //==========================================================================
    // TDNN Generator (200 MHz domain)
    //==========================================================================
    tdnn_generator #(
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACT_WIDTH(ACT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .INPUT_DIM(INPUT_DIM),
        .HIDDEN1_DIM(HIDDEN1_DIM),
        .HIDDEN2_DIM(HIDDEN2_DIM),
        .OUTPUT_DIM(OUTPUT_DIM)
    ) u_generator (
        .clk(clk_nn),
        .rst_n(rst_n),
        
        // Input vector
        .in_vector(input_vector),
        .in_valid(input_vector_valid),
        
        // Weight memory interface
        .weight_addr(nn_weight_addr),
        .weight_data(nn_weight_data),
        .weight_bank_sel(weight_bank_sel),
        
        // Output
        .out_i(dpd_out_i),
        .out_q(dpd_out_q),
        .out_valid(gen_out_valid),
        
        // Status
        .busy(gen_busy)
    );
    
    //==========================================================================
    // Weight BRAM with Shadow Memory (CDC between domains)
    //==========================================================================
    shadow_memory #(
        .DATA_WIDTH(WEIGHT_WIDTH),
        .ADDR_WIDTH(16),
        .DEPTH(TOTAL_WEIGHTS * 3),  // 3 temperature banks
        .NUM_BANKS(3)
    ) u_shadow_mem (
        // NN side (200 MHz) - Read
        .clk_rd(clk_nn),
        .rst_n(rst_n),
        .rd_addr(nn_weight_addr),
        .rd_bank_sel(weight_bank_sel),
        .rd_data(nn_weight_data),
        
        // SPSA side (1 MHz) - Write
        .clk_wr(clk_spsa),
        .wr_addr(spsa_weight_addr),
        .wr_data(spsa_weight_data),
        .wr_en(spsa_weight_we),
        .wr_bank_sel(weight_bank_sel),
        
        // CDC handshake
        .swap_req(shadow_swap_req),
        .swap_ack(shadow_swap_ack),
        .busy(shadow_busy)
    );
    
    //==========================================================================
    // Error Metric Calculator (1 MHz domain)
    //==========================================================================
    error_metric #(
        .DATA_WIDTH(DATA_WIDTH)
    ) u_error_metric (
        .clk(clk_spsa),
        .rst_n(rst_n),
        
        // Desired output (what we expect)
        .desired_i(s_axis_adc_i),
        .desired_q(s_axis_adc_q),
        
        // Actual output (what PA produced)
        .actual_i(s_axis_fb_i),
        .actual_q(s_axis_fb_q),
        .sample_valid(s_axis_fb_valid),
        
        // Error output
        .error_metric(error_evm),
        .metric_valid(error_valid)
    );
    
    //==========================================================================
    // A-SPSA Engine (1 MHz domain) with Deadband Control
    //==========================================================================
    aspsa_engine #(
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .NUM_WEIGHTS(TOTAL_WEIGHTS),
        .LR_WIDTH(16)
    ) u_aspsa (
        .clk(clk_spsa),
        .rst_n(rst_n),
        
        // Control
        .enable(spsa_enable),
        .anneal_reset(anneal_reset | temp_changed),
        .deadband_state(deadband_state),
        .gain_mult(deadband_gain_mult),
        
        // Error input
        .error_metric(error_evm),
        .error_valid(error_valid),
        
        // Weight memory interface
        .weight_addr(spsa_weight_addr),
        .weight_data(spsa_weight_data),
        .weight_we(spsa_weight_we),
        
        // Shadow memory handshake
        .update_req(spsa_weight_update_req),
        .update_ack(spsa_weight_update_ack),
        
        // Status
        .iteration(spsa_iteration),
        .learning_rate(spsa_learning_rate),
        .spsa_state(spsa_fsm_state),
        .busy(spsa_busy_int)
    );
    
    //==========================================================================
    // Bypass Multiplexer (200 MHz domain - output stage)
    //==========================================================================
    wire [DATA_WIDTH-1:0] mux_out_i;
    wire [DATA_WIDTH-1:0] mux_out_q;
    
    dpd_bypass_mux u_bypass_mux (
        .clk_data(clk_nn),
        .rst_n(rst_n),
        
        // ADC passthrough path
        .adc_i(s_axis_adc_i),
        .adc_q(s_axis_adc_q),
        
        // DPD path
        .dpd_i(dpd_out_i),
        .dpd_q(dpd_out_q),
        
        // Control from safety monitor
        .bypass_active(safety_bypass_active),
        
        // Output to DAC
        .dac_i(mux_out_i),
        .dac_q(mux_out_q)
    );
    
    //==========================================================================
    // Output Stage (DPD enable gating)
    //==========================================================================
    
    // When DPD disabled, pass through input directly (before bypass mux)
    assign m_axis_dac_i = dpd_enable ? mux_out_i : s_axis_adc_i;
    assign m_axis_dac_q = dpd_enable ? mux_out_q : s_axis_adc_q;
    assign m_axis_dac_valid = dpd_enable ? gen_out_valid : s_axis_adc_valid;
    
    //==========================================================================
    // Status Outputs
    //==========================================================================
    assign curr_temp_state = temp_state;
    assign temp_change_flag = temp_changed;
    assign curr_spsa_iter = spsa_iteration;
    assign curr_lr = spsa_learning_rate;
    assign dpd_busy = gen_busy;
    assign spsa_busy = spsa_busy_int | shadow_busy;

endmodule
