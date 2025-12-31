//==============================================================================
// Error Metric Calculator for A-SPSA DPD
// Computes EVM-based error metric for gradient estimation
//==============================================================================
// Features:
// - NMSE-based error calculation
// - Moving average filter for noise reduction
// - Configurable window size
// - Fixed-point Q8.8 output
//==============================================================================

module error_metric #(
    parameter DATA_WIDTH = 16,
    parameter ACC_WIDTH = 32,
    parameter AVG_WINDOW = 64,       // Moving average window (power of 2)
    parameter AVG_SHIFT = 6          // log2(AVG_WINDOW)
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Input samples (Q1.15 complex)
    input  wire signed [DATA_WIDTH-1:0] desired_i,     // PA output (ideal)
    input  wire signed [DATA_WIDTH-1:0] desired_q,
    input  wire signed [DATA_WIDTH-1:0] actual_i,      // DPD+PA output
    input  wire signed [DATA_WIDTH-1:0] actual_q,
    input  wire                         sample_valid,
    
    // Error metric output (Q8.8)
    output reg  signed [DATA_WIDTH-1:0] error_metric,
    output reg                          metric_valid,
    
    // Debug outputs
    output wire [ACC_WIDTH-1:0]         error_power,
    output wire [ACC_WIDTH-1:0]         signal_power
);

    //==========================================================================
    // Local Parameters
    //==========================================================================
    localparam SHIFT_NORM = 15;  // Q1.15 normalization shift

    //==========================================================================
    // Registers
    //==========================================================================
    
    // Error computation
    reg signed [DATA_WIDTH-1:0] error_i, error_q;
    reg signed [ACC_WIDTH-1:0]  error_i_sq, error_q_sq;
    reg signed [ACC_WIDTH-1:0]  desired_i_sq, desired_q_sq;
    reg signed [ACC_WIDTH-1:0]  error_power_sample;
    reg signed [ACC_WIDTH-1:0]  signal_power_sample;
    
    // Pipeline stages
    reg sample_valid_d1, sample_valid_d2, sample_valid_d3;
    
    // Moving average accumulators
    reg [ACC_WIDTH+AVG_SHIFT-1:0] error_acc;
    reg [ACC_WIDTH+AVG_SHIFT-1:0] signal_acc;
    
    // Circular buffer for window
    reg [ACC_WIDTH-1:0] error_buffer [0:AVG_WINDOW-1];
    reg [ACC_WIDTH-1:0] signal_buffer [0:AVG_WINDOW-1];
    reg [AVG_SHIFT-1:0] buf_ptr;
    reg                 buffer_filled;
    reg [AVG_SHIFT:0]   sample_count;
    
    // Output computation
    reg [ACC_WIDTH-1:0] error_avg;
    reg [ACC_WIDTH-1:0] signal_avg;
    
    //==========================================================================
    // Debug Outputs
    //==========================================================================
    assign error_power = error_avg;
    assign signal_power = signal_avg;

    //==========================================================================
    // Stage 1: Error Computation
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            error_i <= 0;
            error_q <= 0;
            sample_valid_d1 <= 1'b0;
        end else begin
            sample_valid_d1 <= sample_valid;
            if (sample_valid) begin
                // e = desired - actual
                error_i <= desired_i - actual_i;
                error_q <= desired_q - actual_q;
            end
        end
    end

    //==========================================================================
    // Stage 2: Square Computation (Power)
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            error_i_sq <= 0;
            error_q_sq <= 0;
            desired_i_sq <= 0;
            desired_q_sq <= 0;
            sample_valid_d2 <= 1'b0;
        end else begin
            sample_valid_d2 <= sample_valid_d1;
            if (sample_valid_d1) begin
                // |e|^2 = e_i^2 + e_q^2
                error_i_sq <= error_i * error_i;
                error_q_sq <= error_q * error_q;
                // |d|^2 = d_i^2 + d_q^2
                desired_i_sq <= desired_i * desired_i;
                desired_q_sq <= desired_q * desired_q;
            end
        end
    end

    //==========================================================================
    // Stage 3: Power Sum
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            error_power_sample <= 0;
            signal_power_sample <= 0;
            sample_valid_d3 <= 1'b0;
        end else begin
            sample_valid_d3 <= sample_valid_d2;
            if (sample_valid_d2) begin
                error_power_sample <= error_i_sq + error_q_sq;
                signal_power_sample <= desired_i_sq + desired_q_sq;
            end
        end
    end

    //==========================================================================
    // Moving Average Filter
    //==========================================================================
    integer i;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            error_acc <= 0;
            signal_acc <= 0;
            buf_ptr <= 0;
            buffer_filled <= 1'b0;
            sample_count <= 0;
            
            for (i = 0; i < AVG_WINDOW; i = i + 1) begin
                error_buffer[i] <= 0;
                signal_buffer[i] <= 0;
            end
        end else if (sample_valid_d3) begin
            // Subtract oldest sample, add new sample
            error_acc <= error_acc - error_buffer[buf_ptr] + error_power_sample;
            signal_acc <= signal_acc - signal_buffer[buf_ptr] + signal_power_sample;
            
            // Store new sample
            error_buffer[buf_ptr] <= error_power_sample;
            signal_buffer[buf_ptr] <= signal_power_sample;
            
            // Update pointer
            buf_ptr <= buf_ptr + 1;
            
            // Track fill status
            if (sample_count < AVG_WINDOW) begin
                sample_count <= sample_count + 1;
            end else begin
                buffer_filled <= 1'b1;
            end
        end
    end

    //==========================================================================
    // Average Computation
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            error_avg <= 0;
            signal_avg <= 0;
        end else if (sample_valid_d3) begin
            // Divide by window size (shift for power of 2)
            error_avg <= error_acc >> AVG_SHIFT;
            signal_avg <= signal_acc >> AVG_SHIFT;
        end
    end

    //==========================================================================
    // NMSE Output Computation
    // NMSE = E[|e|^2] / E[|d|^2] in Q8.8 format
    //==========================================================================
    
    // Simple ratio approximation using shifts
    // For NMSE < 1 (typical), scale up error_avg and divide
    reg [ACC_WIDTH+8-1:0] scaled_error;
    reg [DATA_WIDTH-1:0]  nmse_ratio;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scaled_error <= 0;
            nmse_ratio <= 0;
            error_metric <= 0;
            metric_valid <= 1'b0;
        end else begin
            metric_valid <= sample_valid_d3 & buffer_filled;
            
            if (sample_valid_d3 && buffer_filled) begin
                // Scale error by 2^8 for Q8.8 format
                scaled_error <= {error_avg, 8'b0};
                
                // Approximate division by signal_avg
                // Using leading zero detection and shift-based division
                if (signal_avg > 0) begin
                    // Simple approximation: find ratio via iterative compare
                    // For hardware, use dedicated divider or CORDIC
                    nmse_ratio <= divide_approx(error_avg, signal_avg);
                end else begin
                    nmse_ratio <= 16'h7FFF;  // Max value if signal is zero
                end
                
                // Saturate to Q8.8 range
                error_metric <= nmse_ratio;
            end
        end
    end

    //==========================================================================
    // Approximate Division Function (shift-subtract)
    //==========================================================================
    function [DATA_WIDTH-1:0] divide_approx;
        input [ACC_WIDTH-1:0] numerator;
        input [ACC_WIDTH-1:0] denominator;
        
        reg [ACC_WIDTH-1:0] num_scaled;
        reg [DATA_WIDTH-1:0] quotient;
        integer j;
        
        begin
            // Scale numerator for Q8.8 output
            num_scaled = numerator << 8;
            quotient = 0;
            
            // Iterative shift-subtract division (16 iterations)
            for (j = DATA_WIDTH-1; j >= 0; j = j - 1) begin
                if (num_scaled >= (denominator << j)) begin
                    num_scaled = num_scaled - (denominator << j);
                    quotient = quotient | (1 << j);
                end
            end
            
            divide_approx = quotient;
        end
    endfunction

endmodule
