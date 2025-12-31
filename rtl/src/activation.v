//==============================================================================
// Activation Functions Module for TDNN
// Provides LeakyReLU and Tanh (LUT-based)
//==============================================================================
// Features:
// - LeakyReLU with configurable slope (default 0.25 via >>2)
// - Tanh via 256-entry LUT with linear interpolation
// - Pipelined for 200MHz operation
// - Q8.8 input/output format
//==============================================================================

`timescale 1ns / 1ps

module activation #(
    parameter DATA_WIDTH = 16,
    parameter LUT_ADDR_WIDTH = 8,
    parameter LUT_DATA_WIDTH = 16
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Input (Q8.8)
    input  wire signed [DATA_WIDTH-1:0] in_data,
    input  wire                         in_valid,
    input  wire [1:0]                   act_sel,  // 0=None, 1=LeakyReLU, 2=Tanh
    
    // Output (Q8.8)
    output reg  signed [DATA_WIDTH-1:0] out_data,
    output reg                          out_valid
);

    //==========================================================================
    // Activation Select Encoding
    //==========================================================================
    localparam ACT_NONE = 2'd0;
    localparam ACT_LEAKYRELU = 2'd1;
    localparam ACT_TANH = 2'd2;

    //==========================================================================
    // LeakyReLU Implementation
    // f(x) = x if x >= 0, else 0.25*x
    // 0.25 implemented as arithmetic right shift by 2
    //==========================================================================
    reg signed [DATA_WIDTH-1:0] leakyrelu_out;
    
    always @(*) begin
        if (in_data >= 0) begin
            leakyrelu_out = in_data;
        end else begin
            // Arithmetic right shift preserves sign
            leakyrelu_out = in_data >>> 2;
        end
    end

    //==========================================================================
    // Tanh LUT
    // 256 entries covering input range [-4, 4) in Q8.8
    // Output in Q1.15 (tanh output is [-1, 1])
    //==========================================================================
    
    // LUT ROM (initialized from file)
    reg signed [LUT_DATA_WIDTH-1:0] tanh_lut [0:255];
    
    // Initialize LUT from file
    initial begin
        $readmemh("tanh_lut.hex", tanh_lut);
    end
    
    // LUT address generation
    // Map Q8.8 input to 8-bit address
    // Input range [-4, 4) maps to address [0, 255]
    wire [LUT_ADDR_WIDTH-1:0] lut_addr_base;
    wire [DATA_WIDTH-1:0] lut_frac;
    
    // Saturate input to [-4, 4)
    wire signed [DATA_WIDTH-1:0] in_sat;
    assign in_sat = (in_data > 16'h0400) ? 16'h03FF :   // > 4 -> saturate
                    (in_data < 16'hFC00) ? 16'hFC00 :   // < -4 -> saturate
                    in_data;
    
    // Convert to unsigned address: add offset of 4.0 (0x0400)
    wire [DATA_WIDTH-1:0] in_offset = in_sat + 16'h0400;  // Now [0, 8)
    
    // Use upper 8 bits as LUT address (Q8.8 >> 5 = Q3.0)
    assign lut_addr_base = in_offset[12:5];
    
    // Lower 5 bits for interpolation fraction
    assign lut_frac = {11'b0, in_offset[4:0]};
    
    // Registered LUT lookup with interpolation
    reg signed [LUT_DATA_WIDTH-1:0] lut_val0, lut_val1;
    reg [LUT_ADDR_WIDTH-1:0] lut_addr_d1;
    reg [DATA_WIDTH-1:0] lut_frac_d1;
    reg in_valid_d1, in_valid_d2;
    reg [1:0] act_sel_d1, act_sel_d2;
    reg signed [DATA_WIDTH-1:0] leakyrelu_d1, leakyrelu_d2;
    reg signed [DATA_WIDTH-1:0] in_data_d1, in_data_d2;
    
    // Stage 1: LUT read
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lut_val0 <= 0;
            lut_val1 <= 0;
            lut_addr_d1 <= 0;
            lut_frac_d1 <= 0;
            in_valid_d1 <= 1'b0;
            act_sel_d1 <= ACT_NONE;
            leakyrelu_d1 <= 0;
            in_data_d1 <= 0;
        end else begin
            in_valid_d1 <= in_valid;
            act_sel_d1 <= act_sel;
            leakyrelu_d1 <= leakyrelu_out;
            in_data_d1 <= in_data;
            
            if (in_valid) begin
                lut_val0 <= tanh_lut[lut_addr_base];
                lut_val1 <= tanh_lut[lut_addr_base + 1];
                lut_addr_d1 <= lut_addr_base;
                lut_frac_d1 <= lut_frac;
            end
        end
    end
    
    // Stage 2: Linear interpolation
    reg signed [DATA_WIDTH+5-1:0] interp_result;
    reg signed [LUT_DATA_WIDTH-1:0] lut_diff;
    reg signed [DATA_WIDTH-1:0] tanh_out;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lut_diff <= 0;
            interp_result <= 0;
            tanh_out <= 0;
            in_valid_d2 <= 1'b0;
            act_sel_d2 <= ACT_NONE;
            leakyrelu_d2 <= 0;
            in_data_d2 <= 0;
        end else begin
            in_valid_d2 <= in_valid_d1;
            act_sel_d2 <= act_sel_d1;
            leakyrelu_d2 <= leakyrelu_d1;
            in_data_d2 <= in_data_d1;
            
            if (in_valid_d1) begin
                // Linear interpolation: out = val0 + frac * (val1 - val0) / 32
                lut_diff <= lut_val1 - lut_val0;
                interp_result <= lut_val0 + ((lut_diff * lut_frac_d1) >>> 5);
                
                // Convert Q1.15 to Q8.8 (shift right by 7)
                tanh_out <= interp_result >>> 7;
            end
        end
    end

    //==========================================================================
    // Output Multiplexer
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_data <= 0;
            out_valid <= 1'b0;
        end else begin
            out_valid <= in_valid_d2;
            
            case (act_sel_d2)
                ACT_NONE:      out_data <= in_data_d2;
                ACT_LEAKYRELU: out_data <= leakyrelu_d2;
                ACT_TANH:      out_data <= tanh_out;
                default:       out_data <= in_data_d2;
            endcase
        end
    end

endmodule
