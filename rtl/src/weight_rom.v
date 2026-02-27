//==============================================================================
// Weight ROM Stub for PN-TDNN DPD
// Dummy weights for synthesis - replace with actual trained weights later
//==============================================================================
// Architecture: FC1(24→32) → FC2(32→16) → FC3(16→2)
// Total parameters: 1,362 per bank
// 
// Weight layout:
//   [0-767]:    FC1 weights (24×32)
//   [768-799]:  FC1 biases (32)
//   [800-1311]: FC2 weights (32×16)
//   [1312-1327]: FC2 biases (16)
//   [1328-1359]: FC3 weights (16×2)
//   [1360-1361]: FC3 biases (2)
//==============================================================================

`timescale 1ns / 1ps

module weight_rom #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 16,
    parameter NUM_BANKS = 4,        // Temperature compensation banks
    parameter BANK_SIZE = 1362      // Parameters per bank
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Read interface (active for all 5 lanes)
    input  wire [ADDR_WIDTH-1:0]    addr_0,
    input  wire [ADDR_WIDTH-1:0]    addr_1,
    input  wire [ADDR_WIDTH-1:0]    addr_2,
    input  wire [ADDR_WIDTH-1:0]    addr_3,
    input  wire [ADDR_WIDTH-1:0]    addr_4,
    
    input  wire [1:0]               bank_sel,    // Select temperature bank
    
    output reg  [DATA_WIDTH-1:0]    data_0,
    output reg  [DATA_WIDTH-1:0]    data_1,
    output reg  [DATA_WIDTH-1:0]    data_2,
    output reg  [DATA_WIDTH-1:0]    data_3,
    output reg  [DATA_WIDTH-1:0]    data_4
);

    // Total ROM size: 4 banks × 1362 = 5448 entries
    localparam ROM_DEPTH = NUM_BANKS * BANK_SIZE;
    
    // ROM array (initialized with pseudo-random but synthesizable values)
    (* rom_style = "block" *) reg signed [DATA_WIDTH-1:0] rom [0:ROM_DEPTH-1];
    
    // Bank base address calculation
    wire [ADDR_WIDTH-1:0] bank_base = bank_sel * BANK_SIZE;
    
    // Initialize ROM with deterministic pattern for synthesis
    // These are NOT trained weights - just placeholder values
    // Pattern: small values that won't cause overflow
    integer i;
    initial begin
        for (i = 0; i < ROM_DEPTH; i = i + 1) begin
            // Generate pseudo-random small weights
            // Use linear congruential pattern: (a*i + c) mod m
            // Values scaled to ~0.01 range in Q1.15 (≈ 328)
            rom[i] = ((i * 1103515245 + 12345) % 65536) - 32768;
            // Scale down to prevent overflow
            rom[i] = rom[i] >>> 4;  // Divide by 16 for smaller weights
        end
    end
    
    // 5-port read (1 cycle latency)
    always @(posedge clk) begin
        if (!rst_n) begin
            data_0 <= 0;
            data_1 <= 0;
            data_2 <= 0;
            data_3 <= 0;
            data_4 <= 0;
        end else begin
            data_0 <= rom[bank_base + addr_0];
            data_1 <= rom[bank_base + addr_1];
            data_2 <= rom[bank_base + addr_2];
            data_3 <= rom[bank_base + addr_3];
            data_4 <= rom[bank_base + addr_4];
        end
    end

endmodule
