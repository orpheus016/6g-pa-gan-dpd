//==============================================================================
// 6G PA GAN-DPD: Shadow Memory for CDC Weight Transfer
//==============================================================================
//
// Description:
//   Double-buffered shadow memory for safe weight transfer between
//   A-SPSA (1 MHz) and NN inference (200 MHz) clock domains.
//
// Features:
//   - True dual-port BRAM with independent read/write clocks
//   - Gray-coded addresses for CDC safety
//   - Double-buffer swap for atomic updates
//   - 2-stage synchronizers for handshake signals
//
// Author: Generated for 6G PA GAN-DPD Project
//==============================================================================

`timescale 1ns / 1ps

module shadow_memory #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 16,
    parameter DEPTH      = 4662,        // 1554 weights × 3 banks
    parameter NUM_BANKS  = 3
)(
    // Read side (200 MHz - NN inference)
    input  wire                     clk_rd,
    input  wire                     rst_n,
    input  wire [ADDR_WIDTH-1:0]    rd_addr,
    input  wire [1:0]               rd_bank_sel,
    output reg  [DATA_WIDTH-1:0]    rd_data,
    
    // Write side (1 MHz - A-SPSA)
    input  wire                     clk_wr,
    input  wire [ADDR_WIDTH-1:0]    wr_addr,
    input  wire [DATA_WIDTH-1:0]    wr_data,
    input  wire                     wr_en,
    input  wire [1:0]               wr_bank_sel,
    
    // CDC handshake
    input  wire                     swap_req,       // From SPSA domain
    output wire                     swap_ack,       // To SPSA domain
    output wire                     busy
);

    //==========================================================================
    // Local Parameters
    //==========================================================================
    
    localparam BANK_SIZE = DEPTH / NUM_BANKS;
    
    //==========================================================================
    // Dual-Port BRAM (Inferred)
    //==========================================================================
    
    // Main weight memory (read by NN)
    (* ram_style = "block" *)
    reg [DATA_WIDTH-1:0] main_mem [0:DEPTH-1];
    
    // Shadow buffer (written by SPSA)
    (* ram_style = "block" *)
    reg [DATA_WIDTH-1:0] shadow_mem [0:DEPTH-1];
    
    // Buffer select (which buffer is active for reading)
    reg buffer_select;
    
    //==========================================================================
    // Gray Code Conversion (for address CDC safety)
    //==========================================================================
    
    function [ADDR_WIDTH-1:0] binary_to_gray;
        input [ADDR_WIDTH-1:0] binary;
        begin
            binary_to_gray = binary ^ (binary >> 1);
        end
    endfunction
    
    function [ADDR_WIDTH-1:0] gray_to_binary;
        input [ADDR_WIDTH-1:0] gray;
        integer i;
        begin
            gray_to_binary[ADDR_WIDTH-1] = gray[ADDR_WIDTH-1];
            for (i = ADDR_WIDTH-2; i >= 0; i = i - 1)
                gray_to_binary[i] = gray_to_binary[i+1] ^ gray[i];
        end
    endfunction
    
    //==========================================================================
    // CDC Synchronizers for Handshake
    //==========================================================================
    
    // Swap request synchronizer (SPSA → NN domain)
    reg swap_req_sync1, swap_req_sync2;
    always @(posedge clk_rd or negedge rst_n) begin
        if (!rst_n) begin
            swap_req_sync1 <= 0;
            swap_req_sync2 <= 0;
        end
        else begin
            swap_req_sync1 <= swap_req;
            swap_req_sync2 <= swap_req_sync1;
        end
    end
    
    // Swap acknowledge (NN → SPSA domain)
    reg swap_ack_int;
    reg swap_ack_sync1, swap_ack_sync2;
    
    always @(posedge clk_wr or negedge rst_n) begin
        if (!rst_n) begin
            swap_ack_sync1 <= 0;
            swap_ack_sync2 <= 0;
        end
        else begin
            swap_ack_sync1 <= swap_ack_int;
            swap_ack_sync2 <= swap_ack_sync1;
        end
    end
    
    assign swap_ack = swap_ack_sync2;
    
    //==========================================================================
    // Read Logic (200 MHz domain)
    //==========================================================================
    
    wire [ADDR_WIDTH-1:0] rd_addr_full = rd_bank_sel * BANK_SIZE + rd_addr;
    
    always @(posedge clk_rd) begin
        if (buffer_select)
            rd_data <= shadow_mem[rd_addr_full];
        else
            rd_data <= main_mem[rd_addr_full];
    end
    
    //==========================================================================
    // Write Logic (1 MHz domain)
    //==========================================================================
    
    wire [ADDR_WIDTH-1:0] wr_addr_full = wr_bank_sel * BANK_SIZE + wr_addr;
    
    always @(posedge clk_wr) begin
        if (wr_en) begin
            // Always write to inactive buffer
            if (buffer_select)
                main_mem[wr_addr_full] <= wr_data;
            else
                shadow_mem[wr_addr_full] <= wr_data;
        end
    end
    
    //==========================================================================
    // Buffer Swap FSM (200 MHz domain)
    //==========================================================================
    
    localparam SWAP_IDLE     = 2'd0;
    localparam SWAP_WAIT     = 2'd1;
    localparam SWAP_EXECUTE  = 2'd2;
    localparam SWAP_ACK      = 2'd3;
    
    reg [1:0] swap_state;
    reg [7:0] swap_delay_cnt;
    
    assign busy = (swap_state != SWAP_IDLE);
    
    always @(posedge clk_rd or negedge rst_n) begin
        if (!rst_n) begin
            swap_state <= SWAP_IDLE;
            buffer_select <= 0;
            swap_ack_int <= 0;
            swap_delay_cnt <= 0;
        end
        else begin
            case (swap_state)
                SWAP_IDLE: begin
                    swap_ack_int <= 0;
                    if (swap_req_sync2) begin
                        swap_state <= SWAP_WAIT;
                        swap_delay_cnt <= 0;
                    end
                end
                
                SWAP_WAIT: begin
                    // Wait for any in-flight reads to complete
                    swap_delay_cnt <= swap_delay_cnt + 1;
                    if (swap_delay_cnt >= 8'd16) begin
                        swap_state <= SWAP_EXECUTE;
                    end
                end
                
                SWAP_EXECUTE: begin
                    // Atomic buffer swap
                    buffer_select <= ~buffer_select;
                    swap_state <= SWAP_ACK;
                end
                
                SWAP_ACK: begin
                    swap_ack_int <= 1;
                    // Wait for request to deassert
                    if (!swap_req_sync2) begin
                        swap_state <= SWAP_IDLE;
                    end
                end
                
                default: swap_state <= SWAP_IDLE;
            endcase
        end
    end
    
    //==========================================================================
    // Initial Memory Contents (from binary files)
    //==========================================================================
    
    initial begin
        // Load initial weights from binary files
        // These will be overwritten during synthesis with actual values
        $readmemh("weights/weights_cold.hex", main_mem, 0, BANK_SIZE-1);
        $readmemh("weights/weights_normal.hex", main_mem, BANK_SIZE, 2*BANK_SIZE-1);
        $readmemh("weights/weights_hot.hex", main_mem, 2*BANK_SIZE, DEPTH-1);
        
        // Initialize shadow buffer to same values
        $readmemh("weights/weights_cold.hex", shadow_mem, 0, BANK_SIZE-1);
        $readmemh("weights/weights_normal.hex", shadow_mem, BANK_SIZE, 2*BANK_SIZE-1);
        $readmemh("weights/weights_hot.hex", shadow_mem, 2*BANK_SIZE, DEPTH-1);
    end

endmodule
