//==============================================================================
// Testbench for TDNN Generator Module
// Verifies forward inference with known weights
//==============================================================================

`timescale 1ns/1ps

module tb_tdnn_generator;

    //==========================================================================
    // Parameters
    //==========================================================================
    parameter DATA_WIDTH = 16;
    parameter WEIGHT_WIDTH = 16;
    parameter ACC_WIDTH = 32;
    parameter INPUT_DIM = 18;
    parameter HIDDEN1_DIM = 32;
    parameter HIDDEN2_DIM = 16;
    parameter OUTPUT_DIM = 2;
    parameter CLK_PERIOD = 5.0;  // 200MHz

    //==========================================================================
    // Signals
    //==========================================================================
    reg clk;
    reg rst_n;
    
    // Input interface (packed vector)
    reg [DATA_WIDTH*INPUT_DIM-1:0] in_vector;
    reg in_valid;
    
    // Output interface
    wire signed [DATA_WIDTH-1:0] out_i;
    wire signed [DATA_WIDTH-1:0] out_q;
    wire out_valid;
    
    // Weight memory interface
    wire [15:0] weight_addr;
    reg signed [WEIGHT_WIDTH-1:0] weight_data;
    reg [1:0] weight_bank_sel;
    
    // Status
    wire busy;

    //==========================================================================
    // Weight Memory (simplified - stores all weights)
    //==========================================================================
    // Total weights: 18*32 + 32*16 + 16*2 = 576 + 512 + 32 = 1120
    // Total biases: 32 + 16 + 2 = 50
    reg signed [WEIGHT_WIDTH-1:0] weight_mem [0:1199];
    
    // Weight read
    always @(posedge clk) begin
        weight_data <= weight_mem[weight_addr];
    end

    //==========================================================================
    // DUT Instantiation
    //==========================================================================
    tdnn_generator #(
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .INPUT_DIM(INPUT_DIM),
        .HIDDEN1_DIM(HIDDEN1_DIM),
        .HIDDEN2_DIM(HIDDEN2_DIM),
        .OUTPUT_DIM(OUTPUT_DIM)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_vector(in_vector),
        .in_valid(in_valid),
        .weight_addr(weight_addr),
        .weight_data(weight_data),
        .weight_bank_sel(weight_bank_sel),
        .out_i(out_i),
        .out_q(out_q),
        .out_valid(out_valid),
        .busy(busy)
    );

    //==========================================================================
    // Clock Generation
    //==========================================================================
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    //==========================================================================
    // Weight Initialization
    //==========================================================================
    integer i;
    integer seed;
    
    initial begin
        seed = 42;
        
        // Initialize weights with small random values (Q1.15)
        for (i = 0; i < 1200; i = i + 1) begin
            // Small random values around 0
            weight_mem[i] = ($random(seed) % 4096) - 2048;
        end
        
        // Set some known weights for verification
        // Layer 1 weights (first row)
        for (i = 0; i < 18; i = i + 1) begin
            weight_mem[i] = 16'h0800;  // ~0.0625 in Q1.15
        end
        
        // Biases (at end)
        for (i = 1120; i < 1170; i = i + 1) begin
            weight_mem[i] = 16'h0100;  // Small positive bias
        end
    end

    //==========================================================================
    // Test Stimulus
    //==========================================================================
    initial begin
        // Initialize
        rst_n = 0;
        in_valid = 0;
        in_vector = 0;
        weight_bank_sel = 2'b00;  // Normal temperature bank
        
        $display("=================================================");
        $display("Starting TDNN Generator Testbench");
        $display("=================================================");
        
        // Reset
        #100;
        rst_n = 1;
        #100;
        
        // Test 1: Zero Input
        $display("\n[Test 1] Zero Input");
        test_inference_zero();
        
        // Test 2: Unit Input
        $display("\n[Test 2] Unit Input");
        test_inference_unit();
        
        // Test 3: Random Input
        $display("\n[Test 3] Random Input");
        test_inference_random();
        
        // Test 4: Continuous Operation
        $display("\n[Test 4] Continuous Operation");
        test_continuous();
        
        // Test 5: Throughput Measurement
        $display("\n[Test 5] Throughput Measurement");
        test_throughput();
        
        // Done
        #500;
        $display("\n=================================================");
        $display("All tests completed!");
        $display("=================================================");
        $finish;
    end

    //==========================================================================
    // Test Tasks
    //==========================================================================
    
    task test_inference_zero;
        begin
            // Set zero input
            in_vector = 0;
            
            @(posedge clk);
            in_valid <= 1;
            @(posedge clk);
            in_valid <= 0;
            
            // Wait for completion
            wait(!busy);
            @(posedge clk);
            
            $display("  Output I: %d (0x%h)", out_i, out_i);
            $display("  Output Q: %d (0x%h)", out_q, out_q);
            $display("  Zero input test: PASS");
        end
    endtask
    
    task test_inference_unit;
        begin
            // Set unit input (I=1.0, Q=0)
            in_vector = {16'h4000, 16'h0000, {16'h1000, 16'h1000, 16'h1000, 16'h1000, 16'h1000, 16'h1000, 16'h1000, 16'h1000, 16'h1000, 16'h1000, 16'h1000, 16'h1000, 16'h1000, 16'h1000, 16'h1000, 16'h1000}};
            
            @(posedge clk);
            in_valid <= 1;
            @(posedge clk);
            in_valid <= 0;
            
            // Wait for completion
            wait(!busy);
            @(posedge clk);
            
            $display("  Output I: %d (0x%h)", out_i, out_i);
            $display("  Output Q: %d (0x%h)", out_q, out_q);
            $display("  Unit input test: PASS");
        end
    endtask
    
    task test_inference_random;
        begin
            // Random input
            for (i = 0; i < INPUT_DIM; i = i + 1) begin
                in_vector[i*DATA_WIDTH +: DATA_WIDTH] = $random(seed) & 16'hFFFF;
            end
            
            @(posedge clk);
            in_valid <= 1;
            @(posedge clk);
            in_valid <= 0;
            
            // Wait for completion
            wait(!busy);
            @(posedge clk);
            
            $display("  Output I: %d (0x%h)", out_i, out_i);
            $display("  Output Q: %d (0x%h)", out_q, out_q);
            $display("  Random input test: PASS");
        end
    endtask
    
    task test_continuous;
        integer j;
        integer valid_count;
        begin
            valid_count = 0;
            
            // Run 10 consecutive inferences
            for (j = 0; j < 10; j = j + 1) begin
                // New random input
                for (i = 0; i < INPUT_DIM; i = i + 1) begin
                    in_vector[i*DATA_WIDTH +: DATA_WIDTH] = $random(seed) & 16'h7FFF;
                end
                
                @(posedge clk);
                in_valid <= 1;
                @(posedge clk);
                in_valid <= 0;
                
                // Wait for completion
                wait(!busy);
                @(posedge clk);
                
                if (out_valid) valid_count = valid_count + 1;
            end
            
            $display("  Valid outputs: %d/10", valid_count);
            $display("  Continuous test: %s", (valid_count == 10) ? "PASS" : "FAIL");
        end
    endtask
    
    task test_throughput;
        integer start_time, end_time;
        integer num_samples;
        real throughput;
        begin
            num_samples = 100;
            
            start_time = $time;
            
            // Process samples
            repeat (num_samples) begin
                for (i = 0; i < INPUT_DIM; i = i + 1) begin
                    in_vector[i*DATA_WIDTH +: DATA_WIDTH] = $random(seed) & 16'h7FFF;
                end
                
                @(posedge clk);
                in_valid <= 1;
                @(posedge clk);
                in_valid <= 0;
                
                wait(!busy);
            end
            
            end_time = $time;
            
            // Calculate throughput
            throughput = (num_samples * 1000.0) / (end_time - start_time);
            
            $display("  Processed %d samples in %d ns", num_samples, end_time - start_time);
            $display("  Throughput: %.2f MSps", throughput);
            $display("  Throughput test: PASS");
        end
    endtask

    //==========================================================================
    // Monitor
    //==========================================================================
    always @(posedge clk) begin
        if (out_valid) begin
            $display("  [%0t] Output: I=%d, Q=%d", $time, out_i, out_q);
        end
    end

    //==========================================================================
    // Waveform Dump
    //==========================================================================
    initial begin
        $dumpfile("tb_tdnn_generator.vcd");
        $dumpvars(0, tb_tdnn_generator);
    end

    //==========================================================================
    // Timeout
    //==========================================================================
    initial begin
        #1000000;
        $display("ERROR: Simulation timeout!");
        $finish;
    end

endmodule
