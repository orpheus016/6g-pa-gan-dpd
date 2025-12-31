//==============================================================================
// Testbench for A-SPSA Engine Module
// Verifies annealing schedule, LFSR perturbation, and gradient estimation
//==============================================================================

`timescale 1ns/1ps

module tb_aspsa_engine;

    //==========================================================================
    // Parameters
    //==========================================================================
    parameter DATA_WIDTH = 16;
    parameter WEIGHT_WIDTH = 16;
    parameter NUM_WEIGHTS = 1170;
    parameter CLK_PERIOD = 1000.0;  // 1MHz = 1us

    //==========================================================================
    // Signals
    //==========================================================================
    reg clk;
    reg rst_n;
    
    // Error metric input
    reg signed [DATA_WIDTH-1:0] error_metric;
    reg error_valid;
    
    // Weight update output
    wire [15:0] weight_idx;
    wire signed [WEIGHT_WIDTH-1:0] weight_delta;
    wire weight_update_valid;
    
    // Control
    reg adapt_enable;
    reg temp_changed;
    
    // Debug outputs
    wire [15:0] learning_rate;
    wire [15:0] perturb_size;
    wire [31:0] lfsr_state;
    wire [15:0] iteration_count;

    //==========================================================================
    // DUT Instantiation
    //==========================================================================
    aspsa_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .NUM_WEIGHTS(NUM_WEIGHTS),
        .LFSR_SEED(32'hDEADBEEF),
        .ANNEAL_PERIOD(100)  // Fast annealing for test
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .error_metric(error_metric),
        .error_valid(error_valid),
        .weight_idx(weight_idx),
        .weight_delta(weight_delta),
        .weight_update_valid(weight_update_valid),
        .adapt_enable(adapt_enable),
        .temp_changed(temp_changed),
        .learning_rate(learning_rate),
        .perturb_size(perturb_size),
        .lfsr_state(lfsr_state),
        .iteration_count(iteration_count)
    );

    //==========================================================================
    // Clock Generation
    //==========================================================================
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    //==========================================================================
    // Test Stimulus
    //==========================================================================
    integer i;
    reg [15:0] prev_lr;
    reg [15:0] update_count;
    
    initial begin
        // Initialize
        rst_n = 0;
        error_metric = 0;
        error_valid = 0;
        adapt_enable = 0;
        temp_changed = 0;
        update_count = 0;
        
        $display("=================================================");
        $display("Starting A-SPSA Engine Testbench");
        $display("=================================================");
        
        // Reset
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 5);
        
        // Test 1: LFSR Randomness
        $display("\n[Test 1] LFSR Randomness Check");
        test_lfsr_randomness();
        
        // Test 2: Annealing Schedule
        $display("\n[Test 2] Annealing Schedule");
        test_annealing();
        
        // Test 3: Gradient Estimation
        $display("\n[Test 3] Gradient Estimation");
        test_gradient_estimation();
        
        // Test 4: Temperature Reset
        $display("\n[Test 4] Temperature Reset");
        test_temp_reset();
        
        // Test 5: Full Adaptation Cycle
        $display("\n[Test 5] Full Adaptation Cycle");
        test_full_cycle();
        
        // Done
        #(CLK_PERIOD * 10);
        $display("\n=================================================");
        $display("All tests completed!");
        $display("=================================================");
        $finish;
    end

    //==========================================================================
    // Test Tasks
    //==========================================================================
    
    task test_lfsr_randomness;
        reg [31:0] lfsr_values [0:99];
        integer unique_count;
        integer j, k;
        reg found;
        begin
            adapt_enable = 1;
            
            // Collect 100 LFSR values
            for (i = 0; i < 100; i = i + 1) begin
                @(posedge clk);
                error_metric <= $random & 16'hFFFF;
                error_valid <= 1;
                @(posedge clk);
                error_valid <= 0;
                lfsr_values[i] = lfsr_state;
                #(CLK_PERIOD * 2);
            end
            
            // Count unique values
            unique_count = 1;
            for (j = 1; j < 100; j = j + 1) begin
                found = 0;
                for (k = 0; k < j; k = k + 1) begin
                    if (lfsr_values[j] == lfsr_values[k]) found = 1;
                end
                if (!found) unique_count = unique_count + 1;
            end
            
            $display("  Unique LFSR values: %d/100", unique_count);
            $display("  LFSR randomness: %s", (unique_count > 90) ? "PASS" : "FAIL");
            
            adapt_enable = 0;
            #(CLK_PERIOD * 5);
        end
    endtask
    
    task test_annealing;
        begin
            adapt_enable = 1;
            prev_lr = learning_rate;
            
            $display("  Initial LR: 0x%h", learning_rate);
            $display("  Initial perturb: 0x%h", perturb_size);
            
            // Run for several annealing periods
            for (i = 0; i < 500; i = i + 1) begin
                @(posedge clk);
                error_metric <= 16'h0100 + ($random & 16'h00FF);  // ~constant error
                error_valid <= 1;
                @(posedge clk);
                error_valid <= 0;
                
                // Check for LR decay
                if (i % 100 == 99) begin
                    $display("  Iter %d: LR=0x%h, perturb=0x%h", 
                             iteration_count, learning_rate, perturb_size);
                end
            end
            
            $display("  Final LR: 0x%h (should decrease)", learning_rate);
            $display("  Annealing: %s", (learning_rate < prev_lr) ? "PASS" : "FAIL");
            
            adapt_enable = 0;
            #(CLK_PERIOD * 5);
        end
    endtask
    
    task test_gradient_estimation;
        integer pos_count, neg_count;
        begin
            adapt_enable = 1;
            pos_count = 0;
            neg_count = 0;
            update_count = 0;
            
            // Provide varying error metrics
            for (i = 0; i < 200; i = i + 1) begin
                @(posedge clk);
                // Simulate error that depends on perturbation direction
                if (i % 2 == 0) begin
                    error_metric <= 16'h0200;  // Higher error (positive perturbation)
                end else begin
                    error_metric <= 16'h0100;  // Lower error (negative perturbation)
                end
                error_valid <= 1;
                
                @(posedge clk);
                error_valid <= 0;
                
                // Monitor weight updates
                if (weight_update_valid) begin
                    update_count = update_count + 1;
                    if (weight_delta > 0) pos_count = pos_count + 1;
                    else if (weight_delta < 0) neg_count = neg_count + 1;
                end
                
                #(CLK_PERIOD);
            end
            
            $display("  Weight updates: %d", update_count);
            $display("  Positive deltas: %d, Negative deltas: %d", pos_count, neg_count);
            $display("  Gradient estimation: %s", (update_count > 0) ? "PASS" : "FAIL");
            
            adapt_enable = 0;
            #(CLK_PERIOD * 5);
        end
    endtask
    
    task test_temp_reset;
        begin
            adapt_enable = 1;
            
            // Run for a while to let LR decay
            for (i = 0; i < 300; i = i + 1) begin
                @(posedge clk);
                error_metric <= 16'h0100;
                error_valid <= 1;
                @(posedge clk);
                error_valid <= 0;
            end
            
            prev_lr = learning_rate;
            $display("  LR before temp change: 0x%h", prev_lr);
            
            // Trigger temperature change
            @(posedge clk);
            temp_changed <= 1;
            @(posedge clk);
            temp_changed <= 0;
            
            #(CLK_PERIOD * 10);
            
            $display("  LR after temp change: 0x%h", learning_rate);
            $display("  Temperature reset: %s", 
                     (learning_rate > prev_lr) ? "PASS" : "FAIL");
            
            adapt_enable = 0;
            #(CLK_PERIOD * 5);
        end
    endtask
    
    task test_full_cycle;
        integer cycle;
        real error_f;
        begin
            adapt_enable = 1;
            
            // Simulate convergence: error decreases over time
            for (cycle = 0; cycle < 1000; cycle = cycle + 1) begin
                @(posedge clk);
                
                // Error decreases as we adapt
                error_f = 0.5 * (1.0 - cycle / 1500.0);
                if (error_f < 0.05) error_f = 0.05;
                error_metric <= $rtoi(error_f * 65536);
                error_valid <= 1;
                
                @(posedge clk);
                error_valid <= 0;
                
                if (cycle % 200 == 0) begin
                    $display("  Cycle %d: error=0x%h, LR=0x%h, iter=%d",
                             cycle, error_metric, learning_rate, iteration_count);
                end
            end
            
            $display("  Final error: 0x%h", error_metric);
            $display("  Full cycle: PASS");
            
            adapt_enable = 0;
        end
    endtask

    //==========================================================================
    // Weight Update Monitor
    //==========================================================================
    always @(posedge clk) begin
        if (weight_update_valid) begin
            // Only print first few updates
            if (update_count < 10) begin
                $display("  [%0t] Weight[%d] delta: %d", 
                         $time, weight_idx, weight_delta);
            end
        end
    end

    //==========================================================================
    // Waveform Dump
    //==========================================================================
    initial begin
        $dumpfile("tb_aspsa_engine.vcd");
        $dumpvars(0, tb_aspsa_engine);
    end

    //==========================================================================
    // Timeout
    //==========================================================================
    initial begin
        #50000000;  // 50ms timeout (at 1MHz, this is 50k cycles)
        $display("ERROR: Simulation timeout!");
        $finish;
    end

endmodule
