//==============================================================================
// FEX Layer II=1: Comprehensive Testbench
//==============================================================================
// Tests:
// 1. Latency: Measure cycles to first output
// 2. Throughput: Measure initiation interval (should be 1)
// 3. Accuracy: Verify numerical correctness (CORDIC, phase norm, amplitudes)
// 4. Memory tapeline: Verify feature history is captured correctly
//==============================================================================

`timescale 1ns / 1ps

module fex_layer_ii1_tb;

    reg clk, rst_n;
    reg signed [15:0] in_i, in_q;
    reg in_valid;
    
    wire signed [15:0] out_features [0:23];
    wire out_valid, busy;
    
    fex_layer_ii1_fixed dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_i(in_i),
        .in_q(in_q),
        .in_valid(in_valid),
        .out_features(out_features),
        .out_valid(out_valid),
        .busy(busy)
    );
    
    always begin
        clk = 0; #2;
        clk = 1; #2;
    end
    
    integer cycle, output_count, last_output_cycle;
    integer ii_vals [0:19];
    integer ii, ii_min, ii_max, ii_total, i;
    real i_norm, q_norm, amp, amp_cubed;
    
    initial begin
        rst_n = 0;
        in_valid = 0;
        in_i = 0;
        in_q = 0;
        
        #10;
        rst_n = 1;
        #10;
        
        $display("\n================================================================================");
        $display("FEX LAYER II=1: THROUGHPUT & ACCURACY TEST");
        $display("================================================================================\n");
        
        //======================================================================
        // TEST 1: Latency Measurement
        //======================================================================
        
        $display("TEST 1: Latency to First Output\n");
        
        in_i = 16'h4000;
        in_q = 16'h0000;
        in_valid = 1;
        #4;
        in_valid = 0;
        
        cycle = 0;
        output_count = 0;
        wait(out_valid);
        begin : LATENCY_LOOP
            integer temp_cycle;
            for (temp_cycle = 0; temp_cycle < 100; temp_cycle = temp_cycle + 1) begin
                @(posedge clk);
                if (out_valid) begin
                    $display("  ✓ First output at cycle %d", temp_cycle);
                    $display("  Expected: ~21-22 cycles (16 CORDIC + 1 A² + 4 phase norm)\n");
                    last_output_cycle = temp_cycle;
                    output_count = 1;
                    disable LATENCY_LOOP;
                end
            end
        end
        
        if (output_count == 0) begin
            $display("  ✗ FAIL: No output within 100 cycles\n");
        end
        
        #100;
        
        //======================================================================
        // TEST 2: Initiation Interval (II=1 verification)
        //======================================================================
        
        $display("TEST 2: Initiation Interval Measurement (II should be 1)\n");
        $display("Sending 20 continuous inputs...\n");
        
        in_i = 16'h4000;
        in_q = 16'h0000;
        in_valid = 1;
        output_count = 0;
        
        begin : II_LOOP
            integer temp_cycle;
            for (temp_cycle = 0; temp_cycle < 300; temp_cycle = temp_cycle + 1) begin
                @(posedge clk);
                
                if (out_valid) begin
                    if (output_count == 0) begin
                        $display("Output 1: Cycle %d (first)", temp_cycle);
                        last_output_cycle = temp_cycle;
                        output_count = 1;
                    end
                    else if (output_count < 20) begin
                        ii = temp_cycle - last_output_cycle;
                        $display("Output %d: Cycle %d (II = %d)", output_count + 1, temp_cycle, ii);
                        ii_vals[output_count - 1] = ii;
                        last_output_cycle = temp_cycle;
                        output_count = output_count + 1;
                    end
                    else begin
                        disable II_LOOP;
                    end
                end
            end
        end
        
        in_valid = 0;
        
        // Analyze II
        if (output_count > 1) begin
            ii_min = ii_vals[0];
            ii_max = ii_vals[0];
            ii_total = 0;
            
            for (i = 0; i < output_count - 1; i = i + 1) begin
                if (ii_vals[i] < ii_min) ii_min = ii_vals[i];
                if (ii_vals[i] > ii_max) ii_max = ii_vals[i];
                ii_total = ii_total + ii_vals[i];
            end
            
            $display("\n--- II Statistics ---");
            $display("Total outputs: %d", output_count);
            $display("II values: min=%d, max=%d, avg=%.1f", ii_min, ii_max, 
                     real'(ii_total) / real'(output_count - 1));
            
            if (ii_min == 1 && ii_max == 1) begin
                $display("✓ PASS: Perfect II=1 (pipelined)\n");
            end
            else if (ii_min <= 2 && ii_max <= 2) begin
                $display("⚠ WARNING: II mostly 1, occasional 2\n");
            end
            else begin
                $display("✗ FAIL: II > 2 (not properly pipelined)\n");
            end
        end
        
        #200;
        
        //======================================================================
        // TEST 3: Accuracy - DC Input
        //======================================================================
        
        $display("TEST 3: Accuracy - DC Input (I=0.5, Q=0)\n");
        
        in_i = 16'h4000;
        in_q = 16'h0000;
        in_valid = 1;
        #4;
        in_valid = 0;
        
        wait(out_valid);
        @(posedge clk);
        
        begin
            i_norm = real'($signed(out_features[0])) / 32768.0;
            q_norm = real'($signed(out_features[1])) / 32768.0;
            amp = real'($signed(out_features[8])) / 32768.0;
            amp_cubed = real'($signed(out_features[12])) / 32768.0;
            
            $display("  I_norm = 0x%04x (%.4f), expected ≈1.0 (normalized unit vector)", out_features[0], i_norm);
            $display("  Q_norm = 0x%04x (%.4f), expected ≈0.0", out_features[1], q_norm);
            $display("  A(0)   = 0x%04x (%.4f), expected ≈0.5", out_features[8], amp);
            $display("  A³(0)  = 0x%04x (%.4f), expected ≈0.125\n", out_features[12], amp_cubed);
            
            if (i_norm > 0.95 && i_norm <= 1.0 && q_norm < 0.01) begin
                $display("  ✓ PASS: DC input normalized correctly\n");
            end
            else begin
                $display("  ⚠ FAIL: Normalization incorrect\n");
            end
        end
        
        #100;
        
        //======================================================================
        // TEST 4: Accuracy - 45° Signal
        //======================================================================
        
        $display("TEST 4: Accuracy - 45° Signal (I=Q=0.3535)\n");
        
        in_i = 16'h2d41;
        in_q = 16'h2d41;
        in_valid = 1;
        #4;
        in_valid = 0;
        
        wait(out_valid);
        @(posedge clk);
        
        begin
            i_norm = real'($signed(out_features[0])) / 32768.0;
            q_norm = real'($signed(out_features[1])) / 32768.0;
            amp = real'($signed(out_features[8])) / 32768.0;
            
            $display("  I_norm = 0x%04x (%.4f)", out_features[0], i_norm);
            $display("  Q_norm = 0x%04x (%.4f)", out_features[1], q_norm);
            $display("  A(0)   = 0x%04x (%.4f), expected ≈0.707\n", out_features[8], amp);
            
            if (amp > 0.65 && amp < 0.75) begin
                $display("  ✓ PASS: 45° amplitude correct\n");
            end
        end
        
        #100;
        
        //======================================================================
        // TEST 5: Memory Tapeline
        //======================================================================
        
        $display("TEST 5: Memory Tapeline Verification\n");
        $display("Sending continuous stream to fill pipeline...\n");
        
        // Fill pipeline with known sequence: 0, 0.125, 0.25, 0.375, 0.5, ...
        // After 23+ cycles, memory taps should show last 4 inputs
        in_valid = 1;
        in_i = 16'h0000; in_q = 0; #4;  // Sample 0
        in_i = 16'h1000; in_q = 0; #4;  // Sample 1
        in_i = 16'h2000; in_q = 0; #4;  // Sample 2
        in_i = 16'h3000; in_q = 0; #4;  // Sample 3
        in_i = 16'h4000; in_q = 0; #4;  // Sample 4
        in_i = 16'h5000; in_q = 0; #4;  // Sample 5
        in_i = 16'h6000; in_q = 0; #4;  // Sample 6
        in_i = 16'h7000; in_q = 0; #4;  // Sample 7
        in_valid = 0;
        
        // Wait for output corresponding to sample 7
        wait(out_valid);
        @(posedge clk);
        @(posedge clk);
        @(posedge clk);
        @(posedge clk);
        @(posedge clk);
        @(posedge clk);
        @(posedge clk);
        
        begin
            $display("  Output for sample 7 (I=0.875):");
            $display("  I(0)  = 0x%04x (%.4f), expected 0.875 (tap-0, current)", out_features[16], 
                     real'($signed(out_features[16])) / 32768.0);
            $display("  I(-1) = 0x%04x (%.4f), expected 0.75 (tap-1, -1 sample)", out_features[18],
                     real'($signed(out_features[18])) / 32768.0);
            $display("  I(-2) = 0x%04x (%.4f), expected 0.625 (tap-2, -2 samples)", out_features[20],
                     real'($signed(out_features[20])) / 32768.0);
            $display("  I(-3) = 0x%04x (%.4f), expected 0.5 (tap-3, -3 samples)\n", out_features[22],
                     real'($signed(out_features[22])) / 32768.0);
            
            // Check if tapeline shows consecutive sequence
            if (out_features[16] == 16'h7000 && 
                out_features[18] == 16'h6000 &&
                out_features[20] == 16'h5000 &&
                out_features[22] == 16'h4000) begin
                $display("  ✓ PASS: Memory tapeline correct\n");
            end
            else begin
                $display("  ⚠ FAIL: Memory tapeline shows wrong delay\n");
            end
        end
        
        #100;
        
        $display("================================================================================");
        $display("TEST COMPLETE");
        $display("================================================================================\n");
        
        $finish;
    end

endmodule
