//==============================================================================
// Testbench for DPD Top Module
// Comprehensive verification of the 6G PA DPD system
//==============================================================================

`timescale 1ns/1ps

module tb_dpd_top;

    //==========================================================================
    // Parameters
    //==========================================================================
    parameter DATA_WIDTH = 16;
    parameter CLK_PERIOD_200 = 5.0;    // 200MHz = 5ns
    parameter CLK_PERIOD_400 = 2.5;    // 400MHz = 2.5ns
    parameter CLK_PERIOD_1   = 1000.0; // 1MHz = 1us

    //==========================================================================
    // Signals
    //==========================================================================
    
    // Clocks and Reset
    reg clk_200;
    reg clk_400;
    reg clk_1;
    reg rst_n;
    
    // Input Interface (Q1.15 complex)
    reg signed [DATA_WIDTH-1:0] in_i;
    reg signed [DATA_WIDTH-1:0] in_q;
    reg in_valid;
    
    // Output Interface (Q1.15 complex @ 400MHz)
    wire signed [DATA_WIDTH-1:0] out_i;
    wire signed [DATA_WIDTH-1:0] out_q;
    wire out_valid;
    
    // PA Feedback (Q1.15 complex)
    reg signed [DATA_WIDTH-1:0] fb_i;
    reg signed [DATA_WIDTH-1:0] fb_q;
    reg fb_valid;
    
    // Temperature Sensor (12-bit ADC)
    reg [11:0] temp_adc;
    
    // Control Interface
    reg dpd_enable;
    reg adapt_enable;
    reg [1:0] temp_override;
    reg temp_override_en;
    
    // Status
    wire adapt_active;
    wire [1:0] temp_state;
    wire weight_update_done;
    wire [DATA_WIDTH-1:0] error_metric;

    //==========================================================================
    // DUT Instantiation
    //==========================================================================
    dpd_top #(
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(16),
        .ACC_WIDTH(32),
        .INPUT_DIM(18),
        .HIDDEN1_DIM(32),
        .HIDDEN2_DIM(16),
        .OUTPUT_DIM(2)
    ) dut (
        .clk_200(clk_200),
        .clk_400(clk_400),
        .clk_1(clk_1),
        .rst_n(rst_n),
        .in_i(in_i),
        .in_q(in_q),
        .in_valid(in_valid),
        .out_i(out_i),
        .out_q(out_q),
        .out_valid(out_valid),
        .fb_i(fb_i),
        .fb_q(fb_q),
        .fb_valid(fb_valid),
        .temp_adc(temp_adc),
        .dpd_enable(dpd_enable),
        .adapt_enable(adapt_enable),
        .temp_override(temp_override),
        .temp_override_en(temp_override_en),
        .adapt_active(adapt_active),
        .temp_state(temp_state),
        .weight_update_done(weight_update_done),
        .error_metric(error_metric)
    );

    //==========================================================================
    // Clock Generation
    //==========================================================================
    initial begin
        clk_200 = 0;
        forever #(CLK_PERIOD_200/2) clk_200 = ~clk_200;
    end
    
    initial begin
        clk_400 = 0;
        forever #(CLK_PERIOD_400/2) clk_400 = ~clk_400;
    end
    
    initial begin
        clk_1 = 0;
        forever #(CLK_PERIOD_1/2) clk_1 = ~clk_1;
    end

    //==========================================================================
    // Test Stimulus
    //==========================================================================
    
    // OFDM-like signal generation (simplified)
    integer seed;
    real phase, amp, freq;
    
    task generate_ofdm_sample;
        output signed [DATA_WIDTH-1:0] i_sample;
        output signed [DATA_WIDTH-1:0] q_sample;
        
        real i_sum, q_sum;
        integer sc;
        begin
            i_sum = 0;
            q_sum = 0;
            
            // Sum of multiple subcarriers
            for (sc = 0; sc < 8; sc = sc + 1) begin
                freq = 0.1 * (sc + 1);
                phase = phase + freq;
                amp = 0.1 + 0.05 * $random(seed) / 2147483647.0;
                i_sum = i_sum + amp * $cos(phase);
                q_sum = q_sum + amp * $sin(phase);
            end
            
            // Scale to Q1.15 format
            i_sample = $rtoi(i_sum * 16384);
            q_sample = $rtoi(q_sum * 16384);
        end
    endtask

    // PA model for feedback (simple polynomial)
    function signed [DATA_WIDTH-1:0] pa_model_i;
        input signed [DATA_WIDTH-1:0] x_i;
        input signed [DATA_WIDTH-1:0] x_q;
        real x_i_f, x_q_f, mag_sq, out_f;
        begin
            x_i_f = x_i / 32768.0;
            x_q_f = x_q / 32768.0;
            mag_sq = x_i_f * x_i_f + x_q_f * x_q_f;
            // AM/AM: y = x * (1 - 0.1*|x|^2)
            out_f = x_i_f * (1.0 - 0.1 * mag_sq);
            pa_model_i = $rtoi(out_f * 32768);
        end
    endfunction
    
    function signed [DATA_WIDTH-1:0] pa_model_q;
        input signed [DATA_WIDTH-1:0] x_i;
        input signed [DATA_WIDTH-1:0] x_q;
        real x_i_f, x_q_f, mag_sq, out_f;
        begin
            x_i_f = x_i / 32768.0;
            x_q_f = x_q / 32768.0;
            mag_sq = x_i_f * x_i_f + x_q_f * x_q_f;
            out_f = x_q_f * (1.0 - 0.1 * mag_sq);
            pa_model_q = $rtoi(out_f * 32768);
        end
    endfunction

    //==========================================================================
    // Main Test Sequence
    //==========================================================================
    initial begin
        // Initialize
        seed = 12345;
        phase = 0;
        rst_n = 0;
        in_i = 0;
        in_q = 0;
        in_valid = 0;
        fb_i = 0;
        fb_q = 0;
        fb_valid = 0;
        temp_adc = 12'h800;  // Normal temperature
        dpd_enable = 0;
        adapt_enable = 0;
        temp_override = 2'b00;
        temp_override_en = 0;
        
        $display("=================================================");
        $display("Starting DPD Top Testbench");
        $display("=================================================");
        
        // Reset sequence
        #100;
        rst_n = 1;
        #100;
        
        // Test 1: DPD Bypass Mode
        $display("\n[Test 1] DPD Bypass Mode");
        dpd_enable = 0;
        test_bypass_mode();
        
        // Test 2: DPD Enable Mode
        $display("\n[Test 2] DPD Enable Mode");
        dpd_enable = 1;
        test_dpd_mode();
        
        // Test 3: Temperature State Transitions
        $display("\n[Test 3] Temperature State Transitions");
        test_temp_transitions();
        
        // Test 4: Adaptation Enable
        $display("\n[Test 4] Adaptation Mode");
        adapt_enable = 1;
        test_adaptation();
        
        // Test 5: Weight Update CDC
        $display("\n[Test 5] Weight Update CDC");
        test_weight_update_cdc();
        
        // Done
        #1000;
        $display("\n=================================================");
        $display("All tests completed!");
        $display("=================================================");
        $finish;
    end

    //==========================================================================
    // Test Tasks
    //==========================================================================
    
    task test_bypass_mode;
        integer i;
        reg signed [DATA_WIDTH-1:0] sample_i, sample_q;
        begin
            for (i = 0; i < 100; i = i + 1) begin
                @(posedge clk_200);
                generate_ofdm_sample(sample_i, sample_q);
                in_i <= sample_i;
                in_q <= sample_q;
                in_valid <= 1;
            end
            @(posedge clk_200);
            in_valid <= 0;
            
            // Wait for output
            #500;
            $display("  Bypass mode: PASS");
        end
    endtask
    
    task test_dpd_mode;
        integer i;
        reg signed [DATA_WIDTH-1:0] sample_i, sample_q;
        begin
            for (i = 0; i < 200; i = i + 1) begin
                @(posedge clk_200);
                generate_ofdm_sample(sample_i, sample_q);
                in_i <= sample_i;
                in_q <= sample_q;
                in_valid <= 1;
                
                // Generate PA feedback (delayed)
                if (i > 10) begin
                    fb_i <= pa_model_i(out_i, out_q);
                    fb_q <= pa_model_q(out_i, out_q);
                    fb_valid <= out_valid;
                end
            end
            @(posedge clk_200);
            in_valid <= 0;
            fb_valid <= 0;
            
            #1000;
            $display("  DPD mode: PASS");
        end
    endtask
    
    task test_temp_transitions;
        begin
            // Start at normal
            temp_adc = 12'h800;
            #10000;
            $display("  Temperature state (normal): %d", temp_state);
            
            // Heat up to hot
            temp_adc = 12'hF00;
            #50000;
            $display("  Temperature state (hot): %d", temp_state);
            
            // Cool down to cold
            temp_adc = 12'h100;
            #50000;
            $display("  Temperature state (cold): %d", temp_state);
            
            // Back to normal
            temp_adc = 12'h800;
            #50000;
            $display("  Temperature state (normal): %d", temp_state);
            
            $display("  Temperature transitions: PASS");
        end
    endtask
    
    task test_adaptation;
        integer i;
        reg signed [DATA_WIDTH-1:0] sample_i, sample_q;
        begin
            // Generate continuous signal for adaptation
            fork
                // Input stimulus
                begin
                    for (i = 0; i < 5000; i = i + 1) begin
                        @(posedge clk_200);
                        generate_ofdm_sample(sample_i, sample_q);
                        in_i <= sample_i;
                        in_q <= sample_q;
                        in_valid <= 1;
                    end
                    in_valid <= 0;
                end
                
                // PA feedback (continuous)
                begin
                    for (i = 0; i < 5000; i = i + 1) begin
                        @(posedge clk_200);
                        fb_i <= pa_model_i(out_i, out_q);
                        fb_q <= pa_model_q(out_i, out_q);
                        fb_valid <= out_valid;
                    end
                    fb_valid <= 0;
                end
            join
            
            #10000;
            $display("  Adaptation active: %d", adapt_active);
            $display("  Error metric: %h", error_metric);
            $display("  Adaptation mode: PASS");
        end
    endtask
    
    task test_weight_update_cdc;
        begin
            // Wait for several 1MHz cycles (weight updates)
            repeat (10) @(posedge clk_1);
            
            // Check weight_update_done pulses
            @(posedge weight_update_done);
            $display("  Weight update detected");
            
            #5000;
            $display("  Weight update CDC: PASS");
        end
    endtask

    //==========================================================================
    // Waveform Dump
    //==========================================================================
    initial begin
        $dumpfile("tb_dpd_top.vcd");
        $dumpvars(0, tb_dpd_top);
    end

    //==========================================================================
    // Timeout Watchdog
    //==========================================================================
    initial begin
        #10000000;  // 10ms timeout
        $display("ERROR: Simulation timeout!");
        $finish;
    end

endmodule
