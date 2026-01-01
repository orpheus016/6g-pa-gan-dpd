`timescale 1ns/1ps

module tb_tdnn_simple;
    parameter DATA_WIDTH = 16;
    parameter INPUT_DIM = 22;  // UPDATED: 2 + 2*5*2 = 22 features
    
    reg clk, rst_n;
    reg [DATA_WIDTH*INPUT_DIM-1:0] in_vector;
    reg in_valid;
    wire signed [DATA_WIDTH-1:0] out_i, out_q;
    wire out_valid, busy;
    
    wire [15:0] weight_addr;
    reg signed [15:0] weight_data;
    reg [1:0] weight_bank_sel;
    
    // Weight memory (1298 params per bank × 3 banks = 3894 total)
    reg signed [15:0] weight_mem [0:3999];
    
    initial begin
        integer i;
        // Initialize with test values (extended for 22 inputs)
        for (i = 0; i < 4000; i = i + 1) begin
            weight_mem[i] = 16'h1000;  // 0.125 in Q1.15 (4096/32768)
        end
        $display("Initialized %0d weights to 0x1000 (0.125 in Q1.15)", 4000);
    end
    
    always @(posedge clk) begin
        weight_data <= weight_mem[weight_addr];
    end
    
    // DUT
    tdnn_generator dut (
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
    
    // Clock
    initial begin
        clk = 0;
        forever #2.5 clk = ~clk;  // 200MHz
    end
    
    // Test
    integer i, timeout_cnt;
    
    initial begin
        $dumpfile("tb_tdnn_simple.vcd");
        $dumpvars(0, tb_tdnn_simple);
        
        rst_n = 0;
        in_valid = 0;
        in_vector = 0;
        weight_bank_sel = 0;
        
        #100;
        rst_n = 1;
        #50;
        
        // Test with simple input
        $display("=== Starting Simple TDNN Test (22 inputs) ===");
        $display("Input: Setting I=0.5, Q=0.25, rest=0.1");
        
        in_vector[0*16 +: 16] = 16'h4000;  // I = 0.5
        in_vector[1*16 +: 16] = 16'h2000;  // Q = 0.25
        
        for (i = 2; i < 22; i = i + 1) begin
            in_vector[i*16 +: 16] = 16'h0CCC;  // 0.1
        end
        
        @(posedge clk);
        in_valid <= 1;
        $display("[%0t] Input valid asserted", $time);
        
        @(posedge clk);
        in_valid <= 0;
        $display("[%0t] Input valid deasserted", $time);
        
        // Wait for out_valid with timeout
        timeout_cnt = 0;
        while (!out_valid && timeout_cnt < 20000) begin
            @(posedge clk);
            timeout_cnt = timeout_cnt + 1;
        end
        
        if (timeout_cnt >= 20000) begin
            $display("[%0t] ✗ TIMEOUT waiting for out_valid", $time);
            $display("  Current state: %0d", dut.state);
            $display("  out_idx: %0d, in_idx: %0d, mac_cnt: %0d", dut.out_idx, dut.in_idx, dut.mac_cnt);
            $display("  busy: %0d", busy);
            $finish;
        end
        
        // Capture outputs when out_valid is high
        $display("[%0t] ✓ out_valid asserted after %0d cycles", $time, timeout_cnt);
        $display("  out_i = %d (0x%h) = %f", out_i, out_i, $itor(out_i)/32768.0);
        $display("  out_q = %d (0x%h) = %f", out_q, out_q, $itor(out_q)/32768.0);
        
        // Capture outputs when out_valid is high
        $display("[%0t] ✓ out_valid asserted after %0d cycles", $time, timeout_cnt);
        $display("  out_i = %d (0x%h) = %f", out_i, out_i, $itor(out_i)/32768.0);
        $display("  out_q = %d (0x%h) = %f", out_q, out_q, $itor(out_q)/32768.0);
        
        #100;
        
        if (out_i == 0 && out_q == 0) begin
            $display("\n✗ FAIL: Outputs are zero (weights not being applied)");
            $display("   This usually means:");
            $display("   1. Weight memory reads are returning 0");
            $display("   2. MAC accumulation is not working");
            $display("   3. Pipeline timing is incorrect");
        end else begin
            $display("\n✓ PASS: Non-zero output detected!");
            $display("   TDNN inference is working correctly");
        end
        
        #200;
        $display("=== Test Complete ===");
        $finish;
    end
    
    // Monitor state changes with more detail
    always @(dut.state) begin
        case (dut.state)
            0: $display("[%0t] STATE: IDLE", $time);
            1: $display("[%0t] STATE: LOAD", $time);
            2: $display("[%0t] STATE: FC1 (will compute 32 neurons of 22 weights each = 704 total)", $time);
            3: $display("[%0t] STATE: ACT1 (out_idx=%0d, should be 32)", $time, dut.out_idx);
            4: $display("[%0t] STATE: FC2 (will compute 16 neurons of 32 weights each = 512 total)", $time);
            5: $display("[%0t] STATE: ACT2 (out_idx=%0d, should be 16)", $time, dut.out_idx);
            6: $display("[%0t] STATE: FC3 (will compute 2 neurons of 16 weights each = 32 total)", $time);
            7: $display("[%0t] STATE: TANH (out_idx=%0d, should be 2)", $time, dut.out_idx);
            8: $display("[%0t] STATE: OUTPUT", $time);
        endcase
    end
    
    // Monitor neuron completion in FC1 with accumulator values
    always @(posedge clk) begin
        if (dut.state == 2 && dut.in_idx == 21 && dut.out_idx < 3) begin  // FC1, last input of first 3 neurons (22-1=21)
            $display("[%0t] FC1: Completed neuron %0d - acc[0]=0x%08h (%0d decimal)", 
                     $time, dut.out_idx, dut.acc[0], $signed(dut.acc[0]));
        end
        if (dut.state == 2 && dut.in_idx == 0 && dut.out_idx < 3 && dut.mac_cnt > 0) begin
            $display("[%0t] FC1: Starting neuron %0d - acc[0]=0x%08h (should be 0)", 
                     $time, dut.out_idx, dut.acc[0]);
        end
    end
    
    // Monitor MAC operations for first neuron
    reg monitor_mac;
    initial monitor_mac = 1;
    
    always @(posedge clk) begin
        if (monitor_mac && dut.state == 2 && dut.out_idx == 0 && dut.in_idx < 5) begin
            $display("[%0t] MAC[neuron=0, input=%0d]: weight=0x%04h, input=0x%04h, product=0x%08h, acc=0x%08h",
                     $time, dut.in_idx, dut.mac_weight, dut.mac_input, 
                     dut.mac_product, dut.acc[0]);
        end
        if (dut.state == 2 && dut.out_idx == 1) begin
            monitor_mac = 0;  // Stop monitoring after first neuron
        end
    end
    
    // Monitor weight reads (limit to first few to avoid spam)
    reg [15:0] last_weight_addr;
    integer weight_read_count;
    
    initial begin
        weight_read_count = 0;
        last_weight_addr = 16'hFFFF;
    end
    
    always @(posedge clk) begin
        if (weight_addr != last_weight_addr && weight_read_count < 30) begin
            $display("[%0t] Weight[%4d] = 0x%04h (busy=%0d, state=%0d)", 
                     $time, weight_addr, weight_data, busy, dut.state);
            last_weight_addr = weight_addr;
            weight_read_count = weight_read_count + 1;
        end
    end
    
    // Timeout
    initial begin
        #100000;
        $display("✗ TIMEOUT");
        $finish;
    end
    
endmodule
