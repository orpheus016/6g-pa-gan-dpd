`timescale 1ns / 10ps
`include "interpolator1_5.v"

module tb_interpolator1_5;

    //==========================================================================
    // Parameters
    //==========================================================================
    parameter DATA_WIDTH     = 16;
    parameter CLK_200_PERIOD = 5.0;  // 200 MHz (5ns)
    parameter CLK_1K_PERIOD  = 1.0;  // 1 GHz (1ns)

    //==========================================================================
    // Signal Declarations
    //==========================================================================
    reg                       clk_200;
    reg                       clk_1k;
    reg                       rst_n;

    // Inputs (Q1.15)
    reg  signed [DATA_WIDTH-1:0] in_i;
    reg  signed [DATA_WIDTH-1:0] in_q;
    reg                       in_valid;

    // Outputs (Q1.15)
    wire signed [DATA_WIDTH-1:0] out_i;
    wire signed [DATA_WIDTH-1:0] out_q;
    wire                      out_valid;

    // Simulation Variables
    integer i;
    real    freq_sig   = 10.0e6;       // 10 MHz Sine Wave
    real    fs         = 200.0e6;      // 200 MHz Input Sample Rate
    real    amplitude  = 15000.0;      // Amplitude (Safe range for Q1.15)
    real    pi         = 3.14159265359;
    real    val_i, val_q;

    //==========================================================================
    // Clock Generation
    //==========================================================================
    // 200 MHz Clock
    initial begin
        clk_200 = 0;
        forever #(CLK_200_PERIOD / 2.0) clk_200 = ~clk_200;
    end

    // 1 GHz Clock (Phase aligned for easier viewing, though not required)
    initial begin
        clk_1k = 0;
        forever #(CLK_1K_PERIOD / 2.0) clk_1k = ~clk_1k;
    end

    //==========================================================================
    // DUT Instantiation
    //==========================================================================
    interpolator1_5 #(
        .DATA_WIDTH(DATA_WIDTH)
    ) uut (
        .clk_200    (clk_200),
        .clk_1k     (clk_1k),
        .rst_n      (rst_n),
        .in_i       (in_i),
        .in_q       (in_q),
        .in_valid   (in_valid),
        .out_i      (out_i),
        .out_q      (out_q),
        .out_valid  (out_valid)
    );

    //==========================================================================
    // Main Stimulus
    //==========================================================================
    initial begin
        // 1. Initialize
        rst_n    = 0;
        in_i     = 0;
        in_q     = 0;
        in_valid = 0;

        // Setup Waveform Dump
        $dumpfile("tb_interpolator1_5.vcd");
        $dumpvars(0, tb_interpolator1_5);

        // 2. Reset Sequence
        #(CLK_200_PERIOD * 10);
        rst_n = 1;
        #(CLK_200_PERIOD * 5);

        $display("Starting Sine Wave Input @ 200Msps...");

        // 3. Generate 50 Input Samples (Creates 250 Output Samples)
        for (i = 0; i < 50; i = i + 1) begin
            @(negedge clk_200); // Drive on negedge to ensure setup time
            
            // Calculate float values
            val_i = amplitude * $cos(2.0 * pi * freq_sig * i / fs);
            val_q = amplitude * $sin(2.0 * pi * freq_sig * i / fs);

            // Implicit cast to Q1.15 integer
            in_i     = val_i;
            in_q     = val_q;
            in_valid = 1;
        end

        // 4. Stop Inputs
        @(negedge clk_200);
        in_valid = 0;
        in_i     = 0;
        in_q     = 0;

        // 5. Wait for pipeline flush (Filter Delay + Buffer)
        #(CLK_200_PERIOD * 20);
        
        $display("Simulation Finished.");
        $finish;
    end

endmodule