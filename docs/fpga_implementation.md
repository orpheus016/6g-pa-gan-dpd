# FPGA Implementation Guide

## Overview

This guide covers the FPGA implementation of the 6G PA DPD system for both proof-of-concept (PYNQ-Z1) and production (ZCU104) platforms.

## Target Platforms

### PYNQ-Z1 (Proof of Concept)
- **FPGA:** Zynq-7020 (XC7Z020-1CLG400C)
- **Resources:** 53K LUTs, 220 DSP48s, 4.9 Mb BRAM
- **Use Case:** Algorithm validation, demo, development

### ZCU104 (Production)
- **FPGA:** Zynq UltraScale+ (XCZU7EV-2FFVC1156)
- **Resources:** 230K LUTs, 1,728 DSPs, 11 Mb BRAM, 27 Mb URAM
- **Use Case:** Real 6G deployment with high-speed ADC/DAC

## Build Flow

### Prerequisites

```bash
# Vivado 2023.2 or later
source /tools/Xilinx/Vivado/2023.2/settings64.sh

# Python dependencies
pip install pynq numpy pyyaml
```

### RTL Simulation

```bash
cd rtl

# Run all testbenches (Icarus Verilog)
make sim_all

# Run specific testbench
make sim_dpd_top
make sim_tdnn
make sim_aspsa

# View waveforms
make wave_dpd

# Lint check (Verilator)
make lint
```

### Vivado Build (PYNQ-Z1)

```bash
# Create Vivado project
cd rtl
vivado -mode batch -source scripts/create_project_pynq.tcl

# Or using Makefile
make vivado_pynq
```

### Vivado Build (ZCU104)

```bash
cd rtl
vivado -mode batch -source scripts/create_project_zcu104.tcl

# Or using Makefile
make vivado_zcu104
```

## RTL Module Hierarchy

```
dpd_top
├── input_buffer          # Memory tap assembly
├── tdnn_generator        # Neural network inference
│   ├── fc_layer (x3)     # Fully connected layers
│   └── activation        # LeakyReLU/Tanh
├── interpolator          # 2x upsampler
├── shadow_memory         # CDC weight buffer
├── temp_controller       # Temperature state machine
├── aspsa_engine          # Online adaptation
└── error_metric          # NMSE calculation
```

## Clocking Architecture

```
                    ┌─────────────────────────────────────┐
                    │              MMCM                    │
   clk_125 ────────►│  ┌─────────────────────────────┐   │
   (Input)          │  │ VCO = 1000 MHz              │   │
                    │  │                              │   │
                    │  │  ÷5  → clk_200 (200 MHz)    │───► NN Domain
                    │  │  ÷2.5→ clk_400 (400 MHz)    │───► PA Output
                    │  │  ÷1000→clk_1   (1 MHz)      │───► Adaptation
                    │  └─────────────────────────────┘   │
                    └─────────────────────────────────────┘
```

### Clock Domain Crossing

**200MHz ↔ 1MHz CDC:**
- Double-buffer shadow memory
- Gray-coded pointers
- 2-stage synchronizers on control signals

```verilog
// Gray code synchronizer
always @(posedge clk_dest or negedge rst_n) begin
    if (!rst_n) begin
        sync_stage1 <= 0;
        sync_stage2 <= 0;
    end else begin
        sync_stage1 <= src_gray;
        sync_stage2 <= sync_stage1;
    end
end
```

## Weight Memory Architecture

### Bank Organization

```
           ┌──────────────────────────────────────────────┐
           │              Weight Memory (BRAM)             │
           │                                               │
           │  ┌─────────────────────────────────────────┐ │
           │  │    Bank 0: COLD (-40°C to 10°C)         │ │
           │  │    1,170 × 16-bit = 2,340 bytes         │ │
           │  └─────────────────────────────────────────┘ │
           │  ┌─────────────────────────────────────────┐ │
           │  │    Bank 1: NORMAL (10°C to 45°C)        │ │
           │  │    1,170 × 16-bit = 2,340 bytes         │ │
           │  └─────────────────────────────────────────┘ │
           │  ┌─────────────────────────────────────────┐ │
           │  │    Bank 2: HOT (45°C to 85°C)           │ │
           │  │    1,170 × 16-bit = 2,340 bytes         │ │
           │  └─────────────────────────────────────────┘ │
           │                                               │
           │  Total: 7,020 bytes → 4 BRAM18K              │
           └──────────────────────────────────────────────┘
```

### Weight Loading

```python
# Export weights from trained model
python export.py --checkpoint checkpoints/best.pt \
                 --output rtl/weights \
                 --temperature-banks

# Generated files:
# rtl/weights/cold/weights.hex
# rtl/weights/normal/weights.hex
# rtl/weights/hot/weights.hex
```

### BRAM Initialization

```verilog
// In tdnn_generator.v
(* ram_style = "block" *)
reg signed [15:0] weight_mem [0:1169];

initial begin
    $readmemh("weights/normal/weights.hex", weight_mem);
end

// Runtime bank switching
always @(posedge clk) begin
    case (temp_state)
        COLD:   weight_data <= weight_mem_cold[weight_addr];
        NORMAL: weight_data <= weight_mem_normal[weight_addr];
        HOT:    weight_data <= weight_mem_hot[weight_addr];
    endcase
end
```

## MAC Pipeline Design

### Single MAC Unit (Area Optimized)

```
                    ┌─────────────────────────────────────────┐
                    │           MAC Pipeline                   │
                    │                                          │
   weight ─────────►│  ┌───────┐   ┌───────┐   ┌───────┐     │
                    │  │ MUL   │──►│ ADD   │──►│ ACC   │─────│──► result
   activation ─────►│  │16×16  │   │ 32+32 │   │ 32-bit│     │
                    │  │=32-bit│   │       │   │       │     │
                    │  └───────┘   └───────┘   └───────┘     │
                    │                                          │
                    │  Stages: 1      2          3             │
                    └─────────────────────────────────────────┘

Throughput: 1 MAC per cycle
Latency: 3 cycles
```

### DSP48E2 Mapping (UltraScale+)

```
DSP48E2 Configuration:
├── A input: 30-bit (weight, sign-extended)
├── B input: 18-bit (activation)
├── P output: 48-bit (accumulated result)
├── OPMODE: Multiply-accumulate (MACC)
└── Pipeline: A1, B1, M, P registers enabled
```

## Activation Functions

### LeakyReLU (α = 0.25)

```verilog
// Efficient shift-based implementation
wire signed [DATA_WIDTH-1:0] leaky_result;
assign leaky_result = (in_data >= 0) ? in_data : (in_data >>> 2);
```

**Timing:** Combinational (0 cycles)

### Tanh (LUT-based)

```verilog
// 256-entry LUT with linear interpolation
reg signed [15:0] tanh_lut [0:255];

// Address mapping: x ∈ [-4, 4) → addr ∈ [0, 255]
wire [7:0] lut_addr = (in_data + 16'h0400) >> 5;
wire [4:0] frac = in_data[4:0];

// Linear interpolation
wire signed [15:0] y0 = tanh_lut[lut_addr];
wire signed [15:0] y1 = tanh_lut[lut_addr + 1];
wire signed [15:0] result = y0 + ((y1 - y0) * frac) >>> 5;
```

**Timing:** 2 cycles (LUT read + interpolation)

## A-SPSA Implementation

### LFSR Perturbation Generator

```verilog
// 32-bit Galois LFSR
// Polynomial: x^32 + x^22 + x^2 + x^1 + 1
// Taps: 32, 22, 2, 1

reg [31:0] lfsr;
wire feedback = lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0];

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        lfsr <= SEED;
    else if (shift_en)
        lfsr <= {lfsr[30:0], feedback};
end

// Bernoulli output: each bit is ±1
wire [31:0] bernoulli = lfsr;  // 1 = +1, 0 = -1
```

### Shift-Based Annealing

```verilog
// Learning rate decay via right shift
reg [15:0] learning_rate;
reg [3:0] anneal_shift;

always @(posedge clk) begin
    if (iteration_count[9:0] == 0) begin  // Every 1024 iterations
        anneal_shift <= (anneal_shift < 8) ? anneal_shift + 1 : 8;
    end
end

// Effective LR = base_lr >> anneal_shift
wire [15:0] effective_lr = base_lr >> anneal_shift;
```

### Gradient Computation

```verilog
// SPSA gradient estimate
// g = (L+ - L-) / (2c) * delta^(-1)
// Since delta ∈ {-1, +1}, delta^(-1) = delta

reg signed [31:0] gradient;

always @(posedge clk) begin
    if (grad_compute_en) begin
        // Scale by perturbation size (shift for division)
        gradient <= (loss_plus - loss_minus) >> (perturb_shift + 1);
        
        // Apply Bernoulli sign
        if (bernoulli_bit)
            gradient <= gradient;  // +1 case
        else
            gradient <= -gradient; // -1 case
    end
end
```

## Temperature Controller

### State Encoding

```verilog
localparam TEMP_COLD   = 2'b00;
localparam TEMP_NORMAL = 2'b01;
localparam TEMP_HOT    = 2'b10;
```

### Moving Average Filter

```verilog
// 16-sample moving average
reg [11:0] temp_history [0:15];
reg [15:0] temp_sum;
reg [3:0] hist_ptr;

always @(posedge clk) begin
    temp_sum <= temp_sum - temp_history[hist_ptr] + temp_adc;
    temp_history[hist_ptr] <= temp_adc;
    hist_ptr <= hist_ptr + 1;
end

wire [11:0] temp_avg = temp_sum >> 4;
```

## Interpolator Design

### Half-Band Filter Coefficients

```python
# Design script (scipy.signal)
from scipy import signal
import numpy as np

# Half-band filter design
N = 23  # Number of taps (odd)
h = signal.firwin(N, 0.5)  # Cutoff at Nyquist/2

# Quantize to Q1.15
h_fixed = np.round(h * 32768).astype(np.int16)
```

### Polyphase Implementation

```verilog
// Phase 0: Pass-through (center tap = 0.5)
// Phase 1: Interpolated sample

always @(posedge clk_200) begin
    // Symmetric filter: h[n] = h[N-1-n]
    // Compute only half the taps
    phase1_acc <= 
        coef[0]  * (delay[0]  + delay[22]) +
        coef[2]  * (delay[2]  + delay[20]) +
        coef[4]  * (delay[4]  + delay[18]) +
        coef[6]  * (delay[6]  + delay[16]) +
        coef[8]  * (delay[8]  + delay[14]) +
        coef[10] * (delay[10] + delay[12]);
end
```

## Debug and Monitoring

### ILA Integration

```verilog
// Add ILA probes for debug
(* mark_debug = "true" *) wire [15:0] debug_dpd_out_i;
(* mark_debug = "true" *) wire [15:0] debug_dpd_out_q;
(* mark_debug = "true" *) wire [15:0] debug_error_metric;
(* mark_debug = "true" *) wire [1:0]  debug_temp_state;
```

### VIO Control

```verilog
// Virtual I/O for runtime control
vio_0 vio_inst (
    .clk(clk_200),
    .probe_in0(error_metric),
    .probe_in1(temp_state),
    .probe_in2(adapt_active),
    .probe_out0(dpd_enable_override),
    .probe_out1(adapt_enable_override),
    .probe_out2(temp_override)
);
```

## Performance Verification

### Checklist

- [ ] Timing closure at 200 MHz
- [ ] CDC paths properly constrained
- [ ] Weight memory initialized correctly
- [ ] Temperature transitions verified
- [ ] A-SPSA convergence tested
- [ ] EVM < -25 dB achieved
- [ ] ACPR < -45 dBc achieved

### Post-Implementation Timing Report

```tcl
# Check timing in Vivado
report_timing_summary -delay_type min_max -report_unconstrained \
    -check_timing_verbose -max_paths 10 -input_pins -routable_nets \
    -name timing_1
```

Expected results:
- WNS (Worst Negative Slack): > 0.5 ns
- WHS (Worst Hold Slack): > 0.1 ns
- No timing violations in CDC paths

## Deployment

### PYNQ Deployment

```python
# Python overlay loading
from pynq import Overlay, allocate
import numpy as np

# Load bitstream
ol = Overlay('dpd_system.bit')

# Access DPD IP
dpd = ol.dpd_top_0

# Configure
dpd.register_map.CTRL = 0x03  # Enable DPD + Adapt
dpd.register_map.TEMP_CFG = 0x0A28  # 10°C/40°C thresholds

# Allocate DMA buffers
in_buffer = allocate(shape=(1024,), dtype=np.int16)
out_buffer = allocate(shape=(1024,), dtype=np.int16)

# Run DPD
in_buffer[:] = input_signal
ol.dma_0.sendchannel.transfer(in_buffer)
ol.dma_0.recvchannel.transfer(out_buffer)
ol.dma_0.sendchannel.wait()
ol.dma_0.recvchannel.wait()
```

### ZCU104 Deployment

```tcl
# Petalinux device tree overlay
/ {
    fragment@0 {
        target = <&fpga_full>;
        __overlay__ {
            dpd_system: dpd@a0000000 {
                compatible = "xlnx,dpd-system-1.0";
                reg = <0x0 0xa0000000 0x0 0x10000>;
                interrupts = <0 89 4>;
                clocks = <&zynqmp_clk 71>;
            };
        };
    };
};
```
