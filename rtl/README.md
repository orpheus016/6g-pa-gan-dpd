# RTL Implementation for 6G PA GAN-DPD

This directory contains the Verilog RTL implementation for FPGA deployment.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            dpd_top.v                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                   200MHz Clock Domain (clk_nn)                        │  │
│  │  ┌─────────────┐    ┌─────────────────┐    ┌──────────────┐           │  │
│  │  │ input_      │    │ tdnn_           │    │ interpolator │           │  │
│  │  │ buffer.v    │───►│ generator.v     │───►│ .v           │──► DAC    │  │
│  │  │             │    │                 │    │ 2× upsample  │           │  │
│  │  └─────────────┘    │ ┌─────────────┐ │    └──────────────┘           │  │
│  │                     │ │ fc_layer.v  │ │                               │  │
│  │                     │ │ (×3 layers) │ │                               │  │
│  │                     │ └─────────────┘ │                               │  │
│  │                     │ ┌─────────────┐ │                               │  │
│  │                     │ │ activation.v│ │                               │  │
│  │                     │ │ LReLU, Tanh │ │                               │  │
│  │                     │ └─────────────┘ │                               │  │
│  │                     │ ┌─────────────┐ │                               │  │
│  │                     │ │ weight_     │ │                               │  │
│  │                     │ │ bram.v      │◄├───── Shadow Memory            │  │
│  │                     │ └─────────────┘ │                               │  │
│  │                     └─────────────────┘                               │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │ CDC                                          │
│  ┌───────────────────────────▼───────────────────────────────────────────┐  │
│  │                   1MHz Clock Domain (clk_spsa)                        │  │
│  │  ┌─────────────┐    ┌─────────────────┐    ┌──────────────┐           │  │
│  │  │ error_      │    │ aspsa_          │    │ shadow_      │           │  │
│  │  │ metric.v    │───►│ engine.v        │───►│ memory.v     │           │  │
│  │  │ EVM calc    │    │ Gradient est.   │    │ CDC write    │           │  │
│  │  └─────────────┘    │ Weight update   │    └──────────────┘           │  │
│  │                     └─────────────────┘                               │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │ temp_controller.v - Temperature state machine                   │  │  │
│  │  │ Cold/Normal/Hot detection → Bank select + Anneal reset          │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ pa_digital_twin.v - PA model for simulation (optional in hardware)    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## File Structure

```
rtl/
├── README.md
├── Makefile
├── src/
│   ├── dpd_top.v              # Top-level module
│   ├── tdnn_generator.v       # TDNN inference engine
│   ├── fc_layer.v             # Fully-connected layer
│   ├── activation.v           # LeakyReLU, Tanh LUT
│   ├── weight_bram.v          # Weight storage (3 banks)
│   ├── input_buffer.v         # Input buffering + memory taps
│   ├── aspsa_engine.v         # A-SPSA update logic
│   ├── shadow_memory.v        # CDC weight transfer
│   ├── temp_controller.v      # Temperature state machine
│   ├── interpolator.v         # 2× polyphase upsampler
│   ├── error_metric.v         # EVM/error calculation
│   ├── pa_digital_twin.v      # Volterra PA (simulation)
│   └── cdc_sync.v             # CDC synchronization primitives
├── tb/
│   ├── tb_dpd_top.v           # Top-level testbench
│   ├── tb_tdnn_generator.v    # Generator testbench
│   └── tb_aspsa_engine.v      # A-SPSA testbench
├── constraints/
│   ├── pynq_z1.xdc            # PYNQ-Z1 constraints
│   └── zcu104.xdc             # ZCU104 constraints
└── weights/
    ├── weights_cold.bin       # Cold temperature weights
    ├── weights_normal.bin     # Normal temperature weights
    └── weights_hot.bin        # Hot temperature weights
```

## Quantization Formats

| Signal Type | Format | Bits | Range |
|-------------|--------|------|-------|
| Input IQ | Q1.15 | 16 | [-1.0, +0.99997] |
| Weights | Q1.15 | 16 | [-1.0, +0.99997] |
| Activations | Q8.8 | 16 | [-128.0, +127.996] |
| Accumulator | Q16.16 | 32 | [-32768.0, +32767.99998] |
| Output IQ | Q1.15 | 16 | [-1.0, +0.99997] |
| Learning Rate | Q0.16 | 16 | [0, 0.99998] |

## Building

### Simulation (Icarus Verilog)
```bash
make sim_all          # Run all testbenches
make sim_generator    # Run generator testbench only
make sim_aspsa        # Run A-SPSA testbench only
```

### Synthesis (Vivado)
```bash
make synth_pynq       # Synthesize for PYNQ-Z1
make synth_zcu104     # Synthesize for ZCU104
```

## Resource Estimates

### PYNQ-Z1 (xc7z020clg400-1)
| Resource | Used | Available | % |
|----------|------|-----------|---|
| LUT | ~3,500 | 53,200 | 6.6% |
| FF | ~2,200 | 106,400 | 2.1% |
| BRAM | 7 | 140 | 5.0% |
| DSP48 | 14 | 220 | 6.4% |

### ZCU104 (xczu7ev-ffvc1156-2-e)
| Resource | Used | Available | % |
|----------|------|-----------|---|
| LUT | ~3,500 | 230,400 | 1.5% |
| FF | ~2,200 | 460,800 | 0.5% |
| BRAM | 7 | 312 | 2.2% |
| DSP48 | 14 | 1,728 | 0.8% |

## Clock Domains

1. **clk_nn (200 MHz)**: NN inference, input/output processing
2. **clk_spsa (1 MHz)**: A-SPSA weight updates
3. **clk_dac (400 MHz)**: Output interpolation (optional, can use clk_nn with 2× logic)

## CDC (Clock Domain Crossing)

Weight updates from A-SPSA (1 MHz) to TDNN (200 MHz) use:
- Double-buffer shadow memory
- Gray-coded addresses
- 2-stage synchronizers for handshake signals
- Atomic buffer swap during vertical blanking

## Temperature Control

Temperature state transitions:
- **COLD** (< 15°C): Bank 0, reset annealing
- **NORMAL** (15-40°C): Bank 1
- **HOT** (> 40°C): Bank 2, reset annealing

State changes trigger:
1. Weight bank switch (atomic via shadow memory)
2. A-SPSA annealing reset (lr ← lr_initial)
