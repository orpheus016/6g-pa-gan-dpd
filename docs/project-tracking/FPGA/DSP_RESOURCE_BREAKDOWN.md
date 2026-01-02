# DSP48 Resource Breakdown

## Total DSP48 Count: 10

### 1. Nonlinear Feature Computation (2 DSP blocks)
**Location:** `input_buffer.v`

```verilog
// DSP #1: Envelope squared computation
envelope_sq = I² + Q²  // Uses 1 DSP48 for complex magnitude squared
// DSP #2: Envelope 4th power computation  
envelope_4th = envelope_sq²  // Uses 1 DSP48 for squaring
```

**Why only 2 DSPs for 6 envelope computations?**
- We compute envelope², envelope⁴ for current sample only
- Historical taps (n-1 through n-5) reuse buffered values from previous cycles
- Shift registers store pre-computed `env_sq_buffer[0..5]` and `env_4th_buffer[0..5]`

### 2. MAC Operations for TDNN Layers (6 DSP blocks)
**Location:** `tdnn_generator.v`

```verilog
// Parallel MAC architecture with 6 units
genvar g;
for (g = 0; g < NUM_MACS; g = g + 1) begin : mac_array
    // Each DSP computes: acc += weight * input
    mac_result[g] = weight_operand * input_operand + accumulator
end
```

**Layer processing:**
- FC1: 30×32 = 960 multiply-accumulates (160 cycles with 6 MACs)
- FC2: 32×16 = 512 MACs (86 cycles)
- FC3: 16×2 = 32 MACs (6 cycles)

### 3. Interpolation (2 DSP blocks)
**Location:** `interpolator.v` (upsampling from 200 MHz to RF rate)

```verilog
// DSP #9: I-channel interpolation
out_i = coeff_0 * sample_0 + coeff_1 * sample_1 + ...

// DSP #10: Q-channel interpolation  
out_q = coeff_0 * sample_0 + coeff_1 * sample_1 + ...
```

## Summary

| Module | DSP Count | Purpose |
|--------|-----------|---------|
| `input_buffer.v` | 2 | Envelope² and Envelope⁴ computation |
| `tdnn_generator.v` | 6 | Parallel MAC units for FC layers |
| `interpolator.v` | 2 | I/Q channel interpolation |
| **TOTAL** | **10** | |

## Optimization Notes

1. **Feature Reuse:** By buffering nonlinear features, we avoid recomputing them for each memory tap
2. **MAC Parallelism:** 6 parallel MACs balance throughput (3.3 Msps) with resource usage
3. **Pipeline Efficiency:** Total latency ~60 cycles includes FC1 (160/6≈27), FC2 (86/6≈14), FC3 (6/6≈1) plus control overhead

## FPGA-Specific Resource Mapping

### PYNQ-Z1 (XC7Z020)
- DSP48E1 slices: 220 available
- Usage: 10/220 = **4.5%**
- Headroom: 21x for future optimization

### ZCU104 (XCZU7EV)
- DSP48E2 slices: 1,728 available  
- Usage: 10/1728 = **0.6%**
- Headroom: 172x for scaling to larger models
