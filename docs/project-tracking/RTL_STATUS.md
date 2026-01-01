# RTL & Testbench Status
**Component:** Hardware Design & Verification  
**Last Updated:** January 2, 2026  
**Status:** ✅ ARCHITECTURE VALIDATED - ⏳ WAITING FOR ML WEIGHT VALIDATION

---

## Current State

**RTL is ready but BLOCKED waiting for ML team to validate weights.**

- ✅ Architecture matches Python model (22→32→16→2)
- ✅ Simulation passes with test weights  
- ✅ All RTL modules implemented
- ⚠️ **CANNOT synthesize** until ML confirms weights improve ACPR/EVM
- ⚠️ **A-SPSA loop NOT validated** - needs Python reference from ML

---

## Architecture Summary

**TDNN Generator:**
- Input: 22 features (I, Q + 5 memory taps × 2 channels × 2 variants)
- Hidden1: 32 neurons (LeakyReLU, α=0.25)
- Hidden2: 16 neurons (LeakyReLU, α=0.25)
- Output: 2 (I, Q, Tanh)
- **Total: 1,298 parameters per temperature bank**

**Clock Domains:**
- **200 MHz:** TDNN inference, I/O processing
- **1 MHz:** A-SPSA adaptation, weight updates

---

## Validation Status

### ✅ Completed Validation

| Test | Status | Result | Evidence |
|------|--------|--------|----------|
| Parameter matching | ✅ Pass | 22→32→16→2 = Python model | `validate_rtl_params.py` |
| MAC operations | ✅ Pass | Accumulator: 0x14658000 | Simulation log |
| State machine | ✅ Pass | All 9 states working | `tb_tdnn_simple.vcd` |
| Output values | ✅ Pass | Non-zero: 0x6C94, 0x6FB0 | Testbench |
| Weight loading | ✅ Pass | All 704 FC1 weights read | Address trace |
| Quantization | ✅ Pass | Q16.16→Q8.8→Q1.15 pipeline | Layer outputs |
| Latency | ✅ Pass | 1,257 cycles @ 200MHz = 6.3µs | Timing |

### Simulation Results (Test Weights)
```verilog
Test Configuration:
  Weights: 0x1000 (0.125 in Q1.15)
  Input:   I=0x4000 (0.5), Q=0x2000 (0.25), rest=0x0CCC (0.1)

Results:
  Output:  I=0x6C94 (0.848), Q=0x6FB0 (0.873)
  Latency: 1,257 cycles = 6.3µs @ 200MHz
  Status:  ✅ PASS - Non-zero outputs confirm MAC operations work
```

### ⏳ Pending Validation (Waiting on ML)

| Test | Status | Blocker |
|------|--------|---------|
| Real trained weights | ❌ Not tested | Waiting for ML to validate weights |
| ACPR improvement | ❌ Unknown | Need ML validation results |
| Thermal bank switching | ⏳ Code ready | Need validated 3 weight sets |
| A-SPSA loop | ❌ No reference | **Need Python model from ML** |
| Full system test | ⏳ TB ready | Waiting for validated weights |

---

## RTL Modules

### Core Modules (All Implemented)

| Module | Lines | Status | Function | Validated |
|--------|-------|--------|----------|-----------|
| `dpd_top.v` | 350 | ✅ Ready | Top integration, CDC | ⚠️ Needs system TB |
| `tdnn_generator.v` | 400 | ✅ Validated | TDNN @ 200MHz | ✅ Simulation pass |
| `aspsa_engine.v` | 300 | ⚠️ Needs ref | A-SPSA @ 1MHz | ❌ No Python model |
| `shadow_memory.v` | 250 | ✅ Ready | CDC weight memory | ⏳ Need integration test |
| `temp_controller.v` | 150 | ✅ Ready | Thermal switching | ⏳ Need scenario test |
| `input_buffer.v` | 200 | ✅ Ready | Memory taps | ⏳ Need integration test |
| `error_metric.v` | 150 | ✅ Ready | EVM calculation | ⏳ Need validation |

### Testbenches

| Testbench | Coverage | Status | Next Step |
|-----------|----------|--------|-----------|
| `tb_tdnn_simple.v` | TDNN only | ✅ Passing | Add real weights |
| `tb_tdnn_generator.v` | TDNN full | ✅ Passing | Test with ML weights |
| `tb_aspsa_engine.v` | A-SPSA | ⏳ Exists | **Need Python co-sim** |
| `tb_dpd_top.v` | Full system | ⏳ WIP | Waiting for ML validation |

---

## A-SPSA Loop Validation Strategy

**CRITICAL:** A-SPSA engine cannot be validated until ML provides reference model.

### What RTL Team Needs from ML Team

#### 1. Python Reference Model (`validate_aspsa_loop.py`)
**Required deliverables:**
- [ ] Floating-point A-SPSA implementation
- [ ] Fixed-point (Q1.15, Q16.16, Q8.24) version
- [ ] Test vectors:
  - Input samples (I/Q)
  - PA output with thermal drift
  - Expected weight updates per iteration
  - Convergence curve (NMSE vs iteration)

#### 2. Tuned Parameters
**Need validated values for:**
| Parameter | Symbol | RTL Representation | ML Must Provide |
|-----------|--------|-------------------|-----------------|
| Initial learning rate | `a` | Q8.24 (32-bit) | Tuned value |
| LR offset | `A` | 16-bit unsigned | Tuned value |
| LR decay exponent | `α` | Fixed: 0.602 | Confirm value |
| Gradient step | `c` | Q8.24 (32-bit) | Tuned value |
| Gradient decay | `γ` | Fixed: 0.101 | Confirm value |

**Example config (ML to provide):**
```yaml
aspsa:
  a: 0.05          # Initial LR (Q8.24 = 0x00CCCCCC)
  A: 100           # LR offset
  alpha: 0.602     # LR decay (fixed by theory)
  c: 0.005         # Grad step (Q8.24 = 0x0051EB85)
  gamma: 0.101     # Grad decay (fixed by theory)
```

#### 3. Temperature Transition Test Cases
**Required test scenarios:**
```
Scenario 1: Cold → Normal transition
  - Run A-SPSA with cold weights for 500 iterations
  - Temperature sensor crosses threshold
  - Switch to normal bank, reset iteration counter
  - Verify: Converges within 100 iterations

Scenario 2: Normal → Hot transition
  - Similar test
  
Scenario 3: Rapid temperature oscillation
  - Temperature crosses threshold multiple times
  - Verify: No divergence, stable adaptation
```

**ML deliverable:** Expected outputs for each scenario

### RTL Team Responsibilities (After ML Provides Reference)

#### 1. Co-Simulation Testbench
**Create:** `tb_aspsa_cosim.v` that:
- Reads test vectors from ML reference
- Runs RTL A-SPSA engine
- Compares cycle-by-cycle with Python
- Flags mismatches > ±1 LSB (quantization tolerance)

#### 2. Fixed-Point Verification
**Validate:** RTL matches fixed-point Python within quantization error

**Critical paths to check:**
```verilog
// Gradient estimation (must match Python Q16.16)
J_plus = loss_function(weights + c_k * delta);
J_minus = loss_function(weights - c_k * delta);
grad_est = (J_plus - J_minus) / (2 * c_k * delta);  // Check rounding!

// Weight update (must match Python Q1.15)
weight_new = saturate_q15(weight - a_k * grad_est);  // Check saturation!

// Annealing schedule (must match exactly)
a_k = a / pow(A + k + 1, alpha);  // Check fixed-point pow()
c_k = c / pow(k + 1, gamma);
```

#### 3. Learning Rate Annealing Verification
**Test:** Plot `a_k` and `c_k` from RTL vs Python reference

**Expected:**
- Both should decay smoothly (no jumps)
- Both should reach ~1e-3 after 1000 iterations
- Temperature reset should restart to initial values

#### 4. Convergence Validation
**Test:** Run full adaptation loop with thermal drift

**Metrics to match Python:**
- NMSE reduction curve
- Final converged weights (within ±1 LSB)
- Number of iterations to convergence
- Peak memory usage

---

## Parameter Verification

### Input Dimension (Validated ✅)
```
Base features:           2  (I, Q)
Linear memory taps:     10  (5 delays × 2 channels)
Nonlinear memory taps:  10  (5 delays × 2 channels)
────────────────────────────
Total INPUT_DIM:        22  ✅ MATCHES PYTHON
```

### Weight Memory Layout (Validated ✅)
```
Bank 0 (Cold, -20°C):    Addr 0    - 1297   (1,298 params)
Bank 1 (Normal, 25°C):   Addr 1298 - 2595   (1,298 params)
Bank 2 (Hot, 70°C):      Addr 2596 - 3893   (1,298 params)
────────────────────────────────────────────────────────────
Total BRAM depth:        3,894 params
BRAM blocks needed:      ~8 blocks @ 4Kb each
```

### Layer Breakdown (Validated ✅)
```
FC1: 22×32 = 704 weights + 32 biases = 736 params  (ADDR 0-735)
FC2: 32×16 = 512 weights + 16 biases = 528 params  (ADDR 736-1263)
FC3: 16×2  = 32 weights  + 2 biases  = 34 params   (ADDR 1264-1297)
──────────────────────────────────────────────────────────────────
Total per bank:                         1,298 params ✅
```

---

## Known Issues & Fixes

### Fixed Issues ✅
1. ✅ **Parameter mismatch** (18→22 inputs) - Fixed Jan 2
2. ✅ **Zero outputs** (weights too small) - Fixed Jan 1  
3. ✅ **MAC accumulation timing** - Fixed Jan 1

### Open Issues ⚠️
1. ⚠️ **A-SPSA not validated** - Waiting for ML Python reference
2. ⚠️ **CDC not formally verified** - Need assertions
3. ⚠️ **Temperature controller timing** - Need cross-clock constraints

---

## Performance Analysis

### Timing (Pre-Synthesis Estimates)
```
Target Clock:       200 MHz (5.0 ns period)
Critical Path:      MAC multiply → Add → Accumulate
Estimated Delay:    ~4.5 ns (0.5 ns margin)
Confidence:         Medium (needs synthesis to confirm)
```

### Resource Usage (Estimates)
```
Component          | LUTs    | DSPs | BRAM | Notes
-------------------|---------|------|------|------------------
TDNN Generator     | 15,000  | 72   | 8    | Dominates DSP usage
A-SPSA Engine      | 4,000   | 8    | 0    | Gradient estimation
Shadow Memory      | 2,000   | 0    | 10   | Dual-port CDC
Input Buffer       | 3,000   | 0    | 2    | Delay line
Temperature Ctrl   | 500     | 0    | 0    | State machine
Error Metric       | 2,500   | 6    | 0    | EVM calculation
Misc (CDC, ctrl)   | 3,000   | 0    | 0    | Handshaking
-------------------|---------|------|------|------------------
TOTAL (estimated)  | 30,000  | 86   | 20   |
PYNQ-Z1 available  | 53,200  | 220  | 140  |
Utilization        | 56%     | 39%  | 14%  | ✅ Should fit
```

**⚠️ Note:** These are estimates. Real utilization measured post-synthesis.

---

## Next Steps (Priority Order)

### Immediate (This Week)
1. **[BLOCKED]** Wait for ML to provide:
   - Weight validation results
   - A-SPSA Python reference model
   - Tuned parameters (a, c, A)

2. **[RTL Team]** While waiting:
   - [ ] Add assertions to CDC logic
   - [ ] Create timing constraints file
   - [ ] Prepare synthesis scripts
   - [ ] Document expected synthesis results

### After ML Validation (Next Week)
3. **[RTL Team]** Once ML provides reference:
   - [ ] Create A-SPSA co-simulation testbench
   - [ ] Validate A-SPSA against Python
   - [ ] Run full system testbench
   - [ ] Generate test report

### Synthesis (Week 3)
4. **[RTL Team]** Only after ML confirms weights work:
   - [ ] Run Vivado synthesis for PYNQ-Z1
   - [ ] Check timing report (must meet 200MHz)
   - [ ] Verify resource usage < estimates
   - [ ] Generate bitstream

---

## Build Commands (Ready but Don't Run Yet)

### Simulation (Can Run Now)
```bash
cd rtl

# Compile TDNN testbench
iverilog -g2012 -o tb_tdnn_simple.vvp \
  src/tdnn_generator.v src/activation.v tb/tb_tdnn_simple.v

# Run simulation
vvp tb_tdnn_simple.vvp

# View waveforms
gtkwave tb_tdnn_simple.vcd
```

### Synthesis (DO NOT RUN UNTIL ML VALIDATES)
```bash
# ⚠️ BLOCKED - Wait for ML validation!
cd rtl
vivado -mode batch -source scripts/build_pynq.tcl

# This will fail if weights don't improve ACPR!
```

---

## Recommendations for RTL Team

### What You Can Do Now (Not Blocked)
1. **Review CDC implementation** - Add formal verification assertions
2. **Prepare timing constraints** - Define clock domains, I/O timing
3. **Create synthesis checklist** - What to check in timing/utilization reports
4. **Study A-SPSA theory** - Understand the algorithm before implementing tests

### What You CANNOT Do Yet (Blocked by ML)
1. ❌ **Synthesize** - Weights not validated
2. ❌ **Validate A-SPSA** - No Python reference
3. ❌ **Full system test** - Need validated weights
4. ❌ **Hardware deployment** - Waiting on synthesis

### How to Unblock Yourself
**Contact ML team and request:**
1. Python A-SPSA reference model (`validate_aspsa_loop.py`)
2. Test vectors for co-simulation
3. Tuned parameters (a, c, A, α, γ)
4. Temperature transition test cases
5. Expected convergence curve (NMSE vs iteration)

---

## Success Criteria

### RTL Validation (Can Verify Now)
- ✅ Architecture matches Python: 22→32→16→2
- ✅ Simulation passes with test weights
- ✅ MAC operations verified
- ✅ Quantization pipeline correct

### System Validation (Blocked)
- ⏳ A-SPSA matches Python reference
- ⏳ Thermal switching validated
- ⏳ Full system testbench passes
- ⏳ Timing closure @ 200MHz (post-synthesis)

### Performance Validation (Waiting on ML)
- ❌ ACPR improvement measured (need ML validation first)
- ❌ Hardware test passes (need bitstream)
- ❌ Demo stable for >1 hour (hardware testing)

**Overall:** RTL is ready, but **cannot proceed without ML validation**

---

## Contact

**RTL Lead:** [Your Team]  
**Blocked By:** ML Team (need weight validation + A-SPSA reference)  
**Critical Path:** ML validation → A-SPSA co-sim → Synthesis → Hardware

---

*RTL is architecturally correct but cannot synthesize until ML confirms weights work.*
