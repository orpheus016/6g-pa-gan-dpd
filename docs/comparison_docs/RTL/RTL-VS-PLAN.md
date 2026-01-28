## FINAL-SPSA.md vs Current RTL Implementation: Gap Analysis

### ✅ **Features Already Implemented**

| FINAL-SPSA.md Requirement | RTL Location | Status |
|---------------------------|--------------|--------|
| Core SPSA algorithm | aspsa_engine.v | ✅ Complete |
| Bernoulli perturbation (LFSR) | aspsa_engine.v | ✅ Complete |
| Annealing schedule (shift-based) | aspsa_engine.v | ✅ Complete |
| Thermal zone detection | temp_controller.v | ✅ Complete |
| Hysteresis on zone transitions | temp_controller.v | ✅ Complete |
| Anneal reset on temp change | temp_controller.v | ✅ Complete |
| CDC shadow memory | shadow_memory.v | ✅ Complete |
| Double-buffered weight banks | shadow_memory.v | ✅ Complete |
| 2-stage CDC synchronizers | shadow_memory.v | ⚠️ Partial (spec says 3-stage) |
| Atomic bank swap | shadow_memory.v | ✅ Complete |
| 3 thermal weight banks | shadow_memory.v | ✅ Complete |
| Error metric (EVM-based) | error_metric.v | ✅ Complete |

---

### ❌ **Critical Missing Features**

| FINAL-SPSA.md Requirement | Section | RTL Status | Severity |
|---------------------------|---------|------------|----------|
| **Deadband FSM (IDLE/TRACK/PANIC/BYPASS)** | §3 | ❌ Missing | **CRITICAL** |
| **Safety overflow monitor** | §4.3 | ❌ Missing | **CRITICAL** |
| **Bypass MUX** | §4.4 | ❌ Missing | **CRITICAL** |
| **Pre-commit validation** | §5 | ❌ Missing | **HIGH** |
| **Divergence rate monitor** | §9.5 | ❌ Missing | **MEDIUM** |
| Variable gain scheduling (4× in PANIC) | §3.3 | ❌ Missing | **HIGH** |
| Variable update rate (1 kHz / 10 kHz) | §3.3 | ❌ Missing | **HIGH** |

---

### ⚠️ **Discrepancies / Partial Implementations**

| Feature | FINAL-SPSA.md Spec | Current RTL | Issue |
|---------|-------------------|-------------|-------|
| **CDC sync stages** | 3-stage (MTBF >100 years) | 2-stage | Lower metastability protection |
| **Annealing formula** | $a_k = a/(A+k)^\alpha$ | `a >> (k/period)` | RTL uses step-wise decay, not continuous |
| **Weight count** | 1362 | 1170 (aspsa) / 1554 (dpd_top) | Mismatch in parameters |
| **Clock frequency** | 250 MHz data / 1 MHz adapt | 200 MHz data / 1 MHz adapt | Different from spec |
| **EVM thresholds** | Q8.8 (-45 dB = 0xD300) | Not used | No deadband logic |
| **Thermal zones** | COLD<25°C, HOT>50°C | COLD<15°C, HOT>40°C | Different thresholds |

---

### **What Current RTL Does**

```
aspsa_engine.v State Machine:
ST_IDLE → ST_PERTURB_POS → ST_WAIT_POS → ST_PERTURB_NEG → 
ST_WAIT_NEG → ST_GRADIENT → ST_UPDATE → ST_SYNC → 
ST_WAIT_SYNC → ST_ANNEAL → ST_IDLE
```

**Missing:** No state machine considers EVM level (IDLE/TRACK/PANIC/BYPASS). SPSA runs unconditionally when enabled.

---

### **What Needs to Be Added**

#### 1. **Deadband FSM Module** (New file: `deadband_fsm.v`)

```verilog
module deadband_fsm (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] evm_db,           // From error_metric
    input  wire        overflow_flag,    // From safety_monitor
    input  wire        arm_reset,
    output reg  [1:0]  state,            // IDLE/TRACK/PANIC/BYPASS
    output reg         spsa_enable,
    output reg  [1:0]  gain_mult         // 0=off, 1=1×, 2=4×
);
```

#### 2. **Safety Monitor Module** (New file: `dpd_safety_monitor.v`)

```verilog
module dpd_safety_monitor (
    input  wire        clk_data,         // 250 MHz
    input  wire signed [15:0] dpd_i,
    input  wire signed [15:0] dpd_q,
    output reg         bypass_active,
    output reg         overflow_alarm
);
```

#### 3. **Bypass MUX Module** (New file: `dpd_bypass_mux.v`)

```verilog
module dpd_bypass_mux (
    input  wire        bypass_active,
    input  wire signed [15:0] adc_i, adc_q,
    input  wire signed [15:0] dpd_i, dpd_q,
    output wire signed [15:0] dac_i, dac_q
);
```

#### 4. **Modify aspsa_engine.v**

Current:
```verilog
input wire enable,  // Simple enable
```

Required:
```verilog
input wire [1:0] spsa_state,      // From deadband_fsm
input wire [1:0] gain_multiplier, // 1× or 4×
// Accept/reject logic for pre-commit validation
output reg       candidate_ready,
input  wire      candidate_accept  // From ARM or validation logic
```

#### 5. **Modify shadow_memory.v**

Current: 2-stage CDC sync
```verilog
reg swap_req_sync1, swap_req_sync2;
```

Required: 3-stage for higher MTBF
```verilog
reg swap_req_sync1, swap_req_sync2, swap_req_sync3;
```

---

### **Integration in dpd_top.v**

Current structure:
```
temp_controller → weight_bank_sel
input_buffer → tdnn_generator → output
error_metric → aspsa_engine → shadow_memory
```

Required structure:
```
temp_controller ─────────────────────────────────────┐
                                                     ▼
error_metric → deadband_fsm ──→ aspsa_engine → shadow_memory
                    │                              ▼
safety_monitor ─────┴────────→ bypass_mux ──→ DAC output
                                   ▲
                          tdnn_generator
```

---

### **Summary Table**

| Category | FINAL-SPSA.md Requirements | RTL Status |
|----------|---------------------------|------------|
| Core SPSA | 5/5 | ✅ 100% |
| Annealing | 3/4 | ⚠️ 75% (formula differs) |
| Thermal zones | 4/4 | ✅ 100% |
| CDC/Shadow RAM | 4/5 | ⚠️ 80% (2-stage not 3) |
| **Deadband FSM** | 0/4 | ❌ **0%** |
| **Safety system** | 0/3 | ❌ **0%** |
| **Pre-commit validation** | 0/3 | ❌ **0%** |
| **Divergence monitor** | 0/2 | ❌ **0%** |

**Overall: ~50% complete.** The fundamental SPSA algorithm works, but all safety and jitter-prevention mechanisms from FINAL-SPSA.md are missing.