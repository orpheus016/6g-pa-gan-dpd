# Final A-SPSA Implementation Specification

**Version:** 1.0  
**Last Updated:** January 28, 2026  
**Status:** Complete Specification  
**Dependencies:** !ARCHITECTURE.md Section 7

---

## Executive Summary

This document consolidates all A-SPSA (Annealed Simultaneous Perturbation Stochastic Approximation) requirements for online DPD adaptation. It merges:
- Steady-state jitter prevention (REVISED-SPSA.md)
- Safety and compliance requirements (PREVENT-JAMMING-RISK.md)
- Current architecture plan (!ARCHITECTURE.md §7)
- Gap analysis (COMPARISON-SPSA.md)

**Core Principle:** Test first, commit only if proven better. Never accept a weight update blindly.

---

## 1. Problem Statement

### 1.1 Why Online Adaptation is Required

PA characteristics drift with operating conditions:

| Factor | Effect on PA | Time Constant | Magnitude |
|--------|--------------|---------------|-----------|
| Temperature | Gain compression shifts | ~10 seconds | ~0.5%/10°C |
| Aging | Bias point drift | ~months | ~1-2% |
| Supply voltage | Operating point shift | ~ms | ~0.1%/mV |
| Self-heating | Thermal memory | ~100 µs | ~0.2% |

**Consequence:** Static DPD trained at 25°C degrades at other temperatures. ACPR can worsen by 5-10 dB without adaptation.

### 1.2 Why SPSA (Not Backpropagation)

| Method | Requires | Works with Real PA | Evaluations/Iteration |
|--------|----------|-------------------|----------------------|
| Backpropagation | Differentiable PA model | ❌ No | N/A |
| Finite Difference | N perturbations | ✅ Yes | 2N |
| **SPSA** | Random perturbation | ✅ Yes | **2** |

SPSA only requires loss function evaluation (EVM/ACPR measurement), not gradient computation through the PA. This is critical because the real PA is not differentiable.

**Source:** Spall, J.C. "Implementation of the Simultaneous Perturbation Algorithm for Stochastic Optimization," IEEE Trans. Aerospace & Electronic Systems, 1998.

---

## 2. A-SPSA Algorithm Specification

### 2.1 Core Update Rule

$$w_{k+1} = w_k - a_k \hat{g}_k$$

Where gradient estimate:

$$\hat{g}_k = \frac{L(w_k + c_k \Delta_k) - L(w_k - c_k \Delta_k)}{2 c_k} \odot \Delta_k^{-1}$$

- $\Delta_k$: Random perturbation vector (±1 Bernoulli for each weight)
- $a_k$: Step size (learning rate)
- $c_k$: Perturbation magnitude
- $L(\cdot)$: Loss function (EVM or ACPR in dB)

### 2.2 Gain Annealing Schedule

Standard annealing (Spall, 1998):

$$a_k = \frac{a}{(A + k)^\alpha}, \quad c_k = \frac{c}{k^\gamma}$$

**Optimal exponents:**
- $\alpha = 0.602$ (theory)
- $\gamma = 0.101$ (theory)

**Hardware-friendly simplification:**
- $\alpha = 1.0$ (integer division by $A+k$)
- $\gamma = 0.167 \approx 1/6$ (piecewise LUT)

### 2.3 Parameter Values

| Parameter | Symbol | Value | Q Format | Hardware | Justification |
|-----------|--------|-------|----------|----------|---------------|
| Initial step | $a$ | 0.01 | Q0.16 (655) | 16-bit | Conservative start |
| Stability const | $A$ | 100 | uint16 | 16-bit | Prevents $\div 0$ |
| Perturbation | $c$ | 0.001 | Q0.16 (65) | 16-bit | ~1 LSB of Q1.15 weight |
| Step decay | $\alpha$ | 1.0 | N/A | Division | Simplest hardware |
| Perturb decay | $\gamma$ | 0.167 | LUT | 8-entry | Piecewise approximation |
| Max iterations | $k_{max}$ | 10000 | uint14 | 14-bit | Reset on thermal change |

**Convergence behavior:**
- k=100: $a_{100} = 0.01/200 = 5×10^{-5}$
- k=1000: $a_{1000} = 0.01/1100 ≈ 9×10^{-6}$
- k=10000: $a_{10000} = 0.01/10100 ≈ 1×10^{-6}$ (effectively frozen)

---

## 3. Deadband State Machine (Jitter Prevention)

### 3.1 The Problem: Steady-State Jitter

Standard SPSA never stops perturbing. Even at optimum, it continues "wiggling" weights to check for improvements. This injects AM noise into the transmitted signal.

**Quantified impact:**
- Theoretical EVM floor: -48 dB
- With continuous perturbation: -45 dB (3 dB degradation from jitter alone)

### 3.2 Solution: 3-State Deadband FSM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEADBAND STATE MACHINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                           EVM > -35 dB                                      │
│                      ┌──────────────────┐                                   │
│                      │                  │                                   │
│                      ▼                  │                                   │
│   ┌──────────┐   EVM>-40dB   ┌──────────┴─┐   EVM>-30dB   ┌──────────┐     │
│   │   IDLE   │──────────────►│   TRACK    │──────────────►│  PANIC   │     │
│   │          │               │            │               │          │     │
│   │ SPSA OFF │◄──────────────│ SPSA ON    │◄──────────────│ SPSA ON  │     │
│   │ a_k = 0  │   EVM<-45dB   │ a_k = 1×   │   EVM<-35dB   │ a_k = 4× │     │
│   └──────────┘               └────────────┘               └──────────┘     │
│        │                           │                           │            │
│        │                           │                           │            │
│        └───────────────────────────┴───────────────────────────┘            │
│                                    │                                        │
│                            ANY OVERFLOW                                     │
│                                    ▼                                        │
│                            ┌──────────┐                                     │
│                            │  BYPASS  │  ← SAFETY STATE (new)               │
│                            │ DPD OFF  │                                     │
│                            └──────────┘                                     │
│                                                                             │
│   Hysteresis: 5 dB band prevents rapid state oscillation                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 State Parameters

| State | SPSA Active | $a_k$ Multiplier | $c_k$ Multiplier | Update Rate | Entry Condition |
|-------|-------------|------------------|------------------|-------------|-----------------|
| **IDLE** | ❌ OFF | 0× | 0× | 0 Hz | EVM < -45 dB |
| **TRACK** | ✅ ON | 1× | 1× | 1 kHz | -45 dB ≤ EVM < -35 dB |
| **PANIC** | ✅ ON | 4× | 2× | 10 kHz | EVM ≥ -35 dB |
| **BYPASS** | ❌ OFF | N/A | N/A | N/A | Overflow detected |

### 3.4 Hysteresis Implementation

```verilog
module deadband_fsm (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] evm_db,          // EVM in dB (Q8.8, negative)
    input  wire        overflow_flag,   // From safety monitor
    input  wire        arm_reset,       // ARM clears BYPASS
    output reg  [1:0]  state,
    output reg         spsa_enable,
    output reg  [3:0]  gain_shift       // Right shift for a_k
);
    // States
    localparam IDLE   = 2'b00;
    localparam TRACK  = 2'b01;
    localparam PANIC  = 2'b10;
    localparam BYPASS = 2'b11;
    
    // Thresholds (Q8.8, e.g., -45 dB = 0xD300)
    localparam EVM_IDLE_ENTER  = 16'hD300;  // -45 dB
    localparam EVM_IDLE_EXIT   = 16'hD800;  // -40 dB
    localparam EVM_PANIC_ENTER = 16'hDD00;  // -35 dB
    localparam EVM_PANIC_EXIT  = 16'hDD00;  // -35 dB (with hysteresis via state)
    localparam EVM_CRITICAL    = 16'hE200;  // -30 dB
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= TRACK;
            spsa_enable <= 1'b1;
            gain_shift <= 4'd0;
        end else if (overflow_flag) begin
            // SAFETY: Immediate transition to BYPASS
            state <= BYPASS;
            spsa_enable <= 1'b0;
            gain_shift <= 4'd15;  // Effectively zero
        end else if (state == BYPASS && arm_reset) begin
            // ARM must explicitly clear BYPASS
            state <= TRACK;
            spsa_enable <= 1'b1;
            gain_shift <= 4'd0;
        end else begin
            case (state)
                IDLE: begin
                    spsa_enable <= 1'b0;
                    gain_shift <= 4'd15;
                    if (evm_db > EVM_IDLE_EXIT)  // Worse than -40 dB
                        state <= TRACK;
                end
                
                TRACK: begin
                    spsa_enable <= 1'b1;
                    gain_shift <= 4'd0;  // 1× gain
                    if (evm_db < EVM_IDLE_ENTER)      // Better than -45 dB
                        state <= IDLE;
                    else if (evm_db > EVM_PANIC_ENTER) // Worse than -35 dB
                        state <= PANIC;
                end
                
                PANIC: begin
                    spsa_enable <= 1'b1;
                    gain_shift <= 4'd2;  // 4× gain (shift left, not right)
                    if (evm_db < EVM_PANIC_EXIT)
                        state <= TRACK;
                end
                
                BYPASS: begin
                    // Latched until ARM reset
                    spsa_enable <= 1'b0;
                end
            endcase
        end
    end
endmodule
```

---

## 4. Safety System (Jamming Prevention)

### 4.1 The Risk: Unstable DPD = Jammer

If SPSA diverges (accepts bad perturbation), the DPD output can:
1. Exceed DAC range → clipping → broadband spectral splatter
2. Oscillate → AM/FM noise across entire spectrum
3. Violate FCC/ETSI spectral masks → illegal transmission

**Time to disaster:** At 250 MSps, one bad weight update affects 250,000 samples before the next 1 kHz EVM check.

### 4.2 Solution: Real-Time Overflow Monitor

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REAL-TIME SAFETY MONITOR                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   DATA PATH (250 MHz)                                                       │
│   ═══════════════════                                                       │
│                                                                             │
│   DPD Output ──┬──► Magnitude ──► Threshold ──► Overflow? ──► Bypass MUX   │
│                │    Calculator    Comparator    (Latched)      Control     │
│                │                                    │                       │
│                │                                    ▼                       │
│                │                              ┌──────────┐                  │
│                │                              │  Alarm   │──► ARM IRQ      │
│                │                              │  Counter │                  │
│                │                              └──────────┘                  │
│                │                                                            │
│                ▼                                                            │
│   ┌────────────────────────────────────────────────────────────────┐       │
│   │                     BYPASS MUX                                  │       │
│   │  bypass_active=0: DAC ← DPD_out (normal operation)             │       │
│   │  bypass_active=1: DAC ← ADC_in  (passthrough, DPD disabled)    │       │
│   └────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Hardware Implementation

```verilog
module dpd_safety_monitor (
    input  wire        clk_data,        // 250 MHz
    input  wire        rst_n,
    input  wire signed [15:0] dpd_i,    // DPD I output (Q1.15)
    input  wire signed [15:0] dpd_q,    // DPD Q output (Q1.15)
    input  wire        arm_clear,       // ARM clears alarm
    output reg         bypass_active,   // Force DPD bypass
    output reg         alarm,           // Interrupt to ARM
    output reg  [15:0] overflow_count   // Statistics
);
    // Threshold: 87.5% of full scale = 0.875 × 32768 = 28672
    // Allows 1.2 dB headroom for DAC + analog chain peaks
    localparam [15:0] CLIP_THRESHOLD = 16'd28672;
    
    // L1 norm approximation (fast, no multiplier)
    wire [15:0] abs_i = dpd_i[15] ? (~dpd_i + 1'b1) : dpd_i;
    wire [15:0] abs_q = dpd_q[15] ? (~dpd_q + 1'b1) : dpd_q;
    wire [16:0] magnitude_l1 = abs_i + abs_q;  // Max = 65534
    
    // L1 to L2 approximation: L2 ≈ 0.707 × L1 for equal I/Q
    // For safety, use L1 directly (conservative)
    wire overflow_detect = (magnitude_l1 > {1'b0, CLIP_THRESHOLD});
    
    // Consecutive overflow counter (debounce single spikes)
    reg [3:0] overflow_streak;
    localparam STREAK_THRESHOLD = 4'd3;  // 3 consecutive = real problem
    
    always @(posedge clk_data or negedge rst_n) begin
        if (!rst_n) begin
            bypass_active <= 1'b0;
            alarm <= 1'b0;
            overflow_count <= 16'd0;
            overflow_streak <= 4'd0;
        end else if (arm_clear) begin
            bypass_active <= 1'b0;
            alarm <= 1'b0;
            overflow_streak <= 4'd0;
            // overflow_count preserved for diagnostics
        end else begin
            if (overflow_detect) begin
                overflow_streak <= (overflow_streak < 4'd15) ? 
                                   overflow_streak + 1'b1 : overflow_streak;
                overflow_count <= overflow_count + 1'b1;
                
                if (overflow_streak >= STREAK_THRESHOLD) begin
                    bypass_active <= 1'b1;  // LATCH: requires ARM reset
                    alarm <= 1'b1;
                end
            end else begin
                overflow_streak <= 4'd0;  // Reset streak on clean sample
            end
        end
    end
endmodule
```

### 4.4 Bypass MUX

```verilog
module dpd_bypass_mux (
    input  wire        clk_data,
    input  wire        bypass_active,
    input  wire signed [15:0] adc_i,    // Original input (passthrough)
    input  wire signed [15:0] adc_q,
    input  wire signed [15:0] dpd_i,    // DPD output
    input  wire signed [15:0] dpd_q,
    output reg  signed [15:0] dac_i,    // To DAC
    output reg  signed [15:0] dac_q
);
    always @(posedge clk_data) begin
        if (bypass_active) begin
            // Bypass mode: passthrough (no DPD)
            // PA without DPD is nonlinear but won't jam
            dac_i <= adc_i;
            dac_q <= adc_q;
        end else begin
            // Normal mode: DPD active
            dac_i <= dpd_i;
            dac_q <= dpd_q;
        end
    end
endmodule
```

---

## 5. Pre-Commit Validation (Shadow Mode)

### 5.1 The Problem: Blind Acceptance

Standard SPSA:
```
perturb → measure → update → (repeat)
```

Problem: If perturbation makes things worse, it's still applied. You discover the problem on the *next* iteration, after the damage is done.

### 5.2 Solution: Validate Before Commit

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRE-COMMIT VALIDATION FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ARM Processor (1 MHz loop)                                                │
│   ══════════════════════════                                                │
│                                                                             │
│   1. BASELINE: Measure EVM with current weights (w_current)                 │
│      └─► evm_baseline = measure_evm(N=1024)  // ~4 µs                      │
│                                                                             │
│   2. COMPUTE: Calculate candidate weights                                   │
│      └─► delta = bernoulli_random(±1, size=1362)                           │
│      └─► gradient = (L+ - L-) / (2 × c_k × delta)                          │
│      └─► w_candidate = w_current - a_k × gradient                          │
│                                                                             │
│   3. TEST: Write candidate to shadow bank, temporarily activate             │
│      └─► write_shadow_bank(w_candidate)                                    │
│      └─► toggle_bank()  // Shadow → Active                                 │
│      └─► wait(10 µs)    // Let pipeline flush                              │
│      └─► evm_candidate = measure_evm(N=1024)                               │
│                                                                             │
│   4. DECIDE: Compare and commit or revert                                   │
│      └─► if (evm_candidate < evm_baseline - MARGIN):                       │
│              // Improvement confirmed, keep new weights                     │
│              w_current = w_candidate                                        │
│              accept_count++                                                 │
│          else:                                                              │
│              // No improvement or worse, revert                             │
│              toggle_bank()  // Back to original                             │
│              reject_count++                                                 │
│                                                                             │
│   5. MONITOR: Track acceptance ratio                                        │
│      └─► if (reject_count > 10 consecutive):                               │
│              state = PANIC  // SPSA is stuck, increase gains               │
│      └─► if (accept_count > 50):                                           │
│              // Converged, check if can enter IDLE                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 ARM Software Implementation

```c
// ARM Cortex-A9 SPSA Controller (C pseudocode)

#define N_WEIGHTS       1362
#define EVM_MARGIN_DB   0.5f    // Require 0.5 dB improvement to accept
#define MAX_REJECTS     10
#define EVM_IDLE_THRESH -45.0f
#define EVM_PANIC_THRESH -35.0f

typedef enum {
    STATE_IDLE,
    STATE_TRACK,
    STATE_PANIC,
    STATE_BYPASS
} spsa_state_t;

typedef struct {
    float weights[N_WEIGHTS];
    float a, A, c;
    float alpha, gamma;
    uint32_t k;                 // Iteration counter
    spsa_state_t state;
    uint32_t reject_streak;
    uint32_t accept_count;
    uint8_t active_bank;        // 0 or 1
} spsa_context_t;

void spsa_iteration(spsa_context_t* ctx) {
    // Skip if in IDLE or BYPASS
    if (ctx->state == STATE_IDLE || ctx->state == STATE_BYPASS) {
        return;
    }
    
    // 1. Baseline measurement
    float evm_baseline = measure_evm_db(1024);
    
    // 2. Check state transitions based on EVM
    update_state(ctx, evm_baseline);
    if (ctx->state == STATE_IDLE) return;
    
    // 3. Compute annealed gains
    float a_k = ctx->a / powf(ctx->A + ctx->k, ctx->alpha);
    float c_k = ctx->c / powf(ctx->k + 1, ctx->gamma);
    
    // Apply state multipliers
    if (ctx->state == STATE_PANIC) {
        a_k *= 4.0f;
        c_k *= 2.0f;
    }
    
    // 4. Generate perturbation (Bernoulli ±1)
    int8_t delta[N_WEIGHTS];
    for (int i = 0; i < N_WEIGHTS; i++) {
        delta[i] = (rand() & 1) ? 1 : -1;
    }
    
    // 5. Evaluate L+ and L-
    float w_plus[N_WEIGHTS], w_minus[N_WEIGHTS];
    for (int i = 0; i < N_WEIGHTS; i++) {
        w_plus[i]  = ctx->weights[i] + c_k * delta[i];
        w_minus[i] = ctx->weights[i] - c_k * delta[i];
    }
    
    // L+ measurement
    write_shadow_bank(w_plus);
    toggle_bank();
    usleep(10);  // Pipeline flush
    float evm_plus = measure_evm_db(1024);
    
    // L- measurement
    write_shadow_bank(w_minus);
    toggle_bank();
    usleep(10);
    float evm_minus = measure_evm_db(1024);
    
    // 6. Compute gradient estimate and candidate weights
    float w_candidate[N_WEIGHTS];
    for (int i = 0; i < N_WEIGHTS; i++) {
        float grad_i = (evm_plus - evm_minus) / (2.0f * c_k * delta[i]);
        w_candidate[i] = ctx->weights[i] - a_k * grad_i;
    }
    
    // 7. Test candidate
    write_shadow_bank(w_candidate);
    toggle_bank();
    usleep(10);
    float evm_candidate = measure_evm_db(1024);
    
    // 8. Decision: accept or reject
    if (evm_candidate < evm_baseline - EVM_MARGIN_DB) {
        // ACCEPT: Keep candidate weights
        memcpy(ctx->weights, w_candidate, sizeof(ctx->weights));
        ctx->reject_streak = 0;
        ctx->accept_count++;
    } else {
        // REJECT: Revert to original
        write_shadow_bank(ctx->weights);
        toggle_bank();
        ctx->reject_streak++;
        
        // Too many rejections = stuck
        if (ctx->reject_streak > MAX_REJECTS && ctx->state != STATE_PANIC) {
            ctx->state = STATE_PANIC;
            ctx->k = 0;  // Reset annealing
        }
    }
    
    ctx->k++;
}

void update_state(spsa_context_t* ctx, float evm_db) {
    switch (ctx->state) {
        case STATE_TRACK:
            if (evm_db < EVM_IDLE_THRESH) {
                ctx->state = STATE_IDLE;
            } else if (evm_db > EVM_PANIC_THRESH) {
                ctx->state = STATE_PANIC;
                ctx->k = 0;  // Reset for fast tracking
            }
            break;
            
        case STATE_PANIC:
            if (evm_db < EVM_PANIC_THRESH) {
                ctx->state = STATE_TRACK;
            }
            break;
            
        case STATE_IDLE:
            if (evm_db > EVM_IDLE_THRESH + 5.0f) {  // Hysteresis
                ctx->state = STATE_TRACK;
            }
            break;
            
        case STATE_BYPASS:
            // Only ARM explicit command exits BYPASS
            break;
    }
}
```

---

## 6. Thermal Zone Management

### 6.1 Problem: Annealed SPSA Can't Track Fast Changes

After 10,000 iterations, $a_k ≈ 10^{-6}$. If temperature jumps 30°C, the gains are too small to track.

### 6.2 Solution: Zone-Based Reset with Warm Start

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THERMAL ZONE ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Temperature ADC ──► Zone Detector ──► Zone Change? ──► Actions           │
│                            │                  │                             │
│                            ▼                  ▼                             │
│                     ┌──────────────┐   ┌────────────────────┐              │
│                     │ Zone 0: COLD │   │ 1. Load pre-trained │              │
│                     │   < 25°C     │   │    weights for zone │              │
│                     ├──────────────┤   │ 2. Reset k = 0      │              │
│                     │ Zone 1: NORMAL│   │ 3. Force TRACK state│              │
│                     │  25-50°C     │   └────────────────────┘              │
│                     ├──────────────┤                                        │
│                     │ Zone 2: HOT  │                                        │
│                     │   > 50°C     │                                        │
│                     └──────────────┘                                        │
│                                                                             │
│   Pre-trained weight banks (offline, stored in BRAM/DDR):                   │
│   - weights_cold[1362]   : Trained at 15°C                                  │
│   - weights_normal[1362] : Trained at 35°C                                  │
│   - weights_hot[1362]    : Trained at 60°C                                  │
│                                                                             │
│   SPSA fine-tunes from warm start → faster convergence                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Hardware Implementation

```verilog
module thermal_controller (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [11:0] temp_adc,        // 12-bit temperature ADC
    output reg  [1:0]  thermal_zone,    // 00=COLD, 01=NORMAL, 10=HOT
    output reg         zone_changed,    // Pulse on transition
    output reg  [1:0]  weight_bank_sel  // Select pre-trained bank
);
    // ADC thresholds (calibrated to sensor)
    // Assuming 0.1°C/LSB, 0°C = 0x000
    localparam [11:0] THRESH_COLD   = 12'd250;   // 25°C
    localparam [11:0] THRESH_HOT    = 12'd500;   // 50°C
    localparam [11:0] HYSTERESIS    = 12'd30;    // 3°C hysteresis
    
    reg [1:0] zone_prev;
    reg [1:0] zone_raw;
    
    // Zone determination with hysteresis
    always @(*) begin
        case (thermal_zone)
            2'b00: begin  // Currently COLD
                if (temp_adc > THRESH_COLD + HYSTERESIS)
                    zone_raw = 2'b01;
                else
                    zone_raw = 2'b00;
            end
            2'b01: begin  // Currently NORMAL
                if (temp_adc < THRESH_COLD - HYSTERESIS)
                    zone_raw = 2'b00;
                else if (temp_adc > THRESH_HOT + HYSTERESIS)
                    zone_raw = 2'b10;
                else
                    zone_raw = 2'b01;
            end
            2'b10: begin  // Currently HOT
                if (temp_adc < THRESH_HOT - HYSTERESIS)
                    zone_raw = 2'b01;
                else
                    zone_raw = 2'b10;
            end
            default: zone_raw = 2'b01;
        endcase
    end
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            thermal_zone <= 2'b01;      // Start NORMAL
            zone_prev <= 2'b01;
            zone_changed <= 1'b0;
            weight_bank_sel <= 2'b01;
        end else begin
            zone_prev <= thermal_zone;
            thermal_zone <= zone_raw;
            
            // Detect zone transition
            zone_changed <= (zone_raw != zone_prev);
            
            // Update weight bank selector
            weight_bank_sel <= zone_raw;
        end
    end
endmodule
```

---

## 7. CDC Architecture for Weight Updates

### 7.1 Challenge: Clock Domain Crossing

- SPSA runs at 1 MHz (ARM clock domain)
- Inference runs at 250 MHz (FPGA fabric)
- Writing weights during inference corrupts output

### 7.2 Solution: Double-Buffered Shadow RAM with Atomic Swap

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CDC SHADOW RAM ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ADAPTATION DOMAIN (ARM, ~1 MHz)    │     DATA DOMAIN (FPGA, 250 MHz)     │
│   ═══════════════════════════════    │     ════════════════════════════    │
│                                      │                                      │
│   ┌─────────────────────┐            │     ┌─────────────────────┐         │
│   │  SHADOW BANK        │            │     │  ACTIVE BANK        │         │
│   │  (weights_shadow)   │◄───────────┼─────│  (weights_active)   │──► FC   │
│   │                     │   Atomic   │     │                     │         │
│   │  Written by ARM     │    Swap    │     │  Read by systolic   │         │
│   │  during compute     │  (1-bit)   │     │  array at 250 MHz   │         │
│   └─────────────────────┘            │     └─────────────────────┘         │
│            ▲                         │              ▲                       │
│            │                         │              │                       │
│   ┌────────┴────────┐                │     ┌────────┴────────┐             │
│   │  SPSA Engine    │                │     │  Bank Select    │             │
│   │  (ARM software) │    CDC Sync    │     │  (1-bit FF)     │             │
│   │                 │────────────────┼────►│                 │             │
│   │  swap_request   │    (3-stage)   │     │  0 = Bank A     │             │
│   │                 │                │     │  1 = Bank B     │             │
│   └─────────────────┘                │     └─────────────────┘             │
│                                      │                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 CDC Synchronizer

```verilog
module cdc_pulse_sync (
    input  wire clk_src,        // ARM clock (~1 MHz)
    input  wire clk_dst,        // FPGA clock (250 MHz)
    input  wire rst_n,
    input  wire pulse_in,       // Pulse in source domain
    output reg  pulse_out       // Pulse in destination domain
);
    // Toggle in source domain
    reg toggle_src;
    always @(posedge clk_src or negedge rst_n) begin
        if (!rst_n)
            toggle_src <= 1'b0;
        else if (pulse_in)
            toggle_src <= ~toggle_src;
    end
    
    // 3-stage synchronizer in destination domain
    reg [2:0] sync_chain;
    always @(posedge clk_dst or negedge rst_n) begin
        if (!rst_n)
            sync_chain <= 3'b000;
        else
            sync_chain <= {sync_chain[1:0], toggle_src};
    end
    
    // Edge detect for pulse
    always @(posedge clk_dst or negedge rst_n) begin
        if (!rst_n)
            pulse_out <= 1'b0;
        else
            pulse_out <= sync_chain[2] ^ sync_chain[1];
    end
endmodule
```

### 7.4 Bank Swap Controller

```verilog
module weight_bank_controller (
    input  wire        clk_data,        // 250 MHz
    input  wire        rst_n,
    input  wire        swap_request,    // From CDC sync
    output reg         active_bank,     // 0 or 1
    output wire [10:0] weight_addr,     // To BRAM
    output wire        weight_bank_sel  // Which BRAM to read
);
    // Atomic bank swap on synchronized request
    always @(posedge clk_data or negedge rst_n) begin
        if (!rst_n)
            active_bank <= 1'b0;
        else if (swap_request)
            active_bank <= ~active_bank;
    end
    
    assign weight_bank_sel = active_bank;
endmodule
```

---

## 8. Complete System Integration

### 8.1 Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE A-SPSA SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        ARM PROCESSOR (PS)                            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │   SPSA      │  │   State     │  │   Weight    │  │   EVM      │ │   │
│  │  │   Engine    │─►│   Machine   │─►│   Manager   │─►│   Compute  │ │   │
│  │  │             │  │             │  │             │  │            │ │   │
│  │  │ - Gradient  │  │ - IDLE      │  │ - Validate  │  │ - FFT      │ │   │
│  │  │ - Perturb   │  │ - TRACK     │  │ - Shadow    │  │ - Metric   │ │   │
│  │  │ - Anneal    │  │ - PANIC     │  │ - Commit    │  │ - Compare  │ │   │
│  │  └─────────────┘  └─────────────┘  └──────┬──────┘  └─────┬──────┘ │   │
│  │                                           │                │        │   │
│  └───────────────────────────────────────────┼────────────────┼────────┘   │
│                                              │                │            │
│  ════════════════════════════════════════════╪════════════════╪════════    │
│                                   CDC Sync   │      Feedback  │            │
│  ════════════════════════════════════════════╪════════════════╪════════    │
│                                              │                │            │
│  ┌───────────────────────────────────────────┼────────────────┼────────┐   │
│  │                        FPGA FABRIC (PL)   │                │        │   │
│  │                                           ▼                │        │   │
│  │  ┌───────────┐   ┌───────────┐   ┌───────────────┐   ┌────┴─────┐ │   │
│  │  │   ADC     │──►│   FEx +   │──►│   PN-TDNN     │──►│  Safety  │ │   │
│  │  │   IQ In   │   │   Phase   │   │   Systolic    │   │  Monitor │ │   │
│  │  │           │   │   Norm    │   │   FC Layers   │   │          │ │   │
│  │  └───────────┘   └───────────┘   └───────┬───────┘   └────┬─────┘ │   │
│  │                                          │                │        │   │
│  │                  ┌───────────────────────┴────────────────┘        │   │
│  │                  │                                                  │   │
│  │                  ▼                                                  │   │
│  │          ┌───────────────┐   ┌───────────────┐   ┌───────────┐    │   │
│  │          │  Bypass MUX   │──►│  Phase        │──►│   DAC     │    │   │
│  │          │  (Safety)     │   │  Denorm       │   │   IQ Out  │    │   │
│  │          └───────────────┘   └───────────────┘   └───────────┘    │   │
│  │                  ▲                                                  │   │
│  │                  │                                                  │   │
│  │          ┌───────┴───────┐                                         │   │
│  │          │ Thermal Zone  │◄─── Temperature ADC                     │   │
│  │          │ Controller    │                                         │   │
│  │          └───────────────┘                                         │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Timing Budget

| Phase | Duration | Frequency |
|-------|----------|-----------|
| EVM measurement (1024 samples) | 4.1 µs | — |
| SPSA gradient computation | ~10 µs | — |
| Weight write to shadow bank | ~50 µs | — |
| Bank swap (CDC sync) | ~12 ns | — |
| **Full iteration (TRACK)** | **~1 ms** | **1 kHz** |
| **Full iteration (PANIC)** | **~100 µs** | **10 kHz** |

### 8.3 Resource Summary

| Component | BRAM (36Kb) | DSP48 | LUT | FF |
|-----------|-------------|-------|-----|-----|
| Shadow RAM (2 banks) | 4 | 0 | 128 | 64 |
| CDC Sync | 0 | 0 | 32 | 24 |
| Deadband FSM | 0 | 0 | 64 | 32 |
| Safety Monitor | 0 | 0 | 96 | 48 |
| Thermal Controller | 0 | 0 | 48 | 24 |
| Divergence Monitor | 0 | 0 | 64 | 32 |
| **Total A-SPSA overhead** | **4** | **0** | **432** | **224** |

---

## 9. Gap Analysis Cross-Reference

This section maps COMPARISON-SPSA.md concerns to implementation sections in this document.

| Concern | Status | Implementation Location | Notes |
|---------|--------|------------------------|-------|
| **Spectral mask guard** | ✅ Implemented | §4.2, §4.3 | Real-time overflow monitor at 250 MHz |
| **Bypass on divergence** | ✅ Implemented | §3.2 (BYPASS state), §4.4 | Automatic failover to passthrough |
| **Pre-commit validation** | ✅ Implemented | §5 (entire section) | Shadow mode with accept/reject logic |
| **Output clipping detection** | ✅ Implemented | §4.3 (`dpd_safety_monitor`) | L1 norm vs 87.5% DAC threshold |
| **Divergence rate monitoring** | ✅ Implemented | §9.5 (below) | EVM derivative + streak counter |
| Jitter prevention | ✅ Implemented | §3 | Deadband FSM with IDLE state |
| Thermal tracking | ✅ Implemented | §6 | Zone-based reset with warm start |
| CDC-safe updates | ✅ Implemented | §7 | Double-buffered shadow RAM |

### 9.5 Divergence Rate Monitor (NEW)

**Problem:** Reject streak counter detects stuck SPSA, but not rapid oscillation (EVM bouncing ±5 dB).

**Solution:** Monitor EVM rate-of-change. If variance exceeds threshold, trigger BYPASS.

```c
// ARM software addition to spsa_iteration()

#define EVM_WINDOW_SIZE  10     // Moving average window
#define EVM_VARIANCE_THRESH 4.0f  // 4 dB^2 variance = unstable

typedef struct {
    float evm_history[EVM_WINDOW_SIZE];
    uint8_t history_idx;
    float evm_variance;
} divergence_monitor_t;

void update_divergence_monitor(divergence_monitor_t* mon, float evm_db) {
    // Update circular buffer
    mon->evm_history[mon->history_idx] = evm_db;
    mon->history_idx = (mon->history_idx + 1) % EVM_WINDOW_SIZE;
    
    // Compute variance over window
    float mean = 0.0f;
    for (int i = 0; i < EVM_WINDOW_SIZE; i++) {
        mean += mon->evm_history[i];
    }
    mean /= EVM_WINDOW_SIZE;
    
    float variance = 0.0f;
    for (int i = 0; i < EVM_WINDOW_SIZE; i++) {
        float diff = mon->evm_history[i] - mean;
        variance += diff * diff;
    }
    variance /= EVM_WINDOW_SIZE;
    mon->evm_variance = variance;
    
    // Check for divergence
    if (variance > EVM_VARIANCE_THRESH) {
        // Rapid oscillation detected
        trigger_bypass_mode();
        log_error("SPSA divergence: variance = %.2f dB^2", variance);
    }
}

// Call in spsa_iteration() after each EVM measurement:
// update_divergence_monitor(&div_mon, evm_baseline);
```

**Hardware Support (Optional):**

```verilog
module divergence_detector (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] evm_db,       // New EVM sample (Q8.8)
    input  wire        evm_valid,    // Sample strobe
    output reg         divergence    // Flag for ARM
);
    // Simple rate-of-change detector
    reg signed [15:0] evm_prev;
    reg signed [16:0] evm_delta;
    reg [15:0] abs_delta;
    
    // Threshold: 5 dB change in one iteration = suspicious
    localparam [15:0] DELTA_THRESH = 16'd1280;  // 5 dB in Q8.8
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            evm_prev <= 16'd0;
            divergence <= 1'b0;
        end else if (evm_valid) begin
            evm_delta <= evm_db - evm_prev;
            abs_delta <= evm_delta[16] ? (~evm_delta[15:0] + 1'b1) : evm_delta[15:0];
            
            if (abs_delta > DELTA_THRESH) begin
                divergence <= 1'b1;  // Latch until ARM clears
            end
            
            evm_prev <= evm_db;
        end
    end
endmodule
```

**Integration:**
- Add `divergence_detector` instance in FPGA fabric
- Connect to same ARM IRQ as `dpd_safety_monitor`
- ARM decides: gradual drift (PANIC) vs rapid oscillation (BYPASS)

---

## 10. Verification Checklist

### 10.1 Functional Tests

| Test | Pass Criteria | Priority |
|------|---------------|----------|
| Deadband transitions | IDLE↔TRACK↔PANIC at correct thresholds | HIGH |
| Hysteresis band | No rapid oscillation near thresholds | HIGH |
| Safety bypass trigger | Activates within 3 samples of overflow | CRITICAL |
| Bypass passthrough | Zero additional latency, unity gain | CRITICAL |
| CDC metastability | No glitches on bank swap (10M swaps) | HIGH |
| Thermal reset | k=0 on zone change, weights loaded | MEDIUM |
| Pre-commit validation | Rejects bad perturbations >95% | HIGH |
| Reject streak detection | PANIC triggered after 10 consecutive | MEDIUM |
| **Divergence rate monitor** | Variance <4 dB² in steady state | **MEDIUM** |
| **Rapid oscillation detect** | BYPASS within 10 samples if Δ>5dB | **HIGH** |

### 10.2 Performance Tests

| Test | Target | Method |
|------|--------|--------|
| Tracking convergence | <1 second to -45 dB EVM | Step temperature 25→50°C |
| Steady-state EVM | <-45 dB with SPSA idle | Run 1 hour at constant temp |
| Jitter in IDLE | <0.1 dB EVM fluctuation | Monitor EVM variance |
| PANIC recovery | <100 ms to TRACK | Inject artificial error spike |

---

## 11. References

1. Spall, J.C., "Implementation of the Simultaneous Perturbation Algorithm for Stochastic Optimization," IEEE Trans. Aerospace & Electronic Systems, Vol. 34, No. 3, 1998.

2. Spall, J.C., *Introduction to Stochastic Search and Optimization*, Wiley, 2003. (Chapter 7: SPSA)

3. Kushner, H.J. and Yin, G.G., *Stochastic Approximation and Recursive Algorithms and Applications*, Springer, 2003.

4. Cripps, S.C., *RF Power Amplifiers for Wireless Communications*, 2nd ed., Artech House, 2006.

5. Raich, R. et al., "Orthogonal polynomials for power amplifier modeling and predistorter design," IEEE Trans. Vehicular Technology, Vol. 53, No. 5, 2004.

---

## Appendix A: Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         A-SPSA QUICK REFERENCE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STATES:                                                                    │
│  ─────────────────────────────────────────────────                         │
│  IDLE   : EVM < -45 dB     → SPSA OFF, weights frozen                      │
│  TRACK  : -45 ≤ EVM < -35  → SPSA ON, normal gains                         │
│  PANIC  : EVM ≥ -35 dB     → SPSA ON, 4× gain, 10× rate                    │
│  BYPASS : Overflow detected → DPD disabled, ARM must clear                  │
│                                                                             │
│  PARAMETERS:                                                                │
│  ─────────────────────────────────────────────────                         │
│  a = 0.01    (initial step size)                                           │
│  A = 100     (stability constant)                                          │
│  c = 0.001   (perturbation size)                                           │
│  α = 1.0     (step decay exponent)                                         │
│  γ = 0.167   (perturbation decay exponent)                                 │
│                                                                             │
│  SAFETY:                                                                    │
│  ─────────────────────────────────────────────────                         │
│  Overflow threshold: 87.5% DAC range (28672/32768)                         │
│  Streak to trigger: 3 consecutive samples                                   │
│  Recovery: ARM must explicitly clear bypass                                 │
│                                                                             │
│  THERMAL ZONES:                                                             │
│  ─────────────────────────────────────────────────                         │
│  COLD   : < 25°C  → weights_cold[]                                         │
│  NORMAL : 25-50°C → weights_normal[]                                       │
│  HOT    : > 50°C  → weights_hot[]                                          │
│  Zone change → Reset k=0, load pre-trained weights                         │
│                                                                             │
│  VALIDATION RULE:                                                           │
│  ─────────────────────────────────────────────────                         │
│  Accept weight update ONLY if: EVM_new < EVM_old - 0.5 dB                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
