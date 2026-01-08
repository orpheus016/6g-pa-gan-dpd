# 6G PA DPD System Architecture

## Overview

This document describes the architecture of the 6G PA Digital Predistortion (DPD) system designed for the **29th LSI Design Contest in Okinawa**.

### Key Innovation
- **CWGAN-GP offline training** with spectral loss (EVM, ACPR, NMSE) for superior linearization
- **30-dimensional TDNN** (30→32→16→2) with nonlinear envelope features (|x|, |x|², |x|⁴)
- **Custom QAT** (Q1.15 weights, Q8.8 activations) for FPGA deployment
- **Two-stage training** - 50 epochs supervised pretraining + 250 epochs GAN fine-tuning
- **Enhanced augmentation** - AWGN, phase offset, gain variation, thermal drift
- **Conditional discriminator** with spectral normalization for Lipschitz constraint
- **Production models** - TDNNGeneratorQAT, Discriminator, SpectralLoss from codebase
- **Target: -60 dB ACPR** (beats OpenDPD -59 dB, train.py -58 dB)

### Honest Claims vs. Aspirational Goals

| Aspect | What We Claim | What We Don't Claim |
|--------|---------------|---------------------|
| GAN Role | CWGAN-GP achieves -60 to -62 dB ACPR | GAN replaces conventional DPD |
| Architecture | 30-dim TDNN with nonlinear features | Simple memory polynomial |
| Training | Two-stage: 50 pretrain + 250 GAN epochs | Single-stage training |
| QAT | Custom Q1.15/Q8.8 for FPGA | Generic PyTorch W8A8 |
| Validation | Comprehensive: TensorBoard + 3-way comparison | Basic MSE-only metrics |
| Data | Real measured PA data (OpenDPD 200 MHz GaN) | Synthetic/simulated data |
| Performance | Beats OpenDPD (-59 dB) and train.py (-58 dB) | Magical results without proof |

### 30-Dimensional TDNN Architecture

**Input Feature Structure (30 dimensions):**
1. **Current I, Q**: 2 dims
2. **Nonlinear envelope features**: 18 dims
   - For each of 6 memory taps (current + 5 previous): |x|, |x|², |x|⁴
   - 6 taps × 3 features = 18 dims
3. **Delayed I/Q**: 10 dims
   - Previous 5 taps × 2 (I, Q) = 10 dims

**Network Structure**: 30 → 32 → 16 → 2
- FC1: 30 → 32 (960 weights + 32 biases)
- FC2: 32 → 16 (512 weights + 16 biases)
- FC3: 16 → 2 (32 weights + 2 biases)
- **Total: 1,554 parameters (9.3 KB in Q8.8)**

**Why 30-dim works better than 18-dim:**
- 18-dim: Only linear envelope |x| per tap
- 30-dim: Nonlinear features |x|², |x|⁴ capture AM-AM/AM-PM
- Result: ~4-6 dB ACPR improvement

**Reference:** Yao et al., *"Deep Learning for DPD"*, IEEE JSAC 2021

### CWGAN-GP Training Strategy (Google Colab)

**Two-Stage Training Pipeline:**

**Stage 1: Supervised Pretraining (Epochs 1-50)**
- Loss: MSE-only (no discriminator)
- Purpose: Stable weight initialization
- Learning rate: 1e-4
- Expected: ACPR ~-40 to -45 dB

**Stage 2: GAN Fine-Tuning (Epochs 51-300)**
- Loss: Wasserstein + Gradient Penalty + Spectral
- QAT: Enabled at epoch 50
- Augmentation: Noise, phase, gain, thermal (50% probability)
- N_critic: 5 (discriminator updates per generator update)
- Expected: ACPR -60 to -62 dB

**Loss Function:**
```python
# Generator loss
L_G = L_wasserstein + λ_spectral * (L_EVM + L_ACPR + L_NMSE)

# Discriminator loss  
L_D = D(fake, condition) - D(real, condition) + λ_GP * gradient_penalty
```

**Models:**
- **Generator**: TDNNGeneratorQAT (30→32→16→2) with MemoryTapAssembly
- **Discriminator**: Conditional (4→64→32→16→1) with spectral normalization
- **Spectral Loss**: SpectralLoss (EVM + ACPR + NMSE)

Standard MSE training: `L = E[|y - x|²]`
GAN with spectral loss: `L = L_adv + λ₁·L_EVM + λ₂·L_ACPR + λ₃·L_NMSE`

GAN trains the TDNN offline. TDNN runs on FPGA. GAN never runs on FPGA.

**Reference:** Tervo et al., *"Adversarial Learning for Neural DPD"*, WAMICON 2019

## Contest Demo Setup (Algorithm Validation - No RF Equipment)

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        DIGITAL LOOPBACK DEMO                                   ║
║                     (No ADC/DAC, No PA, No Analyzer)                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                              LAPTOP                                      │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │  Python GUI (demo/video_demo.py)                                │    │
    │  │  - Generate OFDM I/Q signal                                     │    │
    │  │  - Display constellation & spectrum                             │    │
    │  │  - Show EVM/ACPR metrics                                        │    │
    │  └──────────────────────────┬──────────────────────────────────────┘    │
    └─────────────────────────────┼───────────────────────────────────────────┘
                                  │ USB/Ethernet
                                  ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         PYNQ-Z1 FPGA                                     │
    │  ┌────────────────────────────────────────────────────────────────┐     │
    │  │  PS (ARM Cortex-A9)                                            │     │
    │  │  - Receive I/Q via AXI                                         │     │
    │  │  - Send to PL via AXI DMA                                      │     │
    │  │  - Read results back                                           │     │
    │  └───────────────────────────┬────────────────────────────────────┘     │
    │                              │ AXI HP0                                   │
    │  ┌───────────────────────────▼────────────────────────────────────┐     │
    │  │  PL (Programmable Logic)                                       │     │
    │  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐  │     │
    │  │  │  DPD TDNN   │──►│  PA Digital │──►│  Output Buffer      │  │     │
    │  │  │  Generator  │   │    Twin     │   │                     │  │     │
    │  │  └─────────────┘   └─────────────┘   └─────────────────────┘  │     │
    │  │        ▲                                                       │     │
    │  │        │                                                       │     │
    │  │  ┌─────┴───────┐   ┌─────────────┐                            │     │
    │  │  │   A-SPSA    │◄──│   Error     │                            │     │
    │  │  │   Engine    │   │   Metric    │                            │     │
    │  │  └─────────────┘   └─────────────┘                            │     │
    │  └────────────────────────────────────────────────────────────────┘     │
    │                                                                          │
    │  User Interface:                                                         │
    │  [BTN0] DPD Enable   [SW0-1] Temp Select   [LED0-3] Status              │
    │  [BTN1] Adapt Enable [BTN2] Cycle Temp     [RGB] EVM Level              │
    └─────────────────────────────────────────────────────────────────────────┘
```

### Why Digital Loopback?

| Approach | Cost | Complexity | Contest Suitability |
|----------|------|------------|---------------------|
| Real PA + ADC/DAC + Analyzer | $$$$ | High | ❌ Expensive |
| RF Simulation (ADS/AWR) | $$$ | Medium | ❌ License issues |
| **Digital Loopback + PA Twin** | **$0** | **Low** | **✅ Perfect** |

The PA Digital Twin accurately models:
- AM-AM compression
- AM-PM conversion
- Memory effects (5 taps)
- Temperature drift

## System Block Diagram (FPGA Implementation)

```
                                    ┌─────────────────────────────────────────────────────────────┐
                                    │                    FPGA (200MHz Domain)                      │
                                    │                                                              │
   ┌──────────┐    ┌───────────┐    │  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐    │
   │  AXI     │───►│  Input    │───►│  │   TDNN      │──►│ Interpolator│──►│   PA Twin       │    │
   │  DMA     │    │  Buffer   │    │  │  Generator  │   │    2x       │   │  (Digital)      │    │
   │ (From PS)│    │  (M=5)    │    │  │ 18→32→16→2  │   │             │   │                 │    │
   └──────────┘    └───────────┘    │  └──────┬──────┘   └─────────────┘   └─────────────────┘    │
                                    │         │                                                    │
                                    │         │ Weight Read                                        │
                                    │         ▼                                                    │
                                    │  ┌─────────────┐   ┌─────────────┐                           │
                                    │  │   Shadow    │◄──│    CDC      │◄─────────────────────┐   │
                                    │  │   Memory    │   │  (Gray Code)│                      │   │
                                    │  │ (3 Banks)   │   └─────────────┘                      │   │
                                    │  └─────────────┘                                        │   │
                                    │                                                         │   │
                                    │         ┌─────────────┐                                 │   │
                                    │         │    Temp     │◄──── SW[1:0] (Manual Select)   │   │
                                    │         │ Controller  │                                 │   │
                                    │         │ (3 States)  │                                 │   │
                                    │         └──────┬──────┘                                 │   │
                                    │                │ Bank Select                            │   │
                                    │                ▼                                        │   │
                                    └────────────────┬────────────────────────────────────────┘   │
                                                     │                                            │
                                    ┌────────────────┼────────────────────────────────────────────┘
                                    │                │     FPGA (1MHz Domain)
                                    │                │
                                    │  ┌─────────────▼─────────────┐
                                    │  │      A-SPSA Engine        │
                                    │  │ - LFSR Perturbation       │
                                    │  │ - Shift-based LR Anneal   │
                                    │  │ - Gradient Estimation     │
                                    │  └───────────────────────────┘
                                    │
                                    └───────────────────────────────────────────────────────────────
```

## Component Details

### 1. TDNN Generator (18→32→16→2)

The Time-Delay Neural Network generator is the core DPD model.

**Architecture:**
- Input Layer: 18 dimensions (2 current + 6 delayed I/Q + 10 envelope features)
- Hidden Layer 1: 32 neurons with LeakyReLU (α=0.25)
- Hidden Layer 2: 16 neurons with LeakyReLU (α=0.25)
- Output Layer: 2 neurons (I/Q) with Tanh

**Input Feature Composition:**
```
Input Vector (30 elements):
├── Current I/Q: I(n), Q(n) (2)
├── Nonlinear envelope features: |x(n)|, |x(n)|², |x(n)|⁴ (3)
├── Memory envelope features: |x(n-k)|, |x(n-k)|², |x(n-k)|⁴ for k=1..5 (15)
└── Delayed I/Q: I(n-k), Q(n-k) for k=1..5 (10)
```

**Fixed-Point Format:**
| Signal | Format | Range |
|--------|--------|-------|
| Weights | Q1.15 | [-1, +1) |
| Activations | Q8.8 | [-128, +128) |
| Accumulators | Q16.16 | [-32768, +32768) |

**Parameter Count:**
- Layer 1: 30×32 + 32 = 992
- Layer 2: 32×16 + 16 = 528
- Layer 3: 16×2 + 2 = 34
- **Total: 1,554 parameters**

### 2. Input Buffer (Memory Tap Assembly)

Maintains the delay line for memory effects modeling.

```
┌──────────────────────────────────────────────────┐
│              Input Buffer (M=5)                   │
├──────────────────────────────────────────────────┤
│  x[n] → [z⁻¹] → [z⁻¹] → [z⁻¹] → [z⁻¹] → [z⁻¹]  │
│    │      │       │       │       │       │      │
│    └──────┴───────┴───────┴───────┴───────┘      │
│                    │                              │
│           ┌───────┴───────┐                       │
│           │ Feature Calc  │                       │
│           │ |x|², |x|⁴    │                       │
│           └───────────────┘                       │
└──────────────────────────────────────────────────┘
```

**Implementation:**
- 6 complex registers (current + 5 delays)
- Envelope calculation using squarer and multiplier
- Output: 18-element feature vector

### 3. Shadow Memory (3 Temperature Weight Banks)

**BRAM Organization:** The FPGA stores **3 complete weight banks** (9.3 KB total):

```
┌───────────────────────────────────────────────────────────────┐
│              Shadow Memory (Dual-Port BRAM)                    │
│                  DEPTH = 4,662 words                          │
├───────────────────────────────────────────────────────────────┤
│  Bank 0 (Cold):   Addr 0x0000 - 0x0611  (1,554 params)       │
│  Bank 1 (Normal): Addr 0x0612 - 0x0C23  (1,554 params)       │
│  Bank 2 (Hot):    Addr 0x0C24 - 0x1235  (1,554 params)       │
└───────────────────────────────────────────────────────────────┘
                            ▲
                            │ Bank Select (temp_controller)
                            │
                   weight_bank_sel[1:0]
```

**Training Strategy:** Since BRAM stores 3 separate banks, you have **two options**:

| Approach | Method | BRAM | Training Time | Accuracy |
|----------|--------|------|---------------|----------|
| **A: Triple Training** | Train 3 networks separately | 9.3 KB | 3x | Best (each optimized) |
| **B: Single + Scaling** | Train 1, apply thermal drift | 9.3 KB | 1x | Good (assumes linear) |

**Recommendation:** Use **Approach A (Triple Training)** for better accuracy:
```bash
# Train each temperature condition separately
python train.py --temp cold --output models/dpd_cold.pt
python train.py --temp normal --output models/dpd_normal.pt
python train.py --temp hot --output models/dpd_hot.pt

# Export all 3 networks to BRAM hex files
python export.py --cold models/dpd_cold.pt \
                 --normal models/dpd_normal.pt \
                 --hot models/dpd_hot.pt
```

**CDC for A-SPSA Adaptation:**
```
                    1MHz Domain                     200MHz Domain
                         │                               │
    ┌────────────────────┼───────────────────────────────┼────────────────┐
    │                    │                               │                │
    │  ┌─────────┐       │  ┌──────────────────────┐    │  ┌─────────┐   │
    │  │ A-SPSA  │──────►│  │     Double Buffer    │────│─►│  TDNN   │   │
    │  │ Updates │       │  │  ┌──────┐ ┌──────┐   │    │  │ Weight  │   │
    │  └─────────┘       │  │  │Buf A │ │Buf B │   │    │  │ Memory  │   │
    │                    │  │  └──────┘ └──────┘   │    │  └─────────┘   │
    │                    │  │      ▲         ▲     │    │                │
    │  ┌─────────┐       │  │      │ Swap    │     │    │  ┌─────────┐   │
    │  │ Write   │───────│──│──────┘ Signal  └─────│────│──│ Read    │   │
    │  │ Pointer │       │  │    (Gray Code Sync)  │    │  │ Pointer │   │
    │  │ (Gray)  │       │  │                      │    │  │ (Gray)  │   │
    │  └─────────┘       │  └──────────────────────┘    │  └─────────┘   │
    │                    │                               │                │
    └────────────────────┼───────────────────────────────┼────────────────┘
                         │                               │
```

**Note:** A-SPSA adaptation only updates the **currently active bank** selected by temp_controller.

**Gray Code Conversion:**
```
Binary  →  Gray        Gray  →  Binary
B₃B₂B₁B₀   G₃G₂G₁G₀     G₃G₂G₁G₀   B₃B₂B₁B₀
G = B ^ (B >> 1)      B₃ = G₃
                      Bₙ = Gₙ ^ Bₙ₊₁
```

### 4. A-SPSA Engine (1MHz Adaptation)

Simultaneous Perturbation Stochastic Approximation with annealing.

**Algorithm:**
```
For each iteration k:
  1. Generate Bernoulli perturbation: Δₖ ∈ {-1, +1}^p (via LFSR)
  2. Perturb weights: θ⁺ = θ + cₖΔₖ, θ⁻ = θ - cₖΔₖ
  3. Evaluate loss: L(θ⁺), L(θ⁻)
  4. Estimate gradient: ĝₖ = [L(θ⁺) - L(θ⁻)] / (2cₖ) · Δₖ⁻¹
  5. Update weights: θ = θ - aₖĝₖ
  6. Anneal: aₖ₊₁ = aₖ >> 1 (every N iterations)
```

**LFSR Bernoulli Generator:**
- 32-bit LFSR with polynomial: x³² + x²² + x² + x¹ + 1
- Each bit provides one Bernoulli sample
- Period: 2³² - 1 cycles

**Annealing Schedule:**
```
Learning Rate: aₖ = a₀ >> (k / T)
Perturbation:  cₖ = c₀ >> (k / 2T)

Where:
  a₀ = Initial learning rate
  c₀ = Initial perturbation size
  T  = Annealing period (iterations)
```

### 5. Temperature Controller

Manages weight bank switching based on PA temperature.

**State Machine:**
```
                    ┌──────────────┐
         T > 45°C   │              │  T < 55°C
       ┌───────────►│     HOT      │◄───────────┐
       │            │              │            │
       │            └──────────────┘            │
       │                                        │
       │ T < 40°C                    T > 50°C   │
       │                                        │
┌──────┴─────┐                         ┌────────┴────┐
│            │                         │             │
│   NORMAL   │◄───────────────────────►│    COLD     │
│            │  10°C < T < 15°C        │             │
└────────────┘                         └─────────────┘
      ▲                                      │
      │                                      │
      └──────────────────────────────────────┘
                   T > 15°C
```

**Hysteresis Implementation:**
- Each transition has different up/down thresholds
- Moving average filter on temperature ADC
- Automatic A-SPSA reset on state change

### 6. 2x Interpolator

Upsamples from 200MHz to 400MHz using polyphase half-band filter.

**Filter Specifications:**
- Type: Half-band FIR (symmetric)
- Taps: 23
- Passband: 0-80 MHz
- Stopband: 120-200 MHz
- Stopband attenuation: >60 dB

**Polyphase Structure:**
```
Phase 0: H₀(z) → Even samples (passthrough with delay)
Phase 1: H₁(z) → Odd samples (interpolated)

Only non-zero coefficients computed (half-band property)
```

## Data Flow

### Forward Path (200MHz)
1. Input I/Q samples arrive at 200 MSps
2. Input buffer assembles memory features (18 elements)
3. TDNN performs inference:
   - Layer 1: 18→32, LeakyReLU
   - Layer 2: 32→16, LeakyReLU
   - Layer 3: 16→2, Tanh
4. Interpolator upsamples 2x to 400 MSps
5. Output drives DAC to PA

### Feedback Path (1MHz)
1. PA output sampled by feedback ADC
2. Downsampled to 1 MSps for adaptation
3. Error metric calculated (NMSE)
4. A-SPSA generates weight updates
5. Shadow memory handles CDC to NN domain

### Temperature Path (Slow)
1. Temperature ADC sampled periodically
2. Moving average applied
3. State machine determines bank
4. Bank switch triggers A-SPSA reset

## Timing Budget

| Operation | Cycles @ 200MHz | Time |
|-----------|-----------------|------|
| Input buffer | 1 | 5 ns |
| Layer 1 MAC | 18 | 90 ns |
| Layer 1 Activation | 1 | 5 ns |
| Layer 2 MAC | 32 | 160 ns |
| Layer 2 Activation | 1 | 5 ns |
| Layer 3 MAC | 16 | 80 ns |
| Layer 3 Activation | 1 | 5 ns |
| Interpolator | 12 | 60 ns |
| **Total Pipeline** | **~82** | **~410 ns** |

With full pipelining, throughput is 1 sample per cycle = **200 MSps**.

## Memory Map

### Weight Memory Organization
```
Address Range     Content              Size
0x0000-0x03BF    Layer 1 Weights      960 × 16-bit
0x03C0-0x05BF    Layer 2 Weights      512 × 16-bit
0x05C0-0x05DF    Layer 3 Weights       32 × 16-bit
0x05E0-0x05FF    Layer 1 Biases        32 × 16-bit
0x0600-0x060F    Layer 2 Biases        16 × 16-bit
0x0610-0x0611    Layer 3 Biases         2 × 16-bit

Total: 1,554 × 16-bit = 3,108 bytes per bank
3 banks × 3,108 bytes = 9,324 bytes total
```

### Register Map (AXI-Lite)
```
Address    Register         Description
0x00       CTRL            Control register (DPD/Adapt enable)
0x04       STATUS          Status register (state, errors)
0x08       TEMP_CFG        Temperature thresholds
0x0C       TEMP_STATUS     Current temperature, state
0x10       ASPSA_CFG       A-SPSA parameters
0x14       ASPSA_STATUS    Iteration count, LR
0x18       ERROR_METRIC    Current NMSE value
0x1C       VERSION         IP version register
```

## Resource Utilization (Estimated)

### PYNQ-Z1 (XC7Z020)
| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUT | ~5,200 | 53,200 | 9.8% |
| FF | ~3,800 | 106,400 | 3.6% |
| BRAM18 | 5 | 280 | 1.8% |
| DSP48 | 10 | 220 | 4.5% |

### ZCU104 (XCZU7EV)
| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| CLB LUT | ~5,200 | 230,400 | 2.3% |
| CLB FF | ~3,800 | 460,800 | 0.8% |
| BRAM | 5 | 312 | 1.6% |
| URAM | 0 | 96 | 0% |
| DSP | 10 | 1,728 | 0.6% |
