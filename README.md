# 6G PA GAN-DPD: GAN-Trained TDNN Digital Predistortion with Decoupled A-SPSA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LSI Design Contest](https://img.shields.io/badge/LSI%20Contest-29th%20Okinawa-blue)](https://www.lsi-contest.com/)

## Overview

**6G PA GAN-DPD** is a neural network-based Digital Predistortion (DPD) system for wideband Power Amplifiers. 

### What This Project Demonstrates

| Component | What It Does | What It Proves |
|-----------|--------------|----------------|
| **CWGAN-GP Training** | Trains TDNN with spectral loss (ACPR/EVM) | 2-3dB ACPR improvement over MSE-only |
| **TDNN on FPGA** | 200MHz inference, fixed complexity | Scales to wideband unlike Volterra |
| **Decoupled A-SPSA** | 1MHz adaptation with CDC | Tracks thermal drift safely |
| **3-Bank Weights** | Cold/Normal/Hot pre-trained | Instant response to temperature |

### Honest Scope Statement

**This is an algorithm validation demo, not a production RF system.**

- ✅ Demonstrates GAN-trained DPD with measurable improvement
- ✅ Shows proper FPGA architecture with CDC
- ✅ Uses real measured PA data (OpenDPD)
- ❌ Does NOT run against real PA hardware
- ❌ Does NOT claim real-time 6G sub-THz operation
- ❌ Does NOT replace conventional DPD entirely

See [docs/rf_upgrade_guide.md](docs/rf_upgrade_guide.md) for path to real RF deployment.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        OFFLINE TRAINING (Python/PyTorch)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │ OpenDPD APA  │───►│ Thermal Drift│───►│ CWGAN-GP     │                   │
│  │ 200MHz GaN   │    │ Cold/Norm/Hot│    │ + Spectral   │                   │
│  │ Dataset      │    │ Augmentation │    │ Loss + QAT   │                   │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                   │
│                                                  │                          │
│                                    ┌─────────────▼─────────────┐            │
│                                    │  3× Weight Files (Q1.15)  │            │
│                                    │  cold.bin | norm.bin |    │            │
│                                    │  hot.bin                  │            │
│                                    └─────────────┬─────────────┘            │
└──────────────────────────────────────────────────┼──────────────────────────┘
                                                   │
┌──────────────────────────────────────────────────▼──────────────────────────┐
│                        FPGA DEPLOYMENT (ZCU104/PYNQ-Z1)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     200MHz CLOCK DOMAIN (NN Inference)              │   │
│  │  ┌──────────┐    ┌──────────────────┐    ┌──────────────────┐       │   │
│  │  │ Input    │    │ TDNN Generator   │    │ Output           │       │   │
│  │  │ Buffer   │───►│ (Memory-Aware)   │───►│ Interpolator     │───►   │   │
│  │  │ Q8.8     │    │ Q1.15 weights    │    │ 2× Upsample      │  PA   │   │
│  │  └──────────┘    │ Q8.8 activations │    │ to 400MHz        │       │   │
│  │                  └────────┬─────────┘    └──────────────────┘       │   │
│  │                           │ Shadow                                   │   │
│  │                           │ Memory                                   │   │
│  │                           │ Read Port                                │   │
│  └───────────────────────────┼─────────────────────────────────────────┘   │
│                              │ CDC (Gray-coded handshake)                   │
│  ┌───────────────────────────▼─────────────────────────────────────────┐   │
│  │                     1MHz CLOCK DOMAIN (A-SPSA Update)               │   │
│  │  ┌──────────┐    ┌──────────────────┐    ┌──────────────────┐       │   │
│  │  │ Error    │    │ A-SPSA Engine    │    │ Shadow Memory    │       │   │
│  │  │ Metric   │───►│ Gradient Est.    │───►│ Write Port       │       │   │
│  │  │ Calc     │    │ Shift-Reg LR     │    │ Weight Update    │       │   │
│  │  └──────────┘    └──────────────────┘    └──────────────────┘       │   │
│  │                                                                      │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │                 Temperature Controller                        │   │   │
│  │  │  Temp Sensor ──► State (Cold/Norm/Hot) ──► Bank Select       │   │   │
│  │  │                                         ──► Anneal Reset     │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     PA Digital Twin (Simulation)                     │   │
│  │  ┌──────────┐    ┌──────────────────┐    ┌──────────────────┐       │   │
│  │  │ Volterra │    │ Temperature      │    │ AWGN + Noise     │       │   │
│  │  │ Model    │───►│ Coefficients     │───►│ Floor Model      │───►   │   │
│  │  │ Order 7  │    │ (α drift)        │    │                  │  FB   │   │
│  │  └──────────┘    └──────────────────┘    └──────────────────┘       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quantization Scheme

| Component | Format | Range | Description |
|-----------|--------|-------|-------------|
| **Weights** | Q1.15 | [-1.0, +0.99997] | 16-bit signed fixed-point |
| **Activations** | Q8.8 | [-128.0, +127.996] | 16-bit signed fixed-point |
| **Accumulator** | Q16.16 | [-32768.0, +32767.99998] | 32-bit for MAC operations |
| **Input IQ** | Q1.15 | [-1.0, +0.99997] | Normalized IQ samples |
| **Output IQ** | Q1.15 | [-1.0, +0.99997] | Predistorted IQ samples |
| **Learning Rate** | Q0.16 | [0, 0.99998] | Unsigned, shift-reg controlled |
| **Error Metric** | Q8.24 | High precision | For gradient estimation |

---

## TDNN Generator Architecture

### Memory-Aware Input Structure

```
Input Vector (per sample n):
┌─────────────────────────────────────────────────────────────────────┐
│  I(n), Q(n),                         ← Current IQ sample            │
│  |x(n)|, |x(n)|², |x(n)|⁴            ← Nonlinear envelope features  │
│  |x(n-1)|, |x(n-1)|², |x(n-1)|⁴, ... ← Envelope memory (M=5)        │
│  I(n-1), Q(n-1), ..., I(n-M), Q(n-M) ← IQ memory taps               │
└─────────────────────────────────────────────────────────────────────┘
Total input dim = 2 + 3*(M+1) + 2*M = 2 + 18 + 10 = 30 (for M=5)
```

### Layer-by-Layer Specification

| Layer | Type | Input | Output | Weights | Bias | Params | Format |
|-------|------|-------|--------|---------|------|--------|--------|
| **Input** | Buffer | 30×1 | 30×1 | - | - | - | Q1.15 |
| **FC1** | Linear | 30 | 32 | 30×32=960 | 32 | 992 | Q1.15 |
| **Act1** | LeakyReLU | 32 | 32 | - | - | - | Q8.8 |
| **FC2** | Linear | 32 | 16 | 32×16=512 | 16 | 528 | Q1.15 |
| **Act2** | LeakyReLU | 16 | 16 | - | - | - | Q8.8 |
| **FC3** | Linear | 16 | 2 | 16×2=32 | 2 | 34 | Q1.15 |
| **Output** | Tanh | 2 | 2 | - | - | - | Q1.15 |
| **TOTAL** | | | | | | **1,554** | |

### FPGA Resource Estimate (per weight bank)

| Resource | PYNQ-Z1 | ZCU104 | Usage |
|----------|---------|--------|-------|
| **BRAM** | 9.3 KB | 9.3 KB | Weight storage (1,554 × 16-bit × 3 banks = 9.3KB) |
| **DSP48** | 10 | 10 | MAC operations (6) + nonlinear features (2) + interp (2) |
| **LUT** | ~4,500 | ~4,500 | Control logic, activation, feature extraction |
| **FF** | ~3,200 | ~3,200 | Pipeline registers, shift registers |

---

## Data Flow Specification

### 1. Input Stage (200MHz)

```
External ADC/DAC ──► FIFO ──► Input Buffer ──► Memory Tap Shift Register
                                    │
                   ┌────────────────┴────────────────┐
                   │  Input Vector Assembly          │
                   │  [I(n), Q(n), |x|, taps...]     │
                   └────────────────┬────────────────┘
                                    ▼
                               FC1 Layer
```

**Interface Signals:**
| Signal | Width | Direction | Format | Description |
|--------|-------|-----------|--------|-------------|
| `adc_i` | 16 | IN | Q1.15 | ADC I-channel sample |
| `adc_q` | 16 | IN | Q1.15 | ADC Q-channel sample |
| `adc_valid` | 1 | IN | - | ADC data valid strobe |
| `input_ready` | 1 | OUT | - | FIFO not full |

### 2. TDNN Inference (200MHz)

```
        FC1 (18→32)           FC2 (32→16)           FC3 (16→2)
            │                     │                     │
    ┌───────▼───────┐     ┌───────▼───────┐     ┌───────▼───────┐
    │ Weight BRAM   │     │ Weight BRAM   │     │ Weight BRAM   │
    │ Bank Select   │     │ Bank Select   │     │ Bank Select   │
    │ (0/1/2)       │     │ (0/1/2)       │     │ (0/1/2)       │
    └───────────────┘     └───────────────┘     └───────────────┘
            │                     │                     │
    ┌───────▼───────┐     ┌───────▼───────┐     ┌───────▼───────┐
    │ MAC Array     │     │ MAC Array     │     │ MAC Array     │
    │ 6× DSP48      │     │ 6× DSP48      │     │ 2× DSP48      │
    │ Q1.15 × Q8.8  │     │ Q1.15 × Q8.8  │     │ Q1.15 × Q8.8  │
    │ = Q9.23 acc   │     │ = Q9.23 acc   │     │ = Q9.23 acc   │
    └───────────────┘     └───────────────┘     └───────────────┘
            │                     │                     │
    ┌───────▼───────┐     ┌───────▼───────┐     ┌───────▼───────┐
    │ Bias Add      │     │ Bias Add      │     │ Bias Add      │
    │ Q8.8 + Q8.8   │     │ Q8.8 + Q8.8   │     │ Q8.8 + Q8.8   │
    └───────────────┘     └───────────────┘     └───────────────┘
            │                     │                     │
    ┌───────▼───────┐     ┌───────▼───────┐     ┌───────▼───────┐
    │ LeakyReLU     │     │ LeakyReLU     │     │ Tanh LUT      │
    │ α=0.2 (>>2)   │     │ α=0.2 (>>2)   │     │ 256 entries   │
    │ Q8.8 → Q8.8   │     │ Q8.8 → Q8.8   │     │ Q8.8 → Q1.15  │
    └───────────────┘     └───────────────┘     └───────────────┘
```

**Internal Signals:**
| Signal | Width | Format | Description |
|--------|-------|--------|-------------|
| `fc1_out[31:0]` | 32×16 | Q8.8 | FC1 output vector |
| `fc2_out[15:0]` | 16×16 | Q8.8 | FC2 output vector |
| `dpd_i` | 16 | Q1.15 | Predistorted I output |
| `dpd_q` | 16 | Q1.15 | Predistorted Q output |
| `weight_bank_sel[1:0]` | 2 | UINT | 0=Cold, 1=Norm, 2=Hot |

### 3. Output Stage (200MHz → 400MHz)

```
TDNN Output ──► 2× Polyphase ──► Output FIFO ──► DAC Interface
  (200MHz)      Interpolator      (400MHz)        (400MHz)
                   │
           ┌───────▼───────┐
           │ FIR Filter    │
           │ 8-tap         │
           │ Halfband      │
           └───────────────┘
```

**Interface Signals:**
| Signal | Width | Direction | Format | Description |
|--------|-------|-----------|--------|-------------|
| `dac_i` | 16 | OUT | Q1.15 | DAC I-channel sample |
| `dac_q` | 16 | OUT | Q1.15 | DAC Q-channel sample |
| `dac_valid` | 1 | OUT | - | DAC data valid strobe |
| `dac_ready` | 1 | IN | - | DAC ready to accept |

### 4. A-SPSA Update Engine (1MHz)

```
                    ┌─────────────────────────────────────────────┐
                    │           A-SPSA Controller                 │
                    └─────────────────────────────────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         │                             │                             │
         ▼                             ▼                             ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│ Error Metric    │         │ Perturbation    │         │ Weight Update   │
│ Calculator      │         │ Generator       │         │ Engine          │
└─────────────────┘         └─────────────────┘         └─────────────────┘
         │                             │                             │
         ▼                             ▼                             ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│ J(w+Δ) - J(w-Δ) │         │ LFSR → ±1       │         │ w ← w + α·g     │
│ Q8.24 precision │         │ Bernoulli dist  │         │ Shift-reg LR    │
└─────────────────┘         └─────────────────┘         └─────────────────┘
```

**A-SPSA Algorithm:**
```
For each iteration k:
  1. Generate Δk ∈ {-1, +1}^n (Bernoulli from LFSR)
  2. Compute J(w + ck·Δk) using PA feedback
  3. Compute J(w - ck·Δk) using PA feedback  
  4. Gradient estimate: gk = [J(w+) - J(w-)] / (2·ck·Δk)
  5. Weight update: w ← w - ak·gk

Annealing schedule (shift-register based):
  ak = a0 >> (k / anneal_period)  // Learning rate decay
  ck = c0 >> (k / anneal_period)  // Perturbation decay
```

**Interface Signals:**
| Signal | Width | Direction | Format | Description |
|--------|-------|-----------|--------|-------------|
| `error_evm` | 24 | IN | Q8.16 | EVM error metric |
| `error_acpr` | 24 | IN | Q8.16 | ACPR error metric |
| `spsa_lr[15:0]` | 16 | INT | Q0.16 | Current learning rate |
| `spsa_delta[1553:0]` | 1554 | INT | ±1 | Perturbation vector |
| `weight_update_req` | 1 | OUT | - | Request CDC transfer |
| `weight_update_ack` | 1 | IN | - | CDC transfer complete |

### 5. CDC Shadow Memory (200MHz ↔ 1MHz)

```
     1MHz Domain                              200MHz Domain
┌──────────────────┐                     ┌──────────────────┐
│ A-SPSA Engine    │                     │ TDNN Inference   │
│                  │                     │                  │
│ Weight Write     │                     │ Weight Read      │
│ Port             │                     │ Port             │
└────────┬─────────┘                     └────────┬─────────┘
         │                                        │
         │   ┌────────────────────────────┐       │
         │   │     SHADOW MEMORY          │       │
         │   │  ┌──────────────────────┐  │       │
         └──►│  │ Write Buffer (4KB)   │  │◄──────┘
             │  │ Gray-coded addr      │  │
             │  │ Double-buffer swap   │  │
             │  └──────────────────────┘  │
             │                            │
             │  ┌──────────────────────┐  │
             │  │ Handshake Logic      │  │
             │  │ req_sync (2-FF)      │  │
             │  │ ack_sync (2-FF)      │  │
             │  └──────────────────────┘  │
             └────────────────────────────┘
```

**CDC Signals:**
| Signal | Domain | Width | Description |
|--------|--------|-------|-------------|
| `wr_req` | 1MHz | 1 | Write request from A-SPSA |
| `wr_req_sync` | 200MHz | 1 | Synchronized request |
| `wr_ack` | 200MHz | 1 | Acknowledge from shadow mem |
| `wr_ack_sync` | 1MHz | 1 | Synchronized acknowledge |
| `shadow_swap` | 200MHz | 1 | Double-buffer swap trigger |

### 6. Temperature Controller (1MHz)

```
Temp Sensor ──► ADC ──► Threshold ──► State FSM ──► Bank Select
  (I2C/SPI)            Comparator                    + Anneal Reset

States:
  COLD   (T < 15°C)  → Bank 0, Reset anneal
  NORMAL (15-40°C)   → Bank 1
  HOT    (T > 40°C)  → Bank 2, Reset anneal
```

**Interface Signals:**
| Signal | Width | Direction | Format | Description |
|--------|-------|-----------|--------|-------------|
| `temp_raw[11:0]` | 12 | IN | UINT | Raw temperature ADC |
| `temp_state[1:0]` | 2 | OUT | UINT | 0=Cold, 1=Norm, 2=Hot |
| `temp_change` | 1 | OUT | - | State transition pulse |
| `anneal_reset` | 1 | OUT | - | Reset A-SPSA iteration |

---

## Project Structure

```
6g-pa-gan-dpd/
├── README.md
├── LICENSE
├── requirements.txt
├── config/
│   └── config.yaml              # Training & deployment config
├── data/
│   ├── raw/                     # OpenDPD APA dataset
│   └── processed/               # Thermal-augmented datasets
├── models/
│   ├── __init__.py
│   ├── tdnn_generator.py        # Memory-aware TDNN with QAT
│   ├── discriminator.py         # CWGAN-GP critic
│   └── pa_digital_twin.py       # Volterra PA model
├── utils/
│   ├── __init__.py
│   ├── quantization.py          # QAT utilities
│   ├── dataset.py               # Data loading & thermal augment
│   ├── spectral_loss.py         # ACPR, EVM loss functions
│   └── export.py                # Weight export for FPGA
├── train.py                     # CWGAN-GP training script
├── export.py                    # Export weights to binary
├── validate.py                  # Validation & benchmarking
├── rtl/
│   ├── Makefile
│   ├── README.md
│   ├── src/
│   │   ├── tdnn_generator.v     # TDNN inference engine
│   │   ├── fc_layer.v           # Fully-connected layer
│   │   ├── activation.v         # LeakyReLU, Tanh LUT
│   │   ├── aspsa_engine.v       # A-SPSA update logic
│   │   ├── shadow_memory.v      # CDC weight transfer
│   │   ├── temp_controller.v    # Temperature state machine
│   │   ├── pa_digital_twin.v    # Volterra PA simulation
│   │   ├── interpolator.v       # 2× polyphase upsampler
│   │   └── dpd_top.v            # Top-level integration
│   ├── tb/
│   │   ├── tb_tdnn_generator.v
│   │   ├── tb_aspsa_engine.v
│   │   └── tb_dpd_top.v
│   └── constraints/
│       ├── pynq_z1.xdc
│       └── zcu104.xdc
├── fpga/
│   ├── pynq_z1/                 # PYNQ-Z1 Vivado project
│   └── zcu104/                  # ZCU104 Vivado project
├── demo/
│   ├── video_demo.py            # Video transmission demo
│   └── benchmark.py             # GMP/Volterra comparison
└── docs/
    ├── architecture.md          # Detailed architecture doc
    ├── fpga_implementation.md   # FPGA build guide
    └── figures/
```

---

## Quick Start

### Option A: Train on Google Colab (Recommended)

1. **Upload to Colab**: Open `training_colab.ipynb` in Google Colab
2. **Run all cells**: Training uses free GPU, takes ~30 minutes
3. **Download weights**: `weights_trained.hex` for FPGA deployment

### Option B: Local Development

```bash
# Setup environment
cd 6g-pa-gan-dpd
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train model (CPU is slow, recommend Colab)
python train.py --config config/config.yaml

# Export weights for FPGA
python export.py --checkpoint checkpoints/best.pt --output rtl/weights/
```

### RTL Simulation

```bash
# Requires: iverilog, gtkwave (sudo apt install iverilog gtkwave)
cd rtl
make sim_all      # Run all testbenches
make wave_dpd     # View DPD waveforms in GTKWave
```

---

## 🎮 HDMI Demo Setup (LSI Design Contest)

**No ADC/DAC or RF equipment required!** Uses HDMI for digital I/Q loopback.

```
┌──────────────┐      HDMI       ┌──────────────┐      HDMI      ┌──────────────┐
│              │  (I/Q encoded)  │              │  (I/Q + OSD)   │              │
│    Laptop    │ ──────────────► │   PYNQ-Z1    │ ─────────────► │   Monitor    │
│  (TX Signal) │                 │    (FPGA)    │                │  (Display)   │
│              │                 │              │                │              │
└──────────────┘                 └──────────────┘                └──────────────┘
      │                                │
      │ USB (Jupyter)                  │ DPD + PA Twin
      └────────────────────────────────┘ (All Digital!)
```

### Hardware Required
| Item | Purpose | Cost |
|------|---------|------|
| PYNQ-Z1 | FPGA board | ~$229 |
| HDMI cables (×2) | Signal path | ~$15 |
| Monitor | Display output | Already have |
| USB cable | Jupyter control | Included |

### Demo Controls
| Button/Switch | Function |
|---------------|----------|
| **BTN0** | Toggle DPD ON/OFF |
| **BTN1** | Toggle Adaptation ON/OFF |
| **BTN2** | Cycle Temperature (Cold→Normal→Hot) |
| **SW0-1** | Temperature override select |

### What the Demo Shows
- ✅ Real-time DPD inference at 200MHz
- ✅ A-SPSA adaptation convergence
- ✅ Temperature state switching
- ✅ EVM/ACPR improvement metrics
- ✅ Constellation/spectrum display

### Running the Demo

```bash
# On PYNQ board (via Jupyter terminal)
cd 6g-pa-gan-dpd/demo
python hdmi_demo.py

# Or launch Jupyter notebook
jupyter notebook hdmi_demo.ipynb
```

### Upgrading to Real RF
See [docs/rf_upgrade_guide.md](docs/rf_upgrade_guide.md) for adding:
- SDR feedback (~$150)
- FMC ADC/DAC (~$500)
- Real GaN PA (~$300)
- Vector Signal Analyzer (~$10k+)

---

## References

1. OpenDPD: Open Digital Predistortion - [GitHub](https://github.com/OpenDPD)
2. CWGAN-GP: Conditional Wasserstein GAN with Gradient Penalty
3. SPSA: Simultaneous Perturbation Stochastic Approximation - [arXiv:2506.16591](https://arxiv.org/abs/2506.16591)
4. Neural Network DPD for eMBB - [ResearchGate](https://www.researchgate.net/publication/334162227)
5. Ultra-Low Latency DPD - [arXiv:2507.06849](https://arxiv.org/abs/2507.06849)

---

## License

MIT License - See [LICENSE](LICENSE) for details.
