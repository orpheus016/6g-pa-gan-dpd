# Video Rescue Demo: FPGA-Based DPD Visualization

**Version:** 1.0  
**Last Updated:** January 12, 2026  
**Target Platform:** PYNQ-Z1 (XC7Z020) or ZCU104  
**Purpose:** Visually demonstrate PA linearization via video quality degradation/recovery

---

## 1. Concept Overview

This demo uses HDMI video as a **proxy payload** to visualize PA distortion and DPD correction. The video content is not an RF signal—it is mapped to baseband symbols, passed through a PA behavioral model, and reconstructed. PA distortion causes visible artifacts; DPD removes them.

**Key Principle:** A PA does not know what a pixel is. It only sees a complex IQ envelope. Video artifacts are a human-observable proxy for EVM degradation.

**Why This Works:**
- Same math: nonlinear memory system (AM/AM, AM/PM, memory effects)
- Same optimization target: error minimization (EVM, ACPR)
- Same adaptation problem: slow coefficient update, fast inference

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FPGA (PL)                                     │
│                                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │  HDMI RX │───▶│ Frame-to │───▶│ OFDM TX  │───▶│ PA Model │          │
│  │          │    │ Symbol   │    │ (QAM+CP) │    │ (GMP)    │          │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘          │
│                                                       │                 │
│                                                       ▼                 │
│                                              ┌──────────────┐           │
│                                              │   MUX        │           │
│                                              │ DPD ON/OFF   │           │
│                                              └───────┬──────┘           │
│                                                      │                  │
│       ┌──────────────────────────────────────────────┴───────┐          │
│       │                                                      │          │
│       ▼                                                      ▼          │
│  ┌──────────┐                                         ┌──────────┐     │
│  │ No DPD   │                                         │ PN-TDNN  │     │
│  │ (bypass) │                                         │   DPD    │     │
│  └────┬─────┘                                         └────┬─────┘     │
│       │                                                    │            │
│       └────────────────────┬───────────────────────────────┘            │
│                            ▼                                            │
│                     ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│                     │ OFDM RX  │───▶│ Symbol-  │───▶│ HDMI TX  │       │
│                     │ (Demod)  │    │ to-Frame │    │          │       │
│                     └──────────┘    └──────────┘    └──────────┘       │
│                            │                                            │
│                            ▼                                            │
│                     ┌──────────────────┐                                │
│                     │ Telemetry Unit   │                                │
│                     │ (EVM, ACPR)      │───▶ UART / OSD Overlay         │
│                     └──────────────────┘                                │
│                            │                                            │
│                            ▼                                            │
│                     ┌──────────────────┐                                │
│                     │ A-SPSA Engine    │ (1 MHz clock domain)           │
│                     │ (Adaptation)     │                                │
│                     └──────────────────┘                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Block Specifications

### 3.1 HDMI RX/TX

| Parameter        | Value                          |
|------------------|--------------------------------|
| Resolution       | 720p60 (1280×720 @ 60 Hz)      |
| Pixel Format     | RGB 8-bit per channel          |
| Pixel Clock      | 74.25 MHz                      |
| Interface        | AXI4-Stream (32-bit TDATA)     |

**IP Core:** Xilinx HDMI 1.4/2.0 RX/TX Subsystem or Digilent DVI2RGB/RGB2DVI

### 3.2 Frame-to-Symbol Mapper

**Function:** Convert RGB pixels to complex baseband symbols.

**Mapping Strategy:**
```
RGB pixel (24-bit) → 3 bytes → 12 QAM-256 symbols (2 bits/symbol × 4 symbols/byte)
                   or → 6 QAM-64 symbols (6 bits/symbol)
```

**Implementation:**
- Serial-to-parallel conversion
- Gray-coded QAM constellation mapping
- Output: I[15:0], Q[15:0] (Q1.15 format)

**Latency:** 4 cycles (pipelined)

### 3.3 OFDM TX (Baseband Generator)

| Parameter           | Value                     |
|---------------------|---------------------------|
| FFT Size            | 64 or 128 subcarriers     |
| Active Subcarriers  | 52 or 104 (data + pilot)  |
| CP Length           | 16 samples (1/4 FFT)      |
| Modulation          | QAM-64 or QAM-256         |
| Symbol Rate         | ~3.125 MSym/s             |
| Sample Rate         | 20 MSps (demo) or 250 MSps (full) |

**Implementation:**
- Xilinx FFT IP (radix-2, streaming)
- Pilot insertion (BPSK, known sequence)
- Cyclic prefix prepend

**Latency:** FFT size + CP = 80 cycles (64+16)

### 3.4 PA Behavioral Model (GMP)

**Function:** Emulate PA nonlinearity (AM/AM, AM/PM, memory effects).

**Model:** Generalized Memory Polynomial (GMP)
$$
y(n) = \sum_{k=1,3,5}^{K} \sum_{m=0}^{M} a_{km} \cdot x(n-m) \cdot |x(n-m)|^{k-1}
$$

| Parameter        | Value                     |
|------------------|---------------------------|
| Nonlinear Order  | K = 7                     |
| Memory Depth     | M = 3                     |
| Coefficients     | Pre-loaded from CSV       |
| Thermal Modes    | Cold / Normal / Hot       |

**Distortion Intensity Control:**
- `distortion_level[7:0]`: 0 = linear, 255 = severe compression
- Allows smooth demonstration of distortion effects

**Latency:** M + 1 = 4 cycles

### 3.5 PN-TDNN DPD

**Architecture:** Phase-Normalized TDNN with systolic FC layers.

| Parameter        | Value                     |
|------------------|---------------------------|
| Feature Dim      | 24 (A, A³, I_norm, Q_norm, I, Q × M=3) |
| Topology         | 24 → 32 → 16 → 2          |
| Parameters       | 1,362                     |
| Quantization     | Q1.15 weights, Q8.8 activations |
| Latency          | 81 cycles                 |
| Throughput       | II = 1 (250 MSps max)     |

**Weight Banks:** 3 sets (Cold / Normal / Hot) for thermal gain scheduling.

### 3.6 OFDM RX (Demodulator)

**Function:** Recover symbols from OFDM signal.

**Implementation:**
- CP removal
- FFT (inverse of TX)
- Channel equalization (optional, use known pilots)
- QAM demapping

**Latency:** FFT size + overhead = ~80 cycles

### 3.7 Symbol-to-Frame Mapper

**Function:** Reconstruct RGB pixels from demodulated symbols.

**Implementation:**
- Inverse of Frame-to-Symbol
- Parallel-to-serial conversion
- Pixel clock domain crossing (FIFO)

**Latency:** 4 cycles

### 3.8 Telemetry Unit

**Function:** Compute real-time EVM and ACPR for display.

**Metrics:**
| Metric | Formula | Update Rate |
|--------|---------|-------------|
| EVM (dB) | $20 \log_{10}\left(\frac{\|e\|_{rms}}{\|x\|_{rms}}\right)$ | Per OFDM symbol |
| ACPR (dB) | $10 \log_{10}\left(\frac{P_{adj}}{P_{main}}\right)$ | Per 1024 samples |

**Output:**
- 7-segment display (on-board)
- UART ASCII stream
- OSD overlay on HDMI output (optional)

### 3.9 A-SPSA Adaptation Engine

**Function:** Online weight adaptation to track PA drift.

| Parameter        | Value                     |
|------------------|---------------------------|
| Clock Domain     | 1 MHz (decoupled)         |
| Perturbation     | LFSR-based Bernoulli      |
| Learning Rate    | α = 0.001 (annealed)      |
| Deadband         | EVM < -45 dB → suspend    |
| Update Rate      | 1 kHz – 10 kHz            |

**CDC:** Double-buffered weight RAM with CDC synchronizers.

---

## 4. Clock Domains

| Domain           | Frequency   | Function                    |
|------------------|-------------|-----------------------------|
| `clk_pixel`      | 74.25 MHz   | HDMI RX/TX, frame buffers   |
| `clk_sample`     | 20–250 MHz  | OFDM, PA model, DPD         |
| `clk_adapt`      | 1 MHz       | A-SPSA, telemetry update    |
| `clk_axi`        | 100 MHz     | ARM interface, control regs |

**CDC Crossings:**
1. `clk_pixel` ↔ `clk_sample`: Async FIFO for frame/symbol data
2. `clk_sample` ↔ `clk_adapt`: Double-buffer for weights, gray-coded for telemetry
3. `clk_axi` ↔ all: Register file with CDC synchronizers

**Source:** Clifford Cummings, "Clock Domain Crossing (CDC) Design & Verification," SNUG 2008

---

## 5. PA Distortion → Video Artifact Mapping

| RF Metric        | Video Manifestation          | Why                                   |
|------------------|------------------------------|---------------------------------------|
| AM/AM            | Contrast compression         | High-amplitude symbols clipped        |
| AM/PM            | Color phase shift            | Phase error rotates constellation     |
| Memory effects   | Temporal ghosting            | ISI across symbols                    |
| EVM              | Pixel noise / block errors   | Symbol errors → wrong pixel values    |
| ACPR             | Adjacent block interference  | Spectral regrowth → cross-talk        |

This mapping is artificial but **causally consistent**—same nonlinear system, same error minimization objective.

---

## 6. Demo Features

### 6.1 User Controls

| Control            | Type        | Function                           |
|--------------------|-------------|------------------------------------|
| DPD ON/OFF         | Toggle SW   | Enable/disable predistortion       |
| Distortion Level   | Rotary/Slider | Adjust PA compression severity   |
| Thermal Mode       | 3-pos SW    | Force Cold/Normal/Hot              |
| Adaptation ON/OFF  | Toggle SW   | Enable/disable A-SPSA              |

### 6.2 Visual Indicators

| Indicator          | Location    | Function                           |
|--------------------|-------------|------------------------------------|
| EVM (dB)           | OSD / LED   | Real-time error vector magnitude   |
| ACPR (dBc)         | OSD / UART  | Adjacent channel power ratio       |
| DPD Status         | LED         | Green = ON, Red = OFF              |
| Thermal State      | LED         | Blue/Green/Red = Cold/Normal/Hot   |
| Adaptation Active  | LED         | Blinking = adapting, Solid = idle  |

### 6.3 Demo Sequence (Suggested)

1. **Baseline:** Show clean video (DPD ON, low distortion)
2. **Distortion:** Turn DPD OFF → visible artifacts appear
3. **Recovery:** Turn DPD ON → artifacts disappear
4. **Adaptation:** Force thermal jump → brief degradation → A-SPSA recovers
5. **Metrics:** Show EVM improvement (e.g., -25 dB → -45 dB)

---

## 7. Resource Estimates

| Block              | LUTs   | FFs    | DSPs  | BRAM  |
|--------------------|--------|--------|-------|-------|
| HDMI RX/TX         | 2,000  | 2,500  | 0     | 4     |
| Frame-Symbol Map   | 500    | 400    | 0     | 2     |
| OFDM TX (64-FFT)   | 1,200  | 1,500  | 12    | 4     |
| PA Model (GMP)     | 800    | 600    | 8     | 2     |
| PN-TDNN DPD        | 3,500  | 2,800  | 62    | 8     |
| OFDM RX (64-FFT)   | 1,200  | 1,500  | 12    | 4     |
| Telemetry Unit     | 1,000  | 800    | 4     | 2     |
| A-SPSA Engine      | 600    | 500    | 8     | 2     |
| Control / CDC      | 500    | 600    | 0     | 2     |
| **Total**          | **11,300** | **11,200** | **106** | **30** |

**PYNQ-Z1 (XC7Z020) Capacity:**
- LUTs: 53,200 → 21% utilization
- FFs: 106,400 → 11% utilization
- DSPs: 220 → 48% utilization
- BRAM: 140 (36Kb) → 21% utilization

**Verdict:** Fits comfortably on PYNQ-Z1.

---

## 8. Verification Hierarchy

### Stage A: Python Functional Model

- `demo/video_demo.py`: Bit-accurate reference model
- Input: Test image (Lena, color bars)
- Output: Distorted/corrected image, EVM values
- Purpose: Verify algorithm before RTL

### Stage B: RTL Simulation

- Tool: Vivado Behavioral Simulation
- Testbench: UVM environment (see [UVM.md](UVM.md "UVM.md"))
- Metrics: Latency (cycles), EVM (numeric), bit-exactness

### Stage C: Hardware Validation

- Input: HDMI test pattern generator or laptop
- Output: HDMI monitor, UART terminal
- Metrics: Visual quality, real-time EVM readout

---

## 9. What NOT to Do (Demo Killers)

1. **Do NOT claim "real Wi-Fi"** unless 802.11 compliant
2. **Do NOT mix clock domains** without proper CDC
3. **Do NOT adapt weights at video frame rate** (too fast, will oscillate)
4. **Do NOT rely only on "looks better"**—always show numeric EVM
5. **Do NOT skip telemetry**—judges want quantitative proof

---

## 10. File Structure

```
demo/
├── hdmi_demo.py          # Python reference model
├── video_demo.py         # End-to-end simulation
└── benchmark.py          # Performance metrics

rtl/
├── src/
│   ├── hdmi_rx_wrapper.v
│   ├── hdmi_tx_wrapper.v
│   ├── frame_to_symbol.v
│   ├── symbol_to_frame.v
│   ├── ofdm_tx.v
│   ├── ofdm_rx.v
│   ├── pa_digital_twin.v
│   ├── tdnn_generator.v
│   ├── spsa_engine.v
│   ├── telemetry_unit.v
│   └── top_video_demo.v
├── tb/
│   └── tb_video_demo.sv  # UVM testbench
└── constraints/
    └── pynq_z1_hdmi.xdc
```

---

## 11. References

- Cripps, *RF Power Amplifiers for Wireless Communications*, 2nd ed., Artech House, 2006
- Proakis & Salehi, *Digital Communications*, 5th ed., McGraw-Hill, 2008
- Morgan et al., "A Generalized Memory Polynomial Model for Digital Predistortion," IEEE Trans. SP, 2006
- Spall, "Implementation of the Simultaneous Perturbation Algorithm," IEEE TAC, 1998
- Haykin, *Adaptive Filter Theory*, 5th ed., Pearson, 2014
- Cummings, "Clock Domain Crossing Design & Verification," SNUG 2008
- Xilinx UG585, "Zynq-7000 SoC Technical Reference Manual"
- Xilinx PG235, "HDMI 1.4/2.0 Transmitter Subsystem"

---

## 12. Summary

This demo proves the DPD architecture works by making PA distortion **visible** and DPD correction **observable**. It is not a Wi-Fi product—it is an *argument* that the PN-TDNN + A-SPSA architecture achieves real-time, adaptive linearization. The numeric telemetry (EVM, ACPR) provides quantitative proof; the video provides intuitive proof. Both are necessary for a credible demo.

**Final Verdict:** This architecture is credible, reviewable, and defensible. It fits on PYNQ-Z1 and demonstrates all key innovations: phase-normalized TDNN, systolic II=1 inference, thermal gain scheduling, and SPSA-based online adaptation.
