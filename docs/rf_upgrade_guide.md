# 🔧 RF Demo Upgrade Guide

## From Digital Loopback to Real RF System

This guide explains how to upgrade the contest demo to a full RF measurement setup like OpenDPDv2.

---

## 📊 Demo Levels

| Level | Equipment | Cost | Purpose |
|-------|-----------|------|---------|
| **Level 0** | FPGA + Laptop + HDMI | ~$200 | Contest Demo |
| **Level 1** | + USB SDR (RTL-SDR) | ~$250 | Basic RF Test |
| **Level 2** | + Eval Board ADC/DAC | ~$500 | Accurate Loopback |
| **Level 3** | + Real PA + Attenuators | ~$1000 | Real DPD Test |
| **Level 4** | + Vector Signal Analyzer | ~$5000+ | Publication Quality |

---

## Level 0: Digital Loopback (Contest Demo)

**This is what we use for the LSI Design Contest.**

```
┌─────────────┐      HDMI        ┌─────────────┐      HDMI       ┌─────────────┐
│   Laptop    │ ───────────────► │  PYNQ-Z1    │ ──────────────► │   Monitor   │
│  (TX I/Q)   │                  │   (FPGA)    │                 │  (Display)  │
└─────────────┘                  └─────────────┘                 └─────────────┘
                                       │
                                       │ PA Digital Twin
                                       │ (Software Model)
                                       ▼
                                 ┌─────────────┐
                                 │  DPD + PA   │
                                 │  Simulation │
                                 └─────────────┘
```

### Components
- **PYNQ-Z1**: $229 (academic) / $299 (commercial)
- **HDMI Cable**: ~$10
- **Laptop**: Already have

### What it demonstrates
- ✅ TDNN inference at 200MHz
- ✅ A-SPSA online adaptation
- ✅ Temperature bank switching
- ✅ Real-time EVM/ACPR display
- ✅ CDC between clock domains

### Limitations
- ❌ No actual RF signals
- ❌ PA model is simplified
- ❌ No real temperature effects

---

## Level 1: USB SDR Loopback

Add a cheap SDR for real RF signals (low bandwidth).

```
┌─────────────┐                  ┌─────────────┐                 ┌─────────────┐
│   Laptop    │ ───── USB ─────► │  RTL-SDR    │ ◄── Coax ──────│  HackRF/    │
│  (Control)  │                  │   (RX)      │    (Loopback)  │  PlutoSDR   │
└─────────────┘                  └─────────────┘                 └─────────────┘
                                                                       │
                                                                       │ USB
                                                                       ▼
                                                                 ┌─────────────┐
                                                                 │   Laptop    │
                                                                 │    (TX)     │
                                                                 └─────────────┘
```

### Additional Components
- **RTL-SDR**: ~$25 (receive only)
- **HackRF One**: ~$300 (TX/RX)
- **PlutoSDR**: ~$150 (TX/RX, better for DPD)
- **SMA cables & attenuators**: ~$30

### Bandwidth Limitations
| SDR | Bandwidth | Suitable for |
|-----|-----------|--------------|
| RTL-SDR | 2.4 MHz | Narrowband test |
| HackRF | 20 MHz | LTE-like signals |
| PlutoSDR | 20 MHz | Good balance |
| USRP B200 | 56 MHz | Wideband (expensive) |

### What it adds
- ✅ Real RF signal path
- ✅ True ADC/DAC effects
- ✅ Can test with real PA later

---

## Level 2: FPGA ADC/DAC Integration

Connect high-speed ADC/DAC to FPGA via FMC.

```
┌─────────────┐    AXI     ┌─────────────┐    FMC      ┌─────────────┐
│   Laptop    │ ◄────────► │   ZCU104    │ ◄─────────► │  AD-FMCDAQ2 │
│  (Control)  │  Ethernet  │   (FPGA)    │             │  ADC + DAC  │
└─────────────┘            └─────────────┘             └─────────────┘
                                                              │
                                                         SMA (RF)
                                                              │
                                                              ▼
                                                        ┌───────────┐
                                                        │ Loopback  │
                                                        │ or PA     │
                                                        └───────────┘
```

### Recommended FMC Cards
| Card | ADC | DAC | Bandwidth | Price |
|------|-----|-----|-----------|-------|
| AD-FMCDAQ2 | 1 GSPS 14-bit | 2.5 GSPS 16-bit | 500 MHz | ~$1000 |
| AD-FMCOMMS3 | 12-bit 61.44 MSPS | Same | 30 MHz | ~$600 |
| AD9361 | 12-bit 61.44 MSPS | Same | 56 MHz | ~$400 |

### ZCU104 Upgrade
- **ZCU104**: ~$1500
- **FMC ADC/DAC**: ~$500-1000
- Better for production deployment

---

## Level 3: Real PA Testing

Add an actual GaN PA for true DPD validation.

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│    FPGA     │ ──DAC──►│     PA      │ ──────► │   Load/     │
│    DPD      │         │   (GaN)     │         │  Antenna    │
└─────────────┘         └─────────────┘         └─────────────┘
       ▲                       │
       │                       │ Coupler (-20dB)
       │                       ▼
       │                ┌─────────────┐
       └────── ADC ◄────│ Attenuator  │
                        │   (-30dB)   │
                        └─────────────┘
```

### Components for PA Testing
| Component | Purpose | Price |
|-----------|---------|-------|
| GaN PA Eval Board | DUT | $200-500 |
| Directional Coupler | Sample output | $50-100 |
| Attenuators (30-40dB) | Protect ADC | $30-50 |
| RF Cables (SMA) | Connections | $50 |
| DC Power Supply | PA bias | $100-200 |
| Heat Sink / Fan | PA cooling | $20-50 |

### Recommended GaN PA Eval Boards
| Part | Freq | Power | Price |
|------|------|-------|-------|
| **CGD15SG00D2** | DC-6 GHz | 15W | ~$200 |
| **TGA2594** | 2-6 GHz | 10W | ~$300 |
| **CGHV14250** | DC-4 GHz | 250W | ~$500 |

### Safety Considerations
⚠️ **WARNING**: High-power RF can cause burns and equipment damage!

1. Always use proper attenuators
2. Never operate PA without load
3. Start with low input power
4. Monitor PA temperature
5. Use safety goggles for high power

---

## Level 4: Lab-Grade Measurement

Full characterization setup for publications.

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   Vector    │ ──────► │     PA      │ ──────► │   Vector    │
│   Signal    │         │   (DUT)     │         │   Signal    │
│  Generator  │         │             │         │  Analyzer   │
└─────────────┘         └─────────────┘         └─────────────┘
       │                                               │
       │              ┌─────────────┐                  │
       └───────────── │    FPGA     │ ◄────────────────┘
           Trigger    │    DPD      │    Feedback
                      └─────────────┘
```

### Required Equipment
| Equipment | Purpose | Price |
|-----------|---------|-------|
| Vector Signal Generator | Generate test signals | $5,000-50,000 |
| Vector Signal Analyzer | Measure EVM, ACPR | $10,000-100,000 |
| Spectrum Analyzer | Basic spectral view | $2,000-20,000 |
| Power Meter | Calibrate levels | $500-2,000 |
| Network Analyzer | S-parameters | $5,000-50,000 |

### Alternative: Rent Equipment
- Many universities have RF labs
- Equipment rental services (~$500/week)
- Collaborate with companies that have equipment

---

## OpenDPDv2 Reference Setup

The OpenDPDv2 project uses:

```
┌─────────────────────────────────────────────────────────────────┐
│                      OpenDPDv2 Setup                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PC (MATLAB)                                                     │
│      │                                                           │
│      ▼                                                           │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐             │
│  │  Keysight  │───►│  Mini-     │───►│  Keysight  │             │
│  │  M9381A    │    │  Circuits  │    │  N9030A    │             │
│  │  VSG       │    │  ZHL-4240  │    │  PXA       │             │
│  └────────────┘    │  (PA)      │    └────────────┘             │
│                    └────────────┘                                │
│                                                                  │
│  Total Cost: ~$150,000+                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### OpenDPDv2 Key Specs
- **VSG**: Keysight M9381A (up to 6 GHz, 160 MHz BW)
- **PA**: Mini-Circuits ZHL-4240+ (10-4200 MHz, 1W)
- **VSA**: Keysight N9030A PXA (up to 26.5 GHz)
- **Interface**: LAN/GPIB to MATLAB

### Our Equivalent (Budget Version)
| OpenDPDv2 | Our Replacement | Cost Saving |
|-----------|-----------------|-------------|
| VSG ($30k) | FPGA DAC + SDR | $29k |
| PA ($200) | Same or eval board | $0 |
| VSA ($50k) | FPGA ADC + Python | $49k |
| MATLAB | Python (free) | $2k/year |

---

## Migration Path

### Step 1: Validate with Digital Loopback
```bash
# Run simulation
cd rtl && make sim_all

# Build for PYNQ
vivado -mode batch -source scripts/build_pynq.tcl

# Run demo
python demo/hdmi_demo.py
```

### Step 2: Add SDR Feedback (Optional)
```python
# Install SDR support
pip install pyrtlsdr

# Modify demo to read from SDR
import rtlsdr
sdr = rtlsdr.RtlSdr()
sdr.sample_rate = 2.4e6
sdr.center_freq = 2.4e9
samples = sdr.read_samples(1024)
```

### Step 3: Integrate FMC ADC/DAC
```tcl
# Add ADC/DAC IP to Vivado project
create_bd_cell -type ip -vlnv analog.com:user:axi_ad9361:1.0 axi_ad9361
connect_bd_intf_net [get_bd_intf_pins axi_ad9361/s_axi] \
                    [get_bd_intf_pins axi_interconnect/M00_AXI]
```

### Step 4: Connect Real PA
```python
# Calibrate feedback path
def calibrate_feedback(input_power_dbm, measured_power_dbm):
    path_loss = input_power_dbm - measured_power_dbm
    return path_loss

# Set safe operating point
MAX_INPUT_POWER = -10  # dBm (adjust for your PA)
```

---

## Recommended Upgrade Timeline

| Phase | Duration | Goal |
|-------|----------|------|
| **Contest Prep** | 2 months | Digital demo working |
| **Post-Contest** | 1 month | Add SDR feedback |
| **Publication** | 2 months | Real PA measurements |
| **Production** | 3 months | ZCU104 + FMC deployment |

---

## Resources

### Tutorials
- [OpenDPDv2 GitHub](https://github.com/ctarver/OpenDPDv2)
- [Analog Devices DPD](https://www.analog.com/en/applications/technology/digital-predistortion.html)
- [Keysight DPD App Note](https://www.keysight.com/us/en/assets/7018-02418/application-notes/5990-5800.pdf)

### Papers
- "Digital Predistortion of Power Amplifiers" - IEEE
- "FPGA Implementation of DPD" - Various conferences
- "GaN PA Linearization" - MTT-S papers

### Communities
- RF-Design subreddit
- EEVblog forums
- Analog Devices EngineerZone

---

## Summary

**For LSI Design Contest**: Level 0 (Digital Loopback) is sufficient and demonstrates all key innovations without expensive RF equipment.

**For Publication**: Level 3-4 required for credible RF measurements.

**Our Recommendation**: Win the contest first with Level 0, then pursue Level 3 for a journal paper using university equipment or collaboration.
