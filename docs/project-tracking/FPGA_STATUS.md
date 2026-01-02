# FPGA Deployment Status
**Component:** Hardware Deployment & Demo  
**Last Updated:** January 2, 2026  
**Status:** ⏳ WAITING - Cannot Start Until RTL Synthesis Complete

---

## Current State

**FPGA deployment is BLOCKED** waiting for:
1. ML team to validate weights
2. RTL team to synthesize (after ML validation)
3. Bitstream generation
4. Hardware testing

**Progress: 0% - Nothing can be deployed yet**

---

## Target Platforms

### Proof-of-Concept: PYNQ-Z1
**Specs:**
- FPGA: Xilinx Zynq-7000 (XC7Z020-CLG400-1)
- LUTs: 53,200
- DSPs: 220
- BRAM: 140 blocks (4.9 Mb)
- ARM: Dual-core Cortex-A9 @ 650 MHz
- **Status:** ⏳ Waiting for bitstream

**Estimated utilization:**
- LUTs: ~30k / 53k (56%) ✅ Should fit
- DSPs: ~86 / 220 (39%) ✅ Should fit
- BRAM: ~20 / 140 (14%) ✅ Should fit

### Production: ZCU104
**Specs:**
- FPGA: Xilinx Zynq UltraScale+ (XCZU7EV-2FFVC1156)
- LUTs: 230,400
- DSPs: 1,728
- BRAM: 624 blocks
- ARM: Quad-core Cortex-A53 + Dual R5
- **Status:** ⏳ Future target if PYNQ-Z1 insufficient

---

## Demo Scenarios (Planned)

### Scenario 1: HDMI Video Demo (Primary)
**Concept:** Visual proof of DPD linearization without RF hardware

**How it works:**
```
┌─────────────────┐
│  Input Pattern  │  Generate test signal (colored bars)
│  Generator      │
└────────┬────────┘
         │ I/Q samples
         ▼
┌─────────────────┐
│  PA Digital     │  Simulate nonlinear distortion
│  Twin (RTL)     │  (polynomial model in FPGA)
└────────┬────────┘
         │ Distorted I/Q
         ▼
┌─────────────────┐
│  TDNN DPD       │  Predistort signal
│  (RTL)          │  (our implementation)
└────────┬────────┘
         │ Corrected I/Q
         ▼
┌─────────────────┐
│  Constellation  │  Convert I/Q to image
│  Visualizer     │  Display on HDMI
└─────────────────┘
```

**What judges will see:**
- Left side: Distorted constellation (without DPD)
- Right side: Clean constellation (with DPD)
- Live metrics: ACPR, EVM, NMSE

**Advantages:**
- No ADC/DAC needed
- No PA hardware needed
- Visual, easy to understand
- Can demonstrate thermal switching live

**Status:** ⏳ Script exists (`demo/hdmi_demo.py`) but not tested

### Scenario 2: Benchmark Comparison
**Concept:** Side-by-side comparison with baselines

**Display:**
```
┌──────────────────────────────────────┐
│  DPD Linearization Comparison        │
├──────────────┬───────┬───────┬───────┤
│ Method       │ ACPR  │ EVM   │ NMSE  │
├──────────────┼───────┼───────┼───────┤
│ No DPD       │ -25dB │ 8.5%  │ 0dB   │
│ Volterra     │ -35dB │ 3.2%  │ -15dB │
│ GMP          │ -38dB │ 2.8%  │ -18dB │
│ Our TDNN     │ -42dB │ 1.5%  │ -22dB │ ← Target
└──────────────┴───────┴───────┴───────┘
```

**Requirements:**
- Volterra baseline (ML team needs to provide)
- GMP baseline (optional)
- Our TDNN results (waiting for ML validation)

**Status:** ❌ Cannot create until ML provides baselines

### Scenario 3: Thermal Adaptation Demo
**Concept:** Show real-time weight bank switching

**Demo flow:**
1. Start at 25°C (normal weights)
2. Simulate temperature increase (force_temp_state = HOT)
3. Show A-SPSA adapting to new temperature
4. Display convergence curve live
5. Switch back to normal temperature

**What judges will see:**
- Temperature gauge (simulated)
- Active weight bank indicator (Cold/Normal/Hot)
- A-SPSA iteration counter
- NMSE convergence curve
- EVM metric improving

**Status:** ⏳ Requires A-SPSA validation first

---

## Implementation Status

### Hardware Setup
| Component | Status | Notes |
|-----------|--------|-------|
| PYNQ-Z1 board | ⏳ Need to acquire | Check lab inventory |
| HDMI monitor | ⏳ Need to test | Any 1080p display |
| Power supply | ⏳ Ready | 12V adapter |
| USB cable | ⏳ Ready | For JTAG programming |
| Ethernet cable | ⏳ Optional | For Jupyter access |

### Software Setup
| Component | Status | Next Step |
|-----------|--------|-----------|
| Vivado 2023.2 | ⏳ Installed? | Check version |
| PYNQ image | ⏳ Need to download | v3.0.1 recommended |
| Python overlay | ⏳ Not created | After bitstream ready |
| Jupyter notebooks | ⏳ Not created | Demo scripts |

### Demo Scripts
| Script | Purpose | Status |
|--------|---------|--------|
| `demo/hdmi_demo.py` | HDMI visualization | ⏳ Exists, not tested |
| `demo/video_demo.py` | Video generation | ⏳ Exists, not tested |
| `demo/benchmark.py` | Baseline comparison | ❌ Missing Volterra |

---

## Deployment Checklist (Cannot Start Yet)

### Phase 1: Pre-Deployment (Blocked)
- [ ] **ML validation complete** ← BLOCKER
- [ ] **RTL synthesis done** ← BLOCKER  
- [ ] **Bitstream generated** ← BLOCKER
- [ ] Timing report reviewed (meets 200MHz?)
- [ ] Resource utilization acceptable?

### Phase 2: Hardware Setup (Can Prepare)
- [ ] Acquire PYNQ-Z1 board
- [ ] Download PYNQ image
- [ ] Flash SD card with PYNQ
- [ ] Test board boots (LED blink test)
- [ ] Connect to network
- [ ] Access Jupyter interface

### Phase 3: Bitstream Deployment (After Synthesis)
- [ ] Copy `.bit` file to PYNQ
- [ ] Load bitstream via Jupyter
- [ ] Verify FPGA configured (status LEDs)
- [ ] Test register access from PS

### Phase 4: Functional Testing
- [ ] Load weight files to BRAM
- [ ] Send test I/Q samples
- [ ] Verify TDNN output (non-zero)
- [ ] Check latency (should be 6.3µs)
- [ ] Test weight bank switching

### Phase 5: Demo Creation
- [ ] Run HDMI demo script
- [ ] Verify constellation display
- [ ] Test live metric updates
- [ ] Record demo video
- [ ] Create presentation slides

### Phase 6: Contest Preparation
- [ ] Practice demo (no crashes!)
- [ ] Prepare backup plan (if hardware fails)
- [ ] Print poster
- [ ] Prepare technical paper

---

## Demo Video Plan

### Video Structure (3-5 minutes)
1. **Introduction (30s)**
   - Problem: PA nonlinearity in 6G systems
   - Solution: TDNN-based DPD with thermal adaptation

2. **Architecture Overview (1min)**
   - Block diagram animation
   - TDNN structure (22→32→16→2)
   - A-SPSA adaptation loop
   - 3-bank thermal switching

3. **HDMI Demo (2min)**
   - Show distorted constellation (no DPD)
   - Enable DPD, show correction
   - Display metrics improving (ACPR, EVM)
   - Demonstrate thermal switching

4. **Benchmark Comparison (1min)**
   - Bar chart: Our method vs Volterra vs GMP
   - Highlight improvements

5. **Conclusion (30s)**
   - Key contributions
   - Future work (if time)

### Video Recording Checklist
- [ ] Screen capture software installed
- [ ] Audio narration script written
- [ ] Demo runs smoothly (no bugs)
- [ ] Metrics look good (ACPR >10dB improvement)
- [ ] Backup recording (in case primary fails)

---

## Risks & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Bitstream doesn't fit | Low | High | Use ZCU104 instead |
| Timing doesn't close | Medium | High | Reduce clock to 100MHz |
| Demo crashes during presentation | Medium | Critical | Backup video + slides |
| PYNQ board not available | Low | High | Order now, 2-week lead time |
| HDMI output not working | Medium | Medium | Test early, debug HDMI IP |
| No improvement in metrics | **High** | **Critical** | **ML must validate first!** |

---

## Success Criteria

### Minimum Viable Demo (Must Have)
- [ ] Bitstream loads on PYNQ without errors
- [ ] TDNN produces non-zero outputs
- [ ] HDMI displays constellation
- [ ] Metrics update in real-time

### Good Demo (Should Have)
- [ ] ACPR improvement visible (>5 dB)
- [ ] EVM reduction visible (<3%)
- [ ] Thermal switching works
- [ ] Demo runs for >5 minutes without crash

### Excellent Demo (Nice to Have)
- [ ] ACPR improvement >10 dB
- [ ] Better than Volterra baseline
- [ ] A-SPSA converges visually
- [ ] Professional video recording

**Current Status: 0/3 tiers achievable (waiting on ML validation)**

---

## Timeline (Depends on ML)

### If ML Validates This Week (Optimistic)
```
Week 1 (Jan 2-8):   ML validation + RTL synthesis
Week 2 (Jan 9-15):  Bitstream generation + PYNQ setup
Week 3 (Jan 16-22): Demo testing + video recording
Week 4 (Jan 23-29): Polish + backup plan
Contest (Feb):      Ready to present ✅
```

### If ML Takes 2 Weeks (Realistic)
```
Week 1-2 (Jan 2-15):  ML validation + retraining (if needed)
Week 3 (Jan 16-22):   RTL synthesis + bitstream
Week 4 (Jan 23-29):   Rush PYNQ setup + basic demo
Contest (Feb):        Minimal demo, risky ⚠️
```

### If ML Takes >3 Weeks (Pessimistic)
```
Week 1-3 (Jan 2-22):  ML struggles, RTL waits
Week 4 (Jan 23-29):   No time for proper demo
Contest (Feb):        Simulation results only, no hardware ❌
```

**Recommendation: Pressure ML team to validate ASAP!**

---

## Equipment Needed (Purchase List)

| Item | Quantity | Status | Notes |
|------|----------|--------|-------|
| PYNQ-Z1 board | 1 | ⏳ Check inventory | ~$200 if need to buy |
| HDMI cable | 1 | ✅ Available | Standard cable |
| HDMI monitor | 1 | ✅ Use lab monitor | 1080p minimum |
| MicroSD card | 1 | ⏳ Need 16GB+ | For PYNQ image |
| USB cable (micro) | 1 | ✅ Available | Programming |
| Power adapter (12V) | 1 | ✅ Included with board | - |
| Ethernet cable | 1 | ✅ Optional | For network access |

**Total cost if buying new: ~$220**

---

## Next Steps (Priority Order)

### Immediate (This Week)
1. **[BLOCKED]** Wait for ML to validate weights
2. **[BLOCKED]** Wait for RTL to synthesize
3. **[Can Do]** Locate PYNQ-Z1 board (check lab)
4. **[Can Do]** Download PYNQ image (v3.0.1)

### After Bitstream Ready (Week 2-3)
5. Flash PYNQ SD card
6. Test board boots
7. Load bitstream
8. Run basic functionality test

### Demo Preparation (Week 3-4)
9. Test HDMI demo script
10. Record demo video
11. Create presentation slides
12. Practice presentation

---

## Contact

**FPGA Lead:** [Needs assignment]  
**Blocked By:** ML validation → RTL synthesis  
**Critical Path:** Bitstream generation is hard blocker

---

*FPGA deployment cannot proceed until ML validates weights and RTL synthesizes successfully.*
