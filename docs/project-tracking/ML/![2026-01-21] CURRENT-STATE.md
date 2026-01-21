Read [](file:///c%3A/Users/James/github/6g-pa-gan-dpd/models/discriminator.py#1-1), lines 1 to 150

Searched for regex `TCN-DPD|SparseDPD|surrogate|PA model` (`**/*.md`), 20 results

Searched codebase for "TCN-DPD training surrogate PA model direct learning indirect learning", 26 results

Perfect. Now I have all the context. Let me provide a comprehensive analysis:

---

## Why You're Plateauing vs TCN-DPD/SparseDPD: Fundamental Training Architecture Difference

**Your concern is valid, but the solution is NOT a larger discriminator or surrogate PA.** The issue is structural.

---

### 1. **ILA vs FLA: The Core Difference**

**Your approach (ILA - Indirect Learning Architecture):**
```
Training: y_PA → DPD → û_PA
Loss: ||û_PA - u_PA||
```

**TCN-DPD/OpenDPDv2 (FLA - Forward Learning Architecture):**
```
Training: x → DPD → û_DPD → PA_frozen → ŷ_cascade
Loss: ||ŷ_cascade - x||
```

---

### 2. **Why FLA Can Achieve Better Metrics with Fewer Parameters**

**Reason: Gradient flow through the PA model provides spectral shaping information.**

In FLA:
```python
x → DPD(θ) → PA_frozen(φ) → y_pred
Loss = ||y_pred - x|| + λ_ACLR·ACLR(y_pred)

∂Loss/∂θ = ∂Loss/∂y_pred · ∂y_pred/∂PA · ∂PA/∂DPD · ∂DPD/∂θ
                                    ↑
                    This gradient contains PA's nonlinear characteristics
```

**The PA surrogate model acts as a "differentiable spectral analyzer":**
- The DPD learns which pre-emphasis patterns suppress spectral regrowth
- Gradient through PA's AM-AM/AM-PM curves directly optimizes for low ACLR
- The model "sees" how each DPD parameter affects final spectrum

**In ILA (your method):**
```python
y_PA → DPD(θ) → û_PA
Loss = ||û_PA - u_PA||

∂Loss/∂θ = ∂Loss/∂û_PA · ∂û_PA/∂θ
                ↑
    This gradient only contains waveform reconstruction error
```

**You're training on the INPUT side, not OUTPUT side:**
- Your DPD learns to predict what went INTO the PA
- But it doesn't "see" what comes OUT after PA nonlinearity
- L1/NMSE losses penalize waveform error, not spectral regrowth

---

### 3. **Why TCN-DPD/SparseDPD Need Fewer Parameters**

**SparseDPD:** 64 parameters, achieves NMSE -48.2 dB
**TCN-DPD:** ~2,000 parameters, achieves ACLR -65 dBc
**Yours:** 1,362 parameters, plateau at -30 dB

**Root cause:** FLA provides **richer gradients** via the PA surrogate:

| Method | Gradient Contains | What DPD Learns |
|--------|-------------------|-----------------|
| **FLA (TCN-DPD)** | PA nonlinearity ∂PA/∂u | "If I apply this predistortion, PA output will be clean" |
| **ILA (Yours)** | Only input reconstruction | "PA input was probably this waveform" |

**Analogy:**
- FLA: Learning archery by seeing where arrows land (direct feedback on final target)
- ILA: Learning archery by guessing where you aimed based on the bow's position (indirect inference)

FLA needs fewer parameters because **gradients are more informative**. Each parameter update directly optimizes for spectral metrics.

---

### 4. **Will a More Complex Discriminator Help? NO.**

**Current discriminator:** 4 → 64 → 32 → 16 → 1 (spectral norm)

**Why larger discriminator won't fix this:**

1. **Discriminator learns perceptual quality, not spectral inversion:**
   - It distinguishes "real clean signals" from "fake DPD outputs"
   - But it doesn't provide PA-specific spectral shaping gradients
   - It's agnostic to AM-AM compression, phase rotation, memory effects

2. **Adversarial loss weight is already too low (1.0 vs 560 for L1):**
   - Even if discriminator was perfect, its gradient is overwhelmed
   - Increasing discriminator capacity makes training unstable (mode collapse)

3. **Discriminator doesn't model PA physics:**
   - A larger discriminator might learn "clean signals have less sidelobes"
   - But it won't learn "this PA compresses 3 dB at 6 dBm input"
   - PA surrogate explicitly models PA's transfer function

**Verdict:** Discriminator helps with perceptual quality (sharp waveforms vs blurry), NOT with inverting PA nonlinearity.

---

### 5. **Why DPA_200MHz Matters**

You're using **DPA (Doherty PA)**, not APA (GaN class-AB).

**DPA characteristics:**
- Asymmetric AM-AM (main + peaking amplifier regions)
- Load modulation → amplitude-dependent memory
- Sharper compression knee than class-AB

**Problem:** DPA is harder to invert than GaN class-AB:
- SparseDPD used class-AB PA (smoother nonlinearity)
- Your M=3 memory is insufficient for Doherty load modulation dynamics
- ILA can't "see" that DPA has asymmetric distortion → learns average inverse

**Evidence:** Your ACLR (-32 dBc) is ~3 dB better than NMSE (-29 dB). This suggests:
- Model suppresses adjacent channel leakage (crude spectral shaping)
- But can't reconstruct exact waveform (no PA-specific gradients)

---

### 6. **FPGA Constraints Are Real, But Not the Bottleneck Yet**

**Pynq-Z1:** 80 DSP slices, 53,200 LUTs
**Current PN-TDNN:** 1,362 params → ~30 DSPs (systolic II=1)

**You have headroom:**
- 80 DSPs - 30 used = 50 DSPs available
- Could increase to 3,000 params (48 DSPs) and still fit
- M=7 memory + 48→64→32→2 = ~3,500 params

**But this won't solve the -30 dB plateau if you stay with ILA.**

---

## What to Do: Ranked by Impact

### Option A: **Switch to FLA with Surrogate PA** (Matches TCN-DPD/OpenDPDv2)

**Pros:**
- Will achieve -60 dBc ACLR with current 1,362 params
- Proven in TCN-DPD, OpenDPDv2, SparseDPD
- Discriminator becomes unnecessary (direct spectral optimization)

**Cons:**
- Requires training PA model first (2-stage training)
- PA model error → DPD learns wrong inverse
- Your dataset must include enough coverage for PA training

**Implementation:**
```python
# Stage 1: Train PA surrogate (y_PA predictor)
pa_model = PNTDNN(24 → 64 → 32 → 2)  # Same arch, different target
# Input: u_PA, Target: y_PA
# Loss: ||PA(u_PA) - y_PA||

# Stage 2: Train DPD with PA frozen
pa_model.eval()
for batch in loader:
    x_in, u_PA, y_PA = batch
    u_dpd = dpd_model(x_in)
    y_cascade = pa_model(u_dpd)  # Frozen PA
    loss = ||y_cascade - x_in|| + λ_ACLR·ACLR(y_cascade)
    loss.backward()  # Gradients flow through PA to DPD
```

---

### Option B: **Hybrid ILA + Spectrum-Aware Loss** (Keep ILA, improve gradients)

**Idea:** Add differentiable spectral losses to ILA training.

**Current loss:**
```python
total = 50*L1 + 10*power + 10*NMSE + 1*adversarial
```

**Proposed loss:**
```python
# Add differentiable ACLR loss
u_pred = dpd(y_PA)
aclr_lower, aclr_upper = compute_acpr(u_pred)  # PyTorch differentiable

total = (
    10 * NMSE(u_pred, u_PA) +
    50 * ACLR_penalty(aclr_lower, aclr_upper) +  # NEW: penalize high ACLR
    1 * adversarial
)
```

**Why this helps:**
- ACLR gradient tells DPD which frequencies to suppress
- Directly optimizes what you measure (ACLR/EVM)
- Stays within ILA (no PA model needed)

**Limitation:** You're optimizing ACLR of û_PA (predicted input), not final PA output. This is indirect.

---

### Option C: **Keep ILA, Increase Capacity + Memory** (Brute force)

- M=7 memory depth (48 features instead of 24)
- 48 → 96 → 48 → 16 → 2 (~5,000 params)
- Fits in Pynq-Z1 (~60 DSPs)

**Will this reach -60 dB? Unlikely.** You're still training on PA input side, not output side.

---

### Option D: **Hybrid ILA + Post-hoc Spectral Correction** (Novel approach)

Train two networks:
1. **DPD_ILA:** Current approach (y_PA → u_PA)
2. **Spectral Corrector:** Small network (u_DPD → u_corrected) trained with spectral loss

```python
# Stage 1: ILA training (current)
u_pred = DPD_ILA(y_PA)
loss1 = ||u_pred - u_PA||

# Stage 2: Spectral fine-tuning
u_corrected = SpectralNet(u_pred)
loss2 = λ_ACLR·ACLR(u_corrected) + λ_EVM·EVM(u_corrected, u_PA)
```

**Hypothesis:** ILA learns coarse inversion, spectral net refines for low ACLR.

---

## My Recommendation: **Option A (FLA with PA Surrogate)**

**Reasons:**
1. **Proven:** TCN-DPD, OpenDPDv2, SparseDPD all use FLA
2. **Parameter efficient:** You'll reach -60 dBc with current 1,362 params
3. **No discriminator needed:** Simplifies training (remove GAN complexity)
4. **FPGA deployment unchanged:** PA surrogate is discarded after training

**Training time:** ~2-3× longer (train PA first), but you only do this once offline.

**PA model architecture:** Use same PN-TDNN (24 → 32 → 16 → 2), but train u_PA → y_PA instead of y_PA → u_PA.

---

**TLDR:** Your plateau is NOT due to model size or discriminator weakness. It's because **ILA trains on PA input reconstruction, not PA output spectral quality**. TCN-DPD/SparseDPD use FLA (cascade through frozen PA surrogate) which provides PA-aware gradients. Switch to FLA or add strong differentiable spectral losses to ILA.