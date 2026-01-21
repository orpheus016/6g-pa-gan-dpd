Read [](file:///c%3A/Users/James/github/6g-pa-gan-dpd/training_colab_v2.ipynb#1797-1797), lines 1797 to 1943

Now let me check the training output to see the plateau pattern: 

Searched for regex `Val EVM:.*dB.*NMSE:.*dB.*ACLR` (`**/training_colab_v2.ipynb`), no results

Searched for regex `Val EVM|ACLR.*dBc|NMSE.*dB` (`**/training_colab_v2.ipynb`), 20 results

Read [](file:///c%3A/Users/James/github/6g-pa-gan-dpd/training_colab_v2.ipynb#2193-2193), lines 2193 to 2193

Let me check the training loss components and model architecture: 

Searched for regex `g_spectral|g_total|g_l1|g_power` (`**/training_colab_v2.ipynb`), 19 results

Read [](file:///c%3A/Users/James/github/6g-pa-gan-dpd/training_colab_v2.ipynb#1520-1520), lines 1520 to 1570

Now let me check the SpectralLoss configuration and the model architecture to understand the plateau: 

Read [](file:///c%3A/Users/James/github/6g-pa-gan-dpd/training_colab_v2.ipynb#800-800), lines 800 to 900

Read [](file:///c%3A/Users/James/github/6g-pa-gan-dpd/training_colab_v2.ipynb#900-900), lines 900 to 1000

Now let me check the train_step to understand the loss composition: 

Read [](file:///c%3A/Users/James/github/6g-pa-gan-dpd/training_colab_v2.ipynb#990-990), lines 990 to 1110

Now let me check the SpectralLoss class implementation: 

Read [](file:///c%3A/Users/James/github/6g-pa-gan-dpd/utils/spectral_loss.py#150-150), lines 150 to 250

Read [](file:///c%3A/Users/James/github/6g-pa-gan-dpd/utils/spectral_loss.py#350-350), lines 350 to 450

Read [](file:///c%3A/Users/James/github/6g-pa-gan-dpd/utils/spectral_loss.py#450-450), lines 450 to 530

Now I have a clear picture. Let me check the model architecture: 

Searched for files matching `**/pn_tdnn*.py`, 1 match

Read [](file:///c%3A/Users/James/github/6g-pa-gan-dpd/models/pn_tdnn_generator.py#1-1), lines 1 to 150

Now let me understand the complete picture. Let me also check the training data characteristics: 

Read [](file:///c%3A/Users/James/github/6g-pa-gan-dpd/data)

Read [](file:///c%3A/Users/James/github/6g-pa-gan-dpd/data/DPA_200MHz)

Read [](file:///c%3A/Users/James/github/6g-pa-gan-dpd/data/DPA_200MHz/spec.json)

Now I have all the context. Here's my analysis:

---

## Root Cause Analysis: Why Plateau at -30 dB Instead of -60 dB

Based on the training output trajectory:

| Epoch | Val EVM | Val NMSE | Val ACLR |
|-------|---------|----------|----------|
| 1 | -12.68 dB | -9.52 dB | -16.92/-15.65 dBc |
| ~50 | -24.41 dB | -21.39 dB | -26.86/-26.46 dBc |
| ~100 | -27.46 dB | -25.02 dB | -30.63/-29.07 dBc |
| ~150 | -28.44 dB | -26.28 dB | -31.78/-29.91 dBc |
| ~200 | -29.22 dB | -27.51 dB | -32.39/-30.38 dBc |
| ~250 | -29.39 dB | -27.64 dB | plateau |

**Observation:** ACLR is ~3 dB better than NMSE. This is expected because ACLR measures adjacent channel leakage (out-of-band), while NMSE measures in-band error. The model suppresses spectral regrowth better than it reconstructs the exact waveform.

---

### 1. **Model Capacity Limitation (Primary Cause)**

Your PN-TDNN has **1,362 parameters**:
- FC1: 24×32 + 32 = 800
- FC2: 32×16 + 16 = 528
- FC3: 16×2 + 2 = 34

**Problem:** GaN PA DPD typically requires modeling:
- Memory effects (thermal, trapping) over 5-20 samples, not just M=3
- Higher-order nonlinearities (5th, 7th order) beyond A³ features
- Cross-term interactions (AM-AM, AM-PM coupling)

**Evidence:** The plateau at -30 dB suggests the model has exhausted its representational capacity. A polynomial DPD with similar complexity would also plateau around -30 to -35 dB.

**Reference:** OpenDPDv2 achieves -45 dB NMSE using ~5,000-10,000 parameters with M=7 memory depth.

---

### 2. **Memory Depth M=3 Is Too Short**

Your feature extraction uses M=3 (4 taps total: n, n-1, n-2, n-3).

**Problem:** At 800 MSps with 200 MHz bandwidth:
- One sample = 1.25 ns
- M=3 = 5 ns total memory span
- GaN thermal time constants: 10-100 μs (8,000-80,000 samples!)
- GaN trapping time constants: 0.1-10 μs (80-8,000 samples)

M=3 only captures **short-term electrical memory** (matching network, bias circuits), not the dominant thermal/trapping nonlinearities.

**Why ACLR is better:** Adjacent channel power comes from instantaneous nonlinearity (AM-AM compression), which M=3 can partially model. But in-band EVM/NMSE requires precise waveform reconstruction, which needs longer memory.

---

### 3. **Loss Function Mismatch**

Your training loss:
```python
g_total = 1.0 * g_adv + 50.0 * L1 + 10.0 * spectral
# where spectral = 50*L1 + 10*power + 10*NMSE
```

**Issue:** L1 loss is used twice (once directly, once inside spectral loss). Effective L1 weight = 50 + 50×10 = 550. This dominates everything.

**Problem with L1 dominance:**
- L1 minimizes pointwise error but doesn't penalize spectral leakage properly
- L1 gradient is constant regardless of error magnitude → poor convergence for small residuals
- At -30 dB, L1 error ≈ 0.03 (linear), but NMSE is still -30 dB → L1 can't push lower

**Better approach:** Use NMSE as primary loss (it directly optimizes what you measure).

---

### 4. **Adversarial Loss Weight Too Low**

`g_adv` weight = 1.0, but reconstruction = 550× total.

**Problem:** The discriminator provides:
- Perceptual quality (sharp waveforms vs blurry averages)
- Implicit spectral matching (discriminator sees frequency content)

At weight 1.0, the adversarial gradient is overwhelmed by L1. The GAN essentially degrades to supervised L1 training.

**Evidence:** If your discriminator loss `d_wasserstein` is near 0 throughout training, the GAN is collapsed.

---

### 5. **Training Data Quality/Diversity**

**Question:** What's the PA operating point in your dataset?
- If data only covers -10 dB to -2 dB backoff, the model can't learn saturation behavior
- If data is from a single temperature/bias, it can't generalize

Check your training data's amplitude histogram. If it's concentrated in linear region, the model won't learn compression.

---

## Recommendations (Ordered by Impact)

### A. Increase Model Capacity

```python
# Current: 24 → 32 → 16 → 2 (1,362 params)
# Proposed: 24 → 64 → 32 → 16 → 2 (~3,000 params)
hidden_dims = [64, 32, 16]
```

Or use M=7 memory depth (48 features → larger first layer needed).

### B. Increase Memory Depth

```python
memory_depth = 7  # 48 features instead of 24
# Requires adjusting hidden layers: 48 → 64 → 32 → 2
```

This captures more of the PA's memory nonlinearity.

### C. Fix Loss Weighting

```python
# Remove L1 from train_step, keep only spectral loss
g_total = (
    1.0 * g_adv +
    1.0 * spectral  # spectral already has L1, power, NMSE
)

# Inside SpectralLoss:
l1_weight = 10.0       # Reduced
power_weight = 5.0     # Reduced  
nmse_weight = 50.0     # PRIMARY: optimize directly
```

### D. Check Discriminator Health

Add logging:
```python
print(f"D_real: {d_real.mean():.4f}, D_fake: {d_fake.mean():.4f}")
```

If `|D_real - D_fake| < 0.1`, discriminator is collapsed. Consider:
- Increase `n_critic` from 5 to 10
- Reduce generator LR relative to discriminator

---

## Why ACLR > NMSE (Answering Your Specific Question)

ACLR measures **out-of-band power** relative to **in-band peak power**:
```
ACLR = P_adjacent / P_main (dBc)
```

NMSE measures **in-band error power** relative to **signal power**:
```
NMSE = ||y - ŷ||² / ||y||²
```

**The model can achieve better ACLR than NMSE because:**

1. **L1/MSE losses penalize all frequencies equally**, but ACLR only cares about spectral regrowth
2. **Phase errors hurt NMSE but don't affect ACLR** (ACLR is power-based, phase-agnostic)
3. **In-band noise floor from quantization/numerical error affects NMSE** but not ACLR

To close the gap, you need:
- Higher model capacity for precise waveform reconstruction
- Longer memory depth for phase accuracy
- NMSE-dominant loss function