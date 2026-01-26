## Diagnosis: Why ACLR is Stuck at -35 dB Instead of -45 dB

Based on my analysis of your codebase, I've identified **5 root causes** and their interactions:

---

### **Problem 1: Sample-by-Sample Training vs. Sequence-Based (CRITICAL)**

**Your Implementation:**
```python
# create_dpd_dataset returns individual samples
inputs = np.zeros((num_samples, memory_depth + 1, 2), dtype=np.float32)  # [N, M+1, 2]
targets = np.zeros((num_samples, 2), dtype=np.float32)                   # [N, 2]
```

**OpenDPD expects:**
```python
# [batch, frame_length, 2] where frame_length = nperseg (2560)
# input_signal: 2D array [batch, time_sequence]
```

**Why this matters for ACLR:**
- ACLR is computed via Welch PSD with `nperseg=2560`
- Your batch only contains `M+1=4` samples per input
- Spectral metrics need **long coherent sequences** to resolve adjacent channel power
- You're effectively training on **noise-like** data from the frequency domain perspective

**Evidence:** OpenDPD uses `frame_length = nperseg = 2560` samples per training sequence (see metrics.py).

---

### **Problem 2: Loss Function Doesn't Directly Optimize ACLR**

**Your SpectralLoss:**
```python
# From spectral_loss.py line 456-494
total = (
    self.l1_weight * l1 +           # L1 reconstruction
    self.power_weight * power_loss + # Power regularization  
    self.nmse_weight * nmse_loss     # NMSE (time-domain)
)
```

**Issue:** None of these terms directly penalize adjacent channel power:
- L1/NMSE = time-domain error → doesn't isolate spectral regrowth
- Power regularization = total power → doesn't differentiate in-band vs. out-of-band

**OpenDPD's approach:** They use MSE only (no spectral loss during training) and achieve -59.9 dBc ACLR because they train on **long sequences** where MSE implicitly minimizes spectral content.

**Why your spectral loss doesn't help:**
- `compute_acpr()` in training uses short batches → high variance, bad gradients
- ACLR computed at validation on aggregated sequences ≠ what you're optimizing during training

---

### **Problem 3: PA Model Quality (-20dB vs -30dB)**

If your PA surrogate model has -20dB NMSE instead of OpenDPD's -39.6dB:

**Mathematical consequence:**
$$\text{ACLR}_{\text{DPD}} \approx \text{ACLR}_{\text{PA}} + \text{NMSE}_{\text{PA\_model}}$$

A PA model with -20dB NMSE introduces ~1% error per sample. Over 2560 samples (Welch window), this accumulates to significant spectral spreading into adjacent channels.

**First-principles reasoning:**
- DPD cascaded through inaccurate PA → linearization residual spreads into OOB
- You can't linearize better than your PA model predicts

---

### **Problem 4: Batch Size Trade-off (32/64 vs 256)**

| Batch Size | Gradient Variance | Sequence Context | Memory |
|------------|------------------|------------------|--------|
| 32 | High | None (independent samples) | Low |
| 256 | Lower | Still none | Higher |

**The real problem isn't batch size—it's that each sample is independent.**

OpenDPD's batch of 64 contains `64 × 2560 = 163,840` total samples with temporal coherence within each sequence. Your batch of 64 contains only `64 × 1 = 64` unrelated point estimates.

---

### **Problem 5: ILA vs FLA Training Paradigm**

**ILA (your notebook):**
```
y_PA (distorted) → DPD → û_PA (approximation of u_PA)
Loss: ||û_PA - u_PA||
```

**FLA (train_fla.py):**
```
u_PA → DPD → û_DPD → PA_frozen → ŷ_cas
Loss: ||ŷ_cas - G·u_PA||  (linearized output)
```

**Why FLA should be better for ACLR:**
- FLA optimizes the **cascaded output spectrum**, not just inverse mapping
- ACLR is a property of `ŷ_cas`, not `û_PA`
- ILA assumes perfect PA model inversion → doesn't exist for wideband signals

---

## Recommended Fixes (Priority Order)

### **1. Fix Data Pipeline: Sequence-Based Training**

```python
def create_dpd_dataset_sequence(u_pa, y_pa, seq_len=2560, stride=1280):
    """Create dataset with long sequences for proper spectral learning."""
    num_seqs = (len(y_pa) - seq_len) // stride
    
    # [num_seqs, seq_len, 2]
    inputs = np.zeros((num_seqs, seq_len, 2), dtype=np.float32)
    targets = np.zeros((num_seqs, seq_len, 2), dtype=np.float32)
    
    for i in range(num_seqs):
        start = i * stride
        inputs[i, :, 0] = y_pa[start:start+seq_len].real
        inputs[i, :, 1] = y_pa[start:start+seq_len].imag
        targets[i, :, 0] = u_pa[start:start+seq_len].real
        targets[i, :, 1] = u_pa[start:start+seq_len].imag
    
    return TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
```

**Requires:** Modify `PNTDNNGenerator` to process sequences, not single samples.

---

### **2. Add Differentiable ACLR Loss**

```python
def compute_aclr_differentiable(signal, fs, bw_main, nperseg):
    """PyTorch-differentiable ACLR for training."""
    # FFT-based (not Welch - Welch isn't differentiable)
    spectrum = torch.fft.fft(signal, dim=-1)
    psd = (spectrum.abs() ** 2) / signal.shape[-1]
    
    freq = torch.fft.fftfreq(signal.shape[-1], d=1/fs)
    
    main_mask = torch.abs(freq) <= bw_main / 2
    adj_mask = (torch.abs(freq) > bw_main/2) & (torch.abs(freq) <= bw_main)
    
    main_power = (psd * main_mask).sum(dim=-1)
    adj_power = (psd * adj_mask).sum(dim=-1)
    
    aclr_linear = adj_power / (main_power + 1e-10)
    return aclr_linear  # Minimize this during training
```

---

### **3. Improve PA Model First**

Train PA model to -30dB+ NMSE before DPD training. Check:
- Is your PA model trained on same data distribution as DPD?
- Are you using the correct DGRU feature extraction (6 features: I, Q, |z|, |z|³, sin, cos)?

---

### **4. Use FLA with Sequence-Based Loss**

Modify train_fla.py to:
1. Load sequences of length `nperseg=2560`
2. Compute loss on cascaded output spectrum
3. Use ACLR as direct loss term

---

## Where to Learn More

| Concept | Resource |
|---------|----------|
| Why sequence length matters for PSD | Welch's method derivation (Oppenheim & Schafer Ch. 10) |
| FLA vs ILA trade-offs | OpenDPDv2 paper (Wu et al., 2025, Section III) |
| WGAN-GP for regression | Gulrajani et al., "Improved Training of WGANs" (2017) |
| PA memory effects | Ding et al., "A Robust Digital Baseband Predistorter" (2004) |

---

**Bottom line:** Your training loop sees each sample in isolation with no spectral context. ACLR is fundamentally a property of **sequences**, not individual IQ points. Fix the data pipeline first—the other issues are secondary.