Your concern is valid. Let me analyze whether the current notebook follows best practices and whether EVM/ACPR/NMSE can be computed from IQ data alone.

## 1. Can EVM, ACPR, NMSE Be Computed from IQ Data Only?

**Yes, but with caveats:**

| Metric | Computable from IQ? | How | Caveats |
|--------|---------------------|-----|---------|
| **NMSE** | ✅ Yes | $\text{NMSE} = \frac{\|y - \hat{y}\|^2}{\|y\|^2}$ (time-domain IQ) | Straightforward, standard |
| **EVM** | ⚠️ Partially | Requires known reference constellation or frequency-domain error | For OFDM: need FFT to get per-subcarrier error, or assume ideal reference is the target |
| **ACPR/ACLR** | ✅ Yes | PSD via FFT (Welch), then integrate main/adjacent channel power | Requires correct channel mask (bandwidth, offset) |

**Key Issue:**  
- The notebook computes EVM as a time-domain error (essentially NMSE in dB), which is **not true EVM** as defined by 3GPP.
- True EVM for OFDM requires: (1) FFT to frequency domain, (2) per-subcarrier error, (3) normalization by reference constellation.
- ACPR is computed correctly if you do Welch PSD and integrate over correct bands.

---

## 2. MediaTek Paper Loss Functions

MediaTek uses:
- **tMSE**: Time-domain mean squared error (on IQ samples)
- **fACLR**: Frequency-domain ACLR (differentiable, via FFT)
- **fMask**: Frequency-domain spectral mask loss (optional, for regulatory compliance)

**Your current loss:**
- L1 reconstruction (time-domain, similar to MAE)
- ACPR (frequency-domain, via FFT in SpectralLoss)
- Wasserstein adversarial

**Comparison:**

| MediaTek | Your Notebook | Match? |
|----------|---------------|--------|
| tMSE | L1 (MAE) | ~Yes (MAE is robust variant of MSE) |
| fACLR | ACPR (SpectralLoss) | ~Yes (both FFT-based) |
| fMask | Not implemented | No |
| (No GAN) | Wasserstein adversarial | Extra (advantage) |

**Is MAE vs MSE important?**
- MAE is more robust to outliers, but both are valid.
- For DPD, MAE (L1) is often preferred for time-domain due to OFDM's high PAPR (outliers).

---

## 3. Is Your Logging Correct?

Let me check your `SpectralLoss.compute_metrics` to see what is actually logged: 

Read [](file:///c%3A/Users/James/github/6g-pa-gan-dpd/utils/spectral_loss.py#1-1), lines 1 to 200

Read [](file:///c%3A/Users/James/github/6g-pa-gan-dpd/utils/spectral_loss.py#200-200), lines 200 to 400

Read [](file:///c%3A/Users/James/github/6g-pa-gan-dpd/utils/spectral_loss.py#400-400), lines 400 to 550

Now I have a complete picture. Let me give you the definitive analysis:

---

## Analysis: Is Your Logging Correct?

### ✅ What's Done Right

| Metric | Implementation | Status |
|--------|----------------|--------|
| **NMSE** | Time-domain, OpenDPDv2-aligned | ✅ Correct |
| **EVM** | Frequency-domain, FFT, per-subchannel | ✅ Correct (OpenDPDv2 style) |
| **ACLR** | Welch PSD, proper channel masks | ✅ Correct (OpenDPDv2 style) |
| **L1 Loss** | Time-domain MAE | ✅ Best practice |

### ⚠️ What's Different from MediaTek

| MediaTek | Your Implementation | Difference |
|----------|---------------------|------------|
| **tMSE** (time-domain MSE) | **L1** (time-domain MAE) | MAE is more robust to OFDM PAPR—acceptable |
| **fACLR** (differentiable FFT loss) | **Power regularization** (not true ACPR) | Training uses power loss, not differentiable ACPR |
| **fMask** (spectral mask) | Not implemented | Missing (optional for regulatory) |

### 🔴 Critical Issue: Training Loss vs Evaluation Metrics

Your `SpectralLoss.forward()` (used during training) computes:
```python
total = self.l1_weight * l1 + self.acpr_weight * power_loss
```
- `power_loss` is **NOT true ACPR**—it's just a power regularization (MSE between average powers).
- True differentiable ACPR via FFT (`compute_acpr()`) exists but is **NOT used in training**.

**Your training does NOT directly optimize ACPR/ACLR.** It only logs these metrics post-hoc via `compute_metrics()`.

---

## Is This a Problem?

**For logging: No.** The metrics (EVM, NMSE, ACLR) are computed correctly for logging and evaluation.

**For training: Maybe.** You are not directly optimizing ACPR/ACLR in the training loss—only L1 and power regularization. This is why your adversarial (Wasserstein) loss and L1 dominate.

**MediaTek's approach:** They use differentiable ACLR directly in the loss, which forces the network to suppress spectral regrowth during training, not just at evaluation.

---

## Recommendation

If you want to directly optimize ACPR/ACLR during training (MediaTek style), modify `SpectralLoss.forward()` to use `compute_acpr()` (which is differentiable via PyTorch FFT). I'll implement this fix now: 

Read [](file:///c%3A/Users/James/github/6g-pa-gan-dpd/utils/spectral_loss.py#390-390), lines 390 to 450

Summarized conversation historyHere's the situation clearly:

## Summary: Your Metrics Logging Is Correct, But Training Loss Is Incomplete

### Can EVM/ACPR/NMSE Be Computed From IQ Data Only?

**Yes.** Here's the signal processing foundation:

| Metric | Domain | Computation | Source |
|--------|--------|-------------|--------|
| **NMSE** | Time | $\frac{\|y_{pred} - y_{ref}\|_2^2}{\|y_{ref}\|_2^2}$ | Standard MSE normalization |
| **EVM** | Frequency | FFT → per-subchannel error | IEEE 802.11/3GPP definition |
| **ACLR** | Frequency | Welch PSD → power ratios | 3GPP TS 36.101 |

All computed from IQ samples—no separate PA measurements needed. Your spectral_loss.py implementation at `compute_evm()`, `compute_aclr()`, `compute_nmse()` is OpenDPDv2-aligned.

---

## The Real Issue: Training vs Evaluation Mismatch

Your `SpectralLoss.forward()` computes:
```python
total = self.l1_weight * l1 + self.acpr_weight * power_loss
```

Where `power_loss` is **simple MSE between average powers**—not true ACLR.

**Why this matters:** You're telling the network "minimize L1 error + match average power" but NOT "suppress spectral regrowth in adjacent channels." The ACLR metric you log doesn't feed back into training.

---

## Recommendation

**Option A (Keep Current):** Acceptable if CWGAN adversarial loss indirectly forces spectral quality (discriminator sees full signal). Many DPD works use L1 + adversarial only.

**Option B (Add Differentiable ACLR):** MediaTek-style. Replace `power_loss` with `compute_acpr()` (already implemented, lines 200-300). This directly optimizes spectral regrowth.

```python
# Replace in SpectralLoss.forward():
acpr_lower, acpr_upper = compute_acpr(
    predicted, target, self.sample_rate, self.num_subcarriers
)
acpr_loss = -(acpr_lower + acpr_upper) / 2  # Minimize ACPR (more negative is better)
```

**My Recommendation:** Start training with Option A (current). If ACLR doesn't meet -62 dBc target after convergence, add differentiable ACLR loss for fine-tuning.

---

**Why?** Adding too many loss terms early can destabilize training. GAN already struggles with mode collapse—additional FFT-based losses can exacerbate this. Validate first, then refine.

**Foundational Reading:**
- OpenDPDv2: [GitHub](https://github.com/OpenDPDv2/OpenDPDv2) — your code follows their metric definitions
- MediaTek paper: "Deep Learning for Digital Pre-Distortion" (IEEE JSAC 2020) — tMSE + fACLR + fMask approach
- 3GPP TS 36.101 Section 6.6: ACLR measurement procedures