## Critical Architecture Critique

### Problem 1: **THREE Different Architectures Across Files**

| Location | Input Dim | Feature Extraction | Memory Depth |
|----------|-----------|-------------------|--------------|
| **BACKUP Colab (working)** | 18 | `[I,Q] + delayed + |x|²·x` | M=5 |
| **tdnn_generator.py** | 30 | `[I,Q] + |x|,|x|²,|x|⁴ + delayed IQ` | M=5 |
| **ARCHITECTURE.md (proposed)** | **24** | `[A, A³, I_norm, Q_norm, I, Q] × 4 taps` | **M=3** |

**This is your core problem**: You're documenting one architecture, implementing another, and training a third.

---

### Problem 2: **Colab Notebooks Have Fundamental Bugs**

#### BACKUP Notebook Issues:

```python
# Line ~90: Feature extraction is WRONG
def create_input_features(self, x):
    for d in range(1, self.memory_depth + 1):
        # BUG: This creates BATCH-level delays, not SAMPLE-level
        delayed = torch.cat([torch.zeros(d, 2, device=x.device), x[:-d]], dim=0)
```

**Problem**: This treats samples as independent (batch dimension) but applies delays across the batch. You're creating:
- `x[0]` with delay = zeros
- `x[1]` with delay = x[0]

This is **sequence-to-sequence**, but your dataloader treats each sample independently. The delays are nonsensical.

#### Second Notebook Issues:

```python
# Line ~340: PA Digital Twin used INCORRECTLY in training loop
pa_output = pa_twin(dpd_output)  # Using synthetic PA!
```

**Problem**: You're using a **synthetic PA** during training but claim ILA with measured data. This is **circular**:
1. Train PADigitalTwin on measured data
2. Train DPD to invert PADigitalTwin
3. Test DPD on PADigitalTwin

You're optimizing for your own model, not the real PA.

---

### Problem 3: **Phase Normalization NOT Implemented**

Your proposed architecture (ARCHITECTURE.md) specifies:

```
I_norm(n-k) = (I_k·I_0 + Q_k·Q_0) / A_0
Q_norm(n-k) = (Q_k·I_0 - I_k·Q_0) / A_0
```

**Neither notebook implements this**. Instead:
- BACKUP: Uses `|x|²·x` (magnitude-weighted IQ)
- Second: Uses `|x|, |x|², |x|⁴` (envelope powers)

Phase normalization is **critical** for:
1. Reducing learning complexity by 40% (SparseDPD claim)
2. FPGA: Complex multiply + divide is expensive; doing it once in FEx amortizes cost

---

### Problem 4: **ILA Flow Executed Incorrectly**

Your train.py (lines 270-310) correctly describes ILA:

```python
# In ILA:
# - Input: PA output (distorted signal y_PA)
# - Generator produces: Predistorted signal (should match clean PA input u_PA)
# - Target: PA input (clean signal u_PA)
```

But your Colab notebooks do:

```python
# WRONG: Using synthetic PA in loop
fake_pa_out = pa_twin(dpd_out)  # This defeats ILA!
```

**Correct ILA** (what train.py does):
```python
# NO PA in training loop
dpd_output = generator(input_seq)  # input_seq = y_PA (measured)
loss = L1(dpd_output, target)       # target = u_PA (measured)
```

---

## Comparison: Your Implementation vs OpenDPD

| Aspect | OpenDPD | Your train.py | Your Colab |
|--------|---------|---------------|------------|
| **PA Model** | Frozen surrogate (FLA) | None (ILA) ✅ | PADigitalTwin in loop ❌ |
| **Training Data** | Measured CSV | Measured CSV ✅ | Measured but processed wrong |
| **Feature Extraction** | Raw IQ + memory | Memory taps | Different per notebook |
| **Architecture** | RNN/DGRU | TDNN (FC-only) ✅ | FC layers |
| **Loss** | MSE | WGAN-GP + Spectral + L1 ✅ | WGAN-GP + Spectral |
| **QAT** | Optional | Implemented ✅ | Partially |

**Your train.py is actually closer to correct than your Colab notebooks.**

---

## What a Senior Engineer Would Do

### Step 1: **Freeze the Architecture (Do This First)** ✓

Create ONE authoritative specification:

```python
# PN-TDNN-DPD Final Specification
MEMORY_DEPTH = 3  # M=3 (not 5)
INPUT_DIM = 24    # Phase-normalized features
HIDDEN_DIMS = [32, 16]
OUTPUT_DIM = 2

# Feature vector per sample:
# [A(n-k), A³(n-k), I_norm(n-k), Q_norm(n-k), I(n-k), Q(n-k)] × 4 taps
# = 6 features × 4 taps = 24 dimensions
```

**Why M=3 not M=5**: 
- GaN memory decays exponentially; M=3 captures >95% of energy
- Latency: 81 cycles (M=3) vs 101 cycles (M=5)
- Parameters: 1,362 (M=3) vs 1,554 (M=5)

### Step 2: **Implement Phase Normalization Correctly** ✓

```python
class PhaseNormalizedFeatureExtraction(nn.Module):
    def __init__(self, memory_depth: int = 3):
        super().__init__()
        self.M = memory_depth
        self.output_dim = 6 * (memory_depth + 1)  # 24 for M=3
        
    def forward(self, iq_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            iq_sequence: [batch, seq_len, 2] - I, Q
        Returns:
            features: [batch, seq_len - M, 24]
        """
        batch, seq_len, _ = iq_sequence.shape
        I = iq_sequence[..., 0]
        Q = iq_sequence[..., 1]
        
        # Envelope
        A = torch.sqrt(I**2 + Q**2 + 1e-8)
        A3 = A ** 3
        
        outputs = []
        for n in range(self.M, seq_len):
            tap_features = []
            
            # Current sample reference
            I_0, Q_0, A_0 = I[:, n], Q[:, n], A[:, n]
            
            for k in range(self.M + 1):
                idx = n - k
                
                # Feature 1-2: Amplitude, Amplitude³
                tap_features.append(A[:, idx:idx+1])
                tap_features.append(A3[:, idx:idx+1])
                
                # Feature 3-4: Phase-normalized IQ
                I_k, Q_k = I[:, idx], Q[:, idx]
                # Complex multiply: (I_k + jQ_k) × (I_0 - jQ_0) / A_0
                I_norm = (I_k * I_0 + Q_k * Q_0) / (A_0 + 1e-8)
                Q_norm = (Q_k * I_0 - I_k * Q_0) / (A_0 + 1e-8)
                tap_features.append(I_norm.unsqueeze(-1))
                tap_features.append(Q_norm.unsqueeze(-1))
                
                # Feature 5-6: Raw IQ (residual/linear path)
                tap_features.append(I[:, idx:idx+1])
                tap_features.append(Q[:, idx:idx+1])
            
            outputs.append(torch.cat(tap_features, dim=-1))
        
        return torch.stack(outputs, dim=1)
```

### Step 3: **Fix Training Flow (Remove PADigitalTwin from Loop)** ✓

```python
def train_step_ila(generator, discriminator, batch, g_opt, d_opt, config):
    """
    CORRECT ILA training - NO PA model in loop.
    """
    y_pa, u_pa = batch  # y_pa: PA output (input to DPD), u_pa: PA input (target)
    
    # ===== Discriminator =====
    for _ in range(n_critic):
        with torch.no_grad():
            dpd_output = generator(y_pa)  # DPD(y_PA)
            
        # Real: clean PA input (what DPD should produce)
        # Fake: DPD output (what DPD actually produces)
        d_real = discriminator(u_pa, y_pa[:, -1, :])  # condition on current y_PA
        d_fake = discriminator(dpd_output.detach(), y_pa[:, -1, :])
        
        d_loss = wasserstein_loss(d_real, d_fake) + gradient_penalty(...)
        d_loss.backward()
        d_opt.step()
    
    # ===== Generator =====
    dpd_output = generator(y_pa)
    
    # Losses
    g_adv = -discriminator(dpd_output, y_pa[:, -1, :]).mean()
    g_l1 = F.l1_loss(dpd_output, u_pa)
    g_spec = spectral_loss(dpd_output, u_pa)
    
    g_total = g_adv + 10 * g_spec + 1 * g_l1  # 11:1 supervised ratio
    g_total.backward()
    g_opt.step()
```

### Step 4: **Create Single Authoritative Training Script** ✓

**Delete**: [BACKUP]training_colab.ipynb and the buggy second version.

**Create**: `training_colab_v2.ipynb` that:
1. Uses `PhaseNormalizedFeatureExtraction` (24-dim)
2. Implements true ILA (no PA model in loop)
3. Uses measured data from `data/*.csv`
4. Matches train.py logic exactly

### Step 5: **Run Ablation Study BEFORE Contest Submission**

| Config | Loss | Expected ACPR |
|--------|------|---------------|
| A | L1 only | ~-55 dBc |
| B | L1 + Spectral | ~-58 dBc |
| C | L1 + Spectral + WGAN-GP | ~-60 dBc (if GAN helps) |

**This is your evidence** that each component contributes.

---

## Immediate Action Items

### Day 1: Consolidate ✓

1. **Delete** all backup/old notebooks
2. **Create** `models/pn_tdnn_generator.py` with correct 24-dim phase-normalized features
3. **Update** train.py to use new generator

### Day 2: Validate

4. **Run** train.py on measured data (no synthetic PA)
5. **Log** ACPR, EVM, NMSE every epoch
6. **Compare** to OpenDPD baseline

### Day 3: Ablation

7. **Run** 3 configs (L1-only, +Spectral, +WGAN-GP)
8. **Document** results for contest paper

### Day 4: Export

9. **Export** weights to hex (Q1.15)
10. **Verify** RTL simulation matches Python inference

---

## The Bottom Line

Your **documentation** (ARCHITECTURE.md) is excellent. Your **train.py** logic is correct. Your **Colab notebooks** are broken in multiple ways.

**Fix priority**:
1. ❌ Feature extraction mismatch (24 vs 30 vs 18)
2. ❌ Phase normalization not implemented
3. ❌ PADigitalTwin in training loop (defeats ILA)
4. ❌ Batch-level vs sample-level delay confusion

Once you unify the architecture and fix the training loop, the rest (WGAN-GP, spectral loss, QAT) will work correctly. Right now you're debugging the wrong code against the wrong spec.