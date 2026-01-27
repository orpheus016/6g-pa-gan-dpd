Let me review your current spectral loss and training code to provide a concrete integration plan. 

Read [](file:///c%3A/Users/James/github/6g-pa-gan-dpd/utils/spectral_loss.py#420-420), lines 420 to 550

Now let me check the notebook's current train_step to see where the loss is computed: 

Read [](file:///c%3A/Users/James/github/6g-pa-gan-dpd/training_colab_v2.ipynb#1050-1050), lines 1050 to 1100

Now I have the complete picture. Let me provide you with a comprehensive integration plan:

---

## Review of Your Current Setup vs. ACLR-LOSS.md

### Current `SpectralLoss.forward()`:
```python
total = (
    self.l1_weight * l1 +           # L1 reconstruction
    self.power_weight * power_loss + # Power matching
    self.nmse_weight * nmse_loss     # NMSE (dB)
)
```

### What's Missing:
**No A³-weighted MSE** — The physics-based ACLR surrogate from your ACLR-LOSS.md is not implemented.

---

## Integration Plan

### Step 1: Add A³-MSE Loss to spectral_loss.py

```python
def compute_a3_weighted_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    input_signal: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute A³-weighted MSE (ACLR surrogate loss).
    
    From Volterra theory: IMD3 amplitude ∝ |x|³
    Adjacent-band energy is dominated by high-amplitude samples.
    
    Formula: L_A³ = E[|x[n]|³ · |e[n]|²]
    
    Args:
        predicted: DPD output [B, 2] (I, Q)
        target: Target signal [B, 2] (I, Q)
        input_signal: PA output / DPD input [B, 2] for amplitude weighting
                     If None, uses target amplitude
    
    Returns:
        A³-weighted MSE (scalar, differentiable)
    
    Reference: Guan & Zhu, IEEE T-MTT 2014
    """
    # Use input signal amplitude if provided, else target
    if input_signal is not None:
        A = torch.sqrt(input_signal[:, 0]**2 + input_signal[:, 1]**2 + 1e-8)
    else:
        A = torch.sqrt(target[:, 0]**2 + target[:, 1]**2 + 1e-8)
    
    # Cubic weighting (physics-based: IMD3 ∝ A³)
    A3 = A ** 3  # [B]
    
    # Normalize to prevent gradient explosion
    A3_normalized = A3 / (A3.mean() + 1e-8)  # [B]
    
    # Error squared per sample
    error_sq = (predicted - target) ** 2  # [B, 2]
    error_magnitude = error_sq.sum(dim=-1)  # [B] = |e_I|² + |e_Q|²
    
    # A³-weighted MSE
    a3_mse = (A3_normalized * error_magnitude).mean()
    
    return a3_mse
```

### Step 2: Modify `SpectralLoss.__init__()` to Add A³ Weight

```python
def __init__(
    self,
    sample_rate: float = 250e6,
    channel_bw: float = 100e6,
    adjacent_offset: float = 100e6,
    bw_main_ch: float = 200e6,
    n_sub_ch: int = 1,
    nperseg: int = 2560,
    l1_weight: float = 1.0,
    power_weight: float = 2.0,
    nmse_weight: float = 5.0,
    a3_mse_weight: float = 0.0       # NEW: A³-MSE weight (ACLR surrogate)
):
    super().__init__()
    
    # ...existing code...
    
    # Loss weights
    self.l1_weight = l1_weight
    self.power_weight = power_weight
    self.nmse_weight = nmse_weight
    self.a3_mse_weight = a3_mse_weight  # NEW
    
    self.l1_loss = nn.L1Loss()
```

### Step 3: Modify `SpectralLoss.forward()` to Include A³-MSE

```python
def forward(
    self,
    predicted: torch.Tensor,
    target: torch.Tensor,
    input_signal: torch.Tensor = None,  # NEW: for A³ weighting
    return_components: bool = False
) -> torch.Tensor:
    """
    Compute combined spectral loss for training.
    
    Loss = L1_weight * L1 + power_weight * power + nmse_weight * NMSE + a3_mse_weight * A³_MSE
    """
    losses = {}
    
    # ===== SHAPE FIX: Squeeze if needed =====
    if predicted.dim() == 3 and predicted.shape[1] == 1:
        predicted = predicted.squeeze(1)
    
    # ===== L1 Reconstruction Loss =====
    l1 = self.l1_loss(predicted, target)
    losses['l1'] = l1
    
    # ===== Power Regularization Loss =====
    if predicted.dim() == 2:
        pred_power = (predicted ** 2).mean(dim=1)
        target_power = (target ** 2).mean(dim=1)
    else:
        pred_power = (predicted ** 2).mean(dim=[1, 2])
        target_power = (target ** 2).mean(dim=[1, 2])
    
    power_loss = torch.nn.functional.mse_loss(pred_power, target_power)
    losses['power'] = power_loss
    
    # ===== Differentiable NMSE Loss =====
    nmse_loss = compute_nmse_differentiable(predicted, target, return_db=True)
    losses['nmse'] = nmse_loss
    
    # ===== A³-Weighted MSE Loss (ACLR Surrogate) =====
    if self.a3_mse_weight > 0:
        a3_mse = compute_a3_weighted_mse(predicted, target, input_signal)
        losses['a3_mse'] = a3_mse
    else:
        a3_mse = torch.tensor(0.0, device=predicted.device)
        losses['a3_mse'] = a3_mse
    
    # ===== Combined Loss =====
    total = (
        self.l1_weight * l1 + 
        self.power_weight * power_loss + 
        self.nmse_weight * nmse_loss +
        self.a3_mse_weight * a3_mse  # NEW
    )
    
    if return_components:
        return total, losses
    
    return total
```

### Step 4: Update Notebook - SpectralLoss Instantiation

```python
# Cell 7: Optimizers and Schedulers
# Replace the spectral_loss_fn instantiation:

spectral_loss_fn = SpectralLoss(
    sample_rate=config['system']['sample_rate'],
    bw_main_ch=config['spectral_loss']['bw_main_ch'],
    n_sub_ch=config['spectral_loss']['n_sub_ch'],
    nperseg=config['spectral_loss']['nperseg'],
    l1_weight=50.0,         # L1 reconstruction
    power_weight=10.0,      # Power regularization
    nmse_weight=10.0,       # NMSE (dB scale)
    a3_mse_weight=5.0,      # NEW: A³-MSE (ACLR surrogate) - start at 5.0
)

print("Spectral Loss initialized:")
print(f"  L1 weight: {spectral_loss_fn.l1_weight}")
print(f"  Power weight: {spectral_loss_fn.power_weight}")
print(f"  NMSE weight: {spectral_loss_fn.nmse_weight}")
print(f"  A³-MSE weight: {spectral_loss_fn.a3_mse_weight}")  # NEW
```

### Step 5: Update Notebook - train_step to Pass Input Signal

```python
# In train_step function, modify the spectral loss call:

# ...existing code...

# Now use dpd_output everywhere else - it's always [B, 2]
condition = input_seq[:, -1, :]  # [B, 2] - most recent sample (PA output)

# Reconstruction loss (L1: DPD output should match PA input)
recon_loss = nn.functional.l1_loss(dpd_output, target)

# Discriminator loss
d_fake = discriminator(dpd_output, condition)  # [B, 1]
g_adv_loss = -d_fake.mean()

# Spectral loss - MODIFIED: pass input_signal for A³ weighting
spectral, spectral_components = spectral_loss(
    dpd_output, 
    target, 
    input_signal=condition,  # NEW: PA output amplitude for A³ weighting
    return_components=True
)

# Combined generator loss with weighted components
g_total = (
    loss_config.get('adversarial', 1.0) * g_adv_loss +
    loss_config.get('reconstruction_l1', 40.0) * recon_loss +
    loss_config.get('spectral', 10.0) * spectral
)

# ...existing code...

# Log A³-MSE component
losses['g_a3_mse'] = spectral_components.get('a3_mse', torch.tensor(0.0)).item()
```

### Step 6: Update TensorBoard Logging

```python
# In training loop, add A³-MSE logging:

# TensorBoard logging per batch
writer.add_scalar('Train/Generator/Total', losses['g_total'], global_step)
writer.add_scalar('Train/Generator/Adversarial', losses['g_adv'], global_step)
writer.add_scalar('Train/Generator/Reconstruction', losses['g_recon'], global_step)
writer.add_scalar('Train/Generator/Spectral', losses['g_spectral'], global_step)
writer.add_scalar('Train/Generator/A3_MSE', losses.get('g_a3_mse', 0), global_step)  # NEW
```

---

## Recommended Weight Tuning Strategy

| Phase | l1_weight | power_weight | nmse_weight | a3_mse_weight | Rationale |
|-------|-----------|--------------|-------------|---------------|-----------|
| **Phase 1** (Epochs 0-100) | 50.0 | 10.0 | 10.0 | **0.0** | Establish baseline EVM/NMSE |
| **Phase 2** (Epochs 100-200) | 50.0 | 10.0 | 10.0 | **2.0** | Gentle ACLR optimization |
| **Phase 3** (Epochs 200+) | 40.0 | 10.0 | 10.0 | **5.0** | Stronger ACLR focus |
| **QAT Phase** | 30.0 | 5.0 | 10.0 | **3.0** | Balance for quantization |

---

## Critical Caveats from Your ACLR-LOSS.md

1. **This is a surrogate, not exact ACLR minimization**
   - A³-MSE penalizes high-amplitude errors → correlates with ACLR
   - True ACLR requires FFT over contiguous time blocks

2. **Normalization is essential**
   - `A3_normalized = A3 / (A3.mean() + 1e-8)` prevents gradient explosion
   - Your IQ is already normalized to -3 dBFS (0.7 linear)

3. **Why A³ not exponential**
   - Volterra theory: IMD3 ∝ |x|³ (physics-based)
   - Exponential has no theoretical justification

4. **Expected improvement**
   - Guan & Zhu (2014): A³-weighted loss improved ACLR by 3-5 dB vs uniform MSE
   - Won't achieve -62 dBc alone — combine with good PA model in FLA

---

## Summary: Files to Modify

| File | Changes |
|------|---------|
| spectral_loss.py | Add `compute_a3_weighted_mse()`, modify `__init__()` and `forward()` |
| training_colab_v2.ipynb | Update SpectralLoss instantiation, modify `train_step()` to pass `input_signal`, add TensorBoard logging |

Do you want me to provide the complete modified spectral_loss.py file?