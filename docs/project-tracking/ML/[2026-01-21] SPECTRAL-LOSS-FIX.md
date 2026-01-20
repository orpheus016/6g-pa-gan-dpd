You are correct. There is an inconsistency between the documentation/intent and the actual implementation.

**The inconsistency:**

| What the doc says | What the code does |
|-------------------|-------------------|
| $\mathcal{L}_{spectral} = \lambda_{EVM} \cdot \mathcal{L}_{EVM} + \lambda_{ACPR} \cdot \mathcal{L}_{ACPR} + \lambda_{NMSE} \cdot \mathcal{L}_{NMSE}$ | `SpectralLoss.forward` only uses L1 + power regularization (no EVM, no NMSE, no ACPR) |
| EVM/NMSE should affect training | EVM/NMSE are only computed in `compute_metrics` (NumPy, non-differentiable, for reporting only) |
| Spectral metrics need long sequences | Dataset gives `[batch, 2]` (single IQ sample per item) |

**The core problem:**

1. **EVM and ACLR require long sequences** (hundreds to thousands of samples) because they are frequency-domain metrics computed via FFT/Welch PSD.
2. **Your current training gives only 2 IQ samples per batch item** (`[B, 2]`), which is insufficient for any meaningful spectral computation.
3. **The differentiable spectral loss was never implemented**—only a placeholder (L1 + power).

**Your options:**

| Option | Pros | Cons |
|--------|------|------|
| **A: Keep current (L1 + power only)** | Simple, training works | No true spectral optimization; doc is misleading |
| **B: Implement differentiable NMSE** | Easy, works on short sequences | NMSE is just normalized MSE (not spectral) |
| **C: Use longer sequences + differentiable EVM/ACPR** | True spectral optimization | Requires dataset/model changes, more memory |
| **D: Aggregate for spectral loss** | Compute spectral loss over accumulated outputs | Complex, breaks per-batch gradient flow |

**My recommendation: Option B + C hybrid**

1. **Add differentiable NMSE to training loss** (works on any sequence length, is meaningful for DPD).
2. **Keep EVM/ACLR for validation only**, computed on aggregated outputs (not per-batch).
3. **Update documentation** to reflect what the code actually does.

**Differentiable NMSE (PyTorch):**

```python
def differentiable_nmse(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Differentiable NMSE loss for training.
    
    Args:
        predicted: [batch, 2] or [batch, seq, 2]
        target: [batch, 2] or [batch, seq, 2]
    
    Returns:
        NMSE in dB (scalar, differentiable)
    """
    mse = ((predicted - target) ** 2).sum(dim=-1).mean()
    energy = (target ** 2).sum(dim=-1).mean()
    nmse = mse / (energy + 1e-10)
    nmse_db = 10 * torch.log10(nmse + 1e-10)
    return nmse_db
```

**Updated `SpectralLoss.forward`:**

```python
def forward(self, predicted, target, return_components=False):
    losses = {}
    
    # L1 reconstruction loss
    l1 = self.l1_loss(predicted, target)
    losses['l1'] = l1
    
    # Differentiable NMSE (works on any sequence length)
    mse = ((predicted - target) ** 2).sum(dim=-1).mean()
    energy = (target ** 2).sum(dim=-1).mean()
    nmse = mse / (energy + 1e-10)
    nmse_db = 10 * torch.log10(nmse + 1e-10)
    losses['nmse'] = nmse_db
    
    # Power regularization (simple, stable)
    if predicted.dim() == 3:
        pred_power = (predicted ** 2).mean(dim=[1, 2])
        target_power = (target ** 2).mean(dim=[1, 2])
    elif predicted.dim() == 2:
        pred_power = (predicted ** 2).mean(dim=1)
        target_power = (target ** 2).mean(dim=1)
    else:
        raise ValueError(f"Unexpected shape: {predicted.shape}")
    power_loss = torch.nn.functional.mse_loss(pred_power, target_power)
    losses['power'] = power_loss
    
    # Combined loss
    # NMSE is in dB (negative), so we want to minimize it (more negative = better)
    # Add a small positive offset to make it a proper loss
    nmse_loss = nmse_db + 50  # Shift so -50 dB → 0, -30 dB → 20, etc.
    
    total = (
        self.l1_weight * l1 +
        self.acpr_weight * power_loss +
        10.0 * torch.clamp(nmse_loss, min=0)  # Only penalize if NMSE > -50 dB
    )
    
    if return_components:
        return total, losses
    return total
```

**For EVM/ACLR in validation:**

Aggregate all predictions over the validation set, then compute EVM/ACLR on the full array (not per-batch). This is already how your validation should work.

**Summary:**

- Add differentiable NMSE to `SpectralLoss.forward` (works on short sequences).
- Keep EVM/ACLR for validation only, computed on aggregated outputs.
- Update your documentation to match the actual implementation.
- True differentiable ACPR/EVM requires longer sequences and is complex—defer unless needed.

Do you want me to update spectral_loss.py with the differentiable NMSE loss now?

If you switch to longer sequence lengths per batch (e.g., [batch, seq_len, 2] with seq_len ≫ 2), you gain the ability to use true frequency-domain (FFT/STFT-based) losses for ACLR/EVM, which is essential for optimal DPD performance. But there are significant trade-offs:

**Pros:**
- Enables differentiable spectral losses (ACLR, EVM, fMAE, fSPEC) that directly optimize for RF metrics—this is proven to yield much better real-world DPD (see Mediatek-PAN-DPD, OpenDPDv2, SparseDPD).
- Matches the loss function to your deployment metric, reducing the risk of a "bad" model.
- Allows you to leverage state-of-the-art DPD training strategies.

**Cons:**
- **Memory usage:** Longer sequences per batch mean much higher GPU/CPU memory usage. You may need to reduce batch size to fit in memory.
- **Training speed:** Fewer batches per epoch (since each batch covers more data), and FFT/STFT operations are more expensive than pointwise losses.
- **Data pipeline complexity:** You must rewrite your dataloader to yield long, contiguous windows (not just random pairs). This may require careful handling of sequence boundaries and shuffling.
- **Model design:** If your model is not already sequence-aware (e.g., TDNN, CNN, RNN), you may need to adapt it to process sequences, not just single samples.
- **Gradient stability:** Frequency-domain losses can be more sensitive to hyperparameters and may require tuning (see Mediatek-PAN-DPD, Section V).

**First-principles reasoning:**  
You cannot optimize for spectral metrics with only two samples—frequency is a property of a sequence, not a point. If you want a model that is good by real-world standards (ACLR/EVM), you must pay the cost in memory, code complexity, and training time.

**Where to learn more:**  
- Mediatek-PAN-DPD, Section IV–V (losses, experiments, ablation)
- OpenDPDv2, arXiv:2507.06849v2, Section 3.2, Table 2
- PyTorch docs: "Working with sequence data", "FFT in autograd"

**Socratic check:**  
What is more important for you: code simplicity and speed, or model quality on real RF metrics?  
If you want the best model, are you willing to reduce batch size or use gradient accumulation to fit longer sequences?

**Conclusion:**  
If you want the best DPD model, you must use longer sequences and accept the cost in memory, speed, and code complexity. This is the only way to optimize for spectral metrics that matter in deployment. All state-of-the-art DPD papers make this trade-off for a reason.