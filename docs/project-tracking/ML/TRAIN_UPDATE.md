# train.py Update Summary

**Date:** January 19, 2026  
**Status:** ✓ Complete and Verified

## Changes Made

### 1. Updated Imports
```python
# Old (removed)
from models import TDNNGenerator, TDNNGeneratorQAT, Discriminator, PADigitalTwin
from models.discriminator import WassersteinLoss

# New
from models import PNTDNNGenerator, create_discriminator, PADigitalTwin
from utils.spectral_loss import SpectralLoss, compute_evm, compute_acpr
```

### 2. Updated `create_models()` Function
- **Old:** Used `TDNNGenerator` (30-dim, M=5) or `TDNNGeneratorQAT`
- **New:** Uses `PNTDNNGenerator` (24-dim, M=3) with built-in QAT support
- QAT is now enabled via `generator.enable_qat()` method

**Parameter counts verified:**
- Generator: 1,362 params (matches ARCHITECTURE.md spec)
- Discriminator: 2,433 params

### 3. Updated `create_dpd_dataset()` Function
- **Old:** Returned pre-extracted features with manual memory taps
- **New:** Returns raw IQ sequences [batch, M+1, 2]
- Phase-normalized feature extraction now handled by `PNTDNNGenerator.forward()`
- Default memory depth changed: M=5 → **M=3** (PN-TDNN spec)

### 4. Updated `train_step()` Function
- **Removed:** Dependency on `WassersteinLoss` helper class
- **Added:** Direct WGAN-GP implementation in `train_step()`:
  - Wasserstein loss calculation (D(real) - D(fake))
  - Gradient penalty computation (enforce 1-Lipschitz constraint)
  - Lambda GP weight from config (default: 10.0)
- **Updated:** Spectral loss integration with loss weighting from config

### 5. Updated Training Loop
- Sample rate updated: 200 MSps → **250 MSps** (matches PYNQ-Z1 spec)
- Default model config now uses M=3 (from config['model']['generator']['memory_depth'], default 3)
- Loss weights now parameterized:
  ```yaml
  loss:
    adversarial: 1.0      # Weight for Wasserstein loss
    reconstruction_l1: 50.0  # Weight for L1 loss
    spectral: 10.0        # Weight for ACPR/EVM/NMSE loss
  ```

## Architecture Alignment

| Spec | ARCHITECTURE.md | train.py (new) | Status |
|------|-----------------|----------------|--------|
| Input dim | 24 | 24 ✓ | **FROZEN** |
| Memory depth | M=3 | M=3 ✓ | **FROZEN** |
| Feature extraction | Phase-normalized | Phase-normalized ✓ | **FROZEN** |
| Parameters | 1,362 | 1,362 ✓ | **VERIFIED** |
| Output | Linear (no Tanh) | Linear ✓ | **FROZEN** |
| QAT | Q1.15/Q8.8 | Enabled ✓ | **FROZEN** |

## Verification Results

```
✓ Models created successfully
  Generator params: 1,362
  Discriminator params: 2,433
  
✓ Phase-normalized feature extraction verified
✓ ILA training flow implemented correctly
✓ QAT integration working
✓ WGAN-GP loss functions implemented
```

## Next Steps

1. **Update config.yaml** to reflect M=3 and sample_rate=250e6
2. **Run training on measured data:**
   ```bash
   python train.py --config config/config.yaml --epochs 500
   ```
3. **Enable QAT fine-tuning:**
   ```bash
   python train.py --config config/config.yaml --qat --resume checkpoints/latest.pth
   ```

## Backward Compatibility

- Old `TDNNGenerator` and `TDNNGeneratorQAT` remain in `models/tdnn_generator.py` (marked deprecated)
- New architecture can coexist with old for ablation studies
- Export function `export_weights_q115()` available for FPGA deployment
