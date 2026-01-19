## Code Block Report: training_colab_v2.ipynb

| Cell | Purpose | Key Implementation |
|------|---------|-------------------|
| 1 | Environment setup | Colab/local detection, repo path setup |
| 2 | Imports | Imports `PNTDNNGenerator`, `PhaseNormalizedFeatureExtraction`, `Discriminator` from `models.pn_tdnn_generator` |
| 3 | Configuration | M=3, 24-dim, hidden=[32,16], QAT starts at epoch 300 |
| 4 | Data loading | CSV loader matching train.py (ILA convention: u_pa=target, y_pa=input) |
| 5 | Dataset creation | Returns raw IQ sequences `[batch, M+1, 2]` — feature extraction handled by generator |
| 6 | Model creation | Creates generator (1,362 params) and discriminator, **verifies param count** |
| 7 | Optimizers | Adam with β₁=0 (WGAN-GP requirement), cosine scheduler |
| 8 | Train step | **ILA: no PA in loop**, matches train.py exactly — WGAN-GP + GP + spectral loss |
| 9 | Validation | Computes EVM/NMSE/L1 metrics |
| 10 | Training loop | **QAT transition at epoch 300** via `generator.enable_qat()` |
| 11 | Plot curves | Training visualization with QAT transition marker |
| 12 | Save model | Best checkpoint with config |
| 13 | Test evaluation | Final metrics vs targets |
| 14 | Export weights | Q1.15 hex files for Verilog `$readmemh` |

## Key Differences from Backup Notebook

| Aspect | [BACKUP] | v2 (New) |
|--------|----------|----------|
| **Input dim** | 18 | **24** ✓ |
| **Memory depth** | M=5 | **M=3** ✓ |
| **Feature extraction** | Manual in dataset | **PhaseNormalizedFeatureExtraction** ✓ |
| **PA model in loop** | PADigitalTwin used | **None (true ILA)** ✓ |
| **QAT transition** | Not implemented | **Automatic at epoch 300** ✓ |
| **Param count** | Not verified | **Asserts 1,362** ✓ |
| **Output activation** | Tanh | **Linear** ✓ |

Made changes.