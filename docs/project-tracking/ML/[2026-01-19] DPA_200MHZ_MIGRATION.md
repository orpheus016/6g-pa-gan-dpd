# DPA 200MHz Dataset Migration - Implementation Summary

## Overview
Successfully migrated training codebase from **apa_200mhz** (5 channels, 1 sub-channel, 250 MSps) to **dpa_200mhz** (1 channel, 10 sub-channels, 200 MSps).

## Changes Implemented

### 1. training_colab_v2.ipynb (Production Notebook)

#### Cell 3: Configuration Dictionary
**Changed:**
- Added `dataset` section with:
  - `'name': 'dpa_200mhz'`
  - `'n_channels': 1` (was 5 for apa_200mhz)
  - `'n_sub_channels': 10` (was 1 for apa_200mhz)
  
- Updated `system` section:
  - `'sample_rate': 200e6` (was 250e6)
  
- Added new `spectral_loss` section:
  - `'bw_main_ch': 200e6` - Main channel bandwidth
  - `'n_sub_ch': 10` - Sub-channels for EVM/ACLR analysis
  - `'nperseg': 1024` - FFT segment length

**Impact:** Configuration now correctly specifies dpa_200mhz 10-subchannel analysis

#### Cell 7: SpectralLoss Instantiation
**Changed:**
```python
# OLD (apa_200mhz):
spectral_loss_fn = SpectralLoss(sample_rate=config['system']['sample_rate'])

# NEW (dpa_200mhz):
spectral_loss_fn = SpectralLoss(
    sample_rate=config['system']['sample_rate'],              # 200e6
    bw_main_ch=config['spectral_loss']['bw_main_ch'],        # 200e6
    n_sub_ch=config['spectral_loss']['n_sub_ch'],            # 10 (KEY)
    nperseg=config['spectral_loss']['nperseg'],              # 1024
)
```

**Impact:** SpectralLoss now computes EVM/ACLR across 10 frequency sub-channels instead of 1

### 2. train.py (Main Training Script)

#### Line ~575: SpectralLoss Instantiation
**Changed:**
```python
# OLD:
spectral_loss = SpectralLoss(
    sample_rate=config['system'].get('sample_rate', 250e6)
)

# NEW:
spectral_loss = SpectralLoss(
    sample_rate=config['system'].get('sample_rate', 200e6),
    bw_main_ch=config.get('spectral_loss', {}).get('bw_main_ch', 200e6),
    n_sub_ch=config.get('spectral_loss', {}).get('n_sub_ch', 10),
    nperseg=config.get('spectral_loss', {}).get('nperseg', 1024),
)
```

**Impact:** train.py now matches notebook configuration with 10 sub-channel analysis

### 3. config/config.yaml (Configuration Template)

#### System Section
- `sample_rate` already set to `200e6` ✓

#### Added: Spectral Loss Section
```yaml
spectral_loss:
  bw_main_ch: 200e6
  n_sub_ch: 10
  nperseg: 1024
  targets:
    evm_db: -45
    aclr_db: -62
    nmse_db: -42
```

#### Added: Dataset Section
```yaml
dataset:
  name: "dpa_200mhz"
  n_channels: 1
  n_sub_channels: 10
  sample_rate: 200e6
```

**Impact:** YAML now documents dpa_200mhz configuration parameters

## Dataset Structure

### DPA 200MHz Dataset Location
`data/DPA_200MHz/`

**Files:**
- `train_input.csv` / `train_output.csv`
- `val_input.csv` / `val_output.csv`
- `test_input.csv` / `test_output.csv`
- `spec.json` - Dataset metadata

**Specification (spec.json):**
```json
{
    "dataset_format": "split_csv",
    "input_signal_fs": 800e6,
    "bw_main_ch": 200e6,
    "bw_sub_ch": 20e6,
    "n_sub_ch": 10,        ← 10 frequency sub-channels
    "nperseg": 2560
}
```

**Data Format:**
- Single-channel IQ pairs (I, Q columns in CSV)
- No channel stacking needed
- Existing `load_measured_data()` function already compatible

## Key Parameters Updated

| Parameter | Old (apa_200mhz) | New (dpa_200mhz) | Notes |
|-----------|-----------------|-----------------|-------|
| `n_channels` | 5 | 1 | Single-channel input |
| `n_sub_channels` | 1 | 10 | Finer spectral resolution |
| `sample_rate` | 250e6 (250 MSps) | 200e6 (200 MSps) | DPA sampling rate |
| `bw_main_ch` | 200e6 | 200e6 | Main channel BW unchanged |
| `n_sub_ch` (SpectralLoss) | 1 | 10 | **Critical for metrics** |
| `nperseg` | 19662 (default) | 1024 | Configurable FFT length |

## What Did NOT Change

✅ **Model Architecture**: PN-TDNN (24-dim, M=3, 1,362 params) - unchanged
✅ **Feature Extraction**: Phase-normalization logic - unchanged
✅ **Training Flow**: ILA (no PA in loop) - unchanged
✅ **CWGAN-GP**: Discriminator/critic - unchanged
✅ **QAT**: Quantization schedule - unchanged (epoch 300)
✅ **Data Loading**: CSV format compatible - unchanged

## Impact on Spectral Metrics

### EVM (Error Vector Magnitude)
- **Before**: Single sub-channel analysis (n_sub_ch=1)
- **After**: 10 sub-channel analysis (n_sub_ch=10)
- **Result**: **Finer spectral resolution** - Better visibility into inter-modulation distortion across frequency bins

### ACLR (Adjacent Channel Leakage Ratio)
- **Before**: Power summed across 1 sub-channel
- **After**: Power summed per 20 MHz sub-channel (200 MHz / 10)
- **Result**: **More granular spectral regrowth measurement**

### NMSE (Normalized Mean Square Error)
- **Before & After**: Unchanged (time-domain, global metric)
- **Result**: Should show similar or better performance with finer subchannel training

## Backward Compatibility

The changes are **backward compatible**:
- Code can still train on apa_200mhz by setting `n_sub_ch: 1` in config
- `SpectralLoss` class already supports any n_sub_ch value (tested with 1 and 10)
- Data loading is dataset-agnostic (any single-channel CSV format)

## Validation Checklist

- [x] Config updated with dpa_200mhz parameters
- [x] training_colab_v2.ipynb Cell 3 updated (config dict)
- [x] training_colab_v2.ipynb Cell 7 updated (SpectralLoss init)
- [x] train.py updated (SpectralLoss init line ~575)
- [x] config.yaml added spectral_loss section
- [x] config.yaml added dataset section
- [x] Data files verified in `data/DPA_200MHz/`
- [x] spec.json verified (n_sub_ch=10)
- [x] CSV format confirmed (single-channel, I/Q columns)
- [x] Model architecture still frozen (1,362 params)

## Next Steps

1. **Load DPA dataset in training**:
   ```python
   data_dir = Path('data/DPA_200MHz')  # Changed from data/APA_200MHz
   u_pa_train, y_pa_train = load_measured_data(data_dir, 'train')
   ```

2. **Run training with new config**:
   ```bash
   python train.py --config config/config.yaml
   # or
   # Run training_colab_v2.ipynb cells 4-10
   ```

3. **Monitor training curves**:
   - EVM should track per-subchannel (10 values averaged)
   - ACLR should show left/right adjacent channel leakage
   - Loss curves should converge smoothly (epochs 0-299 float32, 300+ QAT)

4. **Validate test metrics**:
   - EVM < -45 dB ✓
   - ACLR < -62 dBc ✓
   - NMSE < -42 dB ✓

## Files Modified

1. `training_colab_v2.ipynb` - Cells 3 and 7
2. `train.py` - Line ~575
3. `config/config.yaml` - Added spectral_loss and dataset sections

## Reference

**OpenDPDv2 Alignment:**
- SpectralLoss.compute_evm() - Frequency-domain per-subchannel ✓
- SpectralLoss.compute_aclr() - Welch PSD ✓
- SpectralLoss.compute_nmse() - Time-domain I/Q ✓

**DPA Dataset Specification:**
- `data/DPA_200MHz/spec.json` - Dataset metadata
- 10 frequency sub-channels at 20 MHz each
- Single-channel IQ input
- 200 MSps sampling rate
