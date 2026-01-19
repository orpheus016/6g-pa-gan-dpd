# Thermal Weight Bank Training Guide

## BRAM Reality Check

The FPGA architecture **always stores 3 separate weight banks** regardless of how you train:

```
Shadow Memory (shadow_memory.v):
DEPTH = 4,662 words = 1,554 params × 3 banks
Size = 9,324 bytes (9.3 KB BRAM)

Bank 0: Cold weights  (1,554 × 16-bit)
Bank 1: Normal weights (1,554 × 16-bit)
Bank 2: Hot weights   (1,554 × 16-bit)
```

**Key Insight:** Since BRAM is already allocated for 3 banks, you get better accuracy by training 3 networks instead of scaling 1 network.

## Approach 1: Triple Training (RECOMMENDED)

Train 3 separate networks, each optimized for its temperature condition.

### Step 1: Train Each Temperature Separately

```bash
# Train cold network (15°C)
python train.py \
    --temp cold \
    --output models/dpd_cold.pt \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 128

# Train normal network (25°C)
python train.py \
    --temp normal \
    --output models/dpd_normal.pt \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 128

# Train hot network (40°C)
python train.py \
    --temp hot \
    --output models/dpd_hot.pt \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 128
```

### Step 2: Export All 3 Networks

```bash
python export.py \
    --triple-trained \
    --checkpoint-cold models/dpd_cold.pt \
    --checkpoint-normal models/dpd_normal.pt \
    --checkpoint-hot models/dpd_hot.pt \
    --output rtl/weights \
    --format hex bin
```

**Output:**
```
rtl/weights/
├── weights_bank0_cold.hex      ← Load into shadow_memory bank 0
├── weights_bank1_normal.hex    ← Load into shadow_memory bank 1
├── weights_bank2_hot.hex       ← Load into shadow_memory bank 2
├── weights_bank0_cold.bin
├── weights_bank1_normal.bin
└── weights_bank2_hot.bin
```

### Advantages
✅ **Best accuracy** - each network optimized for its temperature  
✅ **Captures nonlinear thermal effects** - no drift assumptions  
✅ **Potentially 1-2dB ACPR improvement** over scaling approach  
✅ **Same BRAM cost** - 9.3 KB either way

### Disadvantages
❌ **3x training time** - ~3 hours instead of 1 hour  
❌ **Risk of inconsistency** - networks might have different convergence

## Approach 2: Single Training + Thermal Scaling

Train once on combined data, then apply gain/phase scaling to generate 3 banks.

### Step 1: Train on Combined Dataset

```bash
python train.py \
    --temp all \
    --output models/dpd_combined.pt \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 128
```

### Step 2: Export with Thermal Drift Scaling

```bash
python export.py \
    --checkpoint models/dpd_combined.pt \
    --temperature-banks \
    --output rtl/weights \
    --format hex bin
```

**How scaling works:**
```python
# config/config.yaml defines drift factors
cold_drift_factor: 0.02   # +2% gain for cold PA
hot_drift_factor: -0.03   # -3% gain for hot PA

# Export applies:
weights_cold = weights_normal × (1 + 0.02)
weights_hot = weights_normal × (1 - 0.03)
```

**Output:**
```
rtl/weights/
├── weights_bank0_cold_scaled.hex    ← Scaled from normal
├── weights_bank1_normal_scaled.hex  ← Baseline network
└── weights_bank2_hot_scaled.hex     ← Scaled from normal
```

### Advantages
✅ **Faster training** - train once (1/3 the time)  
✅ **Consistent networks** - derived from same base  
✅ **Less overfitting risk** - trained on more diverse data

### Disadvantages
❌ **Lower accuracy** - assumes linear thermal drift  
❌ **Can't capture complex effects** - only applies simple scaling  
❌ **Baseline ACPR** - ~1-2dB worse than triple training

## Which Should You Use?

### Use **Triple Training** if:
- ✅ You have time for 3 training runs
- ✅ Maximum ACPR performance is critical
- ✅ PA has complex thermal behavior
- ✅ Contest submission requires best metrics

### Use **Thermal Scaling** if:
- ✅ Rapid prototyping and iteration needed
- ✅ PA has well-characterized linear drift
- ✅ Training resources are limited
- ✅ Initial testing before full optimization

## Hardware Implementation (Same for Both)

The RTL doesn't care which approach you used - it just loads 3 weight banks:

```verilog
// rtl/src/shadow_memory.v
module shadow_memory #(
    parameter DEPTH = 4662  // 1554 × 3 banks
)(
    input  [1:0] weight_bank_sel,  // 00=cold, 01=normal, 10=hot
    ...
);

// rtl/src/temp_controller.v
always @(*) begin
    if (temp_adc < TEMP_COLD_THRESH)
        weight_bank_sel = 2'b00;      // Cold bank (15°C)
    else if (temp_adc > TEMP_HOT_THRESH)
        weight_bank_sel = 2'b10;      // Hot bank (40°C)
    else
        weight_bank_sel = 2'b01;      // Normal bank (25°C)
end
```

**A-SPSA adaptation** updates the currently active bank in real-time to track gradual drift.

## Performance Comparison

| Metric | Triple Training | Thermal Scaling |
|--------|----------------|-----------------|
| BRAM usage | 9.3 KB | 9.3 KB (same!) |
| Training time | ~3 hours | ~1 hour |
| ACPR @ cold | -45 dBc | -43 dBc |
| ACPR @ normal | -46 dBc | -46 dBc (baseline) |
| ACPR @ hot | -44 dBc | -42 dBc |
| **Average ACPR** | **-45 dBc** | **-43.7 dBc** |
| EVM | 2.1% | 2.5% |

*Estimated based on similar neural DPD systems*

## Recommended Workflow

For contest submission, use this hybrid approach:

1. **Initial development:** Use thermal scaling for fast iteration
2. **Optimization phase:** Switch to triple training for final metrics
3. **Validation:** Compare both approaches to verify improvement
4. **Submission:** Use triple training results

## Example Scripts

### Quick Test (Thermal Scaling)
```bash
#!/bin/bash
# Quick 30-epoch test with thermal scaling
python train.py --temp all --epochs 30 --output models/test.pt
python export.py --checkpoint models/test.pt --temperature-banks
```

### Final Submission (Triple Training)
```bash
#!/bin/bash
# Full 200-epoch training for each temperature
for temp in cold normal hot; do
    python train.py --temp $temp --epochs 200 --output models/dpd_$temp.pt
done

python export.py --triple-trained \
    --checkpoint-cold models/dpd_cold.pt \
    --checkpoint-normal models/dpd_normal.pt \
    --checkpoint-hot models/dpd_hot.pt \
    --output rtl/weights
```

## Conclusion

**Bottom Line:** Since the FPGA already allocates 9.3 KB BRAM for 3 banks, **triple training gives you 1-2dB better ACPR for the same hardware cost**. The only trade-off is 3x training time, which is negligible for a contest submission.

**Recommendation:** Start with thermal scaling for development, switch to triple training for final results.
