# Machine Learning & Training Status
**Component:** Model Training & Validation  
**Last Updated:** January 2, 2026  
**Status:** ⚠️ NEEDS VALIDATION - Weights Generated but Not Verified

---

## Current Status Summary

**⚠️ CRITICAL: ML validation incomplete!**

The training scripts have generated weights, but **NO PROPER VALIDATION** has been done yet. The RTL team is waiting for ML validation before proceeding with synthesis.

---

## What We Have

### Generated Weights
- ✅ 18 weight files generated (cold/normal/hot × 6 layers)
- ⚠️ **NOT validated** against known benchmarks
- ⚠️ **NOT tested** for ACPR/EVM improvement
- ⚠️ **NOT compared** with Volterra/GMP baselines

### Training Scripts
| Script | Purpose | Status | Issues |
|--------|---------|--------|--------|
| `train.py` | Reference (full CWGAN-GP) | 📚 Reference only | Too complex, not executable |
| `train_opendpd.py` | OpenDPD loading | ⚠️ Incomplete | Missing QAT, spectral loss |
| `train_thermal_variants.py` | 3-variant training | ⚠️ Simplified | CPU-only, no validation |
| `validate_model_cpu.py` | Quick architecture test | ✅ Works | Only tests forward pass |
| `verify_gan_vs_supervised.py` | Comparison | ❌ Not updated | Needs Volterra comparison |

---

## CRITICAL ISSUES (From TODO)

### 1. Training Method Not Decided
**Question:** Which training approach should we use?
- **Option A: Full CWGAN-GP (like `train.py`) - Most accurate but complex** add awareness with OpenDPD flow
- Option B: Supervised only (simpler) - Faster but less accurate
- Option C: OpenDPD's method - Need to study their flow

**Blocker:** Need to study **OpenDPDv2 paper** and decide on:
- Training epochs (how many?)
- Loss functions (supervised vs GAN vs hybrid?)
- Hyperparameters (learning rate, batch size, etc.)

### 2. Missing QAT (Quantization-Aware Training)
**Current:** Weights quantized AFTER training (post-training quantization)  
**Problem:** Accuracy degradation not measured  
**Need:** Implement QAT in training loop to minimize EVM loss

### 3. Missing Spectral Loss
**Current:** Only MSE loss used  
**Problem:** Frequency-domain performance not optimized  
**Need:** Add ACPR/EVM loss terms (already in `utils/spectral_loss.py`)

### 4. No Validation Against Baselines
**Missing comparisons:**
- [ ] Volterra series (classic DPD)
- [ ] GMP (generalized memory polynomial)
- [ ] Standard 5G DPD
- [ ] OpenDPDv2 results
- [ ] Our TDNN approach

---

## What Needs to Be Done (Priority Order)

### Phase 1: Understand OpenDPDv2 Training Flow (HIGH PRIORITY)
**Action Items:**
1. [ ] Read OpenDPDv2 paper (https://arxiv.org/abs/2507.06849)
2. [ ] Check their GitHub for training scripts (https://github.com/lab-emi/OpenDPD)
3. [ ] Document their:
   - Number of epochs
   - Loss functions used
   - Data augmentation
   - Validation metrics
4. [ ] Decide: Use their method or adapt ours?

### Phase 2: Choose ONE Training Script and Make It Complete
**Current problem:** We have 3 half-finished training scripts!

**Recommendation:** Pick ONE and make it production-ready:
```
Option A: Extend train_thermal_variants.py
  + Already generates 3 weight sets
  + Simple, understandable
  - Missing QAT, spectral loss, validation

Option B: Integrate train.py with other features
  + Has full CWGAN-GP pipeline
  + Has QAT and spectral loss
  - Too complex, hard to debug
  - No thermal variant
  ! Dont use current config, use OpenDPD APA_200MHz dataset then add thermal variant then train NN on 3 different temp
  ? Check other dataset feasibility on OpenDPD github

Option C: Follow OpenDPDv2 exactly
  + Proven to work
  + Reproducible results
  - Need to study their code first
```

**Decision needed:** Which option? (**Recommend: Integrate Option B with Option C**)

### Phase 3: Implement Missing Features
Once we pick a script, add:
- [ ] **QAT:** Quantize during training, not after
- [ ] **Spectral loss:** Add ACPR + EVM terms to loss function
- [ ] **Proper validation:** Test on held-out data
- [ ] **Baseline comparison:** Train Volterra/GMP for comparison (adjust validate.py to OpenDPD)

### Phase 4: Full Training Run
- [ ] Train on full dataset (not subset)
- [ ] Use GPU (not CPU)
- [ ] Monitor convergence
- [ ] Save checkpoints
- [ ] Log metrics (loss, ACPR, EVM, NMSE)

### Phase 5: Validation & Benchmarking
- [ ] Test trained model on validation set (adjust to OpenDPD)
- [ ] Measure ACPR improvement (target: >10 dB)
- [ ] Measure EVM reduction (target: <2%)
- [ ] Compare with Volterra baseline and other comparisons
- [ ] Generate validation plots ()
- [ ] Document results

---

## Decoupled Loop Validation (For RTL Team)

**Question from RTL team:** How to validate the A-SPSA adaptation loop before implementing in RTL?

### Recommended Validation Approach

#### Step 1: Python Reference Model (RTL after ML Training)
**Create:** `validate_aspsa_loop.py` that simulates the full adaptation loop

**What it should do:**
1. Load trained TDNN weights as starting point
2. Simulate PA output with thermal drift
3. Run A-SPSA gradient estimation:
   ```python
   # Simultaneous perturbation
   delta = random_bernoulli()  # ±1 for each weight
   
   # Two-sided gradient estimation
   J_plus = loss(weights + c_k * delta)
   J_minus = loss(weights - c_k * delta)
   grad_estimate = (J_plus - J_minus) / (2 * c_k * delta)
   
   # Weight update
   weights_new = weights - a_k * grad_estimate
   ```
4. Update weights with annealing schedule:
   ```python
   a_k = a / (A + k + 1)^alpha     # Learning rate (alpha=0.602)
   c_k = c / (k + 1)^gamma         # Gradient step (gamma=0.101)
   ```
5. Measure convergence (NMSE vs iteration)

**Deliverable:** Python script showing A-SPSA converges in <1000 iterations

#### Step 2: Fixed-Point Simulation (RTL after ML Training)
**Convert floating-point A-SPSA to match RTL precision:**

| Variable | Float (Python) | Fixed-Point (RTL) | Range |
|----------|----------------|-------------------|-------|
| Weights | float32 | Q1.15 (16-bit) | [-1, +0.999] |
| Gradients | float32 | Q16.16 (32-bit) | [-65536, +65536] |
| Learning rate `a_k` | float32 | Q8.24 (32-bit) | [0, 256] |
| Loss `J` | float32 | Q16.16 (32-bit) | [0, +65536] |

**Example fixed-point gradient calculation:**
```python
# In Q1.15 fixed-point
weights_q15 = float_to_q15(weights)
delta_q15 = random_sign()  # ±1 in Q1.15

# Perturb (Q1.15 + Q1.15 = Q1.15)
w_plus_q15 = saturate_q15(weights_q15 + c_k_q15 * delta_q15)
w_minus_q15 = saturate_q15(weights_q15 - c_k_q15 * delta_q15)

# Compute loss (returns Q16.16)
J_plus_q16 = compute_loss_fixed(w_plus_q15)
J_minus_q16 = compute_loss_fixed(w_minus_q15)

# Gradient estimate (Q16.16 / Q1.15 = Q16.16)
grad_q16 = (J_plus_q16 - J_minus_q16) / (2 * c_k_q15 * delta_q15)

# Weight update (Q1.15 - Q8.24 * Q16.16 = Q1.15)
weights_new_q15 = saturate_q15(weights_q15 - a_k_q24 * grad_q16 >> 16)
```

**Deliverable:** Bit-accurate Python model matching RTL data types

#### Step 3: Parameter Tuning (RTL Team)
**Critical A-SPSA parameters that need validation:**

| Parameter | Symbol | Typical Range | What It Does | Needs Tuning? |
|-----------|--------|---------------|--------------|---------------|
| Initial LR | `a` | 0.01 - 0.1 | Controls initial step size | **YES - Critical!** |
| LR offset | `A` | 10 - 100 | Delays LR decay | **YES** |
| LR decay | `α` | 0.602 | Asymptotic optimality | **NO - theory** |
| Grad step | `c` | 0.001 - 0.01 | Perturbation size | **YES** |
| Grad decay | `γ` | 0.101 | Perturbation decay | **NO - theory** |

**Validation process:**
1. **Sweep `a`:** Try [0.01, 0.03, 0.1] and measure convergence speed
2. **Sweep `c`:** Try [0.001, 0.005, 0.01] and check gradient noise
3. **Verify annealing:** Plot `a_k` and `c_k` vs iteration, should decay smoothly
4. **Test temperature reset:** When temp changes, reset to `k=0` (restart annealing)

**Deliverable:** Tuned parameters documented in config file

#### Step 4: Thermal Reset Logic (RTL Team)
**What happens when temperature changes:**
```python
def on_temperature_change(new_temp_state):
    # Save current weights
    checkpoint_weights = current_weights
    
    # Switch to new weight bank
    load_weights(bank=new_temp_state)  # cold=0, normal=1, hot=2
    
    # Reset A-SPSA iteration counter
    iteration_k = 0  # Restart annealing schedule
    
    # Recompute learning rate
    a_k = a / (A + 1)^alpha  # Back to initial LR
    c_k = c / 1^gamma        # Back to initial step
```

**Validation:**
- [ ] Simulate temperature switch during adaptation
- [ ] Verify A-SPSA recovers quickly (<100 iterations)
- [ ] Check weights don't diverge

**Deliverable:** Temperature transition test results

#### Step 5: Co-Simulation Setup (RTL Team Responsibility)
**Once ML provides Python reference:**

1. **Generate test vectors:**
   - Input signal samples (I/Q)
   - PA output with drift
   - Expected weight updates

2. **Run RTL simulation:**
   - Feed same inputs
   - Monitor weight memory writes
   - Compare with Python cycle-by-cycle

3. **Verify match:**
   - Weights should match within ±1 LSB (quantization)
   - Learning rate decay should be identical
   - Convergence curves should overlay

**Pass criteria:** <0.1% difference between Python and RTL outputs

---

## Dataset Status

### OpenDPD APA_200MHz
- ✅ Downloaded at `git clone git@github.com:lab-emi/OpenDPD.git`
- ✅ CSV format: `train_input.csv`, `train_output.csv`
- ✅ 58,980 training samples loaded successfully
- ⚠️ **NOT using** `val_input.csv`, `val_output.csv`, `test_*.csv` yet!

**Action needed:** Split data properly:
```
train_*.csv  →  Training (60%, 35,388 samples) ✅ Using this
val_*.csv    →  Validation (20%, 11,796 samples) ❌ NOT USING
test_*.csv   →  Testing (20%, 11,796 samples) ❌ NOT USING
```

### Configuration Files
**Question from TODO:** Do we need `config.yaml` or `dataset.py`?  

**Answer:** OpenDPD already provides dataset - **use theirs directly**, don't reinvent.

**Recommendation:** 
- ✅ Keep OpenDPD's dataset structure
- ✅ Add our thermal augmentation layer
- ❌ Don't create redundant `dataset.py`

---

## Training Infrastructure Status

### What Works
- ✅ Dataset loading (CSV to complex numpy arrays)
- ✅ Model architecture (22→32→16→2 TDNN)
- ✅ Basic supervised training loop
- ✅ Weight export to Q1.15 hex format

### What Doesn't Work / Missing
- ❌ Full CWGAN-GP training (`train.py` too complex to run)
- ❌ QAT (quantization-aware training) integration
- ❌ Spectral loss in training loop (file exists but not used)
- ❌ Validation on held-out test set
- ❌ Baseline comparison scripts
- ❌ A-SPSA loop validation

---

## Immediate Action Items (Priority Order)

### P0: Validate Current Weights
```bash
# Create validation script
python validate_weights.py \
  --weights rtl/weights/normal_*.hex \
  --test-data OpenDPD/datasets/APA_200MHz/test_*.csv \
  --metrics ACPR,EVM,NMSE \
  --output validation_report.pdf
```

**Expected output:**
- ACPR improvement: ? dB (unknown until tested)
- EVM: ? % (unknown)
- NMSE: ? dB (unknown)

**Action:** CREATE THIS SCRIPT FIRST!

### P0: Study OpenDPDv2
1. Read paper: https://arxiv.org/abs/2507.06849
2. Check their GitHub: https://github.com/lab-emi/OpenDPD
3. Document:
   - Their training epochs
   - Their loss functions
   - Their validation method

### P1: Create Volterra Baseline
```python
# Simple Volterra DPD for comparison
def train_volterra_baseline():
    # Memory polynomial model
    order = 7
    memory = 5
    
    # Least squares fitting
    weights = fit_volterra(x_in, y_out, order, memory)
    
    # Test performance
    acpr_volterra = measure_acpr(weights)
    return acpr_volterra
```

**Deliverable:** Baseline ACPR/EVM numbers for comparison

### P1: Implement A-SPSA Reference
**Create:** `validate_aspsa_loop.py` (see Step 1-4 above)

**Deliverable:** Working Python A-SPSA with tuned parameters

---

## Files Status

### Training Scripts
| File | Status | Next Action |
|------|--------|-------------|
| `train.py` | ❌ Too complex | Use as reference only |
| `train_opendpd.py` | ⚠️ Incomplete | Add QAT + spectral loss + integrate with thermal |
| `train_thermal_variants.py` | ⚠️ Simplified | Validate first, then enhance |

### Validation Scripts
| File | Status | Next Action |
|------|--------|-------------|
| `validate_model_cpu.py` | ✅ Works | Extend to measure ACPR/EVM |
| `validate.py` | ⚠️ Outdated | Update with test set metrics |
| `verify_gan_vs_supervised.py` | ❌ Missing baseline | Add Volterra comparison |
| `validate_aspsa_loop.py` | ❌ Doesn't exist | **CREATE THIS!** |

### Utilities
| File | Status | Usage |
|------|--------|-------|
| `utils/spectral_loss.py` | ✅ Ready | Add to training loop |
| `utils/quantization.py` | ✅ Works | Use for QAT |
| `utils/dataset.py` | ⚠️ Redundant? | OpenDPD has this |

---

## Success Criteria (Not Met Yet)

| Criterion | Target | Current | Status |
|-----------|--------|---------|--------|
| Test set ACPR improvement | >10 dB | Unmeasured | ❌ |
| Test set EVM | <2% | Unmeasured | ❌ |
| Better than Volterra | Yes | No baseline | ❌ |
| Thermal compensation validated | Yes | Untested | ❌ |
| QAT implemented | Yes | Post-quant only | ❌ |
| A-SPSA loop validated | Yes | No reference | ❌ |

**Overall: 0/6 criteria met**

---

## Recommendations Summary

### For ML Team
1. **STOP** creating more training variants until current weights are validated
2. **START** with validation script to measure current performance
3. **STUDY** OpenDPDv2 paper and decide on training method
4. **CREATE** A-SPSA Python reference for RTL team
5. **TRAIN** Volterra baseline for honest comparison

### For RTL Team (Waiting on ML)
1. **DO NOT** synthesize until ML confirms weights improve ACPR
2. **CREATE** testbench ready for co-simulation
3. **WAIT** for A-SPSA Python reference from ML
4. **PREPARE** bitstream scripts but don't run yet
