# Full Training Flow: CWGAN-GP with QAT for PN-TDNN-DPD

**Version:** 1.0  
**Last Updated:** January 12, 2026  
**Target:** Pre-6G PA Linearization with FPGA Deployment

---

## 1. Overview

This document specifies the complete training pipeline for the Phase-Normalized TDNN-DPD using:

- **CWGAN-GP** (Conditional Wasserstein GAN with Gradient Penalty)
- **Spectral Loss** (EVM + ACPR + NMSE)
- **QAT** (Quantization-Aware Training for FPGA deployment)

### Training Objectives

| Metric | Target | Loss Component |
|--------|--------|----------------|
| ACPR | < -62 dBc | Spectral loss (ACPR term) |
| EVM | < -45 dB | Spectral loss (EVM term) |
| NMSE | < -42 dB | Reconstruction loss (L1) |
| QAT Error | < 0.5 dB | Fake quantization |

---

## 2. Data Pipeline

### 2.1 Dataset Structure

```
data/
├── train_input.csv   # PA input (clean signal u_PA)
├── train_output.csv  # PA output (distorted signal y_PA)
├── val_input.csv
├── val_output.csv
├── test_input.csv
├── test_output.csv
└── spec.json         # Signal specifications
```

### 2.2 ILA (Indirect Learning Architecture) Data Flow

**Key insight:** DPD training uses **Indirect Learning Architecture (ILA)**, NOT direct learning.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      INDIRECT LEARNING ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Training Phase (Learn the inverse):                                        │
│                                                                             │
│      y_PA (distorted) ──► DPD ──► u_pred                                   │
│                            │                                                │
│                            │  Loss = |u_pred - u_PA|                        │
│                            │  (compare to clean input)                      │
│                            ▼                                                │
│      u_PA (clean) ◄───── Target                                            │
│                                                                             │
│  Inference Phase (Apply predistortion):                                     │
│                                                                             │
│      x_in ──► DPD ──► u_dpd ──► PA ──► y_out (linearized)                  │
│                                                                             │
│  Why ILA works:                                                             │
│      DPD learns: y_PA → u_PA (inverse of PA)                               │
│      At inference: x → DPD(x) → PA(DPD(x)) ≈ linear(x)                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Data Normalization

```python
def normalize_iq(iq_complex, target_power_dbfs=-3.0):
    """
    Normalize IQ signal to target power level.
    
    Args:
        iq_complex: Complex numpy array
        target_power_dbfs: Target power in dBFS (default -3 dB for headroom)
    
    Returns:
        Normalized complex signal
    """
    current_power = np.mean(np.abs(iq_complex)**2)
    target_power = 10 ** (target_power_dbfs / 10)
    scale = np.sqrt(target_power / current_power)
    return iq_complex * scale
```

### 2.4 Feature Extraction (Phase-Normalized)

```python
class PNFeatureExtraction:
    """
    Phase-Normalized Feature Extraction matching FPGA implementation.
    
    For each sample n with memory depth M=3:
    - A(n-k) for k=0..M: Amplitude
    - A³(n-k): Cubic amplitude (odd-order PA model)
    - I_norm(n-k), Q_norm(n-k): Phase-normalized IQ
    - I(n-k), Q(n-k): Original IQ (residual path)
    
    Total: 4×(2+2+2) = 24 features
    """
    
    def __init__(self, memory_depth=3):
        self.M = memory_depth
        self.input_dim = 6 * (memory_depth + 1)  # 24 for M=3
    
    def extract(self, iq_sequence):
        """
        Extract features from IQ sequence.
        
        Args:
            iq_sequence: [batch, seq_len, 2] - I, Q
        
        Returns:
            features: [batch, seq_len - M, 24]
        """
        I, Q = iq_sequence[..., 0], iq_sequence[..., 1]
        
        # Amplitude (use true sqrt, NOT approximation!)
        A = torch.sqrt(I**2 + Q**2 + 1e-8)
        A3 = A ** 3
        
        features = []
        for n in range(self.M, iq_sequence.shape[1]):
            tap_features = []
            
            # Current sample phase factor
            I_0, Q_0 = I[:, n], Q[:, n]
            A_0 = A[:, n]
            
            for k in range(self.M + 1):
                idx = n - k
                
                # Amplitude features
                tap_features.append(A[:, idx:idx+1])
                tap_features.append(A3[:, idx:idx+1])
                
                # Phase-normalized IQ
                if k == 0:
                    # Current sample: no normalization
                    I_norm = I[:, idx:idx+1]
                    Q_norm = Q[:, idx:idx+1]
                else:
                    # Delayed: apply phase normalization
                    I_k, Q_k = I[:, idx], Q[:, idx]
                    I_norm = ((I_k * I_0 + Q_k * Q_0) / A_0).unsqueeze(-1)
                    Q_norm = ((Q_k * I_0 - I_k * Q_0) / A_0).unsqueeze(-1)
                
                tap_features.append(I_norm)
                tap_features.append(Q_norm)
                
                # Original IQ (residual path)
                tap_features.append(I[:, idx:idx+1])
                tap_features.append(Q[:, idx:idx+1])
            
            features.append(torch.cat(tap_features, dim=-1))
        
        return torch.stack(features, dim=1)
```

---

## 3. Model Architecture

### 3.1 Generator (PN-TDNN)

```python
class PNTDNNGenerator(nn.Module):
    """
    Phase-Normalized TDNN Generator for DPD.
    
    Architecture: 24 → 32 → 16 → 2
    Parameters: 1,362
    """
    
    def __init__(self, memory_depth=3, hidden_dims=[32, 16]):
        super().__init__()
        self.memory_depth = memory_depth
        self.input_dim = 6 * (memory_depth + 1)  # 24
        
        self.fex = PNFeatureExtraction(memory_depth)
        
        self.fc1 = nn.Linear(self.input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 2)
        
        self.act = nn.LeakyReLU(0.2)
        
        # QAT components (disabled by default)
        self.qat_enabled = False
        self.weight_bits = 16
        self.act_bits = 16
    
    def enable_qat(self):
        """Enable Quantization-Aware Training."""
        self.qat_enabled = True
    
    def fake_quantize(self, x, bits, is_weight=False):
        """Fake quantization with straight-through estimator."""
        if not self.qat_enabled:
            return x
        
        if is_weight:
            # Q1.15 for weights
            scale = 2 ** 15
            q_min, q_max = -32768, 32767
        else:
            # Q8.8 for activations
            scale = 2 ** 8
            q_min, q_max = -32768, 32767
        
        # Quantize and dequantize (STE for gradients)
        x_scaled = x * scale
        x_quant = torch.round(x_scaled).clamp(q_min, q_max)
        x_dequant = x_quant / scale
        
        # Straight-through: use quantized for forward, gradient unchanged
        return x_dequant.detach() + x - x.detach()
    
    def forward(self, x, pre_assembled=False):
        """
        Forward pass.
        
        Args:
            x: [batch, seq_len, 2] IQ sequence or [batch, 24] pre-assembled features
            pre_assembled: Whether input is already feature-extracted
        
        Returns:
            [batch, seq_len - M, 2] or [batch, 2] - Predistorted IQ
        """
        if not pre_assembled:
            feat = self.fex.extract(x)
            original_shape = feat.shape[:2]
            feat = feat.reshape(-1, self.input_dim)
            reshape_back = True
        else:
            feat = x
            reshape_back = False
        
        # FC1
        w1 = self.fake_quantize(self.fc1.weight, self.weight_bits, is_weight=True)
        h = F.linear(feat, w1, self.fc1.bias)
        h = self.fake_quantize(h, self.act_bits)
        h = self.act(h)
        
        # FC2
        w2 = self.fake_quantize(self.fc2.weight, self.weight_bits, is_weight=True)
        h = F.linear(h, w2, self.fc2.bias)
        h = self.fake_quantize(h, self.act_bits)
        h = self.act(h)
        
        # FC3
        w3 = self.fake_quantize(self.fc3.weight, self.weight_bits, is_weight=True)
        out = F.linear(h, w3, self.fc3.bias)
        
        # Phase denormalization
        if not pre_assembled:
            out = out.reshape(*original_shape, 2)
            out = self.phase_denorm(out, x)
        
        return out
    
    def phase_denorm(self, fc_out, x):
        """
        Apply phase denormalization to restore original phase.
        
        Args:
            fc_out: [batch, seq-M, 2] FC output
            x: [batch, seq, 2] Original input
        
        Returns:
            [batch, seq-M, 2] Phase-restored output
        """
        M = self.memory_depth
        I_0 = x[:, M:, 0]
        Q_0 = x[:, M:, 1]
        A_0 = torch.sqrt(I_0**2 + Q_0**2 + 1e-8)
        
        I_fc, Q_fc = fc_out[..., 0], fc_out[..., 1]
        
        # Complex multiply: (I_fc + jQ_fc) × (I_0 + jQ_0) / A_0
        I_out = (I_fc * I_0 - Q_fc * Q_0) / A_0
        Q_out = (I_fc * Q_0 + Q_fc * I_0) / A_0
        
        return torch.stack([I_out, Q_out], dim=-1)
```

### 3.2 Discriminator

```python
class Discriminator(nn.Module):
    """
    Conditional Discriminator for WGAN-GP.
    
    Input: IQ output [batch, 2] + condition [batch, 2]
    Output: Critic score (unbounded, not probability)
    """
    
    def __init__(self, input_dim=4, hidden_dims=[64, 32]):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[1], 1)
        )
    
    def forward(self, x, condition):
        """
        Forward pass.
        
        Args:
            x: [batch, 2] IQ output (real or fake)
            condition: [batch, 2] Conditioning input (current IQ sample)
        
        Returns:
            [batch, 1] Critic score
        """
        combined = torch.cat([x, condition], dim=-1)
        return self.net(combined)
```

---

## 4. Loss Functions

### 4.1 Wasserstein Loss with Gradient Penalty

```python
def wasserstein_loss_d(d_real, d_fake):
    """Discriminator loss: maximize D(real) - D(fake)."""
    return d_fake.mean() - d_real.mean()

def wasserstein_loss_g(d_fake):
    """Generator loss: maximize D(fake) = minimize -D(fake)."""
    return -d_fake.mean()

def gradient_penalty(discriminator, real_samples, fake_samples, condition, lambda_gp=10):
    """
    Compute gradient penalty for WGAN-GP.
    
    Enforces 1-Lipschitz constraint on discriminator.
    """
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, device=real_samples.device)
    
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)
    
    d_interpolates = discriminator(interpolates, condition)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
    gp = ((gradient_norm - 1) ** 2).mean()
    
    return lambda_gp * gp
```

### 4.2 Spectral Loss (EVM + ACPR)

```python
class SpectralLoss(nn.Module):
    """
    Combined spectral loss for DPD training.
    
    Components:
    - EVM: Error Vector Magnitude
    - ACPR: Adjacent Channel Power Ratio
    - NMSE: Normalized Mean Square Error (optional)
    """
    
    def __init__(self, sample_rate=250e6, channel_bw=200e6,
                 evm_weight=1.0, acpr_weight=0.5, nmse_weight=0.1):
        super().__init__()
        self.sample_rate = sample_rate
        self.channel_bw = channel_bw
        self.evm_weight = evm_weight
        self.acpr_weight = acpr_weight
        self.nmse_weight = nmse_weight
    
    def compute_evm(self, pred, target):
        """
        Compute EVM loss.
        
        EVM = sqrt(mean(|error|²) / mean(|target|²))
        """
        if pred.dim() == 3:
            pred_complex = pred[..., 0] + 1j * pred[..., 1]
            target_complex = target[..., 0] + 1j * target[..., 1]
        else:
            pred_complex = pred
            target_complex = target
        
        error = pred_complex - target_complex
        error_power = (error.abs() ** 2).mean()
        ref_power = (target_complex.abs() ** 2).mean().clamp(min=1e-10)
        
        evm = torch.sqrt(error_power / ref_power)
        return evm
    
    def compute_acpr(self, signal):
        """
        Compute ACPR.
        
        ACPR = P_adjacent / P_main_channel
        """
        if signal.dim() == 3:
            signal_complex = signal[..., 0] + 1j * signal[..., 1]
        else:
            signal_complex = signal
        
        # Flatten to [batch, seq]
        if signal_complex.dim() == 3:
            signal_complex = signal_complex.reshape(-1, signal_complex.shape[-1])
        
        seq_len = signal_complex.shape[-1]
        
        # FFT
        spectrum = torch.fft.fft(signal_complex, dim=-1)
        power_spectrum = (spectrum.abs() ** 2) / seq_len
        
        # Frequency bins
        freq_bins = torch.fft.fftfreq(seq_len, d=1/self.sample_rate)
        freq_bins = freq_bins.to(signal.device)
        
        # Channel masks
        main_mask = freq_bins.abs() <= self.channel_bw / 2
        adj_offset = self.channel_bw  # Adjacent channel offset
        adj_bw = self.channel_bw / 2  # Adjacent channel BW
        
        lower_adj_mask = (freq_bins >= -(adj_offset + adj_bw)) & \
                         (freq_bins <= -(adj_offset - adj_bw))
        upper_adj_mask = (freq_bins >= (adj_offset - adj_bw)) & \
                         (freq_bins <= (adj_offset + adj_bw))
        
        # Power computation
        main_power = (power_spectrum * main_mask).sum(dim=-1).clamp(min=1e-10)
        adj_power = (power_spectrum * (lower_adj_mask | upper_adj_mask)).sum(dim=-1)
        
        acpr = adj_power / main_power
        return acpr.mean()
    
    def forward(self, pred, target):
        """
        Compute combined spectral loss.
        
        Args:
            pred: Predicted IQ [batch, seq, 2]
            target: Target IQ [batch, seq, 2]
        
        Returns:
            Scalar loss value
        """
        evm = self.compute_evm(pred, target)
        acpr = self.compute_acpr(pred)
        
        # NMSE
        error = pred - target
        nmse = (error ** 2).sum() / (target ** 2).sum().clamp(min=1e-10)
        
        loss = (self.evm_weight * evm + 
                self.acpr_weight * acpr + 
                self.nmse_weight * nmse)
        
        return loss, {'evm': evm.item(), 'acpr': acpr.item(), 'nmse': nmse.item()}
```

### 4.3 Combined Generator Loss

```python
def generator_loss(d_fake, pred, target, spectral_loss_fn,
                   lambda_adv=1.0, lambda_spec=10.0, lambda_l1=1.0):
    """
    Combined generator loss.
    
    L_G = λ_adv * L_WGAN + λ_spec * L_spectral + λ_l1 * L_L1
    """
    # Adversarial loss
    l_adv = wasserstein_loss_g(d_fake)
    
    # Spectral loss
    l_spec, spec_metrics = spectral_loss_fn(pred, target)
    
    # L1 reconstruction loss
    l_l1 = F.l1_loss(pred, target)
    
    # Combined
    loss = lambda_adv * l_adv + lambda_spec * l_spec + lambda_l1 * l_l1
    
    metrics = {
        'l_adv': l_adv.item(),
        'l_spec': l_spec.item(),
        'l_l1': l_l1.item(),
        **spec_metrics
    }
    
    return loss, metrics
```

---

## 5. Training Loop

### 5.1 Two-Stage Training Schedule

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TWO-STAGE TRAINING SCHEDULE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Stage 1: Float32 Training (Epochs 1-300)                                   │
│  ├── Purpose: Learn DPD inverse mapping                                     │
│  ├── LR: 1e-4 → 1e-5 (cosine decay)                                        │
│  ├── D:G ratio: 5:1 (WGAN-GP standard)                                     │
│  └── Target: ACPR < -55 dBc, EVM < -40 dB                                  │
│                                                                             │
│  Stage 2: QAT Fine-tuning (Epochs 301-500)                                  │
│  ├── Purpose: Adapt to fixed-point quantization                            │
│  ├── LR: 1e-5 → 1e-6 (cosine decay)                                        │
│  ├── Enable fake quantization                                              │
│  └── Target: <0.5 dB degradation from Stage 1                              │
│                                                                             │
│  Total: 500 epochs                                                          │
│  Expected time: ~4-6 hours on T4 GPU (Colab)                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Training Step

```python
def train_step(generator, discriminator, batch,
               g_optimizer, d_optimizer, spectral_loss_fn,
               config, step, qat_enabled=False):
    """
    Single training step.
    
    Args:
        batch: (y_pa, u_pa) - PA output and input
        step: Current step number
    
    Returns:
        Dictionary of metrics
    """
    y_pa, u_pa = batch  # y_pa: distorted, u_pa: clean target
    device = y_pa.device
    
    # ===== Discriminator Update (n_critic times) =====
    for _ in range(config['n_critic']):
        d_optimizer.zero_grad()
        
        # Generate fake samples
        with torch.no_grad():
            u_pred = generator(y_pa)  # DPD(y_PA) → u_pred
        
        # Condition on current sample (for conditional GAN)
        M = generator.memory_depth
        condition = y_pa[:, M:, :].reshape(-1, 2)  # Current IQ
        real_flat = u_pa[:, M:, :].reshape(-1, 2)
        fake_flat = u_pred.reshape(-1, 2)
        
        # Discriminator scores
        d_real = discriminator(real_flat, condition)
        d_fake = discriminator(fake_flat, condition)
        
        # Wasserstein loss + GP
        d_loss = wasserstein_loss_d(d_real, d_fake)
        gp = gradient_penalty(discriminator, real_flat, fake_flat, condition)
        
        (d_loss + gp).backward()
        d_optimizer.step()
    
    # ===== Generator Update =====
    g_optimizer.zero_grad()
    
    u_pred = generator(y_pa)
    
    fake_flat = u_pred.reshape(-1, 2)
    d_fake = discriminator(fake_flat, condition)
    
    u_target = u_pa[:, M:, :]
    g_loss, metrics = generator_loss(
        d_fake, u_pred, u_target, spectral_loss_fn,
        lambda_adv=config['lambda_adv'],
        lambda_spec=config['lambda_spec'],
        lambda_l1=config['lambda_l1']
    )
    
    g_loss.backward()
    g_optimizer.step()
    
    metrics['d_loss'] = d_loss.item()
    metrics['gp'] = gp.item()
    metrics['g_loss'] = g_loss.item()
    
    return metrics
```

### 5.3 QAT Transition

```python
def transition_to_qat(generator, g_optimizer, epoch, config):
    """
    Transition from float32 to QAT training.
    
    Called at epoch = config['qat_start_epoch'] (default: 300)
    """
    print(f"[Epoch {epoch}] Enabling QAT (Q1.15 weights, Q8.8 activations)")
    
    # Enable fake quantization
    generator.enable_qat()
    
    # Reduce learning rate for fine-tuning
    for param_group in g_optimizer.param_groups:
        param_group['lr'] = config['qat_lr']  # 1e-5 → 1e-6
    
    return generator, g_optimizer
```

---

## 6. Hyperparameters

### 6.1 Default Configuration

```yaml
# config/config.yaml

# Model
model:
  memory_depth: 3
  hidden_dims: [32, 16]
  leaky_slope: 0.2

# Data
data:
  sample_rate: 250e6
  channel_bw: 200e6
  batch_size: 64
  seq_length: 64

# Training
training:
  epochs: 500
  n_critic: 5  # D updates per G update
  
  # Learning rates
  lr_g: 1e-4
  lr_d: 1e-4
  beta1: 0.0   # WGAN-GP: no momentum
  beta2: 0.9
  
  # Loss weights
  lambda_adv: 1.0
  lambda_spec: 10.0
  lambda_l1: 1.0
  lambda_gp: 10.0

# QAT
qat:
  enabled: true
  start_epoch: 300
  qat_lr: 1e-5
  weight_bits: 16  # Q1.15
  act_bits: 16     # Q8.8

# Spectral loss
spectral_loss:
  evm_weight: 1.0
  acpr_weight: 0.5
  nmse_weight: 0.1
```

### 6.2 Hyperparameter Justification

| Parameter | Value | Reason |
|-----------|-------|--------|
| `n_critic=5` | WGAN-GP standard; ensures D learns faster than G |
| `lambda_gp=10` | Original WGAN-GP paper; enforces 1-Lipschitz |
| `lambda_spec=10` | High weight on spectral loss; prioritizes RF metrics |
| `beta1=0.0` | WGAN-GP recommendation; no momentum for stability |
| `qat_start_epoch=300` | Start QAT after G converges; avoid early quantization noise |
| `batch_size=64` | Balance between gradient variance and memory |

---

## 7. Validation & Export

### 7.1 Validation Metrics

```python
def validate(generator, val_loader, spectral_loss_fn, device):
    """
    Validate model on held-out data.
    
    Returns:
        Dictionary of validation metrics
    """
    generator.eval()
    
    metrics_sum = defaultdict(float)
    n_batches = 0
    
    with torch.no_grad():
        for y_pa, u_pa in val_loader:
            y_pa, u_pa = y_pa.to(device), u_pa.to(device)
            
            u_pred = generator(y_pa)
            M = generator.memory_depth
            u_target = u_pa[:, M:, :]
            
            _, batch_metrics = spectral_loss_fn(u_pred, u_target)
            
            for k, v in batch_metrics.items():
                metrics_sum[k] += v
            n_batches += 1
    
    generator.train()
    
    return {k: v / n_batches for k, v in metrics_sum.items()}
```

### 7.2 Weight Export for FPGA

```python
def export_weights_for_fpga(generator, output_dir):
    """
    Export quantized weights in binary format for FPGA.
    
    File format per layer:
    - weights.bin: int16 (Q1.15), row-major
    - biases.bin: int16 (Q1.15)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scale_weight = 2 ** 15  # Q1.15
    
    for name, module in generator.named_modules():
        if isinstance(module, nn.Linear):
            # Quantize weights
            w = module.weight.data
            w_quant = torch.round(w * scale_weight).clamp(-32768, 32767).to(torch.int16)
            
            # Quantize biases
            b = module.bias.data
            b_quant = torch.round(b * scale_weight).clamp(-32768, 32767).to(torch.int16)
            
            # Save as binary
            w_file = output_dir / f"{name.replace('.', '_')}_weights.bin"
            b_file = output_dir / f"{name.replace('.', '_')}_biases.bin"
            
            w_quant.numpy().tofile(w_file)
            b_quant.numpy().tofile(b_file)
            
            print(f"Exported {name}: weights {w.shape}, biases {b.shape}")
    
    print(f"Weights exported to {output_dir}")
```

---

## 8. Expected Results

### 8.1 Training Curves (Typical)

```
Epoch   D_loss   G_loss   EVM(dB)   ACPR(dBc)   NMSE(dB)
--------------------------------------------------------------
  50    -0.82    2.34     -28.5     -42.3       -25.1
 100    -1.05    1.98     -35.2     -50.1       -31.4
 200    -1.23    1.45     -41.8     -56.2       -38.7
 300    -1.31    1.12     -44.5     -60.1       -41.2   ← QAT starts
 350    -1.28    1.18     -43.8     -59.4       -40.5   ← QAT adapting
 400    -1.30    1.08     -44.2     -60.8       -41.0
 500    -1.32    1.02     -45.1     -62.3       -42.1   ← Final
```

### 8.2 Final Performance (Target)

| Metric | Target | Expected | QAT Degradation |
|--------|--------|----------|-----------------|
| ACPR | < -62 dBc | -62 to -65 dBc | < 0.5 dB |
| EVM | < -45 dB | -45 to -50 dB | < 0.3 dB |
| NMSE | < -42 dB | -42 to -45 dB | < 0.2 dB |

---

## 9. Troubleshooting

### 9.1 Common Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| Mode collapse | G loss stuck, D loss → 0 | Reduce `lambda_adv`, increase `n_critic` |
| Poor ACPR | ACPR > -50 dBc | Increase `lambda_spec`, add ACPR-specific term |
| QAT divergence | Metrics degrade after QAT start | Lower QAT LR, start QAT later |
| NaN loss | Loss becomes NaN | Add gradient clipping, check data normalization |

### 9.2 Gradient Clipping

```python
# Add after loss.backward(), before optimizer.step()
torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
```

---

## 10. References

1. Arjovsky et al., "Wasserstein GAN" (2017)
2. Gulrajani et al., "Improved Training of Wasserstein GANs" (2017) - WGAN-GP
3. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (2018) - QAT
4. OpenDPDv2, SparseDPD papers for DPD-specific metrics
