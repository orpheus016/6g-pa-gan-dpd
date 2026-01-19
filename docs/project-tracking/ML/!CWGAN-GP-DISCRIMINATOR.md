# Discriminator Architecture for CWGAN-GP DPD Training

**Version:** 1.0  
**Last Updated:** January 12, 2026  
**Reference:** Gulrajani et al. (2017), FULL-TRAINING-FLOW.md

---

## 1. Why Use a Discriminator for DPD?

### 1.1 Limitations of MSE-Only Training

Traditional DPD training minimizes Mean Squared Error (MSE) between predicted and target outputs:

$$\mathcal{L}_{MSE} = \mathbb{E}\left[ \| \hat{u} - u \|^2 \right]$$

**Problems:**
1. **Blurry outputs:** MSE favors average solutions, smoothing sharp spectral features
2. **Poor perceptual quality:** Does not capture distribution of "good" DPD outputs
3. **ACPR insensitivity:** MSE doesn't directly penalize spectral regrowth

### 1.2 GAN Advantage

A discriminator learns to distinguish "real" (well-linearized) from "fake" (poorly-linearized) outputs. This provides:
- **Distribution matching:** Generator learns to produce realistic linearized outputs
- **Spectral awareness:** Combined with spectral loss, discriminator guides toward low-ACPR solutions
- **Robustness:** Adversarial training prevents overfitting to training set artifacts

**Key insight:** The discriminator is **not deployed on FPGA**—only the generator is. This allows the discriminator to be arbitrarily complex during training.

---

## 2. WGAN-GP Framework

### 2.1 Why Wasserstein over Standard GAN?

| Aspect | Standard GAN | WGAN-GP |
|--------|--------------|---------|
| **Loss** | JS divergence | Wasserstein (Earth Mover) distance |
| **Training** | Minimax game | Minimize Wasserstein distance |
| **Stability** | Mode collapse common | Stable convergence |
| **Discriminator output** | Probability [0, 1] | Critic score (unbounded) |
| **Gradient** | Vanishing gradients | Meaningful gradients everywhere |

**Source:** Arjovsky et al., "Wasserstein GAN," ICML 2017

### 2.2 Wasserstein Distance

The Wasserstein-1 distance (Earth Mover's distance):

$$W(\mathbb{P}_r, \mathbb{P}_g) = \inf_{\gamma \in \Pi(\mathbb{P}_r, \mathbb{P}_g)} \mathbb{E}_{(x,y) \sim \gamma} \left[ \|x - y\| \right]$$

Under 1-Lipschitz constraint, this becomes:

$$W(\mathbb{P}_r, \mathbb{P}_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim \mathbb{P}_r}[f(x)] - \mathbb{E}_{x \sim \mathbb{P}_g}[f(x)]$$

**Meaning:** The discriminator (critic) $f$ estimates how far apart the real and generated distributions are.

### 2.3 Gradient Penalty (GP)

Instead of weight clipping (WGAN original), use gradient penalty to enforce 1-Lipschitz:

$$\mathcal{L}_{GP} = \lambda_{GP} \cdot \mathbb{E}_{\hat{x} \sim \mathbb{P}_{\hat{x}}} \left[ \left( \|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1 \right)^2 \right]$$

where $\hat{x}$ is sampled uniformly along lines between real and fake samples:

$$\hat{x} = \alpha \cdot x_{real} + (1 - \alpha) \cdot x_{fake}, \quad \alpha \sim U(0, 1)$$

**Source:** Gulrajani et al., "Improved Training of Wasserstein GANs," NeurIPS 2017

---

## 3. Discriminator Architecture

### 3.1 Network Design

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
            nn.Linear(hidden_dims[1], 1),
            # No sigmoid! WGAN outputs unbounded critic score
        )
    
    def forward(self, x, condition):
        """
        Args:
            x: DPD output [batch, 2] (I, Q)
            condition: Input signal [batch, 2] (I_in, Q_in)
        Returns:
            Critic score [batch, 1]
        """
        combined = torch.cat([x, condition], dim=-1)  # [batch, 4]
        return self.net(combined)
```

### 3.2 Why Conditional Discriminator?

The discriminator receives **both** the DPD output **and** the input signal:

$$D(y_{DPD}, x_{input}) \rightarrow \text{score}$$

**Reason:** The discriminator must judge whether the output is a valid predistortion **for the given input**. Without conditioning, the discriminator only learns marginal distribution of outputs, not the input-output relationship.

### 3.3 Architecture Justification

| Choice | Reason |
|--------|--------|
| **4-dim input** | 2 (output IQ) + 2 (condition IQ) |
| **64→32→1** | Simple enough to train fast, complex enough to discriminate |
| **LeakyReLU(0.2)** | Prevents dead neurons, WGAN-GP standard |
| **No BatchNorm** | WGAN-GP recommends LayerNorm or none |
| **No sigmoid** | WGAN outputs unbounded critic score |

**Note:** The discriminator can be much larger (e.g., 128→64→32→1) since it's only used during training, not deployed on FPGA.

---

## 4. Loss Functions

### 4.1 Discriminator Loss

The discriminator maximizes the distance between real and fake:

$$\mathcal{L}_D = \underbrace{\mathbb{E}_{x \sim fake}[D(x)]}_{\text{push down fake}} - \underbrace{\mathbb{E}_{x \sim real}[D(x)]}_{\text{push up real}} + \underbrace{\mathcal{L}_{GP}}_{\text{enforce Lipschitz}}$$

```python
def wasserstein_loss_d(d_real, d_fake):
    """Discriminator loss: maximize D(real) - D(fake)."""
    return d_fake.mean() - d_real.mean()
```

### 4.2 Generator Loss

The generator minimizes the negative critic score:

$$\mathcal{L}_G^{adv} = -\mathbb{E}_{x \sim fake}[D(x)]$$

```python
def wasserstein_loss_g(d_fake):
    """Generator loss: maximize D(fake) = minimize -D(fake)."""
    return -d_fake.mean()
```

### 4.3 Gradient Penalty Implementation

```python
def gradient_penalty(discriminator, real_samples, fake_samples, condition, lambda_gp=10):
    """
    Compute gradient penalty for WGAN-GP.
    """
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, device=real_samples.device)
    
    # Interpolate between real and fake
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)
    
    # Get discriminator output
    d_interpolates = discriminator(interpolates, condition)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Gradient penalty: (||grad|| - 1)^2
    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return lambda_gp * gp
```

---

## 5. Spectral Loss Integration

### 5.1 Why Spectral Loss?

The discriminator alone doesn't guarantee good ACPR/EVM. We add explicit spectral loss to directly optimize RF metrics.

### 5.2 Spectral Loss Components

$$\mathcal{L}_{spectral} = \lambda_{EVM} \cdot \mathcal{L}_{EVM} + \lambda_{ACPR} \cdot \mathcal{L}_{ACPR} + \lambda_{NMSE} \cdot \mathcal{L}_{NMSE}$$

**EVM Loss (differentiable):**

$$\text{EVM}_{dB} = 10 \log_{10} \left( \frac{\sum |y_{pred} - y_{ideal}|^2}{\sum |y_{ideal}|^2} \right)$$

**ACPR Loss (differentiable approximation):**

$$\text{ACPR}_{dB} = 10 \log_{10} \left( \frac{P_{adjacent}}{P_{main}} \right)$$

Computed via FFT with learnable band masks.

**NMSE Loss:**

$$\text{NMSE}_{dB} = 10 \log_{10} \left( \frac{\sum |y_{pred} - y_{target}|^2}{\sum |y_{target}|^2} \right)$$

### 5.3 Combined Generator Loss

$$\mathcal{L}_G = \underbrace{\lambda_{adv} \cdot \mathcal{L}_G^{adv}}_{\text{adversarial}} + \underbrace{\lambda_{spec} \cdot \mathcal{L}_{spectral}}_{\text{spectral}} + \underbrace{\lambda_{L1} \cdot \|y_{pred} - y_{target}\|_1}_{\text{reconstruction}}$$

**Typical weights:**
- $\lambda_{adv} = 1.0$
- $\lambda_{spec} = 10.0$ (high to prioritize RF metrics)
- $\lambda_{L1} = 1.0$

---

## 6. Training Protocol

### 6.1 Critic-to-Generator Ratio

WGAN-GP recommends training discriminator more often than generator:

$$n_{critic} = 5$$

For every 5 discriminator updates, perform 1 generator update.

**Reason:** The discriminator must provide accurate Wasserstein distance estimate before generator can improve. Under-trained discriminator gives poor gradients.

### 6.2 Training Loop

```python
for epoch in range(epochs):
    for batch in dataloader:
        # === Train Discriminator (n_critic times) ===
        for _ in range(n_critic):
            # Forward pass
            fake = generator(batch['input'])
            real = batch['target']
            
            d_fake = discriminator(fake.detach(), batch['input'])
            d_real = discriminator(real, batch['input'])
            
            # Discriminator loss
            d_loss = wasserstein_loss_d(d_real, d_fake)
            gp = gradient_penalty(discriminator, real, fake, batch['input'])
            d_total = d_loss + gp
            
            # Update discriminator
            d_optimizer.zero_grad()
            d_total.backward()
            d_optimizer.step()
        
        # === Train Generator (1 time) ===
        fake = generator(batch['input'])
        d_fake = discriminator(fake, batch['input'])
        
        # Generator loss (adversarial + spectral + L1)
        g_adv = wasserstein_loss_g(d_fake)
        g_spec = spectral_loss(fake, batch['target'])
        g_l1 = F.l1_loss(fake, batch['target'])
        
        g_total = lambda_adv * g_adv + lambda_spec * g_spec + lambda_l1 * g_l1
        
        # Update generator
        g_optimizer.zero_grad()
        g_total.backward()
        g_optimizer.step()
```

### 6.3 Optimizer Settings

| Parameter | Discriminator | Generator |
|-----------|---------------|-----------|
| Optimizer | Adam | Adam |
| Learning rate | 1e-4 | 1e-4 |
| $\beta_1$ | 0.0 | 0.0 |
| $\beta_2$ | 0.9 | 0.9 |

**Why $\beta_1 = 0$?** WGAN-GP authors recommend no momentum for stability.

---

## 7. Why the Discriminator Works for DPD

### 7.1 Distribution Perspective

The discriminator learns the **distribution of well-linearized outputs**:
- Real samples: PA outputs that would result from ideal DPD
- Fake samples: Current generator's DPD outputs

The generator is pushed to produce outputs that **look like** well-linearized signals, not just minimize point-wise error.

### 7.2 Mode Coverage

Unlike MSE which averages over modes, adversarial training encourages **mode coverage**:
- Generator must produce diverse, realistic outputs
- Prevents collapse to average solution
- Results in sharper spectral characteristics

### 7.3 Gradient Quality

WGAN-GP provides **meaningful gradients everywhere**:
- Standard GAN: Gradients vanish when discriminator is too good
- WGAN-GP: Wasserstein distance always provides useful gradient signal

**Source:** Arjovsky et al. prove that Wasserstein distance is continuous and differentiable almost everywhere.

---

## 8. Discriminator NOT Deployed

**Critical point:** Only the Generator (PN-TDNN) is deployed on FPGA.

| Component | Training | Inference (FPGA) |
|-----------|----------|------------------|
| Generator (PN-TDNN) | ✅ | ✅ |
| Discriminator | ✅ | ❌ |
| Spectral Loss | ✅ | ❌ |

This means:
- Discriminator can use any architecture (FP32, deep networks, batch norm)
- No quantization needed for discriminator
- Discriminator complexity doesn't affect deployment cost

---

## 9. Summary

| Design Choice | Reason | Source |
|---------------|--------|--------|
| WGAN-GP | Stable training, meaningful gradients | Gulrajani 2017 |
| Conditional | Discriminate input-output relationship | Standard CGAN |
| $n_{critic} = 5$ | Train D more for accurate Wasserstein estimate | WGAN-GP paper |
| $\lambda_{GP} = 10$ | Enforce 1-Lipschitz without weight clipping | WGAN-GP paper |
| Spectral loss | Directly optimize RF metrics (ACPR, EVM) | DPD domain knowledge |
| $\beta_1 = 0$ | No momentum for WGAN stability | WGAN-GP recommendation |

---

## References

1. Arjovsky et al., "Wasserstein GAN," ICML 2017
2. Gulrajani et al., "Improved Training of Wasserstein GANs," NeurIPS 2017
3. Mirza & Osindero, "Conditional Generative Adversarial Nets," 2014
4. OpenDPDv2: arXiv:2507.06849v2 (end-to-end DPD learning)
5. SparseDPD: arXiv:2506.16591v1 (FPGA DPD metrics)
