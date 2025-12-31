# =============================================================================
# 6G PA GAN-DPD: CWGAN-GP Discriminator (Critic)
# =============================================================================
"""
CWGAN-GP DISCRIMINATOR FOR DPD TRAINING
=======================================

This module implements the critic network for Conditional Wasserstein GAN
with Gradient Penalty (CWGAN-GP) training of the DPD model.

The discriminator:
1. Takes both PA output and condition (original signal) as input
2. Outputs a scalar score (Wasserstein distance estimate)
3. Uses spectral normalization for training stability

Architecture:
    Input [4] → FC1 [64] → LeakyReLU → FC2 [32] → LeakyReLU → FC3 [16] → LeakyReLU → FC4 [1]
    
Input: Concatenated [I_pa, Q_pa, I_cond, Q_cond]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class SpectralNormLinear(nn.Module):
    """
    Linear layer with spectral normalization.
    
    Spectral normalization constrains the Lipschitz constant of the layer
    to be at most 1, which is crucial for WGAN training stability.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.utils.spectral_norm(
            nn.Linear(in_features, out_features, bias=bias)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Discriminator(nn.Module):
    """
    CWGAN-GP Critic (Discriminator) for DPD training.
    
    The critic estimates the Wasserstein distance between:
    - Real: (ideal_dpd_output, condition)
    - Fake: (generator_output, condition)
    
    Args:
        input_dim: Input dimension (4 for concatenated IQ pairs)
        hidden_dims: Hidden layer dimensions
        leaky_slope: LeakyReLU negative slope
        use_spectral_norm: Whether to use spectral normalization
    """
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dims: List[int] = [64, 32, 16],
        leaky_slope: float = 0.2,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.leaky_slope = leaky_slope
        self.use_spectral_norm = use_spectral_norm
        
        # Build layers
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            if use_spectral_norm:
                layers.append(SpectralNormLinear(in_dim, hidden_dim))
            else:
                layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=leaky_slope))
            in_dim = hidden_dim
            
        self.features = nn.Sequential(*layers)
        
        # Output layer (no activation - Wasserstein score)
        if use_spectral_norm:
            self.output = SpectralNormLinear(hidden_dims[-1], 1)
        else:
            self.output = nn.Linear(hidden_dims[-1], 1)
            
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=self.leaky_slope)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: PA output [batch, 2] or [batch, seq, 2]
            condition: Original signal [batch, 2] or [batch, seq, 2]
            
        Returns:
            score: Wasserstein score [batch, 1] or [batch, seq, 1]
        """
        # Handle sequence input
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            x = x.reshape(-1, 2)
            condition = condition.reshape(-1, 2)
            reshape_back = True
        else:
            reshape_back = False
            batch_size, seq_len = x.shape[0], 1
            
        # Concatenate PA output and condition
        combined = torch.cat([x, condition], dim=-1)  # [batch, 4]
        
        # Feature extraction
        features = self.features(combined)
        
        # Output score
        score = self.output(features)
        
        if reshape_back:
            score = score.reshape(batch_size, seq_len, 1)
            
        return score


def compute_gradient_penalty(
    discriminator: Discriminator,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    condition: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP.
    
    The gradient penalty enforces the Lipschitz constraint by penalizing
    gradients with norm != 1 along interpolated samples.
    
    Args:
        discriminator: Critic network
        real_samples: Real (target) samples
        fake_samples: Generated (fake) samples
        condition: Conditioning input
        device: Computation device
        
    Returns:
        gradient_penalty: Scalar gradient penalty loss
    """
    # Random interpolation coefficient
    alpha = torch.rand(real_samples.size(0), 1, device=device)
    if real_samples.dim() == 3:
        alpha = alpha.unsqueeze(1)
        
    # Interpolated samples
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)
    
    # Discriminator output on interpolates
    d_interpolates = discriminator(interpolates, condition)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Flatten gradients
    gradients = gradients.reshape(gradients.size(0), -1)
    
    # Compute gradient penalty
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty


class WassersteinLoss:
    """
    Wasserstein loss for CWGAN-GP training.
    
    Discriminator loss: E[D(fake)] - E[D(real)] + λ * GP
    Generator loss: -E[D(fake)]
    """
    def __init__(self, gp_weight: float = 10.0):
        self.gp_weight = gp_weight
        
    def discriminator_loss(
        self,
        discriminator: Discriminator,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
        condition: torch.Tensor,
        device: torch.device
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute discriminator loss.
        
        Returns:
            loss: Total discriminator loss
            info: Dictionary with loss components
        """
        # Critic scores
        real_score = discriminator(real_samples, condition)
        fake_score = discriminator(fake_samples.detach(), condition)
        
        # Wasserstein distance (we want to maximize D(real) - D(fake))
        # So we minimize D(fake) - D(real) = -W_distance
        w_distance = real_score.mean() - fake_score.mean()
        d_loss = -w_distance
        
        # Gradient penalty
        gp = compute_gradient_penalty(
            discriminator, real_samples, fake_samples, condition, device
        )
        
        # Total loss
        total_loss = d_loss + self.gp_weight * gp
        
        info = {
            'd_loss': d_loss.item(),
            'gp': gp.item(),
            'w_distance': w_distance.item(),
            'real_score': real_score.mean().item(),
            'fake_score': fake_score.mean().item()
        }
        
        return total_loss, info
    
    def generator_loss(
        self,
        discriminator: Discriminator,
        fake_samples: torch.Tensor,
        condition: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute generator loss.
        
        Returns:
            loss: Generator adversarial loss
            info: Dictionary with loss components
        """
        fake_score = discriminator(fake_samples, condition)
        g_loss = -fake_score.mean()
        
        info = {
            'g_loss': g_loss.item(),
            'fake_score': fake_score.mean().item()
        }
        
        return g_loss, info


def create_discriminator(config: dict) -> Discriminator:
    """
    Factory function to create discriminator from config.
    """
    disc_config = config['model']['discriminator']
    
    return Discriminator(
        input_dim=disc_config.get('input_dim', 4),
        hidden_dims=disc_config.get('hidden_dims', [64, 32, 16]),
        leaky_slope=disc_config.get('leaky_slope', 0.2),
        use_spectral_norm=disc_config.get('use_spectral_norm', True)
    )


if __name__ == "__main__":
    print("Testing Discriminator")
    print("=" * 50)
    
    # Create discriminator
    disc = Discriminator(
        input_dim=4,
        hidden_dims=[64, 32, 16],
        use_spectral_norm=True
    )
    
    # Parameter count
    total_params = sum(p.numel() for p in disc.parameters())
    print(f"Total parameters: {total_params}")
    
    # Test forward pass
    batch_size = 8
    seq_len = 16
    
    # Sample inputs
    x = torch.randn(batch_size, seq_len, 2)
    condition = torch.randn(batch_size, seq_len, 2)
    
    print(f"\nInput shapes: x={x.shape}, condition={condition.shape}")
    
    # Forward pass
    score = disc(x, condition)
    print(f"Output score shape: {score.shape}")
    
    # Test gradient penalty
    real = torch.randn(batch_size, seq_len, 2)
    fake = torch.randn(batch_size, seq_len, 2)
    
    gp = compute_gradient_penalty(disc, real, fake, condition, torch.device('cpu'))
    print(f"\nGradient penalty: {gp.item():.4f}")
    
    # Test loss computation
    loss_fn = WassersteinLoss(gp_weight=10.0)
    
    d_loss, d_info = loss_fn.discriminator_loss(
        disc, real, fake, condition, torch.device('cpu')
    )
    print(f"\nDiscriminator loss: {d_loss.item():.4f}")
    print(f"  W-distance: {d_info['w_distance']:.4f}")
    print(f"  GP: {d_info['gp']:.4f}")
