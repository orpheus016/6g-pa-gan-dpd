# =============================================================================
# 6G PA GAN-DPD: Spectral Loss Functions (EVM, ACPR)
# =============================================================================
"""
SPECTRAL LOSS FUNCTIONS FOR DPD TRAINING
========================================

This module implements spectral-domain loss functions critical for DPD:
1. EVM (Error Vector Magnitude) - measures constellation distortion
2. ACPR (Adjacent Channel Power Ratio) - measures spectral regrowth
3. NMSE (Normalized Mean Square Error) - measures overall distortion

These losses ensure the GAN generator produces outputs that not only
fool the discriminator but also meet RF performance requirements.
"""

import torch
import torch.nn as nn
import torch.fft as fft
from typing import Dict, Tuple, Optional
import numpy as np


def compute_evm(
    measured: torch.Tensor,
    reference: torch.Tensor,
    return_db: bool = True
) -> torch.Tensor:
    """
    Compute Error Vector Magnitude (EVM).
    
    EVM = sqrt(mean(|measured - reference|²) / mean(|reference|²))
    
    Args:
        measured: Measured/predicted IQ signal [batch, seq, 2]
        reference: Reference/ideal IQ signal [batch, seq, 2]
        return_db: Return EVM in dB (default) or linear
        
    Returns:
        EVM value (scalar or per-batch)
    """
    # Convert to complex
    if measured.dim() == 3 and measured.shape[-1] == 2:
        meas_complex = torch.complex(measured[..., 0], measured[..., 1])
        ref_complex = torch.complex(reference[..., 0], reference[..., 1])
    else:
        meas_complex = measured
        ref_complex = reference
        
    # Error vector
    error = meas_complex - ref_complex
    
    # EVM calculation
    error_power = (error.abs() ** 2).mean(dim=-1)
    ref_power = (ref_complex.abs() ** 2).mean(dim=-1)
    
    # Avoid division by zero
    ref_power = torch.clamp(ref_power, min=1e-10)
    
    evm_linear = torch.sqrt(error_power / ref_power)
    
    if return_db:
        return 20 * torch.log10(evm_linear + 1e-10)
    return evm_linear


def compute_acpr(
    signal: torch.Tensor,
    sample_rate: float,
    channel_bw: float,
    adjacent_offset: float,
    return_db: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Adjacent Channel Power Ratio (ACPR).
    
    ACPR = P_adjacent / P_main_channel
    
    Args:
        signal: IQ signal [batch, seq, 2] or complex [batch, seq]
        sample_rate: Sample rate in Hz
        channel_bw: Main channel bandwidth in Hz
        adjacent_offset: Offset to adjacent channel in Hz
        return_db: Return ACPR in dB
        
    Returns:
        acpr_lower: ACPR for lower adjacent channel
        acpr_upper: ACPR for upper adjacent channel
    """
    # Convert to complex
    if signal.dim() == 3 and signal.shape[-1] == 2:
        signal_complex = torch.complex(signal[..., 0], signal[..., 1])
    else:
        signal_complex = signal
        
    batch_size, seq_len = signal_complex.shape
    
    # Compute FFT
    spectrum = fft.fft(signal_complex, dim=-1)
    power_spectrum = (spectrum.abs() ** 2) / seq_len
    
    # Frequency bins
    freq_bins = fft.fftfreq(seq_len, d=1/sample_rate)
    
    # Define channel masks
    main_mask = torch.abs(freq_bins) <= channel_bw / 2
    lower_adj_mask = (freq_bins >= -(adjacent_offset + channel_bw/2)) & \
                     (freq_bins <= -(adjacent_offset - channel_bw/2))
    upper_adj_mask = (freq_bins >= (adjacent_offset - channel_bw/2)) & \
                     (freq_bins <= (adjacent_offset + channel_bw/2))
    
    # Move masks to device
    main_mask = main_mask.to(signal.device)
    lower_adj_mask = lower_adj_mask.to(signal.device)
    upper_adj_mask = upper_adj_mask.to(signal.device)
    
    # Compute powers
    main_power = (power_spectrum * main_mask).sum(dim=-1)
    lower_adj_power = (power_spectrum * lower_adj_mask).sum(dim=-1)
    upper_adj_power = (power_spectrum * upper_adj_mask).sum(dim=-1)
    
    # Avoid division by zero
    main_power = torch.clamp(main_power, min=1e-10)
    
    acpr_lower = lower_adj_power / main_power
    acpr_upper = upper_adj_power / main_power
    
    if return_db:
        acpr_lower = 10 * torch.log10(acpr_lower + 1e-10)
        acpr_upper = 10 * torch.log10(acpr_upper + 1e-10)
        
    return acpr_lower, acpr_upper


def compute_nmse(
    measured: torch.Tensor,
    reference: torch.Tensor,
    return_db: bool = True
) -> torch.Tensor:
    """
    Compute Normalized Mean Square Error (NMSE).
    
    NMSE = mean(|measured - reference|²) / mean(|reference|²)
    
    Args:
        measured: Measured/predicted signal
        reference: Reference/ideal signal
        return_db: Return in dB
        
    Returns:
        NMSE value
    """
    # Flatten if needed
    if measured.dim() > 2:
        measured = measured.reshape(measured.shape[0], -1)
        reference = reference.reshape(reference.shape[0], -1)
        
    error = measured - reference
    error_power = (error ** 2).sum(dim=-1)
    ref_power = (reference ** 2).sum(dim=-1)
    
    ref_power = torch.clamp(ref_power, min=1e-10)
    nmse = error_power / ref_power
    
    if return_db:
        return 10 * torch.log10(nmse + 1e-10)
    return nmse


class SpectralLoss(nn.Module):
    """
    Combined spectral loss for DPD training.
    
    Combines:
    - L1 reconstruction loss
    - EVM loss
    - ACPR loss
    - Spectral flatness loss (optional)
    
    Args:
        sample_rate: Signal sample rate in Hz
        channel_bw: Channel bandwidth in Hz
        adjacent_offset: Adjacent channel offset in Hz
        evm_weight: Weight for EVM loss
        acpr_weight: Weight for ACPR loss
        l1_weight: Weight for L1 reconstruction loss
    """
    def __init__(
        self,
        sample_rate: float = 200e6,
        channel_bw: float = 100e6,
        adjacent_offset: float = 100e6,
        evm_weight: float = 20.0,
        acpr_weight: float = 10.0,
        l1_weight: float = 50.0
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.channel_bw = channel_bw
        self.adjacent_offset = adjacent_offset
        
        self.evm_weight = evm_weight
        self.acpr_weight = acpr_weight
        self.l1_weight = l1_weight
        
        self.l1_loss = nn.L1Loss()
        
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute combined spectral loss.
        
        Args:
            predicted: Predicted/generated signal [batch, seq, 2]
            target: Target/reference signal [batch, seq, 2]
            return_components: Also return individual loss components
            
        Returns:
            total_loss: Combined loss (or tuple with components if return_components)
        """
        losses = {}
        
        # L1 reconstruction loss
        l1 = self.l1_loss(predicted, target)
        losses['l1'] = l1
        
        # EVM loss (we want to minimize EVM, which is in dB and typically negative for good signals)
        # Convert to positive loss: higher EVM (less negative) = higher loss
        evm_db = compute_evm(predicted, target, return_db=True)
        # Normalize: -40dB EVM is excellent, 0dB is bad
        # Loss = (EVM_dB + 40) / 40, clamped to [0, 2]
        evm_loss = torch.clamp((evm_db.mean() + 40) / 40, 0, 2)
        losses['evm'] = evm_loss
        
        # ACPR loss (similar normalization)
        # Good ACPR is < -45dB, we want to penalize higher values
        acpr_lower, acpr_upper = compute_acpr(
            predicted, self.sample_rate, self.channel_bw, self.adjacent_offset
        )
        acpr_max = torch.max(acpr_lower.mean(), acpr_upper.mean())
        # Loss = (ACPR_dB + 50) / 50, clamped to [0, 2]
        acpr_loss = torch.clamp((acpr_max + 50) / 50, 0, 2)
        losses['acpr'] = acpr_loss
        
        # Total loss
        total = (self.l1_weight * l1 + 
                 self.evm_weight * evm_loss + 
                 self.acpr_weight * acpr_loss)
        
        if return_components:
            return total, losses
        return total
    
    def compute_metrics(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute all spectral metrics for evaluation.
        
        Returns:
            Dictionary of metrics
        """
        with torch.no_grad():
            metrics = {}
            
            # EVM
            evm = compute_evm(predicted, target, return_db=True)
            metrics['evm_db'] = evm.mean().item()
            
            # NMSE
            nmse = compute_nmse(predicted, target, return_db=True)
            metrics['nmse_db'] = nmse.mean().item()
            
            # ACPR
            acpr_l, acpr_u = compute_acpr(
                predicted, self.sample_rate, self.channel_bw, self.adjacent_offset
            )
            metrics['acpr_lower_db'] = acpr_l.mean().item()
            metrics['acpr_upper_db'] = acpr_u.mean().item()
            metrics['acpr_max_db'] = max(metrics['acpr_lower_db'], metrics['acpr_upper_db'])
            
            # L1 error
            l1 = torch.nn.functional.l1_loss(predicted, target)
            metrics['l1_error'] = l1.item()
            
            return metrics


class EVMLoss(nn.Module):
    """
    Standalone EVM loss for fine-tuning.
    
    Directly minimizes EVM in dB scale.
    """
    def __init__(self, target_evm_db: float = -35.0):
        super().__init__()
        self.target_evm_db = target_evm_db
        
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        evm_db = compute_evm(predicted, target, return_db=True)
        # Loss is distance from target EVM
        loss = torch.relu(evm_db.mean() - self.target_evm_db)
        return loss


class ACPRLoss(nn.Module):
    """
    Standalone ACPR loss for fine-tuning.
    
    Penalizes adjacent channel leakage.
    """
    def __init__(
        self,
        sample_rate: float = 200e6,
        channel_bw: float = 100e6,
        adjacent_offset: float = 100e6,
        target_acpr_db: float = -45.0
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.channel_bw = channel_bw
        self.adjacent_offset = adjacent_offset
        self.target_acpr_db = target_acpr_db
        
    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        acpr_l, acpr_u = compute_acpr(
            signal, self.sample_rate, self.channel_bw, self.adjacent_offset
        )
        acpr_max = torch.max(acpr_l.mean(), acpr_u.mean())
        # Loss is distance from target ACPR
        loss = torch.relu(acpr_max - self.target_acpr_db)
        return loss


def create_spectral_loss(config: dict) -> SpectralLoss:
    """Factory function to create spectral loss from config."""
    system_config = config.get('system', {})
    loss_config = config.get('training', {}).get('loss', {})
    
    return SpectralLoss(
        sample_rate=system_config.get('sample_rate', 200e6),
        channel_bw=system_config.get('sample_rate', 200e6) / 2,  # Nyquist
        adjacent_offset=system_config.get('sample_rate', 200e6) / 2,
        evm_weight=loss_config.get('spectral_evm', 20.0),
        acpr_weight=loss_config.get('spectral_acpr', 10.0),
        l1_weight=loss_config.get('reconstruction_l1', 50.0)
    )


if __name__ == "__main__":
    print("Testing Spectral Loss Functions")
    print("=" * 50)
    
    # Create test signals
    batch_size = 4
    seq_len = 1024
    sample_rate = 200e6
    
    # Reference signal (clean)
    t = torch.linspace(0, seq_len / sample_rate, seq_len)
    freq = 50e6  # 50 MHz tone
    ref_i = torch.cos(2 * np.pi * freq * t)
    ref_q = torch.sin(2 * np.pi * freq * t)
    reference = torch.stack([ref_i, ref_q], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Measured signal (with distortion)
    noise = torch.randn_like(reference) * 0.1
    distortion = reference ** 3 * 0.05  # Nonlinear distortion
    measured = reference + noise + distortion
    
    print(f"Reference shape: {reference.shape}")
    print(f"Measured shape: {measured.shape}")
    
    # Test EVM
    evm = compute_evm(measured, reference)
    print(f"\nEVM: {evm.mean():.2f} dB")
    
    # Test NMSE
    nmse = compute_nmse(measured, reference)
    print(f"NMSE: {nmse.mean():.2f} dB")
    
    # Test ACPR
    acpr_l, acpr_u = compute_acpr(measured, sample_rate, 100e6, 100e6)
    print(f"ACPR (lower): {acpr_l.mean():.2f} dB")
    print(f"ACPR (upper): {acpr_u.mean():.2f} dB")
    
    # Test combined loss
    spectral_loss = SpectralLoss(sample_rate=sample_rate)
    total_loss, components = spectral_loss(measured, reference, return_components=True)
    
    print(f"\nCombined Spectral Loss:")
    print(f"  Total: {total_loss.item():.4f}")
    for name, value in components.items():
        print(f"  {name}: {value.item():.4f}")
        
    # Test metrics
    metrics = spectral_loss.compute_metrics(measured, reference)
    print(f"\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
