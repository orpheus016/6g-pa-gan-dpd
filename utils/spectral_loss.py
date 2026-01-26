# =============================================================================
# 6G PA GAN-DPD: Spectral Loss Functions (EVM, ACPR, NMSE)
# =============================================================================
"""
SPECTRAL LOSS FUNCTIONS FOR DPD TRAINING
========================================

This module implements spectral-domain loss functions critical for DPD:
1. EVM (Error Vector Magnitude) - measures constellation distortion (frequency-domain, per-subchannel)
2. ACLR (Adjacent Channel Leakage Ratio) - measures spectral regrowth (Welch PSD)
3. NMSE (Normalized Mean Square Error) - measures overall distortion (time-domain)

Based on OpenDPDv2 metrics implementation (Yizhuo Wu, Chang Gao, TU Delft).
These losses ensure the GAN generator produces outputs that not only
fool the discriminator but also meet RF performance requirements.
"""

import torch
import torch.nn as nn
import torch.fft as fft
from typing import Dict, Tuple, Optional
import numpy as np
from scipy.signal import welch


def compute_evm(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    sample_rate: float = 983.04e6,
    bw_main_ch: float = 200e6,
    n_sub_ch: int = 1,
    nperseg: int = 19662,
    return_db: bool = True
) -> float:
    """
    Compute Error Vector Magnitude (EVM) - OpenDPDv2 style (frequency-domain, per-subchannel).
    
    Based on OpenDPDv2 metrics.EVM() implementation.
    
    Args:
        predicted: Predicted/measured IQ signal [batch, seq, 2] or [seq, 2]
        ground_truth: Ground truth/reference IQ signal [batch, seq, 2] or [seq, 2]
        sample_rate: Sample rate in Hz (default: 983.04 MHz for 5G)
        bw_main_ch: Main channel bandwidth in Hz (default: 200 MHz)
        n_sub_ch: Number of sub-channels for analysis (default: 1)
        nperseg: FFT segment length (default: 19662 for 983.04 MHz at 20ms)
        return_db: Return EVM in dB (default) or linear
        
    Returns:
        EVM value in dB (scalar)
    """
    # Convert torch to numpy if needed
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.cpu().detach().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().detach().numpy()
        
    # Handle batch dimension
    if predicted.ndim == 3:
        # Take only first sample in batch for now (can be averaged over batch if needed)
        predicted = predicted[0]
        ground_truth = ground_truth[0]
    
    # Convert to complex
    predicted_complex = predicted[..., 0] + 1j * predicted[..., 1]
    ground_truth_complex = ground_truth[..., 0] + 1j * ground_truth[..., 1]
    
    # Compute FFT with fftshift
    spectrum_pred = np.fft.fft(predicted_complex, n=nperseg, axis=-1)
    spectrum_pred = np.fft.fftshift(spectrum_pred)
    
    spectrum_gt = np.fft.fft(ground_truth_complex, n=nperseg, axis=-1)
    spectrum_gt = np.fft.fftshift(spectrum_gt)
    
    # Create frequency array
    freq = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1/sample_rate))
    
    # Find main channel indices
    index_left = np.min(np.where(freq >= -bw_main_ch / 2))
    index_right = np.max(np.where(freq <= bw_main_ch / 2))
    
    # Compute sub-channel index length
    channel_index_len = int((index_right - index_left) / n_sub_ch)
    
    # Calculate error per sub-channel
    error = np.zeros(n_sub_ch)
    for c in range(n_sub_ch):
        start_idx = index_left + c * channel_index_len
        end_idx = index_left + (c + 1) * channel_index_len
        
        # Error magnitude per subchannel
        error[c] = np.mean(np.abs(spectrum_pred[start_idx:end_idx] - spectrum_gt[start_idx:end_idx]))
        
        # Normalize by ground truth spectrum magnitude
        error[c] = error[c] / np.mean(np.abs(spectrum_gt[start_idx:end_idx]))
    
    # Average error across sub-channels
    evm_avg = error.mean()
    
    # Convert to dB
    evm_db = 20 * np.log10(evm_avg + 1e-10)
    
    if return_db:
        return evm_db
    return evm_avg


def compute_aclr(
    predicted: np.ndarray,
    sample_rate: float = 983.04e6,
    nperseg: int = 19662,
    bw_main_ch: float = 200e6,
    n_sub_ch: int = 1,
    return_db: bool = True
) -> Tuple[float, float]:
    """
    Compute Adjacent Channel Leakage Ratio (ACLR) - OpenDPDv2 style (Welch PSD).
    
    Based on OpenDPDv2 metrics.ACLR() implementation.
    
    Args:
        predicted: Predicted IQ signal [batch, seq, 2] or [seq, 2]
        sample_rate: Sample rate in Hz
        nperseg: Welch segment length
        bw_main_ch: Main channel bandwidth in Hz
        n_sub_ch: Number of sub-channels
        return_db: Return ACLR in dB
        
    Returns:
        aclr_left, aclr_right in dB
    """
    # Convert torch to numpy if needed
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.cpu().detach().numpy()
        
    # Handle batch dimension
    if predicted.ndim == 3:
        predicted = predicted[0]
    
    # Convert to complex
    complex_signal = predicted[..., 0] + 1j * predicted[..., 1]
    
    # Compute Welch PSD (smoother than FFT)
    freq, psd = welch(complex_signal, fs=sample_rate, nperseg=nperseg,
                      return_onesided=False, scaling='spectrum')
    
    # Shift to center (make frequency axis monotonic)
    half_nfft = int(nperseg / 2)
    freq = np.concatenate((freq[half_nfft:], freq[:half_nfft]))
    psd = np.concatenate((psd[half_nfft:], psd[:half_nfft]))
    
    # Find main channel indices
    index_left = np.min(np.where(freq >= -bw_main_ch / 2))
    index_right = np.max(np.where(freq <= bw_main_ch / 2))
    
    # Sub-channel index length
    sub_ch_index_len = int((index_right - index_left) / n_sub_ch)
    
    # Compute power per sub-channel
    sub_ch_power = np.zeros(n_sub_ch)
    for c in range(n_sub_ch):
        sub_ch_power[c] = np.sum(psd[index_left + c * sub_ch_index_len:index_left + (c + 1) * sub_ch_index_len])
    max_sub_ch_power = sub_ch_power.max()
    
    # Compute ACLR for left and right adjacent channels
    left_side_ch_power = np.sum(psd[index_left - sub_ch_index_len:index_left])
    aclr_left = 10 * np.log10(left_side_ch_power / max_sub_ch_power + 1e-10)
    
    right_side_ch_power = np.sum(psd[index_right:index_right + sub_ch_index_len])
    aclr_right = 10 * np.log10(right_side_ch_power / max_sub_ch_power + 1e-10)
    
    return aclr_left, aclr_right


def compute_acpr(
    signal: torch.Tensor,
    sample_rate: float = 200e6,
    channel_bw: float = 100e6,
    adjacent_offset: float = 100e6,
    return_db: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Adjacent Channel Power Ratio (ACPR) - PyTorch differentiable version.
    
    Used during training for gradient computation.
    For evaluation, use compute_aclr() which uses Welch PSD (matches OpenDPDv2).
    
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
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    return_db: bool = True
) -> float:
    """
    Compute Normalized Mean Square Error (NMSE) - OpenDPDv2 style.
    
    Based on OpenDPDv2 metrics.NMSE() implementation.
    
    NMSE = 10 * log10(MSE / energy) where:
    - MSE = mean((I_true - I_hat)^2 + (Q_true - Q_hat)^2)
    - energy = mean(I_true^2 + Q_true^2)
    
    Args:
        predicted: Predicted IQ signal [batch, seq, 2] or [seq, 2]
        ground_truth: Ground truth IQ signal [batch, seq, 2] or [seq, 2]
        return_db: Return in dB (default) or linear
        
    Returns:
        NMSE value in dB (scalar)
    """
    # Convert torch to numpy if needed
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.cpu().detach().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().detach().numpy()
        
    # Handle batch dimension - use first sample
    if predicted.ndim == 3:
        predicted = predicted[0]
        ground_truth = ground_truth[0]
    
    # Extract I and Q
    I_hat = predicted[..., 0]
    Q_hat = predicted[..., 1]
    I_true = ground_truth[..., 0]
    Q_true = ground_truth[..., 1]
    
    # Calculate MSE
    mse = np.mean((I_true - I_hat) ** 2 + (Q_true - Q_hat) ** 2)
    
    # Calculate energy
    energy = np.mean(I_true ** 2 + Q_true ** 2)
    
    # Avoid division by zero
    energy = np.maximum(energy, 1e-10)
    
    # Calculate NMSE
    nmse = mse / energy
    
    if return_db:
        return 10 * np.log10(nmse + 1e-10)
    return nmse


def compute_nmse_differentiable(
    predicted: torch.Tensor,
    target: torch.Tensor,
    return_db: bool = True
) -> torch.Tensor:
    """
    Compute Normalized Mean Square Error (NMSE) - DIFFERENTIABLE PyTorch version.
    
    Works on any batch/sequence shape [batch, 2] or [batch, seq, 2].
    Fully differentiable for gradient-based optimization.
    
    NMSE = MSE / energy where:
    - MSE = mean((I_true - I_hat)^2 + (Q_true - Q_hat)^2)
    - energy = mean(I_true^2 + Q_true^2)
    
    Args:
        predicted: Predicted IQ signal [batch, seq, 2] or [batch, 2] (PyTorch tensor)
        target: Target IQ signal [batch, seq, 2] or [batch, 2] (PyTorch tensor)
        return_db: Return in dB (default) or linear
        
    Returns:
        NMSE value (differentiable tensor, scalar or per-batch)
    """
    # Ensure tensors are on the same device
    assert predicted.device == target.device, "predicted and target must be on same device"
    
    # Extract I and Q components
    if predicted.dim() == 3:
        # Shape: [batch, seq, 2]
        pred_i = predicted[..., 0]  # [batch, seq]
        pred_q = predicted[..., 1]  # [batch, seq]
        target_i = target[..., 0]   # [batch, seq]
        target_q = target[..., 1]   # [batch, seq]
    elif predicted.dim() == 2:
        # Shape: [batch, 2]
        pred_i = predicted[:, 0]    # [batch]
        pred_q = predicted[:, 1]    # [batch]
        target_i = target[:, 0]     # [batch]
        target_q = target[:, 1]     # [batch]
    else:
        raise ValueError(f"Unexpected tensor shape for predicted: {predicted.shape}")
    
    # Calculate MSE (mean squared error)
    # MSE = mean((I_true - I_hat)^2 + (Q_true - Q_hat)^2) over all samples
    mse = torch.mean((target_i - pred_i) ** 2 + (target_q - pred_q) ** 2)
    
    # Calculate energy (signal power)
    # energy = mean(I_true^2 + Q_true^2) over all samples
    energy = torch.mean(target_i ** 2 + target_q ** 2)
    
    # Avoid division by zero
    energy = torch.clamp(energy, min=1e-10)
    
    # Calculate NMSE (normalized MSE)
    nmse = mse / energy
    
    if return_db:
        # Convert to dB: 10 * log10(NMSE)
        nmse_db = 10.0 * torch.log10(nmse + 1e-10)
        return nmse_db
    
    return nmse


def get_amplitude(IQ_signal: np.ndarray) -> np.ndarray:
    """
    Get amplitude (magnitude) from IQ signal.
    
    Based on OpenDPDv2 util.get_amplitude().
    
    Args:
        IQ_signal: IQ signal [seq, 2] or [batch, seq, 2]
        
    Returns:
        Amplitude array
    """
    I = IQ_signal[..., 0]
    Q = IQ_signal[..., 1]
    power = I ** 2 + Q ** 2
    amplitude = np.sqrt(power)
    return amplitude


def set_target_gain(input_iq: np.ndarray, output_iq: np.ndarray) -> float:
    """
    Calculate the target gain (PA gain) from input and output signals.
    
    Based on OpenDPDv2 util.set_target_gain().
    
    Args:
        input_iq: Input IQ signal [seq, 2]
        output_iq: Output IQ signal [seq, 2]
        
    Returns:
        Target gain (scalar)
    """
    amp_in = get_amplitude(input_iq)
    amp_out = get_amplitude(output_iq)
    max_in_amp = np.max(amp_in)
    max_out_amp = np.max(amp_out)
    target_gain = np.mean(max_out_amp / max_in_amp)
    return target_gain


class SpectralLoss(nn.Module):
    """
    Combined spectral loss for DPD training.
    
    Uses differentiable PyTorch functions during training (compute_acpr for gradients).
    Uses numpy-based OpenDPDv2-compatible functions for evaluation metrics.
    
    Combines:
    - L1 reconstruction loss
    - ACPR loss (differentiable, for training)
    - Spectral loss components
    
    Args:
        sample_rate: Signal sample rate in Hz
        channel_bw: Channel bandwidth in Hz
        adjacent_offset: Offset to adjacent channel in Hz
        bw_main_ch: Main channel bandwidth for ACLR (Hz)
        n_sub_ch: Number of sub-channels for EVM/ACLR
        nperseg: FFT segment length for Welch PSD
        acpr_weight: Weight for ACPR loss
        l1_weight: Weight for L1 reconstruction loss
    """
    # MODIFY SpectralLoss.__init__() - ADD nmse_weight parameter

    def __init__(
        self,
        sample_rate: float = 250e6,
        channel_bw: float = 100e6,
        adjacent_offset: float = 100e6,
        bw_main_ch: float = 200e6,
        n_sub_ch: int = 1,
        nperseg: int = 2560,
        l1_weight: float = 1.0,
        power_weight: float = 2.0,      # Reduced from acpr_weight
        nmse_weight: float = 5.0        # NEW: NMSE loss weight, later add weighted loss for A^3
    ):
        super().__init__()
        
        # Training parameters
        self.sample_rate = sample_rate
        self.channel_bw = channel_bw
        self.adjacent_offset = adjacent_offset
        
        # Evaluation parameters (OpenDPDv2 compatible)
        self.bw_main_ch = bw_main_ch
        self.n_sub_ch = n_sub_ch
        self.nperseg = nperseg
        
        # Loss weights
        self.l1_weight = l1_weight              # L1 reconstruction: 50.0
        self.power_weight = power_weight        # Power regularization: 10.0 (reduced)
        self.nmse_weight = nmse_weight          # NMSE loss: 10.0 (NEW)
        
        self.l1_loss = nn.L1Loss()
        
    # MODIFY SpectralLoss.forward() - ADD NMSE loss computation

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute combined spectral loss for training with differentiable NMSE.
        
        Loss = L1_weight * L1 + power_weight * power + nmse_weight * NMSE_dB
        """
        losses = {}
        
        # ===== SHAPE FIX: Squeeze if needed =====
        # Generator might output [B, 1, 2] for M=3, need [B, 2]
        if predicted.dim() == 3 and predicted.shape[1] == 1:
            predicted = predicted.squeeze(1)  # [B, 1, 2] → [B, 2]
        
        # ===== L1 Reconstruction Loss =====
        l1 = self.l1_loss(predicted, target)
        losses['l1'] = l1
        
        # ===== Power Regularization Loss =====
        if predicted.dim() == 3:
            pred_power = (predicted ** 2).mean(dim=[1, 2])
            target_power = (target ** 2).mean(dim=[1, 2])
        elif predicted.dim() == 2:
            pred_power = (predicted ** 2).mean(dim=1)
            target_power = (target ** 2).mean(dim=1)
        else:
            raise ValueError(f"Unexpected tensor shape: {predicted.shape}")
        
        power_loss = torch.nn.functional.mse_loss(pred_power, target_power)
        losses['power'] = power_loss
        
        # ===== Differentiable NMSE Loss =====
        nmse_loss = compute_nmse_differentiable(predicted, target, return_db=True)
        losses['nmse'] = nmse_loss
        
        # ===== Combined Loss =====
        total = (
            self.l1_weight * l1 + 
            self.power_weight * power_loss + 
            self.nmse_weight * nmse_loss
        )
        
        if return_components:
            return total, losses
        
        return total
    
    def compute_metrics(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute all spectral metrics for evaluation (OpenDPDv2 compatible).
        
        Uses numpy-based implementations that match OpenDPDv2 exactly.
        
        Returns:
            Dictionary of metrics
        """
        with torch.no_grad():
            metrics = {}
            
            # Convert to numpy
            pred_np = predicted.cpu().detach().numpy()
            target_np = target.cpu().detach().numpy()
            
            # EVM (frequency-domain, per-subchannel - OpenDPDv2 style)
            try:
                evm = compute_evm(
                    pred_np, target_np,
                    sample_rate=self.sample_rate,
                    bw_main_ch=self.bw_main_ch,
                    n_sub_ch=self.n_sub_ch,
                    nperseg=self.nperseg,
                    return_db=True
                )
                metrics['evm_db'] = evm
            except Exception as e:
                print(f"Warning: EVM computation failed: {e}")
                metrics['evm_db'] = -50.0
            
            # NMSE (time-domain - OpenDPDv2 style)
            try:
                nmse = compute_nmse(pred_np, target_np, return_db=True)
                metrics['nmse_db'] = nmse
            except Exception as e:
                print(f"Warning: NMSE computation failed: {e}")
                metrics['nmse_db'] = -50.0
            
            # ACLR (Welch PSD - OpenDPDv2 style)
            try:
                aclr_l, aclr_r = compute_aclr(
                    pred_np,
                    sample_rate=self.sample_rate,
                    nperseg=self.nperseg,
                    bw_main_ch=self.bw_main_ch,
                    n_sub_ch=self.n_sub_ch,
                    return_db=True
                )
                metrics['aclr_lower_db'] = aclr_l
                metrics['aclr_upper_db'] = aclr_r
                metrics['aclr_max_db'] = max(aclr_l, aclr_r)
            except Exception as e:
                print(f"Warning: ACLR computation failed: {e}")
                metrics['aclr_lower_db'] = -50.0
                metrics['aclr_upper_db'] = -50.0
                metrics['aclr_max_db'] = -50.0
            
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


# Used for config.yaml but we wont be using config.yaml because config is defined in jupyter notebook explicitly
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
