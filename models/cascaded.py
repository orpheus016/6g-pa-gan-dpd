"""
Cascaded DPD + PA Model for Forward Learning Architecture (FLA)

This module provides the CascadedDPDPA model that chains a trainable DPD
with a frozen PA surrogate for end-to-end learning.

Reference: OpenDPD E2E learning methodology
"""

import torch
import torch.nn as nn
from typing import Tuple


class CascadedDPDPA(nn.Module):
    """
    Cascaded DPD + PA model with SEQUENCE support for RNN-based PA models.
    
    Architecture:
        Input → DPD (trainable) → PA (frozen) → Output
    
    Key features:
        - PA parameters are frozen (requires_grad=False)
        - PA remains in train() mode for cuDNN RNN backward compatibility
        - Supports both sequential [B, seq_len, M+1, 2] and single-step [B, M+1, 2] inputs
        - Gradients backpropagate through frozen PA to update only DPD weights
    
    Why PA.train() mode:
        - cuDNN RNN requires train mode for efficient backward pass
        - Freezing via requires_grad=False prevents weight updates
        - This combination gives best of both worlds: frozen weights + fast RNNs
    """
    
    def __init__(self, dpd_model: nn.Module, pa_model: nn.Module):
        """
        Initialize cascaded model.
        
        Args:
            dpd_model: Trainable DPD model (e.g., PN-TDNN)
            pa_model: Frozen PA surrogate model (e.g., DGRU, VDLSTM)
        """
        super().__init__()
        self.dpd = dpd_model
        self.pa = pa_model

        # Freeze PA parameters but keep in train mode for RNN efficiency
        self.pa.train()
        for param in self.pa.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through cascaded model with sequence support.

        Handles both:
            - Sequential: [B, seq_len, M+1, 2] → [B, seq_len, 2] (for RNN PA models)
            - Single step: [B, M+1, 2] → [B, 2] (for inference/validation)

        Args:
            x: Input IQ with memory taps
               - Sequential: [B, seq_len, M+1, 2] where M is memory depth
               - Single: [B, M+1, 2]

        Returns:
            y_cas: Cascaded output (PA output on predistorted signal)
                   - Sequential: [B, seq_len, 2]
                   - Single: [B, 2]
            u_dpd: DPD output (predistorted signal)
                   - Sequential: [B, seq_len, 2]
                   - Single: [B, 2]
        """
        if x.dim() == 4:
            # Sequential: [B, seq_len, M+1, 2]
            return self._forward_sequential(x)
        else:
            # Single step: [B, M+1, 2]
            return self._forward_single(x)
    
    def _forward_single(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single timestep forward: [B, M+1, 2] → [B, 2]
        
        Used for:
            - Validation on non-sequential data
            - Inference in production
        """
        u_dpd = self.dpd(x)  # [B, 2] or [B, 1, 2]
        
        # Squeeze sequence dimension if present
        if u_dpd.dim() == 3 and u_dpd.shape[1] == 1:
            u_dpd = u_dpd.squeeze(1)  # [B, 2]
        
        y_cas = self.pa(u_dpd)  # [B, 2]
        
        return y_cas, u_dpd
    
    def _forward_sequential(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sequential forward: [B, seq_len, M+1, 2] → [B, seq_len, 2]
        
        Processing strategy:
            1. Vectorize DPD over all timesteps: [B*seq_len, M+1, 2]
            2. DPD produces [B*seq_len, 2] predistorted samples
            3. Reshape to [B, seq_len, 2] for PA
            4. PA processes full sequences (RNN temporal modeling)
        
        Why this matters:
            - RNN-based PA models (DGRU/VDLSTM) need sequences for temporal context
            - Batch-level [B, 2] degrades RNNs to feedforward
            - Sequence-based processing achieves -30dB NMSE vs -20dB batch-level
        """
        B, seq_len, M_plus_1, _ = x.shape
        
        # Vectorized DPD processing: [B, seq_len, M+1, 2] → [B*seq_len, M+1, 2]
        x_flat = x.view(B * seq_len, M_plus_1, 2)
        u_dpd_flat = self.dpd(x_flat)  # [B*seq_len, 2] or [B*seq_len, 1, 2]
        
        # Squeeze if DPD outputs [B*seq_len, 1, 2]
        if u_dpd_flat.dim() == 3:
            u_dpd_flat = u_dpd_flat.squeeze(1)  # [B*seq_len, 2]
        
        # Reshape to sequence: [B, seq_len, 2]
        u_dpd = u_dpd_flat.view(B, seq_len, 2)
        
        # PA processes full sequence (enables RNN temporal modeling)
        y_cas = self.pa(u_dpd)  # [B, seq_len, 2]
        
        return y_cas, u_dpd
