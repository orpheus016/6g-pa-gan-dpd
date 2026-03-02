# ================================================================================
# FIX 1: Update cascaded.py to handle pre-assembled memory taps correctly
# ================================================================================
# filepath: c:\Users\James\github\6g-pa-gan-dpd\models\cascaded.py

"""
Cascaded DPD + PA Model for Forward Learning Architecture (FLA)
FIXED: Proper handling of pre-assembled memory tap inputs
"""

import torch
import torch.nn as nn
from typing import Tuple


class CascadedDPDPA(nn.Module):
    """
    Cascaded DPD + PA model with SEQUENCE support for RNN-based PA models.
    
    FIXED: Handles pre-assembled memory tap inputs [B, seq_len, M+1, 2]
    where the DPD expects [B, M+1, 2] single-step inputs.
    """
    
    def __init__(self, dpd_model: nn.Module, pa_model: nn.Module):
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

        Args:
            x: Input tensor
               - Sequential: [B, seq_len, M+1, 2] (pre-assembled memory taps per timestep)
               - Single step: [B, M+1, 2]

        Returns:
            y_cas: Cascaded output [B, seq_len, 2] or [B, 2]
            u_dpd: DPD output [B, seq_len, 2] or [B, 2]
        """
        if x.dim() == 4:
            return self._forward_sequential(x)
        else:
            return self._forward_single(x)
    
    def _forward_single(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single timestep forward: [B, M+1, 2] → [B, 2]
        """
        # DPD processes pre-assembled memory taps
        u_dpd = self.dpd(x, pre_assembled=True)  # [B, 2]
        
        # Handle potential extra dimension
        if u_dpd.dim() == 3 and u_dpd.shape[1] == 1:
            u_dpd = u_dpd.squeeze(1)
        
        # PA expects [B, seq_len, 2] for RNN, so add seq dim
        u_dpd_for_pa = u_dpd.unsqueeze(1)  # [B, 1, 2]
        y_cas = self.pa(u_dpd_for_pa)  # [B, 1, 2]
        y_cas = y_cas.squeeze(1)  # [B, 2]
        
        return y_cas, u_dpd
    
    def _forward_sequential(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sequential forward: [B, seq_len, M+1, 2] → [B, seq_len, 2]
        
        FIXED: Process each timestep's memory taps through DPD properly.
        """
        B, seq_len, M_plus_1, _ = x.shape
        device = x.device
        
        # Process all timesteps through DPD in parallel
        # Reshape: [B, seq_len, M+1, 2] → [B*seq_len, M+1, 2]
        x_flat = x.view(B * seq_len, M_plus_1, 2)
        
        # DPD processes pre-assembled memory taps (NOT raw IQ sequences)
        # This is the KEY FIX - use pre_assembled=True
        u_dpd_flat = self.dpd(x_flat, pre_assembled=True)  # [B*seq_len, 2]
        
        # Handle potential squeeze
        if u_dpd_flat.dim() == 3 and u_dpd_flat.shape[1] == 1:
            u_dpd_flat = u_dpd_flat.squeeze(1)
        
        # Reshape back to sequences: [B*seq_len, 2] → [B, seq_len, 2]
        u_dpd = u_dpd_flat.view(B, seq_len, 2)
        
        # PA processes full sequences (enables RNN temporal modeling)
        y_cas = self.pa(u_dpd)  # [B, seq_len, 2]
        
        return y_cas, u_dpd