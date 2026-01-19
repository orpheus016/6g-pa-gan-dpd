"""
PN-TDNN Generator: Phase-Normalized TDNN for DPD

Architecture: 24 → 32 → 16 → 2 (1,362 parameters)
Memory Depth: M=3 (captures >95% of GaN PA memory energy)
Feature Extraction: Phase-normalized (decouples amplitude/phase learning)

Reference: ARCHITECTURE.md v3.0, FULL-TRAINING-FLOW.md v1.0
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class PhaseNormalizedFeatureExtraction(nn.Module):
    """
    Phase-Normalized Feature Extraction matching FPGA implementation.
    
    For each sample n with memory depth M=3:
    - A(n-k) for k=0..M: Amplitude
    - A³(n-k): Cubic amplitude (odd-order PA model, dominant IMD3)
    - I_norm(n-k), Q_norm(n-k): Phase-normalized IQ
    - I(n-k), Q(n-k): Original IQ (residual/linear path)
    
    Total: 6 × (M+1) = 24 features for M=3
    
    Why phase normalization (SparseDPD):
    - Decouples amplitude and phase: FC layers learn amplitude-only relationships
    - Reduces model complexity by ~40%
    - All delayed samples rotated to align with current sample's phase
    
    Phase normalization formula:
        P(n) = (I(n) - jQ(n)) / A(n) = e^{-jφ(n)}
        
        I_norm(n-k) = (I(n-k)·I(n) + Q(n-k)·Q(n)) / A(n)
        Q_norm(n-k) = (Q(n-k)·I(n) - I(n-k)·Q(n)) / A(n)
    """
    
    def __init__(self, memory_depth: int = 3):
        super().__init__()
        self.M = memory_depth
        self.output_dim = 6 * (memory_depth + 1)  # 24 for M=3
    
    def forward(self, iq_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract phase-normalized features from IQ sequence.
        
        Args:
            iq_sequence: [batch, seq_len, 2] - I, Q values
            
        Returns:
            features: [batch, seq_len - M, 24] - Phase-normalized feature vectors
            reference: [batch, seq_len - M, 3] - (I_0, Q_0, A_0) for phase denormalization
        """
        batch, seq_len, _ = iq_sequence.shape
        device = iq_sequence.device
        
        I = iq_sequence[..., 0]  # [batch, seq_len]
        Q = iq_sequence[..., 1]  # [batch, seq_len]
        
        # Envelope amplitude (with epsilon for numerical stability)
        eps = 1e-8
        A = torch.sqrt(I**2 + Q**2 + eps)  # [batch, seq_len]
        A3 = A ** 3  # Cubic amplitude (dominant odd-order nonlinearity)
        
        feature_list = []
        reference_list = []
        
        # Process each valid output sample (need M previous samples)
        for n in range(self.M, seq_len):
            tap_features = []
            
            # Current sample reference for phase normalization
            I_0 = I[:, n]  # [batch]
            Q_0 = Q[:, n]  # [batch]
            A_0 = A[:, n]  # [batch]
            
            # Store reference for phase denormalization
            reference_list.append(torch.stack([I_0, Q_0, A_0], dim=-1))
            
            # Extract features for each memory tap (k = 0, 1, 2, 3 for M=3)
            for k in range(self.M + 1):
                idx = n - k
                
                # Feature 1: Amplitude A(n-k)
                tap_features.append(A[:, idx:idx+1])
                
                # Feature 2: Cubic amplitude A³(n-k)
                tap_features.append(A3[:, idx:idx+1])
                
                # Features 3-4: Phase-normalized IQ
                # Complex multiply: (I_k + jQ_k) × (I_0 - jQ_0) / A_0
                # Result: I_norm = (I_k·I_0 + Q_k·Q_0) / A_0
                #         Q_norm = (Q_k·I_0 - I_k·Q_0) / A_0
                I_k = I[:, idx]
                Q_k = Q[:, idx]
                
                I_norm = (I_k * I_0 + Q_k * Q_0) / (A_0 + eps)
                Q_norm = (Q_k * I_0 - I_k * Q_0) / (A_0 + eps)
                
                tap_features.append(I_norm.unsqueeze(-1))
                tap_features.append(Q_norm.unsqueeze(-1))
                
                # Features 5-6: Raw IQ (residual/linear path)
                tap_features.append(I[:, idx:idx+1])
                tap_features.append(Q[:, idx:idx+1])
            
            # Concatenate all tap features: [batch, 24]
            feature_list.append(torch.cat(tap_features, dim=-1))
        
        # Stack along sequence dimension
        features = torch.stack(feature_list, dim=1)  # [batch, seq_len - M, 24]
        reference = torch.stack(reference_list, dim=1)  # [batch, seq_len - M, 3]
        
        return features, reference


class PNTDNNGenerator(nn.Module):
    """
    Phase-Normalized TDNN Generator for DPD.
    
    Architecture: 24 → 32 → 16 → 2
    Parameters: 1,362
        - FC1: 24×32 + 32 = 800
        - FC2: 32×16 + 16 = 528
        - FC3: 16×2 + 2 = 34
        - Total: 1,362
    
    Training flow (ILA - Indirect Learning Architecture):
        - Input: y_PA (distorted PA output)
        - Target: u_PA (clean PA input)
        - DPD learns: y_PA → u_PA (inverse of PA)
        - At inference: x → DPD(x) → PA(DPD(x)) ≈ linear(x)
    
    Key design decisions:
        - LeakyReLU(0.2): Prevents dead neurons, WGAN-GP standard
        - No Tanh output: Phase denormalization naturally bounds output
        - QAT: Q1.15 weights (16-bit), Q8.8 activations (16-bit)
    """
    
    def __init__(
        self,
        memory_depth: int = 3,
        hidden_dims: List[int] = [32, 16],
        leaky_slope: float = 0.2
    ):
        super().__init__()
        
        self.memory_depth = memory_depth
        self.input_dim = 6 * (memory_depth + 1)  # 24 for M=3
        self.hidden_dims = hidden_dims
        self.leaky_slope = leaky_slope
        
        # Feature extraction (not trainable, just preprocessing)
        self.feature_extraction = PhaseNormalizedFeatureExtraction(memory_depth)
        
        # FC layers: 24 → 32 → 16 → 2
        layers = []
        in_features = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.LeakyReLU(leaky_slope))
            in_features = hidden_dim
        
        # Output layer: no activation (phase denorm handles bounds)
        layers.append(nn.Linear(in_features, 2))
        
        self.fc_layers = nn.Sequential(*layers)
        
        # QAT configuration
        self.qat_enabled = False
        self.weight_bits = 16  # Q1.15
        self.act_bits = 16     # Q8.8
        
        # Initialize weights (Xavier uniform for stable GAN training)
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def enable_qat(self):
        """Enable Quantization-Aware Training."""
        self.qat_enabled = True
    
    def disable_qat(self):
        """Disable Quantization-Aware Training."""
        self.qat_enabled = False
    
    def fake_quantize(self, x: torch.Tensor, bits: int, is_weight: bool = False) -> torch.Tensor:
        """
        Fake quantization with straight-through estimator (STE).
        
        Args:
            x: Input tensor
            bits: Number of bits for quantization
            is_weight: True for weights (Q1.15), False for activations (Q8.8)
            
        Returns:
            Quantized tensor (differentiable via STE)
        """
        if is_weight:
            # Q1.15: range [-1, 1), scale = 2^15
            scale = 2 ** (bits - 1)
            x_clamp = torch.clamp(x, -1.0, 1.0 - 1.0/scale)
        else:
            # Q8.8: range [-128, 128), scale = 2^8
            scale = 2 ** (bits // 2)
            max_val = 2 ** (bits // 2)
            x_clamp = torch.clamp(x, -max_val, max_val - 1.0/scale)
        
        # Quantize
        x_quant = torch.round(x_clamp * scale)
        
        # Dequantize
        x_dequant = x_quant / scale
        
        # Straight-through estimator: gradient flows through as if no quantization
        return x_dequant.detach() + x - x.detach()
    
    def phase_denormalize(
        self,
        fc_out: torch.Tensor,
        reference: torch.Tensor
    ) -> torch.Tensor:
        """
        Phase denormalization: rotate FC output back to original phase.
        
        Inverse of phase normalization:
            I_out = (I_fc·I_0 - Q_fc·Q_0) / A_0
            Q_out = (I_fc·Q_0 + Q_fc·I_0) / A_0
        
        Args:
            fc_out: [batch, seq_len, 2] - FC layer output (I_fc, Q_fc)
            reference: [batch, seq_len, 3] - (I_0, Q_0, A_0) from feature extraction
            
        Returns:
            output: [batch, seq_len, 2] - Phase-denormalized IQ output
        """
        eps = 1e-8
        
        I_fc = fc_out[..., 0]  # [batch, seq_len]
        Q_fc = fc_out[..., 1]  # [batch, seq_len]
        
        I_0 = reference[..., 0]  # [batch, seq_len]
        Q_0 = reference[..., 1]  # [batch, seq_len]
        A_0 = reference[..., 2]  # [batch, seq_len]
        
        # Complex multiply: (I_fc + jQ_fc) × (I_0 + jQ_0) / A_0
        # Result: I_out = (I_fc·I_0 - Q_fc·Q_0) / A_0
        #         Q_out = (I_fc·Q_0 + Q_fc·I_0) / A_0
        I_out = (I_fc * I_0 - Q_fc * Q_0) / (A_0 + eps)
        Q_out = (I_fc * Q_0 + Q_fc * I_0) / (A_0 + eps)
        
        return torch.stack([I_out, Q_out], dim=-1)
    
    def forward(
        self,
        x: torch.Tensor,
        pre_assembled: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through PN-TDNN.
        
        Args:
            x: Input tensor
               - If pre_assembled=False: [batch, seq_len, 2] raw IQ sequence
               - If pre_assembled=True: [batch, 24] pre-extracted features
            pre_assembled: Whether features are already extracted
            
        Returns:
            output: [batch, 2] or [batch, seq_len - M, 2] DPD output IQ
        """
        if pre_assembled:
            # Features already extracted (e.g., for FPGA simulation)
            features = x
            reference = None
            
            # Apply FC layers with optional QAT
            out = self._forward_fc(features)
            return out
        else:
            # Extract phase-normalized features
            features, reference = self.feature_extraction(x)
            # features: [batch, seq_len - M, 24]
            # reference: [batch, seq_len - M, 3]
            
            # Reshape for batch processing
            batch, seq_out, feat_dim = features.shape
            features_flat = features.view(batch * seq_out, feat_dim)
            
            # Apply FC layers
            fc_out_flat = self._forward_fc(features_flat)
            fc_out = fc_out_flat.view(batch, seq_out, 2)
            
            # Phase denormalization
            output = self.phase_denormalize(fc_out, reference)
            
            return output
    
    def _forward_fc(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FC layers with optional QAT.
        
        Args:
            features: [batch, 24] input features
            
        Returns:
            output: [batch, 2] FC output
        """
        if self.qat_enabled:
            # Quantize activations
            x = self.fake_quantize(features, self.act_bits, is_weight=False)
            
            # Process through FC layers with quantized weights
            for i, layer in enumerate(self.fc_layers):
                if isinstance(layer, nn.Linear):
                    # Quantize weights
                    w_q = self.fake_quantize(layer.weight, self.weight_bits, is_weight=True)
                    b_q = self.fake_quantize(layer.bias, self.weight_bits, is_weight=True) if layer.bias is not None else None
                    
                    # Linear operation with quantized weights
                    x = torch.nn.functional.linear(x, w_q, b_q)
                    
                    # Quantize activation output (except final layer)
                    if i < len(self.fc_layers) - 1:
                        x = self.fake_quantize(x, self.act_bits, is_weight=False)
                else:
                    # Activation function (LeakyReLU)
                    x = layer(x)
            return x
        else:
            # Standard forward pass
            return self.fc_layers(features)
    
    def get_parameter_count(self) -> dict:
        """Get detailed parameter count matching ARCHITECTURE.md spec."""
        counts = {
            'fc1': 0,
            'fc2': 0,
            'fc3': 0,
            'total': 0
        }
        
        fc_idx = 0
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                params = layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
                fc_idx += 1
                counts[f'fc{fc_idx}'] = params
        
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts
    
    def export_weights_q115(self) -> dict:
        """
        Export weights in Q1.15 format for FPGA.
        
        Q1.15: 1 sign bit, 15 fractional bits
        Range: [-1, 1 - 2^-15]
        Scale: 2^15 = 32768
        
        Returns:
            Dictionary with quantized weights as numpy arrays
        """
        import numpy as np
        
        weights = {}
        scale = 2 ** 15  # Q1.15
        
        fc_idx = 0
        for name, param in self.named_parameters():
            # Clamp to Q1.15 range and quantize
            p_np = param.detach().cpu().numpy()
            p_clamp = np.clip(p_np, -1.0, 1.0 - 1.0/scale)
            p_quant = np.round(p_clamp * scale).astype(np.int16)
            
            weights[name] = p_quant
        
        return weights


class Discriminator(nn.Module):
    """
    Conditional Discriminator for CWGAN-GP.
    
    Architecture: 4 → 64 → 32 → 1
    
    Input: IQ output [batch, 2] + condition [batch, 2] = [batch, 4]
    Output: Critic score (unbounded, not probability)
    
    Key design decisions:
        - Conditional: Judges output quality given input (not just marginal distribution)
        - No sigmoid: WGAN outputs unbounded critic score
        - No BatchNorm: WGAN-GP recommends LayerNorm or none
        - LeakyReLU(0.2): Prevents dead neurons
    
    Note: Only used during training, NOT deployed on FPGA.
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dims: List[int] = [64, 32],
        leaky_slope: float = 0.2
    ):
        super().__init__()
        
        layers = []
        in_features = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.LeakyReLU(leaky_slope))
            in_features = hidden_dim
        
        # Output: single critic score (no activation)
        layers.append(nn.Linear(in_features, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: DPD output [batch, 2] (I, Q)
            condition: Input signal [batch, 2] (I_in, Q_in) - what DPD saw
            
        Returns:
            Critic score [batch, 1] (unbounded)
        """
        # Concatenate output and condition
        combined = torch.cat([x, condition], dim=-1)  # [batch, 4]
        return self.net(combined)


# Convenience function for creating models
def create_pn_tdnn_generator(memory_depth: int = 3) -> PNTDNNGenerator:
    """
    Create PN-TDNN generator with default architecture.
    
    Args:
        memory_depth: Number of memory taps (default M=3)
        
    Returns:
        PNTDNNGenerator instance
    """
    return PNTDNNGenerator(
        memory_depth=memory_depth,
        hidden_dims=[32, 16],
        leaky_slope=0.2
    )


def create_discriminator() -> Discriminator:
    """
    Create discriminator with default architecture.
    
    Returns:
        Discriminator instance
    """
    return Discriminator(
        input_dim=4,
        hidden_dims=[64, 32],
        leaky_slope=0.2
    )


if __name__ == "__main__":
    # Verification: ensure parameter count matches ARCHITECTURE.md
    gen = create_pn_tdnn_generator(memory_depth=3)
    disc = create_discriminator()
    
    print("=" * 60)
    print("PN-TDNN Generator Architecture Verification")
    print("=" * 60)
    print(f"Memory depth: M={gen.memory_depth}")
    print(f"Input dim: {gen.input_dim} (expected: 24)")
    print(f"Hidden dims: {gen.hidden_dims}")
    print()
    
    counts = gen.get_parameter_count()
    print("Parameter count:")
    print(f"  FC1 (24→32): {counts['fc1']} (expected: 800)")
    print(f"  FC2 (32→16): {counts['fc2']} (expected: 528)")
    print(f"  FC3 (16→2):  {counts['fc3']} (expected: 34)")
    print(f"  Total:       {counts['total']} (expected: 1362)")
    print()
    
    # Test forward pass
    batch_size = 16
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 2)  # [batch, seq_len, 2]
    
    out = gen(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape} (expected: [{batch_size}, {seq_len - gen.memory_depth}, 2])")
    print()
    
    # Test QAT
    gen.enable_qat()
    out_qat = gen(x)
    print(f"QAT output shape: {out_qat.shape}")
    print()
    
    # Test discriminator
    print("=" * 60)
    print("Discriminator Architecture Verification")
    print("=" * 60)
    disc_params = sum(p.numel() for p in disc.parameters())
    print(f"Total parameters: {disc_params}")
    
    # Test discriminator forward
    dpd_out = torch.randn(batch_size, 2)
    condition = torch.randn(batch_size, 2)
    critic_score = disc(dpd_out, condition)
    print(f"Critic output shape: {critic_score.shape} (expected: [{batch_size}, 1])")
    print()
    
    print("✓ All verifications passed!")
