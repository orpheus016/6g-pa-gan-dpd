# =============================================================================
# 6G PA GAN-DPD: Memory-Aware TDNN Generator with QAT
# =============================================================================
"""
MEMORY-AWARE TDNN GENERATOR FOR DPD
===================================

This module implements a Time-Delay Neural Network (TDNN) generator designed
for Digital Predistortion (DPD) of GaN Power Amplifiers.

Key Features:
- Memory-effect aware input structure (envelope + IQ memory taps)
- Quantization-Aware Training (QAT) for minimal EVM degradation
- Bounded complexity for FPGA deployment

Architecture:
    Input [30] → FC1 [32] → LeakyReLU → FC2 [16] → LeakyReLU → FC3 [2] → Tanh
    
Input Structure (M=5 memory depth):
    [I(n), Q(n), |x(n)|, |x(n)|², |x(n)|⁴, |x(n-1)|, |x(n-1)|², |x(n-1)|⁴, ...,
     |x(n-M)|, |x(n-M)|², |x(n-M)|⁴, I(n-1), Q(n-1), ..., I(n-M), Q(n-M)]
    = 2 + 3*(M+1) + 2*M = 30 dimensions

Quantization (FPGA):
    Weights: Q1.15 (16-bit signed, range [-1, +0.99997])
    Activations: Q8.8 (16-bit signed, range [-128, +127.996])
    Accumulator: Q16.16 (32-bit for MAC operations)

Total Parameters: 1,554
    FC1: 30×32 + 32 = 992
    FC2: 32×16 + 16 = 528
    FC3: 16×2 + 2 = 34
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import math


# =============================================================================
# Quantization Utilities (Straight-Through Estimator)
# =============================================================================

class StraightThroughQuantize(torch.autograd.Function):
    """
    Straight-through estimator for quantization.
    Forward: quantize → dequantize
    Backward: pass gradient unchanged
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor, 
                zero_point: torch.Tensor, n_bits: int) -> torch.Tensor:
        q_min = -(2 ** (n_bits - 1))
        q_max = 2 ** (n_bits - 1) - 1
        
        # Quantize
        x_scaled = x / scale + zero_point
        x_quant = torch.round(x_scaled)
        x_quant = torch.clamp(x_quant, q_min, q_max)
        
        # Dequantize
        x_dequant = (x_quant - zero_point) * scale
        
        return x_dequant
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        # Straight-through: pass gradient unchanged
        return grad_output, None, None, None


def fake_quantize(x: torch.Tensor, n_bits: int = 16, 
                  per_channel: bool = False, channel_dim: int = 0) -> torch.Tensor:
    """
    Fake quantization for QAT.
    
    Args:
        x: Input tensor
        n_bits: Number of bits (default 16 for Q1.15 or Q8.8)
        per_channel: Use per-channel quantization
        channel_dim: Channel dimension for per-channel
        
    Returns:
        Fake-quantized tensor (same dtype as input)
    """
    q_min = -(2 ** (n_bits - 1))
    q_max = 2 ** (n_bits - 1) - 1
    
    if per_channel:
        # Per-channel scale
        dims = list(range(x.dim()))
        dims.remove(channel_dim)
        abs_max = x.abs().amax(dim=dims, keepdim=True)
    else:
        abs_max = x.abs().max()
    
    abs_max = torch.clamp(abs_max, min=1e-8)
    scale = abs_max / q_max
    zero_point = torch.zeros_like(scale)
    
    return StraightThroughQuantize.apply(x, scale, zero_point, n_bits)


class FakeQuantizeModule(nn.Module):
    """
    Fake quantization module for QAT.
    Tracks running statistics for activation quantization.
    """
    def __init__(self, n_bits: int = 16, per_channel: bool = False, 
                 channel_dim: int = 0, momentum: float = 0.1):
        super().__init__()
        self.n_bits = n_bits
        self.per_channel = per_channel
        self.channel_dim = channel_dim
        self.momentum = momentum
        
        self.register_buffer('running_scale', torch.tensor(1.0))
        self.register_buffer('running_zero_point', torch.tensor(0.0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Update running statistics
            q_max = 2 ** (self.n_bits - 1) - 1
            abs_max = x.abs().max()
            abs_max = torch.clamp(abs_max, min=1e-8)
            scale = abs_max / q_max
            
            self.running_scale = (1 - self.momentum) * self.running_scale + self.momentum * scale
            
        return fake_quantize(x, self.n_bits, self.per_channel, self.channel_dim)


# =============================================================================
# Memory-Aware Input Assembly
# =============================================================================

class MemoryTapAssembly(nn.Module):
    """
    Assembles memory-aware input vector from IQ samples with nonlinear features.
    
    Input: IQ samples over time
    Output: [I(n), Q(n), |x(n)|, |x(n)|², |x(n)|⁴, ..., |x(n-M)|, |x(n-M)|², |x(n-M)|⁴,
             I(n-1), Q(n-1), ..., I(n-M), Q(n-M)]
    
    Features per tap: magnitude, magnitude², magnitude⁴
    Total dims: 2 (current IQ) + 3*(M+1) (nonlinear envelope) + 2*M (IQ memory) = 30 for M=5
    
    This module is for training only - FPGA uses shift registers.
    """
    def __init__(self, memory_depth: int = 5):
        super().__init__()
        self.memory_depth = memory_depth
        # 2 (current IQ) + 3*(M+1) (|x|, |x|², |x|⁴ for each tap) + 2*M (IQ memory)
        self.input_dim = 2 + 3 * (memory_depth + 1) + 2 * memory_depth  # 30 for M=5
        
    def forward(self, iq_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            iq_sequence: [batch, seq_len, 2] - sequence of IQ samples
            
        Returns:
            memory_input: [batch, seq_len - M, input_dim] - memory-aware input vectors
        """
        batch_size, seq_len, _ = iq_sequence.shape
        M = self.memory_depth
        
        if seq_len <= M:
            raise ValueError(f"Sequence length {seq_len} must be > memory depth {M}")
        
        # Compute envelope
        envelope = torch.sqrt(iq_sequence[..., 0]**2 + iq_sequence[..., 1]**2)  # [batch, seq_len]
        
        outputs = []
        for n in range(M, seq_len):
            # Current IQ: [I(n), Q(n)]
            current_iq = iq_sequence[:, n, :]  # [batch, 2]
            
            # Envelope memory: [|x(n)|, |x(n-1)|, ..., |x(n-M)|]
            env_memory = envelope[:, n-M:n+1].flip(dims=[1])  # [batch, M+1]
            
            # IQ memory: [I(n-1), Q(n-1), ..., I(n-M), Q(n-M)]
            iq_memory = iq_sequence[:, n-M:n, :].flip(dims=[1]).reshape(batch_size, -1)  # [batch, 2*M]
            
            # Concatenate all
            memory_vec = torch.cat([current_iq, env_memory, iq_memory], dim=1)  # [batch, 18]
            outputs.append(memory_vec)
            
        return torch.stack(outputs, dim=1)  # [batch, seq_len - M, 18]


# =============================================================================
# TDNN Generator (Float32)
# =============================================================================

class TDNNGenerator(nn.Module):
    """
    Memory-Aware TDNN Generator for DPD with nonlinear feature extraction.
    
    Architecture: FC1(30→32) → LeakyReLU → FC2(32→16) → LeakyReLU → FC3(16→2) → Tanh
    
    Features: [I(n), Q(n), |x(n)|, |x(n)|², |x(n)|⁴, ..., |x(n-M)|, |x(n-M)|², |x(n-M)|⁴,
               I(n-1), Q(n-1), ..., I(n-M), Q(n-M)]
    
    Args:
        memory_depth: Number of memory taps (M)
        hidden_dims: Hidden layer dimensions [32, 16]
        leaky_slope: LeakyReLU negative slope
        
    Total Parameters: 1,554 (increased from 1,170 due to larger input)
    """
    def __init__(
        self,
        memory_depth: int = 5,
        hidden_dims: List[int] = [32, 16],
        leaky_slope: float = 0.2
    ):
        super().__init__()
        
        self.memory_depth = memory_depth
        self.input_dim = 2 + 3 * (memory_depth + 1) + 2 * memory_depth  # 30 for M=5
        self.hidden_dims = hidden_dims
        self.leaky_slope = leaky_slope
        
        # Memory tap assembly (training only)
        self.memory_assembly = MemoryTapAssembly(memory_depth)
        
        # FC1: 30 → 32
        self.fc1 = nn.Linear(self.input_dim, hidden_dims[0])
        self.act1 = nn.LeakyReLU(negative_slope=leaky_slope)
        
        # FC2: 32 → 16
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.act2 = nn.LeakyReLU(negative_slope=leaky_slope)
        
        # FC3: 16 → 2
        self.fc3 = nn.Linear(hidden_dims[1], 2)
        self.output_act = nn.Tanh()
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization for weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor, pre_assembled: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: If pre_assembled=False: [batch, seq_len, 2] IQ sequence
               If pre_assembled=True: [batch, input_dim] memory-assembled vector
            pre_assembled: Whether input is already memory-assembled
            
        Returns:
            dpd_output: [batch, *, 2] predistorted IQ
        """
        if not pre_assembled:
            # Assemble memory-aware input
            x = self.memory_assembly(x)  # [batch, seq_len - M, 18]
            batch_size, seq_len, _ = x.shape
            x = x.reshape(-1, self.input_dim)  # [batch * seq_len, 18]
            reshape_back = True
        else:
            reshape_back = False
            batch_size, seq_len = x.shape[0], 1
            
        # FC layers
        h = self.act1(self.fc1(x))  # [*, 32]
        h = self.act2(self.fc2(h))  # [*, 16]
        out = self.output_act(self.fc3(h))  # [*, 2]
        
        if reshape_back:
            out = out.reshape(batch_size, seq_len, 2)
            
        return out
    
    def get_param_count(self) -> Dict[str, int]:
        """Return parameter count per layer."""
        return {
            'fc1_weights': self.fc1.weight.numel(),
            'fc1_bias': self.fc1.bias.numel(),
            'fc2_weights': self.fc2.weight.numel(),
            'fc2_bias': self.fc2.bias.numel(),
            'fc3_weights': self.fc3.weight.numel(),
            'fc3_bias': self.fc3.bias.numel(),
            'total': sum(p.numel() for p in self.parameters())
        }


# =============================================================================
# TDNN Generator with QAT
# =============================================================================

class TDNNGeneratorQAT(nn.Module):
    """
    Memory-Aware TDNN Generator with Quantization-Aware Training.
    
    Simulates fixed-point arithmetic during training to minimize
    EVM degradation when deployed on FPGA.
    
    Includes nonlinear features: |x|, |x|², |x|⁴ for better PA modeling.
    
    Quantization:
        Weights: Q1.15 (16-bit signed)
        Activations: Q8.8 (16-bit signed)
        Accumulator: Q16.16 (32-bit, implicit in MAC)
        
    Args:
        memory_depth: Number of memory taps (M)
        hidden_dims: Hidden layer dimensions [32, 16]
        leaky_slope: LeakyReLU negative slope
        weight_bits: Weight quantization bits (default 16)
        activation_bits: Activation quantization bits (default 16)
    """
    def __init__(
        self,
        memory_depth: int = 5,
        hidden_dims: List[int] = [32, 16],
        leaky_slope: float = 0.2,
        weight_bits: int = 16,
        activation_bits: int = 16
    ):
        super().__init__()
        
        self.memory_depth = memory_depth
        self.input_dim = 2 + 3 * (memory_depth + 1) + 2 * memory_depth
        self.hidden_dims = hidden_dims
        self.leaky_slope = leaky_slope
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        
        # Memory tap assembly
        self.memory_assembly = MemoryTapAssembly(memory_depth)
        
        # Quantizers for inputs
        self.input_quant = FakeQuantizeModule(n_bits=activation_bits)
        
        # FC1: 30 → 32 (with quantization)
        self.fc1 = nn.Linear(self.input_dim, hidden_dims[0])
        self.fc1_weight_quant = FakeQuantizeModule(n_bits=weight_bits, per_channel=True)
        self.fc1_act_quant = FakeQuantizeModule(n_bits=activation_bits)
        self.act1 = nn.LeakyReLU(negative_slope=leaky_slope)
        
        # FC2: 32 → 16 (with quantization)
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc2_weight_quant = FakeQuantizeModule(n_bits=weight_bits, per_channel=True)
        self.fc2_act_quant = FakeQuantizeModule(n_bits=activation_bits)
        self.act2 = nn.LeakyReLU(negative_slope=leaky_slope)
        
        # FC3: 16 → 2 (with quantization)
        self.fc3 = nn.Linear(hidden_dims[1], 2)
        self.fc3_weight_quant = FakeQuantizeModule(n_bits=weight_bits, per_channel=True)
        self.fc3_act_quant = FakeQuantizeModule(n_bits=activation_bits)
        self.output_act = nn.Tanh()
        
        # Output quantization (Q1.15 for DAC)
        self.output_quant = FakeQuantizeModule(n_bits=16)
        
        # Track QAT state
        self.qat_enabled = True
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with bounded range for quantization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use smaller initialization for quantization-friendly range
                bound = 0.5 / math.sqrt(m.in_features)
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def enable_qat(self):
        """Enable quantization-aware training."""
        self.qat_enabled = True
        
    def disable_qat(self):
        """Disable QAT (for float evaluation)."""
        self.qat_enabled = False
        
    def _quantized_linear(self, x: torch.Tensor, layer: nn.Linear,
                          weight_quant: FakeQuantizeModule,
                          act_quant: FakeQuantizeModule,
                          activation: nn.Module) -> torch.Tensor:
        """Apply quantized linear layer."""
        if self.qat_enabled:
            # Quantize weights
            q_weight = weight_quant(layer.weight)
            # Linear operation
            out = F.linear(x, q_weight, layer.bias)
            # Activation
            out = activation(out)
            # Quantize activation output
            out = act_quant(out)
        else:
            out = activation(layer(x))
        return out
    
    def forward(self, x: torch.Tensor, pre_assembled: bool = False) -> torch.Tensor:
        """
        Forward pass with QAT.
        
        Args:
            x: Input tensor (IQ sequence or pre-assembled)
            pre_assembled: Whether input is already memory-assembled
            
        Returns:
            dpd_output: Predistorted IQ samples
        """
        if not pre_assembled:
            x = self.memory_assembly(x)
            batch_size, seq_len, _ = x.shape
            x = x.reshape(-1, self.input_dim)
            reshape_back = True
        else:
            reshape_back = False
            batch_size, seq_len = x.shape[0], 1
            
        # Quantize input
        if self.qat_enabled:
            x = self.input_quant(x)
            
        # FC1
        h = self._quantized_linear(x, self.fc1, self.fc1_weight_quant,
                                   self.fc1_act_quant, self.act1)
        
        # FC2
        h = self._quantized_linear(h, self.fc2, self.fc2_weight_quant,
                                   self.fc2_act_quant, self.act2)
        
        # FC3 + Tanh
        if self.qat_enabled:
            q_weight = self.fc3_weight_quant(self.fc3.weight)
            out = F.linear(h, q_weight, self.fc3.bias)
            out = self.output_act(out)
            out = self.output_quant(out)
        else:
            out = self.output_act(self.fc3(h))
            
        if reshape_back:
            out = out.reshape(batch_size, seq_len, 2)
            
        return out
    
    def get_quantized_weights(self) -> Dict[str, torch.Tensor]:
        """
        Export quantized weights for FPGA.
        
        Returns:
            Dictionary of quantized weight tensors in integer format
        """
        weights = {}
        
        # FC1 weights: Q1.15
        w1 = self.fc1.weight.data
        scale1 = w1.abs().max() / (2**15 - 1)
        weights['fc1_weight'] = torch.round(w1 / scale1).to(torch.int16)
        weights['fc1_weight_scale'] = scale1
        weights['fc1_bias'] = self.fc1.bias.data
        
        # FC2 weights: Q1.15
        w2 = self.fc2.weight.data
        scale2 = w2.abs().max() / (2**15 - 1)
        weights['fc2_weight'] = torch.round(w2 / scale2).to(torch.int16)
        weights['fc2_weight_scale'] = scale2
        weights['fc2_bias'] = self.fc2.bias.data
        
        # FC3 weights: Q1.15
        w3 = self.fc3.weight.data
        scale3 = w3.abs().max() / (2**15 - 1)
        weights['fc3_weight'] = torch.round(w3 / scale3).to(torch.int16)
        weights['fc3_weight_scale'] = scale3
        weights['fc3_bias'] = self.fc3.bias.data
        
        return weights
    
    def get_param_count(self) -> Dict[str, int]:
        """Return parameter count per layer."""
        return {
            'fc1_weights': self.fc1.weight.numel(),
            'fc1_bias': self.fc1.bias.numel(),
            'fc2_weights': self.fc2.weight.numel(),
            'fc2_bias': self.fc2.bias.numel(),
            'fc3_weights': self.fc3.weight.numel(),
            'fc3_bias': self.fc3.bias.numel(),
            'total': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    @classmethod
    def from_float(cls, float_model: TDNNGenerator, 
                   weight_bits: int = 16, 
                   activation_bits: int = 16) -> 'TDNNGeneratorQAT':
        """
        Convert float model to QAT model.
        
        Args:
            float_model: Pre-trained float32 TDNNGenerator
            weight_bits: Weight quantization bits
            activation_bits: Activation quantization bits
            
        Returns:
            QAT model with copied weights
        """
        qat_model = cls(
            memory_depth=float_model.memory_depth,
            hidden_dims=float_model.hidden_dims,
            leaky_slope=float_model.leaky_slope,
            weight_bits=weight_bits,
            activation_bits=activation_bits
        )
        
        # Copy weights
        qat_model.fc1.weight.data.copy_(float_model.fc1.weight.data)
        qat_model.fc1.bias.data.copy_(float_model.fc1.bias.data)
        qat_model.fc2.weight.data.copy_(float_model.fc2.weight.data)
        qat_model.fc2.bias.data.copy_(float_model.fc2.bias.data)
        qat_model.fc3.weight.data.copy_(float_model.fc3.weight.data)
        qat_model.fc3.bias.data.copy_(float_model.fc3.bias.data)
        
        return qat_model


# =============================================================================
# Utility Functions
# =============================================================================

def create_generator(config: dict, qat: bool = False) -> nn.Module:
    """
    Factory function to create generator from config.
    
    Args:
        config: Configuration dictionary
        qat: Whether to create QAT model
        
    Returns:
        Generator model
    """
    gen_config = config['model']['generator']
    quant_config = config.get('quantization', {})
    
    if qat and quant_config.get('enabled', False):
        return TDNNGeneratorQAT(
            memory_depth=gen_config.get('memory_depth', 5),
            hidden_dims=gen_config.get('hidden_dims', [32, 16]),
            leaky_slope=gen_config.get('leaky_slope', 0.2),
            weight_bits=quant_config.get('weight', {}).get('bits', 16),
            activation_bits=quant_config.get('activation', {}).get('bits', 16)
        )
    else:
        return TDNNGenerator(
            memory_depth=gen_config.get('memory_depth', 5),
            hidden_dims=gen_config.get('hidden_dims', [32, 16]),
            leaky_slope=gen_config.get('leaky_slope', 0.2)
        )


if __name__ == "__main__":
    # Test generator
    print("Testing TDNN Generator")
    print("=" * 50)
    
    # Create models
    gen_float = TDNNGenerator(memory_depth=5, hidden_dims=[32, 16])
    gen_qat = TDNNGeneratorQAT(memory_depth=5, hidden_dims=[32, 16])
    
    # Parameter count
    print("\nParameter Count:")
    for name, count in gen_float.get_param_count().items():
        print(f"  {name}: {count}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 32
    iq_input = torch.randn(batch_size, seq_len, 2)
    
    print(f"\nInput shape: {iq_input.shape}")
    
    # Float model
    out_float = gen_float(iq_input)
    print(f"Float output shape: {out_float.shape}")
    
    # QAT model
    out_qat = gen_qat(iq_input)
    print(f"QAT output shape: {out_qat.shape}")
    
    # Compare outputs
    diff = (out_float - out_qat).abs().mean()
    print(f"\nMean absolute difference (float vs QAT): {diff:.6f}")
    
    # Test quantized weights export
    print("\nQuantized Weights:")
    q_weights = gen_qat.get_quantized_weights()
    for name, tensor in q_weights.items():
        if isinstance(tensor, torch.Tensor):
            print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
        else:
            print(f"  {name}: {tensor:.6f}")
