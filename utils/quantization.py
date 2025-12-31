# =============================================================================
# 6G PA GAN-DPD: Quantization Utilities
# =============================================================================
"""
QUANTIZATION FOR FPGA DEPLOYMENT
================================

This module provides utilities for:
1. Quantization-Aware Training (QAT)
2. Fixed-point conversion for FPGA
3. Weight export in binary format

QUANTIZATION FORMATS:
--------------------
Weights:     Q1.15  (16-bit signed, range [-1, +0.99997])
Activations: Q8.8   (16-bit signed, range [-128, +127.996])
Accumulator: Q16.16 (32-bit signed, range [-32768, +32767.99998])
Input/Output: Q1.15 (normalized IQ samples)

FIXED-POINT MATH:
----------------
For Q1.15:  value = integer / 2^15
For Q8.8:   value = integer / 2^8  
For Q16.16: value = integer / 2^16
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union
from pathlib import Path
import struct


@dataclass
class QuantizationConfig:
    """Configuration for quantization parameters."""
    
    # Weight quantization
    weight_bits: int = 16
    weight_frac_bits: int = 15  # Q1.15
    weight_signed: bool = True
    
    # Activation quantization
    activation_bits: int = 16
    activation_frac_bits: int = 8  # Q8.8
    activation_signed: bool = True
    
    # Accumulator (for MAC operations)
    accumulator_bits: int = 32
    accumulator_frac_bits: int = 16  # Q16.16
    
    # Input/Output (normalized IQ)
    io_bits: int = 16
    io_frac_bits: int = 15  # Q1.15
    
    @property
    def weight_scale(self) -> float:
        """Scale factor for weight quantization."""
        return 2 ** self.weight_frac_bits
    
    @property
    def activation_scale(self) -> float:
        """Scale factor for activation quantization."""
        return 2 ** self.activation_frac_bits
    
    @property
    def io_scale(self) -> float:
        """Scale factor for I/O quantization."""
        return 2 ** self.io_frac_bits
    
    @property
    def weight_range(self) -> Tuple[int, int]:
        """Integer range for weights."""
        if self.weight_signed:
            return (-(2 ** (self.weight_bits - 1)), 2 ** (self.weight_bits - 1) - 1)
        return (0, 2 ** self.weight_bits - 1)
    
    @property
    def activation_range(self) -> Tuple[int, int]:
        """Integer range for activations."""
        if self.activation_signed:
            return (-(2 ** (self.activation_bits - 1)), 2 ** (self.activation_bits - 1) - 1)
        return (0, 2 ** self.activation_bits - 1)


def quantize_to_fixed_point(
    tensor: torch.Tensor,
    scale: float,
    bits: int,
    signed: bool = True
) -> torch.Tensor:
    """
    Quantize floating-point tensor to fixed-point integer.
    
    Args:
        tensor: Input float tensor
        scale: Scale factor (2^frac_bits)
        bits: Total bits
        signed: Whether signed or unsigned
        
    Returns:
        Integer tensor (same dtype for gradients, use .to(torch.int16) for export)
    """
    if signed:
        q_min = -(2 ** (bits - 1))
        q_max = 2 ** (bits - 1) - 1
    else:
        q_min = 0
        q_max = 2 ** bits - 1
        
    # Scale and round
    scaled = tensor * scale
    quantized = torch.round(scaled)
    
    # Clamp to range
    quantized = torch.clamp(quantized, q_min, q_max)
    
    return quantized


def dequantize_from_fixed_point(
    tensor: torch.Tensor,
    scale: float
) -> torch.Tensor:
    """
    Dequantize fixed-point integer to floating-point.
    
    Args:
        tensor: Integer tensor
        scale: Scale factor (2^frac_bits)
        
    Returns:
        Float tensor
    """
    return tensor.float() / scale


def fake_quantize(
    tensor: torch.Tensor,
    scale: float,
    bits: int,
    signed: bool = True
) -> torch.Tensor:
    """
    Fake quantization for QAT (quantize then dequantize).
    
    Uses straight-through estimator for gradients.
    
    Args:
        tensor: Input tensor
        scale: Scale factor
        bits: Total bits
        signed: Whether signed
        
    Returns:
        Fake-quantized tensor (float dtype)
    """
    quantized = quantize_to_fixed_point(tensor, scale, bits, signed)
    return dequantize_from_fixed_point(quantized, scale)


class FakeQuantizeFunction(torch.autograd.Function):
    """Autograd function for fake quantization with STE."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float, bits: int, signed: bool) -> torch.Tensor:
        return fake_quantize(x, scale, bits, signed)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        return grad_output, None, None, None


def fake_quantize_grad(
    tensor: torch.Tensor,
    scale: float,
    bits: int,
    signed: bool = True
) -> torch.Tensor:
    """Fake quantization with gradient support."""
    return FakeQuantizeFunction.apply(tensor, scale, bits, signed)


class QuantizedLinear(nn.Module):
    """
    Linear layer with quantization simulation.
    
    For QAT: simulates fixed-point arithmetic during forward pass
    while maintaining float gradients for backprop.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: QuantizationConfig,
        bias: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        # Float weights (for training)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with fake quantization."""
        # Quantize input
        x_q = fake_quantize_grad(x, self.config.activation_scale,
                                  self.config.activation_bits,
                                  self.config.activation_signed)
        
        # Quantize weights
        w_q = fake_quantize_grad(self.weight, self.config.weight_scale,
                                  self.config.weight_bits,
                                  self.config.weight_signed)
        
        # Linear operation (simulates MAC with accumulator)
        out = torch.nn.functional.linear(x_q, w_q, self.bias)
        
        # Quantize output
        out_q = fake_quantize_grad(out, self.config.activation_scale,
                                    self.config.activation_bits,
                                    self.config.activation_signed)
        
        return out_q
    
    def get_quantized_weights(self) -> Dict[str, torch.Tensor]:
        """Export quantized weights as integers."""
        w_int = quantize_to_fixed_point(
            self.weight.data,
            self.config.weight_scale,
            self.config.weight_bits,
            self.config.weight_signed
        ).to(torch.int16)
        
        result = {'weight': w_int}
        
        if self.bias is not None:
            # Bias uses activation format (Q8.8) for addition
            b_int = quantize_to_fixed_point(
                self.bias.data,
                self.config.activation_scale,
                self.config.activation_bits,
                self.config.activation_signed
            ).to(torch.int16)
            result['bias'] = b_int
            
        return result


def export_weights_binary(
    weights: Dict[str, torch.Tensor],
    output_path: Union[str, Path],
    endianness: str = 'little'
) -> None:
    """
    Export quantized weights to binary file for FPGA.
    
    File format:
    - Header: 4 bytes magic (0xDPD1), 4 bytes total size
    - For each tensor:
      - 32 bytes name (null-padded)
      - 4 bytes shape[0]
      - 4 bytes shape[1] (or 1 if 1D)
      - N × 2 bytes data (int16)
      
    Args:
        weights: Dictionary of weight tensors (int16)
        output_path: Output binary file path
        endianness: 'little' or 'big'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    byte_order = '<' if endianness == 'little' else '>'
    
    with open(output_path, 'wb') as f:
        # Magic number
        f.write(struct.pack(f'{byte_order}I', 0x44504431))  # 'DPD1'
        
        # Placeholder for total size (will update later)
        size_pos = f.tell()
        f.write(struct.pack(f'{byte_order}I', 0))
        
        for name, tensor in weights.items():
            # Name (32 bytes, null-padded)
            name_bytes = name.encode('utf-8')[:31].ljust(32, b'\x00')
            f.write(name_bytes)
            
            # Shape
            if tensor.dim() == 1:
                shape = (tensor.shape[0], 1)
            else:
                shape = (tensor.shape[0], tensor.shape[1])
            f.write(struct.pack(f'{byte_order}II', *shape))
            
            # Data (int16)
            data = tensor.flatten().numpy().astype(np.int16)
            f.write(data.tobytes())
            
        # Update total size
        total_size = f.tell()
        f.seek(size_pos)
        f.write(struct.pack(f'{byte_order}I', total_size))


def load_weights_binary(
    input_path: Union[str, Path],
    endianness: str = 'little'
) -> Dict[str, torch.Tensor]:
    """
    Load quantized weights from binary file.
    
    Args:
        input_path: Input binary file path
        endianness: 'little' or 'big'
        
    Returns:
        Dictionary of weight tensors (int16)
    """
    input_path = Path(input_path)
    byte_order = '<' if endianness == 'little' else '>'
    
    weights = {}
    
    with open(input_path, 'rb') as f:
        # Magic number
        magic = struct.unpack(f'{byte_order}I', f.read(4))[0]
        if magic != 0x44504431:
            raise ValueError(f"Invalid magic number: {hex(magic)}")
            
        # Total size
        total_size = struct.unpack(f'{byte_order}I', f.read(4))[0]
        
        while f.tell() < total_size:
            # Name
            name_bytes = f.read(32)
            name = name_bytes.rstrip(b'\x00').decode('utf-8')
            
            # Shape
            shape = struct.unpack(f'{byte_order}II', f.read(8))
            
            # Data
            num_elements = shape[0] * shape[1]
            data = np.frombuffer(f.read(num_elements * 2), dtype=np.int16)
            
            if shape[1] == 1:
                tensor = torch.from_numpy(data.copy()).reshape(shape[0])
            else:
                tensor = torch.from_numpy(data.copy()).reshape(shape)
                
            weights[name] = tensor
            
    return weights


def generate_tanh_lut(
    num_entries: int = 256,
    input_range: float = 4.0,
    output_bits: int = 16,
    output_frac_bits: int = 15
) -> torch.Tensor:
    """
    Generate tanh lookup table for FPGA.
    
    Args:
        num_entries: Number of LUT entries
        input_range: Input range [-input_range, +input_range]
        output_bits: Output bit width
        output_frac_bits: Output fractional bits
        
    Returns:
        LUT values as int16 tensor
    """
    # Input values
    x = torch.linspace(-input_range, input_range, num_entries)
    
    # Compute tanh
    y = torch.tanh(x)
    
    # Quantize output
    scale = 2 ** output_frac_bits
    y_int = quantize_to_fixed_point(y, scale, output_bits, signed=True)
    
    return y_int.to(torch.int16)


def generate_leaky_relu_params(
    negative_slope: float = 0.2,
    bits: int = 16
) -> Dict[str, int]:
    """
    Generate LeakyReLU parameters for FPGA.
    
    For efficiency, negative_slope is approximated as 1/2^n (shift operation).
    
    Args:
        negative_slope: Desired negative slope (e.g., 0.2)
        bits: Precision bits
        
    Returns:
        Dictionary with shift amount and actual slope
    """
    # Find closest power-of-2 approximation
    # 0.2 ≈ 1/5 ≈ 1/4 = 0.25 (shift by 2) or 1/8 = 0.125 (shift by 3)
    
    best_shift = 2
    best_error = float('inf')
    
    for shift in range(1, 8):
        approx = 1.0 / (2 ** shift)
        error = abs(approx - negative_slope)
        if error < best_error:
            best_error = error
            best_shift = shift
            
    return {
        'shift_amount': best_shift,
        'actual_slope': 1.0 / (2 ** best_shift),
        'target_slope': negative_slope,
        'error': best_error
    }


if __name__ == "__main__":
    print("Testing Quantization Utilities")
    print("=" * 50)
    
    # Test config
    config = QuantizationConfig()
    print(f"\nQuantization Config:")
    print(f"  Weight: Q1.{config.weight_frac_bits} ({config.weight_bits} bits)")
    print(f"  Activation: Q8.{config.activation_frac_bits} ({config.activation_bits} bits)")
    print(f"  Weight range: {config.weight_range}")
    print(f"  Activation range: {config.activation_range}")
    
    # Test quantization
    print("\n\nQuantization Test:")
    x = torch.tensor([0.5, -0.25, 0.999, -1.0, 0.0])
    x_q = quantize_to_fixed_point(x, config.weight_scale, config.weight_bits)
    x_dq = dequantize_from_fixed_point(x_q, config.weight_scale)
    
    print(f"  Original: {x.numpy()}")
    print(f"  Quantized (int): {x_q.numpy()}")
    print(f"  Dequantized: {x_dq.numpy()}")
    print(f"  Error: {(x - x_dq).abs().numpy()}")
    
    # Test tanh LUT
    print("\n\nTanh LUT:")
    lut = generate_tanh_lut(num_entries=16)
    print(f"  Shape: {lut.shape}")
    print(f"  Range: [{lut.min()}, {lut.max()}]")
    
    # Test LeakyReLU params
    print("\n\nLeakyReLU Parameters (α=0.2):")
    params = generate_leaky_relu_params(0.2)
    print(f"  Shift amount: {params['shift_amount']}")
    print(f"  Actual slope: {params['actual_slope']}")
    print(f"  Error: {params['error']:.4f}")
    
    # Test binary export
    print("\n\nBinary Export Test:")
    test_weights = {
        'fc1_weight': torch.randint(-32768, 32767, (32, 18), dtype=torch.int16),
        'fc1_bias': torch.randint(-128, 127, (32,), dtype=torch.int16)
    }
    
    export_weights_binary(test_weights, '/tmp/test_weights.bin')
    loaded = load_weights_binary('/tmp/test_weights.bin')
    
    print(f"  Exported and loaded successfully")
    print(f"  fc1_weight match: {(test_weights['fc1_weight'] == loaded['fc1_weight']).all()}")
    print(f"  fc1_bias match: {(test_weights['fc1_bias'] == loaded['fc1_bias']).all()}")
