# =============================================================================
# 6G PA GAN-DPD: PA Digital Twin (Volterra Model with Thermal Effects)
# =============================================================================
"""
PA DIGITAL TWIN FOR DPD TRAINING AND VALIDATION
================================================

This module implements a digital twin of a GaN Power Amplifier using
Volterra series modeling with:
- Memory effects (temporal nonlinearity)
- AM-AM and AM-PM distortion
- Temperature-dependent coefficient variations
- Additive noise (AWGN + phase noise)

The digital twin is used for:
1. Generating training data from ideal DPD → PA → output
2. Validating DPD performance without hardware
3. Temperature robustness testing

Volterra Model (Memory Polynomial):
    y(n) = Σ_k Σ_m h_k,m · x(n-m) · |x(n-m)|^(k-1)

where:
    k: nonlinearity order (1, 3, 5, 7)
    m: memory depth (0, 1, ..., M-1)
    h_k,m: complex Volterra coefficients
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List
import math


class VolterraPA(nn.Module):
    """
    Volterra Series Power Amplifier Model.
    
    Implements a memory polynomial model commonly used for PA behavioral modeling.
    
    Args:
        memory_depth: Number of memory taps (M)
        nonlinear_order: Maximum nonlinearity order (K, must be odd)
        gain_db: Small-signal gain in dB
        p1db_dbm: 1dB compression point in dBm
        
    Model equation:
        y(n) = Σ_{k=1,3,5,...} Σ_{m=0}^{M-1} h_{k,m} · x(n-m) · |x(n-m)|^{k-1}
    """
    def __init__(
        self,
        memory_depth: int = 5,
        nonlinear_order: int = 7,
        gain_db: float = 30.0,
        p1db_dbm: float = 40.0
    ):
        super().__init__()
        
        self.memory_depth = memory_depth
        self.nonlinear_order = nonlinear_order
        self.gain_db = gain_db
        self.p1db_dbm = p1db_dbm
        
        # Nonlinearity orders (odd only: 1, 3, 5, 7)
        self.orders = list(range(1, nonlinear_order + 1, 2))
        self.num_orders = len(self.orders)
        
        # Number of Volterra coefficients
        self.num_coeffs = self.num_orders * memory_depth
        
        # Initialize Volterra coefficients (complex)
        # Real and imaginary parts stored separately for PyTorch compatibility
        self.register_buffer('coeffs_real', torch.zeros(self.num_orders, memory_depth))
        self.register_buffer('coeffs_imag', torch.zeros(self.num_orders, memory_depth))
        
        # Initialize coefficients based on typical PA characteristics
        self._init_coefficients()
        
    def _init_coefficients(self):
        """
        Initialize Volterra coefficients to model typical GaN PA behavior.
        
        Based on typical values from OpenDPD measurements.
        """
        # Linear gain (small-signal)
        linear_gain = 10 ** (self.gain_db / 20)
        
        # Set linear coefficients (k=1)
        self.coeffs_real[0, 0] = linear_gain
        self.coeffs_imag[0, 0] = 0.0
        
        # Memory taps for linear term (exponential decay)
        for m in range(1, self.memory_depth):
            decay = 0.1 * np.exp(-m / 2)
            self.coeffs_real[0, m] = decay * linear_gain
            self.coeffs_imag[0, m] = 0.02 * decay * linear_gain
            
        # Nonlinear coefficients (decreasing magnitude with order)
        for k_idx, k in enumerate(self.orders[1:], start=1):
            # AM-AM coefficient (gain compression)
            am_am = -0.05 * linear_gain / (k ** 1.5)
            # AM-PM coefficient (phase distortion)
            am_pm = 0.02 * linear_gain / (k ** 1.5)
            
            # Main tap
            self.coeffs_real[k_idx, 0] = am_am
            self.coeffs_imag[k_idx, 0] = am_pm
            
            # Memory taps
            for m in range(1, self.memory_depth):
                decay = 0.3 * np.exp(-m / 1.5)
                self.coeffs_real[k_idx, m] = am_am * decay
                self.coeffs_imag[k_idx, m] = am_pm * decay * 0.5
                
    def get_coefficients(self) -> torch.Tensor:
        """Return complex Volterra coefficients."""
        return torch.complex(self.coeffs_real, self.coeffs_imag)
    
    def set_coefficients(self, coeffs: torch.Tensor):
        """Set Volterra coefficients from complex tensor."""
        self.coeffs_real.copy_(coeffs.real)
        self.coeffs_imag.copy_(coeffs.imag)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply PA model to input signal.
        
        Args:
            x: Input IQ tensor [batch, seq_len, 2] (I, Q channels)
               or complex tensor [batch, seq_len]
               
        Returns:
            y: Output IQ tensor [batch, seq_len - M + 1, 2]
        """
        # Convert to complex if IQ format
        if x.dim() == 3 and x.shape[-1] == 2:
            x_complex = torch.complex(x[..., 0], x[..., 1])
        else:
            x_complex = x
            
        batch_size, seq_len = x_complex.shape
        M = self.memory_depth
        
        if seq_len < M:
            raise ValueError(f"Sequence length {seq_len} must be >= memory depth {M}")
            
        # Output length
        out_len = seq_len - M + 1
        
        # Get complex coefficients
        coeffs = self.get_coefficients()  # [num_orders, M]
        
        # Compute Volterra output
        y = torch.zeros(batch_size, out_len, dtype=torch.complex64, device=x.device)
        
        for n in range(out_len):
            for k_idx, k in enumerate(self.orders):
                for m in range(M):
                    # x(n-m) for current output sample
                    x_nm = x_complex[:, n + M - 1 - m]
                    # |x(n-m)|^(k-1)
                    x_env = x_nm.abs() ** (k - 1) if k > 1 else torch.ones_like(x_nm.abs())
                    # Volterra term
                    y[:, n] += coeffs[k_idx, m] * x_nm * x_env
                    
        # Convert back to IQ format
        y_iq = torch.stack([y.real, y.imag], dim=-1)
        
        return y_iq


class PADigitalTwin(nn.Module):
    """
    Complete PA Digital Twin with thermal effects and noise.
    
    Features:
    - Volterra PA model with memory effects
    - Temperature-dependent coefficient scaling
    - AWGN noise
    - Phase noise
    
    Args:
        memory_depth: PA memory depth
        nonlinear_order: Maximum nonlinearity order
        gain_db: Small-signal gain
        p1db_dbm: 1dB compression point
        noise_floor_db: Noise floor relative to signal
        phase_noise_deg: RMS phase noise in degrees
    """
    def __init__(
        self,
        memory_depth: int = 5,
        nonlinear_order: int = 7,
        gain_db: float = 30.0,
        p1db_dbm: float = 40.0,
        noise_floor_db: float = -60.0,
        phase_noise_deg: float = 0.5
    ):
        super().__init__()
        
        self.memory_depth = memory_depth
        self.noise_floor_db = noise_floor_db
        self.phase_noise_deg = phase_noise_deg
        
        # Base Volterra PA model
        self.pa_model = VolterraPA(
            memory_depth=memory_depth,
            nonlinear_order=nonlinear_order,
            gain_db=gain_db,
            p1db_dbm=p1db_dbm
        )
        
        # Temperature states
        self.temp_states = ['cold', 'normal', 'hot']
        self.current_temp_state = 'normal'
        
        # Temperature-dependent scaling factors
        # Based on typical GaN PA thermal behavior
        self.temp_coefficients = {
            'cold': {
                'gain_scale': 1.02,      # +2% gain
                'phase_offset_deg': -5,   # -5° phase shift
                'am_am_scale': 1.05,     # More compression
                'am_pm_scale': 0.95      # Less AM-PM
            },
            'normal': {
                'gain_scale': 1.0,
                'phase_offset_deg': 0,
                'am_am_scale': 1.0,
                'am_pm_scale': 1.0
            },
            'hot': {
                'gain_scale': 0.95,      # -5% gain
                'phase_offset_deg': 10,   # +10° phase shift
                'am_am_scale': 0.92,     # Less compression
                'am_pm_scale': 1.15      # More AM-PM
            }
        }
        
        # Store base coefficients
        self.register_buffer('base_coeffs_real', self.pa_model.coeffs_real.clone())
        self.register_buffer('base_coeffs_imag', self.pa_model.coeffs_imag.clone())
        
    def set_temperature_state(self, state: str):
        """
        Set temperature state and update PA coefficients.
        
        Args:
            state: Temperature state ('cold', 'normal', 'hot')
        """
        if state not in self.temp_states:
            raise ValueError(f"Invalid temperature state: {state}")
            
        self.current_temp_state = state
        temp_coeff = self.temp_coefficients[state]
        
        # Apply temperature scaling to Volterra coefficients
        gain_scale = temp_coeff['gain_scale']
        phase_offset = temp_coeff['phase_offset_deg'] * np.pi / 180
        am_am_scale = temp_coeff['am_am_scale']
        am_pm_scale = temp_coeff['am_pm_scale']
        
        # Scale linear coefficients (k=1)
        new_coeffs_real = self.base_coeffs_real.clone()
        new_coeffs_imag = self.base_coeffs_imag.clone()
        
        # Linear term: gain + phase rotation
        new_coeffs_real[0] = self.base_coeffs_real[0] * gain_scale * np.cos(phase_offset)
        new_coeffs_imag[0] = self.base_coeffs_real[0] * gain_scale * np.sin(phase_offset)
        
        # Nonlinear terms: AM-AM and AM-PM scaling
        for k_idx in range(1, self.pa_model.num_orders):
            new_coeffs_real[k_idx] = self.base_coeffs_real[k_idx] * am_am_scale
            new_coeffs_imag[k_idx] = self.base_coeffs_imag[k_idx] * am_pm_scale
            
        self.pa_model.coeffs_real.copy_(new_coeffs_real)
        self.pa_model.coeffs_imag.copy_(new_coeffs_imag)
        
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add AWGN and phase noise to signal.
        
        Args:
            x: Input IQ tensor [batch, seq_len, 2]
            
        Returns:
            Noisy signal
        """
        # Signal power
        signal_power = (x ** 2).mean()
        
        # AWGN
        noise_power = signal_power * (10 ** (self.noise_floor_db / 10))
        noise_std = torch.sqrt(noise_power)
        awgn = torch.randn_like(x) * noise_std
        
        # Phase noise
        if self.phase_noise_deg > 0:
            phase_noise_rad = self.phase_noise_deg * np.pi / 180
            phase_noise = torch.randn(x.shape[0], x.shape[1], 1, device=x.device) * phase_noise_rad
            
            # Apply phase rotation
            x_complex = torch.complex(x[..., 0], x[..., 1])
            phase_rotation = torch.exp(1j * phase_noise.squeeze(-1))
            x_rotated = x_complex * phase_rotation
            x = torch.stack([x_rotated.real, x_rotated.imag], dim=-1)
            
        return x + awgn
    
    def forward(
        self, 
        x: torch.Tensor, 
        add_noise: bool = True,
        temp_state: Optional[str] = None
    ) -> torch.Tensor:
        """
        Apply PA model with thermal effects and noise.
        
        Args:
            x: Input IQ tensor [batch, seq_len, 2]
            add_noise: Whether to add noise
            temp_state: Temperature state (None = use current)
            
        Returns:
            PA output IQ tensor
        """
        if temp_state is not None:
            self.set_temperature_state(temp_state)
            
        # Apply Volterra PA model
        y = self.pa_model(x)
        
        # Add noise
        if add_noise:
            y = self.add_noise(y)
            
        return y
    
    def get_transfer_function(
        self, 
        num_points: int = 256,
        max_amplitude: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute AM-AM and AM-PM transfer functions.
        
        Args:
            num_points: Number of amplitude points
            max_amplitude: Maximum input amplitude
            
        Returns:
            input_amp: Input amplitude points
            output_amp: Output amplitude (AM-AM)
            output_phase: Output phase (AM-PM) in degrees
        """
        # Generate test signal (CW at different amplitudes)
        input_amp = torch.linspace(0.01, max_amplitude, num_points)
        
        output_amp = []
        output_phase = []
        
        for amp in input_amp:
            # Single sample test
            x = torch.tensor([[[amp.item(), 0.0]]])  # I only
            
            # Need enough samples for memory
            x_padded = x.repeat(1, self.memory_depth + 1, 1)
            
            with torch.no_grad():
                y = self.pa_model(x_padded)
                
            # Take last output
            y_complex = torch.complex(y[0, -1, 0], y[0, -1, 1])
            output_amp.append(y_complex.abs().item())
            output_phase.append(torch.angle(y_complex).item() * 180 / np.pi)
            
        return input_amp, torch.tensor(output_amp), torch.tensor(output_phase)


def create_pa_digital_twin(config: dict) -> PADigitalTwin:
    """
    Factory function to create PA digital twin from config.
    """
    pa_config = config['pa']
    
    return PADigitalTwin(
        memory_depth=pa_config.get('memory_depth', 5),
        nonlinear_order=pa_config.get('nonlinear_order', 7),
        gain_db=pa_config.get('gain_db', 30.0),
        p1db_dbm=pa_config.get('p1db_dbm', 40.0),
        noise_floor_db=pa_config.get('noise', {}).get('floor_db', -60),
        phase_noise_deg=pa_config.get('noise', {}).get('phase_noise_deg', 0.5)
    )


if __name__ == "__main__":
    print("Testing PA Digital Twin")
    print("=" * 50)
    
    # Create PA model
    pa = PADigitalTwin(
        memory_depth=5,
        nonlinear_order=7,
        gain_db=30.0,
        p1db_dbm=40.0
    )
    
    print(f"Memory depth: {pa.memory_depth}")
    print(f"Number of Volterra coefficients: {pa.pa_model.num_coeffs}")
    
    # Test signal
    batch_size = 4
    seq_len = 128
    
    # Generate random IQ signal
    x = torch.randn(batch_size, seq_len, 2) * 0.3  # Normalized amplitude
    
    print(f"\nInput shape: {x.shape}")
    
    # Test each temperature state
    for temp in ['cold', 'normal', 'hot']:
        pa.set_temperature_state(temp)
        y = pa(x, add_noise=False)
        print(f"Output shape ({temp}): {y.shape}")
        print(f"  Output RMS: {y.pow(2).mean().sqrt():.4f}")
        
    # Test AM-AM/AM-PM curves
    print("\nAM-AM/AM-PM Transfer Functions:")
    for temp in ['cold', 'normal', 'hot']:
        pa.set_temperature_state(temp)
        inp, out_amp, out_phase = pa.get_transfer_function(num_points=10)
        print(f"\n{temp}:")
        print(f"  Gain at 0.1: {20*np.log10(out_amp[1]/inp[1]):.1f} dB")
        print(f"  Gain at 0.9: {20*np.log10(out_amp[-2]/inp[-2]):.1f} dB")
        print(f"  Phase at 0.9: {out_phase[-2]:.1f}°")
