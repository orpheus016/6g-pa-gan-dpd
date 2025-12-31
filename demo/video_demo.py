#!/usr/bin/env python3
"""
Video Demo Script for 6G PA DPD System
LSI Design Contest 29th - Okinawa

============================================
DIGITAL LOOPBACK DEMO (No RF Equipment!)
============================================

This demo runs entirely in software/FPGA without requiring:
- Vector Signal Analyzer
- Real PA hardware  
- ADC/DAC converters
- Temperature sensors

Data Flow:
    Python GUI → USB/Ethernet → PYNQ PS → AXI → DPD PL → PA Twin PL → AXI → PS → GUI

Demonstrates:
1. DPD Enable/Bypass toggle (Button 0)
2. Temperature state switching (Button 2 or Switches)
3. A-SPSA online adaptation convergence
4. Real-time EVM/ACPR improvement metrics
5. Constellation and spectrum visualization

For FPGA Demo:
    python demo/video_demo.py --fpga --ip 192.168.2.99

For Software-Only Demo:
    python demo/video_demo.py --software
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.patches as mpatches
import torch
import yaml
import time
import argparse
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import TDNNGenerator, PADigitalTwin
from utils.spectral_loss import compute_evm, compute_acpr, compute_nmse


class TempState(Enum):
    """Temperature states for demo."""
    COLD = 0
    NORMAL = 1
    HOT = 2


@dataclass
class DemoState:
    """Current demo state."""
    dpd_enabled: bool = True
    adapt_enabled: bool = True
    temp_state: TempState = TempState.NORMAL
    iteration: int = 0
    
    # Metrics history
    evm_history: list = None
    acpr_history: list = None
    convergence_history: list = None
    
    def __post_init__(self):
        self.evm_history = []
        self.acpr_history = []
        self.convergence_history = []


class DPDDemo:
    """
    Real-time DPD demonstration for LSI Design Contest.
    
    Supports both:
    - Software-only simulation
    - FPGA hardware via PYNQ overlay
    """
    
    def __init__(self, config_path: str = 'config/config.yaml',
                 checkpoint_path: str = None,
                 use_fpga: bool = False,
                 fpga_ip: str = '192.168.2.99'):
        
        self.use_fpga = use_fpga
        self.fpga_ip = fpga_ip
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize state
        self.state = DemoState()
        
        # Initialize models (software simulation)
        self._init_models(checkpoint_path)
        
        # Initialize FPGA if requested
        if use_fpga:
            self._init_fpga()
        
        # Signal generation parameters
        self.sample_rate = self.config['system']['sample_rate']
        self.num_samples = 1024
        self.num_subcarriers = 64
        
        # Metrics buffers
        self.history_len = 200
        self.evm_buffer = deque(maxlen=self.history_len)
        self.acpr_buffer = deque(maxlen=self.history_len)
        
        # Temperature drift parameters (for PA twin)
        self.temp_params = {
            TempState.COLD: {'gain': 0.98, 'phase': -5, 'am_am': 1.05},
            TempState.NORMAL: {'gain': 1.0, 'phase': 0, 'am_am': 1.0},
            TempState.HOT: {'gain': 0.95, 'phase': 10, 'am_am': 0.92}
        }
        
    def _init_models(self, checkpoint_path: str):
        """Initialize PyTorch models for software simulation."""
        gen_config = self.config['model']['generator']
        
        self.generator = TDNNGenerator(
            input_dim=gen_config['input_dim'],
            hidden_dims=gen_config['hidden_dims'],
            output_dim=gen_config['output_dim'],
            quantize=True,
            num_bits=self.config['quantization']['weight']['bits']
        )
        
        self.pa_twin = PADigitalTwin(
            memory_depth=self.config['pa']['memory_depth'],
            nonlinear_order=self.config['pa']['nonlinear_order']
        )
        
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            print(f"Loaded checkpoint: {checkpoint_path}")
        
        self.generator.eval()
        
    def _init_fpga(self):
        """Initialize PYNQ FPGA overlay."""
        try:
            from pynq import Overlay, allocate
            
            print(f"Connecting to PYNQ at {self.fpga_ip}...")
            
            # Load overlay (assumes bitstream is pre-loaded)
            self.overlay = Overlay('dpd_system.bit')
            
            # Get DMA engines
            self.dma_send = self.overlay.axi_dma_0.sendchannel
            self.dma_recv = self.overlay.axi_dma_0.recvchannel
            
            # Allocate buffers
            self.input_buffer = allocate(shape=(self.num_samples * 2,), dtype=np.int16)
            self.output_buffer = allocate(shape=(self.num_samples * 2,), dtype=np.int16)
            
            # Get control registers
            self.ctrl_reg = self.overlay.dpd_control_0
            
            print("FPGA initialized successfully!")
            
        except ImportError:
            print("PYNQ library not available. Falling back to software simulation.")
            self.use_fpga = False
        except Exception as e:
            print(f"FPGA initialization failed: {e}")
            print("Falling back to software simulation.")
            self.use_fpga = False
    
    def generate_ofdm_signal(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate OFDM-like test signal."""
        # QAM-64 constellation
        qam_points = np.array([
            complex(i, j) 
            for i in [-7, -5, -3, -1, 1, 3, 5, 7] 
            for j in [-7, -5, -3, -1, 1, 3, 5, 7]
        ]) / 7.0  # Normalize
        
        # Random symbols
        symbols = qam_points[np.random.randint(0, 64, self.num_subcarriers)]
        
        # IFFT to time domain
        freq_domain = np.zeros(self.num_samples, dtype=complex)
        freq_domain[1:self.num_subcarriers+1] = symbols
        freq_domain[-self.num_subcarriers:] = np.conj(symbols[::-1])
        
        time_signal = np.fft.ifft(freq_domain) * np.sqrt(self.num_samples)
        
        # Add cyclic prefix
        cp_len = self.num_samples // 8
        time_signal = np.concatenate([time_signal[-cp_len:], time_signal])
        
        # Scale to avoid clipping
        time_signal = time_signal / (np.max(np.abs(time_signal)) + 1e-6) * 0.8
        
        return time_signal[:self.num_samples].real, time_signal[:self.num_samples].imag
    
    def process_signal(self, in_i: np.ndarray, in_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process signal through DPD and PA."""
        if self.use_fpga:
            return self._process_fpga(in_i, in_q)
        else:
            return self._process_software(in_i, in_q)
    
    def _process_software(self, in_i: np.ndarray, in_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Software simulation of DPD + PA."""
        # Convert to torch
        x = torch.stack([
            torch.tensor(in_i, dtype=torch.float32),
            torch.tensor(in_q, dtype=torch.float32)
        ], dim=-1).unsqueeze(0)  # [1, N, 2]
        
        with torch.no_grad():
            if self.state.dpd_enabled:
                # Apply DPD
                dpd_out = self.generator(x)
            else:
                dpd_out = x
            
            # Apply PA with temperature-dependent parameters
            temp_params = self.temp_params[self.state.temp_state]
            pa_out = self.pa_twin(
                dpd_out, 
                gain_scale=temp_params['gain'],
                phase_offset=temp_params['phase'],
                am_am_scale=temp_params['am_am']
            )
        
        out_i = pa_out[0, :, 0].numpy()
        out_q = pa_out[0, :, 1].numpy()
        
        return out_i, out_q
    
    def _process_fpga(self, in_i: np.ndarray, in_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """FPGA hardware processing via DMA."""
        # Convert to Q1.15 fixed-point
        scale = 2**15 - 1
        self.input_buffer[0::2] = (in_i * scale).astype(np.int16)
        self.input_buffer[1::2] = (in_q * scale).astype(np.int16)
        
        # Set control registers
        self.ctrl_reg.write(0x00, 1 if self.state.dpd_enabled else 0)
        self.ctrl_reg.write(0x04, 1 if self.state.adapt_enabled else 0)
        self.ctrl_reg.write(0x08, self.state.temp_state.value)
        
        # DMA transfer
        self.dma_send.transfer(self.input_buffer)
        self.dma_recv.transfer(self.output_buffer)
        self.dma_send.wait()
        self.dma_recv.wait()
        
        # Convert back to float
        out_i = self.output_buffer[0::2].astype(np.float32) / scale
        out_q = self.output_buffer[1::2].astype(np.float32) / scale
        
        return out_i, out_q
    
    def compute_metrics(self, ideal_i: np.ndarray, ideal_q: np.ndarray,
                        actual_i: np.ndarray, actual_q: np.ndarray) -> dict:
        """Compute EVM, ACPR, NMSE metrics."""
        ideal = ideal_i + 1j * ideal_q
        actual = actual_i + 1j * actual_q
        
        # EVM
        evm = compute_evm(
            torch.tensor(actual_i), torch.tensor(actual_q),
            torch.tensor(ideal_i), torch.tensor(ideal_q)
        ).item()
        
        # ACPR (simplified)
        acpr = compute_acpr(
            torch.tensor(actual_i), torch.tensor(actual_q),
            self.num_subcarriers
        ).item()
        
        # NMSE
        nmse = compute_nmse(
            torch.tensor(actual_i), torch.tensor(actual_q),
            torch.tensor(ideal_i), torch.tensor(ideal_q)
        ).item()
        
        return {
            'evm_db': 20 * np.log10(evm + 1e-10),
            'acpr_db': acpr,
            'nmse_db': 10 * np.log10(nmse + 1e-10)
        }
    
    def toggle_dpd(self):
        """Toggle DPD enable state."""
        self.state.dpd_enabled = not self.state.dpd_enabled
        print(f"DPD: {'ENABLED' if self.state.dpd_enabled else 'BYPASSED'}")
    
    def toggle_adapt(self):
        """Toggle adaptation enable state."""
        self.state.adapt_enabled = not self.state.adapt_enabled
        print(f"Adaptation: {'ENABLED' if self.state.adapt_enabled else 'DISABLED'}")
    
    def cycle_temperature(self):
        """Cycle through temperature states."""
        states = list(TempState)
        current_idx = states.index(self.state.temp_state)
        self.state.temp_state = states[(current_idx + 1) % len(states)]
        print(f"Temperature: {self.state.temp_state.name}")
        
        # Reset adaptation (simulates weight bank switch + anneal reset)
        self.state.convergence_history = []
    
    def run_demo(self):
        """Run the interactive demo with matplotlib animation."""
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('6G PA DPD Demo - LSI Design Contest 29th', fontsize=14, fontweight='bold')
        
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Subplots
        ax_const_in = fig.add_subplot(gs[0, 0])
        ax_const_out = fig.add_subplot(gs[0, 1])
        ax_spectrum = fig.add_subplot(gs[0, 2:])
        ax_evm = fig.add_subplot(gs[1, :2])
        ax_acpr = fig.add_subplot(gs[1, 2:])
        ax_status = fig.add_subplot(gs[2, :2])
        ax_convergence = fig.add_subplot(gs[2, 2:])
        
        # Initialize plots
        ax_const_in.set_title('Input Constellation')
        ax_const_out.set_title('Output Constellation')
        ax_spectrum.set_title('Power Spectrum')
        ax_evm.set_title('EVM (dB)')
        ax_acpr.set_title('ACPR (dB)')
        ax_status.set_title('System Status')
        ax_convergence.set_title('Convergence')
        
        # Status display
        ax_status.axis('off')
        
        # Animation data
        evm_line, = ax_evm.plot([], [], 'b-', lw=2)
        acpr_line, = ax_acpr.plot([], [], 'r-', lw=2)
        
        ax_evm.set_xlim(0, self.history_len)
        ax_evm.set_ylim(-40, 0)
        ax_evm.axhline(y=-25, color='g', linestyle='--', label='Target')
        ax_evm.legend()
        
        ax_acpr.set_xlim(0, self.history_len)
        ax_acpr.set_ylim(-60, -20)
        ax_acpr.axhline(y=-45, color='g', linestyle='--', label='Target')
        ax_acpr.legend()
        
        def animate(frame):
            # Generate signal
            in_i, in_q = self.generate_ofdm_signal()
            
            # Process through DPD + PA
            out_i, out_q = self.process_signal(in_i, in_q)
            
            # Compute metrics (compare output to ideal linear PA)
            ideal_i, ideal_q = in_i * 1.0, in_q * 1.0  # Ideal: just gain
            metrics = self.compute_metrics(ideal_i, ideal_q, out_i, out_q)
            
            # Update buffers
            self.evm_buffer.append(metrics['evm_db'])
            self.acpr_buffer.append(metrics['acpr_db'])
            
            # Update constellation plots
            ax_const_in.cla()
            ax_const_in.scatter(in_i[::4], in_q[::4], c='blue', alpha=0.5, s=10)
            ax_const_in.set_xlim(-1.5, 1.5)
            ax_const_in.set_ylim(-1.5, 1.5)
            ax_const_in.set_title('Input Constellation')
            ax_const_in.grid(True, alpha=0.3)
            ax_const_in.set_aspect('equal')
            
            ax_const_out.cla()
            ax_const_out.scatter(out_i[::4], out_q[::4], c='red', alpha=0.5, s=10)
            ax_const_out.set_xlim(-1.5, 1.5)
            ax_const_out.set_ylim(-1.5, 1.5)
            ax_const_out.set_title(f'Output Constellation (EVM: {metrics["evm_db"]:.1f} dB)')
            ax_const_out.grid(True, alpha=0.3)
            ax_const_out.set_aspect('equal')
            
            # Update spectrum
            ax_spectrum.cla()
            freq = np.fft.fftfreq(len(out_i), 1/self.sample_rate)
            spectrum_out = 20 * np.log10(np.abs(np.fft.fft(out_i + 1j*out_q)) + 1e-10)
            spectrum_in = 20 * np.log10(np.abs(np.fft.fft(in_i + 1j*in_q)) + 1e-10)
            ax_spectrum.plot(freq/1e6, np.fft.fftshift(spectrum_in), 'b-', alpha=0.5, label='Input')
            ax_spectrum.plot(freq/1e6, np.fft.fftshift(spectrum_out), 'r-', label='Output')
            ax_spectrum.set_xlabel('Frequency (MHz)')
            ax_spectrum.set_ylabel('Power (dB)')
            ax_spectrum.set_title(f'Power Spectrum (ACPR: {metrics["acpr_db"]:.1f} dB)')
            ax_spectrum.legend()
            ax_spectrum.grid(True, alpha=0.3)
            ax_spectrum.set_ylim(-80, 20)
            
            # Update metric plots
            x_data = list(range(len(self.evm_buffer)))
            evm_line.set_data(x_data, list(self.evm_buffer))
            acpr_line.set_data(x_data, list(self.acpr_buffer))
            
            # Update status
            ax_status.cla()
            ax_status.axis('off')
            
            status_text = f"""
╔══════════════════════════════════════╗
║  DPD Status:     {'● ENABLED ' if self.state.dpd_enabled else '○ BYPASSED'}       ║
║  Adaptation:     {'● RUNNING ' if self.state.adapt_enabled else '○ STOPPED'}       ║
║  Temperature:    {self.state.temp_state.name:^10}         ║
╠══════════════════════════════════════╣
║  Current Metrics:                    ║
║    EVM:  {metrics['evm_db']:>7.2f} dB                ║
║    ACPR: {metrics['acpr_db']:>7.2f} dB                ║
║    NMSE: {metrics['nmse_db']:>7.2f} dB                ║
╚══════════════════════════════════════╝

Controls (Keyboard):
  [D] Toggle DPD    [A] Toggle Adapt
  [T] Cycle Temp    [Q] Quit
"""
            ax_status.text(0.1, 0.5, status_text, transform=ax_status.transAxes,
                          fontfamily='monospace', fontsize=10, verticalalignment='center')
            
            # Color indicators
            dpd_color = 'green' if self.state.dpd_enabled else 'gray'
            adapt_color = 'green' if self.state.adapt_enabled else 'gray'
            temp_colors = {'COLD': 'blue', 'NORMAL': 'green', 'HOT': 'red'}
            
            self.state.iteration += 1
            
            return evm_line, acpr_line
        
        def on_key(event):
            if event.key == 'd':
                self.toggle_dpd()
            elif event.key == 'a':
                self.toggle_adapt()
            elif event.key == 't':
                self.cycle_temperature()
            elif event.key == 'q':
                plt.close()
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        ani = animation.FuncAnimation(fig, animate, frames=None,
                                       interval=100, blit=False)
        
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='6G PA DPD Demo - LSI Design Contest 29th'
    )
    parser.add_argument('--fpga', action='store_true',
                        help='Use FPGA hardware via PYNQ')
    parser.add_argument('--software', action='store_true',
                        help='Use software simulation only')
    parser.add_argument('--ip', type=str, default='192.168.2.99',
                        help='PYNQ board IP address')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Configuration file path')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Trained model checkpoint')
    
    args = parser.parse_args()
    
    use_fpga = args.fpga and not args.software
    
    print("=" * 60)
    print("6G PA DPD Demo - LSI Design Contest 29th")
    print("=" * 60)
    print(f"Mode: {'FPGA' if use_fpga else 'Software Simulation'}")
    print()
    
    demo = DPDDemo(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        use_fpga=use_fpga,
        fpga_ip=args.ip
    )
    
    demo.run_demo()


if __name__ == '__main__':
    main()
