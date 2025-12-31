#!/usr/bin/env python3
"""
HDMI Video Demo for 6G PA DPD System
LSI Design Contest 29th - Okinawa

============================================
HDMI LOOPBACK DEMO ARCHITECTURE
============================================

This demo uses the HDMI ports on PYNQ-Z1/ZCU104 for a clean signal path:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                              DEMO SETUP                                  │
    │                                                                          │
    │   ┌──────────┐    HDMI OUT    ┌─────────────┐    HDMI IN    ┌────────┐  │
    │   │  LAPTOP  │───────────────►│   FPGA      │──────────────►│MONITOR │  │
    │   │(Generate │                │(DPD + PA    │               │(Display│  │
    │   │ I/Q vis) │                │ Digital Twin)│               │results)│  │
    │   └──────────┘                └─────────────┘               └────────┘  │
    │                                                                          │
    │   Alternative: Single-board demo with PYNQ Jupyter                      │
    │                                                                          │
    │   ┌─────────────────────────────────────────────────────────────────┐   │
    │   │                    PYNQ-Z1 / ZCU104                              │   │
    │   │  ┌───────────────┐                         ┌─────────────────┐  │   │
    │   │  │ PS (Jupyter)  │◄──────AXI──────────────►│ PL (DPD Logic)  │  │   │
    │   │  │ - Python GUI  │                         │ - TDNN Gen      │  │   │
    │   │  │ - Matplotlib  │                         │ - PA Twin       │  │   │
    │   │  │ - Show on HDMI│                         │ - A-SPSA        │  │   │
    │   │  └───────┬───────┘                         └─────────────────┘  │   │
    │   │          │                                                       │   │
    │   │          ▼ HDMI OUT                                              │   │
    │   │   ┌──────────────┐                                               │   │
    │   │   │   MONITOR    │  ← Shows real-time constellation & metrics    │   │
    │   │   └──────────────┘                                               │   │
    │   └─────────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────┘

HDMI Ports Available:
- PYNQ-Z1: HDMI IN + HDMI OUT
- ZCU104: DisplayPort (similar usage)

User Controls:
- BTN0: Toggle DPD Enable/Bypass
- BTN1: Toggle Adaptation
- BTN2: Cycle Temperature State (Cold → Normal → Hot)
- SW0-1: Direct Temperature Select

LED Indicators:
- LED0: DPD Active
- LED1: Adaptation Running
- LED2: Temperature Cold
- LED3: Temperature Hot
- RGB0: Error Level (Green=Good, Red=Bad)
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Or 'Agg' for headless
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle
import matplotlib.animation as animation
import time
import argparse
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List
import threading
import queue

# Try PYNQ imports (will fail gracefully on non-FPGA systems)
try:
    from pynq import Overlay, allocate
    from pynq.lib.video import VideoMode, PIXEL_RGBA
    PYNQ_AVAILABLE = True
except ImportError:
    PYNQ_AVAILABLE = False
    print("PYNQ not available - running in software simulation mode")

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import yaml
    from models import TDNNGenerator, PADigitalTwin
    from utils.spectral_loss import compute_evm, compute_acpr
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch/models not available - using simplified simulation")


class TempState(Enum):
    COLD = 0
    NORMAL = 1
    HOT = 2


@dataclass
class DemoMetrics:
    """Real-time metrics for display."""
    evm_db: float = -20.0
    acpr_db: float = -40.0
    nmse_db: float = -15.0
    iteration: int = 0
    convergence_pct: float = 0.0
    
    # History for plotting
    evm_history: deque = field(default_factory=lambda: deque(maxlen=200))
    acpr_history: deque = field(default_factory=lambda: deque(maxlen=200))
    

@dataclass 
class DemoState:
    """Current demo state controlled by buttons/switches."""
    dpd_enabled: bool = True
    adapt_enabled: bool = True
    temp_state: TempState = TempState.NORMAL
    running: bool = True


class PASimulator:
    """Simplified PA model for software demo."""
    
    def __init__(self, temp_state: TempState = TempState.NORMAL):
        self.temp_state = temp_state
        self._update_coefficients()
    
    def _update_coefficients(self):
        """Update PA coefficients based on temperature."""
        temp_factors = {
            TempState.COLD: {'gain': 0.92, 'am_am': 0.12, 'am_pm': -8},
            TempState.NORMAL: {'gain': 1.0, 'am_am': 0.10, 'am_pm': 0},
            TempState.HOT: {'gain': 0.88, 'am_pm': 15, 'am_am': 0.15}
        }
        self.params = temp_factors[self.temp_state]
    
    def set_temperature(self, temp_state: TempState):
        self.temp_state = temp_state
        self._update_coefficients()
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply PA nonlinearity."""
        # AM-AM: gain compression
        mag = np.abs(x)
        gain = self.params['gain'] * (1 - self.params['am_am'] * mag**2)
        
        # AM-PM: phase rotation
        phase_shift = np.deg2rad(self.params['am_pm']) * mag**2
        
        return x * gain * np.exp(1j * phase_shift)


class DPDSimulator:
    """Simplified DPD model for software demo."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        # Simple polynomial predistorter coefficients
        self.coeffs = [1.0, 0.08, 0.02, -0.01]  # Will be "learned"
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return x
        
        mag = np.abs(x)
        # Polynomial predistortion
        pd_gain = sum(c * mag**(2*i) for i, c in enumerate(self.coeffs))
        return x * pd_gain
    
    def adapt(self, error: float):
        """Simple adaptation step."""
        # Adjust coefficients based on error
        lr = 0.001
        self.coeffs[1] += lr * error * np.random.randn()
        self.coeffs[2] += lr * 0.5 * error * np.random.randn()


class HDMIDemoApp:
    """
    Main HDMI Demo Application.
    
    Can run in three modes:
    1. FPGA + HDMI: Full hardware demo on PYNQ
    2. Software + Display: PyTorch simulation with matplotlib
    3. Headless: Generate video file for offline presentation
    """
    
    def __init__(self, 
                 mode: str = 'software',
                 config_path: str = 'config/config.yaml',
                 checkpoint_path: str = None,
                 output_video: str = None):
        
        self.mode = mode
        self.output_video = output_video
        
        # Load config if available
        self.config = self._load_config(config_path)
        
        # Initialize state
        self.state = DemoState()
        self.metrics = DemoMetrics()
        
        # Signal parameters
        self.sample_rate = 200e6
        self.num_samples = 1024
        self.frame_rate = 30
        
        # Initialize based on mode
        if mode == 'fpga' and PYNQ_AVAILABLE:
            self._init_fpga()
        else:
            self._init_software(checkpoint_path)
        
        # Event queue for button presses
        self.event_queue = queue.Queue()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return {'system': {'sample_rate': 200e6}}
    
    def _init_software(self, checkpoint_path: str):
        """Initialize software simulation."""
        print("Initializing software simulation mode...")
        
        self.pa = PASimulator()
        self.dpd = DPDSimulator()
        
        if TORCH_AVAILABLE and checkpoint_path:
            try:
                self._load_torch_models(checkpoint_path)
            except Exception as e:
                print(f"Could not load PyTorch models: {e}")
    
    def _load_torch_models(self, checkpoint_path: str):
        """Load trained PyTorch models."""
        gen_config = self.config.get('model', {}).get('generator', {})
        
        self.torch_generator = TDNNGenerator(
            input_dim=gen_config.get('input_dim', 18),
            hidden_dims=gen_config.get('hidden_dims', [32, 16]),
            output_dim=gen_config.get('output_dim', 2)
        )
        
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.torch_generator.load_state_dict(
                checkpoint['generator_state_dict']
            )
            self.torch_generator.eval()
            print(f"Loaded model from {checkpoint_path}")
    
    def _init_fpga(self):
        """Initialize FPGA hardware."""
        print("Initializing FPGA mode...")
        
        try:
            # Load overlay
            self.overlay = Overlay('dpd_system.bit')
            
            # Setup HDMI output
            self.hdmi_out = self.overlay.video.hdmi_out
            self.hdmi_out.configure(VideoMode(1280, 720, 24), PIXEL_RGBA)
            self.hdmi_out.start()
            
            # Get DPD control IP
            self.dpd_ctrl = self.overlay.dpd_top_0
            
            # Allocate DMA buffers
            self.input_buf = allocate(shape=(self.num_samples * 2,), dtype=np.int16)
            self.output_buf = allocate(shape=(self.num_samples * 2,), dtype=np.int16)
            
            # Setup button interrupts
            self._setup_button_handlers()
            
            print("FPGA initialized successfully!")
            
        except Exception as e:
            print(f"FPGA init failed: {e}, falling back to software")
            self.mode = 'software'
            self._init_software(None)
    
    def _setup_button_handlers(self):
        """Setup GPIO button handlers on FPGA."""
        if not hasattr(self, 'overlay'):
            return
            
        try:
            btns = self.overlay.btns_gpio
            
            # Button callbacks
            def btn0_handler(flag):
                self.event_queue.put(('dpd_toggle', None))
            
            def btn1_handler(flag):
                self.event_queue.put(('adapt_toggle', None))
            
            def btn2_handler(flag):
                self.event_queue.put(('temp_cycle', None))
            
            # Register handlers (PYNQ-specific)
            # btns.setCallback(btn0_handler, pin=0)
            
        except Exception as e:
            print(f"Button setup failed: {e}")
    
    def generate_test_signal(self) -> np.ndarray:
        """Generate OFDM-like test signal."""
        # QAM-64 symbols
        qam = np.array([complex(i, j) for i in range(-7, 8, 2) 
                        for j in range(-7, 8, 2)]) / 7.0
        
        # Random subcarrier symbols
        n_sc = 64
        symbols = qam[np.random.randint(0, len(qam), n_sc)]
        
        # OFDM modulation (IFFT)
        freq = np.zeros(self.num_samples, dtype=complex)
        freq[1:n_sc+1] = symbols
        time_signal = np.fft.ifft(freq) * np.sqrt(self.num_samples)
        
        # Scale for PA
        time_signal = time_signal / (np.max(np.abs(time_signal)) + 1e-6) * 0.7
        
        return time_signal
    
    def process_signal(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process signal through DPD + PA chain."""
        # Apply DPD (if enabled)
        x_dpd = self.dpd(x) if self.state.dpd_enabled else x
        
        # Apply PA nonlinearity
        self.pa.set_temperature(self.state.temp_state)
        y = self.pa(x_dpd)
        
        return x_dpd, y
    
    def compute_metrics(self, x_ref: np.ndarray, y: np.ndarray):
        """Compute EVM and ACPR metrics."""
        # EVM: Error Vector Magnitude
        error = y - x_ref
        evm_linear = np.sqrt(np.mean(np.abs(error)**2) / np.mean(np.abs(x_ref)**2))
        self.metrics.evm_db = 20 * np.log10(evm_linear + 1e-10)
        
        # ACPR: Adjacent Channel Power Ratio (simplified)
        spectrum = np.abs(np.fft.fft(y))**2
        n = len(spectrum)
        main_power = np.sum(spectrum[n//4:3*n//4])
        adj_power = np.sum(spectrum[:n//4]) + np.sum(spectrum[3*n//4:])
        self.metrics.acpr_db = 10 * np.log10(adj_power / (main_power + 1e-10))
        
        # Update history
        self.metrics.evm_history.append(self.metrics.evm_db)
        self.metrics.acpr_history.append(self.metrics.acpr_db)
        
        # Simple adaptation
        if self.state.adapt_enabled and self.state.dpd_enabled:
            self.dpd.adapt(evm_linear)
            self.metrics.iteration += 1
            
            # Convergence estimate
            if len(self.metrics.evm_history) > 10:
                recent = list(self.metrics.evm_history)[-10:]
                improvement = recent[0] - recent[-1]
                self.metrics.convergence_pct = min(100, max(0, 
                    (1 - evm_linear / 0.3) * 100))
    
    def handle_events(self):
        """Process queued events from buttons/keyboard."""
        while not self.event_queue.empty():
            event, data = self.event_queue.get()
            
            if event == 'dpd_toggle':
                self.state.dpd_enabled = not self.state.dpd_enabled
                print(f"DPD: {'ENABLED' if self.state.dpd_enabled else 'BYPASS'}")
                
            elif event == 'adapt_toggle':
                self.state.adapt_enabled = not self.state.adapt_enabled
                print(f"Adaptation: {'ON' if self.state.adapt_enabled else 'OFF'}")
                
            elif event == 'temp_cycle':
                states = list(TempState)
                idx = (states.index(self.state.temp_state) + 1) % len(states)
                self.state.temp_state = states[idx]
                self.metrics.iteration = 0  # Reset adaptation
                print(f"Temperature: {self.state.temp_state.name}")
                
            elif event == 'quit':
                self.state.running = False
    
    def create_visualization(self) -> Tuple[plt.Figure, dict]:
        """Create matplotlib figure for demo display."""
        fig = plt.figure(figsize=(14, 8), facecolor='#1a1a2e')
        fig.suptitle('6G PA Digital Predistortion Demo\nLSI Design Contest 29th - Okinawa',
                     fontsize=16, color='white', fontweight='bold')
        
        gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3,
                      left=0.06, right=0.98, top=0.90, bottom=0.08)
        
        axes = {}
        
        # Constellation plots
        axes['const_in'] = fig.add_subplot(gs[0, 0])
        axes['const_out'] = fig.add_subplot(gs[0, 1])
        
        # Spectrum
        axes['spectrum'] = fig.add_subplot(gs[0, 2:])
        
        # Metrics over time
        axes['evm_plot'] = fig.add_subplot(gs[1, :2])
        axes['acpr_plot'] = fig.add_subplot(gs[1, 2:])
        
        # Status panel
        axes['status'] = fig.add_subplot(gs[2, :2])
        axes['controls'] = fig.add_subplot(gs[2, 2:])
        
        # Style all axes
        for ax in axes.values():
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='white', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#4a4a6a')
        
        return fig, axes
    
    def update_visualization(self, frame: int, axes: dict,
                            x_in: np.ndarray, y_out: np.ndarray):
        """Update all plots for current frame."""
        
        # Clear axes
        for name, ax in axes.items():
            if name not in ['status', 'controls']:
                ax.clear()
                ax.set_facecolor('#16213e')
        
        # --- Input Constellation ---
        ax = axes['const_in']
        ax.scatter(x_in.real, x_in.imag, c='#00d4ff', s=8, alpha=0.6)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title('Input Signal', color='white', fontsize=10)
        ax.axhline(0, color='#4a4a6a', linewidth=0.5)
        ax.axvline(0, color='#4a4a6a', linewidth=0.5)
        ax.set_aspect('equal')
        
        # --- Output Constellation ---
        ax = axes['const_out']
        color = '#00ff88' if self.state.dpd_enabled else '#ff4444'
        ax.scatter(y_out.real, y_out.imag, c=color, s=8, alpha=0.6)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        title = 'PA Output (DPD ON)' if self.state.dpd_enabled else 'PA Output (DPD OFF)'
        ax.set_title(title, color='white', fontsize=10)
        ax.axhline(0, color='#4a4a6a', linewidth=0.5)
        ax.axvline(0, color='#4a4a6a', linewidth=0.5)
        ax.set_aspect('equal')
        
        # --- Spectrum ---
        ax = axes['spectrum']
        freq = np.fft.fftfreq(len(y_out), 1/self.sample_rate) / 1e6
        spectrum = 20 * np.log10(np.abs(np.fft.fft(y_out)) + 1e-10)
        spectrum = np.fft.fftshift(spectrum)
        freq = np.fft.fftshift(freq)
        ax.plot(freq, spectrum, color='#00d4ff', linewidth=0.8)
        ax.fill_between(freq, spectrum, -100, alpha=0.3, color='#00d4ff')
        ax.set_xlim(-100, 100)
        ax.set_ylim(-60, 10)
        ax.set_xlabel('Frequency (MHz)', color='white', fontsize=9)
        ax.set_ylabel('Power (dB)', color='white', fontsize=9)
        ax.set_title('Output Spectrum', color='white', fontsize=10)
        ax.tick_params(colors='white')
        
        # --- EVM Plot ---
        ax = axes['evm_plot']
        if len(self.metrics.evm_history) > 1:
            ax.plot(list(self.metrics.evm_history), color='#ff6b6b', linewidth=1.5)
        ax.axhline(-25, color='#00ff88', linestyle='--', linewidth=1, label='Target')
        ax.set_ylim(-35, 0)
        ax.set_xlim(0, 200)
        ax.set_xlabel('Iteration', color='white', fontsize=9)
        ax.set_ylabel('EVM (dB)', color='white', fontsize=9)
        ax.set_title(f'EVM: {self.metrics.evm_db:.1f} dB', color='white', fontsize=10)
        ax.tick_params(colors='white')
        ax.legend(loc='upper right', fontsize=8)
        
        # --- ACPR Plot ---
        ax = axes['acpr_plot']
        if len(self.metrics.acpr_history) > 1:
            ax.plot(list(self.metrics.acpr_history), color='#ffd93d', linewidth=1.5)
        ax.axhline(-45, color='#00ff88', linestyle='--', linewidth=1, label='Target')
        ax.set_ylim(-60, -20)
        ax.set_xlim(0, 200)
        ax.set_xlabel('Iteration', color='white', fontsize=9)
        ax.set_ylabel('ACPR (dBc)', color='white', fontsize=9)
        ax.set_title(f'ACPR: {self.metrics.acpr_db:.1f} dBc', color='white', fontsize=10)
        ax.tick_params(colors='white')
        ax.legend(loc='upper right', fontsize=8)
        
        # --- Status Panel ---
        ax = axes['status']
        ax.clear()
        ax.set_facecolor('#16213e')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Status indicators
        dpd_color = '#00ff88' if self.state.dpd_enabled else '#ff4444'
        adapt_color = '#00ff88' if self.state.adapt_enabled else '#666666'
        temp_colors = {TempState.COLD: '#00d4ff', TempState.NORMAL: '#00ff88', 
                       TempState.HOT: '#ff4444'}
        
        ax.text(0.5, 8.5, 'SYSTEM STATUS', fontsize=12, color='white',
                fontweight='bold', transform=ax.transData)
        
        # DPD Status
        ax.add_patch(Circle((1, 6.5), 0.3, color=dpd_color))
        ax.text(1.8, 6.3, f"DPD: {'ENABLED' if self.state.dpd_enabled else 'BYPASS'}",
                fontsize=10, color='white')
        
        # Adaptation Status  
        ax.add_patch(Circle((1, 5), 0.3, color=adapt_color))
        ax.text(1.8, 4.8, f"Adapt: {'ON' if self.state.adapt_enabled else 'OFF'} (Iter: {self.metrics.iteration})",
                fontsize=10, color='white')
        
        # Temperature Status
        ax.add_patch(Circle((1, 3.5), 0.3, color=temp_colors[self.state.temp_state]))
        ax.text(1.8, 3.3, f"Temp: {self.state.temp_state.name}",
                fontsize=10, color='white')
        
        # Convergence bar
        ax.add_patch(Rectangle((0.5, 1.5), 9, 0.8, fill=False, 
                               edgecolor='white', linewidth=1))
        ax.add_patch(Rectangle((0.5, 1.5), 9 * self.metrics.convergence_pct / 100, 0.8,
                               facecolor='#00ff88', alpha=0.7))
        ax.text(5, 1.1, f'Convergence: {self.metrics.convergence_pct:.0f}%',
                fontsize=9, color='white', ha='center')
        
        # --- Controls Panel ---
        ax = axes['controls']
        ax.clear()
        ax.set_facecolor('#16213e')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        ax.text(0.5, 8.5, 'CONTROLS', fontsize=12, color='white',
                fontweight='bold')
        
        controls_text = """
[D] Toggle DPD Enable/Bypass
[A] Toggle Adaptation On/Off
[T] Cycle Temperature State
[R] Reset Adaptation
[Q] Quit Demo

FPGA Buttons:
  BTN0 = DPD Toggle
  BTN1 = Adapt Toggle
  BTN2 = Temp Cycle
  SW0-1 = Temp Select
"""
        ax.text(0.5, 7, controls_text, fontsize=9, color='#aaaaaa',
                verticalalignment='top', family='monospace')
    
    def run_interactive(self):
        """Run interactive matplotlib demo."""
        print("\nStarting interactive demo...")
        print("Press 'D' to toggle DPD, 'A' for adaptation, 'T' for temperature")
        print("Press 'Q' to quit\n")
        
        fig, axes = self.create_visualization()
        
        # Keyboard handler
        def on_key(event):
            if event.key == 'd':
                self.event_queue.put(('dpd_toggle', None))
            elif event.key == 'a':
                self.event_queue.put(('adapt_toggle', None))
            elif event.key == 't':
                self.event_queue.put(('temp_cycle', None))
            elif event.key == 'r':
                self.metrics.iteration = 0
                self.metrics.evm_history.clear()
                self.metrics.acpr_history.clear()
            elif event.key == 'q':
                self.event_queue.put(('quit', None))
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        def animate(frame):
            if not self.state.running:
                plt.close(fig)
                return
            
            self.handle_events()
            
            # Generate and process signal
            x = self.generate_test_signal()
            x_dpd, y = self.process_signal(x)
            
            # Compute metrics
            self.compute_metrics(x, y)
            
            # Update display
            self.update_visualization(frame, axes, x, y)
            
            return []
        
        ani = animation.FuncAnimation(fig, animate, frames=None,
                                       interval=1000//self.frame_rate,
                                       blit=False, cache_frame_data=False)
        plt.show()
    
    def run_fpga(self):
        """Run FPGA hardware demo."""
        if not PYNQ_AVAILABLE:
            print("PYNQ not available, falling back to software mode")
            self.run_interactive()
            return
        
        print("\nStarting FPGA demo...")
        
        # Use HDMI framebuffer for display
        # This would render matplotlib to HDMI
        
        self.run_interactive()  # For now, use same visualization


def main():
    parser = argparse.ArgumentParser(description='6G PA DPD HDMI Demo')
    parser.add_argument('--mode', choices=['software', 'fpga'],
                        default='software', help='Demo mode')
    parser.add_argument('--config', default='config/config.yaml',
                        help='Config file path')
    parser.add_argument('--checkpoint', default=None,
                        help='Model checkpoint path')
    parser.add_argument('--output', default=None,
                        help='Output video file (optional)')
    args = parser.parse_args()
    
    demo = HDMIDemoApp(
        mode=args.mode,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_video=args.output
    )
    
    if args.mode == 'fpga':
        demo.run_fpga()
    else:
        demo.run_interactive()


if __name__ == '__main__':
    main()
