#!/usr/bin/env python3
"""
Validation Script for 6G PA DPD System

Validates:
1. Trained model performance (EVM, ACPR, NMSE)
2. Quantization impact assessment
3. RTL vs Python model comparison
4. Temperature robustness evaluation
"""

import argparse
import numpy as np
import torch
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal

from models import TDNNGenerator, PADigitalTwin
from utils.quantization import quantize_weights_fixed_point, fake_quantize
from utils.spectral_loss import compute_evm, compute_acpr, compute_nmse
from utils.dataset import DPDDataset, create_memory_features


def load_model(checkpoint_path: str, config: dict, quantized: bool = False):
    """Load trained model."""
    model = TDNNGenerator(
        input_dim=config['model']['generator']['input_dim'],
        hidden_dims=config['model']['generator']['hidden_dims'],
        output_dim=config['model']['generator']['output_dim'],
        quantize=quantized,
        num_bits=config['quantization']['weight_bits']
    )
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['generator_state_dict'])
    
    model.eval()
    return model


def create_test_signal(num_samples: int, signal_type: str = 'ofdm',
                       sample_rate: float = 200e6) -> np.ndarray:
    """Generate test signal for validation."""
    t = np.arange(num_samples) / sample_rate
    
    if signal_type == 'ofdm':
        # 5G NR-like OFDM signal
        num_carriers = 256
        signal = np.zeros(num_samples, dtype=complex)
        
        for k in range(num_carriers):
            freq = (k - num_carriers // 2) * sample_rate / num_carriers * 0.8
            phase = np.random.uniform(0, 2 * np.pi)
            # Random QPSK symbols
            symbol = np.exp(1j * np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2]))
            signal += symbol * np.exp(1j * (2 * np.pi * freq * t + phase))
        
        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.7
        
    elif signal_type == 'two_tone':
        # Two-tone test signal
        f1, f2 = 10e6, 15e6
        signal = 0.5 * (np.exp(1j * 2 * np.pi * f1 * t) + 
                        np.exp(1j * 2 * np.pi * f2 * t))
        
    elif signal_type == 'single_carrier':
        # Single carrier QAM
        symbol_rate = 50e6
        samples_per_symbol = int(sample_rate / symbol_rate)
        num_symbols = num_samples // samples_per_symbol
        
        # Random 16-QAM symbols
        constellation = np.array([-3, -1, 1, 3])
        symbols = (np.random.choice(constellation, num_symbols) + 
                   1j * np.random.choice(constellation, num_symbols))
        symbols = symbols / np.max(np.abs(symbols)) * 0.7
        
        # Upsample
        signal = np.repeat(symbols, samples_per_symbol)[:num_samples]
        
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")
    
    return signal.astype(np.complex64)


def run_dpd_inference(model: TDNNGenerator, pa_model: PADigitalTwin,
                      signal: np.ndarray, memory_depth: int = 5) -> dict:
    """Run DPD inference and return results."""
    
    # Create memory features
    features = create_memory_features(
        torch.tensor(signal.real).float(),
        torch.tensor(signal.imag).float(),
        memory_depth=memory_depth
    )
    
    # Run DPD
    with torch.no_grad():
        dpd_out = model(features)
        dpd_signal = dpd_out[:, 0] + 1j * dpd_out[:, 1]
        
        # Run through PA model
        pa_in = torch.stack([dpd_out[:, 0], dpd_out[:, 1]], dim=-1)
        pa_out = pa_model(pa_in)
        pa_output = pa_out[:, 0] + 1j * pa_out[:, 1]
    
    # Also run signal directly through PA (no DPD)
    direct_in = torch.stack([
        torch.tensor(signal.real).float(),
        torch.tensor(signal.imag).float()
    ], dim=-1)
    
    with torch.no_grad():
        pa_direct = pa_model(direct_in)
        pa_direct_output = pa_direct[:, 0] + 1j * pa_direct[:, 1]
    
    return {
        'input': signal,
        'dpd_output': dpd_signal.numpy(),
        'pa_with_dpd': pa_output.numpy(),
        'pa_without_dpd': pa_direct_output.numpy()
    }


def compute_metrics(results: dict, sample_rate: float = 200e6) -> dict:
    """Compute all performance metrics."""
    
    input_signal = torch.tensor(results['input']).cfloat()
    pa_with_dpd = torch.tensor(results['pa_with_dpd']).cfloat()
    pa_without_dpd = torch.tensor(results['pa_without_dpd']).cfloat()
    
    # Normalize for fair comparison
    scale_dpd = torch.sqrt(torch.mean(torch.abs(input_signal)**2) / 
                           torch.mean(torch.abs(pa_with_dpd)**2))
    scale_no_dpd = torch.sqrt(torch.mean(torch.abs(input_signal)**2) / 
                              torch.mean(torch.abs(pa_without_dpd)**2))
    
    pa_with_dpd_norm = pa_with_dpd * scale_dpd
    pa_without_dpd_norm = pa_without_dpd * scale_no_dpd
    
    # Compute metrics with DPD
    evm_dpd = compute_evm(
        torch.stack([pa_with_dpd_norm.real, pa_with_dpd_norm.imag], dim=-1),
        torch.stack([input_signal.real, input_signal.imag], dim=-1)
    )
    
    acpr_dpd = compute_acpr(pa_with_dpd_norm.unsqueeze(0), sample_rate)
    
    nmse_dpd = compute_nmse(
        torch.stack([pa_with_dpd_norm.real, pa_with_dpd_norm.imag], dim=-1),
        torch.stack([input_signal.real, input_signal.imag], dim=-1)
    )
    
    # Compute metrics without DPD
    evm_no_dpd = compute_evm(
        torch.stack([pa_without_dpd_norm.real, pa_without_dpd_norm.imag], dim=-1),
        torch.stack([input_signal.real, input_signal.imag], dim=-1)
    )
    
    acpr_no_dpd = compute_acpr(pa_without_dpd_norm.unsqueeze(0), sample_rate)
    
    nmse_no_dpd = compute_nmse(
        torch.stack([pa_without_dpd_norm.real, pa_without_dpd_norm.imag], dim=-1),
        torch.stack([input_signal.real, input_signal.imag], dim=-1)
    )
    
    return {
        'with_dpd': {
            'evm_db': 20 * np.log10(evm_dpd.item() + 1e-10),
            'acpr_db': acpr_dpd.item(),
            'nmse_db': 10 * np.log10(nmse_dpd.item() + 1e-10)
        },
        'without_dpd': {
            'evm_db': 20 * np.log10(evm_no_dpd.item() + 1e-10),
            'acpr_db': acpr_no_dpd.item(),
            'nmse_db': 10 * np.log10(nmse_no_dpd.item() + 1e-10)
        }
    }


def validate_quantization(model: TDNNGenerator, config: dict,
                          test_signal: np.ndarray) -> dict:
    """Compare float32 vs quantized model performance."""
    
    # Float model
    float_model = load_model(None, config, quantized=False)
    float_model.load_state_dict(model.state_dict())
    
    # Quantized model
    quant_model = load_model(None, config, quantized=True)
    quant_model.load_state_dict(model.state_dict())
    
    # Create features
    features = create_memory_features(
        torch.tensor(test_signal.real).float(),
        torch.tensor(test_signal.imag).float(),
        memory_depth=config['model']['generator']['memory_depth']
    )
    
    with torch.no_grad():
        float_out = float_model(features)
        quant_out = quant_model(features)
    
    # Compute difference
    diff = float_out - quant_out
    mse = torch.mean(diff ** 2).item()
    max_diff = torch.max(torch.abs(diff)).item()
    
    return {
        'mse': mse,
        'max_diff': max_diff,
        'snr_db': 10 * np.log10(torch.mean(float_out**2).item() / (mse + 1e-10))
    }


def validate_temperature_robustness(model: TDNNGenerator, config: dict,
                                    test_signal: np.ndarray) -> dict:
    """Test model across temperature range."""
    
    results = {}
    temp_range = np.linspace(-40, 85, 10)  # Celsius
    
    for temp in temp_range:
        # Create PA model at temperature
        pa_model = PADigitalTwin(
            memory_depth=config['model']['generator']['memory_depth'],
            num_terms=config['pa']['volterra_terms']
        )
        pa_model.set_temperature(temp)
        
        # Run inference
        run_results = run_dpd_inference(model, pa_model, test_signal,
                                        config['model']['generator']['memory_depth'])
        
        # Compute metrics
        metrics = compute_metrics(run_results, config['system']['sample_rate'])
        results[temp] = metrics['with_dpd']
    
    return results


def plot_results(results: dict, metrics: dict, output_path: Path):
    """Generate validation plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Time domain comparison
    ax = axes[0, 0]
    t = np.arange(min(1000, len(results['input'])))
    ax.plot(t, np.abs(results['input'][:len(t)]), 'b-', label='Input', alpha=0.7)
    ax.plot(t, np.abs(results['pa_with_dpd'][:len(t)]), 'g-', label='PA+DPD', alpha=0.7)
    ax.plot(t, np.abs(results['pa_without_dpd'][:len(t)]), 'r-', label='PA only', alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Magnitude')
    ax.set_title('Time Domain')
    ax.legend()
    ax.grid(True)
    
    # AM/AM characteristic
    ax = axes[0, 1]
    in_mag = np.abs(results['input'])
    out_dpd = np.abs(results['pa_with_dpd'])
    out_no_dpd = np.abs(results['pa_without_dpd'])
    
    ax.scatter(in_mag[::10], out_dpd[::10], s=1, c='g', label='PA+DPD', alpha=0.5)
    ax.scatter(in_mag[::10], out_no_dpd[::10], s=1, c='r', label='PA only', alpha=0.5)
    ax.plot([0, max(in_mag)], [0, max(in_mag)], 'b--', label='Ideal')
    ax.set_xlabel('Input Magnitude')
    ax.set_ylabel('Output Magnitude')
    ax.set_title('AM/AM Characteristic')
    ax.legend()
    ax.grid(True)
    
    # AM/PM characteristic
    ax = axes[0, 2]
    in_phase = np.angle(results['input'])
    out_phase_dpd = np.angle(results['pa_with_dpd'])
    out_phase_no_dpd = np.angle(results['pa_without_dpd'])
    
    phase_diff_dpd = np.unwrap(out_phase_dpd - in_phase) * 180 / np.pi
    phase_diff_no_dpd = np.unwrap(out_phase_no_dpd - in_phase) * 180 / np.pi
    
    ax.scatter(in_mag[::10], phase_diff_dpd[::10], s=1, c='g', label='PA+DPD', alpha=0.5)
    ax.scatter(in_mag[::10], phase_diff_no_dpd[::10], s=1, c='r', label='PA only', alpha=0.5)
    ax.set_xlabel('Input Magnitude')
    ax.set_ylabel('Phase Difference (deg)')
    ax.set_title('AM/PM Characteristic')
    ax.legend()
    ax.grid(True)
    
    # Spectrum
    ax = axes[1, 0]
    nfft = 4096
    f = np.fft.fftfreq(nfft, 1/200e6) / 1e6
    
    spec_in = 20 * np.log10(np.abs(np.fft.fft(results['input'][:nfft], nfft)) + 1e-10)
    spec_dpd = 20 * np.log10(np.abs(np.fft.fft(results['pa_with_dpd'][:nfft], nfft)) + 1e-10)
    spec_no_dpd = 20 * np.log10(np.abs(np.fft.fft(results['pa_without_dpd'][:nfft], nfft)) + 1e-10)
    
    ax.plot(np.fft.fftshift(f), np.fft.fftshift(spec_in), 'b-', label='Input', alpha=0.7)
    ax.plot(np.fft.fftshift(f), np.fft.fftshift(spec_dpd), 'g-', label='PA+DPD', alpha=0.7)
    ax.plot(np.fft.fftshift(f), np.fft.fftshift(spec_no_dpd), 'r-', label='PA only', alpha=0.7)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Power (dB)')
    ax.set_title('Power Spectrum')
    ax.set_ylim([-80, 20])
    ax.legend()
    ax.grid(True)
    
    # Metrics comparison bar chart
    ax = axes[1, 1]
    metrics_names = ['EVM (dB)', 'ACPR (dB)', 'NMSE (dB)']
    dpd_values = [metrics['with_dpd']['evm_db'], 
                  metrics['with_dpd']['acpr_db'],
                  metrics['with_dpd']['nmse_db']]
    no_dpd_values = [metrics['without_dpd']['evm_db'],
                     metrics['without_dpd']['acpr_db'],
                     metrics['without_dpd']['nmse_db']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    ax.bar(x - width/2, dpd_values, width, label='With DPD', color='green')
    ax.bar(x + width/2, no_dpd_values, width, label='Without DPD', color='red')
    ax.set_ylabel('dB')
    ax.set_title('Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, axis='y')
    
    # Improvement summary
    ax = axes[1, 2]
    ax.axis('off')
    
    improvements = {
        'EVM': metrics['without_dpd']['evm_db'] - metrics['with_dpd']['evm_db'],
        'ACPR': metrics['without_dpd']['acpr_db'] - metrics['with_dpd']['acpr_db'],
        'NMSE': metrics['without_dpd']['nmse_db'] - metrics['with_dpd']['nmse_db']
    }
    
    text = "Performance Improvement with DPD:\n"
    text += "=" * 40 + "\n\n"
    for metric, improvement in improvements.items():
        text += f"{metric}: {improvement:+.2f} dB\n"
    
    text += "\n" + "=" * 40 + "\n"
    text += "\nWith DPD:\n"
    text += f"  EVM:  {metrics['with_dpd']['evm_db']:.2f} dB\n"
    text += f"  ACPR: {metrics['with_dpd']['acpr_db']:.2f} dB\n"
    text += f"  NMSE: {metrics['with_dpd']['nmse_db']:.2f} dB\n"
    
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path / 'validation_results.png', dpi=150)
    plt.close()
    
    print(f"Saved plots to {output_path / 'validation_results.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate trained DPD model'
    )
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--config', '-f',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs/validation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10000,
        help='Number of test samples'
    )
    parser.add_argument(
        '--signal-type',
        type=str,
        default='ofdm',
        choices=['ofdm', 'two_tone', 'single_carrier'],
        help='Test signal type'
    )
    parser.add_argument(
        '--validate-quant',
        action='store_true',
        help='Validate quantization impact'
    )
    parser.add_argument(
        '--validate-temp',
        action='store_true',
        help='Validate temperature robustness'
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model = load_model(args.checkpoint, config, quantized=True)
    else:
        print("No checkpoint provided, using random weights")
        model = load_model(None, config, quantized=True)
    
    # Create PA model
    pa_model = PADigitalTwin(
        memory_depth=config['model']['generator']['memory_depth'],
        num_terms=config['pa']['volterra_terms']
    )
    
    # Generate test signal
    print(f"\nGenerating {args.signal_type} test signal ({args.num_samples} samples)...")
    test_signal = create_test_signal(args.num_samples, args.signal_type,
                                     config['system']['sample_rate'])
    
    # Run DPD inference
    print("Running DPD inference...")
    results = run_dpd_inference(model, pa_model, test_signal,
                                config['model']['generator']['memory_depth'])
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(results, config['system']['sample_rate'])
    
    # Print results
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    
    print("\nWithout DPD:")
    print(f"  EVM:  {metrics['without_dpd']['evm_db']:.2f} dB")
    print(f"  ACPR: {metrics['without_dpd']['acpr_db']:.2f} dB")
    print(f"  NMSE: {metrics['without_dpd']['nmse_db']:.2f} dB")
    
    print("\nWith DPD:")
    print(f"  EVM:  {metrics['with_dpd']['evm_db']:.2f} dB")
    print(f"  ACPR: {metrics['with_dpd']['acpr_db']:.2f} dB")
    print(f"  NMSE: {metrics['with_dpd']['nmse_db']:.2f} dB")
    
    print("\nImprovement:")
    print(f"  EVM:  {metrics['without_dpd']['evm_db'] - metrics['with_dpd']['evm_db']:+.2f} dB")
    print(f"  ACPR: {metrics['without_dpd']['acpr_db'] - metrics['with_dpd']['acpr_db']:+.2f} dB")
    print(f"  NMSE: {metrics['without_dpd']['nmse_db'] - metrics['with_dpd']['nmse_db']:+.2f} dB")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_results(results, metrics, output_path)
    
    # Quantization validation
    if args.validate_quant:
        print("\n" + "=" * 60)
        print("Quantization Validation")
        print("=" * 60)
        
        quant_results = validate_quantization(model, config, test_signal)
        print(f"\nFloat vs Quantized:")
        print(f"  MSE: {quant_results['mse']:.6e}")
        print(f"  Max diff: {quant_results['max_diff']:.6f}")
        print(f"  SNR: {quant_results['snr_db']:.2f} dB")
    
    # Temperature validation
    if args.validate_temp:
        print("\n" + "=" * 60)
        print("Temperature Robustness Validation")
        print("=" * 60)
        
        temp_results = validate_temperature_robustness(model, config, test_signal)
        
        print("\nEVM vs Temperature:")
        for temp, metrics in temp_results.items():
            print(f"  {temp:6.1f}°C: EVM = {metrics['evm_db']:.2f} dB")
    
    # Save results
    results_dict = {
        'metrics': metrics,
        'signal_type': args.signal_type,
        'num_samples': args.num_samples
    }
    
    import json
    with open(output_path / 'validation_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
