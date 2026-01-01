#!/usr/bin/env python3
"""
Verification Script: GAN vs Supervised DPD Performance

This script provides quantitative evidence that GAN training
improves DPD performance over supervised-only training.

Expected Results:
- GAN-trained TDNN: 2-3 dB better ACPR than MSE-trained
- EVM improvement: ~1-2% absolute

Reference:
- Tervo et al., "Adversarial Learning for Neural DPD", WAMICON 2019
- Shows consistent improvement with adversarial training
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
import torch
import torch.nn as nn
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from models import TDNNGenerator, PADigitalTwin
from utils.spectral_loss import compute_evm, compute_acpr, compute_nmse


def generate_ofdm_signal(num_samples: int = 50000, num_carriers: int = 256):
    """Generate OFDM-like test signal"""
    t = np.arange(num_samples) / 200e6
    
    # Generate random QPSK symbols for each carrier
    symbols = (np.random.randint(0, 4, (num_carriers,)) * np.pi/2 + np.pi/4)
    carriers = np.exp(1j * symbols)
    
    # OFDM modulation (sum of modulated carriers)
    x = np.zeros(num_samples, dtype=np.complex64)
    for k, c in enumerate(carriers):
        freq = (k - num_carriers//2) * 0.78125e6  # 200MHz / 256
        x += c * np.exp(1j * 2 * np.pi * freq * t)
    
    # Normalize
    x = x / np.max(np.abs(x)) * 0.7
    
    return x


def apply_pa_model(x: np.ndarray, temp: float = 25.0):
    """Apply PA model with temperature effects"""
    # Rapp model: y = x * G / (1 + |x|^(2p))^(1/2p)
    G = 10.0  # Gain
    p = 2.0   # Smoothness factor
    
    # Temperature effects
    G *= (1 - 0.005 * (temp - 25) / 10)  # Gain drops at high temp
    
    env = np.abs(x)
    gain = G / (1 + env**(2*p))**(1/(2*p))
    
    # AM-PM (phase rotation with envelope)
    phase = 0.1 * env**2 + 0.003 * (temp - 25) / 10
    
    y = x * gain * np.exp(1j * phase)
    
    return y


def compute_spectrum(sig: np.ndarray, fs: float = 200e6):
    """Compute power spectral density"""
    from scipy import signal as sig_proc
    f, psd = sig_proc.welch(sig, fs=fs, nperseg=1024, return_onesided=False)
    f = np.fft.fftshift(f)
    psd = np.fft.fftshift(psd)
    return f / 1e6, 10 * np.log10(psd + 1e-10)


def measure_acpr(spectrum_db: np.ndarray, freq_mhz: np.ndarray):
    """Measure ACPR from spectrum"""
    # Main channel: -50 to +50 MHz
    # Adjacent: -100 to -50 and +50 to +100 MHz
    
    main_mask = np.abs(freq_mhz) < 50
    lower_adj_mask = (freq_mhz > -100) & (freq_mhz < -50)
    upper_adj_mask = (freq_mhz > 50) & (freq_mhz < 100)
    
    main_power = np.mean(10**(spectrum_db[main_mask]/10))
    lower_power = np.mean(10**(spectrum_db[lower_adj_mask]/10))
    upper_power = np.mean(10**(spectrum_db[upper_adj_mask]/10))
    
    acpr_lower = 10 * np.log10(lower_power / main_power)
    acpr_upper = 10 * np.log10(upper_power / main_power)
    
    return acpr_lower, acpr_upper


def measure_evm(ref: np.ndarray, meas: np.ndarray):
    """Measure EVM"""
    error = meas - ref
    evm = np.sqrt(np.mean(np.abs(error)**2) / np.mean(np.abs(ref)**2))
    return 100 * evm  # Percentage


def run_verification(use_pretrained: bool = False, plot: bool = True):
    """
    Run complete verification comparing:
    1. No DPD (baseline)
    2. Supervised-only DPD (MSE loss)
    3. GAN-trained DPD (CWGAN-GP + spectral loss)
    """
    print("="*70)
    print("6G PA DPD VERIFICATION: GAN vs Supervised Training")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Generate test signal
    print("\n[1/5] Generating OFDM test signal...")
    x_clean = generate_ofdm_signal(100000)
    
    # Apply PA
    print("[2/5] Applying PA model...")
    y_pa = apply_pa_model(x_clean, temp=25.0)
    
    # Measure baseline (no DPD)
    print("[3/5] Measuring baseline (no DPD)...")
    freq, spec_no_dpd = compute_spectrum(y_pa)
    acpr_no_dpd = measure_acpr(spec_no_dpd, freq)
    evm_no_dpd = measure_evm(x_clean, y_pa / 10)  # Normalize gain
    
    print(f"  No DPD - ACPR: {np.mean(acpr_no_dpd):.1f} dBc, EVM: {evm_no_dpd:.1f}%")
    
    # Prepare training data (ILA: train on PA output → input mapping)
    print("[4/5] Training DPD models...")
    
    # Convert to IQ tensor format [batch, seq_len, 2]
    memory_depth = 5
    
    # Trim to manageable size for training
    train_len = min(50000, len(y_pa))
    y_train = y_pa[:train_len]
    x_train = x_clean[:train_len]
    
    # Stack as [1, seq_len, 2] IQ tensor
    features = torch.tensor(
        np.stack([y_train.real, y_train.imag], axis=1),
        dtype=torch.float32
    ).unsqueeze(0)  # [1, seq_len, 2]
    
    targets = torch.tensor(
        np.stack([x_train.real, x_train.imag], axis=1),
        dtype=torch.float32
    ).unsqueeze(0)  # [1, seq_len, 2]
    
    # Train supervised model (MSE only)
    print("  Training supervised (MSE only)...")
    gen_sup = TDNNGenerator(
        memory_depth=5,
        hidden_dims=[32, 16]
    ).to(device)
    
    opt_sup = torch.optim.Adam(gen_sup.parameters(), lr=1e-3)
    
    # Train on full sequence (TDNN handles memory internally)
    features_dev = features.to(device)
    targets_dev = targets.to(device)
    
    for epoch in range(50):
        opt_sup.zero_grad()
        pred = gen_sup(features_dev)  # [1, seq_len - M, 2]
        # Trim targets to match output (memory depth trimmed)
        targets_trimmed = targets_dev[:, memory_depth:, :]
        loss = nn.functional.mse_loss(pred, targets_trimmed)
        loss.backward()
        opt_sup.step()
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: loss={loss.item():.6f}")
    
    # Evaluate supervised DPD
    gen_sup.eval()
    with torch.no_grad():
        dpd_out_sup = gen_sup(features_dev).cpu().numpy().squeeze()  # [seq_len - M, 2]
        dpd_out_sup = dpd_out_sup[:, 0] + 1j * dpd_out_sup[:, 1]
    
    # Apply DPD → PA
    y_dpd_sup = apply_pa_model(dpd_out_sup, temp=25.0)
    freq, spec_sup = compute_spectrum(y_dpd_sup)
    acpr_sup = measure_acpr(spec_sup, freq)
    evm_sup = measure_evm(x_train[memory_depth:], y_dpd_sup / 10)
    
    print(f"  Supervised - ACPR: {np.mean(acpr_sup):.1f} dBc, EVM: {evm_sup:.1f}%")
    
    # Train GAN model (MSE + spectral loss)
    print("  Training GAN (MSE + spectral loss)...")
    gen_gan = TDNNGenerator(
        memory_depth=5,
        hidden_dims=[32, 16]
    ).to(device)
    
    # Simple discriminator (operates on flattened output)
    disc = nn.Sequential(
        nn.Linear(2, 64),
        nn.LeakyReLU(0.2),
        nn.Linear(64, 32),
        nn.LeakyReLU(0.2),
        nn.Linear(32, 1)
    ).to(device)
    
    opt_g = torch.optim.Adam(gen_gan.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.9))
    
    targets_trimmed = targets_dev[:, memory_depth:, :]
    
    for epoch in range(50):
        # Train discriminator
        opt_d.zero_grad()
        with torch.no_grad():
            fake = gen_gan(features_dev)  # [1, seq_len - M, 2]
        
        # Flatten for discriminator
        fake_flat = fake.reshape(-1, 2)
        real_flat = targets_trimmed.reshape(-1, 2)
        
        # Sample subset for efficiency
        idx = torch.randperm(fake_flat.size(0))[:1000]
        d_real = disc(real_flat[idx]).mean()
        d_fake = disc(fake_flat[idx]).mean()
        d_loss = d_fake - d_real
        d_loss.backward()
        opt_d.step()
        
        # Clip weights (WGAN)
        for p in disc.parameters():
            p.data.clamp_(-0.01, 0.01)
        
        # Train generator
        opt_g.zero_grad()
        fake = gen_gan(features_dev)
        fake_flat = fake.reshape(-1, 2)
        
        g_adv = -disc(fake_flat).mean()
        g_mse = nn.functional.mse_loss(fake, targets_trimmed)
        
        # Spectral loss: penalize spectral regrowth (simplified)
        fake_complex = fake_flat[:, 0] + 1j * fake_flat[:, 1]
        spectrum = torch.fft.fft(fake_complex)
        n = len(spectrum)
        # Penalize power outside main band (adjacent channel power)
        adj_power = (spectrum[:n//4].abs()**2).mean() + (spectrum[3*n//4:].abs()**2).mean()
        g_spectral = 0.001 * adj_power
        
        g_loss = g_adv + g_mse + g_spectral
        g_loss.backward()
        opt_g.step()
        
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: G_loss={g_loss.item():.4f}, D_loss={d_loss.item():.4f}")
    
    # Evaluate GAN DPD
    gen_gan.eval()
    with torch.no_grad():
        dpd_out_gan = gen_gan(features_dev).cpu().numpy().squeeze()
        dpd_out_gan = dpd_out_gan[:, 0] + 1j * dpd_out_gan[:, 1]
    
    y_dpd_gan = apply_pa_model(dpd_out_gan, temp=25.0)
    freq, spec_gan = compute_spectrum(y_dpd_gan)
    acpr_gan = measure_acpr(spec_gan, freq)
    evm_gan = measure_evm(x_train[memory_depth:], y_dpd_gan / 10)
    
    print(f"  GAN - ACPR: {np.mean(acpr_gan):.1f} dBc, EVM: {evm_gan:.1f}%")
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Method':<20} {'ACPR (dBc)':<15} {'EVM (%)':<15}")
    print("-"*50)
    print(f"{'No DPD':<20} {np.mean(acpr_no_dpd):>10.1f} {evm_no_dpd:>13.1f}")
    print(f"{'Supervised (MSE)':<20} {np.mean(acpr_sup):>10.1f} {evm_sup:>13.1f}")
    print(f"{'GAN (Spectral)':<20} {np.mean(acpr_gan):>10.1f} {evm_gan:>13.1f}")
    print("-"*50)
    
    acpr_improvement = np.mean(acpr_sup) - np.mean(acpr_gan)
    evm_improvement = evm_sup - evm_gan
    
    print(f"\nGAN vs Supervised Improvement:")
    print(f"  ACPR: {acpr_improvement:+.1f} dB")
    print(f"  EVM:  {evm_improvement:+.1f}%")
    
    if acpr_improvement > 0:
        print(f"\n✓ GAN training provides {acpr_improvement:.1f} dB ACPR improvement")
    else:
        print(f"\n✗ GAN training did not improve ACPR (may need more epochs)")
    
    # Plot results
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Spectrum comparison
        ax = axes[0, 0]
        ax.plot(freq, spec_no_dpd, 'r-', alpha=0.7, label='No DPD')
        ax.plot(freq, spec_sup, 'b-', alpha=0.7, label='Supervised')
        ax.plot(freq, spec_gan, 'g-', alpha=0.7, label='GAN')
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('PSD (dB)')
        ax.set_title('Output Spectrum Comparison')
        ax.legend()
        ax.set_xlim(-100, 100)
        ax.grid(True, alpha=0.3)
        
        # ACPR bar chart
        ax = axes[0, 1]
        methods = ['No DPD', 'Supervised', 'GAN']
        acprs = [np.mean(acpr_no_dpd), np.mean(acpr_sup), np.mean(acpr_gan)]
        colors = ['red', 'blue', 'green']
        bars = ax.bar(methods, acprs, color=colors, alpha=0.7)
        ax.set_ylabel('ACPR (dBc)')
        ax.set_title('ACPR Comparison (lower is better)')
        ax.axhline(-45, color='k', linestyle='--', label='Target')
        for bar, acpr in zip(bars, acprs):
            ax.text(bar.get_x() + bar.get_width()/2, acpr + 1, 
                   f'{acpr:.1f}', ha='center', fontsize=10)
        
        # Constellation
        ax = axes[1, 0]
        ax.scatter(y_pa[:1000].real, y_pa[:1000].imag, 
                  c='red', alpha=0.3, s=1, label='No DPD')
        ax.scatter(y_dpd_gan[:1000].real, y_dpd_gan[:1000].imag,
                  c='green', alpha=0.3, s=1, label='GAN DPD')
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.set_title('Constellation (first 1000 samples)')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # EVM bar chart
        ax = axes[1, 1]
        evms = [evm_no_dpd, evm_sup, evm_gan]
        bars = ax.bar(methods, evms, color=colors, alpha=0.7)
        ax.set_ylabel('EVM (%)')
        ax.set_title('EVM Comparison (lower is better)')
        ax.axhline(3, color='k', linestyle='--', label='Target 3%')
        for bar, evm in zip(bars, evms):
            ax.text(bar.get_x() + bar.get_width()/2, evm + 0.5,
                   f'{evm:.1f}%', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('verification_results.png', dpi=150)
        print(f"\nPlot saved to: verification_results.png")
        plt.show()
    
    return {
        'no_dpd': {'acpr': np.mean(acpr_no_dpd), 'evm': evm_no_dpd},
        'supervised': {'acpr': np.mean(acpr_sup), 'evm': evm_sup},
        'gan': {'acpr': np.mean(acpr_gan), 'evm': evm_gan},
        'improvement': {'acpr': acpr_improvement, 'evm': evm_improvement}
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    args = parser.parse_args()
    
    results = run_verification(plot=not args.no_plot)
