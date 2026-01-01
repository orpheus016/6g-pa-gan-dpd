#!/usr/bin/env python3
"""
Train TDNN DPD using OpenDPD APA_200MHz dataset
with CWGAN-GP and thermal augmentation

This script:
1. Loads OpenDPD measured PA data (real-world, not simulated)
2. Applies thermal drift models to create cold/normal/hot variants
3. Trains TDNN using CWGAN-GP with spectral loss
4. Exports quantized weights for FPGA deployment

Reference:
- OpenDPD: https://github.com/OpenDPD/OpenDPD
- Tervo et al., "Adversarial Learning for Neural DPD", WAMICON 2019
- Yao et al., "Deep Learning for DPD", IEEE JSAC 2021
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import yaml
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local imports
from models import TDNNGenerator, Discriminator, PADigitalTwin
from utils.spectral_loss import compute_evm, compute_acpr, compute_nmse
from utils.quantization import quantize_weights_fixed_point
from utils.dataset import create_memory_features


def load_opendpd_data(dataset_path: str = 'data/APA_200MHz.mat'):
    """
    Load OpenDPD dataset (measured PA input/output)
    
    The APA_200MHz dataset contains:
    - x: PA input signal (complex baseband)
    - y: PA output signal (complex baseband)
    - Measured from actual GaN PA at 200 MSps
    
    Returns:
        x_in: PA input signal (complex numpy array)
        y_out: PA output signal (complex numpy array)
        sample_rate: 200e6 Hz
    """
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please download from: https://github.com/OpenDPD/OpenDPD")
        print("Or run with --use-synthetic to use synthetic PA model")
        return None, None, None
    
    data = loadmat(dataset_path)
    x_in = data['x'].flatten().astype(np.complex64)
    y_out = data['y'].flatten().astype(np.complex64)
    
    # Normalize to prevent overflow
    x_max = np.max(np.abs(x_in))
    x_in = x_in / x_max * 0.7
    y_out = y_out / x_max * 0.7
    
    print(f"Loaded {len(x_in)} samples from OpenDPD")
    print(f"  Input power:  {10*np.log10(np.mean(np.abs(x_in)**2)):.1f} dBFS")
    print(f"  Output power: {10*np.log10(np.mean(np.abs(y_out)**2)):.1f} dBFS")
    
    return x_in, y_out, 200e6


def generate_synthetic_data(num_samples: int = 100000, sample_rate: float = 200e6):
    """
    Generate synthetic PA data using polynomial model
    Used when OpenDPD dataset is not available
    
    This is for development only - real training should use measured data
    """
    print("Generating synthetic PA data (for testing only)")
    print("For accurate training, use OpenDPD measured data")
    
    # Generate OFDM-like signal
    t = np.arange(num_samples) / sample_rate
    
    # Multi-tone signal to mimic OFDM
    x_in = np.zeros(num_samples, dtype=np.complex64)
    for k in range(64):
        freq = (k - 32) * 3.125e6  # 3.125 MHz subcarrier spacing
        phase = np.random.uniform(0, 2*np.pi)
        x_in += np.exp(1j * (2*np.pi*freq*t + phase))
    
    x_in = x_in / np.max(np.abs(x_in)) * 0.7
    
    # Apply PA model (Rapp model + memory)
    pa = PADigitalTwin(memory_depth=5, order=7)
    y_out = pa(torch.tensor(np.stack([x_in.real, x_in.imag], axis=1))).detach().numpy()
    y_out = y_out[:, 0] + 1j * y_out[:, 1]
    
    return x_in, y_out, sample_rate


def apply_thermal_drift(y_pa: np.ndarray, temperature: float, reference_temp: float = 25.0):
    """
    Model PA thermal drift effects on GaN PA
    
    Physical basis (from GaN PA datasheets and literature):
    - Gain drift: ~0.5% per 10°C (due to mobility change)
    - Phase drift: ~0.3° per 10°C (due to capacitance change)
    - AM/AM compression point shifts
    
    Reference: Cripps, "RF Power Amplifiers for Wireless Communications"
    
    Args:
        y_pa: PA output signal (complex)
        temperature: PA junction temperature in °C
        reference_temp: Reference temperature (usually 25°C)
    
    Returns:
        y_thermal: Thermally-drifted PA output
    """
    dT = temperature - reference_temp
    
    # Gain drift: negative tempco for GaN (gain drops at high temp)
    alpha_gain = -0.005  # -0.5% per 10°C
    gain_factor = 1 + alpha_gain * (dT / 10)
    
    # Phase drift: capacitance changes with temperature
    alpha_phase = 0.003  # ~0.3° per 10°C in radians
    phase_shift = alpha_phase * (dT / 10)
    
    # AM/AM compression changes (more compression at high temp)
    # This affects the nonlinearity, modeled as envelope-dependent phase
    env = np.abs(y_pa)
    alpha_amam = 0.01 * (dT / 50)  # Extra compression at hot
    compression = 1 - alpha_amam * env**2
    
    # Apply all effects
    y_thermal = y_pa * gain_factor * compression * np.exp(1j * phase_shift)
    
    return y_thermal


def prepare_dpd_training_data(x_in: np.ndarray, y_pa: np.ndarray, memory_depth: int = 5):
    """
    Prepare data for DPD training using Indirect Learning Architecture (ILA)
    
    In ILA:
    - Train predistorter as post-inverse: DPD(y) ≈ x
    - Then use same function as pre-inverse
    
    This is standard practice in DPD training.
    Reference: Eun & Powers, "A New Volterra Predistorter", IEEE TCOM
    """
    # Create memory features from PA output (for ILA)
    features = create_memory_features(
        torch.tensor(y_pa.real).float(),
        torch.tensor(y_pa.imag).float(),
        memory_depth=memory_depth
    )
    
    # Target is the original input (what we want DPD to produce)
    targets = torch.stack([
        torch.tensor(x_in.real).float(),
        torch.tensor(x_in.imag).float()
    ], dim=1)
    
    # Trim to account for memory depth
    valid_samples = len(targets) - memory_depth
    features = features[:valid_samples]
    targets = targets[memory_depth:]
    
    return features, targets


def gradient_penalty(discriminator, real_samples, fake_samples, device):
    """
    Compute gradient penalty for WGAN-GP
    
    Forces discriminator to be 1-Lipschitz continuous
    Reference: Gulrajani et al., "Improved Training of WGANs", NeurIPS 2017
    """
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    d_interpolates = discriminator(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradient_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
    gp = ((gradient_norm - 1) ** 2).mean()
    
    return gp


def train_single_temperature(
    x_in: np.ndarray,
    y_pa: np.ndarray,
    temperature: float,
    config: dict,
    device: torch.device,
    output_dir: Path
):
    """
    Train TDNN for a single temperature condition
    """
    temp_name = {-20: 'cold', 25: 'normal', 70: 'hot'}.get(temperature, f't{temperature}')
    print(f"\n{'='*60}")
    print(f"Training for {temp_name} temperature ({temperature}°C)")
    print(f"{'='*60}")
    
    # Apply thermal drift to PA output
    y_thermal = apply_thermal_drift(y_pa, temperature)
    
    # Prepare training data
    memory_depth = config['model']['generator'].get('memory_depth', 5)
    features, targets = prepare_dpd_training_data(x_in, y_thermal, memory_depth)
    
    # Create data loader
    dataset = TensorDataset(features, targets)
    loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    # Initialize models
    gen_cfg = config['model']['generator']
    generator = TDNNGenerator(
        input_dim=gen_cfg['input_dim'],
        hidden_dims=gen_cfg['hidden_dims'],
        output_dim=gen_cfg['output_dim'],
        quantize=config['quantization']['enabled'],
        num_bits=config['quantization']['weight_bits']
    ).to(device)
    
    discriminator = Discriminator(
        input_dim=gen_cfg['output_dim'],
        hidden_dims=config['model']['discriminator']['hidden_dims']
    ).to(device)
    
    # Optimizers (WGAN-GP uses different learning rates)
    opt_g = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_d = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    
    # Loss weights
    lambda_mse = config['training']['loss_weights'].get('mse', 1.0)
    lambda_evm = config['training']['loss_weights'].get('evm', 0.1)
    lambda_acpr = config['training']['loss_weights'].get('acpr', 0.1)
    lambda_gp = config['training']['loss_weights'].get('gradient_penalty', 10.0)
    n_critic = config['training'].get('n_critic', 5)
    
    # Training loop
    num_epochs = config['training']['epochs']
    best_evm = float('inf')
    
    history = {'g_loss': [], 'd_loss': [], 'evm': [], 'acpr': []}
    
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        num_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (x_batch, y_batch) in enumerate(pbar):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # ========== Train Discriminator ==========
            for _ in range(n_critic):
                opt_d.zero_grad()
                
                # Generate fake samples
                with torch.no_grad():
                    fake = generator(x_batch)
                
                # Discriminator loss (WGAN)
                d_real = discriminator(y_batch)
                d_fake = discriminator(fake)
                
                # Gradient penalty
                gp = gradient_penalty(discriminator, y_batch, fake, device)
                
                d_loss = d_fake.mean() - d_real.mean() + lambda_gp * gp
                d_loss.backward()
                opt_d.step()
            
            # ========== Train Generator ==========
            opt_g.zero_grad()
            
            fake = generator(x_batch)
            
            # Adversarial loss
            g_adv = -discriminator(fake).mean()
            
            # Reconstruction loss (MSE)
            g_mse = nn.functional.mse_loss(fake, y_batch)
            
            # Spectral losses (computed on batch)
            g_evm = compute_evm(fake, y_batch)
            g_acpr = compute_acpr(fake)
            
            # Combined loss
            g_loss = g_adv + lambda_mse * g_mse + lambda_evm * g_evm + lambda_acpr * g_acpr
            
            g_loss.backward()
            opt_g.step()
            
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'G': f'{g_loss.item():.4f}',
                'D': f'{d_loss.item():.4f}',
                'EVM': f'{g_evm.item():.2f}%'
            })
        
        # Epoch metrics
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        
        # Validation
        generator.eval()
        with torch.no_grad():
            val_features = features[:10000].to(device)
            val_targets = targets[:10000].to(device)
            val_pred = generator(val_features)
            val_evm = compute_evm(val_pred, val_targets).item()
            val_acpr = compute_acpr(val_pred).item()
        
        history['g_loss'].append(avg_g_loss)
        history['d_loss'].append(avg_d_loss)
        history['evm'].append(val_evm)
        history['acpr'].append(val_acpr)
        
        print(f"Epoch {epoch+1}: G_loss={avg_g_loss:.4f}, D_loss={avg_d_loss:.4f}, "
              f"EVM={val_evm:.2f}%, ACPR={val_acpr:.1f}dB")
        
        # Save best model
        if val_evm < best_evm:
            best_evm = val_evm
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'evm': val_evm,
                'acpr': val_acpr,
                'temperature': temperature
            }, output_dir / f'checkpoint_{temp_name}_best.pt')
    
    # Export weights for FPGA
    export_weights_fpga(generator, output_dir / f'weights_{temp_name}.hex', config)
    
    return generator, history


def export_weights_fpga(model: nn.Module, output_path: Path, config: dict):
    """
    Export quantized weights in hex format for FPGA
    
    Format: Q1.15 (1 sign bit, 15 fractional bits)
    Range: [-1.0, +0.999969]
    """
    num_bits = config['quantization']['weight_bits']
    scale = 2 ** (num_bits - 1)
    
    with open(output_path, 'w') as f:
        f.write(f"// TDNN weights - Q1.{num_bits-1} format\n")
        f.write(f"// Generated by train_opendpd.py\n\n")
        
        addr = 0
        for name, param in model.named_parameters():
            if 'weight' in name or 'bias' in name:
                f.write(f"// {name}: shape={list(param.shape)}\n")
                
                # Quantize to fixed-point
                w = param.detach().cpu().numpy().flatten()
                w_clipped = np.clip(w, -1.0, 1.0 - 1/scale)
                w_int = np.round(w_clipped * scale).astype(np.int16)
                
                for val in w_int:
                    # Convert to unsigned for hex representation
                    val_unsigned = val & 0xFFFF
                    f.write(f"{val_unsigned:04X}  // addr={addr}\n")
                    addr += 1
        
        f.write(f"\n// Total: {addr} weights\n")
    
    print(f"Exported {addr} weights to {output_path}")


def compare_gan_vs_supervised(
    x_in: np.ndarray,
    y_pa: np.ndarray,
    config: dict,
    device: torch.device
):
    """
    Compare GAN-trained vs supervised-only TDNN
    
    This demonstrates the value proposition of GAN training:
    - Supervised (MSE only): optimizes sample error
    - GAN (spectral loss): optimizes ACPR/EVM directly
    
    Expected improvement: 2-3 dB ACPR
    """
    print("\n" + "="*60)
    print("COMPARISON: GAN vs Supervised Training")
    print("="*60)
    
    memory_depth = config['model']['generator'].get('memory_depth', 5)
    features, targets = prepare_dpd_training_data(x_in, y_pa, memory_depth)
    
    dataset = TensorDataset(features, targets)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Train supervised-only model
    print("\n[1/2] Training supervised-only (MSE loss)...")
    gen_sup = TDNNGenerator(
        input_dim=config['model']['generator']['input_dim'],
        hidden_dims=config['model']['generator']['hidden_dims'],
        output_dim=config['model']['generator']['output_dim']
    ).to(device)
    
    opt_sup = optim.Adam(gen_sup.parameters(), lr=1e-3)
    
    for epoch in range(50):
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            opt_sup.zero_grad()
            pred = gen_sup(x_batch)
            loss = nn.functional.mse_loss(pred, y_batch)
            loss.backward()
            opt_sup.step()
    
    # Evaluate supervised
    gen_sup.eval()
    with torch.no_grad():
        pred_sup = gen_sup(features[:10000].to(device))
        evm_sup = compute_evm(pred_sup, targets[:10000].to(device)).item()
        acpr_sup = compute_acpr(pred_sup).item()
    
    print(f"  Supervised: EVM={evm_sup:.2f}%, ACPR={acpr_sup:.1f}dB")
    
    # Train GAN model (abbreviated)
    print("\n[2/2] Training GAN (spectral loss)...")
    gen_gan = TDNNGenerator(
        input_dim=config['model']['generator']['input_dim'],
        hidden_dims=config['model']['generator']['hidden_dims'],
        output_dim=config['model']['generator']['output_dim']
    ).to(device)
    
    discriminator = Discriminator(
        input_dim=2,
        hidden_dims=config['model']['discriminator']['hidden_dims']
    ).to(device)
    
    opt_g = optim.Adam(gen_gan.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_d = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    
    for epoch in range(50):
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Train D
            opt_d.zero_grad()
            fake = gen_gan(x_batch).detach()
            d_loss = discriminator(fake).mean() - discriminator(y_batch).mean()
            gp = gradient_penalty(discriminator, y_batch, fake, device)
            (d_loss + 10 * gp).backward()
            opt_d.step()
            
            # Train G
            opt_g.zero_grad()
            fake = gen_gan(x_batch)
            g_loss = -discriminator(fake).mean()
            g_loss += nn.functional.mse_loss(fake, y_batch)
            g_loss += 0.1 * compute_acpr(fake)
            g_loss.backward()
            opt_g.step()
    
    # Evaluate GAN
    gen_gan.eval()
    with torch.no_grad():
        pred_gan = gen_gan(features[:10000].to(device))
        evm_gan = compute_evm(pred_gan, targets[:10000].to(device)).item()
        acpr_gan = compute_acpr(pred_gan).item()
    
    print(f"  GAN:        EVM={evm_gan:.2f}%, ACPR={acpr_gan:.1f}dB")
    
    # Summary
    print("\n" + "-"*40)
    print("IMPROVEMENT FROM GAN TRAINING:")
    print(f"  EVM:  {evm_sup:.2f}% → {evm_gan:.2f}% ({evm_sup-evm_gan:+.2f}%)")
    print(f"  ACPR: {acpr_sup:.1f}dB → {acpr_gan:.1f}dB ({acpr_gan-acpr_sup:+.1f}dB)")
    print("-"*40)
    
    return {
        'supervised': {'evm': evm_sup, 'acpr': acpr_sup},
        'gan': {'evm': evm_gan, 'acpr': acpr_gan}
    }


def main():
    parser = argparse.ArgumentParser(description='Train TDNN DPD with CWGAN-GP')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data', type=str, default='data/APA_200MHz.mat',
                        help='Path to OpenDPD dataset')
    parser.add_argument('--output', type=str, default='checkpoints',
                        help='Output directory')
    parser.add_argument('--use-synthetic', action='store_true',
                        help='Use synthetic PA data (for testing without OpenDPD)')
    parser.add_argument('--compare', action='store_true',
                        help='Run GAN vs supervised comparison')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    if args.use_synthetic:
        x_in, y_pa, fs = generate_synthetic_data()
    else:
        x_in, y_pa, fs = load_opendpd_data(args.data)
        if x_in is None:
            print("Falling back to synthetic data")
            x_in, y_pa, fs = generate_synthetic_data()
    
    # Optional: Compare GAN vs supervised
    if args.compare:
        results = compare_gan_vs_supervised(x_in, y_pa, config, device)
    
    # Train for each temperature
    temperatures = [-20, 25, 70]  # Cold, Normal, Hot
    
    for temp in temperatures:
        train_single_temperature(
            x_in, y_pa, temp, config, device, output_dir
        )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Weights exported to: {output_dir}/weights_*.hex")
    print("Copy these to rtl/weights/ for FPGA synthesis")


if __name__ == '__main__':
    main()
