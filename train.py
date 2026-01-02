#!/usr/bin/env python3
# =============================================================================
# 6G PA GAN-DPD: CWGAN-GP Training Script
# =============================================================================
"""
CWGAN-GP Training for Memory-Aware TDNN DPD
============================================

This script trains the TDNN generator using CWGAN-GP with:
- Wasserstein loss with gradient penalty
- Spectral loss (EVM, ACPR)
- L1 reconstruction loss
- Quantization-Aware Training (QAT)

Usage:
    python train.py --config config/config.yaml
    python train.py --config config/config.yaml --temp all --epochs 500
    python train.py --config config/config.yaml --qat --resume checkpoints/latest.pth
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import TDNNGenerator, TDNNGeneratorQAT, Discriminator, PADigitalTwin
from models.discriminator import WassersteinLoss
from utils.spectral_loss import SpectralLoss, compute_evm, compute_acpr
from utils.dataset import DPDDataset, SyntheticDPDDataset
from utils.quantization import QuantizationConfig


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_measured_data(data_dir: Path, split: str = 'train'):
    """Load measured PA input/output data from CSV files.
    
    Args:
        data_dir: Directory containing train_input.csv, train_output.csv, etc.
        split: 'train', 'val', or 'test'
    
    Returns:
        u_pa: PA input signal (clean, what we want DPD to produce)
        y_pa: PA output signal (distorted, input to DPD)
    """
    input_file = data_dir / f'{split}_input.csv'
    output_file = data_dir / f'{split}_output.csv'
    
    if not input_file.exists() or not output_file.exists():
        raise FileNotFoundError(f"Data files not found in {data_dir}")
    
    # Load CSV files
    input_df = pd.read_csv(input_file)
    output_df = pd.read_csv(output_file)
    
    # Convert to complex arrays
    u_pa = (input_df['I'].values + 1j * input_df['Q'].values).astype(np.complex64)
    y_pa = (output_df['I'].values + 1j * output_df['Q'].values).astype(np.complex64)
    
    # Normalize
    max_val = np.max(np.abs(u_pa))
    u_pa = u_pa / max_val * 0.7
    y_pa = y_pa / max_val * 0.7
    
    print(f"Loaded {len(u_pa):,} {split} samples")
    print(f"  PA input power:  {10*np.log10(np.mean(np.abs(u_pa)**2)):.2f} dBFS")
    print(f"  PA output power: {10*np.log10(np.mean(np.abs(y_pa)**2)):.2f} dBFS")
    
    return u_pa, y_pa


def apply_thermal_drift(y_pa: np.ndarray, temperature: float, reference_temp: float = 25.0):
    """Apply thermal drift to PA output signal.
    
    Physical basis (GaN PA):
    - Gain drift: ~0.5% per 10°C
    - Phase drift: ~0.3° per 10°C
    - AM/AM compression changes
    
    Args:
        y_pa: PA output signal
        temperature: Temperature in °C
        reference_temp: Reference temperature (default 25°C)
    
    Returns:
        y_thermal: Thermally-drifted PA output
    """
    dT = temperature - reference_temp
    
    # Gain drift (negative tempco for GaN)
    alpha_gain = -0.005  # -0.5% per 10°C
    gain_factor = 1 + alpha_gain * (dT / 10)
    
    # Phase drift
    alpha_phase = 0.003  # ~0.3° per 10°C in radians
    phase_shift = alpha_phase * (dT / 10)
    
    # AM/AM compression (more at high temp)
    env = np.abs(y_pa)
    alpha_amam = 0.01 * (dT / 50)
    compression = 1 - alpha_amam * env**2
    
    # Apply all effects
    y_thermal = y_pa * gain_factor * compression * np.exp(1j * phase_shift)
    
    return y_thermal


def create_dpd_dataset(u_pa: np.ndarray, y_pa: np.ndarray, memory_depth: int = 5, 
                       seq_length: int = 256) -> TensorDataset:
    """Create dataset for DPD training using Indirect Learning Architecture.
    
    ILA: Train DPD as post-inverse: DPD(y_PA) ≈ u_PA
    Then use same function as pre-inverse in deployment.
    
    Args:
        u_pa: PA input (target for DPD)
        y_pa: PA output (input to DPD)
        memory_depth: Number of memory taps
        seq_length: Sequence length for batching
    
    Returns:
        TensorDataset with (input_features, target_signal) pairs
    """
    # Create memory features from PA output
    num_samples = len(y_pa) - memory_depth
    num_features = 2 * (memory_depth + 1)  # I/Q for each tap
    
    inputs = np.zeros((num_samples, memory_depth + 1, 2), dtype=np.float32)
    targets = np.zeros((num_samples, 2), dtype=np.float32)
    
    for i in range(num_samples):
        # Memory taps from PA output
        for m in range(memory_depth + 1):
            inputs[i, m, 0] = y_pa[i + memory_depth - m].real
            inputs[i, m, 1] = y_pa[i + memory_depth - m].imag
        
        # Target is PA input (what we want to reconstruct)
        targets[i, 0] = u_pa[i + memory_depth].real
        targets[i, 1] = u_pa[i + memory_depth].imag
    
    # Convert to torch tensors
    inputs_t = torch.from_numpy(inputs)
    targets_t = torch.from_numpy(targets)
    
    return TensorDataset(inputs_t, targets_t)


def create_models(config: dict, device: torch.device, qat: bool = False):
    """Create generator and discriminator models."""
    gen_config = config['model']['generator']
    disc_config = config['model']['discriminator']
    
    # Generator
    if qat:
        quant_config = config.get('quantization', {})
        generator = TDNNGeneratorQAT(
            memory_depth=gen_config.get('memory_depth', 5),
            hidden_dims=gen_config.get('hidden_dims', [32, 16]),
            leaky_slope=gen_config.get('leaky_slope', 0.2),
            weight_bits=quant_config.get('weight', {}).get('bits', 16),
            activation_bits=quant_config.get('activation', {}).get('bits', 16)
        )
    else:
        generator = TDNNGenerator(
            memory_depth=gen_config.get('memory_depth', 5),
            hidden_dims=gen_config.get('hidden_dims', [32, 16]),
            leaky_slope=gen_config.get('leaky_slope', 0.2)
        )
        
    # Discriminator
    discriminator = Discriminator(
        input_dim=disc_config.get('input_dim', 4),
        hidden_dims=disc_config.get('hidden_dims', [64, 32, 16]),
        leaky_slope=disc_config.get('leaky_slope', 0.2),
        use_spectral_norm=disc_config.get('use_spectral_norm', True)
    )
    
    return generator.to(device), discriminator.to(device)


def create_optimizers(generator, discriminator, config: dict):
    """Create optimizers for generator and discriminator."""
    opt_config = config['training']['optimizer']
    
    g_optimizer = optim.Adam(
        generator.parameters(),
        lr=opt_config.get('lr_generator', 1e-4),
        betas=tuple(opt_config.get('betas', [0.0, 0.9])),
        weight_decay=opt_config.get('weight_decay', 1e-5)
    )
    
    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=opt_config.get('lr_discriminator', 1e-4),
        betas=tuple(opt_config.get('betas', [0.0, 0.9])),
        weight_decay=opt_config.get('weight_decay', 1e-5)
    )
    
    return g_optimizer, d_optimizer


def create_schedulers(g_optimizer, d_optimizer, config: dict, num_epochs: int):
    """Create learning rate schedulers."""
    sched_config = config['training'].get('scheduler', {})
    sched_type = sched_config.get('type', 'cosine')
    
    if sched_type == 'cosine':
        g_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            g_optimizer,
            T_max=num_epochs,
            eta_min=sched_config.get('min_lr', 1e-6)
        )
        d_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            d_optimizer,
            T_max=num_epochs,
            eta_min=sched_config.get('min_lr', 1e-6)
        )
    elif sched_type == 'step':
        g_scheduler = optim.lr_scheduler.StepLR(
            g_optimizer,
            step_size=sched_config.get('step_size', 100),
            gamma=sched_config.get('gamma', 0.5)
        )
        d_scheduler = optim.lr_scheduler.StepLR(
            d_optimizer,
            step_size=sched_config.get('step_size', 100),
            gamma=sched_config.get('gamma', 0.5)
        )
    else:
        g_scheduler = None
        d_scheduler = None
        
    return g_scheduler, d_scheduler


def train_step(
    generator: nn.Module,
    discriminator: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    g_optimizer: optim.Optimizer,
    d_optimizer: optim.Optimizer,
    w_loss: WassersteinLoss,
    spectral_loss: SpectralLoss,
    config: dict,
    device: torch.device,
    step: int
) -> Dict[str, float]:
    """
    Single training step using Indirect Learning Architecture (ILA).
    
    In ILA:
    - Input: PA output (distorted signal y_PA)
    - Generator produces: Predistorted signal (should match clean PA input u_PA)
    - Target: PA input (clean signal u_PA)
    
    NO PA model in training loop - we train on measured data!
    
    Returns dictionary of loss values.
    """
    train_config = config['training']
    loss_config = train_config.get('loss', {})
    n_critic = train_config.get('n_critic', 5)
    
    # Unpack batch
    input_seq, target = batch  # [B, M+1, 2], [B, 2]
    input_seq = input_seq.to(device)
    target = target.to(device)
    
    losses = {}
    
    # ===================
    # Train Discriminator
    # ===================
    for _ in range(n_critic):
        d_optimizer.zero_grad()
        
        # Generate DPD output
        with torch.no_grad():
            dpd_output = generator(input_seq)  # [B, 2]
            
        # Discriminator loss
        # Real: clean PA input (target)
        # Fake: DPD output (should also look clean)
        # Condition: PA output (current input to DPD)
        condition = input_seq[:, -1, :]  # Most recent sample
        
        d_loss, d_info = w_loss.discriminator_loss(
            discriminator, target, dpd_output, condition, device
        )
        
        d_loss.backward()
        d_optimizer.step()
        
    losses.update({f'd_{k}': v for k, v in d_info.items()})
    
    # =================
    # Train Generator
    # =================
    g_optimizer.zero_grad()
    
    # Generate DPD output (should match clean PA input)
    dpd_output = generator(input_seq)  # [B, 2]
    
    # Adversarial loss (DPD output should look like clean signal)
    condition = input_seq[:, -1, :]
    g_adv_loss, g_info = w_loss.generator_loss(discriminator, dpd_output, condition)
    
    # Reconstruction loss (DPD output should match PA input)
    recon_loss = nn.functional.l1_loss(dpd_output, target)
    
    # Spectral loss (frequency domain similarity)
    spectral, spectral_components = spectral_loss(dpd_output, target, return_components=True)
    
    # Combined generator loss
    g_total = (
        loss_config.get('adversarial', 1.0) * g_adv_loss +
        loss_config.get('reconstruction_l1', 50.0) * recon_loss +
        spectral
    )
    
    g_total.backward()
    g_optimizer.step()
    
    losses['g_total'] = g_total.item()
    losses['g_adv'] = g_adv_loss.item()
    losses['g_recon'] = recon_loss.item()
    losses['g_spectral'] = spectral.item()
    losses.update({f'g_{k}': v.item() for k, v in spectral_components.items()})
    
    return losses


def validate(
    generator: nn.Module,
    val_loader: DataLoader,
    spectral_loss: SpectralLoss,
    device: torch.device
) -> Dict[str, float]:
    """Validate model on validation set.
    
    In ILA validation:
    - Input: PA output (distorted)
    - DPD output: Should match PA input (clean)
    - Metrics: EVM, NMSE, L1 between DPD output and clean PA input
    """
    generator.eval()
    
    all_evm = []
    all_nmse = []
    all_recon = []
    
    with torch.no_grad():
        for input_seq, target in val_loader:
            input_seq = input_seq.to(device)
            target = target.to(device)
            
            # Generate DPD output
            dpd_output = generator(input_seq)
            
            # Compute metrics (DPD output vs clean PA input)
            metrics = spectral_loss.compute_metrics(dpd_output, target)
            all_evm.append(metrics['evm_db'])
            all_nmse.append(metrics['nmse_db'])
            all_recon.append(metrics['l1_error'])
            
    generator.train()
    
    return {
        'val_evm_db': np.mean(all_evm),
        'val_nmse_db': np.mean(all_nmse),
        'val_l1': np.mean(all_recon)
    }


def save_checkpoint(
    generator: nn.Module,
    discriminator: nn.Module,
    g_optimizer: optim.Optimizer,
    d_optimizer: optim.Optimizer,
    epoch: int,
    best_evm: float,
    config: dict,
    checkpoint_dir: Path,
    is_best: bool = False
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'best_evm': best_evm,
        'config': config
    }
    
    # Save latest
    torch.save(checkpoint, checkpoint_dir / 'latest.pth')
    
    # Save periodic checkpoint
    if (epoch + 1) % config['training'].get('checkpoint_interval', 25) == 0:
        torch.save(checkpoint, checkpoint_dir / f'epoch_{epoch+1}.pth')
        
    # Save best
    if is_best:
        torch.save(checkpoint, checkpoint_dir / 'best.pth')


def main():
    parser = argparse.ArgumentParser(description='Train 6G PA GAN-DPD')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--temp', type=str, default='all',
                        choices=['cold', 'normal', 'hot', 'all'],
                        help='Temperature state for training')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--qat', action='store_true',
                        help='Enable Quantization-Aware Training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--output', type=str, default=None,
                        help='Specific output checkpoint path (e.g., models/dpd_cold.pt)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with args
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
        
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    if args.output:
        # Use specific output path
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_dir = output_path.parent
    else:
        # Use timestamped directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(args.output_dir) / f'{args.temp}_{timestamp}'
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
        
    # Create models
    print("Creating models...")
    generator, discriminator = create_models(config, device, qat=args.qat)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Load measured PA data from CSV files
    print("\nLoading measured PA data from CSV files...")
    data_dir = Path('data')  # Directory with train_input.csv, train_output.csv, etc.
    
    # Load training data
    u_pa_train, y_pa_train = load_measured_data(data_dir, 'train')
    
    # Apply thermal drift based on temperature setting
    print(f"\nApplying thermal drift (temp={args.temp})...")
    if args.temp == 'all':
        # Train on all three temperature variants
        y_cold = apply_thermal_drift(y_pa_train, -20)
        y_normal = y_pa_train.copy()
        y_hot = apply_thermal_drift(y_pa_train, 70)
        
        # Concatenate all variants
        y_pa_combined = np.concatenate([y_cold, y_normal, y_hot])
        u_pa_combined = np.tile(u_pa_train, 3)
        
        print(f"  Combined dataset: {len(u_pa_combined):,} samples (3x thermal variants)")
        y_pa_train = y_pa_combined
        u_pa_train = u_pa_combined
    elif args.temp == 'cold':
        y_pa_train = apply_thermal_drift(y_pa_train, -20)
        print("  Using cold variant (-20°C)")
    elif args.temp == 'hot':
        y_pa_train = apply_thermal_drift(y_pa_train, 70)
        print("  Using hot variant (70°C)")
    else:  # normal
        print("  Using normal temperature (25°C)")
    
    # Create datasets
    memory_depth = config['model']['generator'].get('memory_depth', 5)
    train_dataset = create_dpd_dataset(u_pa_train, y_pa_train, memory_depth)
    
    # Load validation data (always use normal temperature)
    u_pa_val, y_pa_val = load_measured_data(data_dir, 'val')
    val_dataset = create_dpd_dataset(u_pa_val, y_pa_val, memory_depth)
    
    batch_size = config['training'].get('batch_size', 64)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    print(f"\nDataset sizes:")
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")
    
    # Create optimizers and schedulers
    g_optimizer, d_optimizer = create_optimizers(generator, discriminator, config)
    num_epochs = config['training'].get('epochs', 500)
    g_scheduler, d_scheduler = create_schedulers(g_optimizer, d_optimizer, config, num_epochs)
    
    # Create loss functions
    w_loss = WassersteinLoss(gp_weight=config['training'].get('gp_weight', 10.0))
    spectral_loss = SpectralLoss(
        sample_rate=config['system'].get('sample_rate', 200e6)
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_evm = float('inf')
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_evm = checkpoint.get('best_evm', float('inf'))
        
    # TensorBoard
    writer = SummaryWriter(output_dir / 'logs')
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    global_step = start_epoch * len(train_loader)
    
    for epoch in range(start_epoch, num_epochs):
        generator.train()
        discriminator.train()
        
        epoch_losses = {}
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            losses = train_step(
                generator, discriminator, batch,
                g_optimizer, d_optimizer, w_loss, spectral_loss,
                config, device, global_step
            )
            
            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v
                
            # Update progress bar
            pbar.set_postfix({
                'g_loss': f"{losses['g_total']:.4f}",
                'd_loss': f"{losses['d_d_loss']:.4f}",
                'evm': f"{losses.get('g_evm', 0):.4f}"
            })
            
            # Log to TensorBoard
            if global_step % config.get('logging', {}).get('log_interval', 100) == 0:
                for k, v in losses.items():
                    writer.add_scalar(f'train/{k}', v, global_step)
                    
            global_step += 1
            
        # Average epoch losses
        for k in epoch_losses:
            epoch_losses[k] /= len(train_loader)
            
        # Validation
        val_metrics = validate(generator, val_loader, spectral_loss, device)
        
        # Log validation metrics
        for k, v in val_metrics.items():
            writer.add_scalar(f'val/{k}', v, epoch)
            
        # Update schedulers
        if g_scheduler:
            g_scheduler.step()
        if d_scheduler:
            d_scheduler.step()
            
        # Check for best model
        is_best = val_metrics['val_evm_db'] < best_evm
        if is_best:
            best_evm = val_metrics['val_evm_db']
            
        # Save checkpoint
        save_checkpoint(
            generator, discriminator, g_optimizer, d_optimizer,
            epoch, best_evm, config, output_dir, is_best
        )
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  G Loss: {epoch_losses['g_total']:.4f}")
        print(f"  D Loss: {epoch_losses['d_d_loss']:.4f}")
        print(f"  Val EVM: {val_metrics['val_evm_db']:.2f} dB")
        print(f"  Val NMSE: {val_metrics['val_nmse_db']:.2f} dB")
        print(f"  Best EVM: {best_evm:.2f} dB")
        print(f"  LR: {g_optimizer.param_groups[0]['lr']:.2e}")
        
    # Final save
    print(f"\nTraining complete! Best EVM: {best_evm:.2f} dB")
    print(f"Checkpoints saved to: {output_dir}")
    
    # If specific output path provided, save final best model there
    if args.output:
        print(f"Copying best checkpoint to: {args.output}")
        best_checkpoint = torch.load(output_dir / 'best.pth', map_location='cpu')
        torch.save(best_checkpoint, args.output)
    
    writer.close()


if __name__ == '__main__':
    main()
