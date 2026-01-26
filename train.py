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

from models import PNTDNNGenerator, create_discriminator
from utils.spectral_loss import SpectralLoss, compute_nmse, compute_evm, compute_aclr
from utils.dataset_sequence import create_dpd_dataset_sequence, create_dataloaders


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


# Legacy function kept for backward compatibility - use create_dpd_dataset_sequence instead
def create_dpd_dataset(u_pa: np.ndarray, y_pa: np.ndarray, memory_depth: int = 3, 
                       seq_length: int = 64) -> TensorDataset:
    """DEPRECATED: Use create_dpd_dataset_sequence for proper spectral training.
    
    This sample-by-sample approach cannot compute valid ACLR/EVM metrics.
    Keeping for backward compatibility only.
    """
    print("WARNING: Using deprecated sample-by-sample dataset. Use create_dpd_dataset_sequence instead.")
    num_samples = len(y_pa) - memory_depth
    inputs = np.zeros((num_samples, memory_depth + 1, 2), dtype=np.float32)
    targets = np.zeros((num_samples, 2), dtype=np.float32)
    
    for i in range(num_samples):
        for m in range(memory_depth + 1):
            idx = i + memory_depth - m
            inputs[i, m, 0] = y_pa[idx].real
            inputs[i, m, 1] = y_pa[idx].imag
        target_idx = i + memory_depth
        targets[i, 0] = u_pa[target_idx].real
        targets[i, 1] = u_pa[target_idx].imag
    
    return TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))


def create_models(config: dict, device: torch.device, qat: bool = False):
    """Create generator and discriminator models.
    
    Uses PNTDNNGenerator (phase-normalized) with built-in QAT support.
    """
    gen_config = config['model'].get('generator', {})
    
    # Generator: Phase-Normalized TDNN
    # Default: M=3, hidden=[32, 16], 24-dim input -> 1,362 params
    generator = PNTDNNGenerator(
        memory_depth=gen_config.get('memory_depth', 3),
        hidden_dims=gen_config.get('hidden_dims', [32, 16]),
        leaky_slope=gen_config.get('leaky_slope', 0.2)
    )
    
    # Enable QAT if requested
    if qat:
        generator.enable_qat()
        print("QAT enabled: Q1.15 weights, Q8.8 activations")
        
    # Discriminator: Conditional WGAN-GP (fixed architecture)
    discriminator = create_discriminator()
    
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
    spectral_loss: SpectralLoss,
    config: dict,
    device: torch.device,
    step: int,
    memory_depth: int = 3
) -> Dict[str, float]:
    """
    Single training step using Indirect Learning Architecture (ILA).
    
    Sequence-based training: processes [B, seq_length, 2] sequences.
    Generator output: [B, seq_length - M, 2] (trimmed due to memory effects).
    
    In ILA:
    - Input: PA output sequences (distorted signal y_PA)
    - Generator produces: Predistorted sequences (should match clean PA input u_PA)
    - Target: PA input sequences (clean signal u_PA)
    
    Returns dictionary of loss values.
    """
    train_config = config['training']
    loss_config = train_config.get('loss', {})
    n_critic = train_config.get('n_critic', 5)
    gp_weight = train_config.get('gp_weight', 10.0)
    
    # Unpack batch: [B, seq_length, 2] for sequence-based training
    input_seq, target_seq = batch
    input_seq = input_seq.to(device)   # [B, seq_length, 2]
    target_seq = target_seq.to(device) # [B, seq_length, 2]
    
    batch_size = input_seq.size(0)
    seq_length = input_seq.size(1)
    
    losses = {}
    
    # ===================
    # Train Discriminator
    # ===================
    for _ in range(n_critic):
        d_optimizer.zero_grad()
        
        # Generate DPD output on full sequences
        with torch.no_grad():
            dpd_output = generator(input_seq)  # [B, seq_length - M, 2]
        
        output_len = dpd_output.size(1)
        
        # Trim target to match output length (due to memory effects)
        target_trimmed = target_seq[:, memory_depth:memory_depth + output_len, :]
        
        # For discriminator: sample random time indices from sequences
        # This maintains batch independence while using sequence data
        rand_idx = torch.randint(0, output_len, (batch_size,), device=device)
        
        dpd_sample = dpd_output[torch.arange(batch_size), rand_idx, :]  # [B, 2]
        target_sample = target_trimmed[torch.arange(batch_size), rand_idx, :]  # [B, 2]
        condition = input_seq[torch.arange(batch_size), memory_depth + rand_idx, :]  # [B, 2]
        
        # Critic scores
        d_real = discriminator(target_sample, condition)
        d_fake = discriminator(dpd_sample.detach(), condition)
        
        # Wasserstein loss
        d_loss = d_fake.mean() - d_real.mean()
        
        # Gradient penalty
        alpha = torch.rand(batch_size, 1, device=device)
        interpolates = alpha * target_sample + (1 - alpha) * dpd_sample.detach()
        interpolates.requires_grad_(True)
        
        d_interp = discriminator(interpolates, condition)
        
        gradients = torch.autograd.grad(
            outputs=d_interp,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        gradients_norm = gradients.view(batch_size, -1).norm(2, dim=1)
        gp = ((gradients_norm - 1) ** 2).mean()
        
        d_total = d_loss + gp_weight * gp
        d_total.backward()
        d_optimizer.step()
    
    losses['d_wasserstein'] = d_loss.item()
    losses['d_gp'] = gp.item()
    losses['d_total'] = d_total.item()
    losses['d_d_loss'] = d_loss.item()  # For backward compat with logging
    
    # =================
    # Train Generator
    # =================
    g_optimizer.zero_grad()
    
    # Generate DPD output on full sequences
    dpd_output = generator(input_seq)  # [B, seq_length - M, 2]
    output_len = dpd_output.size(1)
    
    # Trim target to match output length
    target_trimmed = target_seq[:, memory_depth:memory_depth + output_len, :]
    
    # Adversarial loss: sample from sequences
    rand_idx = torch.randint(0, output_len, (batch_size,), device=device)
    dpd_sample = dpd_output[torch.arange(batch_size), rand_idx, :]
    condition = input_seq[torch.arange(batch_size), memory_depth + rand_idx, :]
    
    d_fake = discriminator(dpd_sample, condition)
    g_adv_loss = -d_fake.mean()
    
    # Reconstruction loss (L1: DPD output should match PA input) - on full sequences
    recon_loss = nn.functional.l1_loss(dpd_output, target_trimmed)
    
    # Spectral loss on full sequences (now valid with seq_length >= nperseg)
    spectral, spectral_components = spectral_loss(dpd_output, target_trimmed, return_components=True)
    
    # Combined generator loss
    g_total = (
        loss_config.get('adversarial', 1.0) * g_adv_loss +
        loss_config.get('reconstruction_l1', 40.0) * recon_loss +
        loss_config.get('spectral', 10.0) * spectral
    )
    
    g_total.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
    
    g_optimizer.step()
    
    losses['g_adv'] = g_adv_loss.item()
    losses['g_recon'] = recon_loss.item()
    losses['g_spectral'] = spectral.item()
    losses['g_total'] = g_total.item()
    losses.update({f'g_{k}': v.item() for k, v in spectral_components.items()})
    
    return losses


def validate(
    generator: nn.Module,
    val_loader: DataLoader,
    spectral_loss: SpectralLoss,
    device: torch.device,
    memory_depth: int = 3
) -> Dict[str, float]:
    """Validate model on validation set using sequence-based metrics.
    
    In ILA validation:
    - Input: PA output sequences (distorted)
    - DPD output: Should match PA input sequences (clean)
    - Metrics: EVM, NMSE, ACLR computed per-sequence (now valid)
    """
    generator.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for input_seq, target_seq in val_loader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            # Generate DPD output
            dpd_output = generator(input_seq)  # [B, seq_length - M, 2]
            output_len = dpd_output.size(1)
            
            # Trim target to match output length
            target_trimmed = target_seq[:, memory_depth:memory_depth + output_len, :]
            
            all_preds.append(dpd_output.cpu())
            all_targets.append(target_trimmed.cpu())
    
    generator.train()
    
    # Aggregate all predictions
    all_preds = torch.cat(all_preds, dim=0).numpy()    # [N, seq_len, 2]
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Compute metrics per sequence (each sequence is valid for Welch)
    all_evm = []
    all_nmse = []
    all_aclr_lower = []
    all_aclr_upper = []
    
    for i in range(len(all_preds)):
        pred = all_preds[i]    # [seq_len, 2]
        tgt = all_targets[i]   # [seq_len, 2]
        
        # NMSE (time-domain)
        nmse = compute_nmse(pred, tgt, return_db=True)
        all_nmse.append(nmse)
        
        # EVM (frequency-domain per subchannel)
        evm = compute_evm(pred, tgt, return_db=True)
        all_evm.append(evm)
        
        # ACLR (Welch PSD based)
        try:
            aclr_lower, aclr_upper = compute_aclr(pred, tgt, return_db=True)
            all_aclr_lower.append(aclr_lower)
            all_aclr_upper.append(aclr_upper)
        except:
            pass  # Skip if sequence too short
    
    # Compute L1 error
    l1_error = np.mean(np.abs(all_preds - all_targets))
    
    return {
        'val_evm_db': np.mean(all_evm) if all_evm else 0.0,
        'val_nmse_db': np.mean(all_nmse) if all_nmse else 0.0,
        'val_aclr_lower_db': np.mean(all_aclr_lower) if all_aclr_lower else 0.0,
        'val_aclr_upper_db': np.mean(all_aclr_upper) if all_aclr_upper else 0.0,
        'val_l1': l1_error
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
    
    # Create sequence-based datasets for proper spectral training
    memory_depth = config['model'].get('generator', {}).get('memory_depth', 3)
    seq_length = 500
    stride = 1
    batch_size = 32
    
    # Load validation data (always use normal temperature)
    u_pa_val, y_pa_val = load_measured_data(data_dir, 'val')
    
    # Create dataloaders using sequence-based dataset
    train_loader, val_loader, _ = create_dataloaders(
        u_pa_train, y_pa_train,
        u_pa_val, y_pa_val,
        batch_size=batch_size,
        seq_length=seq_length,
        stride=stride,
        memory_depth=memory_depth,
        mode='ila',
        num_workers=0,  # Use 0 for Windows compatibility
        pin_memory=True
    )
    
    print(f"\nSequence-based training:")
    print(f"  seq_length={seq_length}, stride={stride}")
    print(f"  Frequency resolution: {800e6/seq_length/1e3:.1f} kHz (vs 200 MHz for sample-based)")
    
    # Create optimizers and schedulers
    g_optimizer, d_optimizer = create_optimizers(generator, discriminator, config)
    num_epochs = config['training'].get('epochs', 500)
    g_scheduler, d_scheduler = create_schedulers(g_optimizer, d_optimizer, config, num_epochs)
    
    # Create loss functions
    # Note: WassersteinLoss imported from utils.spectral_loss
    spectral_loss = SpectralLoss(
        sample_rate=config['system'].get('sample_rate', 800e6),
        bw_main_ch=config.get('spectral_loss', {}).get('bw_main_ch', 200e6),
        n_sub_ch=config.get('spectral_loss', {}).get('n_sub_ch', 10),
        nperseg=config.get('spectral_loss', {}).get('nperseg', 2560),
        l1_weight=50.0,
        power_weight=10.0,
        nmse_weight=10.0,
        acpr_weight=20.0       # Frequency-domain spectral loss
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
    qat_start_epoch = config.get('quantization', {}).get('qat', {}).get('start_epoch', 300)
    qat_enabled = False
    print(f"QAT scheduled to start at epoch {qat_start_epoch}")
    global_step = start_epoch * len(train_loader)
    
    for epoch in range(start_epoch, num_epochs):
        # Two-stage training: FP32 baseline (0-{qat_start_epoch-1}) then QAT fine-tuning ({qat_start_epoch}+)
        if epoch == qat_start_epoch and not qat_enabled:
            print(f"\n*** QAT Transition at Epoch {epoch} ***")
            # Load best FP32 checkpoint before enabling QAT
            best_checkpoint_path = output_dir / 'best.pth'
            if best_checkpoint_path.exists():
                print(f"Loading best FP32 checkpoint from {best_checkpoint_path}")
                best_ckpt = torch.load(best_checkpoint_path, map_location=device)
                generator.load_state_dict(best_ckpt['generator_state_dict'])
                print(f"Restored best model weights (EVM: {best_ckpt['best_evm']:.2f} dB)")
            else:
                print("Warning: best.pth not found, continuing with current weights")
            # Enable QAT on restored weights
            print("Enabling QAT: Q1.15 weights, Q8.8 activations")
            generator.enable_qat()
            qat_enabled = True
        
        generator.train()
        discriminator.train()
        
        epoch_losses = {}
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            losses = train_step(
                generator, discriminator, batch,
                g_optimizer, d_optimizer, spectral_loss,
                config, device, global_step,
                memory_depth=memory_depth
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
        val_metrics = validate(generator, val_loader, spectral_loss, device, memory_depth)
        
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
        print(f"\nEpoch {epoch+1}/{num_epochs} {'[QAT FINE-TUNING]' if qat_enabled else '[FP32 BASELINE]'}")
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
