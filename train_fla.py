#!/usr/bin/env python3
# =============================================================================
# 6G PA GAN-DPD: Forward Learning Architecture (FLA) Training Script
# =============================================================================
"""
FLA Training for PN-TDNN DPD with Frozen PA Model
==================================================

This script trains the DPD using Forward Learning Architecture (FLA)
following the OpenDPD methodology:

FLA Flow:
    1. Load pre-trained PA surrogate model (DGRU)
    2. Freeze PA model parameters
    3. Train DPD: x → DPD → u_dpd → PA_frozen → y_pred
    4. Loss: ||y_pred - y_target||  (PA output matches desired output)

Key Differences from ILA (train.py):
    - ILA: DPD learns inverse mapping y_PA → u_PA directly
    - FLA: DPD learns predistortion through cascaded PA model

Reference:
    OpenDPD: End-to-End Learning for DPD (Wu et al., ISCAS 2024)
    OpenDPDv2: Unified Learning Framework (Wu et al., 2025)

Usage:
    python train_fla.py --config config/config.yaml --pa-checkpoint pa_checkpoints/pa_dgru_best.pth
    python train_fla.py --config config/config.yaml --epochs 300 --qat
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
from utils.dataset_sequence import create_fla_dataset_sequence, create_dataloaders


# =============================================================================
# PA Surrogate Model (DGRU) - Must match training architecture
# =============================================================================

class PAModelDGRU(nn.Module):
    """
    DGRU-based PA surrogate model (matches OpenDPD architecture).
    
    Learns to predict: u_PA (clean input) → y_PA (distorted output)
    Uses GRU to capture temporal memory effects in PA.
    """
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.fc_out = nn.Linear(hidden_size, 2)
        self.num_params = sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        """Forward pass: u_PA → y_PA"""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        out = self.fc_out(out[:, -1, :])
        return out


# =============================================================================
# Cascaded Model (DPD + Frozen PA) - OpenDPD E2E Architecture
# =============================================================================

class CascadedDPDPA(nn.Module):
    """
    Cascaded DPD + PA model for Forward Learning Architecture.
    
    Flow: x[n] → DPD → u_dpd[n] → PA_frozen → y_cas[n]
    
    Only DPD parameters are trainable; PA is frozen.
    """
    def __init__(self, dpd_model: nn.Module, pa_model: nn.Module):
        super().__init__()
        self.dpd = dpd_model
        self.pa = pa_model
        
        # Freeze PA model
        self.pa.eval()
        for param in self.pa.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass through cascaded model.
        
        Args:
            x: Input IQ sequence [B, M+1, 2] or [B, 2]
            
        Returns:
            y_cas: Cascaded output [B, 2]
            u_dpd: DPD output (for auxiliary loss) [B, 2]
        """
        # DPD: distorted input → predistorted output
        u_dpd = self.dpd(x)  # [B, 2]
        
        # PA (frozen): predistorted → amplified output
        y_cas = self.pa(u_dpd)  # [B, 2]
        
        return y_cas, u_dpd


# =============================================================================
# Data Loading
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_measured_data(data_dir: Path, split: str = 'train'):
    """
    Load measured PA input/output data from CSV files.
    
    FLA convention:
    - u_pa: PA input (clean signal) -> what DPD should produce
    - y_pa: PA output (distorted signal) -> target for cascaded model
    """
    input_file = data_dir / f'{split}_input.csv'
    output_file = data_dir / f'{split}_output.csv'
    
    if not input_file.exists() or not output_file.exists():
        raise FileNotFoundError(f"Data files not found in {data_dir}")
    
    input_df = pd.read_csv(input_file)
    output_df = pd.read_csv(output_file)
    
    u_pa = (input_df['I'].values + 1j * input_df['Q'].values).astype(np.complex64)
    y_pa = (output_df['I'].values + 1j * output_df['Q'].values).astype(np.complex64)
    
    # Normalize to -3 dBFS (0.7 linear)
    max_val = np.max(np.abs(u_pa))
    u_pa = u_pa / max_val * 0.7
    y_pa = y_pa / max_val * 0.7
    
    print(f"Loaded {len(u_pa):,} {split} samples")
    print(f"  PA input power:  {10*np.log10(np.mean(np.abs(u_pa)**2)):.2f} dBFS")
    print(f"  PA output power: {10*np.log10(np.mean(np.abs(y_pa)**2)):.2f} dBFS")
    
    return u_pa, y_pa


# Legacy function kept for backward compatibility - use create_fla_dataset_sequence instead
def create_fla_dataset(u_pa: np.ndarray, y_pa: np.ndarray, memory_depth: int = 3) -> TensorDataset:
    """DEPRECATED: Use create_fla_dataset_sequence for proper spectral training.
    
    This sample-by-sample approach cannot compute valid ACLR/EVM metrics.
    Keeping for backward compatibility only.
    """
    print("WARNING: Using deprecated sample-by-sample dataset. Use create_fla_dataset_sequence instead.")
    num_samples = len(u_pa) - memory_depth
    inputs = np.zeros((num_samples, memory_depth + 1, 2), dtype=np.float32)
    targets = np.zeros((num_samples, 2), dtype=np.float32)
    clean_inputs = np.zeros((num_samples, 2), dtype=np.float32)
    
    for i in range(num_samples):
        for m in range(memory_depth + 1):
            idx = i + memory_depth - m
            inputs[i, m, 0] = u_pa[idx].real
            inputs[i, m, 1] = u_pa[idx].imag
        target_idx = i + memory_depth
        targets[i, 0] = y_pa[target_idx].real
        targets[i, 1] = y_pa[target_idx].imag
        clean_inputs[i, 0] = u_pa[target_idx].real
        clean_inputs[i, 1] = u_pa[target_idx].imag
    
    return TensorDataset(
        torch.from_numpy(inputs),
        torch.from_numpy(targets),
        torch.from_numpy(clean_inputs)
    )


# =============================================================================
# Model Creation
# =============================================================================

def load_pa_model(checkpoint_path: str, device: torch.device) -> PAModelDGRU:
    """Load pre-trained PA surrogate model."""
    print(f"\nLoading PA model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config if available
    hidden_size = 64  # Default
    num_layers = 2    # Default
    
    pa_model = PAModelDGRU(
        input_size=2,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    
    pa_model.load_state_dict(checkpoint['model_state_dict'])
    pa_model = pa_model.to(device)
    pa_model.eval()
    
    # Freeze all parameters
    for param in pa_model.parameters():
        param.requires_grad = False
    
    print(f"  PA Model parameters: {pa_model.num_params:,}")
    if 'nmse_db' in checkpoint:
        print(f"  PA Model NMSE: {checkpoint['nmse_db']:.2f} dB")
    print("  PA Model: FROZEN ✓")
    
    return pa_model


def create_models(config: dict, pa_model: PAModelDGRU, device: torch.device, qat: bool = False):
    """Create DPD generator and cascaded model."""
    gen_config = config['model'].get('generator', {})
    
    # DPD Generator: Phase-Normalized TDNN
    generator = PNTDNNGenerator(
        memory_depth=gen_config.get('memory_depth', 3),
        hidden_dims=gen_config.get('hidden_dims', [32, 16]),
        leaky_slope=gen_config.get('leaky_slope', 0.2)
    )
    
    if qat:
        generator.enable_qat()
        print("QAT enabled for DPD generator")
    
    generator = generator.to(device)
    
    # Cascaded model (DPD + frozen PA)
    cascaded = CascadedDPDPA(generator, pa_model)
    
    # Optional: Discriminator for adversarial training
    discriminator = create_discriminator().to(device)
    
    return generator, discriminator, cascaded


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
            g_optimizer, T_max=num_epochs, eta_min=sched_config.get('min_lr', 1e-6)
        )
        d_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            d_optimizer, T_max=num_epochs, eta_min=sched_config.get('min_lr', 1e-6)
        )
    else:
        g_scheduler = None
        d_scheduler = None
    
    return g_scheduler, d_scheduler


# =============================================================================
# FLA Training Step (OpenDPD E2E Architecture)
# =============================================================================

def train_step_fla(
    cascaded: CascadedDPDPA,
    discriminator: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    g_optimizer: optim.Optimizer,
    d_optimizer: optim.Optimizer,
    spectral_loss: SpectralLoss,
    config: dict,
    device: torch.device,
    step: int,
    memory_depth: int = 3
) -> Dict[str, float]:
    """
    FLA training step with sequence-based approach.
    
    Sequence-based training: processes [B, seq_length, 2] sequences.
    Generator output: [B, seq_length - M, 2] (trimmed due to memory effects).
    
    Flow:
        x[n] → DPD → u_dpd[n] → PA_frozen → y_cas[n]
        Loss = ||y_cas - G*x|| + λ_spectral * L_spectral (on full sequences)
    """
    train_config = config['training']
    loss_config = train_config.get('loss', {})
    n_critic = train_config.get('n_critic', 5)
    gp_weight = train_config.get('gp_weight', 10.0)
    
    # Unpack batch: [B, seq_length, 2] for sequence-based training
    input_seq, target_seq, clean_input_seq = batch
    input_seq = input_seq.to(device)        # [B, seq_length, 2]
    target_seq = target_seq.to(device)      # [B, seq_length, 2]
    clean_input_seq = clean_input_seq.to(device)  # [B, seq_length, 2]
    
    batch_size = input_seq.size(0)
    seq_length = input_seq.size(1)
    
    losses = {}
    
    # Compute desired output (linear gain applied to clean input)
    linear_gain = 1.0
    desired_output_seq = linear_gain * clean_input_seq  # [B, seq_length, 2]
    
    # ===================
    # Train Discriminator
    # ===================
    for _ in range(n_critic):
        d_optimizer.zero_grad()
        
        with torch.no_grad():
            y_cas, u_dpd = cascaded(input_seq)  # Both [B, seq_length - M, 2]
        
        output_len = y_cas.size(1)
        
        # Trim sequences to match output length
        desired_trimmed = desired_output_seq[:, memory_depth:memory_depth + output_len, :]
        
        # Sample random time indices for discriminator
        rand_idx = torch.randint(0, output_len, (batch_size,), device=device)
        
        y_cas_sample = y_cas[torch.arange(batch_size), rand_idx, :]  # [B, 2]
        desired_sample = desired_trimmed[torch.arange(batch_size), rand_idx, :]  # [B, 2]
        condition = input_seq[torch.arange(batch_size), memory_depth + rand_idx, :]  # [B, 2]
        
        # Real: desired linear output, Fake: cascaded model output
        d_real = discriminator(desired_sample, condition)
        d_fake = discriminator(y_cas_sample.detach(), condition)
        
        # Wasserstein loss
        d_loss = d_fake.mean() - d_real.mean()
        
        # Gradient penalty
        alpha = torch.rand(batch_size, 1, device=device)
        interpolates = alpha * desired_sample + (1 - alpha) * y_cas_sample.detach()
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
    
    # =================
    # Train Generator (DPD) via Cascaded Model
    # =================
    g_optimizer.zero_grad()
    
    # Forward through cascaded model
    y_cas, u_dpd = cascaded(input_seq)  # Both [B, seq_length - M, 2]
    output_len = y_cas.size(1)
    
    # Trim sequences to match output length
    desired_trimmed = desired_output_seq[:, memory_depth:memory_depth + output_len, :]
    
    # Adversarial loss: sample from sequences
    rand_idx = torch.randint(0, output_len, (batch_size,), device=device)
    y_cas_sample = y_cas[torch.arange(batch_size), rand_idx, :]
    condition = input_seq[torch.arange(batch_size), memory_depth + rand_idx, :]
    
    d_fake = discriminator(y_cas_sample, condition)
    g_adv_loss = -d_fake.mean()
    
    # Primary FLA loss on full sequences: ||y_cas - desired_output||
    fla_loss = nn.functional.mse_loss(y_cas, desired_trimmed)
    
    # L1 reconstruction loss on full sequences
    recon_loss = nn.functional.l1_loss(y_cas, desired_trimmed)
    
    # Spectral loss on cascaded output (now valid with long sequences)
    spectral, spectral_components = spectral_loss(y_cas, desired_trimmed, return_components=True)
    
    # Combined generator loss
    g_total = (
        loss_config.get('adversarial', 1.0) * g_adv_loss +
        loss_config.get('fla_mse', 100.0) * fla_loss +
        loss_config.get('reconstruction_l1', 10.0) * recon_loss +
        loss_config.get('spectral', 10.0) * spectral
    )
    
    g_total.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(cascaded.dpd.parameters(), max_norm=1.0)
    
    g_optimizer.step()
    
    losses['g_adv'] = g_adv_loss.item()
    losses['g_fla_mse'] = fla_loss.item()
    losses['g_recon'] = recon_loss.item()
    losses['g_spectral'] = spectral.item()
    losses['g_total'] = g_total.item()
    losses.update({f'g_{k}': v.item() for k, v in spectral_components.items()})
    
    return losses


# =============================================================================
# Validation
# =============================================================================

def validate_fla(
    cascaded: CascadedDPDPA,
    val_loader: DataLoader,
    spectral_loss: SpectralLoss,
    device: torch.device,
    memory_depth: int = 3
) -> Dict[str, float]:
    """
    Validate FLA model with sequence-based metrics.
    
    Metrics computed on cascaded output vs desired linear output.
    Each sequence is long enough for valid Welch PSD computation.
    """
    cascaded.dpd.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for input_seq, target_seq, clean_input_seq in val_loader:
            input_seq = input_seq.to(device)
            clean_input_seq = clean_input_seq.to(device)
            
            y_cas, u_dpd = cascaded(input_seq)  # [B, seq_length - M, 2]
            output_len = y_cas.size(1)
            
            # Desired output = linear(clean_input), trimmed to match
            desired_trimmed = clean_input_seq[:, memory_depth:memory_depth + output_len, :]
            
            all_preds.append(y_cas.cpu())
            all_targets.append(desired_trimmed.cpu())
    
    cascaded.dpd.train()
    
    # Aggregate predictions
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
    
    # L1 error
    l1_error = np.mean(np.abs(all_preds - all_targets))
    
    return {
        'val_nmse_db': np.mean(all_nmse) if all_nmse else 0.0,
        'val_evm_db': np.mean(all_evm) if all_evm else 0.0,
        'val_aclr_lower_db': np.mean(all_aclr_lower) if all_aclr_lower else 0.0,
        'val_aclr_upper_db': np.mean(all_aclr_upper) if all_aclr_upper else 0.0,
        'val_l1': l1_error
    }


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(
    generator: nn.Module,
    discriminator: nn.Module,
    g_optimizer: optim.Optimizer,
    d_optimizer: optim.Optimizer,
    epoch: int,
    best_nmse: float,
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
        'best_nmse': best_nmse,
        'config': config,
        'architecture': 'FLA'
    }
    
    torch.save(checkpoint, checkpoint_dir / 'fla_latest.pth')
    
    if (epoch + 1) % config['training'].get('checkpoint_interval', 25) == 0:
        torch.save(checkpoint, checkpoint_dir / f'fla_epoch_{epoch+1}.pth')
    
    if is_best:
        torch.save(checkpoint, checkpoint_dir / 'fla_best.pth')


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train 6G PA GAN-DPD using FLA')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--pa-checkpoint', type=str, 
                        default='pa_ablation_checkpoints/pa_dgru_best.pth',
                        help='Path to pre-trained PA model checkpoint')
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
    parser.add_argument('--output-dir', type=str, default='checkpoints_fla',
                        help='Output directory for checkpoints')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with args
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print("6G PA GAN-DPD: Forward Learning Architecture (FLA) Training")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directory
    checkpoint_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard
    log_dir = checkpoint_dir / 'logs' / datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir)
    
    # Load data
    print(f"\n{'='*70}")
    print("Loading Data")
    print(f"{'='*70}")
    data_dir = Path(config.get('data', {}).get('dir', 'data/DPA_200MHz'))
    
    u_pa_train, y_pa_train = load_measured_data(data_dir, 'train')
    u_pa_val, y_pa_val = load_measured_data(data_dir, 'val')
    
    # Create sequence-based datasets for proper spectral training
    memory_depth = config['model']['generator'].get('memory_depth', 3)
    seq_length = config.get('spectral_loss', {}).get('nperseg', 2560)
    stride = seq_length // 2  # 50% overlap
    batch_size = config['training'].get('batch_size', 8)  # Fewer but larger sequences
    
    # Create dataloaders using sequence-based dataset
    train_loader, val_loader, _ = create_dataloaders(
        u_pa_train, y_pa_train,
        u_pa_val, y_pa_val,
        batch_size=batch_size,
        seq_length=seq_length,
        stride=stride,
        memory_depth=memory_depth,
        mode='fla',
        num_workers=0,  # Use 0 for Windows compatibility
        pin_memory=True
    )
    
    print(f"\nSequence-based training:")
    print(f"  seq_length={seq_length}, stride={stride}")
    print(f"  Frequency resolution: {800e6/seq_length/1e3:.1f} kHz (vs 200 MHz for sample-based)")
    
    # Load PA model
    print(f"\n{'='*70}")
    print("Loading Pre-trained PA Model")
    print(f"{'='*70}")
    pa_model = load_pa_model(args.pa_checkpoint, device)
    
    # Create models
    print(f"\n{'='*70}")
    print("Creating DPD Model")
    print(f"{'='*70}")
    generator, discriminator, cascaded = create_models(config, pa_model, device, args.qat)
    
    print(f"DPD Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Create optimizers and schedulers
    g_optimizer, d_optimizer = create_optimizers(generator, discriminator, config)
    num_epochs = config['training'].get('epochs', 500)
    g_scheduler, d_scheduler = create_schedulers(g_optimizer, d_optimizer, config, num_epochs)
    
    # Spectral loss
    spectral_loss = SpectralLoss(
        sample_rate=config['system'].get('sample_rate', 800e6),
        bw_main_ch=config['spectral_loss'].get('bw_main_ch', 200e6),
        n_sub_ch=config['spectral_loss'].get('n_sub_ch', 10),
        nperseg=config['spectral_loss'].get('nperseg', 2560),
        l1_weight=50.0,
        power_weight=10.0,
        nmse_weight=10.0,
        acpr_weight=20.0       # Frequency-domain spectral loss
    ).to(device)
    
    # Resume from checkpoint
    start_epoch = 0
    best_nmse = float('inf')
    
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_nmse = checkpoint.get('best_nmse', float('inf'))
    
    # Training loop
    print(f"\n{'='*70}")
    print("Starting FLA Training")
    print(f"{'='*70}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {config['training']['optimizer'].get('lr_generator', 1e-4)}")
    print(f"Target NMSE: < -42 dB")
    qat_start_epoch = config.get('quantization', {}).get('qat', {}).get('start_epoch', 300)
    print(f"QAT scheduled to start at epoch {qat_start_epoch}")
    print(f"{'='*70}\n")
    
    global_step = start_epoch * len(train_loader)
    qat_enabled = False
    
    for epoch in range(start_epoch, num_epochs):
        # Two-stage training: FP32 baseline (0-{qat_start_epoch-1}) then QAT fine-tuning ({qat_start_epoch}+)
        if epoch == qat_start_epoch and not qat_enabled:
            print(f"\n*** QAT Transition at Epoch {epoch} ***")
            # Load best FP32 checkpoint before enabling QAT
            best_checkpoint_path = checkpoint_dir / 'best.pth'
            if best_checkpoint_path.exists():
                print(f"Loading best FP32 checkpoint from {best_checkpoint_path}")
                best_ckpt = torch.load(best_checkpoint_path, map_location=device)
                generator.load_state_dict(best_ckpt['generator_state_dict'])
                print(f"Restored best model weights (NMSE: {best_ckpt['best_nmse']:.2f} dB)")
            else:
                print("Warning: best.pth not found, continuing with current weights")
            # Enable QAT on restored weights
            print("Enabling QAT: Q1.15 weights, Q8.8 activations")
            generator.enable_qat()
            qat_enabled = True
        
        generator.train()
        discriminator.train()
        
        epoch_losses = {
            'g_total': [], 'g_fla_mse': [], 'g_adv': [], 'g_recon': [], 'g_spectral': [],
            'd_total': [], 'd_wasserstein': [], 'd_gp': []
        }
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [FLA]')
        
        for batch in pbar:
            losses = train_step_fla(
                cascaded, discriminator, batch,
                g_optimizer, d_optimizer,
                spectral_loss, config, device, global_step,
                memory_depth=memory_depth
            )
            
            for k, v in losses.items():
                if k in epoch_losses:
                    epoch_losses[k].append(v)
            
            # TensorBoard logging
            for k, v in losses.items():
                writer.add_scalar(f'Train/{k}', v, global_step)
            
            global_step += 1
            
            pbar.set_postfix({
                'G': f"{losses['g_total']:.4f}",
                'FLA': f"{losses['g_fla_mse']:.6f}",
                'D': f"{losses['d_total']:.4f}"
            })
        
        # Scheduler step
        if g_scheduler:
            g_scheduler.step()
            d_scheduler.step()
        
        # Validation
        val_metrics = validate_fla(cascaded, val_loader, spectral_loss, device, memory_depth)
        
        # Log validation metrics
        for k, v in val_metrics.items():
            writer.add_scalar(f'Validation/{k}', v, epoch)
        
        # Check for best model (lower NMSE is better)
        is_best = val_metrics['val_nmse_db'] < best_nmse
        if is_best:
            best_nmse = val_metrics['val_nmse_db']
        
        # Save checkpoint
        save_checkpoint(
            generator, discriminator,
            g_optimizer, d_optimizer,
            epoch, best_nmse, config,
            checkpoint_dir, is_best
        )
        
        # Print epoch summary
        avg_g_loss = np.mean(epoch_losses['g_total'])
        avg_fla_loss = np.mean(epoch_losses['g_fla_mse'])
        
        print(f"\nEpoch {epoch+1}/{num_epochs} {'[QAT FINE-TUNING]' if qat_enabled else '[FP32 BASELINE]'}:")
        print(f"  Train: G_total={avg_g_loss:.4f}, FLA_MSE={avg_fla_loss:.6f}")
        print(f"  Val:   NMSE={val_metrics['val_nmse_db']:.2f} dB, "
              f"EVM={val_metrics['val_evm_db']:.2f} dB, "
              f"ACLR={val_metrics['val_aclr_lower_db']:.2f}/{val_metrics['val_aclr_upper_db']:.2f} dBc")
        print(f"  Best NMSE: {best_nmse:.2f} dB {'✓ NEW BEST' if is_best else ''}")
        
        # Check target achievement
        if val_metrics['val_nmse_db'] < -42:
            print(f"\n✅ Target NMSE achieved! ({val_metrics['val_nmse_db']:.2f} dB < -42 dB)")
        if val_metrics['val_evm_db'] < -45:
            print(f"✅ Target EVM achieved! ({val_metrics['val_evm_db']:.2f} dB < -45 dB)")
        if val_metrics['val_aclr_lower_db'] < -62 and val_metrics['val_aclr_upper_db'] < -62:
            print(f"✅ Target ACLR achieved! (Lower: {val_metrics['val_aclr_lower_db']:.2f}, Upper: {val_metrics['val_aclr_upper_db']:.2f} dBc < -62 dBc)")
    
    writer.close()
    
    print(f"\n{'='*70}")
    print("FLA Training Complete!")
    print(f"{'='*70}")
    print(f"Best NMSE: {best_nmse:.2f} dB")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"TensorBoard logs: {log_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
