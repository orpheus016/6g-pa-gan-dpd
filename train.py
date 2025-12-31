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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm
import numpy as np

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
    pa_model: nn.Module,
    batch: Dict[str, torch.Tensor],
    g_optimizer: optim.Optimizer,
    d_optimizer: optim.Optimizer,
    w_loss: WassersteinLoss,
    spectral_loss: SpectralLoss,
    config: dict,
    device: torch.device,
    step: int
) -> Dict[str, float]:
    """
    Single training step.
    
    Returns dictionary of loss values.
    """
    train_config = config['training']
    loss_config = train_config.get('loss', {})
    n_critic = train_config.get('n_critic', 5)
    
    # Move batch to device
    input_seq = batch['input'].to(device)  # [B, seq+M, 2]
    target = batch['target'].to(device)    # [B, seq, 2]
    
    losses = {}
    
    # ===================
    # Train Discriminator
    # ===================
    for _ in range(n_critic):
        d_optimizer.zero_grad()
        
        # Generate DPD output
        with torch.no_grad():
            dpd_output = generator(input_seq)  # [B, seq, 2]
            
        # Pass through PA model (DPD output → PA → should match target)
        # Note: In real DPD, we want G(x) such that PA(G(x)) ≈ x
        # So discriminator compares PA(G(x)) vs x
        with torch.no_grad():
            pa_output = pa_model(dpd_output, add_noise=False)
            
        # Discriminator loss
        # Real: target signal
        # Fake: PA output of DPD (should match target)
        # Condition: input signal
        condition = input_seq[:, -target.shape[1]:, :]  # Align with output
        
        d_loss, d_info = w_loss.discriminator_loss(
            discriminator, target, pa_output, condition, device
        )
        
        d_loss.backward()
        d_optimizer.step()
        
    losses.update({f'd_{k}': v for k, v in d_info.items()})
    
    # =================
    # Train Generator
    # =================
    g_optimizer.zero_grad()
    
    # Generate DPD output
    dpd_output = generator(input_seq)
    
    # Pass through PA
    pa_output = pa_model(dpd_output, add_noise=False)
    
    # Adversarial loss
    condition = input_seq[:, -target.shape[1]:, :]
    g_adv_loss, g_info = w_loss.generator_loss(discriminator, pa_output, condition)
    
    # Reconstruction loss (PA output should match target)
    recon_loss = nn.functional.l1_loss(pa_output, target)
    
    # Spectral loss
    spectral, spectral_components = spectral_loss(pa_output, target, return_components=True)
    
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
    pa_model: nn.Module,
    val_loader: DataLoader,
    spectral_loss: SpectralLoss,
    device: torch.device
) -> Dict[str, float]:
    """Validate model on validation set."""
    generator.eval()
    
    all_evm = []
    all_nmse = []
    all_recon = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_seq = batch['input'].to(device)
            target = batch['target'].to(device)
            
            # Generate DPD output
            dpd_output = generator(input_seq)
            
            # Pass through PA
            pa_output = pa_model(dpd_output, add_noise=False)
            
            # Compute metrics
            metrics = spectral_loss.compute_metrics(pa_output, target)
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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
        
    # Create models
    print("Creating models...")
    generator, discriminator = create_models(config, device, qat=args.qat)
    
    # Create PA digital twin
    pa_model = PADigitalTwin(
        memory_depth=config['pa'].get('memory_depth', 5),
        nonlinear_order=config['pa'].get('nonlinear_order', 7),
        gain_db=config['pa'].get('gain_db', 30.0)
    ).to(device)
    pa_model.eval()
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Create dataset
    print("Loading dataset...")
    data_path = Path(config['data'].get('dataset_path', 'data/raw/'))
    
    train_dataset = DPDDataset(
        data_path=data_path,
        memory_depth=config['model']['generator'].get('memory_depth', 5),
        sequence_length=256,
        temp_state=args.temp,
        normalize=True,
        augment_noise=0.01
    )
    train_dataset.train()
    
    val_dataset = DPDDataset(
        data_path=data_path,
        memory_depth=config['model']['generator'].get('memory_depth', 5),
        sequence_length=256,
        temp_state='normal',
        normalize=True,
        augment_noise=0.0
    )
    val_dataset.eval()
    
    batch_size = config['training'].get('batch_size', 64)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
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
                generator, discriminator, pa_model, batch,
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
        val_metrics = validate(generator, pa_model, val_loader, spectral_loss, device)
        
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
    
    writer.close()


if __name__ == '__main__':
    main()
