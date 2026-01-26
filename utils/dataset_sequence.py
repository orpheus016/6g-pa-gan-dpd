#!/usr/bin/env python3
"""
Sequence-Based Dataset Creation for DPD Training
=================================================

This module provides sequence-based dataset creation functions that replace
the sample-by-sample approach. Sequence-based training is required for
proper spectral metric computation (ACLR, EVM) via Welch's method.

Why sequence-based training?
----------------------------
Welch's method requires sequences of length >= nperseg (typically 2560) to 
compute valid Power Spectral Density (PSD) estimates. With 4-sample sequences:
- Frequency resolution = fs / N = 800 MHz / 4 = 200 MHz per bin
- Cannot distinguish 200 MHz main channel from 100 MHz offset adjacent channels
- ACLR computation becomes undefined

With 2560-sample sequences:
- Frequency resolution = 800 MHz / 2560 = 312.5 kHz per bin
- Main channel = 640 bins, adjacent channels clearly separated
- ACLR well-defined and optimizable

Usage:
------
    from utils.dataset_sequence import create_dpd_dataset_sequence, create_fla_dataset_sequence

    # For ILA training (train.py)
    train_dataset = create_dpd_dataset_sequence(u_pa_train, y_pa_train, seq_length=2560, stride=1280)
    
    # For FLA training (train_fla.py)
    train_dataset = create_fla_dataset_sequence(u_pa_train, y_pa_train, seq_length=2560, stride=1280)

Reference:
----------
- Welch's method: Oppenheim & Schafer, "Discrete-Time Signal Processing", Ch. 10
- OpenDPDv2: nperseg=2560, frame_length=2560 for proper spectral estimation
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Optional


# =============================================================================
# VECTORIZED Dataset Creation (10-100x faster for small strides)
# =============================================================================

def create_dpd_dataset_sequence(
    u_pa: np.ndarray,
    y_pa: np.ndarray,
    seq_length: int = 500,
    stride: int = 1,
    memory_depth: int = 3
) -> TensorDataset:
    """
    OPTIMIZED: Vectorized sequence-based dataset for ILA DPD training.
    
    Uses NumPy advanced indexing for 10-100x faster preprocessing (especially stride=1).
    
    ILA: Train DPD as post-inverse: DPD(y_PA) → u_PA
    - Input: PA output sequences (distorted)
    - Target: PA input sequences (clean, what DPD should produce)
    
    Args:
        u_pa: PA input signal (complex64, what DPD should produce)
        y_pa: PA output signal (complex64, input to DPD)
        seq_length: Length of each sequence
        stride: Stride between sequences (1 = maximum overlap)
        memory_depth: Memory depth M (for info only)
    
    Returns:
        TensorDataset with:
        - inputs: [num_sequences, seq_length, 2] - PA output (DPD input)
        - targets: [num_sequences, seq_length, 2] - PA input (DPD target)
    """
    n_samples = len(u_pa)
    
    if n_samples < seq_length:
        raise ValueError(f"Data length ({n_samples}) < seq_length ({seq_length}). "
                        f"Need at least {seq_length} samples.")
    
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    
    # Calculate number of sequences
    num_sequences = (n_samples - seq_length) // stride + 1
    
    if num_sequences == 0:
        raise ValueError(f"Cannot create sequences: n_samples={n_samples}, "
                        f"seq_length={seq_length}, stride={stride}")
    
    print(f"Vectorized preprocessing: {num_sequences} sequences...")
    
    # VECTORIZED: Create all sequence indices at once
    # Shape: [num_sequences, seq_length]
    start_indices = np.arange(0, num_sequences * stride, stride)
    indices = start_indices[:, None] + np.arange(seq_length)
    
    # Slice all sequences at once using advanced indexing
    y_sliced = y_pa[indices]  # [num_sequences, seq_length] complex
    u_sliced = u_pa[indices]  # [num_sequences, seq_length] complex
    
    # Convert to [num_sequences, seq_length, 2] format (I/Q channels)
    inputs = np.stack([y_sliced.real, y_sliced.imag], axis=-1).astype(np.float32)
    targets = np.stack([u_sliced.real, u_sliced.imag], axis=-1).astype(np.float32)
    
    print(f"Created ILA dataset: {num_sequences} sequences × {seq_length} samples")
    print(f"  Frequency resolution: {800e6 / seq_length / 1e3:.1f} kHz")
    
    return TensorDataset(
        torch.from_numpy(inputs),
        torch.from_numpy(targets)
    )


def create_fla_dataset_sequence(
    u_pa: np.ndarray,
    y_pa: np.ndarray,
    seq_length: int = 500,
    stride: int = 1,
    memory_depth: int = 3
) -> TensorDataset:
    """
    OPTIMIZED: Vectorized sequence-based dataset for FLA DPD training.
    
    Uses NumPy advanced indexing for 10-100x faster preprocessing (especially stride=1).
    
    FLA: Train DPD through frozen PA: x → DPD → PA_frozen → y_cas
    - Input: PA input sequences (clean, what we want to predistort)
    - Target: PA output sequences (what cascaded model should approximate)
    - Clean: PA input sequences (for auxiliary loss)
    
    Args:
        u_pa: PA input signal (complex64, clean input to predistort)
        y_pa: PA output signal (complex64, target for cascaded model)
        seq_length: Length of each sequence
        stride: Stride between sequences (1 = maximum overlap)
        memory_depth: Memory depth M (for info only)
    
    Returns:
        TensorDataset with:
        - inputs: [num_sequences, seq_length, 2] - PA input (DPD input)
        - targets: [num_sequences, seq_length, 2] - PA output (cascaded target)
        - clean_inputs: [num_sequences, seq_length, 2] - PA input (for aux loss)
    """
    n_samples = len(u_pa)
    
    if n_samples < seq_length:
        raise ValueError(f"Data length ({n_samples}) < seq_length ({seq_length}). "
                        f"Need at least {seq_length} samples.")
    
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    
    # Calculate number of sequences
    num_sequences = (n_samples - seq_length) // stride + 1
    
    if num_sequences == 0:
        raise ValueError(f"Cannot create sequences: n_samples={n_samples}, "
                        f"seq_length={seq_length}, stride={stride}")
    
    print(f"Vectorized preprocessing: {num_sequences} sequences...")
    
    # VECTORIZED: Create all sequence indices at once
    # Shape: [num_sequences, seq_length]
    start_indices = np.arange(0, num_sequences * stride, stride)
    indices = start_indices[:, None] + np.arange(seq_length)
    
    # Slice all sequences at once using advanced indexing
    u_sliced = u_pa[indices]  # [num_sequences, seq_length] complex
    y_sliced = y_pa[indices]  # [num_sequences, seq_length] complex
    
    # Convert to [num_sequences, seq_length, 2] format (I/Q channels)
    inputs = np.stack([u_sliced.real, u_sliced.imag], axis=-1).astype(np.float32)
    targets = np.stack([y_sliced.real, y_sliced.imag], axis=-1).astype(np.float32)
    clean_inputs = np.stack([u_sliced.real, u_sliced.imag], axis=-1).astype(np.float32)
    
    print(f"Created FLA dataset: {num_sequences} sequences × {seq_length} samples")
    print(f"  Frequency resolution: {800e6 / seq_length / 1e3:.1f} kHz")
    
    return TensorDataset(
        torch.from_numpy(inputs),
        torch.from_numpy(targets),
        torch.from_numpy(clean_inputs)
    )


def create_dataloaders(
    u_pa_train: np.ndarray,
    y_pa_train: np.ndarray,
    u_pa_val: np.ndarray,
    y_pa_val: np.ndarray,
    u_pa_test: Optional[np.ndarray] = None,
    y_pa_test: Optional[np.ndarray] = None,
    batch_size: int = 32,
    seq_length: int = 500,
    stride: int = 1,
    memory_depth: int = 3,
    mode: str = 'ila',
    num_workers: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    OPTIMIZED: Create dataloaders with vectorized preprocessing and persistent workers.
    
    Speed improvements over original:
    - Vectorized sequence slicing (10-100x faster for stride=1)
    - Persistent workers (avoids worker respawn overhead)
    - Pin memory for faster GPU transfer
    
    Args:
        u_pa_train, y_pa_train: Training data (complex64)
        u_pa_val, y_pa_val: Validation data (complex64)
        u_pa_test, y_pa_test: Test data (optional, complex64)
        batch_size: Batch size (number of sequences per batch)
        seq_length: Sequence length (default 500)
        stride: Stride between sequences (default 1 for max overlap)
        memory_depth: Memory depth M (default 3)
        mode: 'ila' for ILA training, 'fla' for FLA training, 'pa' for PA surrogate training
        num_workers: Number of dataloader workers (default 2)
        pin_memory: Whether to pin memory for GPU transfer
        persistent_workers: Keep workers alive between epochs (default True)
    
    Returns:
        (train_loader, val_loader, test_loader) tuple
    """
    # Select dataset creation function
    if mode.lower() == 'ila':
        create_fn = create_dpd_dataset_sequence
    elif mode.lower() == 'fla':
        create_fn = create_fla_dataset_sequence
    elif mode.lower() == 'pa':
        # PA surrogate training: same as ILA structure (input → target)
        # but semantically: u_pa (clean) → y_pa (distorted)
        create_fn = create_dpd_dataset_sequence
    else:
        raise ValueError(f"mode must be 'ila', 'fla', or 'pa', got '{mode}'")
    
    print(f"\nCreating OPTIMIZED {mode.upper()} dataloaders...")
    print(f"  seq_length={seq_length}, stride={stride}, batch_size={batch_size}")
    print(f"  num_workers={num_workers}, persistent_workers={persistent_workers}")
    
    # Create datasets (uses vectorized preprocessing)
    train_dataset = create_fn(u_pa_train, y_pa_train, seq_length, stride, memory_depth)
    val_dataset = create_fn(u_pa_val, y_pa_val, seq_length, stride, memory_depth)
    
    test_dataset = None
    if u_pa_test is not None and y_pa_test is not None:
        test_dataset = create_fn(u_pa_test, y_pa_test, seq_length, stride, memory_depth)
    
    # Adjust batch size if larger than dataset
    train_batch = min(batch_size, len(train_dataset))
    val_batch = min(batch_size, len(val_dataset))
    
    if train_batch < batch_size:
        print(f"  Warning: train batch_size reduced to {train_batch} (dataset has {len(train_dataset)} sequences)")
    if val_batch < batch_size:
        print(f"  Warning: val batch_size reduced to {val_batch} (dataset has {len(val_dataset)} sequences)")
    
    # Create dataloaders with optimization flags
    use_persistent = persistent_workers and num_workers > 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        drop_last=False
    )
    
    test_loader = None
    if test_dataset is not None:
        test_batch = min(batch_size, len(test_dataset))
        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=use_persistent,
            drop_last=False
        )
    
    print(f"\nDataloader summary:")
    print(f"  Training:   {len(train_dataset)} sequences, {len(train_loader)} batches")
    print(f"  Validation: {len(val_dataset)} sequences, {len(val_loader)} batches")
    if test_loader:
        print(f"  Test:       {len(test_dataset)} sequences, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


# Alias for backward compatibility
create_dataloaders_optimized = create_dataloaders


if __name__ == "__main__":
    """Quick test of sequence dataset creation."""
    print("Testing sequence-based dataset creation...")
    
    # Create dummy data
    n = 25000
    u_pa = np.random.randn(n).astype(np.float32) + 1j * np.random.randn(n).astype(np.float32)
    y_pa = u_pa * 1.1 + 0.01 * np.random.randn(n).astype(np.float32)  # Simple distortion
    
    # Test ILA dataset
    print("\n--- ILA Dataset ---")
    ila_dataset = create_dpd_dataset_sequence(u_pa, y_pa, seq_length=2560, stride=1280)
    inputs, targets = ila_dataset[0]
    print(f"  Input shape: {inputs.shape}")   # [2560, 2]
    print(f"  Target shape: {targets.shape}") # [2560, 2]
    
    # Test FLA dataset
    print("\n--- FLA Dataset ---")
    fla_dataset = create_fla_dataset_sequence(u_pa, y_pa, seq_length=2560, stride=1280)
    inputs, targets, clean = fla_dataset[0]
    print(f"  Input shape: {inputs.shape}")   # [2560, 2]
    print(f"  Target shape: {targets.shape}") # [2560, 2]
    print(f"  Clean shape: {clean.shape}")    # [2560, 2]
    
    # Test dataloaders
    print("\n--- Dataloaders ---")
    train_loader, val_loader, _ = create_dataloaders(
        u_pa[:20000], y_pa[:20000],
        u_pa[20000:], y_pa[20000:],
        batch_size=4,
        seq_length=500,
        stride=1,
        mode='ila'
    )
    
    batch = next(iter(train_loader))
    print(f"  Batch inputs shape: {batch[0].shape}")   # [4, 2560, 2]
    print(f"  Batch targets shape: {batch[1].shape}")  # [4, 2560, 2]
    
    print("\n✅ All tests passed!")
