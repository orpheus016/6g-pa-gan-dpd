# =============================================================================
# 6G PA GAN-DPD: Dataset Utilities with Thermal Augmentation
# =============================================================================
"""
DATASET UTILITIES FOR DPD TRAINING
==================================

This module handles:
1. Loading OpenDPD APA dataset (200MHz GaN PA)
2. Thermal drift augmentation (cold/normal/hot)
3. Memory-aware sample preparation
4. DataLoader creation for training

Dataset Structure:
- Input: PA input IQ samples
- Output: PA output IQ samples (with nonlinearity)
- Target: Ideal DPD output (inverse of PA)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
import h5py
import scipy.io as sio


class DPDDataset(Dataset):
    """
    Dataset for DPD training with thermal augmentation.
    
    Supports:
    - OpenDPD .mat files
    - HDF5 format
    - Numpy .npz files
    
    Args:
        data_path: Path to dataset file or directory
        memory_depth: Number of memory taps for input assembly
        sequence_length: Length of each training sequence
        temp_state: Temperature state ('cold', 'normal', 'hot', 'all')
        normalize: Whether to normalize to [-1, 1]
        augment_noise: Add small noise for regularization
    """
    def __init__(
        self,
        data_path: Union[str, Path],
        memory_depth: int = 5,
        sequence_length: int = 256,
        temp_state: str = 'normal',
        normalize: bool = True,
        augment_noise: float = 0.0
    ):
        self.data_path = Path(data_path)
        self.memory_depth = memory_depth
        self.sequence_length = sequence_length
        self.temp_state = temp_state
        self.normalize = normalize
        self.augment_noise = augment_noise
        
        # Thermal drift parameters
        self.thermal_params = {
            'cold': {
                'gain_drift': -0.02,
                'phase_drift_deg': -5,
                'am_am_scale': 1.05
            },
            'normal': {
                'gain_drift': 0.0,
                'phase_drift_deg': 0,
                'am_am_scale': 1.0
            },
            'hot': {
                'gain_drift': -0.05,
                'phase_drift_deg': 10,
                'am_am_scale': 0.92
            }
        }
        
        # Load data
        self._load_data()
        
        # Prepare sequences
        self._prepare_sequences()
        
    def _load_data(self):
        """Load data from file(s)."""
        if self.data_path.is_dir():
            # Load all .mat or .npz files in directory
            self._load_directory()
        elif self.data_path.suffix == '.mat':
            self._load_mat_file()
        elif self.data_path.suffix == '.npz':
            self._load_npz_file()
        elif self.data_path.suffix in ['.h5', '.hdf5']:
            self._load_hdf5_file()
        else:
            # Generate synthetic data for testing
            print(f"Warning: Data file not found at {self.data_path}")
            print("Generating synthetic data for testing...")
            self._generate_synthetic_data()
            
    def _load_mat_file(self):
        """Load MATLAB .mat file (OpenDPD format)."""
        try:
            # Try scipy.io first (for older .mat files)
            data = sio.loadmat(str(self.data_path))
            
            # OpenDPD format: 'input' and 'output' fields
            if 'input' in data:
                pa_input = data['input'].flatten()
            elif 'x' in data:
                pa_input = data['x'].flatten()
            else:
                raise KeyError("Could not find input data in .mat file")
                
            if 'output' in data:
                pa_output = data['output'].flatten()
            elif 'y' in data:
                pa_output = data['y'].flatten()
            else:
                raise KeyError("Could not find output data in .mat file")
                
        except NotImplementedError:
            # Use mat73 for MATLAB v7.3 files
            import mat73
            data = mat73.loadmat(str(self.data_path))
            pa_input = np.array(data['input']).flatten()
            pa_output = np.array(data['output']).flatten()
            
        self.pa_input_raw = pa_input
        self.pa_output_raw = pa_output
        
    def _load_npz_file(self):
        """Load numpy .npz file."""
        data = np.load(str(self.data_path))
        self.pa_input_raw = data['pa_input']
        self.pa_output_raw = data['pa_output']
        
    def _load_hdf5_file(self):
        """Load HDF5 file."""
        with h5py.File(str(self.data_path), 'r') as f:
            self.pa_input_raw = f['pa_input'][:]
            self.pa_output_raw = f['pa_output'][:]
            
    def _load_directory(self):
        """Load all data files in directory."""
        pa_inputs = []
        pa_outputs = []
        
        for mat_file in sorted(self.data_path.glob('*.mat')):
            try:
                data = sio.loadmat(str(mat_file))
                if 'input' in data and 'output' in data:
                    pa_inputs.append(data['input'].flatten())
                    pa_outputs.append(data['output'].flatten())
            except Exception as e:
                print(f"Warning: Could not load {mat_file}: {e}")
                
        if pa_inputs:
            self.pa_input_raw = np.concatenate(pa_inputs)
            self.pa_output_raw = np.concatenate(pa_outputs)
        else:
            self._generate_synthetic_data()
            
    def _generate_synthetic_data(self):
        """Generate synthetic PA data for testing."""
        num_samples = 100000
        
        # Generate random OFDM-like signal
        np.random.seed(42)
        
        # Complex baseband signal
        i_data = np.random.randn(num_samples) * 0.3
        q_data = np.random.randn(num_samples) * 0.3
        pa_input = i_data + 1j * q_data
        
        # Simulate PA nonlinearity (Saleh model)
        amp = np.abs(pa_input)
        phase = np.angle(pa_input)
        
        # AM-AM: saturating gain
        alpha_a, beta_a = 2.0, 1.0
        gain = alpha_a * amp / (1 + beta_a * amp**2)
        
        # AM-PM: phase distortion
        alpha_p, beta_p = 0.5, 1.0
        phase_dist = alpha_p * amp**2 / (1 + beta_p * amp**2)
        
        pa_output = gain * np.exp(1j * (phase + phase_dist))
        
        # Add memory effect (simple IIR)
        memory_coeff = 0.1
        for i in range(1, len(pa_output)):
            pa_output[i] += memory_coeff * pa_output[i-1]
            
        self.pa_input_raw = pa_input
        self.pa_output_raw = pa_output
        
    def _apply_thermal_drift(self, data: np.ndarray, temp_state: str) -> np.ndarray:
        """Apply thermal drift to PA output data."""
        params = self.thermal_params[temp_state]
        
        # Convert to complex if needed
        if np.isrealobj(data):
            data_complex = data
        else:
            data_complex = data
            
        # Apply gain drift
        gain = 1.0 + params['gain_drift']
        
        # Apply phase drift
        phase_rad = params['phase_drift_deg'] * np.pi / 180
        phase_rotation = np.exp(1j * phase_rad)
        
        # Apply AM-AM scaling (affects nonlinearity)
        amp = np.abs(data_complex)
        phase = np.angle(data_complex)
        
        # Scale amplitude nonlinearity
        amp_scaled = amp * params['am_am_scale']
        
        # Reconstruct
        drifted = amp_scaled * np.exp(1j * phase) * gain * phase_rotation
        
        return drifted
        
    def _prepare_sequences(self):
        """Prepare training sequences with memory taps."""
        M = self.memory_depth
        L = self.sequence_length
        
        # Determine which temperature states to include
        if self.temp_state == 'all':
            temp_states = ['cold', 'normal', 'hot']
        else:
            temp_states = [self.temp_state]
            
        all_sequences = []
        all_labels = []
        all_temps = []
        
        for temp in temp_states:
            # Apply thermal drift to PA output
            pa_output = self._apply_thermal_drift(self.pa_output_raw, temp)
            pa_input = self.pa_input_raw.copy()
            
            # Normalize
            if self.normalize:
                input_max = np.abs(pa_input).max()
                output_max = np.abs(pa_output).max()
                scale = max(input_max, output_max)
                pa_input = pa_input / scale
                pa_output = pa_output / scale
                
            # Convert to IQ format [N, 2]
            input_iq = np.stack([pa_input.real, pa_input.imag], axis=-1)
            output_iq = np.stack([pa_output.real, pa_output.imag], axis=-1)
            
            # Create sequences (with memory overlap)
            num_sequences = (len(pa_input) - M - L) // (L // 2)  # 50% overlap
            
            for i in range(num_sequences):
                start_idx = i * (L // 2)
                end_idx = start_idx + L + M
                
                if end_idx > len(pa_input):
                    break
                    
                # Input sequence (includes memory taps)
                seq_input = input_iq[start_idx:end_idx]
                
                # Output sequence (target for DPD is PA input, given PA output)
                # DPD should produce what goes into PA to get desired output
                seq_output = input_iq[start_idx + M:end_idx]
                
                all_sequences.append((seq_input, seq_output))
                all_temps.append(temp)
                
        self.sequences = all_sequences
        self.temp_labels = all_temps
        
        print(f"Prepared {len(self.sequences)} sequences for training")
        print(f"  Temperature states: {set(self.temp_labels)}")
        print(f"  Sequence length: {L} + {M} memory taps")
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dictionary with:
            - 'input': Input IQ sequence with memory [seq_len + M, 2]
            - 'target': Target DPD output [seq_len, 2]
            - 'temp_state': Temperature state index (0=cold, 1=normal, 2=hot)
        """
        seq_input, seq_target = self.sequences[idx]
        temp = self.temp_labels[idx]
        
        # Convert to tensors
        input_tensor = torch.from_numpy(seq_input.astype(np.float32))
        target_tensor = torch.from_numpy(seq_target.astype(np.float32))
        
        # Add noise augmentation
        if self.augment_noise > 0 and self.training:
            input_tensor += torch.randn_like(input_tensor) * self.augment_noise
            
        # Temperature state index
        temp_idx = {'cold': 0, 'normal': 1, 'hot': 2}[temp]
        
        return {
            'input': input_tensor,
            'target': target_tensor,
            'temp_state': torch.tensor(temp_idx, dtype=torch.long)
        }
    
    @property
    def training(self) -> bool:
        """Check if in training mode (for augmentation)."""
        return getattr(self, '_training', True)
    
    def train(self):
        """Set to training mode."""
        self._training = True
        
    def eval(self):
        """Set to evaluation mode."""
        self._training = False


class SyntheticDPDDataset(Dataset):
    """
    Synthetic dataset using PA digital twin.
    
    Generates training data on-the-fly using the PA model.
    """
    def __init__(
        self,
        pa_model,
        num_samples: int = 10000,
        sequence_length: int = 256,
        memory_depth: int = 5,
        signal_type: str = 'ofdm'
    ):
        self.pa_model = pa_model
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.memory_depth = memory_depth
        self.signal_type = signal_type
        
    def __len__(self) -> int:
        return self.num_samples
    
    def _generate_signal(self) -> torch.Tensor:
        """Generate random input signal."""
        L = self.sequence_length + self.memory_depth
        
        if self.signal_type == 'ofdm':
            # OFDM-like signal (sum of complex exponentials)
            num_subcarriers = 64
            freqs = torch.randn(num_subcarriers) * 0.1
            phases = torch.rand(num_subcarriers) * 2 * np.pi
            
            t = torch.arange(L).float() / L
            signal_i = torch.zeros(L)
            signal_q = torch.zeros(L)
            
            for f, p in zip(freqs, phases):
                signal_i += torch.cos(2 * np.pi * f * t + p)
                signal_q += torch.sin(2 * np.pi * f * t + p)
                
            # Normalize
            signal_i = signal_i / signal_i.abs().max() * 0.7
            signal_q = signal_q / signal_q.abs().max() * 0.7
            
        else:
            # Random Gaussian
            signal_i = torch.randn(L) * 0.3
            signal_q = torch.randn(L) * 0.3
            
        return torch.stack([signal_i, signal_q], dim=-1)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Generate input signal
        input_signal = self._generate_signal()
        
        # Pass through PA model
        with torch.no_grad():
            pa_output = self.pa_model(input_signal.unsqueeze(0))
            
        pa_output = pa_output.squeeze(0)
        
        # Target is the input (DPD should invert PA)
        target = input_signal[self.memory_depth:]
        
        return {
            'input': input_signal,
            'target': target,
            'pa_output': pa_output,
            'temp_state': torch.tensor(1, dtype=torch.long)  # Normal
        }


def create_dataloader(
    config: dict,
    split: str = 'train',
    batch_size: Optional[int] = None
) -> DataLoader:
    """
    Create DataLoader from config.
    
    Args:
        config: Configuration dictionary
        split: 'train', 'val', or 'test'
        batch_size: Override batch size (optional)
        
    Returns:
        DataLoader instance
    """
    data_config = config.get('data', {})
    model_config = config.get('model', {}).get('generator', {})
    
    # Dataset path
    if split == 'train':
        data_path = Path(data_config.get('dataset_path', 'data/raw/'))
    else:
        data_path = Path(data_config.get('processed_path', 'data/processed/'))
        
    # Create dataset
    dataset = DPDDataset(
        data_path=data_path,
        memory_depth=model_config.get('memory_depth', 5),
        sequence_length=256,
        temp_state='all' if split == 'train' else 'normal',
        normalize=data_config.get('normalize', True),
        augment_noise=0.01 if split == 'train' else 0.0
    )
    
    # Set mode
    if split == 'train':
        dataset.train()
    else:
        dataset.eval()
        
    # Batch size
    if batch_size is None:
        batch_size = config.get('training', {}).get('batch_size', 64)
        
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return loader


if __name__ == "__main__":
    print("Testing Dataset Utilities")
    print("=" * 50)
    
    # Test with synthetic data
    dataset = DPDDataset(
        data_path=Path('/tmp/nonexistent'),  # Will generate synthetic
        memory_depth=5,
        sequence_length=128,
        temp_state='all',
        normalize=True
    )
    
    print(f"\nDataset length: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"  Input: {sample['input'].shape}")
    print(f"  Target: {sample['target'].shape}")
    print(f"  Temp state: {sample['temp_state'].item()}")
    
    # Test DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    
    print(f"\nBatch shapes:")
    print(f"  Input: {batch['input'].shape}")
    print(f"  Target: {batch['target'].shape}")
    print(f"  Temp states: {batch['temp_state'].tolist()}")
