#!/usr/bin/env python3
"""
Train TDNN DPD with thermal drift variants
Simplified version for quick weight generation

Generates 3 weight sets:
- Cold (-20°C): weights_cold_*.hex
- Normal (25°C): weights_normal_*.hex  
- Hot (70°C): weights_hot_*.hex
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

print("="*70)
print("🔥 TDNN DPD Training with Thermal Variants")
print("="*70)

# ============================================================================
# 1. Model Definition (must match RTL exactly)
# ============================================================================
class TDNNGenerator(nn.Module):
    """Time-Delay Neural Network for DPD - matches RTL"""
    def __init__(self, memory_depth=5, hidden1=32, hidden2=16):
        super().__init__()
        self.memory_depth = memory_depth
        self.input_size = 2 + 2 * memory_depth * 2  # I, Q + delayed features
        
        self.fc1 = nn.Linear(self.input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 2)
        self.lrelu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        
    def create_input_features(self, x):
        """Create TDNN input: [I, Q, I_{t-1}, Q_{t-1}, ..., |x|^2*I, |x|^2*Q, ...]"""
        features = [x]  # Current sample
        
        for d in range(1, self.memory_depth + 1):
            if d < x.shape[0]:
                delayed = torch.cat([torch.zeros(d, 2, device=x.device), x[:-d]], dim=0)
            else:
                delayed = torch.zeros_like(x)
            features.append(delayed)
            
            # Add nonlinear terms
            mag_sq = (delayed[:, 0]**2 + delayed[:, 1]**2).unsqueeze(1)
            features.append(mag_sq * delayed)
        
        return torch.cat(features, dim=1)
    
    def forward(self, x):
        x = self.create_input_features(x)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.fc3(x)
        return self.tanh(x)


# ============================================================================
# 2. Load OpenDPD Data
# ============================================================================
def load_data():
    """Load OpenDPD APA_200MHz dataset"""
    data_path = Path('/home/james-patrick/eda/designs/github/OpenDPD/datasets/APA_200MHz')
    
    print("\n📂 Loading OpenDPD APA_200MHz dataset...")
    input_df = pd.read_csv(data_path / 'train_input.csv')
    output_df = pd.read_csv(data_path / 'train_output.csv')
    
    # Convert to complex
    u_pa = (input_df['I'].values + 1j * input_df['Q'].values).astype(np.complex64)
    y_pa = (output_df['I'].values + 1j * output_df['Q'].values).astype(np.complex64)
    
    # Normalize
    max_val = np.max(np.abs(u_pa))
    u_pa = u_pa / max_val * 0.7
    y_pa = y_pa / max_val * 0.7
    
    print(f"   Loaded {len(u_pa):,} samples")
    print(f"   Input power:  {10*np.log10(np.mean(np.abs(u_pa)**2)):.2f} dBFS")
    print(f"   Output power: {10*np.log10(np.mean(np.abs(y_pa)**2)):.2f} dBFS")
    
    return u_pa, y_pa


# ============================================================================
# 3. Apply Thermal Drift
# ============================================================================
def apply_thermal_drift(y_pa, temperature, reference_temp=25.0):
    """
    Model PA thermal drift effects
    
    Physical basis (GaN PA):
    - Gain drift: ~0.5% per 10°C
    - Phase drift: ~0.3° per 10°C  
    - AM/AM compression changes
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


# ============================================================================
# 4. Training Function
# ============================================================================
def train_single_temperature(u_pa, y_pa, temperature, num_epochs=100, device='cpu'):
    """Train TDNN for single temperature condition"""
    
    temp_name = {-20: 'cold', 25: 'normal', 70: 'hot'}[temperature]
    
    print(f"\n{'='*70}")
    print(f"🌡️  Training for {temp_name.upper()} temperature ({temperature}°C)")
    print(f"{'='*70}")
    
    # Apply thermal drift
    y_thermal = apply_thermal_drift(y_pa, temperature)
    
    # Convert to PyTorch tensors (ILA: train on PA output → PA input)
    x_train = torch.from_numpy(np.stack([y_thermal.real, y_thermal.imag], axis=1)).float()
    y_train = torch.from_numpy(np.stack([u_pa.real, u_pa.imag], axis=1)).float()
    
    # Use subset for faster training (can use full dataset for final run)
    subset_size = 50000
    x_train = x_train[:subset_size]
    y_train = y_train[:subset_size]
    
    print(f"   Training samples: {len(x_train):,}")
    
    # Initialize model
    model = TDNNGenerator(memory_depth=5, hidden1=32, hidden2=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Mini-batch training
        batch_size = 512
        num_batches = len(x_train) // batch_size
        epoch_loss = 0
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for i in pbar:
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            x_batch = x_train[start_idx:end_idx].to(device)
            y_batch = y_train[start_idx:end_idx].to(device)
            
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = epoch_loss / num_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d}: Loss = {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    print(f"   ✅ Training complete! Best loss: {best_loss:.6f}")
    
    return model


# ============================================================================
# 5. Export Weights to FPGA Format
# ============================================================================
def export_weights_fpga(model, temp_name, output_dir):
    """
    Export weights in Q1.15 fixed-point format for FPGA
    
    Q1.15 format:
    - 1 sign bit, 15 fractional bits
    - Range: [-1.0, +0.999969]
    - Resolution: 1/32768 ≈ 0.000031
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n   📝 Exporting weights to {output_dir}/")
    
    # Extract layer weights
    fc1_w = model.fc1.weight.detach().cpu().numpy()  # (32, 18)
    fc1_b = model.fc1.bias.detach().cpu().numpy()    # (32,)
    fc2_w = model.fc2.weight.detach().cpu().numpy()  # (16, 32)
    fc2_b = model.fc2.bias.detach().cpu().numpy()    # (16,)
    fc3_w = model.fc3.weight.detach().cpu().numpy()  # (2, 16)
    fc3_b = model.fc3.bias.detach().cpu().numpy()    # (2,)
    
    def to_q1_15(weights):
        """Convert float to Q1.15 fixed-point"""
        # Clip to valid range
        weights_clipped = np.clip(weights, -1.0, 0.999969)
        # Scale and round
        weights_int = np.round(weights_clipped * 32768).astype(np.int16)
        # Convert to unsigned for hex
        weights_hex = weights_int.astype(np.uint16)
        return weights_hex
    
    # Export each layer
    layers = [
        (fc1_w.T, f'{temp_name}_fc1_weights'),  # Transpose for RTL layout
        (fc1_b, f'{temp_name}_fc1_bias'),
        (fc2_w.T, f'{temp_name}_fc2_weights'),
        (fc2_b, f'{temp_name}_fc2_bias'),
        (fc3_w.T, f'{temp_name}_fc3_weights'),
        (fc3_b, f'{temp_name}_fc3_bias'),
    ]
    
    for weights, name in layers:
        weights_flat = weights.flatten()
        weights_hex = to_q1_15(weights_flat)
        
        output_file = output_dir / f'{name}.hex'
        with open(output_file, 'w') as f:
            f.write(f"// {name} - Q1.15 format\n")
            f.write(f"// Shape: {weights.shape}\n")
            f.write(f"// Count: {len(weights_flat)}\n\n")
            
            for addr, val in enumerate(weights_hex):
                f.write(f"{val:04X}\n")
        
        print(f"      ✓ {name}.hex ({len(weights_flat)} values)")
    
    print(f"   ✅ Exported 6 weight files for {temp_name} variant")


# ============================================================================
# 6. Main Training Loop
# ============================================================================
def main():
    device = 'cpu'  # Can change to 'cuda' if GPU available
    
    # Load data
    u_pa, y_pa = load_data()
    
    # Train for each temperature
    temperatures = [-20, 25, 70]
    output_dir = Path('rtl/weights')
    
    for temp in temperatures:
        # Train model
        model = train_single_temperature(u_pa, y_pa, temp, num_epochs=100, device=device)
        
        # Export weights
        temp_name = {-20: 'cold', 25: 'normal', 70: 'hot'}[temp]
        export_weights_fpga(model, temp_name, output_dir)
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"\n✅ Generated 18 weight files in {output_dir}/")
    print("\nWeight files for FPGA BRAM:")
    print("  Cold variant (-20°C):")
    print("    - cold_fc1_weights.hex (576 values)")
    print("    - cold_fc1_bias.hex (32 values)")
    print("    - cold_fc2_weights.hex (512 values)")
    print("    - cold_fc2_bias.hex (16 values)")
    print("    - cold_fc3_weights.hex (32 values)")
    print("    - cold_fc3_bias.hex (2 values)")
    print("\n  Normal variant (25°C): (same structure)")
    print("  Hot variant (70°C): (same structure)")
    print("\n📋 Next steps:")
    print("  1. ✅ Weights exported to rtl/weights/")
    print("  2. Update tdnn_generator.v to load from hex files")
    print("  3. Add temperature sensor interface for bank switching")
    print("  4. Run Vivado synthesis for PYNQ-Z1")
    print("  5. Test on hardware with HDMI demo")
    print()


if __name__ == '__main__':
    main()
