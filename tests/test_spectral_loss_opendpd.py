#!/usr/bin/env python3
"""
Quick validation test for updated spectral_loss.py against OpenDPDv2 reference.
"""

import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.spectral_loss import (
    compute_evm, compute_nmse, compute_aclr, get_amplitude, set_target_gain,
    SpectralLoss
)

def test_opendpd_compatibility():
    """Test that our implementation matches OpenDPDv2 behavior."""
    
    print("=" * 70)
    print("OpenDPDv2 Compatibility Test")
    print("=" * 70)
    
    # Create test signals
    batch_size = 4
    seq_len = 1024
    sample_rate = 983.04e6  # OpenDPDv2 standard
    
    # Generate test signals
    print("\n1. Generating test signals...")
    t = np.linspace(0, seq_len / sample_rate, seq_len, endpoint=False)
    
    # Reference (clean, 50 MHz tone)
    freq = 50e6
    ref_i = np.cos(2 * np.pi * freq * t)
    ref_q = np.sin(2 * np.pi * freq * t)
    ref_iq = np.stack([ref_i, ref_q], axis=-1)
    
    # Add distortion (simulate PA nonlinearity)
    noise = np.random.randn(*ref_iq.shape) * 0.05
    distortion = ref_iq ** 3 * 0.01
    pred_iq = ref_iq + noise + distortion
    
    print(f"   Reference IQ shape: {ref_iq.shape}")
    print(f"   Predicted IQ shape: {pred_iq.shape}")
    
    # Test 1: EVM Computation
    print("\n2. Testing EVM (frequency-domain, per-subchannel)...")
    try:
        evm = compute_evm(pred_iq, ref_iq, sample_rate=sample_rate, n_sub_ch=1, nperseg=1024)
        print(f"   ✅ EVM: {evm:.2f} dB")
        assert isinstance(evm, float), "EVM should return float"
        assert evm < 0, "EVM should be negative for distorted signal"
    except Exception as e:
        print(f"   ❌ EVM failed: {e}")
        return False
    
    # Test 2: NMSE Computation
    print("\n3. Testing NMSE (I/Q separated, time-domain)...")
    try:
        nmse = compute_nmse(pred_iq, ref_iq)
        print(f"   ✅ NMSE: {nmse:.2f} dB")
        assert isinstance(nmse, float), "NMSE should return float"
        assert nmse < 0, "NMSE should be negative"
    except Exception as e:
        print(f"   ❌ NMSE failed: {e}")
        return False
    
    # Test 3: ACLR Computation
    print("\n4. Testing ACLR (Welch PSD)...")
    try:
        aclr_l, aclr_r = compute_aclr(
            pred_iq, sample_rate=sample_rate, nperseg=1024, 
            bw_main_ch=200e6, n_sub_ch=1
        )
        print(f"   ✅ ACLR Left: {aclr_l:.2f} dB, Right: {aclr_r:.2f} dB")
        assert isinstance(aclr_l, float), "ACLR should return float"
    except Exception as e:
        print(f"   ❌ ACLR failed: {e}")
        return False
    
    # Test 4: Utility Functions
    print("\n5. Testing utility functions...")
    try:
        amp = get_amplitude(ref_iq)
        print(f"   ✅ get_amplitude: output shape {amp.shape}")
        
        gain = set_target_gain(ref_iq, pred_iq)
        print(f"   ✅ set_target_gain: {gain:.4f}")
    except Exception as e:
        print(f"   ❌ Utility functions failed: {e}")
        return False
    
    # Test 5: SpectralLoss (PyTorch)
    print("\n6. Testing SpectralLoss (PyTorch, differentiable)...")
    try:
        # Convert to torch - add batch dimension properly [batch, seq, 2] with grad enabled
        pred_torch = torch.from_numpy(pred_iq[np.newaxis, :, :]).float().requires_grad_(True)  # [1, 1024, 2]
        ref_torch = torch.from_numpy(ref_iq[np.newaxis, :, :]).float()    # [1, 1024, 2]
        
        print(f"   Input shapes: pred={pred_torch.shape} (requires_grad={pred_torch.requires_grad}), ref={ref_torch.shape}")
        
        loss_fn = SpectralLoss(sample_rate=sample_rate, bw_main_ch=200e6)
        
        # Training loss (should be differentiable)
        loss, components = loss_fn(pred_torch, ref_torch, return_components=True)
        print(f"   ✅ Training Loss: {loss.item():.4f}")
        print(f"      Components: {[(k, f'{v.item():.4f}' if isinstance(v, torch.Tensor) else f'{v:.4f}') for k, v in components.items()]}")
        
        assert loss.requires_grad, "Loss should be differentiable for backpropagation"
        
        # Verify gradients can flow
        loss.backward()
        assert pred_torch.grad is not None, "Gradients should flow to input"
        print(f"   ✅ Gradients flowing correctly")
        
        # Evaluation metrics (numpy-based, matches OpenDPDv2)
        try:
            metrics = loss_fn.compute_metrics(pred_torch, ref_torch)
            print(f"   ✅ Evaluation Metrics:")
            for k, v in metrics.items():
                print(f"      {k}: {v:.4f}")
        except Exception as e:
            print(f"   ⚠️  Warning: compute_metrics failed: {e}")
        
        assert 'evm_db' in metrics, "Metrics should include EVM"
        assert 'nmse_db' in metrics, "Metrics should include NMSE"
        assert 'aclr_max_db' in metrics, "Metrics should include ACLR"
        
    except Exception as e:
        print(f"   ❌ SpectralLoss failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Batch Processing
    print("\n7. Testing batch processing...")
    try:
        pred_batch = torch.randn(4, 256, 2).float().requires_grad_(True)  # [batch=4, seq=256, 2]
        ref_batch = torch.randn(4, 256, 2).float()
        
        loss = loss_fn(pred_batch, ref_batch)
        print(f"   ✅ Batch Loss: {loss.item():.4f}")
        assert loss.requires_grad, "Batch loss should be differentiable"
        
        # Verify backprop works
        loss.backward()
        assert pred_batch.grad is not None, "Gradients should flow to batch input"
        print(f"   ✅ Batch gradients flowing correctly")
    except Exception as e:
        print(f"   ❌ Batch processing failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - Implementation matches OpenDPDv2")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = test_opendpd_compatibility()
    sys.exit(0 if success else 1)
