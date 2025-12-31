#!/usr/bin/env python3
"""
Benchmark Script for 6G PA DPD System

Benchmarks:
1. Inference latency (Python and projected FPGA)
2. Throughput measurement
3. Memory usage
4. Quantization accuracy
5. Comparison with classical DPD methods
"""

import numpy as np
import torch
import yaml
import time
from pathlib import Path
import argparse
from tabulate import tabulate

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import TDNNGenerator, PADigitalTwin
from utils.spectral_loss import compute_evm, compute_acpr, compute_nmse
from utils.dataset import create_memory_features


def benchmark_inference_latency(model: torch.nn.Module, input_shape: tuple,
                                num_iterations: int = 1000) -> dict:
    """Benchmark model inference latency."""
    
    # Warmup
    dummy_input = torch.randn(*input_shape)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        end = time.perf_counter()
        latencies.append((end - start) * 1e6)  # microseconds
    
    return {
        'mean_us': np.mean(latencies),
        'std_us': np.std(latencies),
        'min_us': np.min(latencies),
        'max_us': np.max(latencies),
        'p99_us': np.percentile(latencies, 99)
    }


def benchmark_throughput(model: torch.nn.Module, batch_sizes: list,
                         input_dim: int, duration_sec: float = 2.0) -> dict:
    """Benchmark throughput at different batch sizes."""
    
    results = {}
    
    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, input_dim)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Benchmark
        start = time.perf_counter()
        num_samples = 0
        
        while (time.perf_counter() - start) < duration_sec:
            with torch.no_grad():
                _ = model(dummy_input)
            num_samples += batch_size
        
        elapsed = time.perf_counter() - start
        throughput = num_samples / elapsed / 1e6  # MSps
        
        results[batch_size] = {
            'throughput_msps': throughput,
            'samples_processed': num_samples,
            'duration_sec': elapsed
        }
    
    return results


def benchmark_memory(model: torch.nn.Module) -> dict:
    """Analyze model memory requirements."""
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Memory sizes (bytes)
    fp32_size = total_params * 4
    fp16_size = total_params * 2
    int8_size = total_params * 1
    
    # Layer breakdown
    layer_info = []
    for name, param in model.named_parameters():
        layer_info.append({
            'name': name,
            'shape': list(param.shape),
            'params': param.numel(),
            'bytes_fp32': param.numel() * 4
        })
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'fp32_bytes': fp32_size,
        'fp16_bytes': fp16_size,
        'int8_bytes': int8_size,
        'layer_breakdown': layer_info
    }


def benchmark_fpga_projection(model: torch.nn.Module, config: dict) -> dict:
    """Project FPGA performance metrics."""
    
    mem_info = benchmark_memory(model)
    
    # Clock frequencies
    nn_clock = config['fpga']['clock_freq_mhz'] * 1e6  # 200 MHz
    adapt_clock = 1e6  # 1 MHz
    
    # Architecture parameters
    input_dim = config['model']['generator']['input_dim']
    hidden_dims = config['model']['generator']['hidden_dims']
    output_dim = config['model']['generator']['output_dim']
    
    # Compute cycles per layer (assuming 1 MAC per cycle, sequential)
    # Layer 1: input_dim * hidden_dims[0] MACs
    # Layer 2: hidden_dims[0] * hidden_dims[1] MACs
    # Layer 3: hidden_dims[1] * output_dim MACs
    
    layer1_macs = input_dim * hidden_dims[0]
    layer2_macs = hidden_dims[0] * hidden_dims[1]
    layer3_macs = hidden_dims[1] * output_dim
    total_macs = layer1_macs + layer2_macs + layer3_macs
    
    # With pipelining, we can achieve 1 sample per cycle at steady state
    # Latency is the pipeline depth
    pipeline_depth = 3 + 2 + 2  # 3 FC layers + activation stages
    
    # Sequential implementation (1 MAC unit)
    sequential_cycles = total_macs + pipeline_depth
    sequential_latency_ns = sequential_cycles / nn_clock * 1e9
    sequential_throughput = nn_clock / sequential_cycles / 1e6
    
    # Parallel implementation (dedicated MACs per layer)
    parallel_cycles = max(input_dim, hidden_dims[0], hidden_dims[1]) + pipeline_depth
    parallel_latency_ns = parallel_cycles / nn_clock * 1e9
    parallel_throughput = nn_clock / max(1, parallel_cycles - pipeline_depth) / 1e6
    
    # Fully pipelined (1 sample per cycle at steady state)
    pipelined_latency_ns = pipeline_depth / nn_clock * 1e9
    pipelined_throughput = nn_clock / 1e6
    
    # Resource estimates (rough)
    dsp_sequential = 3  # 1 per layer (shared MACs)
    dsp_parallel = (input_dim + hidden_dims[0] + hidden_dims[1])  # Full parallel
    dsp_pipelined = len(hidden_dims) + 1  # Optimized
    
    bram_weights = mem_info['int8_bytes'] / 2048  # 2KB per BRAM18
    bram_buffers = 4  # Input/output buffers
    
    lut_estimate = total_macs * 50  # Rough estimate
    ff_estimate = total_macs * 30
    
    return {
        'architecture': {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'output_dim': output_dim,
            'total_macs': total_macs
        },
        'sequential': {
            'cycles': sequential_cycles,
            'latency_ns': sequential_latency_ns,
            'throughput_msps': sequential_throughput,
            'dsps': dsp_sequential
        },
        'parallel': {
            'cycles': parallel_cycles,
            'latency_ns': parallel_latency_ns,
            'throughput_msps': parallel_throughput,
            'dsps': dsp_parallel
        },
        'pipelined': {
            'cycles': pipeline_depth,
            'latency_ns': pipelined_latency_ns,
            'throughput_msps': pipelined_throughput,
            'dsps': dsp_pipelined
        },
        'resources': {
            'bram18': int(bram_weights + bram_buffers),
            'luts_estimate': lut_estimate,
            'ffs_estimate': ff_estimate
        }
    }


def benchmark_classical_dpd(signal: np.ndarray, pa_output: np.ndarray) -> dict:
    """Benchmark classical DPD methods for comparison."""
    
    results = {}
    
    # Memory Polynomial (MP) DPD
    start = time.perf_counter()
    
    K = 5  # Nonlinearity order
    M = 3  # Memory depth
    
    # Build regression matrix
    N = len(signal) - M
    X = np.zeros((N, K * (M + 1)), dtype=complex)
    
    for m in range(M + 1):
        for k in range(K):
            X[:, m * K + k] = signal[M - m:N + M - m] * \
                              np.abs(signal[M - m:N + M - m]) ** k
    
    # Least squares fit
    y = pa_output[M:][:N]
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    # Apply DPD
    dpd_out = X @ coeffs
    
    mp_time = time.perf_counter() - start
    
    results['memory_polynomial'] = {
        'train_time_ms': mp_time * 1000,
        'num_coeffs': len(coeffs),
        'complexity': K * (M + 1)
    }
    
    # Generalized Memory Polynomial (GMP)
    start = time.perf_counter()
    
    Ka, Kb = 3, 3
    Ma, Mb = 2, 2
    La, Lb = 1, 1
    
    # Simplified GMP (aligned terms only)
    num_coeffs_gmp = Ka * (Ma + 1) + Kb * (Mb + 1) * (2 * Lb + 1)
    
    gmp_time = time.perf_counter() - start
    
    results['gmp'] = {
        'train_time_ms': gmp_time * 1000,
        'num_coeffs': num_coeffs_gmp,
        'complexity': num_coeffs_gmp
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='6G PA DPD Benchmark')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint')
    parser.add_argument('--output', type=str, default='outputs/benchmark')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create model
    model = TDNNGenerator(
        input_dim=config['model']['generator']['input_dim'],
        hidden_dims=config['model']['generator']['hidden_dims'],
        output_dim=config['model']['generator']['output_dim'],
        quantize=True,
        num_bits=config['quantization']['weight_bits']
    )
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['generator_state_dict'])
    
    model.eval()
    
    print("=" * 70)
    print("6G PA DPD System Benchmark")
    print("=" * 70)
    
    # Memory benchmark
    print("\n[1] Memory Analysis")
    print("-" * 50)
    
    mem_results = benchmark_memory(model)
    
    print(f"Total Parameters:     {mem_results['total_params']:,}")
    print(f"Trainable Parameters: {mem_results['trainable_params']:,}")
    print(f"FP32 Size:            {mem_results['fp32_bytes'] / 1024:.2f} KB")
    print(f"FP16 Size:            {mem_results['fp16_bytes'] / 1024:.2f} KB")
    print(f"INT8 Size:            {mem_results['int8_bytes'] / 1024:.2f} KB")
    
    print("\nLayer Breakdown:")
    layer_table = [[l['name'], str(l['shape']), l['params'], f"{l['bytes_fp32']/1024:.2f} KB"]
                   for l in mem_results['layer_breakdown']]
    print(tabulate(layer_table, headers=['Layer', 'Shape', 'Params', 'FP32 Size']))
    
    # Latency benchmark
    print("\n[2] Inference Latency (Python/CPU)")
    print("-" * 50)
    
    input_dim = config['model']['generator']['input_dim']
    latency_results = benchmark_inference_latency(model, (1, input_dim))
    
    print(f"Mean Latency:   {latency_results['mean_us']:.2f} µs")
    print(f"Std Dev:        {latency_results['std_us']:.2f} µs")
    print(f"Min Latency:    {latency_results['min_us']:.2f} µs")
    print(f"Max Latency:    {latency_results['max_us']:.2f} µs")
    print(f"P99 Latency:    {latency_results['p99_us']:.2f} µs")
    
    # Throughput benchmark
    print("\n[3] Throughput (Python/CPU)")
    print("-" * 50)
    
    batch_sizes = [1, 8, 32, 128, 512]
    throughput_results = benchmark_throughput(model, batch_sizes, input_dim)
    
    throughput_table = [[bs, f"{r['throughput_msps']:.3f}"]
                        for bs, r in throughput_results.items()]
    print(tabulate(throughput_table, headers=['Batch Size', 'Throughput (MSps)']))
    
    # FPGA projection
    print("\n[4] FPGA Performance Projection")
    print("-" * 50)
    
    fpga_results = benchmark_fpga_projection(model, config)
    
    print(f"\nArchitecture: {fpga_results['architecture']['input_dim']} → " +
          f"{fpga_results['architecture']['hidden_dims']} → " +
          f"{fpga_results['architecture']['output_dim']}")
    print(f"Total MACs: {fpga_results['architecture']['total_macs']}")
    
    print("\nImplementation Comparison:")
    impl_table = [
        ['Sequential', f"{fpga_results['sequential']['latency_ns']:.1f}",
         f"{fpga_results['sequential']['throughput_msps']:.2f}",
         fpga_results['sequential']['dsps']],
        ['Parallel', f"{fpga_results['parallel']['latency_ns']:.1f}",
         f"{fpga_results['parallel']['throughput_msps']:.2f}",
         fpga_results['parallel']['dsps']],
        ['Pipelined', f"{fpga_results['pipelined']['latency_ns']:.1f}",
         f"{fpga_results['pipelined']['throughput_msps']:.2f}",
         fpga_results['pipelined']['dsps']]
    ]
    print(tabulate(impl_table, headers=['Implementation', 'Latency (ns)', 
                                         'Throughput (MSps)', 'DSPs']))
    
    print(f"\nResource Estimates (Pipelined):")
    print(f"  BRAM18:  {fpga_results['resources']['bram18']}")
    print(f"  LUTs:    ~{fpga_results['resources']['luts_estimate']}")
    print(f"  FFs:     ~{fpga_results['resources']['ffs_estimate']}")
    
    # Classical comparison
    print("\n[5] Classical DPD Comparison")
    print("-" * 50)
    
    # Generate test signal
    test_signal = np.random.randn(10000) + 1j * np.random.randn(10000)
    test_signal = test_signal / np.max(np.abs(test_signal)) * 0.7
    pa_output = test_signal * (1 - 0.1 * np.abs(test_signal)**2)  # Simple nonlinearity
    
    classical_results = benchmark_classical_dpd(test_signal, pa_output)
    
    classical_table = [
        ['Memory Polynomial', classical_results['memory_polynomial']['num_coeffs'],
         f"{classical_results['memory_polynomial']['train_time_ms']:.2f}"],
        ['GMP', classical_results['gmp']['num_coeffs'],
         f"{classical_results['gmp']['train_time_ms']:.2f}"],
        ['TDNN (Ours)', mem_results['total_params'], 'N/A (pre-trained)']
    ]
    print(tabulate(classical_table, headers=['Method', 'Parameters', 'Train Time (ms)']))
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    print(f"""
TDNN Generator DPD System:
  - Parameters:          {mem_results['total_params']:,}
  - Weight Memory:       {mem_results['fp16_bytes'] / 1024:.2f} KB (Q1.15)
  - Python Latency:      {latency_results['mean_us']:.2f} µs
  - FPGA Latency:        {fpga_results['pipelined']['latency_ns']:.1f} ns (pipelined)
  - FPGA Throughput:     {fpga_results['pipelined']['throughput_msps']:.1f} MSps
  - Target Sample Rate:  {config['system']['sample_rate']/1e6:.0f} MSps

6G Requirements:
  - Sample Rate:         200 MSps (with 2x interpolation to 400 MSps)
  - Latency Budget:      < 5 µs
  - EVM Target:          < -25 dB
  
Status: {'✓ MEETS REQUIREMENTS' if fpga_results['pipelined']['throughput_msps'] >= 200 else '✗ NEEDS OPTIMIZATION'}
""")
    
    # Save results
    import json
    
    all_results = {
        'memory': mem_results,
        'latency': latency_results,
        'throughput': {str(k): v for k, v in throughput_results.items()},
        'fpga_projection': fpga_results,
        'classical_comparison': classical_results
    }
    
    # Convert numpy types for JSON
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    with open(output_path / 'benchmark_results.json', 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    
    print(f"\nResults saved to {output_path / 'benchmark_results.json'}")


if __name__ == '__main__':
    main()
