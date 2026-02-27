#!/usr/bin/env python3
"""
Module-Level Resource Estimator for DPD SSR Architecture
Target: ZCU104 (XCZU7EV-2FFVC1156)

Estimates LUT, FF, DSP, BRAM for each module:
- interpolator1_5_ssr: 5-phase polyphase interpolator
- fex_layer_synth: Feature extraction (magnitude, powers, phase norm)
- tdnn_generator: PN-TDNN neural network (FC1→FC2→FC3)
- weight_rom: Weight storage (BRAM-based)
- dpd_top_ssr: Top-level with 5 parallel lanes

Based on:
- Xilinx UG579 (UltraScale DSP48E2 User Guide)
- Xilinx UG573 (UltraScale Memory Resources)
- First-principles counting from RTL

Author: DPD Project
Date: January 2026
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# =============================================================================
# ZCU104 Device Specifications
# =============================================================================

@dataclass
class ZCU104:
    """XCZU7EV-2FFVC1156 specifications."""
    name: str = "ZCU104 (XCZU7EV-2FFVC1156)"
    dsp_count: int = 1728
    lut_count: int = 230400
    ff_count: int = 460800
    bram_36kb: int = 312
    uram_288kb: int = 96
    
# =============================================================================
# Weight Analysis from Hex Files
# =============================================================================

def count_hex_weights(filepath: str) -> int:
    """Count number of 16-bit weights in a hex file."""
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    return len(lines)

def analyze_weight_files(weight_dir: str) -> Dict[str, int]:
    """Analyze actual weight files to get parameter counts."""
    weights = {}
    
    # Expected files based on attachments
    files = {
        'fc1_weight': 'fc_layers_0_weight.hex',
        'fc1_bias': 'fc_layers_0_bias.hex',
        'fc2_weight': 'fc_layers_2_weight.hex',
        'fc2_bias': 'fc_layers_2_bias.hex',
        'fc3_weight': 'fc_layers_4_weight.hex',
        'fc3_bias': 'fc_layers_4_bias.hex',
    }
    
    for key, filename in files.items():
        path = os.path.join(weight_dir, filename)
        weights[key] = count_hex_weights(path)
    
    return weights

# =============================================================================
# Module Resource Estimates
# =============================================================================

@dataclass
class ModuleResources:
    """Resource usage for a single module."""
    name: str
    lut: int
    ff: int
    dsp: int
    bram36: int
    description: str = ""
    
    def __str__(self) -> str:
        return f"{self.name:25s} | {self.lut:6d} | {self.ff:6d} | {self.dsp:4d} | {self.bram36:4d}"

def estimate_interpolator_resources(num_phases: int = 5, taps_per_phase: int = 12) -> ModuleResources:
    """
    Estimate resources for interpolator1_5_ssr.
    
    Architecture:
    - 5 phases × 12 taps = 60 filter coefficients
    - Each tap: 1 DSP48E2 for multiply-accumulate (I and Q)
    - Coefficient storage: Distributed ROM (LUTs)
    - Pipeline registers for timing closure
    
    DSP Usage:
    - 60 taps × 2 (I/Q) = 120 DSP48E2 for full parallel
    - With resource sharing: 60 DSP (time-multiplexed I/Q)
    
    We use 5 parallel MAC engines per phase, computing all 12 taps.
    """
    total_taps = num_phases * taps_per_phase  # 60
    
    # DSP: Each phase needs taps_per_phase MACs for I and Q
    # Full parallel: 60 × 2 = 120 DSP
    # Time-multiplexed I/Q: 60 DSP
    # Resource-optimized (2 MAC/phase, sequential): 5 × 2 × 2 = 20 DSP
    # Current design: 12-tap FIR × 5 phases, shared between I/Q = 60 DSP
    dsp_count = total_taps  # 60 DSP for parallel implementation
    
    # LUTs:
    # - Coefficient ROM: 60 × 16-bit = 960 bits, packed in LUTs (~60 LUTs)
    # - Delay line control: 5 phases × 12 taps × mux = ~200 LUTs
    # - Accumulator logic: 5 × ~50 = 250 LUTs
    # - Pipeline control: ~100 LUTs
    lut_count = 60 + 200 + 250 + 100  # ~610 LUTs
    
    # FFs:
    # - Delay line: 12 samples × 16-bit × 2 (I/Q) = 384 FF
    # - Coefficient pipeline: 60 × 16-bit = 960 FF
    # - Accumulator registers: 5 × 40-bit × 2 = 400 FF
    # - Output registers: 5 × 16-bit × 2 = 160 FF
    # - Control: ~50 FF
    ff_count = 384 + 960 + 400 + 160 + 50  # ~1954 FF
    
    # BRAM: None - coefficients fit in distributed ROM
    bram_count = 0
    
    return ModuleResources(
        name="interpolator1_5_ssr",
        lut=lut_count,
        ff=ff_count,
        dsp=dsp_count,
        bram36=bram_count,
        description="5-phase polyphase interpolator (1:5), 12 taps/phase"
    )

def estimate_fex_resources(memory_depth: int = 4) -> ModuleResources:
    """
    Estimate resources for fex_layer_synth.
    
    Architecture:
    - Alpha-max-beta-min magnitude approximation (no CORDIC)
    - A², A³ computation
    - Phase normalization (4 taps)
    - 9-stage pipeline
    
    Features per tap: A, A², A³, I_norm, Q_norm, I, Q (but we use 24-dim = 6 × 4)
    """
    # DSP Usage:
    # - Magnitude: 0 (alpha-max-beta-min uses comparators)
    # - A² computation: 1 DSP
    # - A³ computation: 1 DSP  
    # - Phase norm (I/mag, Q/mag): 2 DSP for division approximation
    # Per tap: 4 DSP, but shared across 4 taps = 4 DSP total
    dsp_count = 4
    
    # LUTs:
    # - Magnitude approximation (comparators, mux): ~100 LUTs
    # - Memory delay line control: 4 × ~30 = 120 LUTs
    # - Division/normalization: ~200 LUTs (reciprocal LUT + multiply)
    # - Output mux and packing: ~150 LUTs
    # - Pipeline control: ~80 LUTs
    lut_count = 100 + 120 + 200 + 150 + 80  # ~650 LUTs
    
    # FFs:
    # - Memory buffer: 4 taps × 16-bit × 2 (I/Q) = 128 FF
    # - Pipeline stages (9): 9 × ~80 = 720 FF
    # - Feature output buffer: 24 × 16-bit = 384 FF
    # - Control: ~50 FF
    ff_count = 128 + 720 + 384 + 50  # ~1282 FF
    
    # BRAM: None - small memory fits in distributed RAM
    bram_count = 0
    
    return ModuleResources(
        name="fex_layer_synth",
        lut=lut_count,
        ff=ff_count,
        dsp=dsp_count,
        bram36=bram_count,
        description="Feature extraction: magnitude, A², A³, phase norm"
    )

def estimate_tdnn_resources(
    input_dim: int = 24,
    hidden1: int = 32,
    hidden2: int = 16,
    output_dim: int = 2,
    num_macs: int = 6
) -> ModuleResources:
    """
    Estimate resources for tdnn_generator.
    
    Architecture: FC1(24→32) → LeakyReLU → FC2(32→16) → LeakyReLU → FC3(16→2)
    Uses 6 parallel MAC units (time-multiplexed)
    
    Total params: 24×32+32 + 32×16+16 + 16×2+2 = 800 + 528 + 34 = 1,362
    """
    # Parameters
    fc1_ops = input_dim * hidden1  # 768 MACs
    fc2_ops = hidden1 * hidden2    # 512 MACs  
    fc3_ops = hidden2 * output_dim # 32 MACs
    total_macs = fc1_ops + fc2_ops + fc3_ops  # 1,312 MACs
    
    # DSP Usage:
    # - 6 parallel MAC units for FC computation
    # - Phase denormalization: 4 DSP (complex multiply: I*I - Q*Q, I*Q + Q*I)
    dsp_count = num_macs + 4  # 10 DSP
    
    # LUTs:
    # - MAC control/mux: 6 × ~50 = 300 LUTs
    # - LeakyReLU (2 layers): 2 × (hidden1 + hidden2) × 3 = 288 LUTs
    # - Accumulator management: ~200 LUTs
    # - Weight address generation: ~150 LUTs
    # - State machine: ~200 LUTs
    # - Input buffer mux: ~150 LUTs
    lut_count = 300 + 288 + 200 + 150 + 200 + 150  # ~1288 LUTs
    
    # FFs:
    # - Input buffer: 24 × 16-bit = 384 FF
    # - FC1 output: 32 × 16-bit = 512 FF
    # - FC2 output: 16 × 16-bit = 256 FF
    # - FC3 output: 2 × 16-bit = 32 FF
    # - Accumulators: 6 × 32-bit = 192 FF
    # - Weight pipeline: 6 × 16-bit = 96 FF
    # - Control/counters: ~100 FF
    ff_count = 384 + 512 + 256 + 32 + 192 + 96 + 100  # ~1572 FF
    
    # BRAM: None - weights stored in separate weight_rom
    bram_count = 0
    
    return ModuleResources(
        name="tdnn_generator",
        lut=lut_count,
        ff=ff_count,
        dsp=dsp_count,
        bram36=bram_count,
        description="PN-TDNN: FC1(24→32)→FC2(32→16)→FC3(16→2)"
    )

def estimate_weight_rom_resources(
    params_per_bank: int = 1362,
    num_banks: int = 4,
    data_width: int = 16,
    num_read_ports: int = 5
) -> ModuleResources:
    """
    Estimate resources for weight_rom (BRAM-based).
    
    Architecture:
    - 4 temperature compensation banks
    - 1,362 parameters per bank (16-bit each)
    - 5 read ports for parallel lane access
    
    Memory: 4 × 1,362 × 16 = 87,168 bits = 10.7 KB
    """
    total_bits = num_banks * params_per_bank * data_width  # 87,168 bits
    total_kb = total_bits / 8 / 1024  # 10.66 KB
    
    # BRAM Usage:
    # BRAM36 = 36Kb = 36,864 bits
    # For 5 read ports, need to replicate or use multi-port
    # Option 1: Simple dual-port BRAM with address mux → 3 BRAM36
    # Option 2: True multi-port with replication → 5 × 3 = 15 BRAM36
    # Option 3: Single BRAM + time-multiplex reads → 3 BRAM36
    # 
    # Current design uses time-multiplexed reads (all lanes read same bank)
    # 87,168 bits / 36,864 = 2.4 → 3 BRAM36 (single port)
    # With dual-port for pipeline: 3 BRAM36
    bram_count = 3
    
    # DSP: None
    dsp_count = 0
    
    # LUTs:
    # - Address calculation (bank_sel × BANK_SIZE + addr): ~100 LUTs
    # - Read port mux (if time-multiplexed): ~150 LUTs
    # - Control logic: ~50 LUTs
    lut_count = 100 + 150 + 50  # ~300 LUTs
    
    # FFs:
    # - Output registers (5 ports × 16-bit): 80 FF
    # - Address pipeline: 5 × 16-bit = 80 FF
    # - Control: ~20 FF
    ff_count = 80 + 80 + 20  # ~180 FF
    
    return ModuleResources(
        name="weight_rom",
        lut=lut_count,
        ff=ff_count,
        dsp=dsp_count,
        bram36=bram_count,
        description=f"BRAM weight storage: {num_banks}×{params_per_bank}={num_banks*params_per_bank} params ({total_kb:.1f} KB)"
    )

def estimate_dpd_top_ssr_resources(
    num_lanes: int = 5,
    interp: ModuleResources = None,
    fex: ModuleResources = None,
    tdnn: ModuleResources = None,
    weight_rom: ModuleResources = None
) -> Tuple[ModuleResources, Dict[str, ModuleResources]]:
    """
    Estimate total resources for dpd_top_ssr.
    
    Architecture:
    - 1× interpolator (shared)
    - 1× weight_rom (shared)
    - 5× fex_layer (parallel lanes)
    - 5× tdnn_generator (parallel lanes)
    - Top-level glue logic
    """
    if interp is None:
        interp = estimate_interpolator_resources()
    if fex is None:
        fex = estimate_fex_resources()
    if tdnn is None:
        tdnn = estimate_tdnn_resources()
    if weight_rom is None:
        weight_rom = estimate_weight_rom_resources()
    
    # Top-level glue logic
    # - Output mux/packing: ~200 LUTs
    # - Valid/busy logic: ~100 LUTs
    # - Reset synchronization: ~50 FF
    glue_lut = 300
    glue_ff = 100
    
    # Calculate totals
    # Interpolator: 1×
    # FEX: 5× (one per lane)
    # TDNN: 5× (one per lane)
    # Weight ROM: 1× (shared)
    
    total_lut = (interp.lut + 
                 fex.lut * num_lanes + 
                 tdnn.lut * num_lanes + 
                 weight_rom.lut + 
                 glue_lut)
    
    total_ff = (interp.ff + 
                fex.ff * num_lanes + 
                tdnn.ff * num_lanes + 
                weight_rom.ff + 
                glue_ff)
    
    total_dsp = (interp.dsp + 
                 fex.dsp * num_lanes + 
                 tdnn.dsp * num_lanes + 
                 weight_rom.dsp)
    
    total_bram = (interp.bram36 + 
                  fex.bram36 * num_lanes + 
                  tdnn.bram36 * num_lanes + 
                  weight_rom.bram36)
    
    top = ModuleResources(
        name="dpd_top_ssr (TOTAL)",
        lut=total_lut,
        ff=total_ff,
        dsp=total_dsp,
        bram36=total_bram,
        description=f"SSR=5 lanes @ 200MHz = 1 GSps aggregate"
    )
    
    # Create breakdown dict
    breakdown = {
        'interpolator1_5_ssr': interp,
        'fex_layer_synth': ModuleResources(
            name=f"fex_layer_synth (×{num_lanes})",
            lut=fex.lut * num_lanes,
            ff=fex.ff * num_lanes,
            dsp=fex.dsp * num_lanes,
            bram36=fex.bram36 * num_lanes,
            description=fex.description
        ),
        'tdnn_generator': ModuleResources(
            name=f"tdnn_generator (×{num_lanes})",
            lut=tdnn.lut * num_lanes,
            ff=tdnn.ff * num_lanes,
            dsp=tdnn.dsp * num_lanes,
            bram36=tdnn.bram36 * num_lanes,
            description=tdnn.description
        ),
        'weight_rom': weight_rom,
        'glue_logic': ModuleResources(
            name="glue_logic",
            lut=glue_lut,
            ff=glue_ff,
            dsp=0,
            bram36=0,
            description="Top-level interconnect, mux, control"
        ),
    }
    
    return top, breakdown

# =============================================================================
# Report Generation
# =============================================================================

def generate_report(weight_dir: str = None) -> str:
    """Generate comprehensive resource estimation report."""
    
    device = ZCU104()
    
    # Analyze actual weights if available
    if weight_dir and os.path.exists(weight_dir):
        weights = analyze_weight_files(weight_dir)
        fc1_params = weights.get('fc1_weight', 768) + weights.get('fc1_bias', 32)
        fc2_params = weights.get('fc2_weight', 512) + weights.get('fc2_bias', 16)
        fc3_params = weights.get('fc3_weight', 32) + weights.get('fc3_bias', 2)
        total_params = fc1_params + fc2_params + fc3_params
        weight_info = f"""
Weight File Analysis:
  FC1: {weights.get('fc1_weight', 0)} weights + {weights.get('fc1_bias', 0)} biases = {fc1_params} params
  FC2: {weights.get('fc2_weight', 0)} weights + {weights.get('fc2_bias', 0)} biases = {fc2_params} params
  FC3: {weights.get('fc3_weight', 0)} weights + {weights.get('fc3_bias', 0)} biases = {fc3_params} params
  Total: {total_params} parameters per bank
  Memory: {total_params * 16 / 8 / 1024:.2f} KB per bank, {total_params * 16 * 4 / 8 / 1024:.2f} KB for 4 banks
"""
    else:
        weight_info = "  (Using default parameter counts: 1,362 per bank)"
        total_params = 1362
    
    # Estimate per-module resources
    interp = estimate_interpolator_resources()
    fex = estimate_fex_resources()
    tdnn = estimate_tdnn_resources()
    weight_rom = estimate_weight_rom_resources(params_per_bank=total_params)
    
    # Calculate top-level totals
    top, breakdown = estimate_dpd_top_ssr_resources(
        num_lanes=5,
        interp=interp,
        fex=fex,
        tdnn=tdnn,
        weight_rom=weight_rom
    )
    
    # Build report
    lines = [
        "=" * 80,
        "DPD SSR Module-Level Resource Estimation for ZCU104",
        "=" * 80,
        "",
        f"Target Device: {device.name}",
        f"  DSP48E2:  {device.dsp_count}",
        f"  LUTs:     {device.lut_count:,}",
        f"  FFs:      {device.ff_count:,}",
        f"  BRAM36:   {device.bram_36kb}",
        "",
        weight_info,
        "",
        "-" * 80,
        "PER-MODULE RESOURCE BREAKDOWN",
        "-" * 80,
        f"{'Module':<25s} | {'LUT':>6s} | {'FF':>6s} | {'DSP':>4s} | {'BRAM':>4s}",
        "-" * 80,
    ]
    
    # Individual modules (single instance)
    lines.append(str(interp))
    lines.append(f"  └─ {interp.description}")
    lines.append("")
    
    lines.append(str(fex) + "  (per lane)")
    lines.append(f"  └─ {fex.description}")
    lines.append("")
    
    lines.append(str(tdnn) + "  (per lane)")
    lines.append(f"  └─ {tdnn.description}")
    lines.append("")
    
    lines.append(str(weight_rom))
    lines.append(f"  └─ {weight_rom.description}")
    lines.append("")
    
    lines.append("-" * 80)
    lines.append("SSR=5 PARALLEL LANES (TOTALS)")
    lines.append("-" * 80)
    
    for name, res in breakdown.items():
        lines.append(str(res))
    
    lines.append("-" * 80)
    lines.append(str(top))
    lines.append("-" * 80)
    lines.append("")
    
    # Utilization
    lines.append("UTILIZATION SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  DSP:  {top.dsp:4d} / {device.dsp_count:4d}  ({100*top.dsp/device.dsp_count:5.2f}%)")
    lines.append(f"  LUT:  {top.lut:5d} / {device.lut_count:6d}  ({100*top.lut/device.lut_count:5.2f}%)")
    lines.append(f"  FF:   {top.ff:5d} / {device.ff_count:6d}  ({100*top.ff/device.ff_count:5.2f}%)")
    lines.append(f"  BRAM: {top.bram36:4d} / {device.bram_36kb:4d}  ({100*top.bram36/device.bram_36kb:5.2f}%)")
    lines.append("")
    
    # Performance estimates
    clock_mhz = 200
    throughput_gsps = 5 * clock_mhz / 1000  # 5 samples per clock
    
    lines.append("PERFORMANCE ESTIMATES")
    lines.append("-" * 40)
    lines.append(f"  Clock Frequency:  {clock_mhz} MHz")
    lines.append(f"  SSR Factor:       5 samples/clock")
    lines.append(f"  Throughput:       {throughput_gsps:.1f} GSps aggregate")
    lines.append(f"  Latency:          ~55 cycles ({55*1000/clock_mhz:.1f} ns)")
    lines.append("")
    
    # Power estimate (rough)
    dsp_power = top.dsp * 0.022 * clock_mhz  # mW
    lut_power = top.lut / 1000 * 0.73  # mW
    bram_power = top.bram36 * 6.7  # mW
    static_power = 350  # mW
    total_power = dsp_power + lut_power + bram_power + static_power
    
    lines.append("POWER ESTIMATES (from XPE coefficients)")
    lines.append("-" * 40)
    lines.append(f"  DSP Dynamic:   {dsp_power:6.1f} mW")
    lines.append(f"  Logic Dynamic: {lut_power:6.1f} mW")
    lines.append(f"  BRAM Dynamic:  {bram_power:6.1f} mW")
    lines.append(f"  Static:        {static_power:6.1f} mW")
    lines.append(f"  TOTAL:         {total_power:6.1f} mW ({total_power/1000:.2f} W)")
    lines.append("")
    
    # Energy per sample: power (mW) / throughput (GSps) = pJ/sample
    # P = E/t, E = P * t = P / (samples/sec) = P / (1e9 samples/sec) * 1e12 pJ/J
    # = P_mW * 1e-3 / 1e9 * 1e12 = P_mW * 1 pJ per sample at 1 GSps
    energy_per_sample_pj = total_power / throughput_gsps  # mW / GSps = pJ/sample
    lines.append(f"  Energy/Sample: {energy_per_sample_pj:.1f} pJ")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("Note: These are pre-RTL estimates. Actual utilization will vary after synthesis.")
    lines.append("Run 'vivado -mode batch -source fpga/scripts/synth_dpd_ssr.tcl' for accurate numbers.")
    lines.append("=" * 80)
    
    return "\n".join(lines)

# =============================================================================
# Export Functions
# =============================================================================

def export_json(output_path: str, weight_dir: str = None):
    """Export resource estimates to JSON."""
    
    # Analyze weights
    if weight_dir and os.path.exists(weight_dir):
        weights = analyze_weight_files(weight_dir)
        total_params = sum(weights.values())
    else:
        total_params = 1362
    
    interp = estimate_interpolator_resources()
    fex = estimate_fex_resources()
    tdnn = estimate_tdnn_resources()
    weight_rom = estimate_weight_rom_resources(params_per_bank=total_params)
    top, breakdown = estimate_dpd_top_ssr_resources(5, interp, fex, tdnn, weight_rom)
    
    device = ZCU104()
    
    data = {
        "device": {
            "name": device.name,
            "dsp_count": device.dsp_count,
            "lut_count": device.lut_count,
            "ff_count": device.ff_count,
            "bram_36kb": device.bram_36kb,
        },
        "modules": {
            "interpolator1_5_ssr": {
                "lut": interp.lut, "ff": interp.ff, 
                "dsp": interp.dsp, "bram36": interp.bram36,
                "instances": 1
            },
            "fex_layer_synth": {
                "lut": fex.lut, "ff": fex.ff,
                "dsp": fex.dsp, "bram36": fex.bram36,
                "instances": 5
            },
            "tdnn_generator": {
                "lut": tdnn.lut, "ff": tdnn.ff,
                "dsp": tdnn.dsp, "bram36": tdnn.bram36,
                "instances": 5
            },
            "weight_rom": {
                "lut": weight_rom.lut, "ff": weight_rom.ff,
                "dsp": weight_rom.dsp, "bram36": weight_rom.bram36,
                "instances": 1
            },
        },
        "total": {
            "lut": top.lut,
            "ff": top.ff,
            "dsp": top.dsp,
            "bram36": top.bram36,
        },
        "utilization_percent": {
            "lut": 100 * top.lut / device.lut_count,
            "ff": 100 * top.ff / device.ff_count,
            "dsp": 100 * top.dsp / device.dsp_count,
            "bram": 100 * top.bram36 / device.bram_36kb,
        },
        "performance": {
            "clock_mhz": 200,
            "ssr_factor": 5,
            "throughput_gsps": 1.0,
            "latency_cycles": 55,
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported to {output_path}")

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Find weight directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    weight_dir = os.path.join(project_root, "rtl", "weights")
    
    # Generate and print report
    report = generate_report(weight_dir)
    print(report)
    
    # Export JSON
    json_path = os.path.join(script_dir, "module_resources.json")
    export_json(json_path, weight_dir)
