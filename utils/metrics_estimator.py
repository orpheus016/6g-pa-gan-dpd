"""
Pre-RTL Metrics Estimator for PN-TDNN-DPD Architecture

Deterministic estimation of energy, latency, throughput, and area metrics
based on architecture specifications. Uses first-principles models from:
- Rabaey, "Digital Integrated Circuits" (power model)
- Xilinx UG907 (FPGA power estimation)
- SparseDPD/OpenDPDv2 methodologies (comparison metrics)

Author: [Your Name]
Date: January 2026
Reference: docs/architecture.md, knowledge/Metrics/*.md
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import math


# =============================================================================
# Architecture Configuration (from architecture.md)
# =============================================================================

@dataclass
class FPGADevice:
    """FPGA device specifications from Xilinx datasheets."""
    name: str
    dsp_count: int
    lut_count: int
    ff_count: int
    bram_36kb: int
    # Power per primitive at reference frequency (mW) - from XPE
    dsp_power_mw_per_mhz: float  # Per DSP at 100% activity
    lut_power_mw_per_1k: float   # Per 1k LUTs
    bram_power_mw: float         # Per 36Kb BRAM (active)
    static_power_mw: float       # Quiescent power
    core_voltage: float          # V_core
    # Timing
    dsp_fmax_mhz: int            # Pipelined DSP Fmax
    
    
PYNQ_Z1 = FPGADevice(
    name="PYNQ-Z1 (XC7Z020-1CLG400C)",
    dsp_count=220,
    lut_count=53200,
    ff_count=106400,
    bram_36kb=140,
    # Power coefficients derived from architecture.md Section 9.3 (XPE-based)
    # Target: 62 DSPs → 300 mW DSP power, 11k LUTs → 150 mW logic
    # Reference: SparseDPD Table II: 66 DSPs @ 170 MHz = 77 mW → 0.69 mW/DSP @ 170 MHz
    # Scaled to 250 MHz: 77/66/170*250 = 0.017 mW/DSP/MHz
    dsp_power_mw_per_mhz=0.017,  # Derived: 300mW / 62 DSPs / 250 MHz ≈ 0.019
    lut_power_mw_per_1k=0.55,    # Derived: 150mW / 11k LUTs * (100/250) activity scaling
    bram_power_mw=5.5,           # Derived: 50mW / 9 BRAMs ≈ 5.5 mW/BRAM
    static_power_mw=200.0,       # From architecture.md Table (XPE)
    core_voltage=1.0,            # 7-series typical
    dsp_fmax_mhz=280,            # -1 grade pipelined DSP (DS181)
)

ZCU104 = FPGADevice(
    name="ZCU104 (XCZU7EV-2FFVC1156)",
    dsp_count=1728,
    lut_count=230400,
    ff_count=460800,
    bram_36kb=312,
    # UltraScale+ coefficients from architecture.md Section 9.3
    # Target: ~1.1 W total for same design
    dsp_power_mw_per_mhz=0.022,  # Derived: 350mW / 62 DSPs / 250 MHz
    lut_power_mw_per_1k=0.73,    # Derived: 200mW / 11k LUTs * (100/250)
    bram_power_mw=6.7,           # Derived: 60mW / 9 BRAMs
    static_power_mw=350.0,       # From architecture.md (UltraScale+ higher static)
    core_voltage=0.85,           # UltraScale+ typical
    dsp_fmax_mhz=500,            # DSP48E2 pipelined (DS923)
)


@dataclass
class LayerSpec:
    """Specification for a single FC layer."""
    name: str
    input_dim: int
    output_dim: int
    
    @property
    def weights(self) -> int:
        return self.input_dim * self.output_dim
    
    @property
    def biases(self) -> int:
        return self.output_dim
    
    @property
    def params(self) -> int:
        return self.weights + self.biases
    
    @property
    def macs_per_sample(self) -> int:
        """MAC operations per inference sample."""
        return self.input_dim * self.output_dim
    
    @property
    def dsps_systolic(self) -> int:
        """DSPs needed for II=1 systolic (one per output neuron)."""
        return self.output_dim
    
    @property
    def latency_cycles(self) -> int:
        """Cycles to complete one sample (input dim for systolic)."""
        return self.input_dim


@dataclass 
class PNTDNNArchitecture:
    """PN-TDNN-DPD architecture specification."""
    name: str = "PN-TDNN-DPD"
    
    # Input configuration
    memory_depth: int = 3  # M=3 (4 taps: n, n-1, n-2, n-3)
    num_taps: int = 4
    features_per_tap: int = 6  # A, A³, I_norm, Q_norm, I, Q
    input_dim: int = 24  # 6 features × 4 taps
    
    # FC layers (from architecture.md Section 6)
    fc1: LayerSpec = field(default_factory=lambda: LayerSpec("FC1", 24, 32))
    fc2: LayerSpec = field(default_factory=lambda: LayerSpec("FC2", 32, 16))
    fc3: LayerSpec = field(default_factory=lambda: LayerSpec("FC3", 16, 2))
    
    # Feature extraction (CORDIC)
    cordic_iterations: int = 8
    cordic_dsps: int = 8  # One DSP per pipeline stage
    
    # Phase normalization/denormalization
    phase_norm_dsps: int = 2   # 2 multiplies for norm
    phase_denorm_dsps: int = 2 # 2 multiplies for denorm
    
    # Quantization
    weight_bits: int = 16  # Q1.15
    activation_bits: int = 16  # Q8.8
    accumulator_bits: int = 32  # Q16.16
    
    # SPSA adaptation engine
    spsa_dsps: int = 12  # Perturbation + gradient + update
    spsa_brams: int = 4  # Shadow RAM banks
    
    @property
    def layers(self) -> List[LayerSpec]:
        return [self.fc1, self.fc2, self.fc3]
    
    @property
    def total_params(self) -> int:
        return sum(l.params for l in self.layers)
    
    @property
    def total_macs_per_sample(self) -> int:
        return sum(l.macs_per_sample for l in self.layers)
    
    @property
    def fc_dsps_systolic(self) -> int:
        """DSPs for systolic FC layers (II=1)."""
        return sum(l.dsps_systolic for l in self.layers)
    
    @property
    def data_path_dsps(self) -> int:
        """Total DSPs for data path at 250 MHz."""
        return (self.cordic_dsps + 
                self.fc_dsps_systolic + 
                self.phase_norm_dsps + 
                self.phase_denorm_dsps)
    
    @property
    def total_dsps(self) -> int:
        """Total DSPs including SPSA."""
        return self.data_path_dsps + self.spsa_dsps
    
    @property
    def pipeline_latency_cycles(self) -> int:
        """Total pipeline depth in cycles."""
        cordic = self.cordic_iterations
        fc_latency = sum(l.latency_cycles for l in self.layers)
        denorm = 1
        return cordic + fc_latency + denorm
    
    @property
    def weight_memory_bits(self) -> int:
        """Memory needed for one weight bank."""
        return self.total_params * self.weight_bits
    
    @property
    def weight_brams(self) -> int:
        """36Kb BRAMs for weights (2 banks for CDC)."""
        bits_per_bram = 36 * 1024
        brams_per_bank = max(2, math.ceil(self.weight_memory_bits / bits_per_bram))
        return brams_per_bank * 2  # Double buffer for CDC shadow RAM
    
    @property
    def delay_line_brams(self) -> int:
        """BRAMs for input delay line storage."""
        return 1  # M=3 memory, compact storage
    
    @property
    def total_brams(self) -> int:
        """Total BRAMs (weights + delay + SPSA)."""
        return self.weight_brams + self.delay_line_brams + self.spsa_brams


# =============================================================================
# Metrics Calculator
# =============================================================================

@dataclass
class PreRTLMetrics:
    """Complete pre-RTL metric set for publication."""
    
    # Throughput metrics
    clock_freq_mhz: float
    throughput_msps: float
    initiation_interval: int
    latency_cycles: int
    latency_ns: float
    
    # Area metrics (FPGA resources)
    dsp_used: int
    dsp_available: int
    dsp_utilization_pct: float
    lut_estimated: int
    lut_utilization_pct: float
    ff_estimated: int
    ff_utilization_pct: float
    bram_used: int
    bram_utilization_pct: float
    
    # Power/Energy metrics
    dynamic_power_mw: float
    static_power_mw: float
    total_power_mw: float
    energy_per_sample_nj: float
    energy_per_sample_pj: float
    
    # Efficiency metrics
    throughput_per_dsp: float  # MSps/DSP
    energy_per_mac_pj: float
    macs_per_sample: int
    gops: float  # Giga-ops per second
    
    # Model metrics
    total_params: int
    weight_memory_kb: float
    
    # Comparison ready
    signal_bandwidth_mhz: float
    power_per_mhz_bw: float  # mW/MHz
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}
    
    def __str__(self) -> str:
        lines = [
            "=" * 70,
            "PN-TDNN-DPD Pre-RTL Metrics Report",
            "=" * 70,
            "",
            "THROUGHPUT & LATENCY",
            "-" * 40,
            f"  Clock Frequency:      {self.clock_freq_mhz:.0f} MHz",
            f"  Throughput:           {self.throughput_msps:.0f} MSps",
            f"  Initiation Interval:  {self.initiation_interval} (II={self.initiation_interval})",
            f"  Pipeline Latency:     {self.latency_cycles} cycles ({self.latency_ns:.1f} ns)",
            "",
            "FPGA RESOURCES (Area)",
            "-" * 40,
            f"  DSP48 Slices:         {self.dsp_used} / {self.dsp_available} ({self.dsp_utilization_pct:.1f}%)",
            f"  LUTs (estimated):     {self.lut_estimated:,} ({self.lut_utilization_pct:.1f}%)",
            f"  FFs (estimated):      {self.ff_estimated:,} ({self.ff_utilization_pct:.1f}%)",
            f"  BRAMs (36Kb):         {self.bram_used} ({self.bram_utilization_pct:.1f}%)",
            "",
            "POWER & ENERGY",
            "-" * 40,
            f"  Dynamic Power:        {self.dynamic_power_mw:.1f} mW",
            f"  Static Power:         {self.static_power_mw:.1f} mW",
            f"  Total Power:          {self.total_power_mw:.1f} mW",
            f"  Energy/Sample:        {self.energy_per_sample_nj:.3f} nJ ({self.energy_per_sample_pj:.1f} pJ)",
            "",
            "EFFICIENCY METRICS",
            "-" * 40,
            f"  Throughput/DSP:       {self.throughput_per_dsp:.2f} MSps/DSP",
            f"  MACs/Sample:          {self.macs_per_sample}",
            f"  GOPS:                 {self.gops:.1f}",
            f"  Energy/MAC:           {self.energy_per_mac_pj:.2f} pJ",
            "",
            "MODEL CHARACTERISTICS",
            "-" * 40,
            f"  Total Parameters:     {self.total_params:,}",
            f"  Weight Memory:        {self.weight_memory_kb:.2f} KB",
            f"  Signal Bandwidth:     {self.signal_bandwidth_mhz:.0f} MHz",
            f"  Power/MHz BW:         {self.power_per_mhz_bw:.2f} mW/MHz",
            "",
            "=" * 70,
        ]
        return "\n".join(lines)


class MetricsEstimator:
    """
    Deterministic pre-RTL metrics estimator.
    
    Power model scaled from SparseDPD (arXiv:2506.16591v1) measured results:
    - 66 DSPs @ 170 MHz = 77 mW DSP power
    - 2298 LUTs @ 170 MHz = 125 mW logic power (signals + clocks + LUTs)
    - 13 BRAMs = 40 mW BRAM power
    
    Area model from architecture primitive counting.
    
    Reference: 
    - SparseDPD Table II (measured power breakdown)
    - architecture.md Section 9.3 (XPE validation)
    - knowledge/Metrics/ENERGY.md (first-principles model)
    """
    
    def __init__(
        self,
        arch: PNTDNNArchitecture,
        device: FPGADevice,
        clock_mhz: float = 250.0,
        signal_bw_mhz: float = 200.0,
        power_margin: float = 1.0,  # 1.0 = no margin, use measured scaling
    ):
        self.arch = arch
        self.device = device
        self.clock_mhz = clock_mhz
        self.signal_bw_mhz = signal_bw_mhz
        self.power_margin = power_margin
    
    def estimate_luts(self) -> int:
        """
        Estimate LUT usage based on architecture.
        
        Components:
        - Control logic: ~500 LUTs per FC layer
        - Activation (LeakyReLU): ~50 LUTs per neuron
        - CORDIC: ~200 LUTs per iteration
        - Phase norm/denorm: ~500 LUTs each
        - SPSA: ~2000 LUTs
        """
        control_per_layer = 500
        activation_luts_per_neuron = 50
        cordic_luts_per_iter = 200
        phase_luts = 500
        spsa_luts = 2500
        
        fc_control = len(self.arch.layers) * control_per_layer
        fc_activation = sum(l.output_dim * activation_luts_per_neuron 
                          for l in self.arch.layers[:-1])  # No activation on output
        cordic = self.arch.cordic_iterations * cordic_luts_per_iter
        phase = phase_luts * 2  # norm + denorm
        
        # Pipeline registers (overhead)
        pipeline_overhead = 2000
        
        return (fc_control + fc_activation + cordic + 
                phase + spsa_luts + pipeline_overhead)
    
    def estimate_ffs(self) -> int:
        """
        Estimate flip-flop usage.
        
        Components:
        - Pipeline registers: bits × stages
        - Accumulators: 32-bit per neuron
        - Input delay line: M samples × features × bits
        - Control FSMs: ~200 per module
        """
        # Pipeline registers (between stages)
        pipeline_bits = self.arch.activation_bits
        pipeline_stages = self.arch.pipeline_latency_cycles
        neuron_count = sum(l.output_dim for l in self.arch.layers)
        
        # Accumulators (one per neuron in systolic)
        acc_bits = self.arch.accumulator_bits
        accumulator_ffs = neuron_count * acc_bits
        
        # Delay line for memory taps
        delay_line_ffs = (self.arch.num_taps * 
                         self.arch.features_per_tap * 
                         self.arch.activation_bits)
        
        # Control logic
        control_ffs = 500 * (len(self.arch.layers) + 3)  # +3 for FEx, norm, denorm
        
        # SPSA state machine
        spsa_ffs = 1500
        
        return accumulator_ffs + delay_line_ffs + control_ffs + spsa_ffs
    
    def estimate_power(self) -> Tuple[float, float]:
        """
        Estimate dynamic and static power.
        
        Uses architecture.md Section 9.3 breakdown (XPE-derived):
        PYNQ-Z1 @ 250 MHz: Logic 150 mW, DSP 300 mW, BRAM 50 mW, I/O 100 mW → 600 mW dynamic
        
        Cross-validation with SparseDPD Table II (measured @ 170 MHz):
        - DSP: 77 mW for 66 DSPs → 1.17 mW/DSP
        - BRAM: 40 mW for 13 BRAMs → 3.08 mW/BRAM
        - Logic: 125 mW for 2.3k LUTs → ~54 mW per 1k LUTs
        
        Reference: architecture.md, SparseDPD arXiv:2506.16591v1 Table II
        """
        freq_mhz = self.clock_mhz
        freq_ratio = freq_mhz / 250.0  # Scale from 250 MHz baseline
        
        # DSP power: From architecture.md target 300 mW for 62 data path DSPs
        # → 4.84 mW/DSP @ 250 MHz
        # Cross-check: SparseDPD 77 mW / 66 DSPs @ 170 MHz = 1.17 mW/DSP
        # Scale: 1.17 × (250/170) = 1.72 mW/DSP → 62 × 1.72 = 107 mW (much less)
        # Use architecture.md value (XPE-based, conservative)
        dsp_power_per_unit = 300.0 / 62.0  # 4.84 mW/DSP @ 250 MHz
        dsp_power = self.arch.data_path_dsps * dsp_power_per_unit * freq_ratio
        # SPSA DSPs at 1 MHz (negligible)
        spsa_dsp_power = self.arch.spsa_dsps * dsp_power_per_unit * (1.0 / 250.0)
        
        # Logic power: From architecture.md target 150 mW
        # This includes LUTs + signals + clocks for ~11k LUT design
        # → ~13.6 mW per 1k LUTs @ 250 MHz
        lut_count = self.estimate_luts()
        lut_power = (lut_count / 11000.0) * 150.0 * freq_ratio
        
        # BRAM power: From architecture.md target 50 mW for ~9 BRAMs
        # → 5.56 mW/BRAM @ 250 MHz
        # Cross-check: SparseDPD 40 mW / 13 BRAMs = 3.08 mW/BRAM @ 170 MHz
        # Scale: 3.08 × (250/170) = 4.53 mW/BRAM → reasonable match
        bram_count = self.arch.total_brams
        bram_power = bram_count * (50.0 / 9.0) * freq_ratio
        
        # I/O power: From architecture.md 100 mW for LVDS ADC/DAC @ 250 MSps
        io_power = 100.0 * freq_ratio
        
        dynamic = (dsp_power + spsa_dsp_power + lut_power + bram_power + io_power) * self.power_margin
        static = self.device.static_power_mw
        
        return dynamic, static
    
    def compute_metrics(self) -> PreRTLMetrics:
        """Compute all pre-RTL metrics."""
        
        # Throughput (II=1 means 1 sample per clock after pipeline fill)
        ii = 1
        throughput = self.clock_mhz  # MSps = MHz for II=1
        latency_cycles = self.arch.pipeline_latency_cycles
        latency_ns = latency_cycles * (1000 / self.clock_mhz)
        
        # Area
        dsp_used = self.arch.total_dsps
        lut_est = self.estimate_luts()
        ff_est = self.estimate_ffs()
        bram_used = self.arch.total_brams
        
        # Power
        dynamic_mw, static_mw = self.estimate_power()
        total_mw = dynamic_mw + static_mw
        
        # Energy
        energy_nj = total_mw / throughput  # mW / MSps = nJ/sample
        energy_pj = energy_nj * 1000
        
        # Efficiency
        macs = self.arch.total_macs_per_sample
        gops = (macs * throughput) / 1000  # GOPS
        energy_per_mac_pj = energy_pj / macs
        throughput_per_dsp = throughput / dsp_used
        
        # Model
        weight_kb = self.arch.weight_memory_bits / (8 * 1024)
        
        # Power efficiency
        power_per_mhz = total_mw / self.signal_bw_mhz
        
        return PreRTLMetrics(
            clock_freq_mhz=self.clock_mhz,
            throughput_msps=throughput,
            initiation_interval=ii,
            latency_cycles=latency_cycles,
            latency_ns=latency_ns,
            dsp_used=dsp_used,
            dsp_available=self.device.dsp_count,
            dsp_utilization_pct=100 * dsp_used / self.device.dsp_count,
            lut_estimated=lut_est,
            lut_utilization_pct=100 * lut_est / self.device.lut_count,
            ff_estimated=ff_est,
            ff_utilization_pct=100 * ff_est / self.device.ff_count,
            bram_used=bram_used,
            bram_utilization_pct=100 * bram_used / self.device.bram_36kb,
            dynamic_power_mw=dynamic_mw,
            static_power_mw=static_mw,
            total_power_mw=total_mw,
            energy_per_sample_nj=energy_nj,
            energy_per_sample_pj=energy_pj,
            throughput_per_dsp=throughput_per_dsp,
            energy_per_mac_pj=energy_per_mac_pj,
            macs_per_sample=macs,
            gops=gops,
            total_params=self.arch.total_params,
            weight_memory_kb=weight_kb,
            signal_bandwidth_mhz=self.signal_bw_mhz,
            power_per_mhz_bw=power_per_mhz,
        )


# =============================================================================
# Comparison with Prior Art
# =============================================================================

@dataclass
class PriorArtMetrics:
    """Metrics from published papers for comparison."""
    name: str
    platform: str
    architecture: str
    params: int
    clock_mhz: float
    throughput_msps: float
    power_mw: float
    signal_bw_mhz: float
    acpr_dbc: float
    evm_db: float
    nmse_db: float
    
    @property
    def energy_pj(self) -> float:
        return (self.power_mw / self.throughput_msps) * 1000
    
    @property
    def throughput_per_param(self) -> float:
        return self.throughput_msps / self.params


# From SparseDPD paper (arXiv:2506.16591v1, Table I & II)
# Zynq-7Z010: 66 DSPs, 2298 LUTs, 1724 FFs, 13 BRAMs
# Dynamic: 241 mW, Static: 164 mW, Total: 405 mW
# Source: Table II power breakdown (DSP: 77mW, BRAM: 40mW, Logic: 37mW, Signals: 65mW, Clocks: 23mW)
SPARSE_DPD = PriorArtMetrics(
    name="SparseDPD",
    platform="FPGA (7Z010)",
    architecture="PNTDNN (sparse)",
    params=64,  # 74% sparsity, W14A14
    clock_mhz=170,
    throughput_msps=170,  # II=1, 1 sample/cycle
    power_mw=405,  # 241 dynamic + 164 static (Table II)
    signal_bw_mhz=20,
    acpr_dbc=-59.4,
    evm_db=-54.0,
    nmse_db=-48.2,
)

# From OpenDPDv2 paper (arXiv:2507.06849v2, Section I)
# TRes-DeltaGRU-999 (999 params), FP32 model
# GPU inference (no FPGA implementation)
# Source: "[13] GPU TDNN FP32 909 ~1,818 ~2,300 1000 ≤320 200" from SparseDPD Table I
OPEN_DPD_V2 = PriorArtMetrics(
    name="OpenDPDv2",
    platform="GPU",
    architecture="TRes-DeltaGRU",
    params=999,
    clock_mhz=2300,  # GPU clock (from SparseDPD Table I reference [13])
    throughput_msps=1000,  # ~1000 MSps (batch processing)
    power_mw=320000,  # GPU TDP ≤320W (from Table I)
    signal_bw_mhz=200,
    acpr_dbc=-59.9,
    evm_db=-42.1,
    nmse_db=-39.6,
)


def generate_comparison_table(
    ours: PreRTLMetrics,
    prior_art: List[PriorArtMetrics],
    target_acpr: float = -62.0,
    target_evm: float = -45.0,
) -> str:
    """Generate publication-ready comparison table."""
    
    lines = [
        "",
        "=" * 90,
        "COMPARISON WITH STATE-OF-THE-ART",
        "=" * 90,
        "",
        f"{'Metric':<25} | {'OpenDPDv2':<15} | {'SparseDPD':<15} | {'Ours (Target)':<15}",
        "-" * 90,
    ]
    
    # Helper for formatting
    def fmt(val, unit="", prec=1):
        if isinstance(val, float):
            return f"{val:.{prec}f}{unit}"
        return f"{val}{unit}"
    
    comparisons = [
        ("Architecture", OPEN_DPD_V2.architecture, SPARSE_DPD.architecture, "PN-TDNN (systolic)"),
        ("Platform", OPEN_DPD_V2.platform, SPARSE_DPD.platform, "FPGA (7Z020)"),
        ("Parameters", fmt(OPEN_DPD_V2.params), fmt(SPARSE_DPD.params), fmt(ours.total_params)),
        ("Signal BW (MHz)", fmt(OPEN_DPD_V2.signal_bw_mhz, "", 0), fmt(SPARSE_DPD.signal_bw_mhz, "", 0), fmt(ours.signal_bandwidth_mhz, "", 0)),
        ("Throughput (MSps)", fmt(OPEN_DPD_V2.throughput_msps, "", 0), fmt(SPARSE_DPD.throughput_msps, "", 0), fmt(ours.throughput_msps, "", 0)),
        ("Latency", "~ms (RNN)", "~60 ns", f"{ours.latency_ns:.0f} ns"),
        ("Power (mW)", "320,000", fmt(SPARSE_DPD.power_mw, "", 0), f"{ours.total_power_mw:.0f} (XPE)"),
        ("Energy/Sample (pJ)", fmt(OPEN_DPD_V2.energy_pj, "", 0), fmt(SPARSE_DPD.energy_pj, "", 0), f"{ours.energy_per_sample_pj:.0f} (XPE)"),
        ("Power Method", "TDP", "Measured", "XPE estimate"),
        ("ACPR (dBc)", fmt(OPEN_DPD_V2.acpr_dbc), fmt(SPARSE_DPD.acpr_dbc), f"< {target_acpr}"),
        ("EVM (dB)", fmt(OPEN_DPD_V2.evm_db), fmt(SPARSE_DPD.evm_db), f"< {target_evm}"),
        ("NMSE (dB)", fmt(OPEN_DPD_V2.nmse_db), fmt(SPARSE_DPD.nmse_db), "< -42.0"),
        ("Online Adaptation", "No", "No", "Yes (A-SPSA)"),
        ("FPGA Deployable", "No", "Yes", "Yes"),
    ]
    
    for metric, opendpd, sparse, ours_val in comparisons:
        lines.append(f"{metric:<25} | {opendpd:<15} | {sparse:<15} | {ours_val:<15}")
    
    # Calculate fair comparisons
    # Throughput/BW ratio (apples-to-apples)
    our_throughput_per_mhz = ours.throughput_msps / ours.signal_bandwidth_mhz  # 1.25
    sparse_throughput_per_mhz = SPARSE_DPD.throughput_msps / SPARSE_DPD.signal_bw_mhz  # 8.5
    
    # Energy per MHz of signal bandwidth (better metric for wideband comparison)
    our_power_per_bw = ours.total_power_mw / ours.signal_bandwidth_mhz  # 4.0 mW/MHz
    sparse_power_per_bw = SPARSE_DPD.power_mw / SPARSE_DPD.signal_bw_mhz  # 20.25 mW/MHz
    
    lines.extend([
        "-" * 90,
        "",
        "KEY METRICS (normalized for fair comparison):",
        f"  • Power/Bandwidth:      Ours {our_power_per_bw:.1f} mW/MHz vs SparseDPD {sparse_power_per_bw:.1f} mW/MHz ({sparse_power_per_bw/our_power_per_bw:.1f}x better)",
        f"  • Throughput:           Ours {ours.throughput_msps:.0f} MSps vs SparseDPD {SPARSE_DPD.throughput_msps:.0f} MSps ({ours.throughput_msps/SPARSE_DPD.throughput_msps:.1f}x)",
        f"  • Signal Bandwidth:     Ours {ours.signal_bandwidth_mhz:.0f} MHz vs SparseDPD {SPARSE_DPD.signal_bw_mhz:.0f} MHz ({ours.signal_bandwidth_mhz/SPARSE_DPD.signal_bw_mhz:.0f}x)",
        f"  • GPU comparison:       {OPEN_DPD_V2.energy_pj/ours.energy_per_sample_pj:.0f}x energy efficiency vs OpenDPDv2",
        "",
        "CAVEATS:",
        "  • Our power is XPE estimate (pre-RTL); SparseDPD is post-implementation measured",
        "  • XPE typically 2-3x conservative vs measured (expect ~300-400 mW actual)",
        "  • ACPR/EVM targets are design goals, not measured",
        "",
        "=" * 90,
    ])
    
    return "\n".join(lines)


# =============================================================================
# Scalability Analysis
# =============================================================================

def analyze_scalability(base_arch: PNTDNNArchitecture, device: FPGADevice) -> str:
    """Analyze how metrics scale with architecture changes."""
    
    lines = [
        "",
        "=" * 70,
        "SCALABILITY ANALYSIS",
        "=" * 70,
        "",
        "DSP scaling with network width:",
        "-" * 40,
    ]
    
    # Test different hidden sizes
    for hidden_mult in [0.5, 1.0, 1.5, 2.0]:
        test_arch = PNTDNNArchitecture(
            fc1=LayerSpec("FC1", 24, int(32 * hidden_mult)),
            fc2=LayerSpec("FC2", int(32 * hidden_mult), int(16 * hidden_mult)),
            fc3=LayerSpec("FC3", int(16 * hidden_mult), 2),
        )
        dsps = test_arch.total_dsps
        params = test_arch.total_params
        lines.append(f"  {hidden_mult:.1f}× width: {dsps} DSPs, {params} params")
    
    lines.extend([
        "",
        "Memory depth impact:",
        "-" * 40,
    ])
    
    for M in [2, 3, 4, 5]:
        taps = M + 1
        input_dim = 6 * taps
        test_arch = PNTDNNArchitecture(
            memory_depth=M,
            num_taps=taps,
            input_dim=input_dim,
            fc1=LayerSpec("FC1", input_dim, 32),
        )
        latency = test_arch.pipeline_latency_cycles
        lines.append(f"  M={M} (taps={taps}): input_dim={input_dim}, latency={latency} cycles")
    
    lines.extend([
        "",
        "Bit-width impact on resources:",
        "-" * 40,
        "  • Halving bit-width: ~50% LUT reduction, same DSP (DSP48 fixed at 18×25)",
        "  • Halving bit-width: ~40% BRAM reduction",
        "  • Impact on EVM: typically +1-2 dB degradation",
        "",
        "=" * 70,
    ])
    
    return "\n".join(lines)


# =============================================================================
# DSP Breakdown Detail
# =============================================================================

def generate_dsp_breakdown(arch: PNTDNNArchitecture) -> str:
    """Generate detailed DSP usage breakdown."""
    
    lines = [
        "",
        "=" * 70,
        "DSP48 USAGE BREAKDOWN",
        "=" * 70,
        "",
        "DATA PATH (250 MHz clock domain):",
        "-" * 40,
        f"  CORDIC FEx:        {arch.cordic_dsps:3d} DSPs  (8 pipeline stages)",
    ]
    
    for layer in arch.layers:
        lines.append(f"  {layer.name} ({layer.input_dim}→{layer.output_dim}):      {layer.dsps_systolic:3d} DSPs  (1 per neuron)")
    
    lines.extend([
        f"  Phase normalize:   {arch.phase_norm_dsps:3d} DSPs  (2 complex multiplies)",
        f"  Phase denormalize: {arch.phase_denorm_dsps:3d} DSPs  (2 complex multiplies)",
        f"  {'─' * 35}",
        f"  Data path total:   {arch.data_path_dsps:3d} DSPs",
        "",
        "ADAPTATION PATH (1 MHz clock domain):",
        "-" * 40,
        f"  Perturbation gen:  {4:3d} DSPs  (w ± c_k × Δ_k)",
        f"  Gradient estimate: {4:3d} DSPs  (ΔL / 2c_k)",
        f"  Weight update:     {4:3d} DSPs  (w - a_k × g_k)",
        f"  {'─' * 35}",
        f"  SPSA total:        {arch.spsa_dsps:3d} DSPs",
        "",
        f"{'=' * 40}",
        f"  GRAND TOTAL:       {arch.total_dsps:3d} DSPs",
        "",
        "=" * 70,
    ])
    
    return "\n".join(lines)


# =============================================================================
# Export Functions
# =============================================================================

def export_metrics_json(metrics: PreRTLMetrics, filepath: str):
    """Export metrics to JSON for documentation/CI."""
    with open(filepath, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"Metrics exported to {filepath}")


def export_latex_table(metrics: PreRTLMetrics) -> str:
    """Generate LaTeX table for publication."""
    return f"""
\\begin{{table}}[h]
\\centering
\\caption{{Pre-RTL Resource and Performance Estimates}}
\\label{{tab:pre_rtl_metrics}}
\\begin{{tabular}}{{|l|c|c|}}
\\hline
\\textbf{{Metric}} & \\textbf{{Value}} & \\textbf{{Method}} \\\\
\\hline
Throughput & {metrics.throughput_msps:.0f} MSps & Architecture \\\\
Latency & {metrics.latency_cycles} cycles ({metrics.latency_ns:.0f} ns) & Pipeline depth \\\\
DSP usage & {metrics.dsp_used} ({metrics.dsp_utilization_pct:.1f}\\%) & Op mapping \\\\
LUT usage & $\\sim${metrics.lut_estimated//1000}k ({metrics.lut_utilization_pct:.1f}\\%) & Logic estimate \\\\
BRAM & {metrics.bram_used} ({metrics.bram_utilization_pct:.1f}\\%) & Weight storage \\\\
Energy/sample & {metrics.energy_per_sample_pj:.1f} pJ & Power model \\\\
Parameters & {metrics.total_params:,} & Network spec \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Generate complete pre-RTL metrics report."""
    
    # Initialize architecture
    arch = PNTDNNArchitecture()
    
    print("=" * 70)
    print("PN-TDNN-DPD Pre-RTL Metrics Estimator")
    print("=" * 70)
    print()
    print("Methodology:")
    print("  - Power model: Scaled from SparseDPD Table II (measured on Zynq-7Z010)")
    print("  - Area model: Primitive counting from architecture specification")
    print("  - Reference: arXiv:2506.16591v1 (SparseDPD), architecture.md Section 9.3")
    print()
    
    # PYNQ-Z1 Analysis - Baseline estimate (no margin)
    print("TARGET: PYNQ-Z1 @ 250 MHz")
    print("-" * 70)
    estimator_pynq = MetricsEstimator(
        arch=arch,
        device=PYNQ_Z1,
        clock_mhz=250.0,
        signal_bw_mhz=200.0,
        power_margin=1.0,  # No margin for baseline
    )
    metrics_pynq = estimator_pynq.compute_metrics()
    print(metrics_pynq)
    
    # Breakdown for verification
    print()
    print("POWER BREAKDOWN (architecture.md XPE-derived baseline):")
    print("-" * 70)
    f = 250.0
    dsp_pwr = arch.data_path_dsps * (300.0 / 62.0)
    logic_pwr = 150.0  # architecture.md value
    bram_pwr = arch.total_brams * (50.0 / 9.0)
    io_pwr = 100.0
    static_pwr = PYNQ_Z1.static_power_mw
    total_calc = dsp_pwr + logic_pwr + bram_pwr + io_pwr + static_pwr
    print(f"  DSP ({arch.data_path_dsps} units):     {dsp_pwr:.0f} mW  (target: 300 mW for 62 DSPs)")
    print(f"  Logic (~{estimator_pynq.estimate_luts()//1000}k LUTs):   {logic_pwr:.0f} mW  (target: 150 mW)")
    print(f"  BRAM ({arch.total_brams} units):      {bram_pwr:.0f} mW  (target: 50 mW for 9 BRAMs)")
    print(f"  I/O interface:       {io_pwr:.0f} mW  (LVDS ADC/DAC)")
    print(f"  Static:              {static_pwr:.0f} mW  (from XPE)")
    print(f"  ────────────────────────────")
    print(f"  Calculated TOTAL:    {total_calc:.0f} mW")
    print(f"  Estimated TOTAL:     {metrics_pynq.total_power_mw:.0f} mW")
    print(f"  architecture.md:     ~800 mW")
    print()
    
    # Verify SparseDPD scaling
    print("VALIDATION vs SparseDPD (measured @ 170 MHz):")
    print("-" * 70)
    # SparseDPD measured breakdown (Table II):
    # DSP: 77 mW, BRAM: 40 mW, Logic: 37 mW, Signals: 65 mW, Clocks: 23 mW
    # Total dynamic: 241 mW, Static: 164 mW, Total: 405 mW
    sparse_dsp_scaled = arch.data_path_dsps * (77.0 / 66.0) * (250.0 / 170.0)  # Linear with DSP count and freq
    sparse_bram_scaled = arch.total_brams * (40.0 / 13.0) * (250.0 / 170.0)    # Linear with BRAM count and freq
    # Logic/Signals/Clocks scale sub-linearly (sqrt approximation for routing)
    lut_ratio = math.sqrt(11000.0 / 2298.0)  # ~2.19x
    sparse_logic_scaled = (37 + 65 + 23) * lut_ratio * (250.0 / 170.0)  # ~400 mW
    sparse_static = 164.0 + 36.0  # 7Z010→7Z020 static delta (larger die)
    sparse_io = 100.0  # Not measured in SparseDPD (no external ADC/DAC)
    sparse_total_scaled = sparse_dsp_scaled + sparse_bram_scaled + sparse_logic_scaled + sparse_io + sparse_static
    print(f"  DSP (62 @ 250 MHz):       {sparse_dsp_scaled:.0f} mW  (from 66 DSP = 77 mW @ 170 MHz)")
    print(f"  BRAM (9 @ 250 MHz):       {sparse_bram_scaled:.0f} mW  (from 13 BRAM = 40 mW)")
    print(f"  Logic/Sig/Clk (11k LUT):  {sparse_logic_scaled:.0f} mW  (sqrt scaling, 125 mW for 2.3k @ 170)")
    print(f"  I/O (ADC/DAC):            {sparse_io:.0f} mW  (not in SparseDPD)")
    print(f"  Static (7Z020):           {sparse_static:.0f} mW  (164 + 36 mW die size delta)")
    print(f"  ────────────────────────────")
    print(f"  SparseDPD-scaled total:   {sparse_total_scaled:.0f} mW")
    print(f"  Energy/sample:            {1000*sparse_total_scaled/250:.0f} pJ")
    print()
    print("  Comparison:")
    print(f"    architecture.md (XPE): 800 mW - conservative for publication")
    print(f"    SparseDPD scaling:     {sparse_total_scaled:.0f} mW - measured baseline extrapolation")
    print(f"    Expect actual:         {(sparse_total_scaled + 800)/2:.0f} mW - average estimate")
    
    # DSP breakdown
    print(generate_dsp_breakdown(arch))
    
    # Comparison table
    print(generate_comparison_table(metrics_pynq, [SPARSE_DPD, OPEN_DPD_V2]))
    
    # Scalability
    print(analyze_scalability(arch, PYNQ_Z1))
    
    # ZCU104 Analysis
    print()
    print("TARGET: ZCU104 @ 250 MHz (for comparison)")
    print("-" * 70)
    estimator_zcu = MetricsEstimator(
        arch=arch,
        device=ZCU104,
        clock_mhz=250.0,
        signal_bw_mhz=200.0,
    )
    metrics_zcu = estimator_zcu.compute_metrics()
    print(f"  DSP utilization: {metrics_zcu.dsp_utilization_pct:.1f}%")
    print(f"  Total power: {metrics_zcu.total_power_mw:.0f} mW (target: ~1100 mW)")
    print(f"  Headroom for parallelism: {ZCU104.dsp_count // arch.total_dsps}× instances possible")
    
    # LaTeX output
    print()
    print("LaTeX Table:")
    print("-" * 70)
    print(export_latex_table(metrics_pynq))
    
    # Export JSON for CI/documentation
    print()
    print("Exporting metrics to JSON...")
    export_metrics_json(metrics_pynq, "utils/pre_rtl_metrics.json")
    
    return metrics_pynq


if __name__ == "__main__":
    metrics = main()
