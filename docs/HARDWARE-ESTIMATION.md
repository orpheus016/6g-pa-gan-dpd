**Architecture**: IQ(200MHz) → FEX(23 cyc) → TDNN(~55 cyc) = ~390ns total latency. Interpolator instantiated but bypassed pending.

**Why this design**:
- **FEX→TDNN direct path**: Works now at 200 MSps, provides accurate timing data
- **Interpolator included**: Synthesis sees resource cost (DSPs/LUTs) even though not in critical path yet
- **Clean interfaces**: Weight BRAM external (reusable), latency counter for debug

**Vivado workflow**:
1. Add sources: `dpd_fast_path.v`, `fex_layer_ii1_fixed.v`, `interpolator_skeleton.v`, `tdnn_generator.v`
2. Create clock constraint: `create_clock -period 5.0 [get_ports clk]` (200 MHz)
3. Synthesize → Check Fmax (should be >200 MHz with 80 MHz margin vs 280 MHz TDNN limit)
4. Implement → Power report

**Resource estimates**: ~60 DSPs (FEX sqrt + TDNN MACs), ~10 BRAMs (1362 weights × 3 banks), ~15k LUTs.