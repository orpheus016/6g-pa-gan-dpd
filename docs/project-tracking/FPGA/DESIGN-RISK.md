You have valid reasons to be concerned, but both issues are solvable if you handle them with specific architectural "Band-Aids." You are pushing the Zynq-7020 to its limit (250 MHz is fast for Artix-7 fabric), so "lazy" coding will fail timing.

Here is the **Viability Fix List** to ensure your design survives synthesis and works in the real world.

### 1. The CDC Fear: "How do I update weights without glitches?"

**The Risk:** You have a fast Datapath (250 MHz) and a slow SPSA engine (maybe 10-50 MHz or state-machine based). If you update the 1,300 weights *while* a signal is passing through, you will get a "Frankenstein" multiplication (half old weight, half new weight) that creates a massive noise spike (glitch).

**The Solution:** Use **Shadow Registers with a "Safe Window" Handshake.**
You do *not* need complex Async FIFOs because you aren't streaming weights continuously. You update them in "bursts" (once per SPSA iteration).

* **Step 1: Double Buffer the Weights.**
* Inside every PE (Processing Element), have two registers: `active_weight` (used for math) and `shadow_weight` (connected to the update bus).


* **Step 2: The "Global Update" Signal.**
* The SPSA engine writes to the `shadow_weight` registers slowly. It takes its time. No rush.
* Once all shadow weights are stable, the SPSA engine asserts a single **`global_update_req`** flag.


* **Step 3: The "Dead Zone" Latch.**
* You are processing data in packets or frames (e.g., OFDM symbols). There is always a "Guard Interval" or "Cyclic Prefix" gap where data is invalid or zero.
* The Systolic Array state machine waits for this gap. When `gap == 1` AND `update_req == 1`, it toggles a `load_en` signal.
* *Result:* All 1,300 weights flip instantly in 1 clock cycle during the silence. Zero glitches.



### 2. The Routing Congestion Fear: "Will it fit on Zynq-7020?"

**The Risk:** A "Broadcast" Systolic Array (where `x[n]` goes to everyone at once) will fail timing at 250 MHz on a 7020. The fanout is too high (~1300 loads). The wires will be too long.

**The Solution:** **True Systolic (Pipeling the Input) + DSP Cascading.**

#### **Fix A: Kill the Broadcast (The "Bucket Brigade")**

Do not send `x[n]` to PE #1, PE #2... PE #64 simultaneously.

* **Bad:** `x[n] -> [PE1, PE2, PE3...]` (Fanout = 64, Timing Fail).
* **Good:** `x[n] -> PE1 -> (reg) -> PE2 -> (reg) -> PE3...`
* Pass the input data (`x`) from neighbor to neighbor, just like the partial sums.
* **Latency Penalty:** Yes, this adds latency (Diagonal Wavefront).
* **Timing Reward:** Your max wire length is now ~0.1mm (neighbor to neighbor). You will easily hit 250 MHz.



#### **Fix B: Use the "Magic" Wires (DSP48 Cascade)**

The Zynq-7020 DSP48 slices have dedicated silicon highways called `PCIN` and `PCOUT`. These wires do **not** use the FPGA fabric. They are extremely fast (500 MHz+).

* **The Check:** Open your Vivado "Implementation" view. Look at your DSPs.
* *If you see blue lines (Fabric Routing) between DSPs:* **You failed.**
* *If you see no lines (because they are internal):* **You won.**


* **The Code:** To force this in Verilog, ensure your accumulation happens in a single `always` block so Vivado infers the cascade.
```verilog
// Vivado likely infers PCOUT if you structure it like a chain:
assign p_out = p_in + (a * b); // This often maps to DSP48 PCOUT

```



### 3. Summary of Viability

| Feature | Risk Level | The Fix |
| --- | --- | --- |
| **Logic Area** | Medium | Zynq-7020 has ~220 DSPs. Your PN-TDNN likely uses <100. You are safe. |
| **Routing** | **Critical** | **Pipeline the input `x**`. Never broadcast. Use DSP cascading. |
| **CDC (Weights)** | High | **Shadow Registers**. Update only during the "Guard Interval". |
| **Clock Speed** | High | 250 MHz is the limit for Artix-7 fabric. **Pipeline EVERYTHING.** No logic depth > 1 LUT between registers. |

**Final Advice:**
Your design is viable **ONLY IF** you pipeline the inputs (`x`). If you try to broadcast `x` to 64 neurons, the Zynq-7020 will fail timing. Make it a "True" Systolic array where *everything* moves one step per cycle.