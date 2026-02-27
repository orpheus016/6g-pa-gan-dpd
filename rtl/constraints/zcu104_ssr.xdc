##==============================================================================
## ZCU104 Constraints for 6G PA DPD System - SSR Architecture
## Target: Zynq UltraScale+ (XCZU7EV-2FFVC1156)
## Single 200 MHz clock domain with SSR=5
##==============================================================================

##==============================================================================
## Clock Constraints
##==============================================================================

# User clock (200 MHz - can be derived from Si570 or PS clock)
# Using PS-supplied clock for simplicity
create_clock -period 5.000 -name clk_200 [get_ports clk]

# Clock uncertainty for timing analysis
set_clock_uncertainty 0.100 [get_clocks clk_200]

##==============================================================================
## Input/Output Timing Constraints
##==============================================================================

# Input timing (ADC interface)
# Assume 1ns setup, 0.5ns hold relative to clock
set_input_delay -clock clk_200 -max 1.000 [get_ports {in_i[*] in_q[*] in_valid}]
set_input_delay -clock clk_200 -min 0.500 [get_ports {in_i[*] in_q[*] in_valid}]

# Output timing (DAC interface)
# Assume 1ns setup, 0.5ns hold at destination
set_output_delay -clock clk_200 -max 1.000 [get_ports {out_i_*[*] out_q_*[*] out_valid}]
set_output_delay -clock clk_200 -min 0.500 [get_ports {out_i_*[*] out_q_*[*] out_valid}]

# Temperature bank select (slow-changing, can be relaxed)
set_false_path -from [get_ports temp_bank_sel[*]]

##==============================================================================
## Area Constraints (Optional - for floorplanning)
##==============================================================================

# Keep parallel lanes close together for routing
# Uncomment and adjust if needed for timing closure

# create_pblock pblock_interpolator
# resize_pblock pblock_interpolator -add {SLICE_X0Y0:SLICE_X50Y99}
# add_cells_to_pblock pblock_interpolator [get_cells u_interpolator]

# create_pblock pblock_lanes
# resize_pblock pblock_lanes -add {SLICE_X51Y0:SLICE_X200Y299}
# add_cells_to_pblock pblock_lanes [get_cells gen_lane[*].u_*]

##==============================================================================
## DSP Utilization Hints
##==============================================================================

# Allow aggressive DSP inference for MAC operations
# set_property DSP_SLICE_UTILIZATION 90 [current_design]

##==============================================================================
## Timing Exceptions
##==============================================================================

# Multicycle path for weight ROM (if needed)
# Weight addresses change slowly compared to data processing
# set_multicycle_path 2 -setup -from [get_cells u_weight_rom/*] -to [get_cells gen_lane[*].u_tdnn/*]

##==============================================================================
## Reset Path
##==============================================================================

# Reset is asynchronous assertion, synchronous deassertion
# Allow more time for reset distribution
set_false_path -from [get_ports rst_n]

##==============================================================================
## Debug Constraints (ILA)
##==============================================================================

# If using ILA for debug, constrain debug hub clock
# set_property C_CLK_INPUT_FREQ_HZ 200000000 [get_debug_cores dbg_hub]
# set_property C_ENABLE_CLK_DIVIDER false [get_debug_cores dbg_hub]

##==============================================================================
## Physical Constraints (I/O Standards)
##==============================================================================

# Default I/O standard for data ports (adjust based on actual board routing)
# These are placeholders - update with actual pin assignments

# set_property IOSTANDARD LVCMOS18 [get_ports {in_i[*] in_q[*]}]
# set_property IOSTANDARD LVCMOS18 [get_ports {out_i_*[*] out_q_*[*]}]
# set_property IOSTANDARD LVCMOS18 [get_ports {in_valid out_valid}]
# set_property IOSTANDARD LVCMOS18 [get_ports {rst_n}]
# set_property IOSTANDARD LVCMOS18 [get_ports {temp_bank_sel[*]}]
# set_property IOSTANDARD LVCMOS18 [get_ports {busy}]

##==============================================================================
## Report Configuration
##==============================================================================

# Generate detailed timing reports during implementation
# report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose \
#     -max_paths 10 -input_pins -routable_nets -name timing_1
