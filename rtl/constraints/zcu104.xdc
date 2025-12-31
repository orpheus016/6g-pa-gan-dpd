##==============================================================================
## ZCU104 Constraints for 6G PA DPD System (Production)
## Target: Zynq UltraScale+ (XCZU7EV-2FFVC1156)
##==============================================================================

##==============================================================================
## Clock Constraints
##==============================================================================

# User clock (300 MHz differential)
set_property PACKAGE_PIN G21 [get_ports user_si570_clk_p]
set_property PACKAGE_PIN F21 [get_ports user_si570_clk_n]
set_property IOSTANDARD DIFF_SSTL12 [get_ports user_si570_clk_*]

create_clock -period 3.333 -name user_clk [get_ports user_si570_clk_p]

# PL system clock (125 MHz)
set_property PACKAGE_PIN H9 [get_ports clk_125_p]
set_property PACKAGE_PIN G9 [get_ports clk_125_n]
set_property IOSTANDARD DIFF_SSTL12 [get_ports clk_125_*]

create_clock -period 8.000 -name clk_125 [get_ports clk_125_p]

# Generated clocks from MMCM
create_generated_clock -name clk_200 -source [get_pins clk_wiz_inst/clk_in1] \
    -multiply_by 8 -divide_by 5 [get_pins clk_wiz_inst/clk_out1]

create_generated_clock -name clk_400 -source [get_pins clk_wiz_inst/clk_in1] \
    -multiply_by 16 -divide_by 5 [get_pins clk_wiz_inst/clk_out2]

create_generated_clock -name clk_1 -source [get_pins clk_wiz_inst/clk_in1] \
    -multiply_by 1 -divide_by 125 [get_pins clk_wiz_inst/clk_out3]

##==============================================================================
## Clock Domain Crossing Constraints
##==============================================================================

# Asynchronous clock groups
set_clock_groups -asynchronous \
    -group [get_clocks clk_200] \
    -group [get_clocks clk_1]

# Shadow memory CDC paths
set_max_delay -from [get_clocks clk_1] -to [get_clocks clk_200] \
    -datapath_only 8.0

set_max_delay -from [get_clocks clk_200] -to [get_clocks clk_1] \
    -datapath_only 8.0

# Gray code bus skew
set_bus_skew -from [get_pins */shadow_memory_inst/wr_ptr_gray_reg*/C] \
    -to [get_pins */shadow_memory_inst/rd_ptr_sync*/D] 0.5

##==============================================================================
## FMC HPC0 - High-Speed ADC Interface (JESD204B or LVDS)
##==============================================================================

# JESD204B lanes (4 lanes, 10 Gbps each)
set_property PACKAGE_PIN D2 [get_ports {fmc_dp_m2c_p[0]}]
set_property PACKAGE_PIN D1 [get_ports {fmc_dp_m2c_n[0]}]
set_property PACKAGE_PIN E4 [get_ports {fmc_dp_m2c_p[1]}]
set_property PACKAGE_PIN E3 [get_ports {fmc_dp_m2c_n[1]}]
set_property PACKAGE_PIN F2 [get_ports {fmc_dp_m2c_p[2]}]
set_property PACKAGE_PIN F1 [get_ports {fmc_dp_m2c_n[2]}]
set_property PACKAGE_PIN G4 [get_ports {fmc_dp_m2c_p[3]}]
set_property PACKAGE_PIN G3 [get_ports {fmc_dp_m2c_n[3]}]

# ADC reference clock (via FMC)
set_property PACKAGE_PIN K6 [get_ports fmc_gbtclk0_m2c_p]
set_property PACKAGE_PIN K5 [get_ports fmc_gbtclk0_m2c_n]

create_clock -period 4.000 -name adc_refclk [get_ports fmc_gbtclk0_m2c_p]

# LVDS data inputs (alternative parallel interface)
set_property PACKAGE_PIN AK17 [get_ports {adc_data_p[0]}]
set_property PACKAGE_PIN AK16 [get_ports {adc_data_n[0]}]
set_property PACKAGE_PIN AJ15 [get_ports {adc_data_p[1]}]
set_property PACKAGE_PIN AK15 [get_ports {adc_data_n[1]}]
set_property PACKAGE_PIN AG14 [get_ports {adc_data_p[2]}]
set_property PACKAGE_PIN AH14 [get_ports {adc_data_n[2]}]
set_property PACKAGE_PIN AF15 [get_ports {adc_data_p[3]}]
set_property PACKAGE_PIN AG15 [get_ports {adc_data_n[3]}]
set_property PACKAGE_PIN AE15 [get_ports {adc_data_p[4]}]
set_property PACKAGE_PIN AF14 [get_ports {adc_data_n[4]}]
set_property PACKAGE_PIN AH13 [get_ports {adc_data_p[5]}]
set_property PACKAGE_PIN AJ13 [get_ports {adc_data_n[5]}]
set_property PACKAGE_PIN AK12 [get_ports {adc_data_p[6]}]
set_property PACKAGE_PIN AK13 [get_ports {adc_data_n[6]}]
set_property PACKAGE_PIN AH12 [get_ports {adc_data_p[7]}]
set_property PACKAGE_PIN AJ12 [get_ports {adc_data_n[7]}]

set_property IOSTANDARD LVDS [get_ports {adc_data_*}]
set_property DIFF_TERM_ADV TERM_100 [get_ports {adc_data_*}]

##==============================================================================
## FMC HPC0 - High-Speed DAC Interface (JESD204B or LVDS)
##==============================================================================

# JESD204B TX lanes
set_property PACKAGE_PIN D6 [get_ports {fmc_dp_c2m_p[0]}]
set_property PACKAGE_PIN D5 [get_ports {fmc_dp_c2m_n[0]}]
set_property PACKAGE_PIN C4 [get_ports {fmc_dp_c2m_p[1]}]
set_property PACKAGE_PIN C3 [get_ports {fmc_dp_c2m_n[1]}]
set_property PACKAGE_PIN B6 [get_ports {fmc_dp_c2m_p[2]}]
set_property PACKAGE_PIN B5 [get_ports {fmc_dp_c2m_n[2]}]
set_property PACKAGE_PIN A4 [get_ports {fmc_dp_c2m_p[3]}]
set_property PACKAGE_PIN A3 [get_ports {fmc_dp_c2m_n[3]}]

# LVDS data outputs (alternative)
set_property PACKAGE_PIN AD15 [get_ports {dac_data_p[0]}]
set_property PACKAGE_PIN AD14 [get_ports {dac_data_n[0]}]
set_property PACKAGE_PIN AB15 [get_ports {dac_data_p[1]}]
set_property PACKAGE_PIN AB14 [get_ports {dac_data_n[1]}]
set_property PACKAGE_PIN AA15 [get_ports {dac_data_p[2]}]
set_property PACKAGE_PIN AA14 [get_ports {dac_data_n[2]}]
set_property PACKAGE_PIN Y14 [get_ports {dac_data_p[3]}]
set_property PACKAGE_PIN Y13 [get_ports {dac_data_n[3]}]
set_property PACKAGE_PIN W15 [get_ports {dac_data_p[4]}]
set_property PACKAGE_PIN W14 [get_ports {dac_data_n[4]}]
set_property PACKAGE_PIN V14 [get_ports {dac_data_p[5]}]
set_property PACKAGE_PIN V13 [get_ports {dac_data_n[5]}]
set_property PACKAGE_PIN U15 [get_ports {dac_data_p[6]}]
set_property PACKAGE_PIN U14 [get_ports {dac_data_n[6]}]
set_property PACKAGE_PIN T15 [get_ports {dac_data_p[7]}]
set_property PACKAGE_PIN T14 [get_ports {dac_data_n[7]}]

set_property IOSTANDARD LVDS [get_ports {dac_data_*}]

##==============================================================================
## PA Feedback Path
##==============================================================================

# Feedback ADC (2nd ADC channel or shared with main ADC)
set_property PACKAGE_PIN AL17 [get_ports {fb_adc_data_p[0]}]
set_property PACKAGE_PIN AL16 [get_ports {fb_adc_data_n[0]}]
# ... additional pins as needed

set_property IOSTANDARD LVDS [get_ports {fb_adc_data_*}]

##==============================================================================
## Temperature Sensor Interface (I2C to LM75 or similar)
##==============================================================================

set_property PACKAGE_PIN J24 [get_ports temp_scl]
set_property PACKAGE_PIN J25 [get_ports temp_sda]
set_property IOSTANDARD LVCMOS18 [get_ports temp_*]
set_property PULLUP true [get_ports temp_sda]

# SYSMON internal temperature (backup)
# Accessed via AXI, no external pin needed

##==============================================================================
## User GPIO - Control and Status
##==============================================================================

# Push buttons
set_property PACKAGE_PIN B4 [get_ports btn_dpd_enable]
set_property PACKAGE_PIN C4 [get_ports btn_adapt_enable]
set_property IOSTANDARD LVCMOS12 [get_ports btn_*]

# DIP switches
set_property PACKAGE_PIN AN16 [get_ports {sw_ctrl[0]}]
set_property PACKAGE_PIN AN15 [get_ports {sw_ctrl[1]}]
set_property PACKAGE_PIN AM14 [get_ports {sw_ctrl[2]}]
set_property PACKAGE_PIN AP15 [get_ports {sw_ctrl[3]}]
set_property IOSTANDARD LVCMOS12 [get_ports {sw_ctrl[*]}]

# Status LEDs
set_property PACKAGE_PIN D5 [get_ports led_dpd_active]
set_property PACKAGE_PIN D6 [get_ports led_adapt_active]
set_property PACKAGE_PIN A5 [get_ports led_error]
set_property PACKAGE_PIN B5 [get_ports led_heartbeat]
set_property IOSTANDARD LVCMOS12 [get_ports led_*]

##==============================================================================
## Timing Constraints
##==============================================================================

# ADC input timing (assuming JESD204B with deterministic latency)
set_input_delay -clock adc_refclk -max 0.5 [get_ports {adc_data_*}]
set_input_delay -clock adc_refclk -min 0.0 [get_ports {adc_data_*}]

# DAC output timing
set_output_delay -clock clk_400 -max 0.5 [get_ports {dac_data_*}]
set_output_delay -clock clk_400 -min 0.0 [get_ports {dac_data_*}]

# Multicycle paths for weight update (slow path)
set_multicycle_path 2 -setup -from [get_clocks clk_1] -to [get_clocks clk_200]
set_multicycle_path 1 -hold -from [get_clocks clk_1] -to [get_clocks clk_200]

##==============================================================================
## Physical Constraints
##==============================================================================

# Place MMCM near GTH transceivers
set_property LOC MMCME4_ADV_X0Y1 [get_cells clk_wiz_inst/inst/mmcme4_adv_inst]

# DSP cascade for MAC operations (place near BRAM)
set_property LOC DSP48E2_X0Y0 [get_cells {tdnn_inst/mac_inst/dsp48_inst}]

# Weight BRAM placement
set_property LOC RAMB36_X1Y10 [get_cells {weight_mem_inst/ram_inst}]

# UltraRAM for larger models (if needed)
# set_property LOC URAM288_X0Y0 [get_cells {uram_inst}]

##==============================================================================
## Power and Thermal Constraints
##==============================================================================

# Voltage configuration
set_property INTERNAL_VREF 0.90 [get_iobanks 64]
set_property INTERNAL_VREF 0.90 [get_iobanks 65]

# Power optimization
set_property BITSTREAM.CONFIG.OVERTEMPSHUTDOWN ENABLE [current_design]

##==============================================================================
## Debug (ILA) Constraints
##==============================================================================

# Debug hub for ILA cores
set_property C_CLK_INPUT_FREQ_HZ 200000000 [get_debug_cores dbg_hub]
set_property C_ENABLE_CLK_DIVIDER false [get_debug_cores dbg_hub]
set_property C_USER_SCAN_CHAIN 1 [get_debug_cores dbg_hub]

# Connect debug clock
# connect_debug_port dbg_hub/clk [get_nets clk_200]

##==============================================================================
## Configuration Settings
##==============================================================================

set_property CONFIG_VOLTAGE 1.8 [current_design]
set_property CFGBVS GND [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 85.0 [current_design]
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
