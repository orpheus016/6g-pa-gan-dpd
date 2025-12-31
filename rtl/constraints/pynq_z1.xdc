##==============================================================================
## PYNQ-Z1 Constraints for 6G PA DPD System
## Target: Zynq-7020 (XC7Z020-1CLG400C)
##==============================================================================

##==============================================================================
## Clock Constraints
##==============================================================================

# System clock (125 MHz from PS)
create_clock -period 8.000 -name clk_125 [get_ports clk_125]

# Generated clocks (from MMCM/PLL)
# 200 MHz NN clock
create_generated_clock -name clk_200 -source [get_pins mmcm_inst/CLKIN1] \
    -multiply_by 8 -divide_by 5 [get_pins mmcm_inst/CLKOUT0]

# 400 MHz PA output clock (DDR or serializer)
create_generated_clock -name clk_400 -source [get_pins mmcm_inst/CLKIN1] \
    -multiply_by 16 -divide_by 5 [get_pins mmcm_inst/CLKOUT1]

# 1 MHz adaptation clock
create_generated_clock -name clk_1 -source [get_pins mmcm_inst/CLKIN1] \
    -multiply_by 1 -divide_by 125 [get_pins mmcm_inst/CLKOUT2]

##==============================================================================
## Clock Domain Crossing Constraints
##==============================================================================

# False paths for CDC synchronizers
set_false_path -from [get_clocks clk_200] -to [get_clocks clk_1]
set_false_path -from [get_clocks clk_1] -to [get_clocks clk_200]

# Max delay for shadow memory CDC
set_max_delay -from [get_clocks clk_1] -to [get_clocks clk_200] \
    -datapath_only 10.0

# Gray code constraints for FIFO pointers
set_bus_skew -from [get_pins shadow_mem_inst/wr_ptr_gray_reg*] \
    -to [get_pins shadow_mem_inst/rd_ptr_sync*] 1.0

##==============================================================================
## Pin Assignments - PMOD JA (ADC Input)
##==============================================================================

# PMOD JA - I/Q ADC Interface (directly or via Pmod ADC)
set_property PACKAGE_PIN Y18 [get_ports {adc_data_i[0]}]
set_property PACKAGE_PIN Y19 [get_ports {adc_data_i[1]}]
set_property PACKAGE_PIN Y16 [get_ports {adc_data_i[2]}]
set_property PACKAGE_PIN Y17 [get_ports {adc_data_i[3]}]
set_property PACKAGE_PIN U18 [get_ports {adc_data_i[4]}]
set_property PACKAGE_PIN U19 [get_ports {adc_data_i[5]}]
set_property PACKAGE_PIN W18 [get_ports {adc_data_i[6]}]
set_property PACKAGE_PIN W19 [get_ports {adc_data_i[7]}]

set_property IOSTANDARD LVCMOS33 [get_ports {adc_data_i[*]}]

##==============================================================================
## Pin Assignments - PMOD JB (DAC Output)
##==============================================================================

# PMOD JB - I/Q DAC Interface
set_property PACKAGE_PIN W14 [get_ports {dac_data_i[0]}]
set_property PACKAGE_PIN Y14 [get_ports {dac_data_i[1]}]
set_property PACKAGE_PIN T11 [get_ports {dac_data_i[2]}]
set_property PACKAGE_PIN T10 [get_ports {dac_data_i[3]}]
set_property PACKAGE_PIN V16 [get_ports {dac_data_i[4]}]
set_property PACKAGE_PIN W16 [get_ports {dac_data_i[5]}]
set_property PACKAGE_PIN V12 [get_ports {dac_data_i[6]}]
set_property PACKAGE_PIN W13 [get_ports {dac_data_i[7]}]

set_property IOSTANDARD LVCMOS33 [get_ports {dac_data_i[*]}]

##==============================================================================
## Pin Assignments - Buttons and Switches
##==============================================================================

# Buttons
set_property PACKAGE_PIN D19 [get_ports btn_dpd_enable]
set_property PACKAGE_PIN D20 [get_ports btn_adapt_enable]
set_property PACKAGE_PIN L20 [get_ports btn_temp_override]
set_property PACKAGE_PIN L19 [get_ports btn_reset]

set_property IOSTANDARD LVCMOS33 [get_ports btn_*]

# Switches
set_property PACKAGE_PIN M20 [get_ports {sw_temp_sel[0]}]
set_property PACKAGE_PIN M19 [get_ports {sw_temp_sel[1]}]

set_property IOSTANDARD LVCMOS33 [get_ports {sw_temp_sel[*]}]

##==============================================================================
## Pin Assignments - LEDs (Status)
##==============================================================================

set_property PACKAGE_PIN R14 [get_ports led_dpd_active]
set_property PACKAGE_PIN P14 [get_ports led_adapt_active]
set_property PACKAGE_PIN N16 [get_ports led_temp_cold]
set_property PACKAGE_PIN M14 [get_ports led_temp_hot]

set_property IOSTANDARD LVCMOS33 [get_ports led_*]

##==============================================================================
## Pin Assignments - RGB LEDs (Error Level)
##==============================================================================

set_property PACKAGE_PIN L15 [get_ports led_rgb0_r]
set_property PACKAGE_PIN G17 [get_ports led_rgb0_g]
set_property PACKAGE_PIN N15 [get_ports led_rgb0_b]
set_property PACKAGE_PIN G14 [get_ports led_rgb1_r]
set_property PACKAGE_PIN L14 [get_ports led_rgb1_g]
set_property PACKAGE_PIN M15 [get_ports led_rgb1_b]

set_property IOSTANDARD LVCMOS33 [get_ports led_rgb*]

##==============================================================================
## Pin Assignments - Temperature Sensor (I2C or Analog)
##==============================================================================

# Assuming analog temp sensor on XADC
set_property PACKAGE_PIN E17 [get_ports temp_vp]
set_property PACKAGE_PIN D18 [get_ports temp_vn]

set_property IOSTANDARD ANALOG [get_ports temp_*]

##==============================================================================
## Timing Constraints
##==============================================================================

# Input delay for ADC data (assume 2ns setup from ADC)
set_input_delay -clock clk_200 -max 2.5 [get_ports {adc_data_i[*]}]
set_input_delay -clock clk_200 -min 0.5 [get_ports {adc_data_i[*]}]

# Output delay for DAC data
set_output_delay -clock clk_400 -max 1.5 [get_ports {dac_data_i[*]}]
set_output_delay -clock clk_400 -min 0.5 [get_ports {dac_data_i[*]}]

##==============================================================================
## Physical Constraints
##==============================================================================

# MMCM placement (near clock input)
set_property LOC MMCME2_ADV_X0Y0 [get_cells mmcm_inst]

# Weight BRAM placement (near DSP columns)
set_property LOC RAMB36_X0Y5 [get_cells weight_bram_inst/RAMB36E1_inst]

##==============================================================================
## Power Constraints
##==============================================================================

# CFGBVS for 3.3V banks
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

##==============================================================================
## Debug Hub (Optional - for ILA)
##==============================================================================

# Uncomment for debug
# set_property C_CLK_INPUT_FREQ_HZ 200000000 [get_debug_cores dbg_hub]
# set_property C_ENABLE_CLK_DIVIDER false [get_debug_cores dbg_hub]
