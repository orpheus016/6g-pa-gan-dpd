##==============================================================================
## PYNQ-Z1 Constraints for 6G PA DPD System - DIGITAL DEMO VERSION
## Target: Zynq-7020 (XC7Z020-1CLG400C)
##==============================================================================
## 
## CONTEST DEMO SETUP (No RF Equipment Required!)
## ===============================================
## This configuration uses:
## - Digital I/Q from PS (AXI) - no ADC needed
## - PA Digital Twin in PL - no real PA needed  
## - Software temperature control - no temp sensor needed
## - VGA/HDMI output for visualization
##
## Data Flow:
##   Laptop → USB/Ethernet → PS ARM → AXI → DPD PL → PA Twin PL → AXI → PS → Display
##
##==============================================================================

##==============================================================================
## Clock Constraints (from PS FCLK)
##==============================================================================

# PS provides clocks via FCLK - no external oscillator needed
# FCLK0 = 200 MHz (NN inference)
# FCLK1 = 100 MHz (AXI interface)  
# FCLK2 = 1 MHz (A-SPSA adaptation)

create_clock -period 5.000 -name clk_200 [get_pins PS7_inst/FCLKCLK[0]]
create_clock -period 10.000 -name clk_100 [get_pins PS7_inst/FCLKCLK[1]]
create_clock -period 1000.000 -name clk_1 [get_pins PS7_inst/FCLKCLK[2]]

##==============================================================================
## Clock Domain Crossing
##==============================================================================

set_clock_groups -asynchronous \
    -group [get_clocks clk_200] \
    -group [get_clocks clk_1]

set_clock_groups -asynchronous \
    -group [get_clocks clk_100] \
    -group [get_clocks clk_200]

# Shadow memory CDC
set_max_delay -from [get_clocks clk_1] -to [get_clocks clk_200] -datapath_only 8.0

##==============================================================================
## User Interface - Buttons (directly usable, active low)
##==============================================================================

# BTN0 - DPD Enable/Bypass toggle
set_property PACKAGE_PIN D19 [get_ports btn_dpd_enable]
set_property IOSTANDARD LVCMOS33 [get_ports btn_dpd_enable]

# BTN1 - Adaptation Enable toggle  
set_property PACKAGE_PIN D20 [get_ports btn_adapt_enable]
set_property IOSTANDARD LVCMOS33 [get_ports btn_adapt_enable]

# BTN2 - Temperature state cycle (Cold→Normal→Hot→Cold)
set_property PACKAGE_PIN L20 [get_ports btn_temp_cycle]
set_property IOSTANDARD LVCMOS33 [get_ports btn_temp_cycle]

# BTN3 - Reset / Re-initialize weights
set_property PACKAGE_PIN L19 [get_ports btn_reset]
set_property IOSTANDARD LVCMOS33 [get_ports btn_reset]

##==============================================================================
## User Interface - Switches (directly usable)
##==============================================================================

# SW0-SW1: Temperature state override (manual control for demo)
# 00 = Auto (use software temp), 01 = Cold, 10 = Normal, 11 = Hot
set_property PACKAGE_PIN M20 [get_ports {sw_temp[0]}]
set_property PACKAGE_PIN M19 [get_ports {sw_temp[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {sw_temp[*]}]

##==============================================================================
## Status LEDs (accent indicators)
##==============================================================================

# LD0 - DPD Active (Green when DPD enabled)
set_property PACKAGE_PIN R14 [get_ports led_dpd_active]
set_property IOSTANDARD LVCMOS33 [get_ports led_dpd_active]

# LD1 - Adaptation Active (Green when adapting)
set_property PACKAGE_PIN P14 [get_ports led_adapt_active]
set_property IOSTANDARD LVCMOS33 [get_ports led_adapt_active]

# LD2 - Temperature Cold indicator
set_property PACKAGE_PIN N16 [get_ports led_temp_cold]
set_property IOSTANDARD LVCMOS33 [get_ports led_temp_cold]

# LD3 - Temperature Hot indicator
set_property PACKAGE_PIN M14 [get_ports led_temp_hot]
set_property IOSTANDARD LVCMOS33 [get_ports led_temp_hot]

##==============================================================================
## RGB LEDs - Error Level Visualization
##==============================================================================

# RGB0 - EVM level (Green=good, Yellow=medium, Red=bad)
set_property PACKAGE_PIN L15 [get_ports led_rgb0_r]
set_property PACKAGE_PIN G17 [get_ports led_rgb0_g]
set_property PACKAGE_PIN N15 [get_ports led_rgb0_b]
set_property IOSTANDARD LVCMOS33 [get_ports led_rgb0_*]

# RGB1 - Adaptation convergence indicator
set_property PACKAGE_PIN G14 [get_ports led_rgb1_r]
set_property PACKAGE_PIN L14 [get_ports led_rgb1_g]
set_property PACKAGE_PIN M15 [get_ports led_rgb1_b]
set_property IOSTANDARD LVCMOS33 [get_ports led_rgb1_*]

##==============================================================================
## HDMI Output (directly usable on PYNQ-Z1)
## Used for: Real-time constellation/spectrum visualization
##==============================================================================

# HDMI accent accent accent accent accent accent accent accent accent accent accent accent
set_property PACKAGE_PIN H17 [get_ports hdmi_clk_p]
set_property PACKAGE_PIN H18 [get_ports hdmi_clk_n]
set_property PACKAGE_PIN D19 [get_ports {hdmi_d_p[0]}]
set_property PACKAGE_PIN D20 [get_ports {hdmi_d_n[0]}]
set_property PACKAGE_PIN C20 [get_ports {hdmi_d_p[1]}]
set_property PACKAGE_PIN B20 [get_ports {hdmi_d_n[1]}]
set_property PACKAGE_PIN B19 [get_ports {hdmi_d_p[2]}]
set_property PACKAGE_PIN A20 [get_ports {hdmi_d_n[2]}]

set_property IOSTANDARD TMDS_33 [get_ports hdmi_*]

##==============================================================================
## AXI Interface (directly connected via PS - no pins needed)
## All I/Q data flows through AXI from PS ARM core
##==============================================================================

# AXI HP ports are internal to Zynq - no external pins
# Configuration done in block design

##==============================================================================
## PMOD Headers - Optional External Interface
## Can be used for: Logic analyzer debug, external triggers
##==============================================================================

# PMOD JA - Debug/trigger outputs (directly usable)
set_property PACKAGE_PIN Y18 [get_ports {pmod_ja[0]}]
set_property PACKAGE_PIN Y19 [get_ports {pmod_ja[1]}]
set_property PACKAGE_PIN Y16 [get_ports {pmod_ja[2]}]
set_property PACKAGE_PIN Y17 [get_ports {pmod_ja[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {pmod_ja[*]}]

##==============================================================================
## Timing Constraints
##==============================================================================

# AXI interface timing (internal)
set_input_delay -clock clk_100 -max 2.0 [get_ports {s_axi_*}]
set_output_delay -clock clk_100 -max 2.0 [get_ports {m_axi_*}]

##==============================================================================
## Configuration
##==============================================================================

set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
