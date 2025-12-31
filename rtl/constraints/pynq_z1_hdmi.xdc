##==============================================================================
## PYNQ-Z1 Constraints for 6G PA DPD System (Digital Demo)
## Target: Zynq-7020 (XC7Z020-1CLG400C)
##
## DEMO CONFIGURATION: No ADC/DAC - Pure Digital Loopback via AXI
##==============================================================================

##==============================================================================
## System Clock (125 MHz from Ethernet PHY oscillator)
##==============================================================================
set_property PACKAGE_PIN H16 [get_ports clk_125]
set_property IOSTANDARD LVCMOS33 [get_ports clk_125]
create_clock -period 8.000 -name clk_125 [get_ports clk_125]

##==============================================================================
## Generated Clocks (from MMCM/PLL in PS or PL)
##
## For digital demo, we primarily use:
## - clk_200: NN inference domain
## - clk_1: A-SPSA adaptation domain
## - clk_148_5: HDMI pixel clock (1080p30 or 720p60)
##==============================================================================

# MMCM outputs (generated in block design)
# clk_200 = 200 MHz (NN inference)
# clk_1 = 1 MHz (adaptation)

##==============================================================================
## Clock Domain Crossing Constraints
##==============================================================================

# Async groups between NN and adaptation domains
set_clock_groups -asynchronous \
    -group [get_clocks -include_generated_clocks clk_200] \
    -group [get_clocks -include_generated_clocks clk_1]

# Max delay for shadow memory CDC
set_max_delay -from [get_clocks clk_1] -to [get_clocks clk_200] \
    -datapath_only 10.0
set_max_delay -from [get_clocks clk_200] -to [get_clocks clk_1] \
    -datapath_only 10.0

##==============================================================================
## HDMI Interface (Main I/O for Demo)
##==============================================================================

## HDMI RX (Optional - for receiving test signal from laptop)
#set_property PACKAGE_PIN Y18 [get_ports hdmi_rx_clk_p]
#set_property PACKAGE_PIN Y19 [get_ports hdmi_rx_clk_n]
#set_property IOSTANDARD TMDS_33 [get_ports hdmi_rx_clk_*]

#set_property PACKAGE_PIN W18 [get_ports {hdmi_rx_d_p[0]}]
#set_property PACKAGE_PIN W19 [get_ports {hdmi_rx_d_n[0]}]
#set_property PACKAGE_PIN V16 [get_ports {hdmi_rx_d_p[1]}]
#set_property PACKAGE_PIN W16 [get_ports {hdmi_rx_d_n[1]}]
#set_property PACKAGE_PIN U17 [get_ports {hdmi_rx_d_p[2]}]
#set_property PACKAGE_PIN U18 [get_ports {hdmi_rx_d_n[2]}]
#set_property IOSTANDARD TMDS_33 [get_ports hdmi_rx_d_*]

## HDMI TX (For displaying results to monitor)
set_property PACKAGE_PIN L16 [get_ports hdmi_tx_clk_p]
set_property PACKAGE_PIN L17 [get_ports hdmi_tx_clk_n]
set_property IOSTANDARD TMDS_33 [get_ports hdmi_tx_clk_*]

set_property PACKAGE_PIN K17 [get_ports {hdmi_tx_d_p[0]}]
set_property PACKAGE_PIN K18 [get_ports {hdmi_tx_d_n[0]}]
set_property PACKAGE_PIN K19 [get_ports {hdmi_tx_d_p[1]}]
set_property PACKAGE_PIN J19 [get_ports {hdmi_tx_d_n[1]}]
set_property PACKAGE_PIN J18 [get_ports {hdmi_tx_d_p[2]}]
set_property PACKAGE_PIN H18 [get_ports {hdmi_tx_d_n[2]}]
set_property IOSTANDARD TMDS_33 [get_ports hdmi_tx_d_*]

## HDMI Control Signals
set_property PACKAGE_PIN R19 [get_ports hdmi_tx_hpd]
set_property IOSTANDARD LVCMOS33 [get_ports hdmi_tx_hpd]

set_property PACKAGE_PIN M17 [get_ports hdmi_tx_cec]
set_property IOSTANDARD LVCMOS33 [get_ports hdmi_tx_cec]

##==============================================================================
## Push Buttons (User Control)
##==============================================================================

# BTN0 - Toggle DPD Enable/Bypass
set_property PACKAGE_PIN D19 [get_ports btn_dpd]
set_property IOSTANDARD LVCMOS33 [get_ports btn_dpd]

# BTN1 - Toggle Adaptation On/Off
set_property PACKAGE_PIN D20 [get_ports btn_adapt]
set_property IOSTANDARD LVCMOS33 [get_ports btn_adapt]

# BTN2 - Cycle Temperature State
set_property PACKAGE_PIN L20 [get_ports btn_temp_cycle]
set_property IOSTANDARD LVCMOS33 [get_ports btn_temp_cycle]

# BTN3 - Reset
set_property PACKAGE_PIN L19 [get_ports btn_reset]
set_property IOSTANDARD LVCMOS33 [get_ports btn_reset]

##==============================================================================
## Slide Switches (Temperature Select)
##==============================================================================

# SW0-1: Direct temperature state selection
# 00 = Auto, 01 = Cold, 10 = Normal, 11 = Hot
set_property PACKAGE_PIN M20 [get_ports {sw_temp[0]}]
set_property PACKAGE_PIN M19 [get_ports {sw_temp[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {sw_temp[*]}]

##==============================================================================
## LEDs (Status Indicators)
##==============================================================================

# LED0 - DPD Enabled
set_property PACKAGE_PIN R14 [get_ports led_dpd_on]
set_property IOSTANDARD LVCMOS33 [get_ports led_dpd_on]

# LED1 - Adaptation Active
set_property PACKAGE_PIN P14 [get_ports led_adapt_on]
set_property IOSTANDARD LVCMOS33 [get_ports led_adapt_on]

# LED2 - Cold State
set_property PACKAGE_PIN N16 [get_ports led_temp_cold]
set_property IOSTANDARD LVCMOS33 [get_ports led_temp_cold]

# LED3 - Hot State
set_property PACKAGE_PIN M14 [get_ports led_temp_hot]
set_property IOSTANDARD LVCMOS33 [get_ports led_temp_hot]

##==============================================================================
## RGB LEDs (Error Level Indicator)
##==============================================================================

# RGB LED 0 - EVM Level
# Green = Good (< -25dB), Yellow = Warning, Red = Bad (> -20dB)
set_property PACKAGE_PIN N15 [get_ports led_evm_r]
set_property PACKAGE_PIN G17 [get_ports led_evm_g]
set_property PACKAGE_PIN L15 [get_ports led_evm_b]
set_property IOSTANDARD LVCMOS33 [get_ports led_evm_*]

# RGB LED 1 - Convergence Status
set_property PACKAGE_PIN M15 [get_ports led_conv_r]
set_property PACKAGE_PIN L14 [get_ports led_conv_g]
set_property PACKAGE_PIN G14 [get_ports led_conv_b]
set_property IOSTANDARD LVCMOS33 [get_ports led_conv_*]

##==============================================================================
## PMOD Headers (Optional Debug/Extension)
##==============================================================================

## PMOD JA - Debug outputs (directly tap internal signals)
#set_property PACKAGE_PIN Y18 [get_ports {pmod_ja[0]}]  # debug_dpd_out_i[15]
#set_property PACKAGE_PIN Y19 [get_ports {pmod_ja[1]}]  # debug_dpd_out_i[14]
#set_property PACKAGE_PIN Y16 [get_ports {pmod_ja[2]}]  # debug_error[15]
#set_property PACKAGE_PIN Y17 [get_ports {pmod_ja[3]}]  # debug_error[14]
#set_property IOSTANDARD LVCMOS33 [get_ports {pmod_ja[*]}]

##==============================================================================
## Timing Constraints for AXI Interface
##==============================================================================

# AXI HP port timing is handled by PS constraints
# No additional constraints needed for digital loopback

##==============================================================================
## Physical Constraints
##==============================================================================

# Place MMCM near center of device
set_property LOC MMCME2_ADV_X1Y0 [get_cells -hierarchical -filter {NAME =~ */mmcm_adv_inst}]

##==============================================================================
## Bitstream Configuration
##==============================================================================

# Configuration voltage
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

# Bitstream compression
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]

# Configuration rate
set_property BITSTREAM.CONFIG.CONFIGRATE 33 [current_design]

##==============================================================================
## Debug (ILA) - Enable for development
##==============================================================================

# Debug nets (uncomment to add ILA probes)
#set_property MARK_DEBUG true [get_nets {dpd_inst/dpd_out_i[*]}]
#set_property MARK_DEBUG true [get_nets {dpd_inst/dpd_out_q[*]}]
#set_property MARK_DEBUG true [get_nets {dpd_inst/error_metric[*]}]
#set_property MARK_DEBUG true [get_nets {dpd_inst/temp_state[*]}]
