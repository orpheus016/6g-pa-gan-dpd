# =============================================================================
# Vivado Build Script for PYNQ-Z1 (6G PA DPD System)
# LSI Design Contest 29th - Okinawa
# =============================================================================
#
# Usage:
#   vivado -mode batch -source scripts/build_pynq.tcl
#   vivado -mode batch -source scripts/build_pynq.tcl -tclargs synth_only
#   vivado -mode batch -source scripts/build_pynq.tcl -tclargs bitstream
#
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
set project_name    "dpd_pynq"
set project_dir     "../build/pynq"
set part            "xc7z020clg400-1"
set board           "tul.com.tw:pynq-z1:part0:1.0"
set top_module      "dpd_top"

# Source directories
set rtl_dir         "../src"
set tb_dir          "../tb"
set constraints_dir "../constraints"
set weights_dir     "../weights"

# Clock constraint
set clk_period_ns   5.0  ;# 200MHz

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
set build_target "bitstream"
if {[llength $argv] > 0} {
    set build_target [lindex $argv 0]
}

puts "============================================"
puts "6G PA DPD System - PYNQ-Z1 Build"
puts "Build target: $build_target"
puts "============================================"

# -----------------------------------------------------------------------------
# Create project
# -----------------------------------------------------------------------------
file mkdir $project_dir
cd $project_dir

create_project $project_name . -part $part -force

# Set board (optional - PYNQ-Z1 board files needed)
# set_property board_part $board [current_project]

# Set project properties
set_property target_language Verilog [current_project]
set_property default_lib work [current_project]

# -----------------------------------------------------------------------------
# Add RTL sources
# -----------------------------------------------------------------------------
puts "Adding RTL sources..."

add_files -fileset sources_1 [glob $rtl_dir/*.v]

# Add weight initialization files
if {[file exists $weights_dir]} {
    add_files -fileset sources_1 -norecurse [glob -nocomplain $weights_dir/*.hex]
    add_files -fileset sources_1 -norecurse [glob -nocomplain $weights_dir/*.mem]
}

# Set top module
set_property top $top_module [current_fileset]

# -----------------------------------------------------------------------------
# Add constraints
# -----------------------------------------------------------------------------
puts "Adding constraints..."

# Use HDMI demo constraints for contest
set xdc_file "$constraints_dir/pynq_z1_hdmi.xdc"
if {[file exists $xdc_file]} {
    add_files -fileset constrs_1 $xdc_file
} else {
    # Fall back to standard constraints
    add_files -fileset constrs_1 "$constraints_dir/pynq_z1.xdc"
}

# -----------------------------------------------------------------------------
# Add testbenches (for simulation only)
# -----------------------------------------------------------------------------
if {[file exists $tb_dir]} {
    puts "Adding testbenches..."
    add_files -fileset sim_1 [glob -nocomplain $tb_dir/*.v]
    set_property top tb_dpd_top [get_filesets sim_1]
}

# -----------------------------------------------------------------------------
# Create Block Design (PS + PL Integration)
# -----------------------------------------------------------------------------
proc create_zynq_bd {} {
    puts "Creating Zynq block design..."
    
    create_bd_design "system"
    
    # Add Zynq PS
    create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
    
    # Configure PS for PYNQ-Z1
    set_property -dict [list \
        CONFIG.PCW_USE_M_AXI_GP0 {1} \
        CONFIG.PCW_USE_S_AXI_HP0 {1} \
        CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
        CONFIG.PCW_FPGA1_PERIPHERAL_FREQMHZ {200} \
        CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
        CONFIG.PCW_IRQ_F2P_INTR {1} \
    ] [get_bd_cells ps7]
    
    # Add DPD RTL as IP
    # (In practice, package dpd_top as custom IP first)
    
    # Add AXI DMA for data transfer
    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
    set_property -dict [list \
        CONFIG.c_include_sg {0} \
        CONFIG.c_sg_include_stscntrl_strm {0} \
        CONFIG.c_mm2s_burst_size {256} \
        CONFIG.c_s2mm_burst_size {256} \
    ] [get_bd_cells axi_dma_0]
    
    # Add AXI GPIO for buttons/LEDs
    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_btns
    set_property -dict [list \
        CONFIG.C_GPIO_WIDTH {4} \
        CONFIG.C_ALL_INPUTS {1} \
        CONFIG.C_INTERRUPT_PRESENT {1} \
    ] [get_bd_cells axi_gpio_btns]
    
    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_leds
    set_property -dict [list \
        CONFIG.C_GPIO_WIDTH {4} \
        CONFIG.C_ALL_OUTPUTS {1} \
    ] [get_bd_cells axi_gpio_leds]
    
    # Add clocking wizard for 200MHz domain
    create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 clk_wiz_0
    set_property -dict [list \
        CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {200} \
        CONFIG.CLKOUT2_USED {true} \
        CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {1} \
        CONFIG.USE_LOCKED {true} \
        CONFIG.USE_RESET {true} \
    ] [get_bd_cells clk_wiz_0]
    
    # Connect clocks
    connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins clk_wiz_0/clk_in1]
    
    # Run automation
    apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
        -config {make_external "FIXED_IO, DDR" Master "Disable" Slave "Disable"} \
        [get_bd_cells ps7]
    
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
        -config {Master "/ps7/M_AXI_GP0" Clk "Auto"} \
        [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
    
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
        -config {Master "/ps7/M_AXI_GP0" Clk "Auto"} \
        [get_bd_intf_pins axi_gpio_btns/S_AXI]
    
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
        -config {Master "/ps7/M_AXI_GP0" Clk "Auto"} \
        [get_bd_intf_pins axi_gpio_leds/S_AXI]
    
    # Validate and save
    validate_bd_design
    save_bd_design
    
    # Generate wrapper
    make_wrapper -files [get_files system.bd] -top
    add_files -norecurse [get_files system_wrapper.v]
    
    puts "Block design created successfully"
}

# Optionally create block design
# create_zynq_bd

# -----------------------------------------------------------------------------
# Run Synthesis
# -----------------------------------------------------------------------------
if {$build_target in {"synth_only" "bitstream"}} {
    puts "Running synthesis..."
    
    # Set synthesis options
    set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
    
    launch_runs synth_1 -jobs 4
    wait_on_run synth_1
    
    # Check synthesis status
    if {[get_property STATUS [get_runs synth_1]] != "synth_design Complete!"} {
        puts "ERROR: Synthesis failed!"
        exit 1
    }
    
    # Open synthesized design for analysis
    open_run synth_1
    
    # Report utilization
    report_utilization -file reports/synth_utilization.rpt
    report_timing_summary -file reports/synth_timing.rpt
    
    puts "Synthesis completed successfully"
}

# -----------------------------------------------------------------------------
# Run Implementation
# -----------------------------------------------------------------------------
if {$build_target == "bitstream"} {
    puts "Running implementation..."
    
    # Set implementation strategy
    set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
    
    launch_runs impl_1 -to_step write_bitstream -jobs 4
    wait_on_run impl_1
    
    # Check implementation status
    if {[get_property STATUS [get_runs impl_1]] != "write_bitstream Complete!"} {
        puts "ERROR: Implementation failed!"
        exit 1
    }
    
    # Open implemented design
    open_run impl_1
    
    # Generate reports
    file mkdir reports
    report_utilization -file reports/impl_utilization.rpt
    report_timing_summary -file reports/impl_timing.rpt
    report_power -file reports/impl_power.rpt
    report_drc -file reports/impl_drc.rpt
    
    # Copy bitstream to output
    file mkdir ../output
    file copy -force [get_property DIRECTORY [get_runs impl_1]]/${project_name}.bit ../output/dpd_pynq.bit
    
    # Generate hardware handoff for PYNQ
    write_hwdef -force -file ../output/dpd_pynq.hwh
    
    puts "============================================"
    puts "Build completed successfully!"
    puts "Bitstream: output/dpd_pynq.bit"
    puts "Hardware def: output/dpd_pynq.hwh"
    puts "============================================"
}

# -----------------------------------------------------------------------------
# Resource Summary
# -----------------------------------------------------------------------------
puts ""
puts "=== Expected Resource Utilization ==="
puts "LUTs:        ~5,000 (9% of 53,200)"
puts "DSP48s:      ~24    (11% of 220)"
puts "BRAM:        ~8     (6% of 140)"
puts "Fmax Target: 200 MHz"
puts "======================================"

exit 0
