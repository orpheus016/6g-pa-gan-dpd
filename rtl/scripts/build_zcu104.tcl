# =============================================================================
# Vivado Build Script for ZCU104 (6G PA DPD System - Production)
# LSI Design Contest 29th - Okinawa
# =============================================================================
#
# Usage:
#   vivado -mode batch -source scripts/build_zcu104.tcl
#   vivado -mode batch -source scripts/build_zcu104.tcl -tclargs synth_only
#   vivado -mode batch -source scripts/build_zcu104.tcl -tclargs bitstream
#
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
set project_name    "dpd_zcu104"
set project_dir     "../build/zcu104"
set part            "xczu7ev-ffvc1156-2-e"
set board           "xilinx.com:zcu104:part0:1.1"
set top_module      "dpd_top"

# Source directories
set rtl_dir         "../src"
set tb_dir          "../tb"
set constraints_dir "../constraints"
set weights_dir     "../weights"

# Clock constraints
set clk_200_period  5.0    ;# 200MHz for NN
set clk_400_period  2.5    ;# 400MHz for output
set clk_adapt_period 1000.0 ;# 1MHz for adaptation

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
set build_target "bitstream"
if {[llength $argv] > 0} {
    set build_target [lindex $argv 0]
}

puts "============================================"
puts "6G PA DPD System - ZCU104 Production Build"
puts "Build target: $build_target"
puts "============================================"

# -----------------------------------------------------------------------------
# Create project
# -----------------------------------------------------------------------------
file mkdir $project_dir
cd $project_dir

create_project $project_name . -part $part -force

# Set board files
set_property board_part $board [current_project]

# Set project properties
set_property target_language Verilog [current_project]
set_property default_lib work [current_project]
set_property coreContainer.enable true [current_project]

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

add_files -fileset constrs_1 "$constraints_dir/zcu104.xdc"

# -----------------------------------------------------------------------------
# Add testbenches
# -----------------------------------------------------------------------------
if {[file exists $tb_dir]} {
    puts "Adding testbenches..."
    add_files -fileset sim_1 [glob -nocomplain $tb_dir/*.v]
    set_property top tb_dpd_top [get_filesets sim_1]
}

# -----------------------------------------------------------------------------
# Create Block Design for Zynq UltraScale+
# -----------------------------------------------------------------------------
proc create_zynq_mpsoc_bd {} {
    puts "Creating Zynq MPSoC block design..."
    
    create_bd_design "system"
    
    # Add Zynq MPSoC PS
    create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.4 zynq_mpsoc
    
    # Configure PS for ZCU104
    apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
        -config {apply_board_preset "1"} [get_bd_cells zynq_mpsoc]
    
    set_property -dict [list \
        CONFIG.PSU__USE__M_AXI_GP0 {1} \
        CONFIG.PSU__USE__M_AXI_GP1 {1} \
        CONFIG.PSU__USE__S_AXI_GP0 {1} \
        CONFIG.PSU__USE__S_AXI_GP2 {1} \
        CONFIG.PSU__FPGA_PL0_ENABLE {1} \
        CONFIG.PSU__FPGA_PL1_ENABLE {1} \
        CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} \
        CONFIG.PSU__CRL_APB__PL1_REF_CTRL__FREQMHZ {200} \
    ] [get_bd_cells zynq_mpsoc]
    
    # Add clocking for DPD system
    create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 clk_wiz_dpd
    set_property -dict [list \
        CONFIG.PRIMITIVE {MMCM} \
        CONFIG.PRIM_SOURCE {Global_buffer} \
        CONFIG.CLKOUT1_USED {true} \
        CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {200} \
        CONFIG.CLKOUT2_USED {true} \
        CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {400} \
        CONFIG.CLKOUT3_USED {true} \
        CONFIG.CLKOUT3_REQUESTED_OUT_FREQ {1} \
        CONFIG.USE_LOCKED {true} \
        CONFIG.USE_RESET {true} \
        CONFIG.RESET_TYPE {ACTIVE_LOW} \
    ] [get_bd_cells clk_wiz_dpd]
    
    # Connect PL clock to clocking wizard
    connect_bd_net [get_bd_pins zynq_mpsoc/pl_clk1] [get_bd_pins clk_wiz_dpd/clk_in1]
    connect_bd_net [get_bd_pins zynq_mpsoc/pl_resetn0] [get_bd_pins clk_wiz_dpd/resetn]
    
    # Add AXI DMA for high-speed data transfer
    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_iq
    set_property -dict [list \
        CONFIG.c_include_sg {1} \
        CONFIG.c_sg_length_width {26} \
        CONFIG.c_mm2s_burst_size {256} \
        CONFIG.c_s2mm_burst_size {256} \
        CONFIG.c_m_axi_mm2s_data_width {128} \
        CONFIG.c_m_axi_s2mm_data_width {128} \
        CONFIG.c_m_axis_mm2s_tdata_width {128} \
        CONFIG.c_s_axis_s2mm_tdata_width {128} \
    ] [get_bd_cells axi_dma_iq]
    
    # Add AXI GPIO for control
    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_ctrl
    set_property -dict [list \
        CONFIG.C_GPIO_WIDTH {8} \
        CONFIG.C_GPIO2_WIDTH {4} \
        CONFIG.C_IS_DUAL {1} \
        CONFIG.C_ALL_INPUTS_2 {1} \
        CONFIG.C_INTERRUPT_PRESENT {1} \
    ] [get_bd_cells axi_gpio_ctrl]
    
    # Add RFDC interface for real RF deployment (optional)
    # create_bd_cell -type ip -vlnv xilinx.com:ip:usp_rf_data_converter:2.6 rf_data_converter
    
    # Add AXI Interconnect
    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0
    set_property -dict [list \
        CONFIG.NUM_MI {4} \
        CONFIG.NUM_SI {1} \
    ] [get_bd_cells axi_interconnect_0]
    
    # Connect AXI interfaces
    connect_bd_intf_net [get_bd_intf_pins zynq_mpsoc/M_AXI_HPM0_FPD] \
        [get_bd_intf_pins axi_interconnect_0/S00_AXI]
    
    # Add interrupt controller
    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_intc:4.1 axi_intc_0
    set_property -dict [list \
        CONFIG.C_IRQ_CONNECTION {1} \
    ] [get_bd_cells axi_intc_0]
    
    # Validate and save
    validate_bd_design
    save_bd_design
    
    # Generate wrapper
    make_wrapper -files [get_files system.bd] -top
    add_files -norecurse [get_files system_wrapper.v]
    
    puts "Block design created successfully"
}

# Optionally create block design
# create_zynq_mpsoc_bd

# -----------------------------------------------------------------------------
# Set Synthesis Options
# -----------------------------------------------------------------------------
set_property -dict [list \
    {STEPS.SYNTH_DESIGN.ARGS.RETIMING} {true} \
    {STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY} {rebuilt} \
    {STEPS.SYNTH_DESIGN.ARGS.FSM_EXTRACTION} {one_hot} \
] [get_runs synth_1]

# -----------------------------------------------------------------------------
# Run Synthesis
# -----------------------------------------------------------------------------
if {$build_target in {"synth_only" "bitstream"}} {
    puts "Running synthesis..."
    
    # Use high-performance strategy
    set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
    
    launch_runs synth_1 -jobs 8
    wait_on_run synth_1
    
    # Check synthesis status
    if {[get_property STATUS [get_runs synth_1]] != "synth_design Complete!"} {
        puts "ERROR: Synthesis failed!"
        exit 1
    }
    
    # Open synthesized design
    open_run synth_1
    
    # Generate reports
    file mkdir reports
    report_utilization -file reports/synth_utilization.rpt
    report_timing_summary -file reports/synth_timing.rpt
    report_clock_utilization -file reports/synth_clocks.rpt
    
    puts "Synthesis completed successfully"
}

# -----------------------------------------------------------------------------
# Run Implementation
# -----------------------------------------------------------------------------
if {$build_target == "bitstream"} {
    puts "Running implementation..."
    
    # Set implementation strategy for high-speed design
    set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
    
    # Additional optimization options
    set_property -dict [list \
        {STEPS.PLACE_DESIGN.ARGS.DIRECTIVE} {ExtraNetDelay_high} \
        {STEPS.PHYS_OPT_DESIGN.IS_ENABLED} {true} \
        {STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE} {AggressiveExplore} \
        {STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE} {AggressiveExplore} \
        {STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED} {true} \
        {STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE} {AggressiveExplore} \
    ] [get_runs impl_1]
    
    launch_runs impl_1 -to_step write_bitstream -jobs 8
    wait_on_run impl_1
    
    # Check implementation status
    if {[get_property STATUS [get_runs impl_1]] != "write_bitstream Complete!"} {
        puts "ERROR: Implementation failed!"
        exit 1
    }
    
    # Open implemented design
    open_run impl_1
    
    # Generate comprehensive reports
    report_utilization -file reports/impl_utilization.rpt -hierarchical
    report_timing_summary -file reports/impl_timing.rpt -max_paths 100
    report_timing -delay_type max -max_paths 50 -file reports/impl_setup_paths.rpt
    report_timing -delay_type min -max_paths 50 -file reports/impl_hold_paths.rpt
    report_power -file reports/impl_power.rpt
    report_drc -file reports/impl_drc.rpt
    report_methodology -file reports/impl_methodology.rpt
    report_clock_utilization -file reports/impl_clocks.rpt
    report_clock_interaction -file reports/impl_clock_interaction.rpt
    
    # Copy bitstream to output
    file mkdir ../output
    file copy -force [get_property DIRECTORY [get_runs impl_1]]/${project_name}.bit ../output/dpd_zcu104.bit
    
    # Generate hardware handoff
    write_hwdef -force -file ../output/dpd_zcu104.hwh
    
    # Generate XSA for Vitis
    write_hw_platform -fixed -include_bit -force -file ../output/dpd_zcu104.xsa
    
    puts "============================================"
    puts "Build completed successfully!"
    puts "Bitstream: output/dpd_zcu104.bit"
    puts "Hardware def: output/dpd_zcu104.hwh"
    puts "XSA Platform: output/dpd_zcu104.xsa"
    puts "============================================"
}

# -----------------------------------------------------------------------------
# Resource Summary (ZCU104)
# -----------------------------------------------------------------------------
puts ""
puts "=== Expected Resource Utilization (ZCU104) ==="
puts "CLB LUTs:     ~8,000  (3% of 230,400)"
puts "CLB Regs:     ~12,000 (3% of 460,800)"
puts "DSP48E2:      ~48     (3% of 1,728)"
puts "Block RAM:    ~16     (5% of 312)"
puts "URAM:         ~0      (0% of 96)"
puts "Fmax Target:  400 MHz (output stage)"
puts "=============================================="

exit 0
