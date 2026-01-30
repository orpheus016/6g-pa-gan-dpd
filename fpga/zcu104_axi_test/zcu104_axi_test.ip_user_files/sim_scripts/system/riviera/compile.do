transcript off
onbreak {quit -force}
onerror {quit -force}
transcript on

vlib work
vlib riviera/xilinx_vip
vlib riviera/xpm
vlib riviera/axi_infrastructure_v1_1_0
vlib riviera/axi_vip_v1_1_22
vlib riviera/zynq_ultra_ps_e_vip_v1_0_22
vlib riviera/xil_defaultlib
vlib riviera/proc_sys_reset_v5_0_17
vlib riviera/smartconnect_v1_0
vlib riviera/axi_register_slice_v2_1_36

vmap xilinx_vip riviera/xilinx_vip
vmap xpm riviera/xpm
vmap axi_infrastructure_v1_1_0 riviera/axi_infrastructure_v1_1_0
vmap axi_vip_v1_1_22 riviera/axi_vip_v1_1_22
vmap zynq_ultra_ps_e_vip_v1_0_22 riviera/zynq_ultra_ps_e_vip_v1_0_22
vmap xil_defaultlib riviera/xil_defaultlib
vmap proc_sys_reset_v5_0_17 riviera/proc_sys_reset_v5_0_17
vmap smartconnect_v1_0 riviera/smartconnect_v1_0
vmap axi_register_slice_v2_1_36 riviera/axi_register_slice_v2_1_36

vlog -work xilinx_vip  -incr "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/hdl/axi4stream_vip_axi4streampc.sv" \
"C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/hdl/axi_vip_axi4pc.sv" \
"C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/hdl/xil_common_vip_pkg.sv" \
"C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/hdl/axi4stream_vip_pkg.sv" \
"C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/hdl/axi_vip_pkg.sv" \
"C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/hdl/axi4stream_vip_if.sv" \
"C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/hdl/axi_vip_if.sv" \
"C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/hdl/clk_vip_if.sv" \
"C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/hdl/rst_vip_if.sv" \

vlog -work xpm  -incr "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"C:/AMDDesignTools/2025.2/Vivado/data/ip/xpm/xpm_cdc/hdl/xpm_cdc.sv" \
"C:/AMDDesignTools/2025.2/Vivado/data/ip/xpm/xpm_fifo/hdl/xpm_fifo.sv" \
"C:/AMDDesignTools/2025.2/Vivado/data/ip/xpm/xpm_memory/hdl/xpm_memory.sv" \

vcom -work xpm -93  -incr \
"C:/AMDDesignTools/2025.2/Vivado/data/ip/xpm/xpm_VCOMP.vhd" \

vlog -work axi_infrastructure_v1_1_0  -incr -v2k5 "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl/axi_infrastructure_v1_1_vl_rfs.v" \

vlog -work axi_vip_v1_1_22  -incr "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/b16a/hdl/axi_vip_v1_1_vl_rfs.sv" \

vlog -work zynq_ultra_ps_e_vip_v1_0_22  -incr "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl/zynq_ultra_ps_e_vip_v1_0_vl_rfs.sv" \

vlog -work xil_defaultlib  -incr -v2k5 "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"../../../bd/system/ip/system_zynq_ultra_ps_e_0_0/sim/system_zynq_ultra_ps_e_0_0_vip_wrapper.v" \
"../../../bd/system/ip/system_axi_smc_0/bd_0/sim/bd_44e3.v" \

vcom -work proc_sys_reset_v5_0_17 -93  -incr \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/9438/hdl/proc_sys_reset_v5_0_vh_rfs.vhd" \

vcom -work xil_defaultlib -93  -incr \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_1/sim/bd_44e3_psr_aclk_0.vhd" \

vlog -work smartconnect_v1_0  -incr "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/sc_util_v1_0_vl_rfs.sv" \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/3d9a/hdl/sc_mmu_v1_0_vl_rfs.sv" \

vlog -work xil_defaultlib  -incr "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_2/sim/bd_44e3_s00mmu_0.sv" \

vlog -work smartconnect_v1_0  -incr "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/7785/hdl/sc_transaction_regulator_v1_0_vl_rfs.sv" \

vlog -work xil_defaultlib  -incr "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_3/sim/bd_44e3_s00tr_0.sv" \

vlog -work smartconnect_v1_0  -incr "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/3051/hdl/sc_si_converter_v1_0_vl_rfs.sv" \

vlog -work xil_defaultlib  -incr "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_4/sim/bd_44e3_s00sic_0.sv" \

vlog -work smartconnect_v1_0  -incr "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/852f/hdl/sc_axi2sc_v1_0_vl_rfs.sv" \

vlog -work xil_defaultlib  -incr "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_5/sim/bd_44e3_s00a2s_0.sv" \

vlog -work smartconnect_v1_0  -incr "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/sc_node_v1_0_vl_rfs.sv" \

vlog -work xil_defaultlib  -incr "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_6/sim/bd_44e3_sarn_0.sv" \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_7/sim/bd_44e3_srn_0.sv" \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_8/sim/bd_44e3_sawn_0.sv" \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_9/sim/bd_44e3_swn_0.sv" \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_10/sim/bd_44e3_sbn_0.sv" \

vlog -work smartconnect_v1_0  -incr "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/fca9/hdl/sc_sc2axi_v1_0_vl_rfs.sv" \

vlog -work xil_defaultlib  -incr "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_11/sim/bd_44e3_m00s2a_0.sv" \

vlog -work smartconnect_v1_0  -incr "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/e44a/hdl/sc_exit_v1_0_vl_rfs.sv" \

vlog -work xil_defaultlib  -incr "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_12/sim/bd_44e3_m00e_0.sv" \

vcom -work smartconnect_v1_0 -93  -incr \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/cb42/hdl/sc_ultralite_v1_0_rfs.vhd" \

vlog -work smartconnect_v1_0  -incr "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/cb42/hdl/sc_ultralite_v1_0_rfs.sv" \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/0848/hdl/sc_switchboard_v1_0_vl_rfs.sv" \

vlog -work axi_register_slice_v2_1_36  -incr -v2k5 "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/bc4b/hdl/axi_register_slice_v2_1_vl_rfs.v" \

vlog -work xil_defaultlib  -incr "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"../../../bd/system/ip/system_axi_smc_0/sim/system_axi_smc_0.sv" \

vcom -work xil_defaultlib -93  -incr \
"../../../bd/system/ip/system_rst_ps8_0_100M_0/sim/system_rst_ps8_0_100M_0.vhd" \

vlog -work xil_defaultlib  -incr -v2k5 "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" -l xilinx_vip -l xpm -l axi_infrastructure_v1_1_0 -l axi_vip_v1_1_22 -l zynq_ultra_ps_e_vip_v1_0_22 -l xil_defaultlib -l proc_sys_reset_v5_0_17 -l smartconnect_v1_0 -l axi_register_slice_v2_1_36 \
"../../../bd/system/ipshared/de31/hdl/my_neural_net_slave_lite_v1_0_S00_AXI.v" \
"../../../bd/system/ipshared/de31/hdl/my_neural_net.v" \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/de31/5de9/tdnn_generator.v" \
"../../../bd/system/ip/system_my_neural_net_0_0/sim/system_my_neural_net_0_0.v" \
"../../../bd/system/sim/system.v" \

vlog -work xil_defaultlib \
"glbl.v"

