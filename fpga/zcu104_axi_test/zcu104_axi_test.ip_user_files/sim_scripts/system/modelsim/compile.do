vlib modelsim_lib/work
vlib modelsim_lib/msim

vlib modelsim_lib/msim/xilinx_vip
vlib modelsim_lib/msim/xpm
vlib modelsim_lib/msim/axi_infrastructure_v1_1_0
vlib modelsim_lib/msim/axi_vip_v1_1_22
vlib modelsim_lib/msim/zynq_ultra_ps_e_vip_v1_0_22
vlib modelsim_lib/msim/xil_defaultlib
vlib modelsim_lib/msim/proc_sys_reset_v5_0_17
vlib modelsim_lib/msim/smartconnect_v1_0
vlib modelsim_lib/msim/axi_register_slice_v2_1_36

vmap xilinx_vip modelsim_lib/msim/xilinx_vip
vmap xpm modelsim_lib/msim/xpm
vmap axi_infrastructure_v1_1_0 modelsim_lib/msim/axi_infrastructure_v1_1_0
vmap axi_vip_v1_1_22 modelsim_lib/msim/axi_vip_v1_1_22
vmap zynq_ultra_ps_e_vip_v1_0_22 modelsim_lib/msim/zynq_ultra_ps_e_vip_v1_0_22
vmap xil_defaultlib modelsim_lib/msim/xil_defaultlib
vmap proc_sys_reset_v5_0_17 modelsim_lib/msim/proc_sys_reset_v5_0_17
vmap smartconnect_v1_0 modelsim_lib/msim/smartconnect_v1_0
vmap axi_register_slice_v2_1_36 modelsim_lib/msim/axi_register_slice_v2_1_36

vlog -work xilinx_vip  -incr -mfcu  -sv -L smartconnect_v1_0 -L axi_vip_v1_1_22 -L zynq_ultra_ps_e_vip_v1_0_22 -L xilinx_vip "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/hdl/axi4stream_vip_axi4streampc.sv" \
"C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/hdl/axi_vip_axi4pc.sv" \
"C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/hdl/xil_common_vip_pkg.sv" \
"C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/hdl/axi4stream_vip_pkg.sv" \
"C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/hdl/axi_vip_pkg.sv" \
"C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/hdl/axi4stream_vip_if.sv" \
"C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/hdl/axi_vip_if.sv" \
"C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/hdl/clk_vip_if.sv" \
"C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/hdl/rst_vip_if.sv" \

vlog -work xpm  -incr -mfcu  -sv -L smartconnect_v1_0 -L axi_vip_v1_1_22 -L zynq_ultra_ps_e_vip_v1_0_22 -L xilinx_vip "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"C:/AMDDesignTools/2025.2/Vivado/data/ip/xpm/xpm_cdc/hdl/xpm_cdc.sv" \
"C:/AMDDesignTools/2025.2/Vivado/data/ip/xpm/xpm_fifo/hdl/xpm_fifo.sv" \
"C:/AMDDesignTools/2025.2/Vivado/data/ip/xpm/xpm_memory/hdl/xpm_memory.sv" \

vcom -work xpm  -93  \
"C:/AMDDesignTools/2025.2/Vivado/data/ip/xpm/xpm_VCOMP.vhd" \

vlog -work axi_infrastructure_v1_1_0  -incr -mfcu  "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl/axi_infrastructure_v1_1_vl_rfs.v" \

vlog -work axi_vip_v1_1_22  -incr -mfcu  -sv -L smartconnect_v1_0 -L axi_vip_v1_1_22 -L zynq_ultra_ps_e_vip_v1_0_22 -L xilinx_vip "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/b16a/hdl/axi_vip_v1_1_vl_rfs.sv" \

vlog -work zynq_ultra_ps_e_vip_v1_0_22  -incr -mfcu  -sv -L smartconnect_v1_0 -L axi_vip_v1_1_22 -L zynq_ultra_ps_e_vip_v1_0_22 -L xilinx_vip "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl/zynq_ultra_ps_e_vip_v1_0_vl_rfs.sv" \

vlog -work xil_defaultlib  -incr -mfcu  "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"../../../bd/system/ip/system_zynq_ultra_ps_e_0_0/sim/system_zynq_ultra_ps_e_0_0_vip_wrapper.v" \
"../../../bd/system/ip/system_axi_smc_0/bd_0/sim/bd_44e3.v" \

vcom -work proc_sys_reset_v5_0_17  -93  \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/9438/hdl/proc_sys_reset_v5_0_vh_rfs.vhd" \

vcom -work xil_defaultlib  -93  \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_1/sim/bd_44e3_psr_aclk_0.vhd" \

vlog -work smartconnect_v1_0  -incr -mfcu  -sv -L smartconnect_v1_0 -L axi_vip_v1_1_22 -L zynq_ultra_ps_e_vip_v1_0_22 -L xilinx_vip "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/sc_util_v1_0_vl_rfs.sv" \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/3d9a/hdl/sc_mmu_v1_0_vl_rfs.sv" \

vlog -work xil_defaultlib  -incr -mfcu  -sv -L smartconnect_v1_0 -L axi_vip_v1_1_22 -L zynq_ultra_ps_e_vip_v1_0_22 -L xilinx_vip "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_2/sim/bd_44e3_s00mmu_0.sv" \

vlog -work smartconnect_v1_0  -incr -mfcu  -sv -L smartconnect_v1_0 -L axi_vip_v1_1_22 -L zynq_ultra_ps_e_vip_v1_0_22 -L xilinx_vip "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/7785/hdl/sc_transaction_regulator_v1_0_vl_rfs.sv" \

vlog -work xil_defaultlib  -incr -mfcu  -sv -L smartconnect_v1_0 -L axi_vip_v1_1_22 -L zynq_ultra_ps_e_vip_v1_0_22 -L xilinx_vip "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_3/sim/bd_44e3_s00tr_0.sv" \

vlog -work smartconnect_v1_0  -incr -mfcu  -sv -L smartconnect_v1_0 -L axi_vip_v1_1_22 -L zynq_ultra_ps_e_vip_v1_0_22 -L xilinx_vip "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/3051/hdl/sc_si_converter_v1_0_vl_rfs.sv" \

vlog -work xil_defaultlib  -incr -mfcu  -sv -L smartconnect_v1_0 -L axi_vip_v1_1_22 -L zynq_ultra_ps_e_vip_v1_0_22 -L xilinx_vip "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_4/sim/bd_44e3_s00sic_0.sv" \

vlog -work smartconnect_v1_0  -incr -mfcu  -sv -L smartconnect_v1_0 -L axi_vip_v1_1_22 -L zynq_ultra_ps_e_vip_v1_0_22 -L xilinx_vip "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/852f/hdl/sc_axi2sc_v1_0_vl_rfs.sv" \

vlog -work xil_defaultlib  -incr -mfcu  -sv -L smartconnect_v1_0 -L axi_vip_v1_1_22 -L zynq_ultra_ps_e_vip_v1_0_22 -L xilinx_vip "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_5/sim/bd_44e3_s00a2s_0.sv" \

vlog -work smartconnect_v1_0  -incr -mfcu  -sv -L smartconnect_v1_0 -L axi_vip_v1_1_22 -L zynq_ultra_ps_e_vip_v1_0_22 -L xilinx_vip "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/sc_node_v1_0_vl_rfs.sv" \

vlog -work xil_defaultlib  -incr -mfcu  -sv -L smartconnect_v1_0 -L axi_vip_v1_1_22 -L zynq_ultra_ps_e_vip_v1_0_22 -L xilinx_vip "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_6/sim/bd_44e3_sarn_0.sv" \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_7/sim/bd_44e3_srn_0.sv" \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_8/sim/bd_44e3_sawn_0.sv" \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_9/sim/bd_44e3_swn_0.sv" \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_10/sim/bd_44e3_sbn_0.sv" \

vlog -work smartconnect_v1_0  -incr -mfcu  -sv -L smartconnect_v1_0 -L axi_vip_v1_1_22 -L zynq_ultra_ps_e_vip_v1_0_22 -L xilinx_vip "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/fca9/hdl/sc_sc2axi_v1_0_vl_rfs.sv" \

vlog -work xil_defaultlib  -incr -mfcu  -sv -L smartconnect_v1_0 -L axi_vip_v1_1_22 -L zynq_ultra_ps_e_vip_v1_0_22 -L xilinx_vip "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_11/sim/bd_44e3_m00s2a_0.sv" \

vlog -work smartconnect_v1_0  -incr -mfcu  -sv -L smartconnect_v1_0 -L axi_vip_v1_1_22 -L zynq_ultra_ps_e_vip_v1_0_22 -L xilinx_vip "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/e44a/hdl/sc_exit_v1_0_vl_rfs.sv" \

vlog -work xil_defaultlib  -incr -mfcu  -sv -L smartconnect_v1_0 -L axi_vip_v1_1_22 -L zynq_ultra_ps_e_vip_v1_0_22 -L xilinx_vip "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"../../../bd/system/ip/system_axi_smc_0/bd_0/ip/ip_12/sim/bd_44e3_m00e_0.sv" \

vcom -work smartconnect_v1_0  -93  \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/cb42/hdl/sc_ultralite_v1_0_rfs.vhd" \

vlog -work smartconnect_v1_0  -incr -mfcu  -sv -L smartconnect_v1_0 -L axi_vip_v1_1_22 -L zynq_ultra_ps_e_vip_v1_0_22 -L xilinx_vip "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/cb42/hdl/sc_ultralite_v1_0_rfs.sv" \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/0848/hdl/sc_switchboard_v1_0_vl_rfs.sv" \

vlog -work axi_register_slice_v2_1_36  -incr -mfcu  "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/bc4b/hdl/axi_register_slice_v2_1_vl_rfs.v" \

vlog -work xil_defaultlib  -incr -mfcu  -sv -L smartconnect_v1_0 -L axi_vip_v1_1_22 -L zynq_ultra_ps_e_vip_v1_0_22 -L xilinx_vip "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"../../../bd/system/ip/system_axi_smc_0/sim/system_axi_smc_0.sv" \

vcom -work xil_defaultlib  -93  \
"../../../bd/system/ip/system_rst_ps8_0_100M_0/sim/system_rst_ps8_0_100M_0.vhd" \

vlog -work xil_defaultlib  -incr -mfcu  "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/ec67/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/a0fe/hdl" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/f0b6/hdl/verilog" "+incdir+../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/00fe/hdl/verilog" "+incdir+../../../../../../../../../../AMDDesignTools/2025.2/Vivado/data/rsb/busdef" "+incdir+C:/AMDDesignTools/2025.2/Vivado/data/xilinx_vip/include" \
"../../../bd/system/ipshared/de31/hdl/my_neural_net_slave_lite_v1_0_S00_AXI.v" \
"../../../bd/system/ipshared/de31/hdl/my_neural_net.v" \
"../../../../zcu104_axi_test.gen/sources_1/bd/system/ipshared/de31/5de9/tdnn_generator.v" \
"../../../bd/system/ip/system_my_neural_net_0_0/sim/system_my_neural_net_0_0.v" \
"../../../bd/system/sim/system.v" \

vlog -work xil_defaultlib \
"glbl.v"

