#===============================================================================
# Makefile for 6G PA DPD RTL Simulation
# Supports Icarus Verilog and Verilator
#===============================================================================

# Tool selection
SIMULATOR ?= iverilog
VERILATOR ?= verilator

# Directories
RTL_DIR := src
TB_DIR := tb
WEIGHT_DIR := weights
BUILD_DIR := build
WAVE_DIR := waves

# Source files
RTL_SRC := $(wildcard $(RTL_DIR)/*.v)
TB_SRC := $(wildcard $(TB_DIR)/*.v)

# Top-level modules for testbenches
TB_DPD_TOP := tb_dpd_top
TB_TDNN := tb_tdnn_generator
TB_ASPSA := tb_aspsa_engine

# Include paths
INCDIRS := -I$(RTL_DIR) -I$(WEIGHT_DIR)

# Icarus Verilog flags
IVERILOG_FLAGS := -g2012 -Wall $(INCDIRS)
VVP_FLAGS := -lxt2

# Verilator flags
VERILATOR_FLAGS := --cc --exe --build -Wall $(INCDIRS) \
                   --trace --trace-fst \
                   -Wno-UNUSED -Wno-UNDRIVEN

#===============================================================================
# Default target
#===============================================================================
.PHONY: all clean help

all: $(BUILD_DIR) sim_dpd_top

help:
	@echo "6G PA DPD RTL Simulation Makefile"
	@echo "=================================="
	@echo ""
	@echo "Targets:"
	@echo "  all           - Build and run DPD top testbench (default)"
	@echo "  sim_dpd_top   - Simulate DPD top module"
	@echo "  sim_tdnn      - Simulate TDNN generator"
	@echo "  sim_aspsa     - Simulate A-SPSA engine"
	@echo "  sim_all       - Run all simulations"
	@echo "  lint          - Run Verilator lint checks"
	@echo "  clean         - Remove build artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  SIMULATOR=iverilog|verilator (default: iverilog)"
	@echo ""
	@echo "Examples:"
	@echo "  make sim_dpd_top"
	@echo "  make sim_all SIMULATOR=verilator"
	@echo "  make lint"

#===============================================================================
# Directory creation
#===============================================================================
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(WAVE_DIR):
	mkdir -p $(WAVE_DIR)

#===============================================================================
# Icarus Verilog Simulation
#===============================================================================
ifeq ($(SIMULATOR),iverilog)

# DPD Top simulation
sim_dpd_top: $(BUILD_DIR) $(WAVE_DIR)
	@echo "=== Compiling DPD Top Testbench (Icarus) ==="
	$(SIMULATOR) $(IVERILOG_FLAGS) -o $(BUILD_DIR)/$(TB_DPD_TOP).vvp \
		$(RTL_SRC) $(TB_DIR)/$(TB_DPD_TOP).v
	@echo "=== Running Simulation ==="
	cd $(BUILD_DIR) && vvp $(TB_DPD_TOP).vvp $(VVP_FLAGS)
	@mv $(BUILD_DIR)/*.vcd $(WAVE_DIR)/ 2>/dev/null || true
	@echo "=== Simulation Complete ==="
	@echo "Waveforms: $(WAVE_DIR)/$(TB_DPD_TOP).vcd"

# TDNN Generator simulation
sim_tdnn: $(BUILD_DIR) $(WAVE_DIR)
	@echo "=== Compiling TDNN Testbench (Icarus) ==="
	$(SIMULATOR) $(IVERILOG_FLAGS) -o $(BUILD_DIR)/$(TB_TDNN).vvp \
		$(RTL_DIR)/tdnn_generator.v \
		$(RTL_DIR)/fc_layer.v \
		$(RTL_DIR)/activation.v \
		$(TB_DIR)/$(TB_TDNN).v
	@echo "=== Running Simulation ==="
	cd $(BUILD_DIR) && vvp $(TB_TDNN).vvp $(VVP_FLAGS)
	@mv $(BUILD_DIR)/*.vcd $(WAVE_DIR)/ 2>/dev/null || true
	@echo "=== Simulation Complete ==="

# A-SPSA Engine simulation
sim_aspsa: $(BUILD_DIR) $(WAVE_DIR)
	@echo "=== Compiling A-SPSA Testbench (Icarus) ==="
	$(SIMULATOR) $(IVERILOG_FLAGS) -o $(BUILD_DIR)/$(TB_ASPSA).vvp \
		$(RTL_DIR)/aspsa_engine.v \
		$(TB_DIR)/$(TB_ASPSA).v
	@echo "=== Running Simulation ==="
	cd $(BUILD_DIR) && vvp $(TB_ASPSA).vvp $(VVP_FLAGS)
	@mv $(BUILD_DIR)/*.vcd $(WAVE_DIR)/ 2>/dev/null || true
	@echo "=== Simulation Complete ==="

endif

#===============================================================================
# Verilator Simulation
#===============================================================================
ifeq ($(SIMULATOR),verilator)

# DPD Top simulation
sim_dpd_top: $(BUILD_DIR) $(WAVE_DIR)
	@echo "=== Compiling DPD Top Testbench (Verilator) ==="
	$(VERILATOR) $(VERILATOR_FLAGS) \
		--top-module dpd_top \
		-o $(BUILD_DIR)/V$(TB_DPD_TOP) \
		$(RTL_SRC)
	@echo "=== Running Simulation ==="
	./$(BUILD_DIR)/V$(TB_DPD_TOP)
	@echo "=== Simulation Complete ==="

# TDNN Generator simulation
sim_tdnn: $(BUILD_DIR) $(WAVE_DIR)
	@echo "=== Compiling TDNN Testbench (Verilator) ==="
	$(VERILATOR) $(VERILATOR_FLAGS) \
		--top-module tdnn_generator \
		-o $(BUILD_DIR)/V$(TB_TDNN) \
		$(RTL_DIR)/tdnn_generator.v \
		$(RTL_DIR)/fc_layer.v \
		$(RTL_DIR)/activation.v
	@echo "=== Running Simulation ==="
	./$(BUILD_DIR)/V$(TB_TDNN)
	@echo "=== Simulation Complete ==="

# A-SPSA simulation
sim_aspsa: $(BUILD_DIR) $(WAVE_DIR)
	@echo "=== Compiling A-SPSA Testbench (Verilator) ==="
	$(VERILATOR) $(VERILATOR_FLAGS) \
		--top-module aspsa_engine \
		-o $(BUILD_DIR)/V$(TB_ASPSA) \
		$(RTL_DIR)/aspsa_engine.v
	@echo "=== Running Simulation ==="
	./$(BUILD_DIR)/V$(TB_ASPSA)
	@echo "=== Simulation Complete ==="

endif

#===============================================================================
# Run all simulations
#===============================================================================
sim_all: sim_dpd_top sim_tdnn sim_aspsa
	@echo "=== All Simulations Complete ==="

#===============================================================================
# Lint check (Verilator)
#===============================================================================
lint:
	@echo "=== Running Verilator Lint ==="
	$(VERILATOR) --lint-only -Wall $(INCDIRS) \
		-Wno-UNUSED -Wno-UNDRIVEN -Wno-DECLFILENAME \
		$(RTL_SRC)
	@echo "=== Lint Complete ==="

#===============================================================================
# Synthesis check (Yosys)
#===============================================================================
synth_check:
	@echo "=== Running Yosys Synthesis Check ==="
	yosys -p "read_verilog $(RTL_SRC); synth -top dpd_top; stat"
	@echo "=== Synthesis Check Complete ==="

#===============================================================================
# Waveform viewer
#===============================================================================
wave_dpd:
	gtkwave $(WAVE_DIR)/$(TB_DPD_TOP).vcd &

wave_tdnn:
	gtkwave $(WAVE_DIR)/$(TB_TDNN).vcd &

wave_aspsa:
	gtkwave $(WAVE_DIR)/$(TB_ASPSA).vcd &

#===============================================================================
# Clean
#===============================================================================
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(WAVE_DIR)
	rm -rf obj_dir
	rm -f *.vcd *.fst *.lxt
	@echo "=== Clean Complete ==="

#===============================================================================
# Vivado targets (for actual FPGA builds)
#===============================================================================
.PHONY: vivado_pynq vivado_zcu104

vivado_pynq:
	@echo "=== Starting Vivado build for PYNQ-Z1 ==="
	vivado -mode batch -source scripts/build_pynq.tcl

vivado_zcu104:
	@echo "=== Starting Vivado build for ZCU104 ==="
	vivado -mode batch -source scripts/build_zcu104.tcl
