#!/bin/bash
#===============================================================================
# Setup Script for 6G PA DPD Development Environment
# Creates Python virtual environment and installs dependencies
#===============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

echo "=============================================="
echo "6G PA GAN-DPD Development Environment Setup"
echo "=============================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: ${PYTHON_VERSION}"

if [[ "${PYTHON_VERSION}" < "3.8" ]]; then
    echo "ERROR: Python 3.8+ required"
    exit 1
fi

# Create virtual environment
if [ -d "${VENV_DIR}" ]; then
    echo "Virtual environment already exists at ${VENV_DIR}"
    read -p "Remove and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${VENV_DIR}"
    else
        echo "Using existing virtual environment"
    fi
fi

if [ ! -d "${VENV_DIR}" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU version for development, GPU if available)
echo ""
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, installing CUDA version"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "No GPU detected, installing CPU version"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other dependencies
echo ""
echo "Installing project dependencies..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

# Install Jupyter for notebook development
echo ""
echo "Installing Jupyter..."
pip install jupyter ipykernel ipywidgets

# Register kernel
python -m ipykernel install --user --name=dpd-venv --display-name="DPD (venv)"

# Verify installation
echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="

python3 << 'EOF'
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

import numpy as np
print(f"NumPy: {np.__version__}")

import scipy
print(f"SciPy: {scipy.__version__}")

import yaml
print(f"PyYAML: {yaml.__version__}")

import matplotlib
print(f"Matplotlib: {matplotlib.__version__}")

print("\n✓ All dependencies installed successfully!")
EOF

# Check RTL tools
echo ""
echo "=============================================="
echo "Checking RTL simulation tools..."
echo "=============================================="

if command -v iverilog &> /dev/null; then
    echo "✓ Icarus Verilog: $(iverilog -V 2>&1 | head -1)"
else
    echo "✗ Icarus Verilog not found (install with: sudo apt install iverilog)"
fi

if command -v gtkwave &> /dev/null; then
    echo "✓ GTKWave: $(gtkwave --version 2>&1 | head -1)"
else
    echo "✗ GTKWave not found (install with: sudo apt install gtkwave)"
fi

if command -v verilator &> /dev/null; then
    echo "✓ Verilator: $(verilator --version 2>&1)"
else
    echo "○ Verilator not found (optional, install with: sudo apt install verilator)"
fi

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run training:"
echo "  python train.py --config config/config.yaml"
echo ""
echo "To run RTL simulation:"
echo "  cd rtl && make sim_all"
echo ""
echo "To deactivate:"
echo "  deactivate"
