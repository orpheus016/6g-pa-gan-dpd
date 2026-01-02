#!/usr/bin/env python3
"""
Validate RTL parameter consistency with Python model
"""

import re
from pathlib import Path
import os

print("="*70)
print("🔍 RTL Parameter Validation")
print("="*70)

# Model dimensions (UPDATED for nonlinear features)
MEMORY_DEPTH = 5
INPUT_DIM = 30                              # 2 + 3*(M+1) + 2*M = 30 for M=5
HIDDEN1_DIM = 32
HIDDEN2_DIM = 16
OUTPUT_DIM = 2

# Parameter counts
FC1_WEIGHTS = INPUT_DIM * HIDDEN1_DIM      # 960
FC1_BIASES = HIDDEN1_DIM                   # 32
FC2_WEIGHTS = HIDDEN1_DIM * HIDDEN2_DIM    # 512
FC2_BIASES = HIDDEN2_DIM                   # 16
FC3_WEIGHTS = HIDDEN2_DIM * OUTPUT_DIM     # 32
FC3_BIASES = OUTPUT_DIM                    # 2

TOTAL_PARAMS = FC1_WEIGHTS + FC1_BIASES + FC2_WEIGHTS + FC2_BIASES + FC3_WEIGHTS + FC3_BIASES  # 1554

print(f"\n📊 Python Model (Expected):")
print(f"   Input: {INPUT_DIM}, Hidden1: {HIDDEN1_DIM}, Hidden2: {HIDDEN2_DIM}, Output: {OUTPUT_DIM}")
print(f"   FC1: {INPUT_DIM}×{HIDDEN1_DIM} = {FC1_WEIGHTS} + {FC1_BIASES} = {FC1_WEIGHTS+FC1_BIASES}")
print(f"   FC2: {HIDDEN1_DIM}×{HIDDEN2_DIM} = {FC2_WEIGHTS} + {FC2_BIASES} = {FC2_WEIGHTS+FC2_BIASES}")
print(f"   FC3: {HIDDEN2_DIM}×{OUTPUT_DIM} = {FC3_WEIGHTS} + {FC3_BIASES} = {FC3_WEIGHTS+FC3_BIASES}")
print(f"   TOTAL: {TOTAL_PARAMS} parameters/bank")

# Check RTL files (use relative paths from script location)
script_dir = Path(__file__).parent
tdnn_path = script_dir / 'src' / 'tdnn_generator.v'
top_path = script_dir / 'src' / 'dpd_top.v'

errors = []

print(f"\n📝 Checking {tdnn_path.name}...")
with open(tdnn_path, encoding='utf-8') as f:
    tdnn_content = f.read()
    
    # Check INPUT_DIM (either hardcoded 30 or uses correct constant)
    if 'INPUT_DIM     = 30' in tdnn_content:
        print("   ✅ INPUT_DIM = 30 (correct for M=5 with nonlinear features)")
    else:
        errors.append("INPUT_DIM should be 30 (2 + 3*(M+1) + 2*M for M=5)")
    
    # Check BANK_SIZE
    if 'BANK_SIZE = 1554' in tdnn_content:
        print("   ✅ BANK_SIZE = 1554 (correct)")
    elif 'BANK_SIZE = 1298' in tdnn_content:
        errors.append("BANK_SIZE = 1298 (should be 1554 for 30-dim input)")
    elif 'BANK_SIZE = 1170' in tdnn_content:
        errors.append("BANK_SIZE = 1170 (should be 1554 for 30-dim input)")
    
    # Check weight offsets
    if 'WADDR_FC1 = 0' in tdnn_content and 'WADDR_B1  = 960' in tdnn_content:
        print("   ✅ FC1 weight offsets correct (960 weights)")
    elif 'WADDR_B1  = 704' in tdnn_content:
        errors.append("FC1 uses 704 weights (should be 960 for 30 inputs)")
    elif 'WADDR_B1  = 576' in tdnn_content:
        errors.append("FC1 uses 576 weights (should be 960 for 30 inputs)")


print(f"\n📝 Checking {top_path.name}...")
with open(top_path, encoding='utf-8') as f:
    top_content = f.read()
    
    if 'TOTAL_WEIGHTS   = 1554' in top_content:
        print("   ✅ TOTAL_WEIGHTS = 1554 (correct)")
    elif 'TOTAL_WEIGHTS   = 1298' in top_content:
        errors.append("TOTAL_WEIGHTS = 1298 (should be 1554)")
    elif 'TOTAL_WEIGHTS   = 1170' in top_content:
        errors.append("TOTAL_WEIGHTS = 1170 (should be 1554)")

print("\n" + "="*70)
if not errors:
    print("✅ ALL CHECKS PASSED - RTL matches Python model!")
    print("\n   Architecture: 30 → 32 → 16 → 2")
    print(f"   Total params: {TOTAL_PARAMS} per temperature bank")
    print("   ✅ Ready for training and synthesis!")
else:
    print("❌ VALIDATION FAILED:")
    for err in errors:
        print(f"   • {err}")
    exit(1)
