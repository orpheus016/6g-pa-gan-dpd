#!/usr/bin/env python3
"""
Validate RTL parameter consistency with Python model
"""

import re
from pathlib import Path

print("="*70)
print("🔍 RTL Parameter Validation")
print("="*70)

# Python Model Parameters
MEMORY_DEPTH = 5
INPUT_DIM = 2 + 2 * MEMORY_DEPTH * 2  # 22
HIDDEN1_DIM = 32
HIDDEN2_DIM = 16
OUTPUT_DIM = 2

FC1_WEIGHTS = INPUT_DIM * HIDDEN1_DIM      # 704
FC1_BIASES = HIDDEN1_DIM                   # 32
FC2_WEIGHTS = HIDDEN1_DIM * HIDDEN2_DIM    # 512
FC2_BIASES = HIDDEN2_DIM                   # 16
FC3_WEIGHTS = HIDDEN2_DIM * OUTPUT_DIM     # 32
FC3_BIASES = OUTPUT_DIM                    # 2

TOTAL_PARAMS = FC1_WEIGHTS + FC1_BIASES + FC2_WEIGHTS + FC2_BIASES + FC3_WEIGHTS + FC3_BIASES  # 1298

print(f"\n📊 Python Model (Expected):")
print(f"   Input: {INPUT_DIM}, Hidden1: {HIDDEN1_DIM}, Hidden2: {HIDDEN2_DIM}, Output: {OUTPUT_DIM}")
print(f"   FC1: {INPUT_DIM}×{HIDDEN1_DIM} = {FC1_WEIGHTS} + {FC1_BIASES} = {FC1_WEIGHTS+FC1_BIASES}")
print(f"   FC2: {HIDDEN1_DIM}×{HIDDEN2_DIM} = {FC2_WEIGHTS} + {FC2_BIASES} = {FC2_WEIGHTS+FC2_BIASES}")
print(f"   FC3: {HIDDEN2_DIM}×{OUTPUT_DIM} = {FC3_WEIGHTS} + {FC3_BIASES} = {FC3_WEIGHTS+FC3_BIASES}")
print(f"   TOTAL: {TOTAL_PARAMS} parameters/bank")

# Check RTL files
tdnn_path = Path('/home/james-patrick/eda/designs/github/6g-pa-gan-dpd/rtl/src/tdnn_generator.v')
top_path = Path('/home/james-patrick/eda/designs/github/6g-pa-gan-dpd/rtl/src/dpd_top.v')

errors = []

print(f"\n📝 Checking {tdnn_path.name}...")
with open(tdnn_path) as f:
    tdnn_content = f.read()
    
    # Check for correct formula
    if '2 + 2*MEMORY_DEPTH*2' in tdnn_content or '2*MEMORY_DEPTH*2' in tdnn_content:
        print("   ✅ INPUT_DIM uses parameterized formula")
    else:
        errors.append("INPUT_DIM should use '2 + 2*MEMORY_DEPTH*2'")
    
    # Check BANK_SIZE
    if 'BANK_SIZE = 1298' in tdnn_content:
        print("   ✅ BANK_SIZE = 1298 (correct)")
    elif 'BANK_SIZE = 1170' in tdnn_content:
        errors.append("BANK_SIZE = 1170 (should be 1298)")
    
    # Check weight offsets
    if 'WADDR_FC1 = 0' in tdnn_content and 'WADDR_B1  = 704' in tdnn_content:
        print("   ✅ FC1 weight offsets correct (704 weights)")
    elif 'WADDR_B1  = 576' in tdnn_content:
        errors.append("FC1 uses 576 weights (should be 704 for 22 inputs)")

print(f"\n📝 Checking {top_path.name}...")
with open(top_path) as f:
    top_content = f.read()
    
    if 'TOTAL_WEIGHTS   = 1298' in top_content:
        print("   ✅ TOTAL_WEIGHTS = 1298 (correct)")
    elif 'TOTAL_WEIGHTS   = 1170' in top_content:
        errors.append("TOTAL_WEIGHTS = 1170 (should be 1298)")

print("\n" + "="*70)
if not errors:
    print("✅ ALL CHECKS PASSED - RTL matches Python model!")
    print("\n   Architecture: 22 → 32 → 16 → 2")
    print(f"   Total params: {TOTAL_PARAMS} per temperature bank")
    print("   ✅ Ready for training and synthesis!")
else:
    print("❌ VALIDATION FAILED:")
    for err in errors:
        print(f"   • {err}")
    exit(1)
