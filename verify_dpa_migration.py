#!/usr/bin/env python3
"""
Verification script for dpa_200mhz dataset migration.
Checks that all configuration changes are correctly applied.
"""

import json
import yaml
import sys
from pathlib import Path


def check_config_yaml():
    """Verify config.yaml has dpa_200mhz parameters."""
    print("\n" + "="*60)
    print("CHECKING: config/config.yaml")
    print("="*60)
    
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print("❌ config/config.yaml not found")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Convert to float, handling both int and string types
    def to_float(val):
        if isinstance(val, (int, float)):
            return float(val)
        return float(str(val).replace('e', 'E'))
    
    checks = [
        ("system.sample_rate", to_float(config['system']['sample_rate']), 200e6),
        ("spectral_loss.bw_main_ch", to_float(config['spectral_loss']['bw_main_ch']), 200e6),
        ("spectral_loss.n_sub_ch", config['spectral_loss']['n_sub_ch'], 10),
        ("dataset.name", config['dataset']['name'], 'dpa_200mhz'),
        ("dataset.n_channels", config['dataset']['n_channels'], 1),
        ("dataset.n_sub_channels", config['dataset']['n_sub_channels'], 10),
    ]
    
    passed = True
    for key, actual, expected in checks:
        if actual == expected:
            print(f"✓ {key}: {actual}")
        else:
            print(f"❌ {key}: {actual} (expected {expected})")
            passed = False
    
    return passed


def check_dpa_dataset():
    """Verify DPA dataset exists and has correct structure."""
    print("\n" + "="*60)
    print("CHECKING: DPA 200MHz Dataset")
    print("="*60)
    
    data_dir = Path("data/DPA_200MHz")
    if not data_dir.exists():
        print(f"❌ {data_dir} not found")
        return False
    
    required_files = [
        "train_input.csv", "train_output.csv",
        "val_input.csv", "val_output.csv",
        "test_input.csv", "test_output.csv",
        "spec.json"
    ]
    
    passed = True
    for fname in required_files:
        fpath = data_dir / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            print(f"✓ {fname}: {size_kb:.1f} KB")
        else:
            print(f"❌ {fname}: NOT FOUND")
            passed = False
    
    # Check spec.json
    if (data_dir / "spec.json").exists():
        with open(data_dir / "spec.json", 'r', encoding='utf-8') as f:
            spec = json.load(f)
        
        print(f"\n  Dataset spec:")
        print(f"    n_sub_ch: {spec['n_sub_ch']} (expected: 10)")
        print(f"    bw_main_ch: {spec['bw_main_ch']/1e6:.0f} MHz (expected: 200)")
        print(f"    bw_sub_ch: {spec['bw_sub_ch']/1e6:.0f} MHz (expected: 20)")
        print(f"    nperseg: {spec['nperseg']} (expected: >= 1024)")
        
        if spec['n_sub_ch'] != 10:
            print(f"❌ n_sub_ch mismatch")
            passed = False
        if spec['bw_main_ch'] != 200e6:
            print(f"❌ bw_main_ch mismatch")
            passed = False
    
    return passed


def check_train_py():
    """Verify train.py has dpa_200mhz parameters in SpectralLoss."""
    print("\n" + "="*60)
    print("CHECKING: train.py SpectralLoss initialization")
    print("="*60)
    
    train_path = Path("train.py")
    if not train_path.exists():
        print("❌ train.py not found")
        return False
    
    with open(train_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for key parameters in SpectralLoss instantiation
    checks = [
        ("n_sub_ch", "n_sub_ch=" in content),
        ("bw_main_ch", "bw_main_ch=" in content),
        ("sample_rate 200e6", "200e6" in content),
    ]
    
    passed = True
    for key, found in checks:
        if found:
            print(f"✓ {key} parameter found in SpectralLoss")
        else:
            print(f"❌ {key} parameter NOT found")
            passed = False
    
    return passed


def check_notebook():
    """Verify training_colab_v2.ipynb has dpa_200mhz parameters."""
    print("\n" + "="*60)
    print("CHECKING: training_colab_v2.ipynb")
    print("="*60)
    
    notebook_path = Path("training_colab_v2.ipynb")
    if not notebook_path.exists():
        print("❌ training_colab_v2.ipynb not found")
        return False
    
    import json as json_lib
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json_lib.load(f)
    
    # Search for config dict in cells
    config_found = False
    spectral_loss_found = False
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            if "'dataset'" in source and "'n_sub_channels': 10" in source:
                config_found = True
                print("✓ Cell 3: Dataset configuration with n_sub_channels=10 found")
            
            if "SpectralLoss(" in source and "n_sub_ch=" in source:
                spectral_loss_found = True
                print("✓ Cell 7: SpectralLoss initialization with n_sub_ch parameter found")
    
    if not config_found:
        print("❌ Dataset configuration in Cell 3 not found")
    if not spectral_loss_found:
        print("❌ SpectralLoss with n_sub_ch in Cell 7 not found")
    
    return config_found and spectral_loss_found


def main():
    """Run all verification checks."""
    print("\n" + "="*80)
    print("DPA 200MHz DATASET MIGRATION - VERIFICATION")
    print("="*80)
    
    results = {
        "config.yaml": check_config_yaml(),
        "DPA Dataset": check_dpa_dataset(),
        "train.py": check_train_py(),
        "training_colab_v2.ipynb": check_notebook(),
    }
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check}")
    
    print(f"\nTotal: {passed_count}/{total_count} checks passed")
    
    if passed_count == total_count:
        print("\n✓ All migration checks PASSED - Ready for training!")
        return 0
    else:
        print("\n❌ Some checks failed - Review changes above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
