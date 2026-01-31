import numpy as np
import pandas as pd
import sys

def compute_papr(iq_data):
    x = iq_data['I'].values + 1j * iq_data['Q'].values
    power = np.abs(x) ** 2
    papr_linear = np.max(power) / np.mean(power)
    papr_db = 10 * np.log10(papr_linear)
    return papr_linear, papr_db

def main(csv_path):
    df = pd.read_csv(csv_path)
    if not {'I', 'Q'}.issubset(df.columns):
        raise ValueError("CSV must contain 'I' and 'Q' columns.")
    papr_linear, papr_db = compute_papr(df)
    print(f"PAPR (linear): {papr_linear:.3f}")
    print(f"PAPR (dB): {papr_db:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tests/papr.py <path_to_csv>")
        sys.exit(1)
    main(sys.argv[1])