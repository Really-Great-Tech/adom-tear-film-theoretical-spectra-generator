import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path("/Users/dede/Lab/adom-tear-film-theoretical-spectra-generator")
sys.path.append(str(project_root))

from src.analysis.measurement_utils import (
    load_txt_file_enhanced,
    detrend_signal,
    calculate_snr,
    calculate_snr_profile,
)


def run_snr_on_files(files):
    results = []

    for f in files:
        try:
            df = load_txt_file_enhanced(f)
            wavelengths = df["wavelength"].to_numpy()
            reflectance = df["reflectance"].to_numpy()

            # Use standard wavelength range 600-1120
            mask = (wavelengths >= 600) & (wavelengths <= 1120)
            wl = wavelengths[mask]
            refl = reflectance[mask]

            if len(wl) < 10:
                continue

            detrended = detrend_signal(wl, refl, 0.008, filter_order=3)
            global_snr = calculate_snr(wl, detrended)
            snr_wl, snr_vals = calculate_snr_profile(wl, detrended)

            min_win_snr = np.min(snr_vals) if len(snr_vals) > 0 else 0
            avg_win_snr = np.mean(snr_vals) if len(snr_vals) > 0 else 0

            results.append(
                {
                    "filename": f.name,
                    "global_snr": global_snr,
                    "min_win_snr": min_win_snr,
                    "avg_win_snr": avg_win_snr,
                }
            )
        except Exception as e:
            # print(f"Error processing {f.name}: {e}")
            pass

    return pd.DataFrame(results)


if __name__ == "__main__":
    # 1. More Good Spectras
    print("--- CATEGORY: MORE GOOD SPECTRA ---")
    good_files = sorted(
        list(
            (
                project_root
                / "exploration"
                / "more_good_spectras"
                / "Corrected_Spectra"
            ).glob("*.txt")
        )
    )[:10]
    df_good = run_snr_on_files(good_files)
    print(df_good.describe())

    # 2. New Spectra (often mixed quality)
    print("\n--- CATEGORY: NEW SPECTRA (Mixed) ---")
    new_files = []
    for f in (project_root / "exploration" / "new_spectra").rglob("*.txt"):
        if "_BestFit" not in f.name:
            new_files.append(f)
            if len(new_files) >= 10:
                break
    df_new = run_snr_on_files(new_files)
    print(df_new.describe())

    # 3. Sample Data
    print("\n--- CATEGORY: SAMPLE DATA ---")
    sample_files = []
    for f in (project_root / "exploration" / "sample_data").rglob("*.txt"):
        if "_BestFit" not in f.name:
            sample_files.append(f)
            if len(sample_files) >= 10:
                break
    df_sample = run_snr_on_files(sample_files)
    print(df_sample.describe())
