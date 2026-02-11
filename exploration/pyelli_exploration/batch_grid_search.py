import sys
import os
import time
import re
from pathlib import Path
import pandas as pd
import numpy as np
import multiprocessing

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exploration.pyelli_exploration.pyelli_grid_search import (
    PyElliGridSearch,
    calculate_peak_based_score,
)
from exploration.pyelli_exploration.pyelli_utils import (
    load_measured_spectrum,
    load_bestfit_spectrum,
)


def calculate_deviation_score(
    results_obj, grid_search, bestfit_wl, bestfit_refl, wl_fit, meas_fit
):
    """
    Calculate the 'PyElli vs LTA Deviation Score' exactly as in app.py
    """
    try:
        # Interpolate BestFit to match our fitting wavelengths
        bestfit_interp = np.interp(wl_fit, bestfit_wl, bestfit_refl)

        # IMPORTANT: Align BestFit to measured exactly as app.py does (handles baseline shifts)
        bestfit_aligned = grid_search._align_spectra(
            meas_fit,
            bestfit_interp,
            focus_min=wl_fit[0],
            focus_max=wl_fit[-1],
            wavelengths=wl_fit,
        )

        # PyElli theoretical already calculated on wl_fit equivalents
        pyelli_theo = np.interp(
            wl_fit, results_obj.wavelengths, results_obj.theoretical_spectrum
        )

        # Calculate LTA metrics against measured (using the ALIGNED bestfit)
        bestfit_score_result = calculate_peak_based_score(
            wl_fit, meas_fit, bestfit_aligned
        )

        # 1. MAPE (Mean Absolute Percentage Error)
        lta_abs = np.abs(bestfit_aligned)
        valid_mask = lta_abs > 1e-10
        if valid_mask.any():
            mape = (
                float(
                    np.mean(
                        np.abs(pyelli_theo[valid_mask] - bestfit_aligned[valid_mask])
                        / lta_abs[valid_mask]
                    )
                )
                * 100
            )
        else:
            mape = 0.0

        # 2. Peak match rate deviation
        pyelli_matched = float(results_obj.matched_peaks)
        lta_peaks = float(bestfit_score_result.get("measurement_peaks", 0))
        if lta_peaks > 0:
            peak_match_deviation = (1.0 - (pyelli_matched / lta_peaks)) * 100
            peak_match_deviation = max(0.0, peak_match_deviation)
        else:
            peak_match_deviation = 0.0

        # 3. Alignment deviation (relative to 50nm spacing)
        pyelli_mean_delta = float(results_obj.mean_delta_nm)
        reference_spacing = 50.0
        alignment_deviation = (pyelli_mean_delta / reference_spacing) * 100

        # Composite score: 40% MAPE, 30% Peak Match, 30% Alignment
        composite_deviation = (
            0.40 * mape + 0.30 * peak_match_deviation + 0.30 * alignment_deviation
        )

        return {
            "composite_dev": composite_deviation,
            "mape": mape,
            "peak_match_dev": peak_match_deviation,
            "alignment_dev": alignment_deviation,
        }
    except Exception as e:
        print(f"  Error calculating deviation: {e}")
        return None


def run_batch_fitting(input_dir, output_file, max_files=40, explicit_file_list=None):
    input_path = Path(input_dir)
    bestfit_dir = PROJECT_ROOT / "exploration" / "more_good_spectras" / "BestFit"

    # Match Ground Truth Excel
    excel_path = (
        PROJECT_ROOT
        / "exploration"
        / "more_good_spectras"
        / "Lipid  and  Mucus-Aqueous_Height.xlsx"
    )
    gt_df = pd.read_excel(excel_path)
    gt_lookup = {
        str(row["Absolute time"]): (row["Lipid_Height"], row["Mucus-Aqueous_Height"])
        for _, row in gt_df.iterrows()
    }

    results = []
    processed_files = set()

    # RESUME LOGIC: Load existing results if they exist
    if Path(output_file).exists():
        try:
            existing_df = pd.read_csv(output_file)
            # Convert NaNs to None for consistency with internal logic
            existing_df = existing_df.replace({np.nan: None})
            results = existing_df.to_dict("records")
            # We must be careful: stored filenames might already have (Run) stripped or not.
            # The current script logic below STRIPS (Run) before saving.
            # So if we load old results, they likely have (Run) or not depending on when they were generated.
            # We normalized to matching what is in the CSV.
            processed_files = set(r["filename"] for r in results)
            print(f"Loaded {len(results)} existing results from {output_file}.")
        except Exception as e:
            print(f"Warning: Could not read existing results: {e}. Starting fresh.")

    # Determine files to process
    if explicit_file_list:
        print(f"Using explicit file list with {len(explicit_file_list)} entries.")
        target_files = []
        for fname in explicit_file_list:
            fname = fname.strip()
            if not fname:
                continue

            # Try direct match
            fpath = input_path / fname
            if not fpath.exists():
                # Try adding (Run) prefix if missing
                if not fname.startswith("(Run)"):
                    fpath_run = input_path / f"(Run){fname}"
                    if fpath_run.exists():
                        fpath = fpath_run

                # Try removing (Run) prefix if present (unlikely but possible user input)
                if not fpath.exists() and fname.startswith("(Run)"):
                    fpath_norun = input_path / fname.replace("(Run)", "")
                    if fpath_norun.exists():
                        fpath = fpath_norun

            if fpath.exists():
                target_files.append(fpath)
            else:
                print(f"Warning: File {fname} not found in {input_dir}")

        # When using explicit list, we ignore max_files limit usually, but let's respect it if smaller?
        # Typically explicit list implies "process exactly these".
        # We will NOT apply max_files to the explicit list, unless we really want partial processing.
        # But consistent with previous logic, max_files was a slicer.
        # Let's assume explicit list overrides max_files "selection" behavior.
        total_files = len(target_files)
    else:
        # Default behavior: glob everything
        all_spectrum_files = sorted(list(input_path.glob("(Run)spectra_*.txt")))
        total_files = len(all_spectrum_files)
        # Apply limit
        target_files = all_spectrum_files[:max_files]

    materials_path = PROJECT_ROOT / "data" / "Materials"
    grid_search = PyElliGridSearch(materials_path)

    # EXACT Streamlit App Defaults
    LIPID_RANGE = (9.0, 250.0, 5.0)
    AQUEOUS_RANGE = (800.0, 12000.0, 200.0)
    ROUGHNESS_RANGE = (600.0, 7000.0, 100.0)
    MAX_COMBINATIONS = 30000
    SEARCH_STRATEGY = "Dynamic Search"
    WL_MIN, WL_MAX = 600.0, 1120.0

    # Filter out already processed files
    # Note: processed_files (from CSV) might be "spectra_..." (without Run).
    # target_files (from Path) are "(Run)spectra_..." (with Run).
    # We need to normalize for comparison.
    files_to_process = []
    for f in target_files:
        # Normalized name for CSV storage and checking
        csv_name = f.name.replace("(Run)", "")
        if csv_name not in processed_files:
            files_to_process.append(f)

    total_target = len(target_files)
    num_already_done = total_target - len(files_to_process)

    print(f"Targeting {total_target} files (out of {total_files} total matches).")
    print(f"Skipping {num_already_done} already processed files.")
    print(f"Queueing {len(files_to_process)} new files for processing.")

    for i, file_path in enumerate(files_to_process):
        # Index for display (resume index)
        current_idx = num_already_done + i + 1
        print(f"[{current_idx}/{total_target}] Fitting {file_path.name}...")

        time_match = re.search(r"(\d{2}-\d{2}-\d{2}-\d+)", file_path.name)
        time_id = time_match.group(1) if time_match else None
        gt_l, gt_a = gt_lookup.get(time_id, (None, None))

        if gt_l is None or pd.isna(gt_l):
            print(f"Skipping {file_path.name}: No ground truth found for ID {time_id}")
            continue

        try:
            wavelengths, measured = load_measured_spectrum(file_path)
            wl_mask = (wavelengths >= WL_MIN) & (wavelengths <= WL_MAX)
            wl_fit, meas_fit = wavelengths[wl_mask], measured[wl_mask]

            best_results = []
            for attempt in range(3):
                try:
                    best_results = grid_search.run_grid_search(
                        wl_fit,
                        meas_fit,
                        lipid_range=LIPID_RANGE,
                        aqueous_range=AQUEOUS_RANGE,
                        roughness_range=ROUGHNESS_RANGE,
                        top_k=10,
                        search_strategy=SEARCH_STRATEGY,
                        max_combinations=MAX_COMBINATIONS,
                        enable_roughness=True,
                    )
                    break
                except Exception as e:
                    if "Interrupted system call" in str(e) and attempt < 2:
                        print(f"  Attempt {attempt + 1} interrupted, retrying...")
                        time.sleep(1)
                        continue
                    raise e

            if best_results:
                best = best_results[0]

                # Deviation Analysis (vs LTA BestFit)
                dev_metrics = None
                bestfit_name = file_path.name.replace(".txt", "_BestFit.txt")
                bestfit_path = bestfit_dir / bestfit_name
                if bestfit_path.exists():
                    bf_wl, bf_refl = load_bestfit_spectrum(bestfit_path)
                    dev_metrics = calculate_deviation_score(
                        best, grid_search, bf_wl, bf_refl, wl_fit, meas_fit
                    )

                row = {
                    "filename": file_path.name.replace("(Run)", ""),
                    "lipid_fit": best.lipid_nm,
                    "aqueous_fit": best.aqueous_nm,
                    "roughness_fit": best.mucus_nm,
                    "score": best.score,
                    "corr": best.correlation,
                    "osc_ratio": best.oscillation_ratio,
                    "dev_score": dev_metrics["composite_dev"] if dev_metrics else None,
                    "gt_lipid": gt_l,
                    "gt_aqueous": gt_a,
                }
                results.append(row)

                # Incremental Save
                pd.DataFrame(results).to_csv(output_file, index=False)

                # Format summary text file exactly as requested
                with open(Path(output_file).with_suffix(".txt"), "w") as f:
                    # Fix: Use total_target instead of undefined spectrum_files
                    f.write(
                        f"Batch Spectral Fitting - PROGRESS: {num_already_done + i + 1}/{total_target}\n"
                    )
                    f.write("=" * 48 + "\n\n")
                    header = f"{'Filename':<40} | {'Lipid(nm)':>10} | {'Aqueous(nm)':>12} | {'Roughness(A)':>13} | {'Score':>8} | {'Corr':>8} | {'OscPct':>8} | {'DevScore(%)':>12} | {'GT_L':>6} | {'GT_A':>6}\n"
                    f.write(header)
                    f.write("-" * len(header) + "\n")
                    for r in results:
                        dev_str = (
                            f"{r['dev_score']:>11.1f}%"
                            if r["dev_score"] is not None
                            else "        N/A"
                        )
                        gt_l_str = (
                            f"{r['gt_lipid']:>6.1f}"
                            if r["gt_lipid"] is not None
                            else "   N/A"
                        )
                        gt_a_str = (
                            f"{r['gt_aqueous']:>6.0f}"
                            if r["gt_aqueous"] is not None
                            else "   N/A"
                        )
                        osc_pct = r["osc_ratio"] * 100
                        f.write(
                            f"{r['filename']:<40} | {r['lipid_fit']:>10.1f} | {r['aqueous_fit']:>12.1f} | {r['roughness_fit']:>13.1f} | {r['score']:>8.4f} | {r['corr']:>8.4f} | {osc_pct:>7.1f}% | {dev_str} | {gt_l_str} | {gt_a_str}\n"
                        )

                print(
                    f"  Best: L={best.lipid_nm:.1f}, A={best.aqueous_nm:.1f} | Score: {best.score:.3f}"
                )
            else:
                print("  No fit found.")
        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")


if __name__ == "__main__":
    # macOS multiprocessing fix
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except Exception:
        pass

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, default="exploration/more_good_spectras/Corrected_Spectra"
    )
    parser.add_argument("--out", type=str, default="batch_results.csv")
    parser.add_argument("--max", type=int, default=40)
    parser.add_argument(
        "--list",
        type=str,
        default=None,
        help="Path to text file containing list of filenames to process",
    )
    args = parser.parse_args()

    explicit_list = None
    if args.list and Path(args.list).exists():
        with open(args.list, "r") as f:
            explicit_list = [line.strip() for line in f if line.strip()]

    run_batch_fitting(args.dir, args.out, args.max, explicit_list)
