#!/usr/bin/env python3
"""
Last-two-cycles seed + tune: full grid on one spectrum, refine around that seed on the rest.

Identifies spectra from the last two blink cycles (by time window) that have BestFit,
runs full grid search on one spectrum to get a seed (L, A, R), then refines around
that seed for all other spectra. Tracks how many have deviation ≤10% vs LTA BestFit;
stops early once 50 are reached (goal).

Usage:
  python -m exploration.pyelli_exploration.run_last_two_cycles_seed_tune --run-dir "/path/To/Full test - 0007_2025-12-30_15-12-20"
  Or with explicit Corrected/BestFit:
  python -m exploration.pyelli_exploration.run_last_two_cycles_seed_tune --corrected-dir ... --bestfit-dir ... [--run-dir for time parsing]

Expects run_dir to contain Spectra/GoodSpectra/Corrected and BestFit, or pass
--corrected-dir and --bestfit-dir. Run folder name should end with _HH-MM-SS (e.g. _15-12-20)
for run start time; otherwise last 10 s are taken from max timestamp in Corrected filenames.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exploration.pyelli_exploration.pyelli_utils import (
    load_measured_spectrum,
    load_bestfit_spectrum,
)
from exploration.pyelli_exploration.pyelli_grid_search import PyElliGridSearch
from exploration.pyelli_exploration.optimize_score_weights import (
    compute_deviation_for_params,
    WL_MIN,
    WL_MAX,
)

# Last two cycles = last 10 s of acquisition (InterBlinkInterval ~5 s)
LAST_TWO_CYCLES_SEC = 10.0
GOAL_UNDER_10_PCT = 50

# Roughness (Å): fixed range for good pyelli fits; do not use values outside this.
ROUGHNESS_MIN_ANGSTROM = 6000.0
ROUGHNESS_MAX_ANGSTROM = 7000.0


def parse_timestamp_from_filename(name: str) -> float | None:
    """Parse (Run)spectra_15-13-26-238.txt -> seconds from midnight: 15*3600 + 13*60 + 26.238."""
    m = re.match(r"\(Run\)spectra_(\d+)-(\d+)-(\d+)-(\d+)\.txt", name)
    if not m:
        return None
    h, mm, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return h * 3600 + mm * 60 + s + ms / 1000.0


def get_run_start_sec_from_folder_name(run_dir: Path) -> float | None:
    """Parse run folder name like 'Full test - 0007_2025-12-30_15-12-20' -> 15:12:20 in sec from midnight."""
    name = run_dir.name
    # Match _HH-MM-SS at end (after last _)
    m = re.search(r"_(\d{2})-(\d{2})-(\d{2})$", name)
    if not m:
        return None
    h, mm, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return h * 3600 + mm * 60 + s


def identify_last_two_cycles_spectra(
    corrected_dir: Path,
    bestfit_dir: Path,
    run_start_sec: float | None,
) -> list[tuple[Path, Path, float]]:
    """
    Return list of (measured_path, bestfit_path, time_sec_from_start) for spectra
    in the last two cycles (last 10 s) that have a BestFit file. Sorted by time.
    """
    corrected_dir = Path(corrected_dir)
    bestfit_dir = Path(bestfit_dir)
    files = [
        f
        for f in corrected_dir.iterdir()
        if f.is_file()
        and f.suffix == ".txt"
        and f.name.startswith("(Run)spectra_")
        and not f.name.endswith("_BestFit.txt")
    ]
    # Parse timestamp (from midnight) for each
    with_time: list[tuple[Path, float]] = []
    for f in files:
        ts = parse_timestamp_from_filename(f.name)
        if ts is not None:
            with_time.append((f, ts))

    if not with_time:
        return []

    # If we have run start, use it; else use min timestamp in list as reference
    if run_start_sec is not None:
        ref = run_start_sec
    else:
        ref = min(t for _, t in with_time)
    # Time from ref for each file
    with_sec = [(p, t - ref) for p, t in with_time]
    t_max = max(s for _, s in with_sec)
    t_cut = t_max - LAST_TWO_CYCLES_SEC
    last_two = [(p, s) for p, s in with_sec if s >= t_cut]
    last_two.sort(key=lambda x: x[1])

    # Filter to those that have BestFit
    out: list[tuple[Path, Path, float]] = []
    for meas_path, time_sec in last_two:
        stem = meas_path.stem
        bf_path = bestfit_dir / f"{stem}_BestFit.txt"
        if bf_path.exists():
            out.append((meas_path, bf_path, time_sec))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Last-two-cycles seed + tune: grid on one, refine on rest; count deviation ≤10%."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run folder (e.g. 'Full test - 0007_2025-12-30_15-12-20'). Uses Spectra/GoodSpectra/Corrected and BestFit inside.",
    )
    parser.add_argument(
        "--corrected-dir",
        type=Path,
        default=None,
        help="Override: directory with (Run)spectra_*.txt",
    )
    parser.add_argument(
        "--bestfit-dir",
        type=Path,
        default=None,
        help="Override: directory with *_BestFit.txt",
    )
    parser.add_argument(
        "--materials",
        type=Path,
        default=PROJECT_ROOT / "data" / "Materials",
        help="Materials directory for PyElliGridSearch",
    )
    parser.add_argument(
        "--seed-index",
        type=int,
        default=None,
        help="Index into last-two-cycles list for which spectrum to use as seed (default 0).",
    )
    parser.add_argument(
        "--seed-spectrum",
        type=str,
        default=None,
        help="Use this spectrum as seed by filename, e.g. '(Run)spectra_15-13-16-509.txt'. Overrides --seed-index.",
    )
    parser.add_argument(
        "--max-tune",
        type=int,
        default=None,
        help="Max number of spectra to tune (for quick tests). Default: no limit.",
    )
    args = parser.parse_args()

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        corrected_dir = args.corrected_dir or (run_dir / "Spectra" / "GoodSpectra" / "Corrected")
        bestfit_dir = args.bestfit_dir or (run_dir / "BestFit")
        run_start_sec = get_run_start_sec_from_folder_name(run_dir)
    else:
        if args.corrected_dir is None or args.bestfit_dir is None:
            parser.error("Either --run-dir or both --corrected-dir and --bestfit-dir are required.")
        corrected_dir = Path(args.corrected_dir)
        bestfit_dir = Path(args.bestfit_dir)
        run_start_sec = get_run_start_sec_from_folder_name(corrected_dir) if corrected_dir.parent.name != "Corrected" else None

    corrected_dir = Path(corrected_dir)
    bestfit_dir = Path(bestfit_dir)
    if not corrected_dir.exists():
        print(f"Error: Corrected dir not found: {corrected_dir}", file=sys.stderr)
        return 1
    if not bestfit_dir.exists():
        print(f"Error: BestFit dir not found: {bestfit_dir}", file=sys.stderr)
        return 1

    pairs = identify_last_two_cycles_spectra(corrected_dir, bestfit_dir, run_start_sec)
    if not pairs:
        print("No spectra in last two cycles with BestFit found.", file=sys.stderr)
        return 1

    print(f"Last two cycles: {len(pairs)} spectra with BestFit.")
    if args.seed_spectrum is not None:
        stem = Path(args.seed_spectrum).stem
        if not stem.startswith("(Run)"):
            stem = "(Run)spectra_" + stem.replace("(Run)spectra_", "")
        seed_index = next((i for i, (m, _, _) in enumerate(pairs) if m.stem == stem), None)
        if seed_index is None:
            print(f"Error: --seed-spectrum '{args.seed_spectrum}' not in last-two-cycles list.", file=sys.stderr)
            return 1
    else:
        seed_index = min(args.seed_index if args.seed_index is not None else 0, len(pairs) - 1)
    seed_meas, seed_bf, _ = pairs[seed_index]
    rest = [(m, b, t) for i, (m, b, t) in enumerate(pairs) if i != seed_index]
    if args.max_tune is not None:
        rest = rest[: args.max_tune]
        print(f"  (Limited to --max-tune={args.max_tune} spectra.)")

    materials_path = Path(args.materials)
    if not materials_path.exists():
        print(f"Error: Materials path not found: {materials_path}", file=sys.stderr)
        return 1

    grid_search = PyElliGridSearch(materials_path)

    # Wavelength mask
    def fit_wl_meas(path_meas: Path):
        wl, meas = load_measured_spectrum(path_meas)
        mask = (wl >= WL_MIN) & (wl <= WL_MAX)
        return wl[mask], meas[mask]

    # --- Step 1: Full grid search on seed spectrum ---
    print(f"\n[1/2] Full grid search on seed: {seed_meas.name}")
    wl_fit, meas_fit = fit_wl_meas(seed_meas)
    t0 = time.perf_counter()
    seed_results = grid_search.run_grid_search(
        wl_fit,
        meas_fit,
        lipid_range=(9.0, 250.0, 5.0),
        aqueous_range=(800.0, 12000.0, 200.0),
        roughness_range=(ROUGHNESS_MIN_ANGSTROM, ROUGHNESS_MAX_ANGSTROM, 100.0),
        top_k=10,
        enable_roughness=True,
        search_strategy="Dynamic Search",
        max_combinations=30000,
    )
    elapsed_seed = time.perf_counter() - t0
    if not seed_results:
        print("Seed spectrum: no valid fits. Aborting.")
        return 1
    best_seed = seed_results[0]
    seed_result = best_seed  # PyElliResult
    print(f"  Seed result: L={seed_result.lipid_nm:.1f} A={seed_result.aqueous_nm:.1f} R={seed_result.mucus_nm:.0f} (took {elapsed_seed:.1f}s)")

    # Deviation for seed (mucus_nm in PyElliResult is roughness in Å in this codebase)
    bf_wl, bf_refl = load_bestfit_spectrum(seed_bf)
    seed_dev = compute_deviation_for_params(
        grid_search,
        wl_fit,
        meas_fit,
        bf_wl,
        bf_refl,
        seed_result.lipid_nm,
        seed_result.aqueous_nm,
        seed_result.mucus_nm,
    )
    seed_dev_pct = seed_dev["composite_dev"] if seed_dev else float("nan")
    under_10 = 1 if seed_dev and seed_dev["composite_dev"] <= 10.0 else 0
    print(f"  Seed deviation: {seed_dev_pct:.2f}%  (under 10%: {under_10})")

    # --- Step 2: Narrow grid around seed on rest (refine_around_best assumes roughness in nm; we use Å) ---
    print(f"\n[2/2] Tuning (narrow grid around seed) on {len(rest)} spectra (stop when ≥{GOAL_UNDER_10_PCT} under 10%)")
    under_10_list: list[str] = [seed_meas.stem] if (seed_dev and seed_dev["composite_dev"] <= 10.0) else []
    processed = 0
    t_tune_start = time.perf_counter()
    # Narrow ranges in same units as main grid: L nm, A nm, R Å (roughness 6000–7000 Å only for good pyelli fits)
    l_lo = max(9.0, seed_result.lipid_nm - 30)
    l_hi = min(250.0, seed_result.lipid_nm + 30)
    a_lo = max(800.0, seed_result.aqueous_nm - 300)
    a_hi = min(12000.0, seed_result.aqueous_nm + 300)
    r_lo = max(ROUGHNESS_MIN_ANGSTROM, seed_result.mucus_nm - 500)
    r_hi = min(ROUGHNESS_MAX_ANGSTROM, seed_result.mucus_nm + 500)
    for i, (meas_path, bf_path, _) in enumerate(rest):
        if under_10 >= GOAL_UNDER_10_PCT:
            print(f"  Goal reached ({under_10} ≥ {GOAL_UNDER_10_PCT}). Stopping early after {processed} tuned.")
            break
        wl_fit, meas_fit = fit_wl_meas(meas_path)
        try:
            tuned = grid_search.run_grid_search(
                wl_fit,
                meas_fit,
                lipid_range=(l_lo, l_hi, 5.0),
                aqueous_range=(a_lo, a_hi, 50.0),
                roughness_range=(r_lo, r_hi, 100.0),
                top_k=1,
                enable_roughness=True,
                search_strategy="Dynamic Search",
                max_combinations=5000,
            )
        except Exception as e:
            print(f"  {meas_path.name}  tune failed: {e}  under_10%=no")
            processed += 1
            continue
        if not tuned:
            print(f"  {meas_path.name}  no result  under_10%=no")
            processed += 1
            continue
        best = tuned[0]
        bf_wl, bf_refl = load_bestfit_spectrum(bf_path)
        dev = compute_deviation_for_params(
            grid_search,
            wl_fit,
            meas_fit,
            bf_wl,
            bf_refl,
            best.lipid_nm,
            best.aqueous_nm,
            best.mucus_nm,
        )
        processed += 1
        dev_pct = dev["composite_dev"] if dev else float("nan")
        if dev and dev["composite_dev"] <= 10.0:
            under_10 += 1
            under_10_list.append(meas_path.stem)
        status = "yes" if dev and dev["composite_dev"] <= 10.0 else "no"
        print(f"  {meas_path.name}  dev={dev_pct:.2f}%  under_10%={status}")
        if (i + 1) % 20 == 0 or under_10 >= GOAL_UNDER_10_PCT:
            print(f"  --- Processed {processed}, under 10%: {under_10} ---")
    elapsed_tune = time.perf_counter() - t_tune_start

    total_under_10 = under_10
    total_spectra = 1 + processed  # seed + tuned
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Spectra in last two cycles (with BestFit): {len(pairs)}")
    print(f"  Seed spectrum: {seed_meas.name}")
    print(f"  Tuned: {processed} (stopped early: {total_under_10 >= GOAL_UNDER_10_PCT})")
    print(f"  Spectra with deviation ≤10%: {total_under_10}")
    print(f"  Goal (≥{GOAL_UNDER_10_PCT}): {'Reached' if total_under_10 >= GOAL_UNDER_10_PCT else 'Not reached'}")
    print(f"  Seed grid time: {elapsed_seed:.1f}s  |  Tune time: {elapsed_tune:.1f}s  |  Total: {elapsed_seed + elapsed_tune:.1f}s")
    if under_10_list:
        print(f"  Stems with ≤10%: {under_10_list[:10]}{' ...' if len(under_10_list) > 10 else ''}")
    return 0 if total_under_10 >= GOAL_UNDER_10_PCT else 1


if __name__ == "__main__":
    sys.exit(main())
