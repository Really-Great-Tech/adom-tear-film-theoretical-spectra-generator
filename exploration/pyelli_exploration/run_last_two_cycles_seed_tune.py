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
for run start time.

Last-two-blinks window: if run_dir/Blink.txt exists, its first column gives blink times
(in the same time base as Aqueous_Height / AllLayersGraph, i.e. time from first spectrum).
We take spectra from the second-to-last blink time to end (~8 s for last two cycles).
If Blink.txt is absent, we use the last 8 s of acquisition (matches AllLayersGraph ~32–40 s).
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

# Fallback when Blink.txt is absent: last 8 s (matches AllLayersGraph last two blinks ~32–40 s)
LAST_TWO_CYCLES_SEC_FALLBACK = 8.0
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


def read_blink_times(run_dir: Path) -> list[float]:
    """Read first column of run_dir/Blink.txt (blink times in same base as Aqueous_Height / AllLayersGraph)."""
    blink_path = Path(run_dir) / "Blink.txt"
    if not blink_path.exists():
        return []
    out: list[float] = []
    for line in blink_path.read_text().splitlines():
        parts = line.strip().split()
        if parts:
            try:
                out.append(float(parts[0]))
            except ValueError:
                pass
    return out


def identify_last_two_cycles_spectra(
    corrected_dir: Path,
    bestfit_dir: Path,
    run_start_sec: float | None,
    run_dir: Path | None = None,
) -> list[tuple[Path, Path, float]]:
    """
    Return list of (measured_path, bestfit_path, time_sec_from_start) for spectra
    in the last two blink cycles that have a BestFit file. Sorted by time.

    Window: if run_dir/Blink.txt exists, use from second-to-last blink time to end
    (plot time base = time from first spectrum, so t_cut_run = t_min + blink_start).
    Otherwise use last LAST_TWO_CYCLES_SEC_FALLBACK (8) s.
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
    t_min = min(s for _, s in with_sec)
    t_max = max(s for _, s in with_sec)

    # Last-two-blinks window: prefer Blink.txt (same time base as AllLayersGraph)
    blink_times = read_blink_times(run_dir) if run_dir else []
    if len(blink_times) >= 2:
        # Blink.txt times are "time from first spectrum" (plot time). Map to run time: t_run = t_min + t_plot.
        blink_sorted = sorted(blink_times)
        start_plot = blink_sorted[-2]  # second-to-last blink = start of last two cycles
        t_cut = t_min + start_plot
    else:
        t_cut = t_max - LAST_TWO_CYCLES_SEC_FALLBACK

    last_two = [(p, s) for p, s in with_sec if s >= t_cut]
    last_two.sort(key=lambda x: x[1])

    # Filter to those that have BestFit
    out: list[tuple[Path, Path, float]] = []
    for meas_path, time_sec in last_two:
        stem = meas_path.stem
        bf_path = bestfit_dir / f"{stem}_BestFit.txt"
        if bf_path.exists():
            out.append((meas_path, bf_path, time_sec))

    # Verification info (plot time base = time from first spectrum = run_time - t_min)
    window_run_s = (t_cut, t_max) if out else (None, None)
    plot_start = (t_cut - t_min) if out else None
    plot_end = (t_max - t_min) if out else None
    info = {
        "t_min_run_s": t_min,
        "t_max_run_s": t_max,
        "t_cut_run_s": t_cut,
        "window_run_s": window_run_s,
        "window_plot_s": (plot_start, plot_end) if plot_start is not None else None,
        "used_blink_txt": len(blink_times) >= 2,
        "n_spectra_in_window": len(out),
    }
    return out, info


def run_last_two_cycles_seed_tune_sync(
    run_dir: Path,
    materials_path: Path,
    *,
    corrected_dir: Path | None = None,
    bestfit_dir: Path | None = None,
    seed_index: int = 0,
    max_tune: int | None = None,
) -> dict:
    """
    Run seed+tune on last-two-cycles spectra. Returns a dict suitable for JSON/API.

    Keys: summary (dict with total_spectra, total_under_10, goal_reached, seed_stem, elapsed_seed_sec, elapsed_tune_sec),
          seed_result (dict with lipid_nm, aqueous_nm, mucus_nm, deviation_pct),
          results (list of dicts with stem, lipid_nm, aqueous_nm, mucus_nm, deviation_pct, under_10).
    """
    corrected_dir = corrected_dir or (run_dir / "Spectra" / "GoodSpectra" / "Corrected")
    bestfit_dir = bestfit_dir or (run_dir / "BestFit")
    run_start_sec = get_run_start_sec_from_folder_name(run_dir) if run_dir.name != "Corrected" else None
    if not corrected_dir.exists() or not bestfit_dir.exists():
        raise FileNotFoundError(f"Corrected or BestFit dir not found under {run_dir}")
    pairs, window_info = identify_last_two_cycles_spectra(corrected_dir, bestfit_dir, run_start_sec, run_dir=run_dir)
    if not pairs:
        return {
            "summary": {"error": "No spectra in last two cycles with BestFit found.", "total_spectra": 0},
            "seed_result": None,
            "results": [],
        }
    seed_index = min(seed_index, len(pairs) - 1)
    seed_meas, seed_bf, _ = pairs[seed_index]
    rest = [(m, b, t) for i, (m, b, t) in enumerate(pairs) if i != seed_index]
    if max_tune is not None:
        rest = rest[:max_tune]
    materials_path = Path(materials_path)
    if not materials_path.exists():
        raise FileNotFoundError(f"Materials path not found: {materials_path}")
    grid_search = PyElliGridSearch(materials_path)

    def fit_wl_meas(path_meas: Path):
        wl, meas = load_measured_spectrum(path_meas)
        mask = (wl >= WL_MIN) & (wl <= WL_MAX)
        return wl[mask], meas[mask]

    # Step 1: Full grid on seed
    t0 = time.perf_counter()
    wl_fit, meas_fit = fit_wl_meas(seed_meas)
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
        return {
            "summary": {"error": "Seed spectrum: no valid fits.", "total_spectra": len(pairs)},
            "seed_result": None,
            "results": [],
        }
    seed_result = seed_results[0]
    bf_wl, bf_refl = load_bestfit_spectrum(seed_bf)
    seed_dev = compute_deviation_for_params(
        grid_search, wl_fit, meas_fit, bf_wl, bf_refl,
        seed_result.lipid_nm, seed_result.aqueous_nm, seed_result.mucus_nm,
    )
    seed_dev_pct = seed_dev["composite_dev"] if seed_dev else float("nan")
    under_10_count = 1 if seed_dev and seed_dev["composite_dev"] <= 10.0 else 0
    under_10_list = [seed_meas.stem] if (seed_dev and seed_dev["composite_dev"] <= 10.0) else []

    # Step 2: Tune on rest
    l_lo = max(9.0, seed_result.lipid_nm - 30)
    l_hi = min(250.0, seed_result.lipid_nm + 30)
    a_lo = max(800.0, seed_result.aqueous_nm - 300)
    a_hi = min(12000.0, seed_result.aqueous_nm + 300)
    r_lo = max(ROUGHNESS_MIN_ANGSTROM, seed_result.mucus_nm - 500)
    r_hi = min(ROUGHNESS_MAX_ANGSTROM, seed_result.mucus_nm + 500)
    results_list = []
    t_tune_start = time.perf_counter()
    processed = 0
    for meas_path, bf_path, _ in rest:
        if under_10_count >= GOAL_UNDER_10_PCT:
            break
        wl_fit, meas_fit = fit_wl_meas(meas_path)
        try:
            tuned = grid_search.run_grid_search(
                wl_fit, meas_fit,
                lipid_range=(l_lo, l_hi, 5.0),
                aqueous_range=(a_lo, a_hi, 50.0),
                roughness_range=(r_lo, r_hi, 100.0),
                top_k=1,
                enable_roughness=True,
                search_strategy="Dynamic Search",
                max_combinations=5000,
            )
        except Exception:
            results_list.append({"stem": meas_path.stem, "deviation_pct": None, "under_10": False})
            processed += 1
            continue
        if not tuned:
            results_list.append({"stem": meas_path.stem, "deviation_pct": None, "under_10": False})
            processed += 1
            continue
        best = tuned[0]
        bf_wl, bf_refl = load_bestfit_spectrum(bf_path)
        dev = compute_deviation_for_params(
            grid_search, wl_fit, meas_fit, bf_wl, bf_refl,
            best.lipid_nm, best.aqueous_nm, best.mucus_nm,
        )
        dev_pct = dev["composite_dev"] if dev else float("nan")
        u10 = bool(dev and dev["composite_dev"] <= 10.0)
        if u10:
            under_10_count += 1
            under_10_list.append(meas_path.stem)
        results_list.append({
            "stem": meas_path.stem,
            "lipid_nm": float(best.lipid_nm),
            "aqueous_nm": float(best.aqueous_nm),
            "mucus_nm": float(best.mucus_nm),
            "deviation_pct": float(dev_pct) if dev else None,
            "under_10": u10,
        })
        processed += 1
    elapsed_tune = time.perf_counter() - t_tune_start

    summary = {
        "total_spectra": len(pairs),
        "seed_stem": seed_meas.stem,
        "total_under_10": under_10_count,
        "goal_under_10": GOAL_UNDER_10_PCT,
        "goal_reached": under_10_count >= GOAL_UNDER_10_PCT,
        "elapsed_seed_sec": round(elapsed_seed, 2),
        "elapsed_tune_sec": round(elapsed_tune, 2),
        "elapsed_total_sec": round(elapsed_seed + elapsed_tune, 2),
        "processed": 1 + processed,
        "stems_under_10": under_10_list[:20],
        "window_plot_s": window_info.get("window_plot_s"),
        "window_run_s": window_info.get("window_run_s"),
        "used_blink_txt": window_info.get("used_blink_txt", False),
    }
    seed_result_dict = {
        "lipid_nm": float(seed_result.lipid_nm),
        "aqueous_nm": float(seed_result.aqueous_nm),
        "mucus_nm": float(seed_result.mucus_nm),
        "deviation_pct": float(seed_dev_pct) if seed_dev else None,
        "under_10": seed_dev is not None and seed_dev["composite_dev"] <= 10.0,
    }
    return {"summary": summary, "seed_result": seed_result_dict, "results": results_list}


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
        corrected_dir = bestfit_dir = None
    else:
        if args.corrected_dir is None or args.bestfit_dir is None:
            parser.error("Either --run-dir or both --corrected-dir and --bestfit-dir are required.")
        corrected_dir = Path(args.corrected_dir)
        bestfit_dir = Path(args.bestfit_dir)
        if not corrected_dir.exists() or not bestfit_dir.exists():
            print(f"Error: Corrected or BestFit dir not found.", file=sys.stderr)
            return 1
        run_dir = corrected_dir.parent.parent.parent  # Corrected -> GoodSpectra -> Spectra -> run_dir
        if not (run_dir / "BestFit").exists():
            run_dir = corrected_dir.parent

    if not run_dir.exists():
        print(f"Error: Run dir not found: {run_dir}", file=sys.stderr)
        return 1
    materials_path = Path(args.materials)
    if not materials_path.exists():
        print(f"Error: Materials path not found: {materials_path}", file=sys.stderr)
        return 1

    seed_index = 0
    if args.seed_spectrum is not None:
        pairs_pre, _ = identify_last_two_cycles_spectra(
            run_dir / "Spectra" / "GoodSpectra" / "Corrected",
            run_dir / "BestFit",
            get_run_start_sec_from_folder_name(run_dir),
            run_dir=run_dir,
        )
        stem = Path(args.seed_spectrum).stem
        if not stem.startswith("(Run)"):
            stem = "(Run)spectra_" + stem.replace("(Run)spectra_", "")
        idx = next((i for i, (m, _, _) in enumerate(pairs_pre) if m.stem == stem), None)
        if idx is None:
            print(f"Error: --seed-spectrum '{args.seed_spectrum}' not in last-two-cycles list.", file=sys.stderr)
            return 1
        seed_index = idx
    elif args.seed_index is not None:
        seed_index = args.seed_index

    try:
        out = run_last_two_cycles_seed_tune_sync(
            run_dir,
            materials_path,
            corrected_dir=corrected_dir if args.run_dir is None else None,
            bestfit_dir=bestfit_dir if args.run_dir is None else None,
            seed_index=seed_index,
            max_tune=args.max_tune,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    summary = out.get("summary", {})
    if summary.get("error"):
        print(summary["error"], file=sys.stderr)
        return 1
    print(f"\nLast two cycles: {summary.get('total_spectra', 0)} spectra with BestFit.")
    wp = summary.get("window_plot_s")
    if wp:
        print(f"Window (plot time, like AllLayersGraph): {wp[0]:.2f} s to {wp[1]:.2f} s (~{wp[1]-wp[0]:.1f} s). Used Blink.txt: {summary.get('used_blink_txt')}")
    print(f"Seed: {summary.get('seed_stem', '')}  L={out.get('seed_result', {}).get('lipid_nm')} A={out.get('seed_result', {}).get('aqueous_nm')} R={out.get('seed_result', {}).get('mucus_nm')}")
    print(f"Under 10%: {summary.get('total_under_10')}  Goal reached: {summary.get('goal_reached')}")
    print(f"Elapsed: seed={summary.get('elapsed_seed_sec')}s tune={summary.get('elapsed_tune_sec')}s total={summary.get('elapsed_total_sec')}s")
    return 0 if summary.get("goal_reached") else 1


if __name__ == "__main__":
    sys.exit(main())
