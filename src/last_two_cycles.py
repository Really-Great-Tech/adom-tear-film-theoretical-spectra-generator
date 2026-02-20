"""
Last-two-cycles identification and LTA full-cycle orchestration.

Shared logic: identify spectra in the last two blink cycles from a run folder.
LTA runner: orchestrates seed (one coarse-fine) + tune (refine-only) via callbacks.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

LAST_TWO_CYCLES_SEC_FALLBACK = 8.0


def parse_timestamp_from_filename(name: str) -> Optional[float]:
    """Parse (Run)spectra_15-13-26-238.txt -> seconds from midnight."""
    m = re.match(r"\(Run\)spectra_(\d+)-(\d+)-(\d+)-(\d+)\.txt", name)
    if not m:
        return None
    h, mm, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return h * 3600 + mm * 60 + s + ms / 1000.0


def get_run_start_sec_from_folder_name(run_dir: Path) -> Optional[float]:
    """Parse run folder name like 'Full test - 0007_2025-12-30_15-12-20' -> 15:12:20 in sec from midnight."""
    name = run_dir.name
    m = re.search(r"_(\d{2})-(\d{2})-(\d{2})$", name)
    if not m:
        return None
    h, mm, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return h * 3600 + mm * 60 + s


def read_blink_times(run_dir: Path) -> List[float]:
    """Read first column of run_dir/Blink.txt."""
    blink_path = Path(run_dir) / "Blink.txt"
    if not blink_path.exists():
        return []
    out: List[float] = []
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
    run_start_sec: Optional[float],
    run_dir: Optional[Path] = None,
) -> Tuple[List[Tuple[Path, Path, float]], Dict[str, Any]]:
    """
    Return list of (measured_path, bestfit_path, time_sec_from_start) for spectra
    in the last two blink cycles that have a BestFit file. Sorted by time.
    Also return info dict with window_plot_s, n_spectra_in_window, etc.
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
    with_time: List[Tuple[Path, float]] = []
    for f in files:
        ts = parse_timestamp_from_filename(f.name)
        if ts is not None:
            with_time.append((f, ts))

    if not with_time:
        return [], {}

    ref = run_start_sec if run_start_sec is not None else min(t for _, t in with_time)
    with_sec = [(p, t - ref) for p, t in with_time]
    t_min = min(s for _, s in with_sec)
    t_max = max(s for _, s in with_sec)

    blink_times = read_blink_times(run_dir) if run_dir else []
    if len(blink_times) >= 2:
        blink_sorted = sorted(blink_times)
        start_plot = blink_sorted[-2]
        t_cut = t_min + start_plot
    else:
        t_cut = t_max - LAST_TWO_CYCLES_SEC_FALLBACK

    last_two = [(p, s) for p, s in with_sec if s >= t_cut]
    last_two.sort(key=lambda x: x[1])

    out: List[Tuple[Path, Path, float]] = []
    for meas_path, time_sec in last_two:
        stem = meas_path.stem
        bf_path = bestfit_dir / f"{stem}_BestFit.txt"
        if bf_path.exists():
            out.append((meas_path, bf_path, time_sec))

    plot_start = (t_cut - t_min) if out else None
    plot_end = (t_max - t_min) if out else None
    info = {
        "t_min_run_s": t_min,
        "t_max_run_s": t_max,
        "t_cut_run_s": t_cut,
        "window_run_s": (t_cut, t_max) if out else (None, None),
        "window_plot_s": (plot_start, plot_end) if plot_start is not None else None,
        "used_blink_txt": len(blink_times) >= 2,
        "n_spectra_in_window": len(out),
    }
    return out, info


def load_lta_height_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load LTA height file (time, value columns). Return (times, values) arrays."""
    if not path.exists():
        return np.array([]), np.array([])
    times, values = [], []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                times.append(float(parts[0]))
                values.append(float(parts[1]))
            except ValueError:
                pass
    return np.array(times), np.array(values)


def run_lta_last_two_cycles_sync(
    run_dir: Path,
    *,
    get_seed_result: Callable[[Path], Optional[Dict[str, Any]]],
    get_tune_result: Callable[[Path, float, float, float], Optional[Dict[str, Any]]],
    get_snr: Callable[[Path], Tuple[float, bool]],
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    seed_index: int = 0,
    snr_threshold: float = 20.0,
) -> Dict[str, Any]:
    """
    Run LTA seed + tune on last-two-cycles spectra. Uses callbacks for actual fitting.

    get_seed_result(meas_path) -> {lipid_nm, aqueous_nm, roughness_A, score_composite, deviation_pct?} or None
    get_tune_result(meas_path, L, A, R) -> same
    get_snr(meas_path) -> (snr_value, noisy: bool)

    Returns dict: summary, seed_result, results (list of per-spectrum dicts with stem, time_sec, lipid_nm, ...).
    """
    run_dir = Path(run_dir)
    corrected_dir = run_dir / "Spectra" / "GoodSpectra" / "Corrected"
    bestfit_dir = run_dir / "BestFit"
    if not corrected_dir.exists() or not bestfit_dir.exists():
        return {
            "summary": {"error": "Corrected or BestFit dir not found.", "total_spectra": 0},
            "seed_result": None,
            "results": [],
        }

    pairs, window_info = identify_last_two_cycles_spectra(
        corrected_dir, bestfit_dir,
        get_run_start_sec_from_folder_name(run_dir),
        run_dir=run_dir,
    )
    if not pairs:
        return {
            "summary": {"error": "No spectra in last two cycles with BestFit.", "total_spectra": 0},
            "seed_result": None,
            "results": [],
        }

    total = len(pairs)
    seed_index = min(seed_index, total - 1)
    seed_meas, seed_bf, seed_time = pairs[seed_index]
    rest = [(m, b, t) for i, (m, b, t) in enumerate(pairs) if i != seed_index]

    if progress_callback:
        progress_callback(0, total, "seed")

    t0 = time.perf_counter()
    seed_row = get_seed_result(seed_meas)
    elapsed_seed = time.perf_counter() - t0

    if seed_row is None:
        return {
            "summary": {"error": "Seed spectrum: no valid fit.", "total_spectra": total},
            "seed_result": None,
            "results": [],
        }

    seed_snr, seed_noisy = get_snr(seed_meas)
    seed_result = {
        "stem": seed_meas.stem,
        "lipid_nm": float(seed_row.get("lipid_nm", 0)),
        "aqueous_nm": float(seed_row.get("aqueous_nm", 0)),
        "roughness_A": float(seed_row.get("roughness_A", 0)),
        "deviation_pct": seed_row.get("deviation_pct"),
        "score_composite": seed_row.get("score_composite"),
        "snr_value": seed_snr,
        "noisy": seed_noisy,
        "time_sec": seed_time,
    }

    results_list: List[Dict[str, Any]] = []
    center_L = seed_result["lipid_nm"]
    center_A = seed_result["aqueous_nm"]
    center_R = seed_result["roughness_A"]

    for idx, (meas_path, bf_path, time_sec) in enumerate(rest):
        if progress_callback:
            progress_callback(1 + idx, total, "tune")
        row = get_tune_result(meas_path, center_L, center_A, center_R)
        if row is None:
            continue
        snr_val, noisy = get_snr(meas_path)
        results_list.append({
            "stem": meas_path.stem,
            "lipid_nm": float(row.get("lipid_nm", 0)),
            "aqueous_nm": float(row.get("aqueous_nm", 0)),
            "roughness_A": float(row.get("roughness_A", 0)),
            "deviation_pct": row.get("deviation_pct"),
            "score_composite": row.get("score_composite"),
            "snr_value": snr_val,
            "noisy": noisy,
            "time_sec": time_sec,
        })
        center_L = results_list[-1]["lipid_nm"]
        center_A = results_list[-1]["aqueous_nm"]
        center_R = results_list[-1]["roughness_A"]

    elapsed_total = time.perf_counter() - t0
    n_noisy = sum(1 for r in results_list if r.get("noisy", False)) + (1 if seed_noisy else 0)

    summary = {
        "total_spectra": total,
        "processed": 1 + len(results_list),
        "seed_stem": seed_meas.stem,
        "elapsed_seed_sec": round(elapsed_seed, 2),
        "elapsed_total_sec": round(elapsed_total, 2),
        "total_noisy": n_noisy,
        "snr_noisy_threshold": snr_threshold,
        "window_plot_s": window_info.get("window_plot_s"),
        "used_blink_txt": window_info.get("used_blink_txt", False),
    }

    return {
        "summary": summary,
        "seed_result": seed_result,
        "results": results_list,
    }
