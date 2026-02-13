#!/usr/bin/env python3
"""
Weight optimization script for PyElli grid search.

Phase 1: Run grid search on N spectra from more_good_spectras, saving all candidates
         (params + 5 component scores) to CSV per spectrum.
Phase 2: Try many weight sets; for each set re-rank candidates per spectrum, pick best,
         compute deviation vs LTA BestFit; report best weights and summary.

Usage:
  python -m exploration.pyelli_exploration.optimize_score_weights --spectra-dir ... --cache-dir ... --out ...
  Or use defaults: 50 spectra from more_good_spectras, cache under outputs/weight_optimization.

Goal: Find weights (rmse, amplitude, correlation, peak_delta, peak_count) that achieve
      <=10% composite deviation on 50+ spectra.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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

WL_MIN, WL_MAX = 600.0, 1120.0
DEFAULT_WEIGHTS = {
    "rmse_score": 0.25,
    "amplitude_score": 0.20,
    "correlation_score": 0.15,
    "peak_delta_score": 0.15,
    "peak_count_score": 0.25,
}
COMPONENT_KEYS = list(DEFAULT_WEIGHTS.keys())


def compute_composite_score(row: pd.Series, weights: dict[str, float]) -> float:
    """Weighted sum of component scores; weights should sum to 1."""
    return sum(float(row[k]) * weights[k] for k in COMPONENT_KEYS)


def compute_deviation_for_params(
    grid_search: PyElliGridSearch,
    wl_fit: np.ndarray,
    meas_fit: np.ndarray,
    bestfit_wl: np.ndarray,
    bestfit_refl: np.ndarray,
    lipid_nm: float,
    aqueous_nm: float,
    roughness_angstrom: float,
) -> dict | None:
    """
    Compute composite deviation (vs LTA BestFit) for a single (L, A, R) fit.
    Does not use PyElliResult; computes theoretical and score on the fly.
    """
    try:
        # PyElli theoretical for (L, A, R); roughness in nm for API
        theoretical = grid_search.calculate_theoretical_spectrum(
            wl_fit, lipid_nm, aqueous_nm, roughness_angstrom / 10.0, enable_roughness=True
        )
        pyelli_aligned = grid_search._align_spectra(
            meas_fit, theoretical,
            focus_min=float(wl_fit[0]), focus_max=float(wl_fit[-1]), wavelengths=wl_fit,
        )
        bestfit_interp = np.interp(wl_fit, bestfit_wl, bestfit_refl)
        bestfit_aligned = grid_search._align_spectra(
            meas_fit, bestfit_interp,
            focus_min=float(wl_fit[0]), focus_max=float(wl_fit[-1]), wavelengths=wl_fit,
        )
        bestfit_score_result = calculate_peak_based_score(wl_fit, meas_fit, bestfit_aligned)
        pyelli_score_result = calculate_peak_based_score(wl_fit, meas_fit, pyelli_aligned)

        lta_abs = np.abs(bestfit_aligned)
        valid_mask = lta_abs > 1e-10
        if valid_mask.any():
            mape = float(np.mean(
                np.abs(pyelli_aligned[valid_mask] - bestfit_aligned[valid_mask]) / lta_abs[valid_mask]
            )) * 100
        else:
            mape = 0.0

        pyelli_matched = float(pyelli_score_result.get("matched_peaks", 0))
        lta_peaks = float(bestfit_score_result.get("measurement_peaks", 0))
        if lta_peaks > 0:
            peak_match_deviation = (1.0 - (pyelli_matched / lta_peaks)) * 100
            peak_match_deviation = max(0.0, peak_match_deviation)
        else:
            peak_match_deviation = 0.0

        pyelli_mean_delta = float(pyelli_score_result.get("mean_delta_nm", 0.0))
        reference_spacing = 50.0
        alignment_deviation = (pyelli_mean_delta / reference_spacing) * 100

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
        return None


def run_phase1(
    spectra_dir: Path,
    bestfit_dir: Path,
    cache_dir: Path,
    materials_path: Path,
    max_spectra: int,
    max_combinations: int,
) -> list[str]:
    """Run grid search on up to max_spectra spectra that have a corresponding BestFit; save candidates CSV per spectrum. Returns list of spectrum stems that have cache."""
    spectra_dir = Path(spectra_dir)
    bestfit_dir = Path(bestfit_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build (measured_path, bestfit_path) pairs where both exist; take up to max_spectra
    pairs: list[tuple[Path, Path]] = []
    for meas_path in sorted(spectra_dir.glob("(Run)spectra_*.txt")):
        stem = meas_path.stem
        bestfit_path = bestfit_dir / f"{stem}_BestFit.txt"
        if bestfit_path.exists():
            pairs.append((meas_path, bestfit_path))
    pairs = pairs[:max_spectra]
    if not pairs:
        raise FileNotFoundError(
            f"No (Run)spectra_*.txt in {spectra_dir} with corresponding *_BestFit.txt in {bestfit_dir}"
        )

    print(f"  Phase 1: saving candidates with NO filter (all_candidates_path set); cache_dir={cache_dir}")
    print(f"  Running grid search on {len(pairs)} spectra (each with a BestFit). Candidate CSV per spectrum.")
    grid_search = PyElliGridSearch(materials_path)
    saved: list[str] = []
    for i, (meas_path, _) in enumerate(pairs):
        stem = meas_path.stem
        cache_path = cache_dir / f"{stem}_candidates.csv"
        print(f"  [{i+1}/{len(pairs)}] {meas_path.name} -> {cache_path.name} ...")
        wl, meas = load_measured_spectrum(meas_path)
        wl_mask = (wl >= WL_MIN) & (wl <= WL_MAX)
        wl_fit, meas_fit = wl[wl_mask], meas[wl_mask]
        try:
            # No filter when saving candidates (all_candidates_path set); Phase 2 re-ranks by weights
            grid_search.run_grid_search(
                wl_fit, meas_fit,
                lipid_range=(9.0, 250.0, 5.0),
                aqueous_range=(800.0, 12000.0, 200.0),
                roughness_range=(6000.0, 7000.0, 100.0),
                top_k=10,
                enable_roughness=True,
                search_strategy="Dynamic Search",
                max_combinations=max_combinations,
                all_candidates_path=cache_path,
            )
            if cache_path.exists():
                n_lines = len(cache_path.read_text().strip().splitlines()) - 1  # exclude header
                print(f"      -> OK: wrote {cache_path.name} ({n_lines} candidate rows)")
                saved.append(stem)
            else:
                print(f"      -> WARNING: {cache_path.name} was not created (0 candidates)")
        except Exception as e:
            print(f"      -> Error: {e}")
    print(f"  Phase 1 done: {len(saved)}/{len(pairs)} candidate CSVs saved to {cache_dir}")
    return saved


def run_phase2(
    cache_dir: Path,
    spectra_dir: Path,
    bestfit_dir: Path,
    materials_path: Path,
    stems: list[str],
    weight_grid: list[dict[str, float]],
) -> pd.DataFrame:
    """
    For each weight set, for each spectrum pick best candidate by weighted score,
    compute deviation; return DataFrame of (weight_id, mean_dev, pct_under_10, ...).
    """
    cache_dir = Path(cache_dir)
    spectra_dir = Path(spectra_dir)
    bestfit_dir = Path(bestfit_dir)
    grid_search = PyElliGridSearch(materials_path)

    rows = []
    for wid, weights in enumerate(weight_grid):
        devs = []
        for stem in stems:
            csv_path = cache_dir / f"{stem}_candidates.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            df["_composite"] = df.apply(lambda r: compute_composite_score(r, weights), axis=1)
            best_idx = df["_composite"].idxmax()
            best = df.loc[best_idx]
            L = float(best["lipid_nm"])
            A = float(best["aqueous_nm"])
            R = float(best["roughness_angstrom"])

            meas_path = spectra_dir / f"{stem}.txt"
            bf_path = bestfit_dir / f"{stem}_BestFit.txt"
            if not meas_path.exists() or not bf_path.exists():
                continue
            wl, meas = load_measured_spectrum(meas_path)
            bf_wl, bf_refl = load_bestfit_spectrum(bf_path)
            wl_mask = (wl >= WL_MIN) & (wl <= WL_MAX)
            wl_fit = wl[wl_mask]
            meas_fit = meas[wl_mask]

            dev = compute_deviation_for_params(
                grid_search, wl_fit, meas_fit, bf_wl, bf_refl, L, A, R,
            )
            if dev is not None:
                devs.append(dev["composite_dev"])
        if not devs:
            rows.append({"weight_id": wid, "mean_dev": np.nan, "pct_under_10": np.nan, "n_spectra": 0})
            continue
        devs = np.array(devs)
        n_under = (devs <= 10.0).sum()
        rows.append({
            "weight_id": wid,
            "mean_dev": float(np.mean(devs)),
            "pct_under_10": 100.0 * n_under / len(devs),
            "n_spectra": len(devs),
            "min_dev": float(np.min(devs)),
            "max_dev": float(np.max(devs)),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Optimize grid search score weights for ≤10% deviation on 50 spectra")
    parser.add_argument("--spectra-dir", type=Path,
                        default=PROJECT_ROOT / "exploration" / "more_good_spectras" / "Corrected_Spectra",
                        help="Directory with (Run)spectra_*.txt")
    parser.add_argument("--bestfit-dir", type=Path,
                        default=PROJECT_ROOT / "exploration" / "more_good_spectras" / "BestFit",
                        help="Directory with *_BestFit.txt")
    parser.add_argument("--cache-dir", type=Path,
                        default=PROJECT_ROOT / "outputs" / "weight_optimization" / "candidates",
                        help="Where to save/load per-spectrum candidate CSVs")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "outputs" / "weight_optimization" / "weight_results.csv",
                        help="Output CSV of weight sets and metrics")
    parser.add_argument("--max-spectra", type=int, default=50, help="Max spectra to use (default 50)")
    parser.add_argument("--max-combinations", type=int, default=30000, help="Max grid combinations per spectrum")
    parser.add_argument("--phase", choices=["1", "2", "both"], default="both",
                        help="Run phase 1 (grid search + cache), phase 2 (weight search), or both")
    parser.add_argument("--n-weights", type=int, default=100,
                        help="Number of random weight sets to try in phase 2 (default 100)")
    args = parser.parse_args()

    materials_path = PROJECT_ROOT / "data" / "materials"
    if not materials_path.exists():
        materials_path = PROJECT_ROOT / "configs"

    stems_phase1 = []
    if args.phase in ("1", "both"):
        print("Phase 1: Grid search and save candidates...")
        t0 = time.perf_counter()
        stems_phase1 = run_phase1(
            args.spectra_dir, args.bestfit_dir, args.cache_dir,
            materials_path, args.max_spectra, args.max_combinations,
        )
        print(f"  Saved {len(stems_phase1)} candidate CSVs in {time.perf_counter() - t0:.1f}s")
        if args.phase == "1":
            return

    # Phase 2: use stems from phase 1 if we just ran it, else load from cache dir
    if stems_phase1:
        stems = stems_phase1
    else:
        stems = [p.stem.replace("_candidates", "") for p in Path(args.cache_dir).glob("*_candidates.csv")]
        stems = sorted(set(stems))[: args.max_spectra]
    if not stems:
        print("No candidate CSVs found; run phase 1 first.")
        return

    # Build weight grid: default + random variants (normalized to sum=1)
    rng = np.random.default_rng(42)
    weight_grid = [DEFAULT_WEIGHTS.copy()]
    for _ in range(args.n_weights - 1):
        w = rng.dirichlet(np.ones(5))
        weight_grid.append(dict(zip(COMPONENT_KEYS, w.tolist())))

    print(f"Phase 2: Trying {len(weight_grid)} weight sets on {len(stems)} spectra...")
    t0 = time.perf_counter()
    results_df = run_phase2(
        args.cache_dir, args.spectra_dir, args.bestfit_dir,
        materials_path, stems, weight_grid,
    )
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    results_df = results_df.sort_values("mean_dev")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.out, index=False)
    print(f"  Wrote {args.out}")

    best = results_df.iloc[0]
    print("")
    print("Best weight set (by mean deviation):")
    print(f"  weight_id={int(best['weight_id'])}  mean_dev={best['mean_dev']:.2f}%  pct_under_10={best['pct_under_10']:.1f}%  n_spectra={int(best['n_spectra'])}")
    best_weights = weight_grid[int(best["weight_id"])]
    print("  Weights:", best_weights)
    print("")
    print("Conclusion: Use these weights in grid search to target ≤10% deviation on 50+ spectra.")
    if best["pct_under_10"] >= 100.0 and best["mean_dev"] <= 10.0:
        print("  This set achieves the goal (all spectra ≤10% deviation).")
    else:
        print("  Consider increasing --n-weights or tuning ranges for further improvement.")

    summary_path = args.out.with_suffix(".best_weights.json")
    with open(summary_path, "w") as f:
        json.dump({"weights": best_weights, "mean_dev": float(best["mean_dev"]), "pct_under_10": float(best["pct_under_10"]), "n_spectra": int(best["n_spectra"])}, f, indent=2)
    print(f"  Best weights saved to {summary_path}")


if __name__ == "__main__":
    main()
