"""Command-line grid search scorer for tear-film spectra."""

from __future__ import annotations

import argparse
import copy
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd

from analysis import (
    load_measurement_spectrum,
    prepare_measurement,
    prepare_theoretical_spectrum,
    peak_count_score,
    peak_delta_score,
    phase_overlap_score,
    composite_score,
)
from tear_film_generator import (
    load_config,
    validate_config,
    make_single_spectrum_calculator,
    get_project_path,
)


@dataclass
class CandidateResult:
    lipid_nm: float
    aqueous_nm: float
    roughness_A: float
    scores: Dict[str, float]
    diagnostics: Dict[str, Dict[str, float]]

    @property
    def composite(self) -> float:
        return self.scores["composite"]

    def as_dict(self) -> Dict[str, float]:
        payload: Dict[str, float] = {
            "lipid_nm": self.lipid_nm,
            "aqueous_nm": self.aqueous_nm,
            "roughness_A": self.roughness_A,
        }
        payload.update({f"score_{key}": value for key, value in self.scores.items()})
        for metric, diag in self.diagnostics.items():
            for diag_key, diag_val in diag.items():
                payload[f"{metric}_{diag_key}"] = diag_val
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score theoretical spectra against a measurement")
    parser.add_argument("measurement", type=Path, help="Path to measurement spectrum file")
    parser.add_argument(
        "--config", type=Path, default=None, help="Configuration YAML (defaults to config.yaml)"
    )
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=None,
        help="Directory containing grid.npy and meta.json produced by run_tear_film_generator",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of best candidates to persist in best_fit.json (default: 25)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write results (defaults to outputs/grid_search)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Optional cap on evaluated spectra (useful for dry runs)",
    )
    parser.add_argument(
        "--peak-count-tol",
        type=float,
        default=None,
        help="Override peak-count wavelength tolerance in nanometers",
    )
    parser.add_argument(
        "--peak-delta-tol",
        type=float,
        default=None,
        help="Override paired-peak tolerance window in nanometers",
    )
    parser.add_argument(
        "--peak-delta-tau",
        type=float,
        default=None,
        help="Override exponential decay factor (tau) for paired-peak delta metric",
    )
    parser.add_argument(
        "--peak-delta-penalty",
        type=float,
        default=None,
        help="Override penalty applied per unmatched peak in the delta metric",
    )
    parser.add_argument(
        "--metric-weights",
        type=float,
        nargs=3,
        metavar=("W_PEAK_COUNT", "W_PEAK_DELTA", "W_PHASE"),
        help="Override composite metric weights (peak_count, peak_delta, phase_overlap)",
    )
    return parser.parse_args()


def load_cached_grid(cache_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid_path = cache_dir / "grid.npy"
    meta_path = cache_dir / "meta.json"
    if not grid_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Grid cache missing required files grid.npy/meta.json in {cache_dir}"
        )
    grid = np.load(grid_path, allow_pickle=False)
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    lipid_vals = np.asarray(meta["lipid_nm"], dtype=float)
    aqueous_vals = np.asarray(meta["aqueous_nm"], dtype=float)
    rough_vals = np.asarray(meta["rough_A"], dtype=float)
    wavelengths = np.asarray(meta["wavelengths_nm"], dtype=float)
    return grid, lipid_vals, aqueous_vals, rough_vals, wavelengths


def iter_cached_spectra(
    grid: np.ndarray,
    lipid_vals: np.ndarray,
    aqueous_vals: np.ndarray,
    rough_vals: np.ndarray,
) -> Iterator[Tuple[Tuple[float, float, float], np.ndarray]]:
    for i, lipid in enumerate(lipid_vals):
        for j, aqueous in enumerate(aqueous_vals):
            for k, rough in enumerate(rough_vals):
                yield (float(lipid), float(aqueous), float(rough)), grid[i, j, k, :]


def iter_generated_spectra(
    calculator,
    lipid_vals: np.ndarray,
    aqueous_vals: np.ndarray,
    rough_vals: np.ndarray,
) -> Iterator[Tuple[Tuple[float, float, float], np.ndarray]]:
    for lipid in lipid_vals:
        for aqueous in aqueous_vals:
            for rough in rough_vals:
                spectrum = calculator(float(lipid), float(aqueous), float(rough))
                yield (float(lipid), float(aqueous), float(rough)), spectrum


def prepare_parameter_arrays(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = config["parameters"]
    lipid_cfg = params["lipid"]
    aqueous_cfg = params["aqueous"]
    rough_cfg = params["roughness"]

    lipid_vals = np.arange(lipid_cfg["min"], lipid_cfg["max"], lipid_cfg["step"], dtype=float)
    aqueous_vals = np.arange(aqueous_cfg["min"], aqueous_cfg["max"], aqueous_cfg["step"], dtype=float)
    rough_vals = np.arange(rough_cfg["min"], rough_cfg["max"], rough_cfg["step"], dtype=float)
    return lipid_vals, aqueous_vals, rough_vals


def evaluate_candidate(
    params: Tuple[float, float, float],
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    measurement_features,
    analysis_cfg: Dict[str, Any],
    metrics_cfg: Dict[str, Any],
) -> CandidateResult:
    theoretical = prepare_theoretical_spectrum(wavelengths, spectrum, measurement_features, analysis_cfg)

    peak_count_cfg = metrics_cfg.get("peak_count", {})
    peak_delta_cfg = metrics_cfg.get("peak_delta", {})
    weights = metrics_cfg.get("composite", {}).get("weights", {})

    count_result = peak_count_score(
        measurement_features,
        theoretical,
        tolerance_nm=float(peak_count_cfg.get("wavelength_tolerance_nm", 5.0)),
    )
    delta_result = peak_delta_score(
        measurement_features,
        theoretical,
        tolerance_nm=float(peak_delta_cfg.get("tolerance_nm", 5.0)),
        tau_nm=float(peak_delta_cfg.get("tau_nm", 15.0)),
        penalty_unpaired=float(peak_delta_cfg.get("penalty_unpaired", 0.05)),
    )
    phase_result = phase_overlap_score(measurement_features, theoretical)

    component_scores = {
        "peak_count": count_result.score,
        "peak_delta": delta_result.score,
        "phase_overlap": phase_result.score,
    }
    composite = composite_score(component_scores, weights)
    component_scores["composite"] = composite
    diagnostics = {
        "peak_count": count_result.diagnostics,
        "peak_delta": delta_result.diagnostics,
        "phase_overlap": phase_result.diagnostics,
    }

    return CandidateResult(
        lipid_nm=params[0],
        aqueous_nm=params[1],
        roughness_A=params[2],
        scores=component_scores,
        diagnostics=diagnostics,
    )


def _apply_metric_overrides(metrics_cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    peak_count_cfg = metrics_cfg.setdefault("peak_count", {})
    if args.peak_count_tol is not None:
        peak_count_cfg["wavelength_tolerance_nm"] = float(args.peak_count_tol)

    peak_delta_cfg = metrics_cfg.setdefault("peak_delta", {})
    if args.peak_delta_tol is not None:
        peak_delta_cfg["tolerance_nm"] = float(args.peak_delta_tol)
    if args.peak_delta_tau is not None:
        peak_delta_cfg["tau_nm"] = float(args.peak_delta_tau)
    if args.peak_delta_penalty is not None:
        peak_delta_cfg["penalty_unpaired"] = float(args.peak_delta_penalty)

    if args.metric_weights is not None:
        w_count, w_delta, w_phase = args.metric_weights
        composite_cfg = metrics_cfg.setdefault("composite", {})
        composite_cfg["weights"] = {
            "peak_count": float(w_count),
            "peak_delta": float(w_delta),
            "phase_overlap": float(w_phase),
        }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if not validate_config(config):
        return 1

    analysis_cfg = copy.deepcopy(config.get("analysis", {}))
    metrics_cfg = analysis_cfg.setdefault("metrics", {})
    _apply_metric_overrides(metrics_cfg, args)

    measurement_cfg = config.get("measurements", {})
    measurement_df = load_measurement_spectrum(args.measurement, measurement_cfg)
    measurement_features = prepare_measurement(measurement_df, analysis_cfg)

    output_dir = args.output_dir or get_project_path("outputs/grid_search")
    output_dir.mkdir(parents=True, exist_ok=True)

    top_k: List[Tuple[float, CandidateResult]] = []
    results: List[CandidateResult] = []

    if args.grid_dir:
        grid, lipid_vals, aqueous_vals, rough_vals, wavelengths = load_cached_grid(args.grid_dir)
        iterator = iter_cached_spectra(grid, lipid_vals, aqueous_vals, rough_vals)
    else:
        calculator, wavelengths = make_single_spectrum_calculator(config)
        lipid_vals, aqueous_vals, rough_vals = prepare_parameter_arrays(config)
        iterator = iter_generated_spectra(calculator, lipid_vals, aqueous_vals, rough_vals)

    total_evaluated = 0
    for params, spectrum in iterator:
        if args.max_results is not None and total_evaluated >= args.max_results:
            break
        candidate = evaluate_candidate(
            params,
            spectrum,
            wavelengths,
            measurement_features,
            analysis_cfg,
            metrics_cfg,
        )
        results.append(candidate)
        total_evaluated += 1

        if len(top_k) < args.top_k:
            heapq.heappush(top_k, (candidate.composite, candidate))
        else:
            if candidate.composite > top_k[0][0]:
                heapq.heapreplace(top_k, (candidate.composite, candidate))

    if not results:
        raise RuntimeError("No spectra evaluated; check grid configuration")

    results_df = pd.DataFrame([cand.as_dict() for cand in results])
    results_csv = output_dir / "results.csv"
    results_df.to_csv(results_csv, index=False)

    top_candidates = [cand for _, cand in sorted(top_k, key=lambda entry: entry[0], reverse=True)]
    top_payload = [cand.as_dict() for cand in top_candidates]
    best_json = output_dir / "best_fit.json"
    with best_json.open("w", encoding="utf-8") as handle:
        json.dump(top_payload, handle, indent=2)

    summary = {
        "evaluated": total_evaluated,
        "top_k": args.top_k,
        "results_csv": str(results_csv),
        "best_fit_json": str(best_json),
        "measurement": str(args.measurement),
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Evaluated {total_evaluated} spectra. Results saved to {results_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
