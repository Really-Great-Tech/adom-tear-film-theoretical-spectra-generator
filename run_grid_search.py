#!/usr/bin/env python
"""CLI grid-search driver for tear-film spectra fitting."""

from __future__ import annotations

import argparse
import copy
import heapq
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis import (  # type: ignore  # pylint: disable=wrong-import-position
    prepare_measurement,
    prepare_theoretical_spectrum,
    score_spectrum,
    SpectrumScore,
)
from analysis.measurement_utils import (  # type: ignore  # pylint: disable=wrong-import-position
    load_measurement_spectrum,
)
from tear_film_generator import (  # type: ignore  # pylint: disable=wrong-import-position
    PROJECT_ROOT,
    get_project_path,
    load_config,
    make_single_spectrum_calculator,
    validate_config,
)


@dataclass
class ParameterGrid:
    lipid: np.ndarray
    aqueous: np.ndarray
    roughness: np.ndarray

    @property
    def size(self) -> int:
        return int(len(self.lipid) * len(self.aqueous) * len(self.roughness))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score theoretical spectra against a measurement")
    parser.add_argument(
        "measurement",
        nargs="?",
        type=Path,
        help="Path to measurement spectrum file",
    )
    parser.add_argument(
        "--measurement",
        dest="measurement_flag",
        type=Path,
        help="Explicit measurement spectrum path (overrides positional)",
    )
    parser.add_argument(
        "--config",
        "--config-file",
        dest="config_path",
        type=Path,
        default=None,
        help="Configuration YAML (defaults to config.yaml)",
    )
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=None,
        help="Directory containing grid.npy and meta.json produced by run_tear_film_generator",
    )
    parser.add_argument(
        "--grid-file",
        type=Path,
        default=None,
        help="Specific grid.npy file to score",
    )
    parser.add_argument(
        "--meta-file",
        type=Path,
        default=None,
        help="Metadata JSON describing parameter axes (defaults to sibling meta.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of best candidates to keep (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write results (defaults to outputs/grid_search)",
    )
    parser.add_argument(
        "--max-spectra",
        "--max-results",
        dest="max_spectra",
        type=int,
        default=None,
        help="Optional cap on evaluated spectra (useful for smoke tests)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress updates during evaluation",
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


def _resolve_measurement_path(args: argparse.Namespace) -> Path:
    path = args.measurement_flag or args.measurement
    if path is None:
        raise SystemExit("Measurement path is required")
    return path


def _load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    config = load_config(config_path)
    if not validate_config(config):
        raise SystemExit("Configuration validation failed.")
    return config


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



def _build_axis(cfg: Dict[str, Any]) -> np.ndarray:
    min_val = float(cfg["min"])
    max_val = float(cfg["max"])
    step = float(cfg["step"])
    if step <= 0:
        return np.array([min_val, max_val], dtype=float)

    values = np.arange(min_val, max_val + step * 0.5, step, dtype=float)
    if values.size == 0:
        values = np.array([min_val], dtype=float)

    if values[-1] < max_val - 1e-9:
        values = np.append(values, max_val)
    else:
        values[-1] = max_val
    return values


def _prepare_parameter_arrays(config: Dict[str, Any]) -> ParameterGrid:
    params = config["parameters"]
    lipid_cfg = params["lipid"]
    aqueous_cfg = params["aqueous"]
    rough_cfg = params["roughness"]

    lipid_vals = _build_axis(lipid_cfg)
    aqueous_vals = _build_axis(aqueous_cfg)
    rough_vals = _build_axis(rough_cfg)
    return ParameterGrid(lipid=lipid_vals, aqueous=aqueous_vals, roughness=rough_vals)


def _load_grid_from_directory(cache_dir: Path) -> Tuple[np.ndarray, np.ndarray, ParameterGrid]:
    grid_path = cache_dir / "grid.npy"
    meta_path = cache_dir / "meta.json"
    return _load_grid(grid_path, meta_path)


def _load_grid(grid_path: Path, meta_path: Optional[Path]) -> Tuple[np.ndarray, np.ndarray, ParameterGrid]:
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")
    if meta_path is None:
        meta_path = grid_path.with_name("meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {meta_path}")

    grid = np.load(grid_path, allow_pickle=False)
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    lipid_vals = np.asarray(meta["lipid_nm"], dtype=float)
    aqueous_vals = np.asarray(meta["aqueous_nm"], dtype=float)
    rough_vals = np.asarray(meta["rough_A"], dtype=float)
    wavelengths = np.asarray(meta["wavelengths_nm"], dtype=float)
    return grid, wavelengths, ParameterGrid(lipid_vals, aqueous_vals, rough_vals)


def _iter_cached_spectra(
    grid: np.ndarray,
    params: ParameterGrid,
) -> Iterable[Tuple[Tuple[float, float, float], np.ndarray]]:
    for i, lipid in enumerate(params.lipid):
        for j, aqueous in enumerate(params.aqueous):
            for k, rough in enumerate(params.roughness):
                yield (float(lipid), float(aqueous), float(rough)), grid[i, j, k, :]


def _iter_generated_spectra(
    calculator,
    params: ParameterGrid,
) -> Iterable[Tuple[Tuple[float, float, float], np.ndarray]]:
    for lipid in params.lipid:
        for aqueous in params.aqueous:
            for rough in params.roughness:
                spectrum = calculator(float(lipid), float(aqueous), float(rough))
                yield (float(lipid), float(aqueous), float(rough)), spectrum


def _score_candidate(
    measurement_features,
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    analysis_cfg: Dict[str, Any],
    metrics_cfg: Dict[str, Any],
    params: Tuple[float, float, float],
) -> SpectrumScore:
    theoretical_features = prepare_theoretical_spectrum(
        wavelengths,
        spectrum,
        measurement_features,
        analysis_cfg,
    )
    lipid_nm, aqueous_nm, roughness_A = params
    return score_spectrum(
        measurement_features,
        theoretical_features,
        metrics_cfg,
        lipid_nm=lipid_nm,
        aqueous_nm=aqueous_nm,
        roughness_A=roughness_A,
    )


def main() -> int:
    args = parse_args()
    measurement_path = _resolve_measurement_path(args)
    config = _load_config(args.config_path)

    analysis_cfg = copy.deepcopy(config.get("analysis", {}))
    metrics_cfg = analysis_cfg.setdefault("metrics", {})
    _apply_metric_overrides(metrics_cfg, args)

    measurement_cfg = config.get("measurements", {})
    measurement_df = load_measurement_spectrum(measurement_path, measurement_cfg)
    measurement_features = prepare_measurement(measurement_df, analysis_cfg)

    output_dir = args.output_dir or get_project_path("outputs/grid_search")
    output_dir.mkdir(parents=True, exist_ok=True)

    iterator: Iterable[Tuple[Tuple[float, float, float], np.ndarray]]
    wavelengths: np.ndarray
    params: ParameterGrid

    if args.grid_dir is not None:
        grid, wavelengths, params = _load_grid_from_directory(args.grid_dir)
        iterator = _iter_cached_spectra(grid, params)
    elif args.grid_file is not None:
        grid, wavelengths, params = _load_grid(args.grid_file, args.meta_file)
        iterator = _iter_cached_spectra(grid, params)
    else:
        calculator, wavelengths = make_single_spectrum_calculator(config)
        params = _prepare_parameter_arrays(config)
        iterator = _iter_generated_spectra(calculator, params)

    heap: List[Tuple[float, SpectrumScore]] = []
    evaluated = 0
    total_candidates = params.size

    for (lipid_nm, aqueous_nm, roughness_A), spectrum in iterator:
        if args.max_spectra is not None and evaluated >= args.max_spectra:
            break

        candidate = _score_candidate(
            measurement_features,
            wavelengths,
            spectrum,
            analysis_cfg,
            metrics_cfg,
            (lipid_nm, aqueous_nm, roughness_A),
        )
        evaluated += 1

        if args.verbose and evaluated % 100 == 0:
            progress = 100.0 * evaluated / max(total_candidates, 1)
            print(
                f"{evaluated}/{total_candidates} spectra evaluated ({progress:5.1f}%)",
                flush=True,
            )

        entry = (candidate.composite, candidate)
        if len(heap) < args.top_k:
            heapq.heappush(heap, entry)
        else:
            if candidate.composite > heap[0][0]:
                heapq.heapreplace(heap, entry)

    if not heap:
        print("No spectra evaluated; check configuration and grid parameters.")
        return 1

    top_scores = [score for _, score in sorted(heap, key=lambda item: item[0], reverse=True)]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    measurement_stem = measurement_path.stem
    prefix = output_dir / f"grid_search_{measurement_stem}_{timestamp}"

    df = pd.DataFrame(score.as_dict() for score in top_scores)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    csv_path = prefix.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    summary = {
        "measurement_file": str(measurement_path),
        "config_file": str(args.config_path or PROJECT_ROOT / "config.yaml"),
        "evaluated_spectra": evaluated,
        "top_k": args.top_k,
        "results_csv": str(csv_path),
        "results": df.to_dict(orient="records"),
    }
    summary_path = prefix.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    best_fit_path = prefix.with_name(prefix.name + "_best.json")
    with best_fit_path.open("w", encoding="utf-8") as handle:
        json.dump([score.as_dict() for score in top_scores], handle, indent=2)

    print(f"Evaluated {evaluated} spectra. Top-{len(top_scores)} results saved to:")
    print(f"  - {csv_path}")
    print(f"  - {summary_path}")
    print(f"  - {best_fit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
