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
    measurement_quality_score,
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


def _parse_previous_best_params(raw: str) -> Dict[str, float]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("Expected three comma-separated values for previous best parameters")
    lipid, aqueous, roughness = map(float, parts)
    return {"lipid_nm": lipid, "aqueous_nm": aqueous, "roughness_A": roughness}


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
        "--target-score",
        type=float,
        default=None,
        help="Stop early when a candidate meets or exceeds this composite score",
    )
    parser.add_argument(
        "--search-mode",
        choices=["full", "coarse-fine"],
        default="coarse-fine",
        help="Search strategy to use (default: coarse-fine)",
    )
    parser.add_argument(
        "--previous-best-params",
        type=str,
        default=None,
        help="Comma-separated lipid_nm,aqueous_nm,roughness_A from previous timestep for temporal seeding",
    )
    parser.add_argument(
        "--previous-best-score",
        type=float,
        default=None,
        help="Composite score of the previous timestep to gauge confidence for temporal seeding",
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


def _push_candidate(
    heap: List[Tuple[float, SpectrumScore]],
    score: SpectrumScore,
    top_k: int,
) -> None:
    entry = (score.composite, score)
    if len(heap) < top_k:
        heapq.heappush(heap, entry)
    elif score.composite > heap[0][0]:
        heapq.heapreplace(heap, entry)


def _heap_to_sorted_list(heap: List[Tuple[float, SpectrumScore]]) -> List[SpectrumScore]:
    return [score for _, score in sorted(heap, key=lambda item: item[0], reverse=True)]


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


def _extract_first(cfg: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return float(cfg[key])
    return None


def _prepare_parameter_arrays(config: Dict[str, Any]) -> ParameterGrid:
    params = config["parameters"]
    lipid_cfg = params["lipid"]
    aqueous_cfg = params["aqueous"]
    rough_cfg = params["roughness"]

    lipid_vals = _build_axis(lipid_cfg)
    aqueous_vals = _build_axis(aqueous_cfg)
    rough_vals = _build_axis(rough_cfg)
    return ParameterGrid(lipid=lipid_vals, aqueous=aqueous_vals, roughness=rough_vals)


def _build_coarse_parameter_grid(
    config: Dict[str, Any],
    analysis_cfg: Dict[str, Any],
    previous_params: Optional[Dict[str, float]] = None,
    previous_score: Optional[float] = None,
) -> ParameterGrid:
    params_cfg = config["parameters"]
    grid_cfg = analysis_cfg.get("grid_search", {})
    coarse_cfg = grid_cfg.get("coarse", {})
    use_temporal = bool(grid_cfg.get("use_temporal_seeding"))
    confidence_threshold = float(grid_cfg.get("confidence_threshold_for_narrow_window", 0.0))
    window_multiplier = float(grid_cfg.get("temporal_window_multiplier", 0.3))
    allow_temporal = use_temporal and previous_params is not None
    confident = previous_score is None or previous_score >= confidence_threshold

    axes: Dict[str, np.ndarray] = {}
    for axis_name, value_key in (
        ("lipid", "lipid_nm"),
        ("aqueous", "aqueous_nm"),
        ("roughness", "roughness_A"),
    ):
        axis_override = coarse_cfg.get(axis_name, {})
        global_axis = params_cfg[axis_name]
        min_val = float(axis_override.get("min", global_axis["min"]))
        max_val = float(axis_override.get("max", global_axis["max"]))
        step = _extract_first(axis_override, ("step", "step_nm", "step_A")) or float(global_axis.get("step", 1.0))

        center_value = None
        if allow_temporal and confident:
            center_value = previous_params.get(value_key)  # type: ignore[arg-type]
        if center_value is not None:
            span = float(global_axis["max"] - global_axis["min"])
            half_window = max(span * window_multiplier * 0.5, float(global_axis.get("step", 1.0)))
            min_val = max(float(global_axis["min"]), center_value - half_window)
            max_val = min(float(global_axis["max"]), center_value + half_window)

        axes[axis_name] = _build_axis({"min": min_val, "max": max_val, "step": step})

    return ParameterGrid(lipid=axes["lipid"], aqueous=axes["aqueous"], roughness=axes["roughness"])


def _build_refine_axis(
    axis_name: str,
    center_value: Optional[float],
    global_axis: Dict[str, Any],
    override_cfg: Dict[str, Any],
) -> np.ndarray:
    if center_value is None:
        return np.array(
            [float(global_axis["min"]), float(global_axis["max"])],
            dtype=float,
        )

    if axis_name == "roughness":
        window_keys = ("window_A", "window")
    else:
        window_keys = ("window_nm", "window")

    window = _extract_first(override_cfg, window_keys)
    if window is None:
        span = float(global_axis["max"] - global_axis["min"])
        window = max(span * 0.05, float(global_axis.get("step", 1.0)))
    step = _extract_first(override_cfg, ("step", "step_nm", "step_A"))
    if step is None:
        step = max(float(global_axis.get("step", 1.0)) / 2.0, 1e-3)

    min_val = max(float(global_axis["min"]), center_value - window)
    max_val = min(float(global_axis["max"]), center_value + window)
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    return _build_axis({"min": min_val, "max": max_val, "step": float(step)})


def _build_refine_grid(
    config: Dict[str, Any],
    analysis_cfg: Dict[str, Any],
    center: SpectrumScore,
) -> ParameterGrid:
    params_cfg = config["parameters"]
    refine_cfg = analysis_cfg.get("grid_search", {}).get("refine", {})

    lipid_vals = _build_refine_axis(
        "lipid",
        center.lipid_nm,
        params_cfg["lipid"],
        refine_cfg.get("lipid", {}),
    )
    aqueous_vals = _build_refine_axis(
        "aqueous",
        center.aqueous_nm,
        params_cfg["aqueous"],
        refine_cfg.get("aqueous", {}),
    )
    roughness_vals = _build_refine_axis(
        "roughness",
        center.roughness_A,
        params_cfg["roughness"],
        refine_cfg.get("roughness", {}),
    )
    return ParameterGrid(lipid=lipid_vals, aqueous=aqueous_vals, roughness=roughness_vals)


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
    *,
    measurement_quality=None,
    previous_params: Optional[Dict[str, float]] = None,
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
        measurement_quality=measurement_quality,
        previous_params=previous_params,
    )


def _evaluate_iterator(
    iterator: Iterable[Tuple[Tuple[float, float, float], np.ndarray]],
    *,
    measurement_features,
    wavelengths: np.ndarray,
    analysis_cfg: Dict[str, Any],
    metrics_cfg: Dict[str, Any],
    heap: List[Tuple[float, SpectrumScore]],
    top_k: int,
    measurement_quality=None,
    previous_params: Optional[Dict[str, float]] = None,
    stage_budget: Optional[int] = None,
    target_score: Optional[float] = None,
    verbose: bool = False,
    progress_label: str = "",
) -> Tuple[int, bool]:
    evaluated = 0
    reached_target = False
    for params, spectrum in iterator:
        if stage_budget is not None and evaluated >= stage_budget:
            break

        candidate = _score_candidate(
            measurement_features,
            wavelengths,
            spectrum,
            analysis_cfg,
            metrics_cfg,
            params,
            measurement_quality=measurement_quality,
            previous_params=previous_params,
        )
        evaluated += 1

        if verbose and evaluated % 50 == 0:
            label = f"[{progress_label}] " if progress_label else ""
            print(f"{label}{evaluated} candidates evaluated", flush=True)

        _push_candidate(heap, candidate, top_k)

        if target_score is not None and candidate.composite >= target_score:
            reached_target = True
            break

    return evaluated, reached_target


def _remaining_budget(limit: Optional[int], used: int) -> Optional[int]:
    if limit is None:
        return None
    return max(0, limit - used)


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

    quality_cfg = analysis_cfg.get("quality_gates", {})
    measurement_quality_result = None
    quality_failures: List[str] = []
    if quality_cfg:
        measurement_quality_result, quality_failures = measurement_quality_score(
            measurement_features,
            min_peaks=quality_cfg.get("min_peaks"),
            min_signal_amplitude=quality_cfg.get("min_signal_amplitude"),
            min_wavelength_span_nm=quality_cfg.get("min_wavelength_span_nm"),
        )
        min_quality = float(quality_cfg.get("min_quality_score", 0.0))
        if quality_cfg.get("enabled") and measurement_quality_result.score < min_quality:
            details = ", ".join(quality_failures) if quality_failures else "score below threshold"
            print(
                f"Measurement quality gate failed ({details}). "
                f"Score={measurement_quality_result.score:.3f} < min_quality_score={min_quality:.3f}"
            )
            return 2

    grid_search_cfg = analysis_cfg.get("grid_search", {})
    config_limit = grid_search_cfg.get("max_total_evals")
    config_limit = int(config_limit) if config_limit is not None else None
    max_total_evals: Optional[int] = args.max_spectra
    if max_total_evals is not None and config_limit is not None:
        max_total_evals = min(max_total_evals, config_limit)
    elif max_total_evals is None:
        max_total_evals = config_limit

    output_dir = args.output_dir or get_project_path("outputs/grid_search")
    output_dir.mkdir(parents=True, exist_ok=True)

    previous_best_params = None
    if args.previous_best_params:
        try:
            previous_best_params = _parse_previous_best_params(args.previous_best_params)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    previous_best_score = args.previous_best_score
    target_score = args.target_score
    search_mode = args.search_mode

    using_cached_grid = args.grid_dir is not None or args.grid_file is not None
    if using_cached_grid and search_mode != "full":
        if args.verbose:
            print("Cached grids do not support coarse-fine search; falling back to full search.")
        search_mode = "full"

    total_evaluated = 0
    top_scores: List[SpectrumScore] = []

    wavelengths: np.ndarray
    params: ParameterGrid

    if args.grid_dir is not None:
        grid, wavelengths, params = _load_grid_from_directory(args.grid_dir)
        iterator = _iter_cached_spectra(grid, params)
        heap: List[Tuple[float, SpectrumScore]] = []
        evaluated, _ = _evaluate_iterator(
            iterator,
            measurement_features=measurement_features,
            wavelengths=wavelengths,
            analysis_cfg=analysis_cfg,
            metrics_cfg=metrics_cfg,
            heap=heap,
            top_k=args.top_k,
            measurement_quality=measurement_quality_result,
            previous_params=previous_best_params,
            stage_budget=max_total_evals,
            target_score=target_score,
            verbose=args.verbose,
            progress_label="cached-grid",
        )
        total_evaluated = evaluated
        if not heap:
            print("No spectra evaluated; check configuration and cached grid parameters.")
            return 1
        top_scores = _heap_to_sorted_list(heap)
    elif args.grid_file is not None:
        grid, wavelengths, params = _load_grid(args.grid_file, args.meta_file)
        iterator = _iter_cached_spectra(grid, params)
        heap = []
        evaluated, _ = _evaluate_iterator(
            iterator,
            measurement_features=measurement_features,
            wavelengths=wavelengths,
            analysis_cfg=analysis_cfg,
            metrics_cfg=metrics_cfg,
            heap=heap,
            top_k=args.top_k,
            measurement_quality=measurement_quality_result,
            previous_params=previous_best_params,
            stage_budget=max_total_evals,
            target_score=target_score,
            verbose=args.verbose,
            progress_label="cached-grid",
        )
        total_evaluated = evaluated
        if not heap:
            print("No spectra evaluated; check configuration and cached grid parameters.")
            return 1
        top_scores = _heap_to_sorted_list(heap)
    else:
        calculator, wavelengths = make_single_spectrum_calculator(config)
        params = _prepare_parameter_arrays(config)

        if search_mode == "full":
            iterator_full = _iter_generated_spectra(calculator, params)
            heap = []
            evaluated, _ = _evaluate_iterator(
                iterator_full,
                measurement_features=measurement_features,
                wavelengths=wavelengths,
                analysis_cfg=analysis_cfg,
                metrics_cfg=metrics_cfg,
                heap=heap,
                top_k=args.top_k,
                measurement_quality=measurement_quality_result,
                previous_params=previous_best_params,
                stage_budget=max_total_evals,
                target_score=target_score,
                verbose=args.verbose,
                progress_label="full-grid",
            )
            total_evaluated = evaluated
            if not heap:
                print("No spectra evaluated; check configuration and grid parameters.")
                return 1
            top_scores = _heap_to_sorted_list(heap)
        else:
            coarse_grid = _build_coarse_parameter_grid(
                config,
                analysis_cfg,
                previous_params=previous_best_params,
                previous_score=previous_best_score,
            )
            coarse_cfg = grid_search_cfg.get("coarse", {})
            coarse_top_k = max(1, int(coarse_cfg.get("top_k", max(args.top_k, 10))))
            coarse_heap: List[Tuple[float, SpectrumScore]] = []
            coarse_iterator = _iter_generated_spectra(calculator, coarse_grid)
            stage_budget = max_total_evals
            coarse_evaluated, coarse_hit_target = _evaluate_iterator(
                coarse_iterator,
                measurement_features=measurement_features,
                wavelengths=wavelengths,
                analysis_cfg=analysis_cfg,
                metrics_cfg=metrics_cfg,
                heap=coarse_heap,
                top_k=coarse_top_k,
                measurement_quality=measurement_quality_result,
                previous_params=previous_best_params,
                stage_budget=stage_budget,
                target_score=target_score,
                verbose=args.verbose,
                progress_label="coarse",
            )
            total_evaluated = coarse_evaluated
            if not coarse_heap:
                print("No spectra evaluated; check coarse grid configuration.")
                return 1

            coarse_results = _heap_to_sorted_list(coarse_heap)
            final_heap: List[Tuple[float, SpectrumScore]] = []
            for score in coarse_results:
                _push_candidate(final_heap, score, args.top_k)

            refine_cfg = grid_search_cfg.get("refine", {})
            remaining_budget = _remaining_budget(max_total_evals, total_evaluated)
            hit_target = coarse_hit_target
            refined_evaluations = 0
            refine_limit = refine_cfg.get("max_refine_candidates")
            refine_limit = int(refine_limit) if refine_limit is not None else None

            if not hit_target and refine_cfg:
                for idx, seed in enumerate(coarse_results):
                    if hit_target:
                        break
                    if remaining_budget is not None and remaining_budget <= 0:
                        break
                    local_grid = _build_refine_grid(config, analysis_cfg, seed)
                    if local_grid.size == 0:
                        continue

                    stage_budget = remaining_budget
                    if refine_limit is not None:
                        remaining_refine = max(refine_limit - refined_evaluations, 0)
                        if remaining_refine == 0:
                            break
                        stage_budget = remaining_refine if stage_budget is None else min(stage_budget, remaining_refine)
                    if stage_budget == 0:
                        break

                    refine_iterator = _iter_generated_spectra(calculator, local_grid)
                    evaluated, stage_hit = _evaluate_iterator(
                        refine_iterator,
                        measurement_features=measurement_features,
                        wavelengths=wavelengths,
                        analysis_cfg=analysis_cfg,
                        metrics_cfg=metrics_cfg,
                        heap=final_heap,
                        top_k=args.top_k,
                        measurement_quality=measurement_quality_result,
                        previous_params=previous_best_params,
                        stage_budget=stage_budget,
                        target_score=target_score,
                        verbose=args.verbose,
                        progress_label=f"refine-{idx + 1}",
                    )
                    refined_evaluations += evaluated
                    total_evaluated += evaluated
                    remaining_budget = _remaining_budget(max_total_evals, total_evaluated)
                    if stage_hit:
                        hit_target = True
                        break

            top_scores = _heap_to_sorted_list(final_heap) if final_heap else []
            if not top_scores:
                top_scores = coarse_results

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
        "search_mode": search_mode,
        "evaluated_spectra": total_evaluated,
        "top_k": args.top_k,
        "target_score": target_score,
        "results_csv": str(csv_path),
        "results": df.to_dict(orient="records"),
    }
    summary_path = prefix.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    best_fit_path = prefix.with_name(prefix.name + "_best.json")
    with best_fit_path.open("w", encoding="utf-8") as handle:
        json.dump([score.as_dict() for score in top_scores], handle, indent=2)

    print(f"Evaluated {total_evaluated} spectra. Top-{len(top_scores)} results saved to:")
    print(f"  - {csv_path}")
    print(f"  - {summary_path}")
    print(f"  - {best_fit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
