#!/usr/bin/env python
"""CLI grid-search driver for tear-film spectra fitting."""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
from dataclasses import asdict
from datetime import datetime
from heapq import heappush, heappushpop, nlargest
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd

import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.measurement_utils import load_txt_file_enhanced
from analysis.metrics import (
    MeasurementArtifacts,
    MetricScores,
    prepare_measurement_artifacts,
    score_spectrum,
)
from tear_film_generator import (
    PROJECT_ROOT,
    load_config,
    make_single_spectrum_calculator,
    validate_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Grid search for best-fit tear film spectra")
    parser.add_argument(
        "--config-file",
        type=pathlib.Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Configuration YAML (defaults to project config.yaml)",
    )
    parser.add_argument(
        "--measurement",
        type=pathlib.Path,
        required=True,
        help="Measurement TXT file to match",
    )
    parser.add_argument(
        "--grid-file",
        type=pathlib.Path,
        help="Precomputed grid.npy file to score instead of calling the DLL",
    )
    parser.add_argument(
        "--meta-file",
        type=pathlib.Path,
        help="Metadata JSON describing parameter axes (defaults to grid sibling meta.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top candidates to retain (default: 10)",
    )
    parser.add_argument(
        "--max-spectra",
        type=int,
        help="Optional limit on number of spectra evaluated (for smoke tests)",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=PROJECT_ROOT / "outputs" / "grid_search",
        help="Directory to store scoring artifacts",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress for every spectrum",
    )
    return parser.parse_args()


def _load_config(config_path: pathlib.Path) -> Dict[str, object]:
    config = load_config(config_path)
    if not validate_config(config):
        raise SystemExit("Configuration validation failed.")
    return config


def _load_measurement(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Measurement file not found: {path}")
    df = load_txt_file_enhanced(path)
    if df.empty:
        raise ValueError(f"Measurement contained no usable data: {path}")
    return df.dropna()


def _parameter_ranges(config: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = config["parameters"]  # type: ignore[index]
    lipid_cfg = params["lipid"]
    aqueous_cfg = params["aqueous"]
    rough_cfg = params["roughness"]
    lipid_vals = np.arange(lipid_cfg["min"], lipid_cfg["max"], lipid_cfg["step"], dtype=float)
    aqueous_vals = np.arange(aqueous_cfg["min"], aqueous_cfg["max"], aqueous_cfg["step"], dtype=float)
    rough_vals = np.arange(rough_cfg["min"], rough_cfg["max"], rough_cfg["step"], dtype=float)
    return lipid_vals, aqueous_vals, rough_vals


def _grid_iterator_from_calc(
    calc_func,
    wavelengths: np.ndarray,
    lipid_vals: np.ndarray,
    aqueous_vals: np.ndarray,
    rough_vals: np.ndarray,
) -> Iterator[Tuple[float, float, float, np.ndarray]]:
    total = len(lipid_vals) * len(aqueous_vals) * len(rough_vals)
    count = 0
    for l, a, r in itertools.product(lipid_vals, aqueous_vals, rough_vals):
        count += 1
        yield float(l), float(a), float(r), calc_func(l, a, r)
        if count % 100 == 0:
            progress = 100.0 * count / total
            print(f"{count}/{total} spectra evaluated ({progress:5.1f}%)", flush=True)


def _grid_iterator_from_numpy(
    grid: np.ndarray,
    lipid_vals: np.ndarray,
    aqueous_vals: np.ndarray,
    rough_vals: np.ndarray,
) -> Iterator[Tuple[float, float, float, np.ndarray]]:
    for i, l in enumerate(lipid_vals):
        for j, a in enumerate(aqueous_vals):
            for k, r in enumerate(rough_vals):
                yield float(l), float(a), float(r), grid[i, j, k, :]


def _load_grid_with_meta(grid_path: pathlib.Path, meta_path: pathlib.Path | None):
    grid = np.load(grid_path)
    if meta_path is None:
        meta_path = grid_path.with_name("meta.json")
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    lipid_vals = np.asarray(meta["lipid_nm"], dtype=float)
    aqueous_vals = np.asarray(meta["aqueous_nm"], dtype=float)
    rough_vals = np.asarray(meta["rough_A"], dtype=float)
    wavelengths = np.asarray(meta["wavelengths_nm"], dtype=float)
    return grid, wavelengths, lipid_vals, aqueous_vals, rough_vals


def main() -> int:
    args = parse_args()
    config = _load_config(args.config_file)

    measurement_df = _load_measurement(args.measurement)
    measurement_name = args.measurement.stem

    if args.grid_file:
        grid, wavelengths, lipid_vals, aqueous_vals, rough_vals = _load_grid_with_meta(
            args.grid_file,
            args.meta_file,
        )
        calc_iterator = _grid_iterator_from_numpy(
            grid,
            lipid_vals,
            aqueous_vals,
            rough_vals,
        )
    else:
        calc_func, wavelengths = make_single_spectrum_calculator(config)
        lipid_vals, aqueous_vals, rough_vals = _parameter_ranges(config)
        calc_iterator = _grid_iterator_from_calc(
            calc_func,
            wavelengths,
            lipid_vals,
            aqueous_vals,
            rough_vals,
        )

    analysis_config = config.get("analysis", {}) or {}
    metrics_config = analysis_config.get("metrics", {}) or {}
    measurement_artifacts = prepare_measurement_artifacts(
        measurement_df,
        wavelengths=wavelengths,
        analysis_config=analysis_config,
    )

    heap: List[Tuple[float, MetricScores]] = []
    evaluated = 0

    for lipid_nm, aqueous_nm, roughness_A, spectrum in calc_iterator:
        evaluated += 1
        if args.max_spectra and evaluated > args.max_spectra:
            break

        if args.verbose and evaluated % 25 == 0:
            print(
                f"Evaluated {evaluated} spectra "
                f"(L={lipid_nm:.1f}, A={aqueous_nm:.1f}, R={roughness_A:.1f})",
                flush=True,
            )

        scores = score_spectrum(
            wavelengths,
            spectrum,
            measurement=measurement_artifacts,
            metrics_config=metrics_config,
            lipid_nm=lipid_nm,
            aqueous_nm=aqueous_nm,
            roughness_A=roughness_A,
        )

        entry = (scores.composite_score, scores)
        if len(heap) < args.top_k:
            heappush(heap, entry)
        else:
            heappushpop(heap, entry)

    if not heap:
        print("No spectra evaluated; check configuration.")
        return 1

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = output_dir / f"grid_search_{measurement_name}_{timestamp}"

    top_results = [
        result for _, result in sorted(nlargest(args.top_k, heap), reverse=True)
    ]

    csv_rows = [
        {
            "lipid_nm": res.lipid_nm,
            "aqueous_nm": res.aqueous_nm,
            "roughness_A": res.roughness_A,
            "peak_count_score": res.peak_count_score,
            "peak_delta_score": res.peak_delta_score,
            "phase_overlap_score": res.phase_overlap_score,
            "composite_score": res.composite_score,
            "matched_peaks": res.matched_peaks,
            "unmatched_measured": res.unmatched_measured,
            "unmatched_theoretical": res.unmatched_theoretical,
        }
        for res in top_results
    ]

    pd.DataFrame(csv_rows).to_csv(output_prefix.with_suffix(".csv"), index=False)

    summary = {
        "measurement_file": str(args.measurement),
        "config_file": str(args.config_file),
        "grid_file": str(args.grid_file) if args.grid_file else None,
        "evaluated_spectra": evaluated,
        "top_k": args.top_k,
        "results_csv": str(output_prefix.with_suffix(".csv")),
        "results": [asdict(res) for res in top_results],
        "analysis_config": analysis_config,
    }

    with output_prefix.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Evaluated {evaluated} spectra. Top-{len(top_results)} results saved to:")
    print(f"  - {output_prefix.with_suffix('.csv')}")
    print(f"  - {output_prefix.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
