"""
PyElli grid search and spectrum service. Single place that calls PyElliGridSearch.
Separation of concerns: routes handle HTTP, service handles domain logic.
"""
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from backend import config

# Backend only: heavy deps live here
from exploration.pyelli_exploration.pyelli_grid_search import PyElliGridSearch
from exploration.pyelli_exploration.run_last_two_cycles_seed_tune import run_last_two_cycles_seed_tune_sync

logger = logging.getLogger(__name__)

# Grid search async: stage (idle|running|done|failed), result (list of dicts), error (str)
_grid_search_progress: Dict = {
    "stage": "idle",
    "result": None,
    "error": None,
}


def get_grid_search_progress() -> Dict:
    """Return current grid search progress. Includes result when stage==done, error when stage==failed."""
    return dict(_grid_search_progress)


def set_grid_search_running() -> None:
    _grid_search_progress["stage"] = "running"
    _grid_search_progress["result"] = None
    _grid_search_progress["error"] = None


def set_grid_search_result(results: List) -> None:
    """Store serialized result (list of dicts from GridSearchResponse.from_results)."""
    from backend.schemas import GridSearchResponse
    resp = GridSearchResponse.from_results(results)
    _grid_search_progress["stage"] = "done"
    _grid_search_progress["result"] = resp.results
    _grid_search_progress["error"] = None


def set_grid_search_failed(err: str) -> None:
    _grid_search_progress["stage"] = "failed"
    _grid_search_progress["result"] = None
    _grid_search_progress["error"] = err


# Progress for last-two-cycles full test (run_folder, processed, total, stage). Updated by sync via callback.
_last_two_cycles_progress: Dict[str, Optional[str | int]] = {
    "run_folder": None,
    "processed": 0,
    "total": 0,
    "stage": "idle",  # idle | running | seed | tune | done | failed
}
# When stage=="done", result is stored here; when stage=="failed", error message here.
_last_two_cycles_result: Optional[dict] = None
_last_two_cycles_error: Optional[str] = None


def get_last_two_cycles_progress() -> Dict:
    """Return a copy of current last-two-cycles progress. Includes result when stage==done, error when stage==failed."""
    out = dict(_last_two_cycles_progress)
    if _last_two_cycles_progress.get("stage") == "done" and _last_two_cycles_result is not None:
        out["result"] = _last_two_cycles_result
    if _last_two_cycles_progress.get("stage") == "failed" and _last_two_cycles_error is not None:
        out["error"] = _last_two_cycles_error
    return out


def create_grid_search(
    materials_path: Path,
    lipid_file: Optional[str] = None,
    water_file: Optional[str] = None,
    mucus_file: Optional[str] = None,
    substratum_file: Optional[str] = None,
) -> PyElliGridSearch:
    """Build PyElliGridSearch with given material files. Fail-fast if path invalid."""
    if not materials_path.is_dir():
        logger.error("Materials path not found: %s", materials_path)
        raise FileNotFoundError(f"Materials path not found: {materials_path}")
    logger.debug("create_grid_search materials_path=%s", materials_path)
    return PyElliGridSearch(
        materials_path,
        lipid_file=lipid_file or PyElliGridSearch.DEFAULT_LIPID_FILE,
        water_file=water_file or PyElliGridSearch.DEFAULT_WATER_FILE,
        mucus_file=mucus_file or PyElliGridSearch.DEFAULT_MUCUS_FILE,
        substratum_file=substratum_file or PyElliGridSearch.DEFAULT_SUBSTRATUM_FILE,
    )


def run_grid_search(
    materials_path: Path,
    wavelengths: List[float],
    measured: List[float],
    lipid_range: tuple,
    aqueous_range: tuple,
    roughness_range: tuple,
    top_k: int = 10,
    enable_roughness: bool = True,
    search_strategy: str = "Dynamic Search",
    max_combinations: Optional[int] = 30000,
    lipid_file: Optional[str] = None,
    water_file: Optional[str] = None,
    mucus_file: Optional[str] = None,
    substratum_file: Optional[str] = None,
):
    """Run grid search and return list of PyElliResult. No HTTP here."""
    wl = np.asarray(wavelengths, dtype=float)
    meas = np.asarray(measured, dtype=float)
    if len(wl) != len(meas):
        raise ValueError("wavelengths and measured must have same length")
    if len(wl) < 10:
        raise ValueError("At least 10 points required")

    logger.info("run_grid_search materials_path=%s strategy=%s top_k=%s", materials_path, search_strategy, top_k)
    grid = create_grid_search(
        materials_path,
        lipid_file=lipid_file,
        water_file=water_file,
        mucus_file=mucus_file,
        substratum_file=substratum_file,
    )
    t0 = time.perf_counter()
    results = grid.run_grid_search(
        wl,
        meas,
        lipid_range=lipid_range,
        aqueous_range=aqueous_range,
        roughness_range=roughness_range,
        top_k=top_k,
        enable_roughness=enable_roughness,
        search_strategy=search_strategy,
        max_combinations=max_combinations,
    )
    elapsed = time.perf_counter() - t0
    logger.info("run_grid_search finished count=%s elapsed_sec=%.2f", len(results), elapsed)
    return results


def calculate_theoretical(
    materials_path: Path,
    wavelengths: List[float],
    lipid_nm: float,
    aqueous_nm: float,
    roughness_angstrom: float,
    enable_roughness: bool = True,
    lipid_file: Optional[str] = None,
    water_file: Optional[str] = None,
    mucus_file: Optional[str] = None,
    substratum_file: Optional[str] = None,
) -> List[float]:
    """Compute theoretical spectrum. Returns list for JSON."""
    wl = np.asarray(wavelengths, dtype=float)
    if len(wl) < 10:
        raise ValueError("At least 10 wavelengths required")
    roughness_nm = roughness_angstrom / 10.0  # UI uses Å; PyElli method expects nm then converts

    grid = create_grid_search(
        materials_path,
        lipid_file=lipid_file,
        water_file=water_file,
        mucus_file=mucus_file,
        substratum_file=substratum_file,
    )
    spectrum = grid.calculate_theoretical_spectrum(
        wl, lipid_nm, aqueous_nm, roughness_nm, enable_roughness=enable_roughness
    )
    return spectrum.tolist()


def align_spectra(
    measured: List[float],
    theoretical: List[float],
    focus_min_nm: float = 600.0,
    focus_max_nm: float = 1120.0,
    wavelengths: Optional[List[float]] = None,
) -> List[float]:
    """Align theoretical to measured via linear regression (a*theo + b). Returns aligned list."""
    meas = np.asarray(measured, dtype=float)
    theo = np.asarray(theoretical, dtype=float)
    if len(meas) != len(theo):
        raise ValueError("measured and theoretical must have same length")
    wl = np.asarray(wavelengths, dtype=float) if wavelengths else None
    if wl is not None and len(wl) != len(meas):
        raise ValueError("wavelengths length must match measured")

    if wl is not None:
        mask = (wl >= focus_min_nm) & (wl <= focus_max_nm)
        meas_fit = meas[mask]
        theo_fit = theo[mask]
    else:
        meas_fit = meas
        theo_fit = theo

    if len(theo_fit) < 10 or np.std(theo_fit) < 1e-10:
        scale = np.mean(meas) / np.mean(theo) if np.mean(theo) > 0 else 1.0
        return (theo * scale).tolist()
    A = np.vstack([theo_fit, np.ones_like(theo_fit)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, meas_fit, rcond=None)
    aligned = np.clip(coeffs[0] * theo + coeffs[1], 0, None)
    return aligned.tolist()


def run_last_two_cycles_seed_tune(
    run_folder_name: str,
    full_test_cycles_dir: Path,
    materials_path: Path,
    *,
    seed_index: int = 0,
    max_tune: Optional[int] = None,
) -> dict:
    """Run last-two-cycles seed+tune on a run folder. Updates progress/result/error in module state. Returns dict on success."""
    global _last_two_cycles_result, _last_two_cycles_error
    run_dir = full_test_cycles_dir / run_folder_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")

    _last_two_cycles_result = None
    _last_two_cycles_error = None
    _last_two_cycles_progress["run_folder"] = run_folder_name
    _last_two_cycles_progress["stage"] = "running"
    _last_two_cycles_progress["processed"] = 0
    _last_two_cycles_progress["total"] = 0

    def _progress_callback(processed: int, total: int, stage: str) -> None:
        _last_two_cycles_progress["processed"] = processed
        _last_two_cycles_progress["total"] = total
        _last_two_cycles_progress["stage"] = stage

    try:
        out = run_last_two_cycles_seed_tune_sync(
            run_dir,
            materials_path,
            seed_index=seed_index,
            max_tune=max_tune,
            progress_callback=_progress_callback,
        )
        _last_two_cycles_progress["stage"] = "done"
        _last_two_cycles_progress["processed"] = out.get("summary", {}).get("processed", 0)
        _last_two_cycles_progress["total"] = out.get("summary", {}).get("total_spectra", 0)
        _last_two_cycles_result = out
        return out
    except Exception as e:
        _last_two_cycles_progress["stage"] = "failed"
        _last_two_cycles_error = str(e)
        logger.exception("last-two-cycles-seed-tune failed: %s", e)
        raise


# ---------------------------------------------------------------------------
# Spectrum source browsing (production: Streamlit calls these when local data is missing)
# ---------------------------------------------------------------------------

def _resolve_source_path(source_name: str) -> Path:
    """Resolve a source name to an absolute path on the backend filesystem. Raises FileNotFoundError."""
    rel = config.SPECTRUM_SOURCES.get(source_name)
    if rel is None:
        raise FileNotFoundError(f"Unknown spectrum source: {source_name}")
    abs_path = config.PROJECT_ROOT / rel
    if not abs_path.is_dir():
        raise FileNotFoundError(f"Source directory not found on backend: {abs_path}")
    return abs_path


def _list_spectrum_txt(directory: Path, nested: bool) -> List[Path]:
    """Return sorted list of spectrum .txt files (excluding BestFit) from a directory."""
    files: List[Path] = []
    if nested:
        for subdir in sorted(directory.iterdir()):
            if subdir.is_dir():
                for f in subdir.glob("(Run)spectra_*.txt"):
                    if "_BestFit" not in f.name:
                        files.append(f)
    else:
        files = sorted(
            f for f in directory.glob("(Run)spectra_*.txt") if "_BestFit" not in f.name
        )
    return files


def list_spectrum_sources() -> List[Dict[str, Any]]:
    """Return available spectrum sources that exist on the backend filesystem."""
    sources = []
    for name, rel in config.SPECTRUM_SOURCES.items():
        abs_path = config.PROJECT_ROOT / rel
        if abs_path.is_dir():
            nested = name in config.NESTED_SPECTRUM_SOURCES
            count = len(_list_spectrum_txt(abs_path, nested))
            sources.append({"name": name, "file_count": count})
    return sources


def list_spectrum_files(source_name: str) -> List[Dict[str, str]]:
    """Return list of spectrum files for a given source."""
    abs_path = _resolve_source_path(source_name)
    nested = source_name in config.NESTED_SPECTRUM_SOURCES
    files = _list_spectrum_txt(abs_path, nested)
    return [{"name": f.name, "relative_path": str(f.relative_to(config.PROJECT_ROOT))} for f in files]


def read_spectrum_content(source_name: str, filename: str) -> Dict[str, Any]:
    """Read and parse a spectrum file, returning wavelengths and measured values."""
    abs_path = _resolve_source_path(source_name)
    nested = source_name in config.NESTED_SPECTRUM_SOURCES
    files = _list_spectrum_txt(abs_path, nested)
    target = None
    for f in files:
        if f.name == filename:
            target = f
            break
    if target is None:
        raise FileNotFoundError(f"Spectrum file '{filename}' not found in source '{source_name}'")

    from exploration.pyelli_exploration.pyelli_utils import load_measured_spectrum
    wavelengths, measured = load_measured_spectrum(target)
    return {
        "filename": target.name,
        "wavelengths": wavelengths.tolist(),
        "measured": measured.tolist(),
    }
