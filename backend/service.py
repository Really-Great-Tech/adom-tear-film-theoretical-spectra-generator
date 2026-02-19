"""
PyElli grid search and spectrum service. Single place that calls PyElliGridSearch.
Separation of concerns: routes handle HTTP, service handles domain logic.
"""
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

# Backend only: heavy deps live here
from exploration.pyelli_exploration.pyelli_grid_search import PyElliGridSearch
from exploration.pyelli_exploration.run_last_two_cycles_seed_tune import run_last_two_cycles_seed_tune_sync

logger = logging.getLogger(__name__)


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
    """Run last-two-cycles seed+tune on a run folder. Returns dict with summary, seed_result, results."""
    run_dir = full_test_cycles_dir / run_folder_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")
    return run_last_two_cycles_seed_tune_sync(
        run_dir,
        materials_path,
        seed_index=seed_index,
        max_tune=max_tune,
    )
