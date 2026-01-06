"""
Interactive Streamlit + Plotly app for exploring tear film reflectance spectra.

Features:
- Sliders for lipid thickness, aqueous thickness, and roughness
- Measurement spectra overlay from specified directory
- Side-by-side comparison of measured vs theoretical spectra
- Fit metrics calculation
"""

from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Tuple, Optional
import itertools
import random
import os
import time
import warnings
import logging
from datetime import timedelta

# Suppress Streamlit widget warnings about default value + session state
warnings.filterwarnings(
    'ignore',
    message='.*widget.*key.*was created with a default value but also had its value set via the Session State API.*',
    category=UserWarning
)
# Also suppress via Streamlit logger if it's logged there
streamlit_logger = logging.getLogger('streamlit')
streamlit_logger.setLevel(logging.ERROR)  # Only show errors, not warnings

from analysis import (
    load_measurement_spectrum,
    detrend_signal,
    boxcar_smooth,
    gaussian_smooth,
    detect_peaks,
    detect_valleys,
    prepare_measurement,
    prepare_theoretical_spectrum,
    peak_count_score,
    peak_delta_score,
    phase_overlap_score,
    composite_score,
    score_spectrum,
    measurement_quality_score,
    SpectrumScore,
)

from tear_film_generator import (
    load_config,
    validate_config,
    make_single_spectrum_calculator,
    PROJECT_ROOT,
    get_project_path,
)

from pdf_report import generate_main_app_pdf_report


def clamp_to_step(value: float, min_val: float, step: float) -> float:
    """Snap a value to the nearest step from min_val."""
    return min_val + round((value - min_val) / step) * step


def detrend_dataframe(df: pd.DataFrame, cutoff_frequency: float, filter_order: int) -> pd.DataFrame:
    """Apply shared detrending routine and attach the result to the dataframe."""

    df = df.sort_values(by="wavelength").reset_index(drop=True)
    detrended = detrend_signal(
        df["wavelength"].to_numpy(),
        df["reflectance"].to_numpy(),
        cutoff_frequency,
        filter_order,
    )
    df_result = df.copy()
    df_result["detrended"] = detrended
    return df_result


def detect_peaks_df(df: pd.DataFrame, column: str, prominence: float, height: Optional[float] = None) -> pd.DataFrame:
    """Convenience wrapper using the shared peak detector."""

    peaks = detect_peaks(
        df["wavelength"].to_numpy(),
        df[column].to_numpy(),
        prominence=prominence,
        height=height,
    )
    result = pd.DataFrame({
        "wavelength": peaks["wavelength"],
        column: peaks["amplitude"],
        "peak_prominence": peaks["prominence"],
    })
    return result


def detect_valleys_df(df: pd.DataFrame, column: str, prominence: float) -> pd.DataFrame:
    """Convenience wrapper for detecting valleys."""

    valleys = detect_valleys(
        df["wavelength"].to_numpy(),
        df[column].to_numpy(),
        prominence=prominence,
    )
    result = pd.DataFrame({
        "wavelength": valleys["wavelength"],
        column: valleys["amplitude"],
        "valley_prominence": valleys["prominence"],
    })
    return result


def apply_smoothing(
    df: pd.DataFrame,
    smoothing_type: str,
    boxcar_width_nm: float = 17.0,
    boxcar_passes: int = 1,
    gaussian_kernel: int = 11,
) -> pd.DataFrame:
    """Apply smoothing to a spectrum DataFrame.

    Args:
        df: DataFrame with 'wavelength' and 'reflectance' columns.
        smoothing_type: One of 'none', 'boxcar', or 'gaussian'.
        boxcar_width_nm: Boxcar smoothing width in nanometers.
        boxcar_passes: Number of boxcar smoothing passes.
        gaussian_kernel: Gaussian kernel size in samples.

    Returns:
        DataFrame with smoothed 'reflectance' column (original preserved as 'reflectance_raw').
    """
    if smoothing_type == "none":
        return df

    df_result = df.copy()
    wavelengths = df_result["wavelength"].to_numpy()
    reflectance = df_result["reflectance"].to_numpy()

    # Preserve original reflectance
    df_result["reflectance_raw"] = reflectance

    if smoothing_type == "boxcar":
        smoothed = boxcar_smooth(reflectance, wavelengths, boxcar_width_nm, boxcar_passes)
    elif smoothing_type == "gaussian":
        smoothed = gaussian_smooth(reflectance, gaussian_kernel)
    else:
        smoothed = reflectance

    df_result["reflectance"] = smoothed
    return df_result


def load_bestfit_spectrum(file_path: pathlib.Path) -> pd.DataFrame:
    """Load BestFit theoretical spectrum file.
    
    Args:
        file_path: Path to the BestFit spectrum file
        
    Returns:
        DataFrame with 'wavelength' and 'reflectance' columns
    """
    wavelengths = []
    reflectances = []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            # Skip header line
            if line.startswith('BestFit') or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    wavelengths.append(float(parts[0]))
                    reflectances.append(float(parts[1]))
                except ValueError:
                    continue
    
    return pd.DataFrame({
        'wavelength': wavelengths,
        'reflectance': reflectances
    }).sort_values('wavelength').reset_index(drop=True)


def load_measurement_files(measurements_dir: pathlib.Path, config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Load measurement spectra using the shared loader.
    
    Loads from:
    - exploration/sample_data/good_fit/ (Silas's good fit samples)
    - exploration/sample_data/bad_fit/ (Silas's bad fit samples)
    - spectra_from_shlomo/ (Shlomo's raw spectra samples)
    
    Explicitly skips:
    - _BestFit.txt files (pre-computed fits, not measurements)
    - readme/documentation files
    - Files that don't exist (silently skipped, no error)
    """

    measurements: Dict[str, pd.DataFrame] = {}
    meas_config = config.get("measurements", {})
    file_pattern = meas_config.get("file_pattern", "*.txt")
    
    # Search directories and their display prefixes
    exploration_dir = PROJECT_ROOT / "exploration" / "sample_data"
    shlomo_dir = PROJECT_ROOT / "spectra_from_shlomo"
    search_dirs = []
    dir_prefixes: Dict[pathlib.Path, str] = {}
    
    # Load from good_fit and bad_fit subdirectories
    for subdir_name in ["good_fit", "bad_fit"]:
        subdir = exploration_dir / subdir_name
        if subdir.exists() and subdir.is_dir():
            search_dirs.append(subdir)
            dir_prefixes[subdir] = ""  # Use relative path from exploration_dir
    
    # Also load from spectra_from_shlomo/ (Shlomo's raw spectra)
    if shlomo_dir.exists() and shlomo_dir.is_dir():
        search_dirs.append(shlomo_dir)
        dir_prefixes[shlomo_dir] = "shlomo"  # Prefix with "shlomo/" for display
    
    if not search_dirs:
        # Only show warning if neither directory exists
        if not exploration_dir.exists() and not shlomo_dir.exists():
            st.warning(f"No measurement directories found")
        return measurements
    
    all_file_path_objs = []
    for search_dir in search_dirs:
        # Normalize the base directory path for Windows UNC paths
        try:
            search_dir_normalized = pathlib.Path(os.path.normpath(str(search_dir.resolve())))
        except (OSError, ValueError):
            # Skip directories that can't be resolved (e.g., network issues)
                continue

        # Skip non-spectrum files (BestFit, readme, documentation, etc.)
        def should_skip_file(file_path: pathlib.Path) -> bool:
            """Check if file should be skipped (not a measurement spectrum file)."""
            name_lower = file_path.name.lower()
            
            # Always skip BestFit files (these are pre-computed theoretical fits, not measurements)
            if "_bestfit" in name_lower or name_lower.endswith("_bestfit.txt"):
                return True
            
            # Skip readme files (case-insensitive)
            if "readme" in name_lower:
                return True
            
            # Skip documentation files
            if name_lower.endswith((".md", ".pdf")):
                return True
            
            return False
        
        try:
            file_path_objs = [
                p for p in search_dir_normalized.rglob(file_pattern)
                if p.is_file() and p.exists() and not should_skip_file(p)
            ]
            all_file_path_objs.extend(file_path_objs)
        except (OSError, PermissionError):
            # Skip directories that can't be accessed
                continue

    if not all_file_path_objs:
        st.warning(f"No measurement files found matching pattern: {file_pattern}")
        return measurements

    for file_path_obj in sorted(all_file_path_objs):
        try:
            # Normalize path for Windows UNC paths (handle backslash issues)
            normalized_str = os.path.normpath(str(file_path_obj))
            file_path_obj = pathlib.Path(normalized_str)
            
            # Check file exists before trying to load (silently skip if not)
            if not file_path_obj.exists():
                continue

            # Skip BestFit and readme files (double-check here too)
            name_lower = file_path_obj.name.lower()
            if "_bestfit" in name_lower or "readme" in name_lower:
                continue

            # Determine display name based on which directory the file is from
            file_name = None
            base_dir_normalized = pathlib.Path(os.path.normpath(str(exploration_dir.resolve())))
            shlomo_dir_normalized = pathlib.Path(os.path.normpath(str(shlomo_dir.resolve())))
            
            try:
                rel_path = file_path_obj.relative_to(base_dir_normalized)
                file_name = str(rel_path.with_suffix(""))  # Remove .txt extension
            except ValueError:
                try:
                    rel_path = file_path_obj.relative_to(shlomo_dir_normalized)
                    file_name = "shlomo/" + str(rel_path.with_suffix(""))  # Prefix with shlomo/
                except ValueError:
                    continue  # File is not from either directory

            if file_name is None:
                continue

            meas_df = load_measurement_spectrum(file_path_obj, meas_config)
            if not meas_df.empty:
                measurements[file_name] = meas_df
            # Silently skip files that don't contain spectral data (no error message)
        except FileNotFoundError:
            # File doesn't exist - silently skip (no warning)
            continue
        except Exception as exc:  # pragma: no cover - UI warning path
            # Only show warning for unexpected errors, not file not found or "no spectral data"
            error_str = str(exc).lower()
            if ("no such file" not in error_str and 
                "file not found" not in error_str and
                "no spectral data" not in error_str and 
                "readme" not in file_path_obj.name.lower()):
                st.warning(f"Error loading {file_path_obj}: {exc}")

    return measurements


def flag_edge_cases(
    results_df: pd.DataFrame,
    config: Dict[str, Any],
    threshold_high_score: float = 0.9,
    threshold_low_score: float = 0.3,
    threshold_no_fit: float = 0.4,
    acceptable_ranges: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """Flag edge cases in grid search results.
    
    Edge cases include:
    1. Parameters OUTSIDE acceptable/feasible ranges (e.g., lipid=600nm when acceptable max is 500nm)
    2. Exceptionally high scores (potential overfitting or exceptional performance)
    3. Exceptionally low scores (poor fits)
    4. "No good fit" - best score is below threshold (suggests measurement may not be valid)
    
    Note: We do NOT flag parameters at search boundaries - the coarse-fine search expands beyond
    configured ranges during refinement, so boundary values are expected and valid.
    
    Returns DataFrame with 'edge_case_flag' and 'edge_case_reason' columns.
    """
    if results_df.empty:
        return results_df
    
    # Get acceptable/feasible ranges (wider than search ranges)
    if acceptable_ranges is None:
        acceptable_ranges = {
            "lipid": {"min": 9, "max": 250},
            "aqueous": {"min": 800, "max": 12000},
            "roughness": {"min": 600, "max": 2750},
        }
    
    accept_lipid_min = float(acceptable_ranges.get("lipid", {}).get("min", 9))
    accept_lipid_max = float(acceptable_ranges.get("lipid", {}).get("max", 250))
    accept_aqueous_min = float(acceptable_ranges.get("aqueous", {}).get("min", 800))
    accept_aqueous_max = float(acceptable_ranges.get("aqueous", {}).get("max", 12000))
    accept_rough_min = float(acceptable_ranges.get("roughness", {}).get("min", 600))
    accept_rough_max = float(acceptable_ranges.get("roughness", {}).get("max", 2750))
    
    # Check for "no good fit" scenario (best score is too low)
    best_score = float(results_df["score_composite"].max()) if "score_composite" in results_df.columns else 0.0
    no_good_fit = best_score <= threshold_no_fit
    
    flags = []
    reasons = []
    
    for idx, row in results_df.iterrows():
        edge_case_reasons = []
        is_edge_case = False
        
        lipid = float(row.get("lipid_nm", 0))
        aqueous = float(row.get("aqueous_nm", 0))
        rough = float(row.get("roughness_A", 0))
        composite_score = float(row.get("score_composite", 0))
        
        # Check 1: Parameters OUTSIDE acceptable/feasible ranges (CRITICAL - suggests invalid result)
        if lipid < accept_lipid_min or lipid > accept_lipid_max:
            edge_case_reasons.append(f"lipid_outside_acceptable_range({lipid:.0f}nm, acceptable: {accept_lipid_min:.0f}-{accept_lipid_max:.0f}nm)")
            is_edge_case = True
        
        if aqueous < accept_aqueous_min or aqueous > accept_aqueous_max:
            edge_case_reasons.append(f"aqueous_outside_acceptable_range({aqueous:.0f}nm, acceptable: {accept_aqueous_min:.0f}-{accept_aqueous_max:.0f}nm)")
            is_edge_case = True
        
        if rough < accept_rough_min or rough > accept_rough_max:
            edge_case_reasons.append(f"roughness_outside_acceptable_range({rough:.0f}Å, acceptable: {accept_rough_min:.0f}-{accept_rough_max:.0f}Å)")
            is_edge_case = True
        
        # Check 2: Exceptionally high scores (algorithm outperforming)
        if composite_score >= threshold_high_score:
            edge_case_reasons.append("exceptional_score")
            is_edge_case = True
        
        # Check 3: Exceptionally low scores (poor fit)
        if composite_score <= threshold_low_score:
            edge_case_reasons.append("poor_fit")
            is_edge_case = True
        
        # Check 4: "No good fit" - if this is the best result and score is too low
        if idx == results_df.index[0] and no_good_fit:  # First row is best (sorted by score)
            edge_case_reasons.append("no_good_fit_found")
            is_edge_case = True
        
        flags.append(is_edge_case)
        reasons.append("; ".join(edge_case_reasons) if edge_case_reasons else "")
    
    results_df = results_df.copy()
    results_df["edge_case_flag"] = flags
    results_df["edge_case_reason"] = reasons
    
    # Store "no good fit" info in a special column (we'll check this separately)
    results_df["_no_good_fit"] = no_good_fit
    results_df["_best_score"] = best_score
    
    return results_df


def generate_parameter_values(cfg: Dict[str, Any], stride: int) -> np.ndarray:
    stride = max(1, stride)
    min_val = float(cfg["min"])
    max_val = float(cfg["max"])
    step = float(cfg["step"]) * stride
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


def generate_dynamic_parameter_values(
    cfg: Dict[str, Any],
    promising_regions: Optional[List[Tuple[float, float]]] = None,
    base_step: Optional[float] = None,
    min_step: Optional[float] = None,
    max_evaluations: Optional[int] = None,
) -> np.ndarray:
    """Generate parameter values with dynamic step sizes.
    
    If promising_regions is provided, uses finer steps in those regions and coarser steps elsewhere.
    This adapts to the score landscape to preserve runtime while getting good results.
    
    Args:
        cfg: Parameter configuration dict with min, max, step
        promising_regions: List of (center, score) tuples indicating promising parameter values
        base_step: Base step size (defaults to cfg["step"])
        min_step: Minimum step size for refinement (defaults to base_step / 4)
        max_evaluations: Maximum number of values to generate (for runtime control)
    
    Returns:
        Array of parameter values with adaptive step sizes
    """
    min_val = float(cfg["min"])
    max_val = float(cfg["max"])
    base_step = base_step if base_step is not None else float(cfg["step"])
    min_step = min_step if min_step is not None else base_step / 4.0
    
    # If no promising regions, use uniform coarse steps
    if not promising_regions or len(promising_regions) == 0:
        return generate_parameter_values(cfg, stride=1)
    
    # Sort promising regions by score (highest first)
    promising_regions = sorted(promising_regions, key=lambda x: x[1], reverse=True)
    
    # Collect all values with adaptive step sizes
    all_values = set()
    
    # Add boundary values
    all_values.add(min_val)
    all_values.add(max_val)
    
    # For each promising region, add fine-grained values around it
    for center, score in promising_regions:
        # Determine step size based on score (higher score = finer step)
        if score >= 0.8:
            step = min_step
            window = base_step * 8  # Wider window for high-scoring regions
        elif score >= 0.5:
            step = base_step / 2.0
            window = base_step * 6
        else:
            step = base_step
            window = base_step * 4
        
        # Clamp center to valid range
        center = max(min_val, min(max_val, center))
        
        # Generate values in window around center
        region_min = max(min_val, center - window / 2)
        region_max = min(max_val, center + window / 2)
        
        region_values = np.arange(region_min, region_max + step * 0.5, step, dtype=float)
        for val in region_values:
            if min_val <= val <= max_val:
                all_values.add(val)
    
    # Fill gaps between promising regions with coarse steps
    sorted_centers = sorted([r[0] for r in promising_regions])
    for i in range(len(sorted_centers) - 1):
        gap_start = sorted_centers[i]
        gap_end = sorted_centers[i + 1]
        
        # Use coarse steps in gaps
        gap_step = base_step * 2
        gap_values = np.arange(gap_start, gap_end, gap_step, dtype=float)
        for val in gap_values:
            if min_val <= val <= max_val:
                all_values.add(val)
    
    # Fill before first and after last promising region
    if sorted_centers:
        if sorted_centers[0] > min_val:
            gap_values = np.arange(min_val, sorted_centers[0], base_step * 2, dtype=float)
            for val in gap_values:
                all_values.add(val)
        
        if sorted_centers[-1] < max_val:
            gap_values = np.arange(sorted_centers[-1], max_val + base_step * 2, base_step * 2, dtype=float)
            for val in gap_values:
                if val <= max_val:
                    all_values.add(val)
    
    # Convert to sorted array
    values = np.array(sorted(all_values), dtype=float)
    
    # Limit to max_evaluations if specified (for runtime control)
    if max_evaluations and len(values) > max_evaluations:
        if max_evaluations >= 3:
            indices = np.linspace(0, len(values) - 1, max_evaluations, dtype=int)
            values = values[indices]
        else:
            values = np.array([min_val, max_val], dtype=float)
    
    return values


def score_candidate(
    measurement_features,
    theoretical_features,
    metrics_cfg: Dict[str, Any],
    lipid_nm: Optional[float] = None,
    aqueous_nm: Optional[float] = None,
    roughness_A: Optional[float] = None,
    measurement_quality: Optional[Any] = None,
    previous_params: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Score a candidate spectrum using all available metrics.
    
    Uses the unified score_spectrum function which handles:
    - peak_count, peak_delta, phase_overlap (original metrics)
    - residual (new: RMSE/MAE/R² fit quality)
    - quality (new: measurement quality validation)
    - temporal_continuity (new: for time-series analysis)
    """
    spectrum_score = score_spectrum(
        measurement_features,
        theoretical_features,
        metrics_cfg,
        lipid_nm=lipid_nm,
        aqueous_nm=aqueous_nm,
        roughness_A=roughness_A,
        measurement_quality=measurement_quality,
        previous_params=previous_params,
    )
    
    # Convert SpectrumScore to the format expected by the UI
    component_scores = spectrum_score.scores
    diagnostics = spectrum_score.diagnostics
    
    return component_scores, diagnostics


def run_coarse_fine_grid_search(
    single_spectrum,
    wavelengths: np.ndarray,
    measurement_features,
    analysis_cfg: Dict[str, Any],
    metrics_cfg: Dict[str, Any],
    config: Dict[str, Any],
    max_results: Optional[int],
    *,
    measurement_quality=None,
) -> pd.DataFrame:
    """Run a two-stage coarse-to-fine grid search.
    
    Stage 1: Coarse search with wider parameter ranges (from config.grid_search.coarse)
    Stage 2: Refine around top-K results from coarse stage
    
    During refinement, the search expands beyond the configured parameter ranges (from config.parameters)
    up to the acceptable ranges (from config.analysis.edge_case_detection.acceptable_ranges).
    This allows finding optimal values that may be outside the initial search range but still
    within physically/biologically feasible limits.
    """
    grid_cfg = analysis_cfg.get("grid_search", {})
    coarse_cfg = grid_cfg.get("coarse", {})
    refine_cfg = grid_cfg.get("refine", {})
    top_k_coarse = int(coarse_cfg.get("top_k", 20))
    
    # Stage 1: Coarse search
    coarse_lipid_cfg = coarse_cfg.get("lipid", config["parameters"]["lipid"])
    coarse_aqueous_cfg = coarse_cfg.get("aqueous", config["parameters"]["aqueous"])
    coarse_rough_cfg = coarse_cfg.get("roughness", config["parameters"]["roughness"])
    
    coarse_lipid_vals = generate_parameter_values(coarse_lipid_cfg, stride=1)
    coarse_aqueous_vals = generate_parameter_values(coarse_aqueous_cfg, stride=1)
    coarse_rough_vals = generate_parameter_values(coarse_rough_cfg, stride=1)
    
    st.info(f"Stage 1 (Coarse): Evaluating {len(coarse_lipid_vals) * len(coarse_aqueous_vals) * len(coarse_rough_vals)} combinations...")
    
    coarse_records: List[Dict[str, float]] = []
    for lipid in coarse_lipid_vals:
        for aqueous in coarse_aqueous_vals:
            for rough in coarse_rough_vals:
                spectrum = single_spectrum(float(lipid), float(aqueous), float(rough))
                if spectrum is None or len(spectrum) == 0 or np.all(spectrum == 0):
                    continue
                spectrum_std = np.std(spectrum)
                if spectrum_std < 1e-6:
                    continue
                theoretical = prepare_theoretical_spectrum(
                    wavelengths,
                    spectrum,
                    measurement_features,
                    analysis_cfg,
                )
                scores, diagnostics = score_candidate(
                    measurement_features,
                    theoretical,
                    metrics_cfg,
                    lipid_nm=float(lipid),
                    aqueous_nm=float(aqueous),
                    roughness_A=float(rough),
                    measurement_quality=measurement_quality,
                )
                record = {
                    "lipid_nm": float(lipid),
                    "aqueous_nm": float(aqueous),
                    "roughness_A": float(rough),
                }
                for key, value in scores.items():
                    record[f"score_{key}"] = float(value)
                for metric, diag in diagnostics.items():
                    for diag_key, diag_val in diag.items():
                        record[f"{metric}_{diag_key}"] = float(diag_val)
                coarse_records.append(record)
    
    if not coarse_records:
        return pd.DataFrame(), 0
    
    coarse_df = pd.DataFrame(coarse_records)
    coarse_df = coarse_df.sort_values("score_composite", ascending=False).reset_index(drop=True)
    top_coarse = coarse_df.head(top_k_coarse)
    
    st.success(f"Stage 1 complete: Found {len(coarse_records)} candidates. Top {top_k_coarse} selected for refinement.")
    
    # Stage 2: Refine around top-K coarse results
    refine_records: List[Dict[str, float]] = []
    max_refine = refine_cfg.get("max_refine_candidates", 5000)
    refine_budget = max_results - len(coarse_records) if max_results else max_refine
    
    st.info(f"Stage 2 (Refine): Exploring around {len(top_coarse)} best candidates...")
    
    for _, row in top_coarse.iterrows():
        center_lipid = row["lipid_nm"]
        center_aqueous = row["aqueous_nm"]
        center_rough = row["roughness_A"]
        
        # Build refinement window - expand beyond configured ranges, but clamp to acceptable ranges
        lipid_window = refine_cfg.get("lipid", {}).get("window_nm", 20)
        lipid_step = refine_cfg.get("lipid", {}).get("step_nm", 5)
        aqueous_window = refine_cfg.get("aqueous", {}).get("window_nm", 40)
        aqueous_step = refine_cfg.get("aqueous", {}).get("step_nm", 10)
        rough_window = refine_cfg.get("roughness", {}).get("window_A", 200)
        rough_step = refine_cfg.get("roughness", {}).get("step_A", 25)
        
        # Get acceptable ranges for clamping (wider than configured search ranges)
        edge_case_cfg = analysis_cfg.get("edge_case_detection", {})
        acceptable_ranges = edge_case_cfg.get("acceptable_ranges", {})
        accept_lipid_min = float(acceptable_ranges.get("lipid", {}).get("min", 9))
        accept_lipid_max = float(acceptable_ranges.get("lipid", {}).get("max", 250))
        accept_aqueous_min = float(acceptable_ranges.get("aqueous", {}).get("min", 800))
        accept_aqueous_max = float(acceptable_ranges.get("aqueous", {}).get("max", 12000))
        accept_rough_min = float(acceptable_ranges.get("roughness", {}).get("min", 600))
        accept_rough_max = float(acceptable_ranges.get("roughness", {}).get("max", 2750))
        
        # Expand refinement window beyond configured ranges, but clamp to acceptable ranges
        # This allows finding optimal values outside the initial search range
        lipid_min = max(accept_lipid_min, center_lipid - lipid_window / 2)
        lipid_max = min(accept_lipid_max, center_lipid + lipid_window / 2)
        aqueous_min = max(accept_aqueous_min, center_aqueous - aqueous_window / 2)
        aqueous_max = min(accept_aqueous_max, center_aqueous + aqueous_window / 2)
        rough_min = max(accept_rough_min, center_rough - rough_window / 2)
        rough_max = min(accept_rough_max, center_rough + rough_window / 2)
        
        refine_lipid_vals = np.arange(lipid_min, lipid_max + lipid_step * 0.5, lipid_step)
        refine_aqueous_vals = np.arange(aqueous_min, aqueous_max + aqueous_step * 0.5, aqueous_step)
        refine_rough_vals = np.arange(rough_min, rough_max + rough_step * 0.5, rough_step)
        
        # Ensure max values are included
        if len(refine_lipid_vals) > 0 and refine_lipid_vals[-1] < lipid_max - 1e-9:
            refine_lipid_vals = np.append(refine_lipid_vals, lipid_max)
        if len(refine_aqueous_vals) > 0 and refine_aqueous_vals[-1] < aqueous_max - 1e-9:
            refine_aqueous_vals = np.append(refine_aqueous_vals, aqueous_max)
        if len(refine_rough_vals) > 0 and refine_rough_vals[-1] < rough_max - 1e-9:
            refine_rough_vals = np.append(refine_rough_vals, rough_max)
        
        for lipid in refine_lipid_vals:
            for aqueous in refine_aqueous_vals:
                for rough in refine_rough_vals:
                    if refine_budget is not None and len(refine_records) >= refine_budget:
                        break
                    spectrum = single_spectrum(float(lipid), float(aqueous), float(rough))
                    if spectrum is None or len(spectrum) == 0 or np.all(spectrum == 0):
                        continue
                    spectrum_std = np.std(spectrum)
                    if spectrum_std < 1e-6:
                        continue
                    theoretical = prepare_theoretical_spectrum(
                        wavelengths,
                        spectrum,
                        measurement_features,
                        analysis_cfg,
                    )
                    scores, diagnostics = score_candidate(
                        measurement_features,
                        theoretical,
                        metrics_cfg,
                        lipid_nm=float(lipid),
                        aqueous_nm=float(aqueous),
                        roughness_A=float(rough),
                        measurement_quality=measurement_quality,
                    )
                    record = {
                        "lipid_nm": float(lipid),
                        "aqueous_nm": float(aqueous),
                        "roughness_A": float(rough),
                    }
                    for key, value in scores.items():
                        record[f"score_{key}"] = float(value)
                    for metric, diag in diagnostics.items():
                        for diag_key, diag_val in diag.items():
                            record[f"{metric}_{diag_key}"] = float(diag_val)
                    refine_records.append(record)
                if refine_budget is not None and len(refine_records) >= refine_budget:
                    break
            if refine_budget is not None and len(refine_records) >= refine_budget:
                break
        if refine_budget is not None and len(refine_records) >= refine_budget:
            break
    
    # Combine coarse and refine results, remove duplicates, and sort
    all_records = coarse_records + refine_records
    if not all_records:
        return pd.DataFrame(), 0
    
    results_df = pd.DataFrame(all_records)
    # Remove duplicates based on parameters (keep first occurrence)
    results_df = results_df.drop_duplicates(subset=["lipid_nm", "aqueous_nm", "roughness_A"], keep="first")
    results_df = results_df.sort_values("score_composite", ascending=False).reset_index(drop=True)
    
    total_evaluated = len(coarse_records) + len(refine_records)
    st.success(f"Stage 2 complete: Refined {len(refine_records)} additional candidates. Total: {total_evaluated} evaluated.")
    
    return results_df, total_evaluated


def run_inline_grid_search(
    single_spectrum,
    wavelengths: np.ndarray,
    measurement_features,
    analysis_cfg: Dict[str, Any],
    metrics_cfg: Dict[str, Any],
    lipid_vals: np.ndarray,
    aqueous_vals: np.ndarray,
    rough_vals: np.ndarray,
    max_results: Optional[int],
    *,
    measurement_quality=None,
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    evaluated = 0
    
    # Calculate total search space
    total_combinations = len(lipid_vals) * len(aqueous_vals) * len(rough_vals)
    
    # Use random sampling if max_results is set and covers less than 10% of search space
    # This prevents bias toward lower parameter values
    use_random_sampling = max_results is not None and max_results > 0 and max_results < total_combinations * 0.1
    
    if use_random_sampling:
        # Generate all combinations and randomly sample
        all_combinations = list(itertools.product(lipid_vals, aqueous_vals, rough_vals))
        # Use fixed seed for reproducibility
        random.seed(42)
        sampled_combinations = random.sample(all_combinations, min(max_results, len(all_combinations)))
        
        for lipid, aqueous, rough in sampled_combinations:
            spectrum = single_spectrum(float(lipid), float(aqueous), float(rough))
            # Validate spectrum is not flat/zero
            if spectrum is None or len(spectrum) == 0 or np.all(spectrum == 0):
                # Skip invalid spectra
                continue
            # Check if spectrum has variation (before detrending)
            spectrum_std = np.std(spectrum)
            if spectrum_std < 1e-6:
                # Skip flat spectra
                continue
            theoretical = prepare_theoretical_spectrum(
                wavelengths,
                spectrum,
                measurement_features,
                analysis_cfg,
            )
            scores, diagnostics = score_candidate(
                measurement_features,
                theoretical,
                metrics_cfg,
                lipid_nm=float(lipid),
                aqueous_nm=float(aqueous),
                roughness_A=float(rough),
                measurement_quality=measurement_quality,
            )
            record = {
                "lipid_nm": float(lipid),
                "aqueous_nm": float(aqueous),
                "roughness_A": float(rough),
            }
            for key, value in scores.items():
                record[f"score_{key}"] = float(value)
            for metric, diag in diagnostics.items():
                for diag_key, diag_val in diag.items():
                    record[f"{metric}_{diag_key}"] = float(diag_val)
            records.append(record)
            evaluated += 1
    else:
        # Use systematic grid search (original nested loop approach)
        for lipid in lipid_vals:
            for aqueous in aqueous_vals:
                for rough in rough_vals:
                    if max_results is not None and evaluated >= max_results:
                        break
                    spectrum = single_spectrum(float(lipid), float(aqueous), float(rough))
                    # Validate spectrum is not flat/zero
                    if spectrum is None or len(spectrum) == 0 or np.all(spectrum == 0):
                        # Skip invalid spectra
                        continue
                    # Check if spectrum has variation (before detrending)
                    spectrum_std = np.std(spectrum)
                    if spectrum_std < 1e-6:
                        # Skip flat spectra
                        continue
                    theoretical = prepare_theoretical_spectrum(
                        wavelengths,
                        spectrum,
                        measurement_features,
                        analysis_cfg,
                    )
                    scores, diagnostics = score_candidate(
                        measurement_features,
                        theoretical,
                        metrics_cfg,
                        lipid_nm=float(lipid),
                        aqueous_nm=float(aqueous),
                        roughness_A=float(rough),
                        measurement_quality=measurement_quality,
                    )
                    record = {
                        "lipid_nm": float(lipid),
                        "aqueous_nm": float(aqueous),
                        "roughness_A": float(rough),
                    }
                    for key, value in scores.items():
                        record[f"score_{key}"] = float(value)
                    for metric, diag in diagnostics.items():
                        for diag_key, diag_val in diag.items():
                            record[f"{metric}_{diag_key}"] = float(diag_val)
                    records.append(record)
                    evaluated += 1
                if max_results is not None and evaluated >= max_results:
                    break
            if max_results is not None and evaluated >= max_results:
                break

    if not records:
        return pd.DataFrame(), 0

    results_df = pd.DataFrame(records)
    results_df = results_df.sort_values("score_composite", ascending=False).reset_index(drop=True)
    return results_df, evaluated


def interpolate_measurement_to_theoretical(measured_df: pd.DataFrame, theoretical_wavelengths: np.ndarray) -> np.ndarray:
    """Interpolate measured spectrum to match theoretical wavelength grid."""
    return np.interp(theoretical_wavelengths, measured_df['wavelength'], measured_df['reflectance'])


def calculate_fit_metrics(measured: np.ndarray, theoretical: np.ndarray) -> Dict[str, float]:
    """Calculate goodness-of-fit metrics between measured and theoretical spectra."""
    # R-squared
    ss_res = np.sum((measured - theoretical) ** 2)
    ss_tot = np.sum((measured - np.mean(measured)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((measured - theoretical) ** 2))
    
    # Mean Absolute Error
    mae = np.mean(np.abs(measured - theoretical))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((measured - theoretical) / measured)) * 100 if np.mean(measured) != 0 else 0
    
    return {
        'R²': r_squared,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE (%)': mape
    }


def create_comparison_plot(theoretical_wl: np.ndarray, theoretical_spec: np.ndarray,
                          measured_df: pd.DataFrame, lipid_val: float, aqueous_val: float, 
                          rough_val: float, config: Dict[str, Any], selected_file: str,
                          bestfit_df: Optional[pd.DataFrame] = None,
                          show_bestfit: bool = False,
                          show_both_theoretical: bool = False) -> go.Figure:
    """Create a plot comparing theoretical and measured spectra.
    
    Args:
        theoretical_wl: Wavelength array for theoretical spectrum
        theoretical_spec: Theoretical spectrum values
        measured_df: Measured spectrum DataFrame
        lipid_val, aqueous_val, rough_val: Parameter values for display
        config: Configuration dictionary
        selected_file: Selected measurement file name
        bestfit_df: Optional BestFit spectrum DataFrame
        show_bestfit: Whether to show BestFit instead of theoretical
        show_both_theoretical: Whether to show both theoretical and BestFit
    """
    plot_config = config.get('plotting', {})
    style = plot_config.get('plot_style', {})
    
    # Get wavelength range of interest from config
    wavelength_range_cfg = config.get("analysis", {}).get("wavelength_range", {})
    wl_min = float(wavelength_range_cfg.get("min", 600))
    wl_max = float(wavelength_range_cfg.get("max", 1120))
    
    # Filter data to wavelength range for proper y-axis auto-scaling
    meas_mask = (measured_df['wavelength'] >= wl_min) & (measured_df['wavelength'] <= wl_max)
    meas_wl_filtered = measured_df.loc[meas_mask, 'wavelength']
    meas_refl_filtered = measured_df.loc[meas_mask, 'reflectance']
    
    theo_mask = (theoretical_wl >= wl_min) & (theoretical_wl <= wl_max)
    theo_wl_filtered = theoretical_wl[theo_mask]
    theo_spec_filtered = theoretical_spec[theo_mask]
    
    fig = go.Figure()
    
    # Add measured spectrum (filtered to wavelength range)
    fig.add_trace(go.Scatter(
        x=meas_wl_filtered,
        y=meas_refl_filtered,
        mode='lines',
        name=f'Measured ({selected_file})',
        line=dict(
            color=style.get('measured_color', 'red'),
            width=style.get('line_width', 2)
        ),
        hovertemplate='λ=%{x:.1f}nm<br>R=%{y:.4f}<br>Measured<extra></extra>'
    ))
    
    # Add theoretical spectrum(s) based on toggle
    if show_both_theoretical and bestfit_df is not None:
        # Filter bestfit data
        bf_mask = (bestfit_df['wavelength'] >= wl_min) & (bestfit_df['wavelength'] <= wl_max)
        bf_wl_filtered = bestfit_df.loc[bf_mask, 'wavelength']
        bf_refl_filtered = bestfit_df.loc[bf_mask, 'reflectance']
        
        # Show both LTA theoretical and BestFit
        fig.add_trace(go.Scatter(
            x=theo_wl_filtered,
            y=theo_spec_filtered,
            mode='lines',
            name=f'LTA Theoretical (L={lipid_val:.0f}, A={aqueous_val:.0f}, R={rough_val:.0f})',
            line=dict(
                color=style.get('theoretical_color', 'blue'),
                width=style.get('line_width', 2),
                dash='dash'
            ),
            hovertemplate='λ=%{x:.1f}nm<br>R=%{y:.4f}<br>LTA Theoretical<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=bf_wl_filtered,
            y=bf_refl_filtered,
            mode='lines',
            name='LTA BestFit',
            line=dict(
                color='#db2777',
                width=style.get('line_width', 2),
                dash='dot'
            ),
            hovertemplate='λ=%{x:.1f}nm<br>R=%{y:.4f}<br>LTA BestFit<extra></extra>'
        ))
    elif show_bestfit and bestfit_df is not None:
        # Filter bestfit data
        bf_mask = (bestfit_df['wavelength'] >= wl_min) & (bestfit_df['wavelength'] <= wl_max)
        bf_wl_filtered = bestfit_df.loc[bf_mask, 'wavelength']
        bf_refl_filtered = bestfit_df.loc[bf_mask, 'reflectance']
        
        # Show only BestFit
        fig.add_trace(go.Scatter(
            x=bf_wl_filtered,
            y=bf_refl_filtered,
            mode='lines',
            name='LTA BestFit',
            line=dict(
                color='#db2777',
                width=style.get('line_width', 2),
                dash='dash'
            ),
            hovertemplate='λ=%{x:.1f}nm<br>R=%{y:.4f}<br>LTA BestFit<extra></extra>'
        ))
    else:
        # Show only LTA theoretical (default)
        fig.add_trace(go.Scatter(
            x=theo_wl_filtered,
            y=theo_spec_filtered,
            mode='lines',
            name=f'Theoretical (L={lipid_val:.0f}, A={aqueous_val:.0f}, R={rough_val:.0f})',
            line=dict(
                color=style.get('theoretical_color', 'blue'),
                width=style.get('line_width', 2)
            ),
            hovertemplate='λ=%{x:.1f}nm<br>R=%{y:.4f}<br>Theoretical<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"Measured vs Theoretical Reflectance Spectra ({wl_min:.0f}-{wl_max:.0f}nm)",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Reflectance",
        xaxis_range=[wl_min, wl_max],  # Zoom to wavelength range of interest
        hovermode="x unified",
        template="plotly_white",
        width=style.get('width', 1000),
        height=style.get('height', 600),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def main():
    config = load_config(PROJECT_ROOT / "config.yaml")
    if not validate_config(config):
        st.error("Invalid configuration. See terminal for details.")
        st.stop()

    analysis_cfg: Dict[str, Any] = config.get("analysis", {})
    analysis_cfg.setdefault("detrending", {})
    analysis_cfg.setdefault("peak_detection", {})
    metrics_cfg: Dict[str, Any] = analysis_cfg.get("metrics", {})
    ui_cfg: Dict[str, Any] = config.get("ui", {})
    st.set_page_config(
        page_title=ui_cfg.get("page_title", "Tear Film Spectra Explorer"), 
        layout="wide"
    )
    
    # Hide Streamlit widget warnings via CSS and JavaScript
    st.markdown('''
    <style>
        /* Hide Streamlit widget warnings about session state */
        div[data-testid="stAlert"]:has(> div:contains("widget")) {
            display: none !important;
        }
        
        /* Additional CSS to target specific warning text if needed */
        div[data-testid="stAlert"]:has(> div:contains("lipid_slider")) {
            display: none !important;
        }
        div[data-testid="stAlert"]:has(> div:contains("aqueous_slider")) {
            display: none !important;
        }
        div[data-testid="stAlert"]:has(> div:contains("rough_slider")) {
            display: none !important;
        }
        div[data-testid="stAlert"]:has(> div:contains("Session State API")) {
            display: none !important;
        }
    </style>
    <script>
        // Hide Streamlit widget warnings about session state
        function hideWidgetWarnings() {
            const alerts = document.querySelectorAll('[data-testid="stAlert"]');
            alerts.forEach(alert => {
                const text = alert.textContent || alert.innerText;
                if (text.includes('widget') && text.includes('key') && text.includes('Session State API')) {
                    alert.style.display = 'none';
                }
            });
        }
        
        // Run on page load and after Streamlit updates
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', hideWidgetWarnings);
        } else {
            hideWidgetWarnings();
        }
        
        // Also hide after Streamlit reruns (using MutationObserver for continuous monitoring)
        const observer = new MutationObserver(hideWidgetWarnings);
        observer.observe(document.body, { childList: true, subtree: true });

        // Fallback for older Streamlit versions or if MutationObserver is not enough
        setInterval(hideWidgetWarnings, 500); // Check every 500ms
    </script>
    ''', unsafe_allow_html=True)

    # Load theoretical spectrum calculator
    single_spectrum, wavelengths = make_single_spectrum_calculator(config)

    # Load measurement files
    measurements_enabled = config.get('measurements', {}).get('enabled', False)
    measurements = {}
    
    if measurements_enabled:
        measurements_dir = get_project_path(config['paths']['measurements'])
        measurements = load_measurement_files(measurements_dir, config)

    params = config["parameters"]
    lipid_cfg = params["lipid"]
    aqueous_cfg = params["aqueous"]
    rough_cfg = params["roughness"]
    
    # Slider limits use accepted ranges (lipid: 9-250, aqueous: 800-12000, roughness: 600-2750)
    slider_lipid_min = 9.0
    slider_lipid_max = 250.0
    slider_aqueous_min = 800.0
    slider_aqueous_max = 12000.0
    slider_rough_min = 600.0
    slider_rough_max = 2750.0

    # Defaults: use configured defaults if provided, or midpoints of slider ranges
    defaults = ui_cfg.get("default_values", {})
    def slider_midpoint(min_val, max_val, step):
        return clamp_to_step((min_val + max_val) / 2, min_val, step)

    default_lipid = max(slider_lipid_min, min(slider_lipid_max, defaults.get("lipid", slider_midpoint(slider_lipid_min, slider_lipid_max, lipid_cfg["step"]))))
    default_aqueous = max(slider_aqueous_min, min(slider_aqueous_max, defaults.get("aqueous", slider_midpoint(slider_aqueous_min, slider_aqueous_max, aqueous_cfg["step"]))))
    default_rough = max(slider_rough_min, min(slider_rough_max, defaults.get("roughness", slider_midpoint(slider_rough_min, slider_rough_max, rough_cfg["step"]))))

    # Initialize session state defaults for sliders and analysis controls so they can be reset later
    slider_defaults = {
        "lipid_slider": float(default_lipid),
        "aqueous_slider": float(default_aqueous),
        "rough_slider": float(default_rough),
    }

    # Main content
    st.markdown("# Tear Film Spectra Explorer")
    st.markdown("Adjust layer properties to view theoretical reflectance spectrum and compare with measurements.")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Spectrum Comparison",
        "📈 Spectrum Analysis",
        "⚙️ Parameters",
        "🔍 Grid Search",
    ])
    
    # Detrending parameters - read defaults from config
    detrending_cfg = analysis_cfg.get("detrending", {})
    peak_detection_cfg = analysis_cfg.get("peak_detection", {})
    smoothing_cfg = analysis_cfg.get("smoothing", {})
    default_cutoff = float(detrending_cfg.get("default_cutoff_frequency", 0.01))
    default_prominence = float(peak_detection_cfg.get("default_prominence", 0.005))
    
    # Smoothing defaults from config
    default_smoothing_type = smoothing_cfg.get("default_type", "none")
    boxcar_cfg = smoothing_cfg.get("boxcar", {})
    gaussian_cfg = smoothing_cfg.get("gaussian", {})
    default_boxcar_width = float(boxcar_cfg.get("default_width_nm", 17.0))
    default_boxcar_passes = int(boxcar_cfg.get("default_passes", 1))
    default_gaussian_kernel = int(gaussian_cfg.get("default_kernel_size", 11))
    apply_smoothing_after_detrend = smoothing_cfg.get("apply_after_detrend", True)
    
    analysis_defaults = {
        "analysis_cutoff_freq": default_cutoff,
        "analysis_peak_prominence": default_prominence,
        "cutoff_freq_slider": default_cutoff,
        "peak_prominence_slider": default_prominence,
        "smoothing_type": default_smoothing_type,
        "boxcar_width_nm": default_boxcar_width,
        "boxcar_passes": default_boxcar_passes,
        "gaussian_kernel": default_gaussian_kernel,
    }

    # Initialize session state with config defaults if not set
    for key, value in {**slider_defaults, **analysis_defaults}.items():
        st.session_state.setdefault(key, value)
    
    # Update sliders if a grid-search selection was applied in the previous run
    pending_update = st.session_state.pop("pending_slider_update", None)
    if pending_update:
        # Clamp values to slider ranges (acceptable ranges) to prevent errors
        if "lipid_slider" in pending_update:
            pending_update["lipid_slider"] = max(slider_lipid_min, min(slider_lipid_max, pending_update["lipid_slider"]))
        if "aqueous_slider" in pending_update:
            pending_update["aqueous_slider"] = max(slider_aqueous_min, min(slider_aqueous_max, pending_update["aqueous_slider"]))
        if "rough_slider" in pending_update:
            pending_update["rough_slider"] = max(slider_rough_min, min(slider_rough_max, pending_update["rough_slider"]))
        for slider_key, slider_value in pending_update.items():
            st.session_state[slider_key] = slider_value
    
    # Clamp any existing session_state values to valid ranges (acceptable ranges)
    if "lipid_slider" in st.session_state:
        st.session_state["lipid_slider"] = max(slider_lipid_min, min(slider_lipid_max, st.session_state["lipid_slider"]))
    if "aqueous_slider" in st.session_state:
        st.session_state["aqueous_slider"] = max(slider_aqueous_min, min(slider_aqueous_max, st.session_state["aqueous_slider"]))
    if "rough_slider" in st.session_state:
        st.session_state["rough_slider"] = max(slider_rough_min, min(slider_rough_max, st.session_state["rough_slider"]))
    
    # Sidebar controls
    st.sidebar.markdown("## Layer Parameters")

    if st.sidebar.button("Reset parameters", type="secondary", use_container_width=True):
        reset_values = {**slider_defaults, **analysis_defaults}
        st.session_state.update(reset_values)
        analysis_cfg["detrending"]["default_cutoff_frequency"] = default_cutoff
        analysis_cfg["peak_detection"]["default_prominence"] = default_prominence
        st.rerun()
    
    lipid_val = st.sidebar.slider(
        "Lipid thickness (nm)",
        min_value=slider_lipid_min,
        max_value=slider_lipid_max,
        step=float(lipid_cfg["step"]),
        format="%.0f",
        key="lipid_slider",
    )

    aqueous_val = st.sidebar.slider(
        "Aqueous thickness (nm)",
        min_value=slider_aqueous_min,
        max_value=slider_aqueous_max,
        step=float(aqueous_cfg["step"]),
        format="%.0f",
        key="aqueous_slider",
    )

    rough_val = st.sidebar.slider(
        "Mucus roughness (Å)",
        min_value=slider_rough_min,
        max_value=slider_rough_max,
        step=float(rough_cfg["step"]),
        format="%.0f",
        key="rough_slider",
    )

    # Measurement file selection
    selected_file = None
    if measurements_enabled and measurements:
        st.sidebar.markdown("## Measurement Data")
        measurement_files = list(measurements.keys())
        selected_file = st.sidebar.selectbox(
            "Select measurement file:",
            options=["None"] + measurement_files,
            index=1 if measurement_files else 0
        )

        if selected_file != "None":
            selected_measurement = measurements[selected_file]
            st.sidebar.write(f"**{selected_file}**")
            st.sidebar.write(f"Data points: {len(selected_measurement)}")
            st.sidebar.write(f"Wavelength range: {selected_measurement['wavelength'].min():.1f} - {selected_measurement['wavelength'].max():.1f} nm")
            
            # Check for corresponding BestFit file
            # Reconstruct file path from selected_file key
            bestfit_file_path = None
            exploration_dir = PROJECT_ROOT / "exploration" / "sample_data"
            shlomo_dir = PROJECT_ROOT / "spectra_from_shlomo"
            
            # Construct the measurement file path from selected_file key
            if selected_file.startswith("shlomo/"):
                # Remove "shlomo/" prefix and add .txt
                meas_file_path = shlomo_dir / (selected_file[7:] + ".txt")  # Remove "shlomo/" prefix (7 chars)
            else:
                # Add .txt to the selected_file key
                meas_file_path = exploration_dir / (selected_file + ".txt")
            
            # Check if measurement file exists and find corresponding BestFit
            if meas_file_path.exists():
                bestfit_name = meas_file_path.name.replace('.txt', '_BestFit.txt')
                bestfit_path = meas_file_path.parent / bestfit_name
                if bestfit_path.exists():
                    bestfit_file_path = bestfit_path
            
            # Toggle to show LTA BestFit
            show_bestfit = False
            show_both_theoretical = False
            if bestfit_file_path:
                st.sidebar.markdown("---")
                st.sidebar.markdown("### 🔬 LTA BestFit Comparison")
                view_mode = st.sidebar.radio(
                    'Theoretical Spectrum View',
                    ['LTA Theoretical Only', 'LTA BestFit Only', 'Both (LTA + BestFit)'],
                    key='theoretical_view_mode',
                    help='Compare LTA-generated theoretical spectra with LTA BestFit results'
                )
                show_bestfit = view_mode in ['LTA BestFit Only', 'Both (LTA + BestFit)']
                show_both_theoretical = view_mode == 'Both (LTA + BestFit)'
                st.session_state.show_bestfit = show_bestfit
                st.session_state.show_both_theoretical = show_both_theoretical
                st.session_state.bestfit_file_path = str(bestfit_file_path)
            else:
                st.session_state.show_bestfit = False
                st.session_state.show_both_theoretical = False
                st.session_state.bestfit_file_path = None
        else:
            st.session_state.show_bestfit = False
            st.session_state.show_both_theoretical = False
            st.session_state.bestfit_file_path = None
    
    st.sidebar.markdown("## Analysis Parameters")
    
    # Sliders that don't trigger immediate reload
    cutoff_freq_input = st.sidebar.slider(
        "Detrending Cutoff Frequency", 
        0.001, 0.1, 
        value=st.session_state["analysis_cutoff_freq"], 
        step=0.001, 
        format="%.3f",
        key="cutoff_freq_slider"
    )
    peak_prominence_input = st.sidebar.slider(
        "Peak Prominence", 
        0.0001, 0.05, 
        value=st.session_state["analysis_peak_prominence"], 
        step=0.001, 
        format="%.5f",
        key="peak_prominence_slider"
    )
    
    # Smoothing controls
    st.sidebar.markdown("---")
    smoothing_type_input = st.sidebar.radio(
        "Smoothing Type",
        options=["none", "boxcar", "gaussian"],
        index=["none", "boxcar", "gaussian"].index(st.session_state.get("smoothing_type", "none")),
        horizontal=True,
        key="smoothing_type_radio"
    )
    
    # Conditional smoothing parameters based on type
    boxcar_width_input = default_boxcar_width
    boxcar_passes_input = default_boxcar_passes
    gaussian_kernel_input = default_gaussian_kernel
    
    if smoothing_type_input == "boxcar":
        boxcar_width_input = st.sidebar.slider(
            "Boxcar Width (nm)",
            min_value=5.0,
            max_value=50.0,
            value=st.session_state.get("boxcar_width_nm", default_boxcar_width),
            step=1.0,
            format="%.0f",
            key="boxcar_width_slider"
        )
        boxcar_passes_input = st.sidebar.selectbox(
            "Boxcar Passes",
            options=[1, 2, 3],
            index=[1, 2, 3].index(st.session_state.get("boxcar_passes", default_boxcar_passes)),
            key="boxcar_passes_select"
        )
    elif smoothing_type_input == "gaussian":
        gaussian_kernel_input = st.sidebar.selectbox(
            "Gaussian Kernel Size",
            options=[7, 9, 11],
            index=[7, 9, 11].index(st.session_state.get("gaussian_kernel", default_gaussian_kernel)),
            key="gaussian_kernel_select"
        )
    
    # Apply button to update all parameters at once
    apply_analysis_params = st.sidebar.button("Apply Analysis Parameters", type="primary", use_container_width=True)
    
    if apply_analysis_params:
        st.session_state["analysis_cutoff_freq"] = cutoff_freq_input
        st.session_state["analysis_peak_prominence"] = peak_prominence_input
        st.session_state["smoothing_type"] = smoothing_type_input
        st.session_state["boxcar_width_nm"] = boxcar_width_input
        st.session_state["boxcar_passes"] = boxcar_passes_input
        st.session_state["gaussian_kernel"] = gaussian_kernel_input
        # Update analysis_cfg dynamically for this session
        analysis_cfg["detrending"]["default_cutoff_frequency"] = cutoff_freq_input
        analysis_cfg["peak_detection"]["default_prominence"] = peak_prominence_input
        st.rerun()
    
    # Use session state values (these are the active values)
    cutoff_freq = st.session_state["analysis_cutoff_freq"]
    peak_prominence = st.session_state["analysis_peak_prominence"]
    smoothing_type = st.session_state.get("smoothing_type", "none")
    boxcar_width_nm = st.session_state.get("boxcar_width_nm", default_boxcar_width)
    boxcar_passes = st.session_state.get("boxcar_passes", default_boxcar_passes)
    gaussian_kernel = st.session_state.get("gaussian_kernel", default_gaussian_kernel)
    
    # Show current active values
    smoothing_info = f", Smoothing={smoothing_type}" if smoothing_type != "none" else ""
    st.sidebar.caption(f"Active: Cutoff={cutoff_freq:.3f}, Prominence={peak_prominence:.5f}{smoothing_info}")

    # Compute theoretical spectrum
    spectrum = single_spectrum(lipid_val, aqueous_val, rough_val)
    theoretical_df = pd.DataFrame({
        'wavelength': wavelengths,
        'reflectance': spectrum
    })

    # Tab 1: Spectrum Comparison
    with tab1:
        if measurements_enabled and selected_file and selected_file != "None":
            selected_measurement = measurements[selected_file].copy()
            
            # Apply smoothing to measured spectrum if enabled
            if smoothing_type != "none":
                selected_measurement = apply_smoothing(
                    selected_measurement, smoothing_type,
                    boxcar_width_nm, boxcar_passes, gaussian_kernel
                )
            
            # Load BestFit if available
            bestfit_df = None
            show_bestfit_tab = st.session_state.get('show_bestfit', False)
            show_both_theoretical_tab = st.session_state.get('show_both_theoretical', False)
            bestfit_file_path_str = st.session_state.get('bestfit_file_path', None)
            
            if bestfit_file_path_str and pathlib.Path(bestfit_file_path_str).exists():
                try:
                    bestfit_df = load_bestfit_spectrum(pathlib.Path(bestfit_file_path_str))
                except Exception as e:
                    st.warning(f"Could not load BestFit file: {e}")
                    bestfit_df = None
            
            # Create comparison plot
            fig = create_comparison_plot(
                wavelengths, spectrum, selected_measurement,
                lipid_val, aqueous_val, rough_val, config, selected_file,
                bestfit_df=bestfit_df,
                show_bestfit=show_bestfit_tab,
                show_both_theoretical=show_both_theoretical_tab
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display fit metrics
            interpolated_measured = interpolate_measurement_to_theoretical(selected_measurement, wavelengths)
            
            # Determine which theoretical to use for metrics
            if show_bestfit_tab and not show_both_theoretical_tab and bestfit_df is not None:
                # Use BestFit for metrics
                bestfit_interp = interpolate_measurement_to_theoretical(bestfit_df, wavelengths)
                metrics = calculate_fit_metrics(interpolated_measured, bestfit_interp)
                theoretical_for_residual = bestfit_interp
                residual_label = 'Residuals (Measured - BestFit)'
            else:
                # Use LTA theoretical for metrics (or when showing both)
                metrics = calculate_fit_metrics(interpolated_measured, spectrum)
                theoretical_for_residual = spectrum
                residual_label = 'Residuals (Measured - Theoretical)'
            
            # Show metrics (both if showing both)
            if show_both_theoretical_tab and bestfit_df is not None:
                # Calculate metrics for both
                metrics_lta = calculate_fit_metrics(interpolated_measured, spectrum)
                bestfit_interp = interpolate_measurement_to_theoretical(bestfit_df, wavelengths)
                metrics_bestfit = calculate_fit_metrics(interpolated_measured, bestfit_interp)
                
                st.markdown("## Goodness of Fit Metrics")
                st.markdown("### LTA Theoretical")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("R²", f"{metrics_lta['R²']:.4f}")
                with col2:
                    st.metric("RMSE", f"{metrics_lta['RMSE']:.6f}")
                with col3:
                    st.metric("MAE", f"{metrics_lta['MAE']:.6f}")
                with col4:
                    st.metric("MAPE", f"{metrics_lta['MAPE (%)']:.2f}%")
                
                st.markdown("### LTA BestFit")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("R²", f"{metrics_bestfit['R²']:.4f}")
                with col2:
                    st.metric("RMSE", f"{metrics_bestfit['RMSE']:.6f}")
                with col3:
                    st.metric("MAE", f"{metrics_bestfit['MAE']:.6f}")
                with col4:
                    st.metric("MAPE", f"{metrics_bestfit['MAPE (%)']:.2f}%")
            else:
                st.markdown("## Goodness of Fit Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("R²", f"{metrics['R²']:.4f}")
                with col2:
                    st.metric("RMSE", f"{metrics['RMSE']:.6f}")
                with col3:
                    st.metric("MAE", f"{metrics['MAE']:.6f}")
                with col4:
                    st.metric("MAPE", f"{metrics['MAPE (%)']:.2f}%")
            
            # Residuals plot
            if not show_both_theoretical_tab:
                interpolated_measured = interpolate_measurement_to_theoretical(selected_measurement, wavelengths)
                residuals = interpolated_measured - theoretical_for_residual
                fig_residuals = go.Figure()
                fig_residuals.add_trace(go.Scatter(
                    x=wavelengths,
                    y=residuals,
                    mode='lines',
                    name=residual_label,
                    line=dict(color='green', width=2)
                ))
                fig_residuals.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
                fig_residuals.update_layout(
                    title=residual_label,
                    xaxis_title="Wavelength (nm)",
                    yaxis_title="Residual Reflectance",
                    template="plotly_white",
                    height=400
                )
                st.plotly_chart(fig_residuals, use_container_width=True)
            
        else:
            # Show only theoretical spectrum
            fig = go.Figure()
            fig.add_scatter(
                x=wavelengths,
                y=spectrum,
                mode="lines",
                name=f"Theoretical (L={lipid_val:.0f}, A={aqueous_val:.0f}, R={rough_val:.0f})",
                line=dict(color=config.get('plotting', {}).get('plot_style', {}).get('theoretical_color', 'blue'))
            )
            fig.update_layout(
                title="Theoretical Reflectance Spectrum",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Reflectance",
                hovermode="x unified",
                template="plotly_white",
                margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Spectrum Analysis (zoomed view 600-1120nm with detrended signals)
    with tab2:
        # Get wavelength range from config
        wavelength_range_cfg = config.get("analysis", {}).get("wavelength_range", {})
        wl_min = float(wavelength_range_cfg.get("min", 600))
        wl_max = float(wavelength_range_cfg.get("max", 1120))
        
        st.markdown(f"## Spectrum Analysis ({wl_min:.0f}-{wl_max:.0f}nm)")
        st.caption("Detrended signals with peak and valley detection in the wavelength range of interest")
        
        if measurements_enabled and selected_file and selected_file != "None":
            selected_measurement = measurements[selected_file]
            
            # Load BestFit if available
            bestfit_df_tab2 = None
            show_bestfit_tab2 = st.session_state.get('show_bestfit', False)
            show_both_theoretical_tab2 = st.session_state.get('show_both_theoretical', False)
            bestfit_file_path_str_tab2 = st.session_state.get('bestfit_file_path', None)
            
            if bestfit_file_path_str_tab2 and pathlib.Path(bestfit_file_path_str_tab2).exists():
                try:
                    bestfit_df_tab2 = load_bestfit_spectrum(pathlib.Path(bestfit_file_path_str_tab2))
                except Exception as e:
                    st.warning(f"Could not load BestFit file: {e}")
                    bestfit_df_tab2 = None
            
            # Filter to wavelength range of interest
            filter_order = config.get("analysis", {}).get("detrending", {}).get("filter_order", 3)
            
            # Filter theoretical spectrum to wavelength range
            theoretical_filtered = theoretical_df[
                (theoretical_df['wavelength'] >= wl_min) & 
                (theoretical_df['wavelength'] <= wl_max)
            ].reset_index(drop=True)
            
            # Filter measured spectrum to wavelength range
            measured_filtered = selected_measurement[
                (selected_measurement['wavelength'] >= wl_min) & 
                (selected_measurement['wavelength'] <= wl_max)
            ].reset_index(drop=True)
            
            # Filter BestFit if available
            bestfit_filtered = None
            if bestfit_df_tab2 is not None:
                bestfit_filtered = bestfit_df_tab2[
                    (bestfit_df_tab2['wavelength'] >= wl_min) & 
                    (bestfit_df_tab2['wavelength'] <= wl_max)
                ].reset_index(drop=True)
            
            # Detrend and smooth signals based on toggle selection
            # Order is configurable: apply_smoothing_after_detrend (from config)
            if smoothing_type != "none" and not apply_smoothing_after_detrend:
                # Smooth BEFORE detrending
                measured_filtered = apply_smoothing(
                    measured_filtered, smoothing_type,
                    boxcar_width_nm, boxcar_passes, gaussian_kernel
                )
            
            measured_detrended = detrend_dataframe(measured_filtered, cutoff_freq, filter_order)
            
            if smoothing_type != "none" and apply_smoothing_after_detrend:
                # Smooth AFTER detrending (default behavior)
                wavelengths_arr = measured_detrended['wavelength'].to_numpy()
                detrended_arr = measured_detrended['detrended'].to_numpy()
                if smoothing_type == "boxcar":
                    smoothed_detrended = boxcar_smooth(detrended_arr, wavelengths_arr, boxcar_width_nm, boxcar_passes)
                else:  # gaussian
                    smoothed_detrended = gaussian_smooth(detrended_arr, gaussian_kernel)
                measured_detrended = measured_detrended.copy()
                measured_detrended['detrended'] = smoothed_detrended
            
            meas_peaks = detect_peaks_df(measured_detrended, 'detrended', peak_prominence)
            meas_valleys = detect_valleys_df(measured_detrended, 'detrended', peak_prominence)
            
            if show_both_theoretical_tab2 and bestfit_filtered is not None:
                # Detrend both LTA theoretical and BestFit
                theoretical_detrended = detrend_dataframe(theoretical_filtered, cutoff_freq, filter_order)
                bestfit_detrended = detrend_dataframe(bestfit_filtered, cutoff_freq, filter_order)
                theo_peaks = detect_peaks_df(theoretical_detrended, 'detrended', peak_prominence)
                theo_valleys = detect_valleys_df(theoretical_detrended, 'detrended', peak_prominence)
                bestfit_peaks = detect_peaks_df(bestfit_detrended, 'detrended', peak_prominence)
                bestfit_valleys = detect_valleys_df(bestfit_detrended, 'detrended', peak_prominence)
            elif show_bestfit_tab2 and bestfit_filtered is not None:
                # Use BestFit only
                bestfit_detrended = detrend_dataframe(bestfit_filtered, cutoff_freq, filter_order)
                bestfit_peaks = detect_peaks_df(bestfit_detrended, 'detrended', peak_prominence)
                bestfit_valleys = detect_valleys_df(bestfit_detrended, 'detrended', peak_prominence)
                # Use BestFit variables for display
                theoretical_detrended = bestfit_detrended
                theo_peaks = bestfit_peaks
                theo_valleys = bestfit_valleys
            else:
                # Use LTA theoretical only (default)
                theoretical_detrended = detrend_dataframe(theoretical_filtered, cutoff_freq, filter_order)
                theo_peaks = detect_peaks_df(theoretical_detrended, 'detrended', peak_prominence)
                theo_valleys = detect_valleys_df(theoretical_detrended, 'detrended', peak_prominence)
            
            # Create single zoomed plot showing detrended signals with peaks/valleys
            fig = go.Figure()
            
            # Measured detrended signal (always shown)
            fig.add_trace(go.Scatter(
                x=measured_detrended['wavelength'],
                y=measured_detrended['detrended'],
                mode='lines', name='Measured (Spectra)',
                line=dict(color='blue', width=2)
            ))
            
            # Theoretical detrended signal(s) based on toggle
            if show_both_theoretical_tab2 and bestfit_filtered is not None:
                # Show both LTA theoretical and BestFit
                fig.add_trace(go.Scatter(
                    x=theoretical_detrended['wavelength'],
                    y=theoretical_detrended['detrended'],
                    mode='lines', name='LTA Theoretical (Detrended)',
                    line=dict(color='green', width=2, dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=bestfit_detrended['wavelength'],
                    y=bestfit_detrended['detrended'],
                    mode='lines', name='LTA BestFit (Detrended)',
                    line=dict(color='#db2777', width=2, dash='dot')
                ))
            else:
                # Show single theoretical (LTA or BestFit)
                theo_name = 'LTA BestFit (Detrended)' if (show_bestfit_tab2 and bestfit_filtered is not None) else 'LTA Theoretical (Detrended)'
                theo_color = '#db2777' if (show_bestfit_tab2 and bestfit_filtered is not None) else 'red'
                fig.add_trace(go.Scatter(
                    x=theoretical_detrended['wavelength'],
                    y=theoretical_detrended['detrended'],
                    mode='lines', name=theo_name,
                    line=dict(color=theo_color, width=2, dash='dot')
                ))
            
            # Measured peaks
            if len(meas_peaks) > 0:
                fig.add_trace(go.Scatter(
                    x=meas_peaks['wavelength'],
                    y=meas_peaks['detrended'],
                    mode='markers', name='Spectra Peaks',
                    marker=dict(color='blue', size=8, symbol='circle')
                ))
            
            # Theoretical peaks
            if show_both_theoretical_tab2 and bestfit_filtered is not None:
                # Show peaks for both
                if len(theo_peaks) > 0:
                    fig.add_trace(go.Scatter(
                        x=theo_peaks['wavelength'],
                        y=theo_peaks['detrended'],
                        mode='markers', name='LTA Theoretical Peaks',
                        marker=dict(color='green', size=8, symbol='circle')
                    ))
                if len(bestfit_peaks) > 0:
                    fig.add_trace(go.Scatter(
                        x=bestfit_peaks['wavelength'],
                        y=bestfit_peaks['detrended'],
                        mode='markers', name='BestFit Peaks',
                        marker=dict(color='red', size=8, symbol='circle')
                    ))
            else:
                # Show single theoretical peaks
                if len(theo_peaks) > 0:
                    peak_name = 'BestFit Peaks' if (show_bestfit_tab2 and bestfit_filtered is not None) else 'LTA Theoretical Peaks'
                    peak_color = 'red' if (show_bestfit_tab2 and bestfit_filtered is not None) else 'red'
                    fig.add_trace(go.Scatter(
                        x=theo_peaks['wavelength'],
                        y=theo_peaks['detrended'],
                        mode='markers', name=peak_name,
                        marker=dict(color=peak_color, size=8, symbol='circle')
                    ))
            
            # Measured valleys
            if len(meas_valleys) > 0:
                fig.add_trace(go.Scatter(
                    x=meas_valleys['wavelength'],
                    y=meas_valleys['detrended'],
                    mode='markers', name='Spectra Valleys',
                    marker=dict(color='cyan', size=8, symbol='circle')
                ))
            
            # Theoretical valleys
            if show_both_theoretical_tab2 and bestfit_filtered is not None:
                # Show valleys for both
                if len(theo_valleys) > 0:
                    fig.add_trace(go.Scatter(
                        x=theo_valleys['wavelength'],
                        y=theo_valleys['detrended'],
                        mode='markers', name='LTA Theoretical Valleys',
                        marker=dict(color='lightgreen', size=8, symbol='circle')
                    ))
                if len(bestfit_valleys) > 0:
                    fig.add_trace(go.Scatter(
                        x=bestfit_valleys['wavelength'],
                        y=bestfit_valleys['detrended'],
                        mode='markers', name='BestFit Valleys',
                        marker=dict(color='pink', size=8, symbol='circle')
                    ))
            else:
                # Show single theoretical valleys
                if len(theo_valleys) > 0:
                    valley_name = 'BestFit Valleys' if (show_bestfit_tab2 and bestfit_filtered is not None) else 'LTA Theoretical Valleys'
                    valley_color = 'pink' if (show_bestfit_tab2 and bestfit_filtered is not None) else 'pink'
                    fig.add_trace(go.Scatter(
                        x=theo_valleys['wavelength'],
                        y=theo_valleys['detrended'],
                        mode='markers', name=valley_name,
                        marker=dict(color=valley_color, size=8, symbol='circle')
                    ))
            
            fig.update_layout(
                title=f"Peak and Valley Detection for {selected_file}",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Intensity (Detrended)",
                xaxis_range=[wl_min, wl_max],
                hovermode="x unified",
                template="plotly_white",
                height=600,
                legend=dict(
                    yanchor="top", y=0.99,
                    xanchor="right", x=0.99
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Peak analysis summary
            if show_both_theoretical_tab2 and bestfit_filtered is not None:
                # Show analysis for all three (measured, LTA theoretical, BestFit)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("### Measured Spectrum")
                    st.write(f"**Peaks:** {len(meas_peaks)}")
                    if len(meas_peaks) > 0:
                        peak_wls = [f"{w:.1f}nm" for w in meas_peaks['wavelength']]
                        st.write("Wavelengths:", ", ".join(peak_wls[:5]) + ("..." if len(peak_wls) > 5 else ""))
                    st.write(f"**Valleys:** {len(meas_valleys)}")
                    if len(meas_valleys) > 0:
                        valley_wls = [f"{w:.1f}nm" for w in meas_valleys['wavelength']]
                        st.write("Wavelengths:", ", ".join(valley_wls[:5]) + ("..." if len(valley_wls) > 5 else ""))
                with col2:
                    st.markdown("### LTA Theoretical")
                    st.write(f"**Peaks:** {len(theo_peaks)}")
                    if len(theo_peaks) > 0:
                        peak_wls = [f"{w:.1f}nm" for w in theo_peaks['wavelength']]
                        st.write("Wavelengths:", ", ".join(peak_wls[:5]) + ("..." if len(peak_wls) > 5 else ""))
                    st.write(f"**Valleys:** {len(theo_valleys)}")
                    if len(theo_valleys) > 0:
                        valley_wls = [f"{w:.1f}nm" for w in theo_valleys['wavelength']]
                        st.write("Wavelengths:", ", ".join(valley_wls[:5]) + ("..." if len(valley_wls) > 5 else ""))
                with col3:
                    st.markdown("### LTA BestFit")
                    st.write(f"**Peaks:** {len(bestfit_peaks)}")
                    if len(bestfit_peaks) > 0:
                        peak_wls = [f"{w:.1f}nm" for w in bestfit_peaks['wavelength']]
                        st.write("Wavelengths:", ", ".join(peak_wls[:5]) + ("..." if len(peak_wls) > 5 else ""))
                    st.write(f"**Valleys:** {len(bestfit_valleys)}")
                    if len(bestfit_valleys) > 0:
                        valley_wls = [f"{w:.1f}nm" for w in bestfit_valleys['wavelength']]
                        st.write("Wavelengths:", ", ".join(valley_wls[:5]) + ("..." if len(valley_wls) > 5 else ""))
            else:
                # Show analysis for measured and single theoretical
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Measured Spectrum Analysis")
                    st.write(f"**Peaks detected:** {len(meas_peaks)}")
                    if len(meas_peaks) > 0:
                        peak_wls = [f"{w:.1f}nm" for w in meas_peaks['wavelength']]
                        st.write("Peak wavelengths:", peak_wls)
                    st.write(f"**Valleys detected:** {len(meas_valleys)}")
                    if len(meas_valleys) > 0:
                        valley_wls = [f"{w:.1f}nm" for w in meas_valleys['wavelength']]
                        st.write("Valley wavelengths:", valley_wls)
                
                with col2:
                    theo_label = "LTA BestFit" if (show_bestfit_tab2 and bestfit_filtered is not None) else "LTA Theoretical"
                    st.markdown(f"### {theo_label} Analysis")
                    st.write(f"**Peaks detected:** {len(theo_peaks)}")
                    if len(theo_peaks) > 0:
                        peak_wls = [f"{w:.1f}nm" for w in theo_peaks['wavelength']]
                        st.write("Peak wavelengths:", peak_wls)
                    st.write(f"**Valleys detected:** {len(theo_valleys)}")
                    if len(theo_valleys) > 0:
                        valley_wls = [f"{w:.1f}nm" for w in theo_valleys['wavelength']]
                        st.write("Valley wavelengths:", valley_wls)
            
        else:
            # Show only theoretical detrended (filtered to wavelength range)
            filter_order = config.get("analysis", {}).get("detrending", {}).get("filter_order", 3)
            
            theoretical_filtered = theoretical_df[
                (theoretical_df['wavelength'] >= wl_min) & 
                (theoretical_df['wavelength'] <= wl_max)
            ].reset_index(drop=True)
            
            theoretical_detrended = detrend_dataframe(theoretical_filtered, cutoff_freq, filter_order)
            theo_peaks = detect_peaks_df(theoretical_detrended, 'detrended', peak_prominence)
            theo_valleys = detect_valleys_df(theoretical_detrended, 'detrended', peak_prominence)
            
            fig = go.Figure()
            
            # Detrended signal
            fig.add_trace(go.Scatter(
                x=theoretical_detrended['wavelength'],
                y=theoretical_detrended['detrended'],
                mode='lines', name='Theoretical (Detrended)',
                line=dict(color='blue', width=2)
            ))
            
            # Peaks
            if len(theo_peaks) > 0:
                fig.add_trace(go.Scatter(
                    x=theo_peaks['wavelength'],
                    y=theo_peaks['detrended'],
                    mode='markers', name='Peaks',
                    marker=dict(color='blue', size=8, symbol='circle')
                ))
            
            # Valleys
            if len(theo_valleys) > 0:
                fig.add_trace(go.Scatter(
                    x=theo_valleys['wavelength'],
                    y=theo_valleys['detrended'],
                    mode='markers', name='Valleys',
                    marker=dict(color='cyan', size=8, symbol='circle')
                ))
            
            fig.update_layout(
                title=f"Theoretical Detrended Signal ({wl_min:.0f}-{wl_max:.0f}nm)",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Intensity (Detrended)",
                xaxis_range=[wl_min, wl_max],
                template="plotly_white",
                height=600,
                legend=dict(
                    yanchor="top", y=0.99,
                    xanchor="right", x=0.99
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Analysis Summary")
            st.write(f"**Peaks detected:** {len(theo_peaks)}")
            if len(theo_peaks) > 0:
                st.write("Peak wavelengths:", [f"{w:.1f}nm" for w in theo_peaks['wavelength']])
            st.write(f"**Valleys detected:** {len(theo_valleys)}")
            if len(theo_valleys) > 0:
                st.write("Valley wavelengths:", [f"{w:.1f}nm" for w in theo_valleys['wavelength']])

    # Tab 3: Parameters
    with tab3:
        st.markdown("## Current Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Layer Properties")
            st.json({
                "lipid_thickness_nm": float(lipid_val),
                "aqueous_thickness_nm": float(aqueous_val),
                "mucus_roughness_angstrom": float(rough_val),
                "mucus_thickness_nm": config.get('fixed', {}).get('mucus_thickness', 500),
            })
        
        with col2:
            st.markdown("### Analysis Settings")
            st.json({
                "detrending_cutoff_frequency": float(cutoff_freq),
                "peak_detection_prominence": float(peak_prominence),
                "selected_measurement": selected_file if selected_file and selected_file != "None" else None
            })

        # Measurement data info
        if measurements_enabled:
            st.markdown("### Available Measurement Files")
            if measurements:
                for filename, df in measurements.items():
                    st.write(f"**{filename}**: {len(df)} data points, λ = {df['wavelength'].min():.1f}-{df['wavelength'].max():.1f} nm")
            else:
                st.info(f"No measurement files found in: {get_project_path(config['paths']['measurements'])}")
        else:
            st.info("Measurement comparison is disabled. Enable in config.yaml under 'measurements.enabled' to compare with experimental data.")

    with tab4:
        st.markdown("## Grid Search Ranking")
        if not (measurements_enabled and selected_file and selected_file != "None"):
            st.info("Select a measurement spectrum to run the grid search.")
        else:
            selected_measurement = measurements[selected_file]
            controls = st.columns(3)
            top_k_display = int(
                controls[0].number_input("Top results", min_value=5, max_value=100, value=10, step=5)
            )
            stride = int(
                controls[1].number_input("Stride multiplier", min_value=1, max_value=10, value=1, step=1)
            )
            max_results_input = int(
                controls[2].number_input("Max spectra", min_value=0, value=500, step=50, 
                                        help="Set to 0 for full grid search (all combinations). Warning: ~36,941 combinations may take 15-30 minutes.")
            )
            max_results = None if max_results_input == 0 else max_results_input
            
            # Calculate total combinations for display
            lipid_vals = generate_parameter_values(config["parameters"]["lipid"], stride)
            aqueous_vals = generate_parameter_values(config["parameters"]["aqueous"], stride)
            rough_vals = generate_parameter_values(config["parameters"]["roughness"], stride)
            total_combinations = len(lipid_vals) * len(aqueous_vals) * len(rough_vals)
            
            # Show estimated time for full search
            if max_results is None:
                estimated_minutes = total_combinations * 0.025 / 60  # ~25ms per spectrum
                st.info(f"📊 Full grid search: {total_combinations:,} combinations. Estimated time: ~{estimated_minutes:.0f} minutes. "
                       f"Consider using the CLI (`run_grid_search.py`) for better progress tracking on long runs.")
            else:
                st.caption(f"Total parameter space: {total_combinations:,} combinations. "
                          f"Will evaluate: {max_results:,} ({100*max_results/total_combinations:.1f}% coverage)")

            # Search strategy selector
            search_strategy = st.selectbox(
                "Search Strategy",
                ["random", "coarse-fine", "dynamic"],
                index=1,  # Default to "coarse-fine" (recommended)
                help="random: Random sampling across full parameter space. coarse-fine: Two-stage search (coarse then refine around best results). dynamic: Adaptive step sizes based on score landscape (preserves runtime, gets good results)."
            )
            
            # Client Reference Scoring - compare with known client best fits
            client_ref_cfg = config.get("ui", {}).get("client_reference_scoring", {})
            client_ref_default_enabled = client_ref_cfg.get("enabled", False)
            
            with st.expander("⭐ Client Reference Scoring", expanded=client_ref_default_enabled):
                st.caption("Compare algorithm results with known client best fit parameters")
                
                enable_client_ref = st.checkbox(
                    "Enable client reference scoring",
                    value=client_ref_default_enabled,
                    key="enable_client_ref_scoring"
                )
                
                if enable_client_ref:
                    ref_cols = st.columns(3)
                    client_lipid = ref_cols[0].number_input(
                        "Client Lipid (nm)",
                        min_value=9.0,
                        max_value=250.0,
                        value=float(client_ref_cfg.get("default_lipid", 62.3)),
                        step=0.1,
                        key="client_ref_lipid"
                    )
                    client_aqueous = ref_cols[1].number_input(
                        "Client Aqueous (nm)",
                        min_value=800.0,
                        max_value=12000.0,
                        value=float(client_ref_cfg.get("default_aqueous", 800.0)),
                        step=0.1,
                        key="client_ref_aqueous"
                    )
                    client_roughness = ref_cols[2].number_input(
                        "Client Roughness (Å)",
                        min_value=600.0,
                        max_value=2750.0,
                        value=float(client_ref_cfg.get("default_roughness", 1200.0)),
                        step=1.0,
                        key="client_ref_roughness"
                    )
                else:
                    client_lipid = None
                    client_aqueous = None
                    client_roughness = None
            
            # Full Grid Search button
            col1, col2 = st.columns([1, 1])
            with col1:
                run_pressed = st.button("Run Grid Search", key="run_grid_search_button", use_container_width=True)
            with col2:
                full_search_pressed = st.button("🔍 Full Grid Search", key="full_grid_search_button", use_container_width=True,
                                                help=f"Run complete grid search: {total_combinations:,} combinations (~{int(total_combinations * 0.025 / 60)} minutes)")
            
            # Handle full grid search button - show confirmation dialog
            if full_search_pressed:
                st.session_state["show_full_search_confirm"] = True
            
            # Show confirmation dialog if requested
            if st.session_state.get("show_full_search_confirm", False):
                estimated_minutes = int(total_combinations * 0.025 / 60)
                with st.expander("⚠️ Full Grid Search Confirmation", expanded=True):
                    st.warning(
                        f"**This will evaluate all {total_combinations:,} parameter combinations.**\n\n"
                        f"**Estimated time:** ~{estimated_minutes} minutes\n"
                        f"**Strategy:** {search_strategy}\n\n"
                        f"This is a long-running operation. The Streamlit UI may appear frozen during execution. "
                        f"For better progress tracking, consider using the CLI: `python run_grid_search.py --measurement <file>`"
                    )
                    confirm_col1, confirm_col2 = st.columns([1, 1])
                    with confirm_col1:
                        confirm_full = st.button("✅ Confirm Full Grid Search", key="confirm_full_search", type="primary", use_container_width=True)
                    with confirm_col2:
                        cancel_full = st.button("❌ Cancel", key="cancel_full_search", use_container_width=True)
                    
                    if confirm_full:
                        st.session_state["show_full_search_confirm"] = False
                        st.session_state["confirm_full_search_executed"] = True
                        st.rerun()  # Rerun to execute with max_results=None
                    elif cancel_full:
                        st.session_state["show_full_search_confirm"] = False
                        st.info("Full grid search cancelled. Use 'Run Grid Search' with a limit, or click 'Full Grid Search' again to retry.")
            
            # Quality gates validation (informational only, doesn't block search)
            # Only show once per measurement file to avoid repetition
            quality_check_key = f"quality_check_{selected_file}"
            quality_cfg = analysis_cfg.get("quality_gates", {})
            quality_enabled = quality_cfg.get("enabled", False)
            
            # Only run quality check if we haven't shown it for this measurement yet
            if quality_enabled and not st.session_state.get(quality_check_key, False):
                with st.spinner("Validating measurement quality..."):
                    measurement_features_temp = prepare_measurement(selected_measurement, analysis_cfg)
                    quality_result, quality_failures = measurement_quality_score(
                        measurement_features_temp,
                        min_peaks=quality_cfg.get("min_peaks"),
                        min_signal_amplitude=quality_cfg.get("min_signal_amplitude"),
                        min_wavelength_span_nm=quality_cfg.get("min_wavelength_span_nm"),
                    )
                    if quality_failures:
                        # Only show as info/warning, don't block - some measurements may have fewer peaks but still be valid
                        min_quality = quality_cfg.get("min_quality_score", 0.5)
                        if quality_result.score < min_quality:
                            # Show detailed guidance with actual measurement values
                            fix_guidance = []
                            actual_peaks = int(quality_result.diagnostics.get("peak_count", 0))
                            actual_amplitude = quality_result.diagnostics.get("signal_amplitude", 0)
                            actual_span = quality_result.diagnostics.get("wavelength_span_nm", 0)
                            
                            if "min_peaks" in quality_failures:
                                fix_guidance.append(f"lower `min_peaks` from {quality_cfg.get('min_peaks', 3)} to ≤{actual_peaks} (measurement has {actual_peaks} peak{'s' if actual_peaks != 1 else ''})")
                            if "min_signal_amplitude" in quality_failures:
                                fix_guidance.append(f"lower `min_signal_amplitude` from {quality_cfg.get('min_signal_amplitude', 0.02):.3f} to ≤{actual_amplitude:.3f}")
                            if "min_wavelength_span_nm" in quality_failures:
                                fix_guidance.append(f"lower `min_wavelength_span_nm` from {quality_cfg.get('min_wavelength_span_nm', 150.0):.1f} to ≤{actual_span:.1f}")
                            
                            guidance_text = " To adjust: " + ", or ".join(fix_guidance) + " in `config.yaml` → `analysis.quality_gates`." if fix_guidance else ""
                            
                            st.warning(
                                f"⚠️ Measurement quality below threshold (score: {quality_result.score:.2f} < {min_quality:.2f}). "
                                f"Failed checks: {', '.join(quality_failures)}. "
                                f"Results may be less reliable.{guidance_text}"
                            )
                        else:
                            st.info(
                                f"ℹ️ Measurement quality: {', '.join(quality_failures)} failed, but score ({quality_result.score:.2f}) is acceptable. "
                                f"Proceeding with grid search."
                            )
                    else:
                        st.success(f"✓ Measurement quality validated (score: {quality_result.score:.2f})")
                    
                    # Mark as shown for this measurement
                    st.session_state[quality_check_key] = True
            
            cache_key = f"grid_search_{selected_file}_{search_strategy}"
            
            # Check if full search was confirmed and should override max_results and auto-run
            should_run_full_search = st.session_state.get("confirm_full_search_executed", False)
            if should_run_full_search:
                max_results = None
                run_pressed = True  # Auto-trigger the search
                st.session_state["confirm_full_search_executed"] = False  # Reset flag
            
            if run_pressed:
                start_time = time.time()
                with st.spinner("Scoring theoretical spectra..."):
                    measurement_features = prepare_measurement(selected_measurement, analysis_cfg)
                    quality_cfg = analysis_cfg.get("quality_gates", {})
                    measurement_quality_result = None
                    if quality_cfg:
                        measurement_quality_result, _ = measurement_quality_score(
                            measurement_features,
                            min_peaks=quality_cfg.get("min_peaks"),
                            min_signal_amplitude=quality_cfg.get("min_signal_amplitude"),
                            min_wavelength_span_nm=quality_cfg.get("min_wavelength_span_nm"),
                        )
                    
                    if search_strategy == "coarse-fine":
                        # Use coarse-to-fine workflow
                        results_df, evaluated = run_coarse_fine_grid_search(
                            single_spectrum,
                            wavelengths,
                            measurement_features,
                            analysis_cfg,
                            metrics_cfg,
                            config,
                            max_results,
                            measurement_quality=measurement_quality_result,
                        )
                        # For coarse-fine, we don't have the original parameter arrays
                        lipid_vals = None
                        aqueous_vals = None
                        rough_vals = None
                    elif search_strategy == "dynamic":
                        # Dynamic search: quick coarse pass, then adaptive refinement
                        st.info("Stage 1: Quick coarse scan to identify promising regions...")
                        
                        # Quick coarse pass with larger steps
                        coarse_lipid_vals = generate_parameter_values(config["parameters"]["lipid"], stride=3)
                        coarse_aqueous_vals = generate_parameter_values(config["parameters"]["aqueous"], stride=3)
                        coarse_rough_vals = generate_parameter_values(config["parameters"]["roughness"], stride=3)
                        
                        # Limit coarse pass to reasonable size for speed
                        max_coarse = min(500, len(coarse_lipid_vals) * len(coarse_aqueous_vals) * len(coarse_rough_vals))
                        coarse_records = []
                        coarse_count = 0
                        
                        for lipid in coarse_lipid_vals:
                            for aqueous in coarse_aqueous_vals:
                                for rough in coarse_rough_vals:
                                    if coarse_count >= max_coarse:
                                        break
                                    spectrum = single_spectrum(float(lipid), float(aqueous), float(rough))
                                    if spectrum is None or len(spectrum) == 0 or np.all(spectrum == 0):
                                        continue
                                    spectrum_std = np.std(spectrum)
                                    if spectrum_std < 1e-6:
                                        continue
                                    theoretical = prepare_theoretical_spectrum(
                                        wavelengths,
                                        spectrum,
                                        measurement_features,
                                        analysis_cfg,
                                    )
                                    scores, _ = score_candidate(
                                        measurement_features,
                                        theoretical,
                                        metrics_cfg,
                                        lipid_nm=float(lipid),
                                        aqueous_nm=float(aqueous),
                                        roughness_A=float(rough),
                                        measurement_quality=measurement_quality_result,
                                    )
                                    coarse_records.append({
                                        "lipid_nm": float(lipid),
                                        "aqueous_nm": float(aqueous),
                                        "roughness_A": float(rough),
                                        "score_composite": scores.get("composite", 0.0),
                                    })
                                    coarse_count += 1
                        
                        if len(coarse_records) == 0:
                            st.warning("No valid spectra in coarse pass. Falling back to random search.")
                            lipid_vals = generate_parameter_values(config["parameters"]["lipid"], stride)
                            aqueous_vals = generate_parameter_values(config["parameters"]["aqueous"], stride)
                            rough_vals = generate_parameter_values(config["parameters"]["roughness"], stride)
                            results_df, evaluated = run_inline_grid_search(
                                single_spectrum, wavelengths, measurement_features,
                                analysis_cfg, metrics_cfg,
                                lipid_vals, aqueous_vals, rough_vals,
                                max_results, measurement_quality=measurement_quality_result,
                            )
                        else:
                            # Get top promising results
                            coarse_df = pd.DataFrame(coarse_records)
                            top_k = min(10, len(coarse_df))
                            top_coarse = coarse_df.nlargest(top_k, "score_composite")
                            
                            # Extract promising regions for each parameter
                            lipid_promising = [(row["lipid_nm"], row["score_composite"]) for _, row in top_coarse.iterrows()]
                            aqueous_promising = [(row["aqueous_nm"], row["score_composite"]) for _, row in top_coarse.iterrows()]
                            rough_promising = [(row["roughness_A"], row["score_composite"]) for _, row in top_coarse.iterrows()]
                            
                            st.success(f"Stage 1 complete: Found {len(coarse_records)} candidates. Top {top_k} selected for adaptive refinement.")
                            st.info("Stage 2: Adaptive refinement with dynamic step sizes...")
                            
                            # Generate dynamic parameter values
                            max_per_param = int(np.sqrt(max_results)) if max_results else 50
                            lipid_vals = generate_dynamic_parameter_values(
                                config["parameters"]["lipid"],
                                promising_regions=lipid_promising,
                                max_evaluations=max_per_param,
                            )
                            aqueous_vals = generate_dynamic_parameter_values(
                                config["parameters"]["aqueous"],
                                promising_regions=aqueous_promising,
                                max_evaluations=max_per_param,
                            )
                            rough_vals = generate_dynamic_parameter_values(
                                config["parameters"]["roughness"],
                                promising_regions=rough_promising,
                                max_evaluations=max_per_param,
                            )
                            
                            # Run full grid search with dynamic values
                            results_df, evaluated = run_inline_grid_search(
                                single_spectrum,
                                wavelengths,
                                measurement_features,
                                analysis_cfg,
                                metrics_cfg,
                                lipid_vals,
                                aqueous_vals,
                                rough_vals,
                                max_results,
                                measurement_quality=measurement_quality_result,
                            )
                            
                            # Combine coarse and refined results for completeness
                            if not results_df.empty and len(coarse_records) > 0:
                                coarse_full_records = []
                                for record in coarse_records:
                                    spectrum = single_spectrum(float(record["lipid_nm"]), float(record["aqueous_nm"]), float(record["roughness_A"]))
                                    if spectrum is None or len(spectrum) == 0 or np.all(spectrum == 0):
                                        continue
                                    spectrum_std = np.std(spectrum)
                                    if spectrum_std < 1e-6:
                                        continue
                                    theoretical = prepare_theoretical_spectrum(
                                        wavelengths, spectrum, measurement_features, analysis_cfg,
                                    )
                                    scores, diagnostics = score_candidate(
                                        measurement_features, theoretical, metrics_cfg,
                                        lipid_nm=float(record["lipid_nm"]),
                                        aqueous_nm=float(record["aqueous_nm"]),
                                        roughness_A=float(record["roughness_A"]),
                                        measurement_quality=measurement_quality_result,
                                    )
                                    full_record = {
                                        "lipid_nm": record["lipid_nm"],
                                        "aqueous_nm": record["aqueous_nm"],
                                        "roughness_A": record["roughness_A"],
                                        **{f"score_{k}": v for k, v in scores.items()},
                                    }
                                    coarse_full_records.append(full_record)
                                
                                if coarse_full_records:
                                    coarse_full_df = pd.DataFrame(coarse_full_records)
                                    results_df = pd.concat([results_df, coarse_full_df], ignore_index=True)
                                    results_df = results_df.drop_duplicates(
                                        subset=["lipid_nm", "aqueous_nm", "roughness_A"], keep="first"
                                    )
                                    results_df = results_df.sort_values("score_composite", ascending=False).reset_index(drop=True)
                            
                            evaluated = len(results_df)
                    else:
                        # Use random or systematic grid search
                        lipid_vals = generate_parameter_values(config["parameters"]["lipid"], stride)
                        aqueous_vals = generate_parameter_values(config["parameters"]["aqueous"], stride)
                        rough_vals = generate_parameter_values(config["parameters"]["roughness"], stride)
                        results_df, evaluated = run_inline_grid_search(
                            single_spectrum,
                            wavelengths,
                            measurement_features,
                            analysis_cfg,
                            metrics_cfg,
                            lipid_vals,
                            aqueous_vals,
                            rough_vals,
                            max_results,
                            measurement_quality=measurement_quality_result,
                        )
                    
                    elapsed_time = time.time() - start_time
                    elapsed_timedelta = timedelta(seconds=int(elapsed_time))
                    
                    st.session_state[cache_key] = {
                        "results": results_df,
                        "evaluated": evaluated,
                        "stride": stride,
                        "max_results": max_results_input,
                        "search_strategy": search_strategy,
                        "lipid_vals": lipid_vals,
                        "aqueous_vals": aqueous_vals,
                        "rough_vals": rough_vals,
                        "elapsed_time_seconds": elapsed_time,
                        "elapsed_time_formatted": str(elapsed_timedelta),
                    }

            cache_entry = st.session_state.get(cache_key)
            if cache_entry:
                results_df = cache_entry["results"]
                evaluated = cache_entry.get("evaluated", len(results_df))
                if results_df.empty:
                    st.warning("No spectra were evaluated with the current settings.")
                else:
                    elapsed_time_str = cache_entry.get("elapsed_time_formatted", "N/A")
                    elapsed_seconds = cache_entry.get("elapsed_time_seconds", 0)
                    if elapsed_seconds > 0:
                        minutes = int(elapsed_seconds // 60)
                        seconds = int(elapsed_seconds % 60)
                        if minutes > 0:
                            time_str = f"{minutes}m {seconds}s"
                        else:
                            time_str = f"{seconds}s"
                        st.caption(
                            f"Evaluated {evaluated} spectra (stride ×{cache_entry.get('stride', stride)}) in {time_str}."
                        )
                    else:
                        st.caption(
                            f"Evaluated {evaluated} spectra (stride ×{cache_entry.get('stride', stride)})."
                        )
                    # Show score statistics for debugging
                    if "score_composite" in results_df.columns:
                        score_min = results_df["score_composite"].min()
                        score_max = results_df["score_composite"].max()
                        score_mean = results_df["score_composite"].mean()
                        st.caption(f"Composite scores: min={score_min:.4f}, max={score_max:.4f}, mean={score_mean:.4f}")
                    
                    # Show peak detection diagnostics
                    if "peak_count_measurement_peaks" in results_df.columns:
                        meas_peaks = results_df["peak_count_measurement_peaks"].iloc[0] if len(results_df) > 0 else 0
                        theo_peaks_col = "peak_count_theoretical_peaks"
                        if theo_peaks_col in results_df.columns:
                            theo_peaks_avg = results_df[theo_peaks_col].mean()
                            theo_peaks_max = results_df[theo_peaks_col].max()
                            st.caption(f"Peak detection: measurement={int(meas_peaks)} peaks, theoretical avg={theo_peaks_avg:.1f}, max={int(theo_peaks_max)}")
                            if theo_peaks_max == 0:
                                st.warning("⚠️ No peaks detected in theoretical spectra! Try lowering the prominence threshold in Analysis Parameters.")
                    
                    # Show parameter range diagnostics
                    if "lipid_nm" in results_df.columns:
                        lipid_min = results_df["lipid_nm"].min()
                        lipid_max = results_df["lipid_nm"].max()
                        lipid_unique = results_df["lipid_nm"].nunique()
                        aqueous_min = results_df["aqueous_nm"].min()
                        aqueous_max = results_df["aqueous_nm"].max()
                        aqueous_unique = results_df["aqueous_nm"].nunique()
                        rough_min = results_df["roughness_A"].min()
                        rough_max = results_df["roughness_A"].max()
                        rough_unique = results_df["roughness_A"].nunique()
                        st.caption(f"Parameter ranges explored: lipid={lipid_min:.0f}-{lipid_max:.0f} ({lipid_unique} values), "
                                 f"aqueous={aqueous_min:.0f}-{aqueous_max:.0f} ({aqueous_unique} values), "
                                 f"roughness={rough_min:.0f}-{rough_max:.0f} ({rough_unique} values)")
                        
                        # Warn if search space coverage is low (only for random/systematic search, not coarse-fine or dynamic)
                        cached_lipid_vals = cache_entry.get("lipid_vals")
                        cached_aqueous_vals = cache_entry.get("aqueous_vals")
                        cached_rough_vals = cache_entry.get("rough_vals")
                        search_strategy_cached = cache_entry.get("search_strategy", "random")
                        if (cached_lipid_vals is not None and cached_aqueous_vals is not None and cached_rough_vals is not None 
                            and search_strategy_cached not in ["coarse-fine", "dynamic"]):
                            total_combinations = len(cached_lipid_vals) * len(cached_aqueous_vals) * len(cached_rough_vals)
                            max_results_cached = cache_entry.get("max_results")
                            coverage = 100 * evaluated / total_combinations if total_combinations > 0 else 0
                            
                            # Only warn if coverage is low (< 10%) AND a limit was set
                            # If max_results is 0 or None, it's a full search (100% coverage is expected and good!)
                            # Dynamic search intentionally uses adaptive step sizes and doesn't need full coverage
                            if max_results_cached is not None and max_results_cached > 0 and coverage < 10.0:
                                st.warning(f"⚠️ Only {coverage:.1f}% of parameter space explored ({evaluated}/{total_combinations} combinations). "
                                         f"Grid search may be biased toward lower parameter values. "
                                         f"Consider increasing 'Max spectra', using a larger 'Stride multiplier', or trying 'coarse-fine' or 'dynamic' strategy.")
                            elif coverage >= 99.0:
                                # Full or near-full coverage - show success message
                                st.success(f"✓ Full parameter space explored: {evaluated:,}/{total_combinations:,} combinations ({coverage:.1f}% coverage)")
                    
                    # Flag edge cases (if enabled)
                    edge_case_cfg = analysis_cfg.get("edge_case_detection", {})
                    if edge_case_cfg.get("enabled", True):
                        threshold_high = float(edge_case_cfg.get("threshold_high_score", 0.9))
                        threshold_low = float(edge_case_cfg.get("threshold_low_score", 0.3))
                        threshold_no_fit = float(edge_case_cfg.get("threshold_no_fit", 0.4))
                        acceptable_ranges = edge_case_cfg.get("acceptable_ranges", None)
                        results_df = flag_edge_cases(
                            results_df, 
                            config, 
                            threshold_high, 
                            threshold_low,
                            threshold_no_fit,
                            acceptable_ranges
                        )
                        
                        # Check for "no good fit" scenario
                        if "_no_good_fit" in results_df.columns and results_df["_no_good_fit"].iloc[0]:
                            best_score = results_df["_best_score"].iloc[0]
                            st.error(
                                f"🚨 **No good fit found!** Best composite score ({best_score:.4f}) is below threshold ({threshold_no_fit:.2f}). "
                                f"This suggests:\n"
                                f"- The measurement may not be valid for analysis\n"
                                f"- Parameter ranges may need to be expanded\n"
                                f"- Measurement quality may be too low\n\n"
                                f"Consider checking measurement quality gates and expanding search ranges."
                            )
                    else:
                        # Add empty columns if disabled
                        results_df = results_df.copy()
                        results_df["edge_case_flag"] = False
                        results_df["edge_case_reason"] = ""
                    
                    # Client Reference Scoring - compare with known client best fits (if enabled)
                    if enable_client_ref and client_lipid is not None:
                        # Compute measurement_features if not already computed (e.g., when viewing cached results)
                        if 'measurement_features' not in dir() or measurement_features is None:
                            measurement_features = prepare_measurement(selected_measurement, analysis_cfg)
                        if 'measurement_quality_result' not in dir():
                            quality_cfg = analysis_cfg.get("quality_gates", {})
                            measurement_quality_result, _ = measurement_quality_score(
                                measurement_features,
                                min_peaks=quality_cfg.get("min_peaks"),
                                min_signal_amplitude=quality_cfg.get("min_signal_amplitude"),
                                min_wavelength_span_nm=quality_cfg.get("min_wavelength_span_nm"),
                            )
                        
                        # Remove any existing entries with these exact parameters (we'll add our own at top)
                        results_df = results_df[
                            ~((results_df["lipid_nm"] == client_lipid) & 
                              (results_df["aqueous_nm"] == client_aqueous) & 
                              (results_df["roughness_A"] == client_roughness))
                        ]
                        
                        # Evaluate client's reference parameters
                        ref_spectrum = single_spectrum(float(client_lipid), float(client_aqueous), float(client_roughness))
                        if ref_spectrum is not None and len(ref_spectrum) > 0 and not np.all(ref_spectrum == 0):
                            ref_spectrum_std = np.std(ref_spectrum)
                            if ref_spectrum_std >= 1e-6:
                                ref_theoretical = prepare_theoretical_spectrum(
                                    wavelengths,
                                    ref_spectrum,
                                    measurement_features,
                                    analysis_cfg,
                                )
                                ref_scores, ref_diagnostics = score_candidate(
                                    measurement_features,
                                    ref_theoretical,
                                    metrics_cfg,
                                    lipid_nm=float(client_lipid),
                                    aqueous_nm=float(client_aqueous),
                                    roughness_A=float(client_roughness),
                                    measurement_quality=measurement_quality_result,
                                )
                                
                                # Create reference record
                                ref_record = {
                                    "lipid_nm": float(client_lipid),
                                    "aqueous_nm": float(client_aqueous),
                                    "roughness_A": float(client_roughness),
                                }
                                for key, value in ref_scores.items():
                                    ref_record[f"score_{key}"] = float(value)
                                for metric, diag in ref_diagnostics.items():
                                    for diag_key, diag_val in diag.items():
                                        ref_record[f"{metric}_{diag_key}"] = float(diag_val)
                                
                                # Add edge case flags for reference if column exists
                                if "edge_case_flag" in results_df.columns:
                                    ref_record["edge_case_flag"] = False
                                    ref_record["edge_case_reason"] = ""
                                
                                # Create dataframe from reference record with a flag
                                ref_record["_is_reference"] = True
                                ref_df = pd.DataFrame([ref_record])
                                
                                # Sort results first (without reference)
                                results_df = results_df.sort_values("score_composite", ascending=False).reset_index(drop=True)
                                
                                # Insert reference at the top (always visible for comparison)
                                results_df = pd.concat([ref_df, results_df], ignore_index=True)
                                
                                st.info(f"⭐ **Reference (Client's best fit)**: L={client_lipid}nm, A={client_aqueous}nm, R={client_roughness}Å - Score={ref_scores.get('composite', 0):.4f}")
                        else:
                            st.warning(f"⚠️ Could not evaluate reference parameters (L={client_lipid}nm, A={client_aqueous}nm, R={client_roughness}Å) - invalid spectrum generated")
                    
                    display_df = results_df.head(top_k_display)
                    
                    # Only show edge case summary if edge cases appear in the displayed results
                    if "edge_case_flag" in display_df.columns:
                        edge_cases_in_display = display_df[display_df["edge_case_flag"] == True]
                        if len(edge_cases_in_display) > 0:
                            st.warning(f"⚠️ **{len(edge_cases_in_display)} edge case(s) detected** in displayed results. Review flagged candidates carefully.")
                            
                            # Show breakdown of edge case types (only for displayed results)
                            edge_reasons = edge_cases_in_display["edge_case_reason"].str.split("; ").explode()
                            reason_counts = edge_reasons.value_counts()
                            if len(reason_counts) > 0:
                                reason_text = ", ".join([f"{reason}: {count}" for reason, count in reason_counts.items()])
                                st.caption(f"Edge case types: {reason_text}")
                    
                    # Highlight edge cases in display
                    if "edge_case_flag" in display_df.columns:
                        # Create styled dataframe with edge case indicators
                        display_df_styled = display_df.copy()
                        # Add visual indicator column
                        display_df_styled["⚠️"] = display_df_styled["edge_case_flag"].apply(lambda x: "⚠️" if x else "")
                        # Reorder columns to show flag first, exclude internal columns
                        cols = ["⚠️"] + [c for c in display_df_styled.columns 
                                         if c != "⚠️" and c != "edge_case_flag" and c != "edge_case_reason" 
                                         and not c.startswith("_")]
                        display_df_styled = display_df_styled[cols]
                        st.dataframe(display_df_styled, use_container_width=True)
                        
                        # Show edge case details in expander
                        if display_df["edge_case_flag"].any():
                            with st.expander("🔍 Edge Case Details", expanded=False):
                                edge_display = display_df[display_df["edge_case_flag"] == True][
                                    ["lipid_nm", "aqueous_nm", "roughness_A", "score_composite", "edge_case_reason"]
                                ]
                                for idx, row in edge_display.iterrows():
                                    st.write(f"**Rank {idx + 1}**: {row['edge_case_reason']}")
                                    st.write(f"  Parameters: L={row['lipid_nm']:.0f}nm, A={row['aqueous_nm']:.0f}nm, R={row['roughness_A']:.0f}Å, Score={row['score_composite']:.4f}")
                    else:
                        st.dataframe(display_df, use_container_width=True)
                    options = list(display_df.index)
                    selection = st.selectbox(
                        "Select a candidate to apply", options=options, format_func=lambda idx: f"Rank {idx + 1}"
                    )
                    
                    # Action buttons in columns
                    btn_col1, btn_col2 = st.columns(2)
                    
                    with btn_col1:
                        if st.button("Apply Selection", key="apply_grid_selection"):
                            row = display_df.loc[selection]
                            st.session_state["pending_slider_update"] = {
                                "lipid_slider": float(row["lipid_nm"]),
                                "aqueous_slider": float(row["aqueous_nm"]),
                                "rough_slider": float(row["roughness_A"]),
                            }
                            st.rerun()
                    
                    with btn_col2:
                        # PDF Export button
                        pdf_top_n = st.number_input(
                            "Top N for PDF", min_value=1, max_value=20, value=10, step=1,
                            help="Number of top fits to include in PDF report"
                        )
                        
                        if st.button("📄 Export PDF Report", key="export_pdf_report"):
                            with st.spinner(f"Generating PDF report for top {pdf_top_n} fits..."):
                                try:
                                    pdf_bytes = generate_main_app_pdf_report(
                                        results_df=results_df,
                                        measurement_file=selected_file,
                                        measured_df=selected_measurement,
                                        single_spectrum_func=single_spectrum,
                                        wavelengths=wavelengths,
                                        analysis_cfg=analysis_cfg,
                                        detrend_func=detrend_dataframe,
                                        detect_peaks_func=detect_peaks_df,
                                        detect_valleys_func=detect_valleys_df,
                                        top_n=pdf_top_n,
                                    )
                                    
                                    # Generate filename with timestamp
                                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                                    safe_filename = selected_file.replace('/', '_').replace('\\', '_')
                                    pdf_filename = f"tear_film_report_{safe_filename}_{timestamp}.pdf"
                                    
                                    st.download_button(
                                        label="⬇️ Download PDF Report",
                                        data=pdf_bytes,
                                        file_name=pdf_filename,
                                        mime="application/pdf",
                                        key="download_pdf_report"
                                    )
                                    st.success(f"✅ PDF report generated! Click above to download.")
                                except Exception as e:
                                    st.error(f"❌ Error generating PDF: {str(e)}")
                                    import traceback
                                    st.code(traceback.format_exc())
            else:
                st.info("Run the grid search to see ranked candidates.")


if __name__ == "__main__":
    main()


