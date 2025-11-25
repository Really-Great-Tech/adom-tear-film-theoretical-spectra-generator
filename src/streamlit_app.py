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
from datetime import timedelta

from analysis import (
    load_measurement_spectrum,
    detrend_signal,
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


def load_measurement_files(measurements_dir: pathlib.Path, config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Load measurement spectra using the shared loader.
    
    Only loads from:
    - exploration/sample_data/good_fit/ (Silas's good fit samples)
    - exploration/sample_data/bad_fit/ (Silas's bad fit samples)
    
    Explicitly skips:
    - _BestFit.txt files (pre-computed fits, not measurements)
    - readme/documentation files
    - Files that don't exist (silently skipped, no error)
    """

    measurements: Dict[str, pd.DataFrame] = {}
    meas_config = config.get("measurements", {})
    file_pattern = meas_config.get("file_pattern", "*.txt")
    
    # Only search in exploration/sample_data/good_fit and bad_fit directories
    exploration_dir = PROJECT_ROOT / "exploration" / "sample_data"
    search_dirs = []
    
    # Only load from good_fit and bad_fit subdirectories
    for subdir_name in ["good_fit", "bad_fit"]:
        subdir = exploration_dir / subdir_name
        if subdir.exists() and subdir.is_dir():
            search_dirs.append(subdir)
    
    if not search_dirs:
        # Only show warning if exploration directory doesn't exist at all
        if not exploration_dir.exists():
            st.warning(f"Exploration sample data directory not found: {exploration_dir}")
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
        st.warning(f"No measurement files found in exploration/sample_data/good_fit or bad_fit matching pattern: {file_pattern}")
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

            # All files are relative to exploration_dir
            base_dir_normalized = pathlib.Path(os.path.normpath(str(exploration_dir.resolve())))
            try:
                rel_path = file_path_obj.relative_to(base_dir_normalized)
            except ValueError:
                # File is not relative to exploration_dir, skip it
                continue

            file_name = str(rel_path.with_suffix(""))  # Remove .txt extension
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
            "lipid": {"min": 0, "max": 500},
            "aqueous": {"min": 0, "max": 2000},  # Note: aqueous CAN be 0
            "roughness": {"min": 0, "max": 5000},
        }
    
    accept_lipid_min = float(acceptable_ranges.get("lipid", {}).get("min", 0))
    accept_lipid_max = float(acceptable_ranges.get("lipid", {}).get("max", 500))
    accept_aqueous_min = float(acceptable_ranges.get("aqueous", {}).get("min", 0))
    accept_aqueous_max = float(acceptable_ranges.get("aqueous", {}).get("max", 2000))
    accept_rough_min = float(acceptable_ranges.get("roughness", {}).get("min", 0))
    accept_rough_max = float(acceptable_ranges.get("roughness", {}).get("max", 5000))
    
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
        
        # Aqueous can be 0, so only flag if > max (not if == 0)
        if aqueous < accept_aqueous_min:
            if aqueous != 0:  # 0 is valid for aqueous
                edge_case_reasons.append(f"aqueous_below_min({aqueous:.0f}nm, min: {accept_aqueous_min:.0f}nm)")
                is_edge_case = True
        elif aqueous > accept_aqueous_max:
            edge_case_reasons.append(f"aqueous_above_max({aqueous:.0f}nm, max: {accept_aqueous_max:.0f}nm)")
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
        accept_lipid_min = float(acceptable_ranges.get("lipid", {}).get("min", 0))
        accept_lipid_max = float(acceptable_ranges.get("lipid", {}).get("max", 500))
        accept_aqueous_min = float(acceptable_ranges.get("aqueous", {}).get("min", 0))
        accept_aqueous_max = float(acceptable_ranges.get("aqueous", {}).get("max", 2000))
        accept_rough_min = float(acceptable_ranges.get("roughness", {}).get("min", 0))
        accept_rough_max = float(acceptable_ranges.get("roughness", {}).get("max", 5000))
        
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
                          rough_val: float, config: Dict[str, Any], selected_file: str) -> go.Figure:
    """Create a plot comparing theoretical and measured spectra."""
    plot_config = config.get('plotting', {})
    style = plot_config.get('plot_style', {})
    
    fig = go.Figure()
    
    # Add measured spectrum
    fig.add_trace(go.Scatter(
        x=measured_df['wavelength'],
        y=measured_df['reflectance'],
        mode='lines',
        name=f'Measured ({selected_file})',
        line=dict(
            color=style.get('measured_color', 'red'),
            width=style.get('line_width', 2)
        ),
        hovertemplate='λ=%{x:.1f}nm<br>R=%{y:.4f}<br>Measured<extra></extra>'
    ))
    
    # Add theoretical spectrum
    fig.add_trace(go.Scatter(
        x=theoretical_wl,
        y=theoretical_spec,
        mode='lines',
        name=f'Theoretical (L={lipid_val:.0f}, A={aqueous_val:.0f}, R={rough_val:.0f})',
        line=dict(
            color=style.get('theoretical_color', 'blue'),
            width=style.get('line_width', 2)
        ),
        hovertemplate='λ=%{x:.1f}nm<br>R=%{y:.4f}<br>Theoretical<extra></extra>'
    ))
    
    fig.update_layout(
        title="Measured vs Theoretical Reflectance Spectra",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Reflectance",
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
    
    # Get acceptable ranges for sliders (wider than parameter ranges, allows using grid search results)
    edge_case_cfg = analysis_cfg.get("edge_case_detection", {})
    acceptable_ranges = edge_case_cfg.get("acceptable_ranges", {})
    
    # Use acceptable ranges for slider bounds, but keep step sizes from parameter config
    slider_lipid_min = float(acceptable_ranges.get("lipid", {}).get("min", lipid_cfg["min"]))
    slider_lipid_max = float(acceptable_ranges.get("lipid", {}).get("max", lipid_cfg["max"]))
    slider_aqueous_min = float(acceptable_ranges.get("aqueous", {}).get("min", aqueous_cfg["min"]))
    slider_aqueous_max = float(acceptable_ranges.get("aqueous", {}).get("max", aqueous_cfg["max"]))
    slider_rough_min = float(acceptable_ranges.get("roughness", {}).get("min", rough_cfg["min"]))
    slider_rough_max = float(acceptable_ranges.get("roughness", {}).get("max", rough_cfg["max"]))

    # Defaults: use configured defaults if provided, or midpoints snapped to step
    defaults = ui_cfg.get("default_values", {})
    def midpoint(cfg):
        return clamp_to_step((cfg["min"] + cfg["max"]) / 2, cfg["min"], cfg["step"])

    default_lipid = defaults.get("lipid", midpoint(lipid_cfg))
    default_aqueous = defaults.get("aqueous", midpoint(aqueous_cfg))
    default_rough = defaults.get("roughness", midpoint(rough_cfg))

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
        "📈 Detrended Analysis",
        "⚙️ Parameters",
        "🔍 Grid Search",
    ])
    
    # Detrending parameters - read defaults from config
    detrending_cfg = analysis_cfg.get("detrending", {})
    peak_detection_cfg = analysis_cfg.get("peak_detection", {})
    default_cutoff = float(detrending_cfg.get("default_cutoff_frequency", 0.01))
    default_prominence = float(peak_detection_cfg.get("default_prominence", 0.005))
    
    analysis_defaults = {
        "analysis_cutoff_freq": default_cutoff,
        "analysis_peak_prominence": default_prominence,
        "cutoff_freq_slider": default_cutoff,
        "peak_prominence_slider": default_prominence,
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
        value=st.session_state["lipid_slider"],
        step=float(lipid_cfg["step"]),
        format="%.0f",
        key="lipid_slider",
    )

    aqueous_val = st.sidebar.slider(
        "Aqueous thickness (nm)",
        min_value=slider_aqueous_min,
        max_value=slider_aqueous_max,
        value=st.session_state["aqueous_slider"],
        step=float(aqueous_cfg["step"]),
        format="%.0f",
        key="aqueous_slider",
    )

    rough_val = st.sidebar.slider(
        "Mucus roughness (Å)",
        min_value=slider_rough_min,
        max_value=slider_rough_max,
        value=st.session_state["rough_slider"],
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
    
    # Apply button to update both parameters at once
    apply_analysis_params = st.sidebar.button("Apply Analysis Parameters", type="primary", use_container_width=True)
    
    if apply_analysis_params:
        st.session_state["analysis_cutoff_freq"] = cutoff_freq_input
        st.session_state["analysis_peak_prominence"] = peak_prominence_input
        # Update analysis_cfg dynamically for this session
        analysis_cfg["detrending"]["default_cutoff_frequency"] = cutoff_freq_input
        analysis_cfg["peak_detection"]["default_prominence"] = peak_prominence_input
        st.rerun()
    
    # Use session state values (these are the active values)
    cutoff_freq = st.session_state["analysis_cutoff_freq"]
    peak_prominence = st.session_state["analysis_peak_prominence"]
    
    # Show current active values
    st.sidebar.caption(f"Active: Cutoff={cutoff_freq:.3f}, Prominence={peak_prominence:.5f}")

    # Compute theoretical spectrum
    spectrum = single_spectrum(lipid_val, aqueous_val, rough_val)
    theoretical_df = pd.DataFrame({
        'wavelength': wavelengths,
        'reflectance': spectrum
    })

    # Tab 1: Spectrum Comparison
    with tab1:
        if measurements_enabled and selected_file and selected_file != "None":
            selected_measurement = measurements[selected_file]
            
            # Create comparison plot
            fig = create_comparison_plot(
                wavelengths, spectrum, selected_measurement,
                lipid_val, aqueous_val, rough_val, config, selected_file
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display fit metrics
            interpolated_measured = interpolate_measurement_to_theoretical(selected_measurement, wavelengths)
            metrics = calculate_fit_metrics(interpolated_measured, spectrum)
            
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
            residuals = interpolated_measured - spectrum
            fig_residuals = go.Figure()
            fig_residuals.add_trace(go.Scatter(
                x=wavelengths,
                y=residuals,
                mode='lines',
                name='Residuals (Measured - Theoretical)',
                line=dict(color='green', width=2)
            ))
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
            fig_residuals.update_layout(
                title="Residuals (Measured - Theoretical)",
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

    # Tab 2: Detrended Analysis
    with tab2:
        st.markdown("## Detrended Signal Analysis with Peak Detection")
        
        if measurements_enabled and selected_file and selected_file != "None":
            selected_measurement = measurements[selected_file]
            
            # Detrend both theoretical and measured signals
            filter_order = config.get("analysis", {}).get("detrending", {}).get("filter_order", 3)
            theoretical_detrended = detrend_dataframe(theoretical_df, cutoff_freq, filter_order)
            measured_detrended = detrend_dataframe(selected_measurement, cutoff_freq, filter_order)
            
            # Detect peaks in detrended signals
            theo_peaks = detect_peaks_df(theoretical_detrended, 'detrended', peak_prominence)
            meas_peaks = detect_peaks_df(measured_detrended, 'detrended', peak_prominence)
            
            # Detect valleys in detrended signals
            theo_valleys = detect_valleys_df(theoretical_detrended, 'detrended', peak_prominence)
            meas_valleys = detect_valleys_df(measured_detrended, 'detrended', peak_prominence)
            
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Original vs Detrended Signals", "Peak and Valley Detection"),
                vertical_spacing=0.1
            )
            
            # Top plot: Original vs Detrended
            fig.add_trace(go.Scatter(
                x=theoretical_detrended['wavelength'], 
                y=theoretical_detrended['reflectance'],
                mode='lines', name='Theoretical Original',
                line=dict(color='blue', width=2)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=measured_detrended['wavelength'], 
                y=measured_detrended['reflectance'],
                mode='lines', name='Measured Original',
                line=dict(color='red', width=2)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=theoretical_detrended['wavelength'], 
                y=theoretical_detrended['detrended'],
                mode='lines', name='Theoretical Detrended',
                line=dict(color='lightblue', width=2, dash='dash')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=measured_detrended['wavelength'], 
                y=measured_detrended['detrended'],
                mode='lines', name='Measured Detrended',
                line=dict(color='lightcoral', width=2, dash='dash')
            ), row=1, col=1)
            
            # Bottom plot: Detrended with peaks and valleys
            fig.add_trace(go.Scatter(
                x=theoretical_detrended['wavelength'], 
                y=theoretical_detrended['detrended'],
                mode='lines', name='Theoretical Detrended',
                line=dict(color='blue', width=2), showlegend=False
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=measured_detrended['wavelength'], 
                y=measured_detrended['detrended'],
                mode='lines', name='Measured Detrended',
                line=dict(color='red', width=2), showlegend=False
            ), row=2, col=1)
            
            # Add peaks
            if len(theo_peaks) > 0:
                fig.add_trace(go.Scatter(
                    x=theo_peaks['wavelength'], 
                    y=theo_peaks['detrended'],
                    mode='markers', name='Theoretical Peaks',
                    marker=dict(color='blue', size=8, symbol='triangle-up')
                ), row=2, col=1)
            
            if len(meas_peaks) > 0:
                fig.add_trace(go.Scatter(
                    x=meas_peaks['wavelength'], 
                    y=meas_peaks['detrended'],
                    mode='markers', name='Measured Peaks',
                    marker=dict(color='red', size=8, symbol='triangle-up')
                ), row=2, col=1)
            
            # Add valleys
            if len(theo_valleys) > 0:
                fig.add_trace(go.Scatter(
                    x=theo_valleys['wavelength'], 
                    y=theo_valleys['detrended'],
                    mode='markers', name='Theoretical Valleys',
                    marker=dict(color='blue', size=8, symbol='triangle-down')
                ), row=2, col=1)
            
            if len(meas_valleys) > 0:
                fig.add_trace(go.Scatter(
                    x=meas_valleys['wavelength'], 
                    y=meas_valleys['detrended'],
                    mode='markers', name='Measured Valleys',
                    marker=dict(color='red', size=8, symbol='triangle-down')
                ), row=2, col=1)
            
            fig.update_xaxes(title_text="Wavelength (nm)")
            fig.update_yaxes(title_text="Reflectance", row=1, col=1)
            fig.update_yaxes(title_text="Detrended Reflectance", row=2, col=1)
            fig.update_layout(height=800, template="plotly_white")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Peak analysis summary
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Theoretical Signal Analysis")
                st.write(f"**Peaks detected:** {len(theo_peaks)}")
                if len(theo_peaks) > 0:
                    st.write("Peak wavelengths:", [f"{w:.1f}nm" for w in theo_peaks['wavelength'].head(5)])
                st.write(f"**Valleys detected:** {len(theo_valleys)}")
                if len(theo_valleys) > 0:
                    st.write("Valley wavelengths:", [f"{w:.1f}nm" for w in theo_valleys['wavelength'].head(5)])
            
            with col2:
                st.markdown("### Measured Signal Analysis")
                st.write(f"**Peaks detected:** {len(meas_peaks)}")
                if len(meas_peaks) > 0:
                    st.write("Peak wavelengths:", [f"{w:.1f}nm" for w in meas_peaks['wavelength'].head(5)])
                st.write(f"**Valleys detected:** {len(meas_valleys)}")
                if len(meas_valleys) > 0:
                    st.write("Valley wavelengths:", [f"{w:.1f}nm" for w in meas_valleys['wavelength'].head(5)])
            
        else:
            # Show only theoretical detrended
            filter_order = config.get("analysis", {}).get("detrending", {}).get("filter_order", 3)
            theoretical_detrended = detrend_dataframe(theoretical_df, cutoff_freq, filter_order)
            theo_peaks = detect_peaks_df(theoretical_detrended, 'detrended', peak_prominence)
            theo_valleys = detect_valleys_df(theoretical_detrended, 'detrended', peak_prominence)
            
            fig = go.Figure()
            
            # Original signal
            fig.add_trace(go.Scatter(
                x=theoretical_detrended['wavelength'],
                y=theoretical_detrended['reflectance'],
                mode='lines', name='Original',
                line=dict(color='blue', width=2)
            ))
            
            # Detrended signal
            fig.add_trace(go.Scatter(
                x=theoretical_detrended['wavelength'],
                y=theoretical_detrended['detrended'],
                mode='lines', name='Detrended',
                line=dict(color='lightblue', width=2, dash='dash')
            ))
            
            # Peaks
            if len(theo_peaks) > 0:
                fig.add_trace(go.Scatter(
                    x=theo_peaks['wavelength'],
                    y=theo_peaks['detrended'],
                    mode='markers', name='Peaks',
                    marker=dict(color='green', size=8, symbol='triangle-up')
                ))
            
            # Valleys
            if len(theo_valleys) > 0:
                fig.add_trace(go.Scatter(
                    x=theo_valleys['wavelength'],
                    y=theo_valleys['detrended'],
                    mode='markers', name='Valleys',
                    marker=dict(color='orange', size=8, symbol='triangle-down')
                ))
            
            fig.update_layout(
                title="Theoretical Signal: Original vs Detrended with Peak/Valley Detection",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Reflectance",
                template="plotly_white",
                height=600
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
                ["random", "coarse-fine"],
                index=1,  # Default to "coarse-fine" (recommended)
                help="random: Random sampling across full parameter space. coarse-fine: Two-stage search (coarse then refine around best results)."
            )
            
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
                        
                        # Warn if search space coverage is low (only for random/systematic search, not coarse-fine)
                        cached_lipid_vals = cache_entry.get("lipid_vals")
                        cached_aqueous_vals = cache_entry.get("aqueous_vals")
                        cached_rough_vals = cache_entry.get("rough_vals")
                        search_strategy_cached = cache_entry.get("search_strategy", "random")
                        if (cached_lipid_vals is not None and cached_aqueous_vals is not None and cached_rough_vals is not None 
                            and search_strategy_cached != "coarse-fine"):
                            total_combinations = len(cached_lipid_vals) * len(cached_aqueous_vals) * len(cached_rough_vals)
                            max_results_cached = cache_entry.get("max_results")
                            coverage = 100 * evaluated / total_combinations if total_combinations > 0 else 0
                            
                            # Only warn if coverage is low (< 10%) AND a limit was set
                            # If max_results is 0 or None, it's a full search (100% coverage is expected and good!)
                            if max_results_cached is not None and max_results_cached > 0 and coverage < 10.0:
                                st.warning(f"⚠️ Only {coverage:.1f}% of parameter space explored ({evaluated}/{total_combinations} combinations). "
                                         f"Grid search may be biased toward lower parameter values. "
                                         f"Consider increasing 'Max spectra', using a larger 'Stride multiplier', or trying 'coarse-fine' strategy.")
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
                    if st.button("Apply Selection", key="apply_grid_selection"):
                        row = display_df.loc[selection]
                        st.session_state["pending_slider_update"] = {
                            "lipid_slider": float(row["lipid_nm"]),
                            "aqueous_slider": float(row["aqueous_nm"]),
                            "rough_slider": float(row["roughness_A"]),
                        }
                        st.rerun()
            else:
                st.info("Run the grid search to see ranked candidates.")


if __name__ == "__main__":
    main()


