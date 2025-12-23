"""
PyElli Grid Search with Peak-Based Scoring

This module integrates pyElli's Transfer Matrix Method with peak count and peak alignment
scoring for finding the best fit.

This demonstrates how to use PyElli for auto-fitting tear film spectra.

Roughness Modeling:
    Uses pyElli's BruggemanEMA + VaryingMixtureLayer for proper interface roughness
    modeling between the mucus layer and corneal epithelium substrate.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import os

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.special import erf

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exploration.pyelli_exploration.pyelli_utils import (
    load_measured_spectrum,
    load_material_data,
    get_available_materials,
)
from src.analysis.measurement_utils import (
    detrend_signal,
    detect_peaks,
    detect_valleys,
)
from src.analysis.metrics import _match_peaks

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PyElliResult:
    """Result from PyElli grid search."""
    lipid_nm: float
    aqueous_nm: float
    mucus_nm: float  # Maps conceptually to ADOM's roughness
    score: float
    rmse: float
    correlation: float
    crossing_count: int
    theoretical_spectrum: np.ndarray
    wavelengths: np.ndarray


# =============================================================================
# PyElli Library Integration
# =============================================================================

def create_bruggeman_roughness_layer(
    wavelengths_nm: np.ndarray,
    mucus_n: np.ndarray,
    mucus_k: np.ndarray,
    substrate_n: np.ndarray,
    substrate_k: np.ndarray,
    roughness_angstrom: float,
    num_divisions: int = 20,
    use_error_function: bool = True,
):
    """
    Create interface roughness layer using pyElli's BruggemanEMA + VaryingMixtureLayer.
    
    This is the proper way to model interface roughness between mucus and substrate.
    Uses Bruggeman Effective Medium Approximation which is the gold standard for
    rough interface modeling in ellipsometry.
    
    Args:
        wavelengths_nm: Wavelength array in nm
        mucus_n, mucus_k: Mucus layer optical constants
        substrate_n, substrate_k: Substrate optical constants
        roughness_angstrom: Interface roughness in Angstroms (LTA range: 300-3000 Å)
        num_divisions: Number of slices for the graded layer (default: 20)
        use_error_function: If True, use error function profile (more physical).
                           If False, use linear profile.
    
    Returns:
        VaryingMixtureLayer configured with BruggemanEMA
    """
    from elli.dispersions.table_index import Table
    from elli.materials import IsotropicMaterial, BruggemanEMA
    from elli.structure import VaryingMixtureLayer
    
    roughness_nm = roughness_angstrom / 10.0
    
    # Create dispersions from n,k data
    mucus_disp = Table(wavelengths_nm, mucus_n + 1j * mucus_k)
    substrate_disp = Table(wavelengths_nm, substrate_n + 1j * substrate_k)
    
    # Create isotropic materials
    mucus_mat = IsotropicMaterial(mucus_disp)
    substrate_mat = IsotropicMaterial(substrate_disp)
    
    # Create Bruggeman EMA mixture
    # host = mucus (fraction=0 means 100% mucus/host)
    # guest = substrate (fraction=1 means 100% substrate/guest)
    bruggeman_mixture = BruggemanEMA(mucus_mat, substrate_mat, fraction=0.5)
    
    # Define fraction modulation profile
    if use_error_function:
        def fraction_profile(z_normalized: float) -> float:
            """
            Error function profile for smooth physical transition.
            z_normalized: 0 (top/mucus side) to 1 (bottom/substrate side)
            Returns: fraction of guest (substrate), 0 → 1
            """
            # Scale to ±3σ range for proper error function coverage
            z_centered = (z_normalized - 0.5) * 6
            return float(0.5 * (1 + erf(z_centered / np.sqrt(2))))
    else:
        def fraction_profile(z_normalized: float) -> float:
            """Linear profile: z=0 → f=0 (mucus), z=1 → f=1 (substrate)."""
            return float(z_normalized)
    
    # Create VaryingMixtureLayer with Bruggeman EMA
    # pyElli handles the internal subdivision and EMA calculations
    roughness_layer = VaryingMixtureLayer(
        material=bruggeman_mixture,
        thickness=roughness_nm,
        div=num_divisions,
        fraction_modulation=fraction_profile,
    )
    
    logger.debug(
        f'🔬 Created Bruggeman roughness layer: {roughness_angstrom:.0f} Å '
        f'({roughness_nm:.1f} nm), {num_divisions} divisions, '
        f'profile={"error_function" if use_error_function else "linear"}'
    )
    
    return roughness_layer


def calculate_reflectance_pyelli(
    wavelengths_nm: np.ndarray,
    lipid_n: np.ndarray,
    lipid_k: np.ndarray,
    lipid_thickness_nm: float,
    aqueous_n: np.ndarray,
    aqueous_k: np.ndarray,
    aqueous_thickness_nm: float,
    mucus_n: np.ndarray,
    mucus_k: np.ndarray,
    mucus_thickness_nm: float,
    substratum_n: np.ndarray,
    substratum_k: np.ndarray,
    roughness_angstrom: float = 0.0,
    enable_roughness: bool = True,
    num_roughness_divisions: int = 20,
    use_error_function_profile: bool = True,
) -> np.ndarray:
    """
    Calculate theoretical reflectance using pyElli with Bruggeman EMA roughness.
    
    This uses pyElli's Transfer Matrix Method with proper interface roughness
    modeling via BruggemanEMA + VaryingMixtureLayer.
    
    Structure: Air → Lipid → Aqueous → Mucus → [Roughness Layer] → Substrate
    
    The roughness layer models the graded interface between the mucus (glycocalyx)
    and the corneal epithelium substrate using Bruggeman Effective Medium
    Approximation - the gold standard for interface roughness in ellipsometry.
    
    Args:
        wavelengths_nm: Wavelength array in nanometers
        lipid_n, lipid_k: Lipid layer optical constants
        lipid_thickness_nm: Lipid layer thickness in nm
        aqueous_n, aqueous_k: Aqueous layer optical constants
        aqueous_thickness_nm: Aqueous layer thickness in nm
        mucus_n, mucus_k: Mucus layer optical constants
        mucus_thickness_nm: Mucus layer thickness in nm (LTA uses fixed 500nm)
        substratum_n, substratum_k: Substrate (struma) optical constants
        roughness_angstrom: Interface roughness in Angstroms (LTA range: 300-3000 Å)
        enable_roughness: If True, model interface roughness with Bruggeman EMA
        num_roughness_divisions: Number of slices for roughness gradient (default: 20)
        use_error_function_profile: If True, use error function (more physical).
                                    If False, use linear transition.
        
    Returns:
        Theoretical reflectance array
    """
    try:
        import elli
        from elli.dispersions.table_index import Table
        from elli.materials import IsotropicMaterial
        
        # Create material dispersions from n,k data
        lipid_nk = lipid_n + 1j * lipid_k
        aqueous_nk = aqueous_n + 1j * aqueous_k
        mucus_nk = mucus_n + 1j * mucus_k
        substratum_nk = substratum_n + 1j * substratum_k
        
        # Create Table dispersions (wavelength in nm, complex nk)
        lipid_disp = Table(wavelengths_nm, lipid_nk)
        aqueous_disp = Table(wavelengths_nm, aqueous_nk)
        mucus_disp = Table(wavelengths_nm, mucus_nk)
        substratum_disp = Table(wavelengths_nm, substratum_nk)
        
        # Wrap in IsotropicMaterial
        lipid_mat = IsotropicMaterial(lipid_disp)
        aqueous_mat = IsotropicMaterial(aqueous_disp)
        mucus_mat = IsotropicMaterial(mucus_disp)
        substrate_mat = IsotropicMaterial(substratum_disp)
        
        # Build layer stack: Air → Lipid → Aqueous → Mucus → [Roughness] → Substrate
        layers = [
            elli.Layer(lipid_mat, lipid_thickness_nm),
            elli.Layer(aqueous_mat, aqueous_thickness_nm),
            elli.Layer(mucus_mat, mucus_thickness_nm),
        ]
        
        # Add Bruggeman roughness layer if enabled
        if enable_roughness and roughness_angstrom > 0:
            roughness_layer = create_bruggeman_roughness_layer(
                wavelengths_nm,
                mucus_n, mucus_k,
                substratum_n, substratum_k,
                roughness_angstrom,
                num_divisions=num_roughness_divisions,
                use_error_function=use_error_function_profile,
            )
            layers.append(roughness_layer)
        else:
            logger.debug('⚠️ Roughness modeling disabled or roughness=0')
        
        # Build structure
        structure = elli.Structure(
            elli.AIR,
            layers,
            substrate_mat,
        )
        
        # Evaluate at normal incidence
        result = structure.evaluate(wavelengths_nm, theta_i=0.0)
        
        return np.array(result.R)
        
    except ImportError:
        logger.error('❌ pyElli not installed. Run: pip install pyElli')
        raise
    except Exception as e:
        logger.error(f'❌ Error calculating pyElli reflectance: {e}')
        logger.warning('⚠️ Falling back to custom TMM implementation')
        return transfer_matrix_reflectance_fallback(
            wavelengths_nm,
            [
                (lipid_n, lipid_k, lipid_thickness_nm),
                (aqueous_n, aqueous_k, aqueous_thickness_nm),
                (mucus_n, mucus_k, mucus_thickness_nm),
            ],
            substratum_n, substratum_k
        )


def transfer_matrix_reflectance_fallback(
    wavelengths_nm: np.ndarray,
    layers: List[Tuple[np.ndarray, np.ndarray, float]],
    substratum_n: Optional[np.ndarray] = None,
    substratum_k: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fallback custom TMM implementation (used if pyElli fails).
    
    This is the original custom implementation kept as backup.
    
    Each layer is (n_array, k_array, thickness_nm).
    Structure: Air -> layers -> substrate
    
    Args:
        wavelengths_nm: Array of wavelengths in nm
        layers: List of (n, k, thickness) tuples for each layer
        
    Returns:
        Reflectance array
    """
    n_air = 1.0
    reflectance = np.zeros_like(wavelengths_nm, dtype=float)
    
    for i, wl in enumerate(wavelengths_nm):
        # Build complex refractive indices
        N = [n_air]  # Start with air
        d = [0]  # Air has no thickness
        
        for n_arr, k_arr, thickness in layers:
            N.append(n_arr[i] + 1j * k_arr[i])
            d.append(thickness)
        
        # Substrate (approximate as semi-infinite, use last layer's index)
        N.append(N[-1])
        
        # Transfer matrix calculation
        M = np.eye(2, dtype=complex)
        
        for j in range(1, len(N) - 1):
            # Interface matrix (Fresnel coefficients)
            r_jk = (N[j-1] - N[j]) / (N[j-1] + N[j])
            t_jk = 2 * N[j-1] / (N[j-1] + N[j])
            
            I_jk = np.array([
                [1, r_jk],
                [r_jk, 1]
            ], dtype=complex) / t_jk
            
            # Propagation matrix (phase shift through layer)
            delta = 2 * np.pi * N[j] * d[j] / wl
            L_j = np.array([
                [np.exp(-1j * delta), 0],
                [0, np.exp(1j * delta)]
            ], dtype=complex)
            
            M = M @ I_jk @ L_j
        
        # Final interface to substrate
        r_final = (N[-2] - N[-1]) / (N[-2] + N[-1])
        t_final = 2 * N[-2] / (N[-2] + N[-1])
        I_final = np.array([
            [1, r_final],
            [r_final, 1]
        ], dtype=complex) / t_final
        
        M = M @ I_final
        
        # Reflectance from transfer matrix
        r = M[1, 0] / M[0, 0]
        reflectance[i] = float(np.abs(r) ** 2)
    
    return reflectance


# =============================================================================
# Peak-Based Scoring (Replacing Monotonic Alignment)
# =============================================================================

def calculate_peak_based_score(
    wavelengths: np.ndarray,
    measured: np.ndarray,
    theoretical: np.ndarray,
    cutoff_frequency: float = 0.008,
    filter_order: int = 3,
    peak_prominence: float = 0.0001,
    tolerance_nm: float = 25.0,  # Increased from 5.0 to 25.0 nm for better peak matching
    tau_nm: float = 30.0,  # Increased from 15.0 to 30.0 nm for more lenient delta scoring
    penalty_unpaired: float = 0.03,  # Reduced from 0.05 to 0.03 for less harsh penalties
) -> Dict[str, float]:
    """
    Calculate peak-based score using peak count and peak alignment.
    
    This replaces monotonic alignment scoring with Silas's approach:
    1. Detrend both signals
    2. Detect peaks in both
    3. Match peaks and calculate alignment score
    
    Args:
        wavelengths: Wavelength array
        measured: Measured reflectance
        theoretical: Theoretical reflectance
        cutoff_frequency: Detrending cutoff frequency
        filter_order: Butterworth filter order
        peak_prominence: Peak detection prominence threshold
        tolerance_nm: Peak matching tolerance in nm
        tau_nm: Decay constant for peak delta scoring
        penalty_unpaired: Penalty per unpaired peak
        
    Returns:
        Dictionary with scores and diagnostics
    """
    min_len = min(len(wavelengths), len(measured), len(theoretical))
    if min_len < 10:
        return {
            "score": 0.0,
            "peak_count_score": 0.0,
            "peak_delta_score": 0.0,
            "matched_peaks": 0,
            "mean_delta_nm": 0.0,
        }
    
    wavelengths = wavelengths[:min_len]
    measured = measured[:min_len]
    theoretical = theoretical[:min_len]
    
    # Detrend both signals
    try:
        meas_detrended = detrend_signal(wavelengths, measured, cutoff_frequency, filter_order)
        theo_detrended = detrend_signal(wavelengths, theoretical, cutoff_frequency, filter_order)
    except Exception:
        return {
            "score": 0.0,
            "peak_count_score": 0.0,
            "peak_delta_score": 0.0,
            "matched_peaks": 0,
            "mean_delta_nm": 0.0,
        }
    
    # Detect peaks
    meas_peaks_df = detect_peaks(wavelengths, meas_detrended, prominence=peak_prominence)
    theo_peaks_df = detect_peaks(wavelengths, theo_detrended, prominence=peak_prominence)
    
    meas_peaks = meas_peaks_df["wavelength"].to_numpy(dtype=float) if len(meas_peaks_df) > 0 else np.array([], dtype=float)
    theo_peaks = theo_peaks_df["wavelength"].to_numpy(dtype=float) if len(theo_peaks_df) > 0 else np.array([], dtype=float)
    
    # Match peaks
    matched_meas, matched_theo, deltas = _match_peaks(meas_peaks, theo_peaks, tolerance_nm)
    
    # Peak count score - more lenient scoring
    meas_count = len(meas_peaks)
    theo_count = len(theo_peaks)
    matched_count = len(matched_meas)
    
    if meas_count == 0:
        peak_count_score = 1.0 if theo_count == 0 else 0.0
    elif matched_count == 0:
        # If no peaks match, give a small score based on how close the counts are
        # This prevents all zero-match results from getting 0.0 score
        count_ratio = min(meas_count, theo_count) / max(meas_count, theo_count) if max(meas_count, theo_count) > 0 else 0.0
        peak_count_score = 0.1 * count_ratio  # Small score for similar counts even if no matches
    else:
        # Reward matching: matched/measured ratio, with bonus for matching most peaks
        match_ratio = matched_count / float(meas_count)
        # Also consider if we're matching a good fraction of theoretical peaks
        theo_match_ratio = matched_count / float(theo_count) if theo_count > 0 else 0.0
        peak_count_score = 0.7 * match_ratio + 0.3 * theo_match_ratio
        peak_count_score = max(0.0, min(1.0, peak_count_score))
    
    # Peak delta score - more lenient
    unmatched_measurement = len(meas_peaks) - len(matched_meas)
    unmatched_theoretical = len(theo_peaks) - len(matched_theo)
    
    if deltas.size == 0:
        mean_delta = 0.0
        if unmatched_measurement == 0 and unmatched_theoretical == 0:
            peak_delta_score = 1.0
        else:
            # If no matches but similar peak counts, give small score
            total_peaks = len(meas_peaks) + len(theo_peaks)
            if total_peaks > 0:
                unmatched_ratio = (unmatched_measurement + unmatched_theoretical) / float(total_peaks)
                peak_delta_score = 0.05 * (1.0 - unmatched_ratio)  # Small score for similar counts
            else:
                peak_delta_score = 0.0
    else:
        mean_delta = float(np.mean(deltas))
        # More lenient scoring: use larger tau for exponential decay
        peak_delta_score = float(np.exp(-mean_delta / max(tau_nm, 1e-6)))
    
    # Reduced penalty for unpaired peaks
    penalty = penalty_unpaired * float(unmatched_measurement + unmatched_theoretical)
    peak_delta_score = max(0.0, min(1.0, peak_delta_score - penalty))
    
    # Composite score: Prioritize peak alignment (70% peak delta, 30% peak count)
    # Peak delta is more important as it measures how well peaks align
    # Increased weight on delta to prioritize alignment quality
    composite_score = 0.7 * peak_delta_score + 0.3 * peak_count_score
    
    return {
        "score": float(np.clip(composite_score, 0.0, 1.0)),
        "peak_count_score": peak_count_score,
        "peak_delta_score": peak_delta_score,
        "matched_peaks": float(matched_count),
        "mean_delta_nm": mean_delta,
        "measurement_peaks": float(meas_count),
        "theoretical_peaks": float(len(theo_peaks)),
        "unpaired_measurement": float(unmatched_measurement),
        "unpaired_theoretical": float(unmatched_theoretical),
    }


def calculate_monotonic_alignment_score(
    wavelengths: np.ndarray,
    measured: np.ndarray,
    theoretical: np.ndarray,
    focus_wavelength_min: float = 600.0,  # Match LTA focus region
    focus_wavelength_max: float = 1200.0,  # Match LTA focus region
    focus_reflectance_max: float = 0.10,  # More permissive to include more data
    strict_crossing_rejection: bool = False,  # Use soft penalty (not hard rejection)
) -> Dict[str, float]:
    """
    Visual quality metric that evaluates fit quality.
    
    SOFT PENALTY: Crossings reduce score but don't reject fits.
    This balances visual quality with finding reasonable fits.
    
    Args:
        wavelengths: Wavelength array
        measured: Measured reflectance
        theoretical: Theoretical reflectance
        focus_wavelength_min: Start of focus region (nm)
        focus_wavelength_max: End of focus region (nm)
        focus_reflectance_max: Maximum reflectance in focus region (0.10 = 10%)
        strict_crossing_rejection: If True, use hard rejection. If False (default), use soft penalty.
        
    Returns:
        Dictionary with score and diagnostics
    """
    min_len = min(len(wavelengths), len(measured), len(theoretical))
    if min_len < 10:
        return {"score": 0.5, "crossing_count": 0, "rmse": 1.0, "correlation": 0.0}
    
    wavelengths = wavelengths[:min_len]
    measured = measured[:min_len]
    theoretical = theoretical[:min_len]
    
    # Calculate residual
    residual = measured - theoretical
    
    # Apply focus region mask (wavelength only, more permissive on reflectance)
    focus_mask = (
        (wavelengths >= focus_wavelength_min) & 
        (wavelengths <= focus_wavelength_max)
    )
    
    # Additional filter for low reflectance if needed
    if focus_reflectance_max < 1.0:
        focus_mask = focus_mask & (measured <= focus_reflectance_max)
    
    points_in_focus = int(np.sum(focus_mask))
    
    if points_in_focus < 10:
        # Use full range if focus region has too few points
        residual_focus = residual
        meas_focus = measured
        theo_focus = theoretical
        wl_focus = wavelengths
        points_in_focus = len(residual)
    else:
        residual_focus = residual[focus_mask]
        meas_focus = measured[focus_mask]
        theo_focus = theoretical[focus_mask]
        wl_focus = wavelengths[focus_mask]
    
    # === CRITERION 1: SHAPE SIMILARITY (40% weight) ===
    if np.std(meas_focus) > 1e-10 and np.std(theo_focus) > 1e-10:
        correlation = float(np.corrcoef(meas_focus, theo_focus)[0, 1])
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0
    
    shape_score = max(0.0, correlation)
    
    # === CRITERION 2: CLOSENESS / RMSE (40% weight) ===
    rmse = float(np.sqrt(np.mean(residual_focus ** 2)))
    meas_mean = float(np.mean(meas_focus))
    meas_mean = max(meas_mean, 0.001)
    
    # Normalized RMSE (relative to mean reflectance)
    nrmse = rmse / meas_mean
    
    # Score: low NRMSE = high score
    # NRMSE of 0.05 (5%) → score ~0.9
    # NRMSE of 0.20 (20%) → score ~0.45
    closeness_score = float(np.exp(-nrmse * 5))
    
    # === CRITERION 3: CRISS-CROSSING DETECTION (soft penalty) ===
    # Count sign changes in residual (measured - theoretical)
    # Use soft penalty - crossings reduce score but don't reject
    sign_changes_focus = int(np.sum(np.abs(np.diff(np.sign(residual_focus))) > 0))
    
    # Normalize crossings by number of points (crossings per 100 points)
    crossings_per_100 = (sign_changes_focus / max(points_in_focus, 1)) * 100
    
    # Soft penalty for crossings (not hard rejection)
    # 0 crossings → score = 1.0
    # 1-2 crossings → score ~0.8
    # 5+ crossings → score < 0.5
    crossing_score = float(np.exp(-crossings_per_100 * 0.05))
    
    # === COMBINED SCORE ===
    # Weight: shape 40%, closeness 40%, crossing penalty 20% (soft rejection)
    score = (
        0.40 * shape_score +
        0.40 * closeness_score +
        0.20 * crossing_score
    )
    
    # Bonus for excellent shape match
    if shape_score > 0.95:
        score = min(1.0, score * 1.1)
    
    # Penalty for poor shape (anti-correlation)
    if shape_score < 0.5:
        score = score * 0.5
    
    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "crossing_count": sign_changes_focus,
        "crossings": sign_changes_focus,  # Alias for app compatibility
        "crossings_per_100": crossings_per_100,
        "rmse": rmse,
        "nrmse": nrmse,
        "correlation": correlation,
        "shape_score": shape_score,
        "closeness_score": closeness_score,
        "crossing_score": crossing_score,
        "points_in_focus": points_in_focus,
    }


# =============================================================================
# Parallel Processing Worker Functions
# =============================================================================

def _evaluate_single_combination(
    wavelengths: np.ndarray,
    measured: np.ndarray,
    lipid: float,
    aqueous: float,
    roughness_angstrom: float,
    material_data: Dict[str, Any],
    enable_roughness: bool = True,
) -> Optional[PyElliResult]:
    """
    Worker function for parallel grid search evaluation.
    
    This function is designed to be pickled for multiprocessing.
    
    Args:
        wavelengths: Wavelength array
        measured: Measured reflectance
        lipid: Lipid thickness in nm
        aqueous: Aqueous thickness in nm
        roughness_angstrom: Interface roughness in Angstroms
        material_data: Dictionary with material DataFrames
        enable_roughness: Whether to enable roughness modeling
        
    Returns:
        PyElliResult or None if calculation fails
    """
    try:
        # Get material n,k values
        lipid_n = np.interp(wavelengths, material_data['lipid']['wavelength_nm'], material_data['lipid']['n'])
        lipid_k = np.interp(wavelengths, material_data['lipid']['wavelength_nm'], material_data['lipid']['k'])
        water_n = np.interp(wavelengths, material_data['water']['wavelength_nm'], material_data['water']['n'])
        water_k = np.interp(wavelengths, material_data['water']['wavelength_nm'], material_data['water']['k'])
        mucus_n = np.interp(wavelengths, material_data['mucus']['wavelength_nm'], material_data['mucus']['n'])
        mucus_k = np.interp(wavelengths, material_data['mucus']['wavelength_nm'], material_data['mucus']['k'])
        substratum_n = np.interp(wavelengths, material_data['substratum']['wavelength_nm'], material_data['substratum']['n'])
        substratum_k = np.interp(wavelengths, material_data['substratum']['wavelength_nm'], material_data['substratum']['k'])
        
        # Calculate theoretical spectrum
        roughness_nm = roughness_angstrom / 10.0
        mucus_thickness_nm = 500.0  # Fixed in LTA
        
        theoretical = calculate_reflectance_pyelli(
            wavelengths,
            lipid_n, lipid_k, lipid,
            water_n, water_k, aqueous,
            mucus_n, mucus_k, mucus_thickness_nm,
            substratum_n, substratum_k,
            roughness_angstrom=roughness_angstrom,
            enable_roughness=enable_roughness,
            num_roughness_divisions=20,
            use_error_function_profile=True,
        )
        
        # Align using linear regression (focus region 600-1100 nm)
        focus_mask = (wavelengths >= 600.0) & (wavelengths <= 1100.0)
        if focus_mask.sum() > 0:
            meas_focus = measured[focus_mask]
            theo_focus = theoretical[focus_mask]
            if np.std(theo_focus) > 1e-10:
                scale = np.dot(meas_focus, theo_focus) / np.dot(theo_focus, theo_focus)
                theoretical_scaled = theoretical * scale
            else:
                theoretical_scaled = theoretical
        else:
            theoretical_scaled = theoretical
        
        # Score using peak-based scoring
        score_result = calculate_peak_based_score(
            wavelengths, measured, theoretical_scaled
        )
        
        # Calculate RMSE and correlation
        residual = measured - theoretical_scaled
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        if np.std(measured) > 1e-10 and np.std(theoretical_scaled) > 1e-10:
            correlation = float(np.corrcoef(measured, theoretical_scaled)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        return PyElliResult(
            lipid_nm=float(lipid),
            aqueous_nm=float(aqueous),
            mucus_nm=float(roughness_angstrom),
            score=score_result['score'],
            rmse=rmse,
            correlation=correlation,
            crossing_count=0,
            theoretical_spectrum=theoretical_scaled,
            wavelengths=wavelengths,
        )
    except Exception as e:
        logger.debug(f'Error evaluating combination (L={lipid}, A={aqueous}, R={roughness_angstrom}Å): {e}')
        return None


# =============================================================================
# Grid Search Implementation
# =============================================================================

class PyElliGridSearch:
    """
    Grid search for tear film parameters using PyElli TMM and monotonic alignment scoring.
    
    This class:
    1. Uses PyElli's Transfer Matrix Method for spectrum generation
    2. Applies monotonic_alignment scoring (not simple RMSE)
    3. Rejects criss-crossing fits automatically
    """
    
    def __init__(self, materials_path: Path):
        """
        Initialize with material data.
        
        Args:
            materials_path: Path to Materials directory with CSV files
        """
        self.materials_path = materials_path
        self.materials = get_available_materials(materials_path)
        
        # Load tear film materials
        self.lipid_df = load_material_data(
            materials_path / "lipid_05-02621extrapolated.csv"
        )
        self.water_df = load_material_data(
            materials_path / "water_Bashkatov1353extrapolated.csv"
        )
        # Mucus layer material (from LTA Stack XML: water_Bashkatov1353extrapolated)
        self.mucus_df = load_material_data(
            materials_path / "water_Bashkatov1353extrapolated.csv"
        )
        # Substrate material (from LTA Stack XML: struma_Bashkatov140extrapolated)
        self.substratum_df = load_material_data(
            materials_path / "struma_Bashkatov140extrapolated.csv"
        )
        
        logger.info("✅ Loaded tear film materials for PyElli")
    
    def _get_nk(self, mat_df: pd.DataFrame, wavelengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate material n,k values to target wavelengths."""
        n = np.interp(wavelengths, mat_df['wavelength_nm'].values, mat_df['n'].values)
        k = np.interp(wavelengths, mat_df['wavelength_nm'].values, mat_df['k'].values)
        return n, k
    
    def calculate_theoretical_spectrum(
        self,
        wavelengths: np.ndarray,
        lipid_nm: float,
        aqueous_nm: float,
        roughness_nm: float,
        enable_roughness: bool = True,
        use_error_function_profile: bool = True,
    ) -> np.ndarray:
        """
        Calculate theoretical spectrum using pyElli with Bruggeman EMA roughness.
        
        This uses pyElli's TMM with proper interface roughness modeling via
        BruggemanEMA + VaryingMixtureLayer.
        
        Args:
            wavelengths: Wavelength array in nm
            lipid_nm: Lipid layer thickness in nm
            aqueous_nm: Aqueous layer thickness in nm
            roughness_nm: Interface roughness in nm (will be converted to Angstroms)
                         LTA range: 30-300 nm = 300-3000 Å
                         Note: Mucus layer thickness is fixed at 500nm in LTA
            enable_roughness: If True, model interface roughness with Bruggeman EMA
            use_error_function_profile: If True, use error function profile
            
        Returns:
            Theoretical reflectance array
        """
        # Get material n,k values at target wavelengths
        lipid_n, lipid_k = self._get_nk(self.lipid_df, wavelengths)
        water_n, water_k = self._get_nk(self.water_df, wavelengths)
        mucus_n, mucus_k = self._get_nk(self.mucus_df, wavelengths)
        substratum_n, substratum_k = self._get_nk(self.substratum_df, wavelengths)
        
        # LTA uses fixed mucus thickness of 500nm
        mucus_thickness_nm = 500.0
        
        # Convert roughness from nm to Angstroms (LTA uses Angstroms internally)
        roughness_angstrom = roughness_nm * 10.0
        
        # Use pyElli with Bruggeman EMA roughness modeling
        return calculate_reflectance_pyelli(
            wavelengths,
            lipid_n, lipid_k, lipid_nm,
            water_n, water_k, aqueous_nm,
            mucus_n, mucus_k, mucus_thickness_nm,
            substratum_n, substratum_k,
            roughness_angstrom=roughness_angstrom,
            enable_roughness=enable_roughness,
            num_roughness_divisions=20,
            use_error_function_profile=use_error_function_profile,
        )
    
    def _align_spectra(
        self,
        measured: np.ndarray,
        theoretical: np.ndarray,
        focus_min: float = 600.0,
        focus_max: float = 1100.0,
        wavelengths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Align theoretical spectrum to measured using linear regression.
        
        Instead of just scaling by mean, fit: theoretical_aligned = a * theoretical + b
        This handles both amplitude and baseline differences.
        """
        if wavelengths is not None:
            # Focus on middle wavelength region for fitting
            mask = (wavelengths >= focus_min) & (wavelengths <= focus_max)
            meas_fit = measured[mask]
            theo_fit = theoretical[mask]
        else:
            meas_fit = measured
            theo_fit = theoretical
        
        if len(theo_fit) < 10 or np.std(theo_fit) < 1e-10:
            # Fallback to simple scaling
            scale = np.mean(measured) / np.mean(theoretical) if np.mean(theoretical) > 0 else 1.0
            return theoretical * scale
        
        # Linear regression: measured = a * theoretical + b
        # Solve for a, b using least squares
        A = np.vstack([theo_fit, np.ones_like(theo_fit)]).T
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, meas_fit, rcond=None)
            a, b = coeffs
            
            # Apply transformation to full spectrum
            aligned = a * theoretical + b
            
            # Ensure non-negative reflectance
            aligned = np.clip(aligned, 0, None)
            return aligned
        except:
            # Fallback
            scale = np.mean(measured) / np.mean(theoretical) if np.mean(theoretical) > 0 else 1.0
            return theoretical * scale
    
    def run_grid_search(
        self,
        wavelengths: np.ndarray,
        measured: np.ndarray,
        lipid_range: Tuple[float, float, float] = (0, 400, 20),  # Matches LTA app coarse search
        aqueous_range: Tuple[float, float, float] = (-20, 6000, 200),  # Matches LTA app coarse search
        roughness_range: Tuple[float, float, float] = (300, 3000, 300),  # Matches LTA app: 300-3000 Å, step 300 Å
        top_k: int = 10,
        enable_roughness: bool = True,
        fine_search: bool = True,  # Enable fine refinement around top candidates
        fine_refinement_factor: float = 0.2,  # Refine to 20% of original step size
    ) -> List[PyElliResult]:
        """
        Run coarse grid search with monotonic alignment scoring.
        
        Uses Bruggeman EMA for interface roughness modeling.
        
        Args:
            wavelengths: Wavelength array
            measured: Measured reflectance
            lipid_range: (min, max, step) for lipid thickness in nm
            aqueous_range: (min, max, step) for aqueous thickness in nm
            roughness_range: (min, max, step) for interface roughness in Angstroms (Å)
                            LTA range: 300-3000 Å, step 300 Å
            top_k: Number of top results to return
            enable_roughness: If True, use Bruggeman EMA roughness modeling
            
        Returns:
            List of top results sorted by score (descending)
        """
        results: List[PyElliResult] = []
        
        # Generate parameter grid
        lipid_values = np.arange(lipid_range[0], lipid_range[1] + 1, lipid_range[2])
        aqueous_values = np.arange(aqueous_range[0], aqueous_range[1] + 1, aqueous_range[2])
        roughness_values_angstrom = np.arange(roughness_range[0], roughness_range[1] + 1, roughness_range[2])
        
        # Filter out negative values
        lipid_values = lipid_values[lipid_values >= 0]
        aqueous_values = aqueous_values[aqueous_values >= 0]
        roughness_values_angstrom = roughness_values_angstrom[roughness_values_angstrom >= 0]
        
        total = len(lipid_values) * len(aqueous_values) * len(roughness_values_angstrom)
        roughness_status = 'Bruggeman EMA' if enable_roughness else 'disabled'
        
        # Prepare material data for worker processes
        material_data = {
            'lipid': {
                'wavelength_nm': self.lipid_df['wavelength_nm'].values,
                'n': self.lipid_df['n'].values,
                'k': self.lipid_df['k'].values,
            },
            'water': {
                'wavelength_nm': self.water_df['wavelength_nm'].values,
                'n': self.water_df['n'].values,
                'k': self.water_df['k'].values,
            },
            'mucus': {
                'wavelength_nm': self.mucus_df['wavelength_nm'].values,
                'n': self.mucus_df['n'].values,
                'k': self.mucus_df['k'].values,
            },
            'substratum': {
                'wavelength_nm': self.substratum_df['wavelength_nm'].values,
                'n': self.substratum_df['n'].values,
                'k': self.substratum_df['k'].values,
            },
        }
        
        # Determine number of workers (use CPU count, but cap at 10 for safety)
        num_workers = min(os.cpu_count() or 4, 10)
        logger.info(f'🔍 Running parallel grid search: {total} combinations, {num_workers} workers (roughness: {roughness_status})')
        
        # Generate all parameter combinations
        combinations = [
            (lipid, aqueous, roughness_angstrom)
            for lipid in lipid_values
            for aqueous in aqueous_values
            for roughness_angstrom in roughness_values_angstrom
        ]
        
        # Parallel evaluation
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    _evaluate_single_combination,
                    wavelengths,
                    measured,
                    lipid,
                    aqueous,
                    roughness_angstrom,
                    material_data,
                    enable_roughness,
                ): (lipid, aqueous, roughness_angstrom)
                for lipid, aqueous, roughness_angstrom in combinations
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 100 == 0:
                    logger.debug(f'Progress: {completed}/{total} ({100*completed/total:.1f}%)')
                
                result = future.result()
                if result is not None:
                    results.append(result)
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Fine refinement around top candidates
        if fine_search and results:
            logger.info(f'🔬 Refining top {min(5, len(results))} candidates with fine search...')
            top_candidates = results[:min(5, len(results))]
            refined_results = []
            
            # Prepare material data for worker processes
            material_data = {
                'lipid': {
                    'wavelength_nm': self.lipid_df['wavelength_nm'].values,
                    'n': self.lipid_df['n'].values,
                    'k': self.lipid_df['k'].values,
                },
                'water': {
                    'wavelength_nm': self.water_df['wavelength_nm'].values,
                    'n': self.water_df['n'].values,
                    'k': self.water_df['k'].values,
                },
                'mucus': {
                    'wavelength_nm': self.mucus_df['wavelength_nm'].values,
                    'n': self.mucus_df['n'].values,
                    'k': self.mucus_df['k'].values,
                },
                'substratum': {
                    'wavelength_nm': self.substratum_df['wavelength_nm'].values,
                    'n': self.substratum_df['n'].values,
                    'k': self.substratum_df['k'].values,
                },
            }
            
            # Generate fine search combinations
            fine_combinations = []
            for candidate in top_candidates:
                # Create fine search ranges around each candidate
                fine_lipid_step = max(1.0, lipid_range[2] * fine_refinement_factor)
                fine_aqueous_step = max(10.0, aqueous_range[2] * fine_refinement_factor)
                fine_roughness_step = max(10.0, roughness_range[2] * fine_refinement_factor)
                
                # Search in a window around the candidate (±2 coarse steps)
                fine_lipid_min = max(lipid_range[0], candidate.lipid_nm - 2 * lipid_range[2])
                fine_lipid_max = min(lipid_range[1], candidate.lipid_nm + 2 * lipid_range[2])
                fine_lipid_values = np.arange(fine_lipid_min, fine_lipid_max + fine_lipid_step, fine_lipid_step)
                
                fine_aqueous_min = max(aqueous_range[0], candidate.aqueous_nm - 2 * aqueous_range[2])
                fine_aqueous_max = min(aqueous_range[1], candidate.aqueous_nm + 2 * aqueous_range[2])
                fine_aqueous_values = np.arange(fine_aqueous_min, fine_aqueous_max + fine_aqueous_step, fine_aqueous_step)
                
                fine_roughness_min = max(roughness_range[0], candidate.mucus_nm - 2 * roughness_range[2])
                fine_roughness_max = min(roughness_range[1], candidate.mucus_nm + 2 * roughness_range[2])
                fine_roughness_values = np.arange(fine_roughness_min, fine_roughness_max + fine_roughness_step, fine_roughness_step)
                
                # Filter out negative values
                fine_lipid_values = fine_lipid_values[fine_lipid_values >= 0]
                fine_aqueous_values = fine_aqueous_values[fine_aqueous_values >= 0]
                fine_roughness_values = fine_roughness_values[fine_roughness_values >= 0]
                
                for lipid in fine_lipid_values:
                    for aqueous in fine_aqueous_values:
                        for roughness_angstrom in fine_roughness_values:
                            # Skip if this is the original candidate (already evaluated)
                            if (abs(lipid - candidate.lipid_nm) < 0.1 and 
                                abs(aqueous - candidate.aqueous_nm) < 1.0 and 
                                abs(roughness_angstrom - candidate.mucus_nm) < 1.0):
                                continue
                            fine_combinations.append((lipid, aqueous, roughness_angstrom))
            
            # Parallel fine search
            num_workers = min(os.cpu_count() or 4, 10)
            logger.info(f'🔬 Refining {len(fine_combinations)} combinations with {num_workers} workers...')
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        _evaluate_single_combination,
                        wavelengths,
                        measured,
                        lipid,
                        aqueous,
                        roughness_angstrom,
                        material_data,
                        enable_roughness,
                    ): (lipid, aqueous, roughness_angstrom)
                    for lipid, aqueous, roughness_angstrom in fine_combinations
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        refined_results.append(result)
        
            # Combine coarse and refined results, sort by score
            all_results = results + refined_results
            all_results.sort(key=lambda x: x.score, reverse=True)
            
            # Remove duplicates (keep best score for same parameters)
            seen = set()
            unique_results = []
            for r in all_results:
                key = (round(r.lipid_nm, 1), round(r.aqueous_nm, 1), round(r.mucus_nm, 0))
                if key not in seen:
                    seen.add(key)
                    unique_results.append(r)
            
            results = unique_results[:top_k]
            logger.info(f'✅ Fine search completed. Best refined score: {results[0].score:.4f}')
        
        if results:
            best = results[0]
            # Get peak metrics from the best result by recalculating
            best_theoretical = best.theoretical_spectrum
            best_score_result = calculate_peak_based_score(
                wavelengths, measured, best_theoretical
            )
            logger.info(
                f'✅ Best fit: Lipid={best.lipid_nm:.1f}nm, '
                f'Aqueous={best.aqueous_nm:.1f}nm, Roughness={best.mucus_nm:.0f}Å, '
                f'Score={best.score:.4f}, Matched Peaks={best_score_result.get("matched_peaks", 0):.0f}, '
                f'Mean Delta={best_score_result.get("mean_delta_nm", 0):.2f}nm, '
                f'Meas Peaks={best_score_result.get("measurement_peaks", 0):.0f}, '
                f'Theo Peaks={best_score_result.get("theoretical_peaks", 0):.0f}, '
                f'Correlation={best.correlation:.3f}'
            )
            # Log top 3 results for debugging
            for i, r in enumerate(results[:min(3, len(results))]):
                r_score = calculate_peak_based_score(wavelengths, measured, r.theoretical_spectrum)
                logger.debug(
                    f'  Rank {i+1}: L={r.lipid_nm:.1f}, A={r.aqueous_nm:.1f}, R={r.mucus_nm:.0f}Å, '
                    f'Score={r.score:.4f}, Matched={r_score.get("matched_peaks", 0):.0f}, '
                    f'Delta={r_score.get("mean_delta_nm", 0):.2f}nm'
                )
        
        return results[:top_k]
    
    def refine_around_best(
        self,
        wavelengths: np.ndarray,
        measured: np.ndarray,
        best_result: PyElliResult,
        lipid_step: float = 5.0,
        aqueous_step: float = 50.0,
        roughness_step: float = 25.0,
        enable_roughness: bool = True,
    ) -> List[PyElliResult]:
        """
        Fine-grain search around best coarse result.
        
        Args:
            wavelengths: Wavelength array
            measured: Measured reflectance
            best_result: Best result from coarse search
            lipid_step: Fine step for lipid (nm)
            aqueous_step: Fine step for aqueous (nm)
            roughness_step: Fine step for roughness (nm)
            enable_roughness: If True, use Bruggeman EMA roughness
            
        Returns:
            Refined top results
        """
        # Define search window around best
        lipid_range = (
            max(0, best_result.lipid_nm - 30),
            min(400, best_result.lipid_nm + 30),
            lipid_step
        )
        aqueous_range = (
            max(0, best_result.aqueous_nm - 300),
            min(6000, best_result.aqueous_nm + 300),
            aqueous_step
        )
        roughness_range = (
            max(30, best_result.mucus_nm - 100),
            min(300, best_result.mucus_nm + 100),
            roughness_step
        )
        
        logger.info(f'🔬 Refining around best result...')
        return self.run_grid_search(
            wavelengths, measured,
            lipid_range, aqueous_range, roughness_range,
            top_k=10,
            enable_roughness=enable_roughness,
        )


# =============================================================================
# Demo / CLI
# =============================================================================

def demo():
    """Run a demo of PyElli grid search with Bruggeman EMA roughness modeling."""
    import matplotlib.pyplot as plt
    
    # Setup paths
    materials_path = PROJECT_ROOT / 'data' / 'Materials'
    sample_data_path = PROJECT_ROOT / 'exploration' / 'sample_data' / 'good_fit' / '1'
    
    # Find sample file
    sample_files = list(sample_data_path.glob('(Run)spectra_*.txt'))
    sample_files = [f for f in sample_files if '_BestFit' not in f.name]
    
    if not sample_files:
        logger.error('❌ No sample files found!')
        return
    
    sample_file = sample_files[0]
    logger.info(f'📂 Loading sample: {sample_file.name}')
    
    # Load measured spectrum
    wavelengths, measured = load_measured_spectrum(sample_file)
    
    # Initialize grid search
    grid_search = PyElliGridSearch(materials_path)
    
    # Run coarse search with Bruggeman EMA roughness
    print('\n' + '=' * 70)
    print('COARSE GRID SEARCH (Bruggeman EMA roughness + monotonic alignment)')
    print('=' * 70)
    
    coarse_results = grid_search.run_grid_search(
        wavelengths, measured,
        lipid_range=(20, 150, 20),
        aqueous_range=(1000, 4500, 250),
        roughness_range=(30, 270, 60),  # 30-270 nm = 300-2700 Å
        enable_roughness=True,
    )
    
    print('\nTop 5 results:')
    print('-' * 90)
    header = f"{'Rank':<6} {'Lipid(nm)':<12} {'Aqueous(nm)':<14} {'Roughness(nm)':<14} {'Score':<10} {'RMSE':<10} {'Crossings':<10}"
    print(header)
    print('-' * 90)
    
    for i, r in enumerate(coarse_results[:5], 1):
        print(f'{i:<6} {r.lipid_nm:<12.1f} {r.aqueous_nm:<14.1f} {r.mucus_nm:<14.1f} {r.score:<10.4f} {r.rmse:<10.6f} {r.crossing_count:<10}')
    
    # Refine around best
    if coarse_results:
        print('\n' + '=' * 70)
        print('FINE REFINEMENT (Bruggeman EMA)')
        print('=' * 70)
        
        refined_results = grid_search.refine_around_best(
            wavelengths, measured, coarse_results[0],
            enable_roughness=True,
        )
        
        print('\nRefined Top 5:')
        print('-' * 90)
        for i, r in enumerate(refined_results[:5], 1):
            print(f'{i:<6} {r.lipid_nm:<12.1f} {r.aqueous_nm:<14.1f} {r.mucus_nm:<14.1f} {r.score:<10.4f} {r.rmse:<10.6f} {r.crossing_count:<10}')
        
        # Plot best result
        best = refined_results[0]
        
        plt.figure(figsize=(12, 5))
        
        # Main comparison plot
        plt.subplot(1, 2, 1)
        plt.plot(wavelengths, measured, 'b-', linewidth=2, label='Measured')
        plt.plot(best.wavelengths, best.theoretical_spectrum, 'r--', linewidth=2, 
                label=f'PyElli+Bruggeman (L={best.lipid_nm:.0f}, A={best.aqueous_nm:.0f}, R={best.mucus_nm:.0f}nm)')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance')
        plt.title('PyElli Auto-Fit with Bruggeman EMA Roughness')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Residual plot
        plt.subplot(1, 2, 2)
        residual = measured[:len(best.theoretical_spectrum)] - best.theoretical_spectrum
        plt.plot(wavelengths[:len(residual)], residual, 'g-', linewidth=1)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Residual')
        plt.title(f'Residual (Crossings: {best.crossing_count}, Score: {best.score:.3f})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = PROJECT_ROOT / 'exploration' / 'pyelli_exploration' / 'grid_search_result.png'
        plt.savefig(output_path, dpi=150)
        plt.show()
        
        print(f'\n✅ Plot saved to: {output_path}')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    demo()

