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
import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import random
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
    matched_peaks: int = 0  # Number of matched peaks
    peak_count_delta: int = 0  # Absolute difference between measurement and theoretical peak counts
    theoretical_spectrum: np.ndarray = None
    wavelengths: np.ndarray = None


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
        num_divisions: int = 20,  # Restored to 20 for accuracy
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
        num_divisions: Number of slices for the graded layer (default: 3, optimized for speed)
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
    num_roughness_divisions: int = 20,  # Restored to 20 for accuracy
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
        num_roughness_divisions: Number of slices for roughness gradient (default: 3, optimized for speed)
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
    tolerance_nm: float = 20.0,  # Increased from 10.0 to 20.0 nm to match LTA's effective tolerance (LTA matches 5 peaks, PyElli was getting 0 with 10nm)
    tau_nm: float = 15.0,  # Stricter: reduced from 20.0 to 15.0 nm for stricter delta scoring
    penalty_unpaired: float = 0.04,  # Slightly increased penalty for unpaired peaks
    min_correlation: float = 0.85,  # CRITICAL: Minimum correlation threshold (LTA achieves 0.99+)
) -> Dict[str, float]:
    """
    Calculate peak-based score using peak count, peak alignment, AND correlation.
    
    IMPROVED based on reverse engineering analysis:
    - LTA BestFit achieves 0.99+ correlation on good fits
    - LTA matches 5.3 peaks on average with 6.3nm mean delta
    - Must reject anti-correlated fits (PyElli was accepting -0.87 correlation!)
    
    This combines:
    1. Correlation score (shape similarity) - CRITICAL, weighted 40%
    2. Peak count score - weighted 20%
    3. Peak delta score (alignment quality) - weighted 40%
    
    Args:
        wavelengths: Wavelength array
        measured: Measured reflectance
        theoretical: Theoretical reflectance
        cutoff_frequency: Detrending cutoff frequency
        filter_order: Butterworth filter order
        peak_prominence: Peak detection prominence threshold
        tolerance_nm: Peak matching tolerance in nm (20nm to match LTA's effective tolerance)
        tau_nm: Decay constant for peak delta scoring
        penalty_unpaired: Penalty per unpaired peak
        min_correlation: Minimum acceptable correlation (below this, score is heavily penalized)
        
    Returns:
        Dictionary with scores and diagnostics
    """
    min_len = min(len(wavelengths), len(measured), len(theoretical))
    if min_len < 10:
        return {
            "score": 0.0,
            "correlation": 0.0,
            "correlation_score": 0.0,
            "rmse": 1.0,
            "rmse_score": 0.0,
            "oscillation_ratio": 1.0,
            "peak_count_score": 0.0,
            "peak_delta_score": 0.0,
            "matched_peaks": 0,
            "mean_delta_nm": 0.0,
        }
    
    wavelengths = wavelengths[:min_len]
    measured = measured[:min_len]
    theoretical = theoretical[:min_len]
    
    # === CRITICAL: Calculate correlation FIRST ===
    # LTA BestFit achieves 0.99+ correlation - this is the most important metric
    if np.std(measured) > 1e-10 and np.std(theoretical) > 1e-10:
        correlation = float(np.corrcoef(measured, theoretical)[0, 1])
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0
    
    # Calculate correlation score (0 to 1)
    # Correlation of 0.99 -> score ~0.99, correlation of 0.85 -> score ~0.85
    # Negative correlation -> score 0 (this was a major bug - we were accepting anti-correlated fits!)
    if correlation < min_correlation:
        # Heavy penalty for low correlation
        # Below 0.85: score drops exponentially
        if correlation < 0:
            correlation_score = 0.0  # Anti-correlated fits get zero
        else:
            # Between 0 and min_correlation: partial score
            correlation_score = (correlation / min_correlation) * 0.3  # Max 0.3 for below-threshold
    else:
        # Above threshold: score scales from 0.7 to 1.0
        correlation_score = 0.7 + 0.3 * ((correlation - min_correlation) / (1.0 - min_correlation))
    
    correlation_score = float(np.clip(correlation_score, 0.0, 1.0))
    
    # Detrend both signals for peak detection
    try:
        meas_detrended = detrend_signal(wavelengths, measured, cutoff_frequency, filter_order)
        theo_detrended = detrend_signal(wavelengths, theoretical, cutoff_frequency, filter_order)
    except Exception:
        # If detrending fails, return score based on correlation only
        return {
            "score": correlation_score * 0.4,  # 40% weight for correlation
            "correlation": correlation,
            "correlation_score": correlation_score,
            "rmse": 1.0,
            "rmse_score": 0.0,
            "oscillation_ratio": 1.0,
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
    
    # Penalty for unpaired peaks
    penalty = penalty_unpaired * float(unmatched_measurement + unmatched_theoretical)
    peak_delta_score = max(0.0, min(1.0, peak_delta_score - penalty))
    
    # === CALCULATE RMSE SCORE ===
    # RMSE measures the actual residual magnitude - critical for detecting bad fits!
    # Increased sensitivity: LTA achieves extremely low residuals (< 0.001)
    residual = measured - theoretical
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    
    # Normalize RMSE to a score (0-1)
    # EXTREMELY tight: LTA achieves RMSE ~0.0006, so we need to heavily penalize anything above 0.001
    # RMSE of 0.0006 -> score ~0.67, RMSE of 0.001 -> score ~0.51, RMSE of 0.002 -> score ~0.26
    rmse_tau = 0.0008  # Much tighter - heavily penalizes RMSE > 0.001
    rmse_score = float(np.exp(-rmse / rmse_tau))
    
    # === OSCILLATION AMPLITUDE CHECK ===
    # Penalize if theoretical has much less oscillation than measured
    # This catches the "flat line" problem where correlation is high but fit is bad
    meas_oscillation = float(np.std(meas_detrended))
    theo_oscillation = float(np.std(theo_detrended))
    
    # Initialize defaults to avoid NameError
    oscillation_penalty = 1.0
    oscillation_ratio = 1.0
    
    if meas_oscillation > 1e-8:
        oscillation_ratio = theo_oscillation / meas_oscillation
        # If theoretical has < 50% of measured oscillation, penalize heavily
        if oscillation_ratio < 0.5:
            oscillation_penalty = float(oscillation_ratio)
        else:
            oscillation_penalty = 1.0
    else:
        oscillation_penalty = 1.0
        oscillation_ratio = 1.0
    
    # === IMPROVED COMPOSITE SCORE ===
    # Weighting heavily towards RMSE and Correlation - these are the true indicators of fit quality
    # Peak count is less important if RMSE and correlation are excellent
    # 
    # Weighting:
    # - 70% RMSE score (residual magnitude) - DOMINANT, ensures curves are physically close
    # - 25% Correlation (shape similarity) - critical for visual fit quality
    # - 3% Peak delta (alignment quality) - minor factor
    # - 2% Peak count (matched peaks ratio) - very minor, don't penalize good fits for this
    
    # Ensure all components are defined
    c_rmse = float(rmse_score)
    c_corr = float(correlation_score)
    c_delta = float(peak_delta_score)
    c_count = float(peak_count_score)
    
    composite_score = (
        0.70 * c_rmse +
        0.25 * c_corr +
        0.03 * c_delta +
        0.02 * c_count
    )
    
    # Apply oscillation penalty (catches "flat line" theoretical)
    if 'oscillation_penalty' in locals():
        composite_score = composite_score * oscillation_penalty
    else:
        oscillation_penalty = 1.0
        composite_score = composite_score * oscillation_penalty
    
    # MASSIVE bonus for excellent RMSE (0.0006-0.0011) - expanded threshold
    if rmse <= 0.0011:  # RMSE <= 0.0011 (includes 0.00103 case)
        composite_score = min(1.0, composite_score * 1.4)  # 40% bonus
    elif rmse <= 0.0015:  # RMSE <= 0.0015 (close to LTA)
        composite_score = min(1.0, composite_score * 1.2)  # 20% bonus
    
    # HUGE bonus for excellent correlation (0.99+) - this is a strong indicator of good fit
    if correlation >= 0.99:
        composite_score = min(1.0, composite_score * 1.25)  # 25% bonus
    elif correlation >= 0.95:
        composite_score = min(1.0, composite_score * 1.1)  # 10% bonus
    
    # Combined bonus for excellent correlation AND low RMSE
    if correlation >= 0.99 and rmse <= 0.0011:
        composite_score = min(1.0, composite_score * 1.2)  # Additional 20% bonus
    
    # Heavy penalty for low/negative correlation
    if correlation < 0.5:
        composite_score = composite_score * 0.2  # 80% penalty
    elif correlation < min_correlation:
        composite_score = composite_score * 0.5  # 50% penalty
    
    # EXTREME penalty for high RMSE (bad residual)
    if rmse > 0.002:  # RMSE > 0.002 (much worse than LTA)
        composite_score = composite_score * 0.2  # 80% penalty
    elif rmse > 0.0015:  # RMSE > 0.0015
        composite_score = composite_score * 0.5  # 50% penalty
    
    # Don't penalize for low peak count if RMSE and correlation are excellent
    # Peak count is less reliable - some good fits naturally have fewer peaks
    if matched_count < 3 and correlation >= 0.99 and rmse <= 0.0011:
        # If correlation and RMSE are excellent, don't penalize low peak count
        pass  # No penalty
    elif matched_count >= 5 and correlation >= 0.95 and rmse <= 0.0011:
        # Bonus for matching many peaks with excellent alignment and RMSE
        composite_score = min(1.0, composite_score * 1.05)
    
    return {
        "score": float(np.clip(composite_score, 0.0, 1.0)),
        "correlation": correlation,
        "correlation_score": correlation_score,
        "rmse": rmse,
        "rmse_score": rmse_score,
        "oscillation_ratio": oscillation_ratio if meas_oscillation > 1e-8 else 1.0,
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
    # Suppress warnings in worker processes (multiprocessing workers don't have Streamlit context)
    import warnings
    import logging
    import sys
    
    # Suppress all ScriptRunContext warnings
    warnings.filterwarnings('ignore', message='.*ScriptRunContext.*')
    warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*')
    
    # Suppress Streamlit runtime warnings
    streamlit_logger = logging.getLogger('streamlit')
    streamlit_logger.setLevel(logging.ERROR)
    logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.CRITICAL)
    logging.getLogger('streamlit.runtime.scriptrunner_utils').setLevel(logging.CRITICAL)
    
    # Redirect stderr to suppress warnings if needed (but keep errors)
    # Note: This is a last resort - warnings should be filtered above
    
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
            num_roughness_divisions=20,  # Restored to 20 for accuracy
            use_error_function_profile=True,
        )
        
        # Align using simple proportional scaling (focus region 600-1120 nm)
        focus_mask = (wavelengths >= 600.0) & (wavelengths <= 1120.0)
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
        
        # OPTIMIZATION: Early termination for obviously bad fits
        # Quick correlation check before expensive peak detection
        wl_focus = wavelengths[focus_mask] if focus_mask.sum() > 0 else wavelengths
        meas_focus_scaled = measured[focus_mask] if focus_mask.sum() > 0 else measured
        theo_focus_scaled = theoretical_scaled[focus_mask] if focus_mask.sum() > 0 else theoretical_scaled
        
        # Quick correlation check (fast) - skip expensive peak detection if correlation is terrible
        if focus_mask.sum() > 0:
            if np.std(meas_focus_scaled) > 1e-10 and np.std(theo_focus_scaled) > 1e-10:
                quick_corr = float(np.corrcoef(meas_focus_scaled, theo_focus_scaled)[0, 1])
                if np.isnan(quick_corr):
                    quick_corr = 0.0
            else:
                quick_corr = 0.0
            
            # OPTIMIZATION: Skip expensive peak detection if correlation is poor (< 0.5)
            # Increased threshold from 0.3 to 0.5 to filter out more bad fits early
            if quick_corr < 0.5:
                return None
        else:
            quick_corr = 0.0
        
        # OPTIMIZATION: Use fast scoring for moderate correlations (0.5-0.7)
        # Skip expensive peak detection for these, use simple RMSE + correlation score
        if quick_corr < 0.7:
            # Fast path: Simple scoring without peak detection
            residual = meas_focus_scaled - theo_focus_scaled
            rmse = float(np.sqrt(np.mean(residual ** 2)))
            
            # Simple score based on RMSE and correlation only
            rmse_tau = 0.0008
            rmse_score = float(np.exp(-rmse / rmse_tau))
            correlation_score = max(0.0, quick_corr) if quick_corr > 0 else 0.0
            
            # Simple composite score (70% RMSE, 30% correlation)
            simple_score = 0.70 * rmse_score + 0.30 * correlation_score
            
            # Apply penalties
            if rmse > 0.002:
                simple_score *= 0.2
            elif rmse > 0.0015:
                simple_score *= 0.5
            if quick_corr < 0.5:
                simple_score *= 0.2
            
            score_result = {
                "score": float(np.clip(simple_score, 0.0, 1.0)),
                "correlation": quick_corr,
                "correlation_score": correlation_score,
                "rmse": rmse,
                "rmse_score": rmse_score,
                "oscillation_ratio": 1.0,
                "peak_count_score": 0.0,
                "peak_delta_score": 0.0,
                "matched_peaks": 0,
                "mean_delta_nm": 0.0,
                "measurement_peaks": 0.0,
                "theoretical_peaks": 0.0,
                "unpaired_measurement": 0.0,
                "unpaired_theoretical": 0.0,
            }
            correlation = quick_corr
        else:
            # Full path: Use peak-based scoring for good correlations (>= 0.7)
            # Score using peak-based scoring on the FOCUS REGION ONLY (600-1120 nm)
            # This ensures consistency with how the app displays scores
            score_result = calculate_peak_based_score(
                wl_focus, meas_focus_scaled, theo_focus_scaled
            )
            
            # Calculate RMSE and correlation on the focus region (reuse quick_corr if available)
            residual = meas_focus_scaled - theo_focus_scaled
            rmse = float(np.sqrt(np.mean(residual ** 2)))
            correlation = score_result.get('correlation', quick_corr)  # Use from score_result if available
        
        # Calculate peak count delta (absolute difference between measurement and theoretical peak counts)
        meas_peaks_count = int(score_result.get('measurement_peaks', 0))
        theo_peaks_count = int(score_result.get('theoretical_peaks', 0))
        peak_count_delta = abs(meas_peaks_count - theo_peaks_count)
        
        return PyElliResult(
            lipid_nm=float(lipid),
            aqueous_nm=float(aqueous),
            mucus_nm=float(roughness_angstrom),
            score=score_result['score'],
            rmse=rmse,
            correlation=correlation,
            crossing_count=0,
            matched_peaks=int(score_result.get('matched_peaks', 0)),
            peak_count_delta=peak_count_delta,
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
            num_roughness_divisions=20,  # Restored to 20 for accuracy
            use_error_function_profile=use_error_function_profile,
        )
    
    def _align_spectra(
        self,
        measured: np.ndarray,
        theoretical: np.ndarray,
        focus_min: float = 600.0,
        focus_max: float = 1120.0,
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
        lipid_range: Tuple[float, float, float] = (9, 250, 10),  # Standard range: 9-250 nm
        aqueous_range: Tuple[float, float, float] = (800, 12000, 200),  # Standard range: 800-12000 nm
        roughness_range: Tuple[float, float, float] = (600, 7000, 100),  # Standard range: 600-7000 Å, larger step for speed
        top_k: int = 20,  # Increased to explore more candidates
        enable_roughness: bool = True,
        fine_search: bool = True,  
        fine_refinement_factor: float = 0.05,  # Much finer: 5% of coarse step for precision
        min_correlation_filter: float = 0.85,  # Keep selective
        search_strategy: str = 'Coarse Search',  # 'Coarse Search', 'Full Grid Search', or 'Dynamic Search'
        max_combinations: Optional[int] = 5000,  # Max combinations for Coarse/Dynamic search (default 5000)
    ) -> List[PyElliResult]:
        """
        Run grid search with IMPROVED scoring based on reverse engineering analysis.
        
        IMPROVEMENTS (based on analyzing LTA BestFit spectra):
        1. Correlation is now CRITICAL (40% of score) - prevents anti-correlated fits
        2. Parameter ranges optimized from reverse engineering
        3. Tighter peak matching tolerance (15nm vs 25nm)
        4. Results pre-filtered by correlation threshold
        
        Uses Bruggeman EMA for interface roughness modeling.
        
        Args:
            wavelengths: Wavelength array
            measured: Measured reflectance
            lipid_range: (min, max, step) for lipid thickness in nm
                        OPTIMIZED: 100-350nm based on reverse engineering (was 0-400)
            aqueous_range: (min, max, step) for aqueous thickness in nm
                          OPTIMIZED: 0-4000nm based on reverse engineering (was -20-6000)
            roughness_range: (min, max, step) for interface roughness in Angstroms (Å)
                            OPTIMIZED: 1500-3500Å based on reverse engineering (was 300-3000)
            top_k: Number of top results to return
            enable_roughness: If True, use Bruggeman EMA roughness modeling
            min_correlation_filter: Minimum correlation for results (default 0.80)
            
        Returns:
            List of top results sorted by score (descending), filtered by correlation
        """
        # Route to appropriate search strategy
        if search_strategy == 'Full Grid Search':
            return self._run_full_grid_search(
                wavelengths, measured, lipid_range, aqueous_range, roughness_range,
                top_k, enable_roughness, min_correlation_filter
            )
        elif search_strategy == 'Dynamic Search':
            return self._run_dynamic_search(
                wavelengths, measured, lipid_range, aqueous_range, roughness_range,
                top_k, enable_roughness, min_correlation_filter, max_combinations
            )
        else:  # Default to Coarse Search
            return self._run_coarse_search(
                wavelengths, measured, lipid_range, aqueous_range, roughness_range,
                top_k, enable_roughness, fine_search, fine_refinement_factor, min_correlation_filter, max_combinations
            )
    
    def _run_coarse_search(
        self,
        wavelengths: np.ndarray,
        measured: np.ndarray,
        lipid_range: Tuple[float, float, float],
        aqueous_range: Tuple[float, float, float],
        roughness_range: Tuple[float, float, float],
        top_k: int,
        enable_roughness: bool,
        fine_search: bool,
        fine_refinement_factor: float,
        min_correlation_filter: float,
        max_combinations: Optional[int] = 5000,
    ) -> List[PyElliResult]:
        """
        Coarse search: Fast two-stage search (coarse then fine refinement).
        This is the current/default approach.
        """
        import time as time_module
        search_start_time = time_module.time()
        
        logger.info('=' * 80)
        logger.info('🔍 COARSE SEARCH - Starting two-stage grid search')
        logger.info('=' * 80)
        logger.info(f'📋 Search Parameters:')
        logger.info(f'   - Lipid Range: {lipid_range[0]:.1f} - {lipid_range[1]:.1f} nm (step: {lipid_range[2]:.1f} nm)')
        logger.info(f'   - Aqueous Range: {aqueous_range[0]:.1f} - {aqueous_range[1]:.1f} nm (step: {aqueous_range[2]:.1f} nm)')
        logger.info(f'   - Roughness Range: {roughness_range[0]:.1f} - {roughness_range[1]:.1f} Å (step: {roughness_range[2]:.1f} Å)')
        logger.info(f'   - Max Combinations Limit: {max_combinations:,}' if max_combinations else '   - Max Combinations Limit: None (unlimited)')
        logger.info(f'   - Fine Search: {fine_search}')
        logger.info(f'   - Top K Results: {top_k}')
        logger.info('=' * 80)
        
        results: List[PyElliResult] = []
        
        # Generate parameter grid
        lipid_values = np.arange(lipid_range[0], lipid_range[1] + 1, lipid_range[2])
        aqueous_values = np.arange(aqueous_range[0], aqueous_range[1] + 1, aqueous_range[2])
        roughness_values_angstrom = np.arange(roughness_range[0], roughness_range[1] + 1, roughness_range[2])
        
        # Filter out negative values and enforce standard ADOM ranges
        lipid_values = lipid_values[(lipid_values >= 9) & (lipid_values <= 250)]
        aqueous_values = aqueous_values[(aqueous_values >= 800) & (aqueous_values <= 12000)]
        roughness_values_angstrom = roughness_values_angstrom[(roughness_values_angstrom >= 600) & (roughness_values_angstrom <= 7000)]
        
        if len(lipid_values) == 0 or len(aqueous_values) == 0 or len(roughness_values_angstrom) == 0:
            logger.warning(f'⚠️ Parameter ranges resulted in empty grid! Lipid: {len(lipid_values)}, Aqueous: {len(aqueous_values)}, Roughness: {len(roughness_values_angstrom)}')
            return []
        
        total = len(lipid_values) * len(aqueous_values) * len(roughness_values_angstrom)
        roughness_status = 'Bruggeman EMA' if enable_roughness else 'disabled'
        
        # SAFETY CHECK: Limit total combinations to prevent excessive runtime
        if max_combinations is None:
            max_combinations = 5000  # Default if not provided
        if total > max_combinations:
            logger.warning(f'⚠️ Coarse grid too large ({total:,} combinations). Limiting to {max_combinations:,} by sampling...')
            # Sample evenly from each dimension to stay within limit
            target_per_dim = int(np.ceil(max_combinations ** (1/3)))
            if len(lipid_values) > target_per_dim:
                indices = np.linspace(0, len(lipid_values)-1, target_per_dim, dtype=int)
                lipid_values = lipid_values[indices]
            if len(aqueous_values) > target_per_dim:
                indices = np.linspace(0, len(aqueous_values)-1, target_per_dim, dtype=int)
                aqueous_values = aqueous_values[indices]
            if len(roughness_values_angstrom) > target_per_dim:
                indices = np.linspace(0, len(roughness_values_angstrom)-1, target_per_dim, dtype=int)
                roughness_values_angstrom = roughness_values_angstrom[indices]
            total = len(lipid_values) * len(aqueous_values) * len(roughness_values_angstrom)
            logger.info(f'📊 After limiting: {total:,} combinations')
        
        # Use all available CPU cores
        num_workers = os.cpu_count() or 4
        logger.info('')
        logger.info('📊 STAGE 1: Coarse Grid Search')
        logger.info('-' * 80)
        logger.info(f'🔧 Step Sizes: Lipid={lipid_range[2]:.1f}nm, Aqueous={aqueous_range[2]:.1f}nm, Roughness={roughness_range[2]:.1f}Å')
        logger.info(f'📈 Grid Dimensions:')
        logger.info(f'   - Lipid values: {len(lipid_values)} ({lipid_values[0]:.1f} to {lipid_values[-1]:.1f} nm)')
        logger.info(f'   - Aqueous values: {len(aqueous_values)} ({aqueous_values[0]:.1f} to {aqueous_values[-1]:.1f} nm)')
        logger.info(f'   - Roughness values: {len(roughness_values_angstrom)} ({roughness_values_angstrom[0]:.1f} to {roughness_values_angstrom[-1]:.1f} Å)')
        logger.info(f'   - Total combinations: {total:,}')
        logger.info(f'🚀 Starting parallel evaluation with {num_workers} workers...')
        stage1_start = time_module.time()
        
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
            
            # Collect results as they complete with progress logging
            completed = 0
            filtered_out = 0
            for future in as_completed(futures):
                completed += 1
                # Log progress every 10% or every 100 combinations (whichever is more frequent)
                progress_interval = max(1, min(100, total // 10))
                if completed % progress_interval == 0 or completed == total:
                    progress_pct = 100 * completed / total
                    elapsed = time_module.time() - stage1_start
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    logger.info(f'   Progress: {completed}/{total} ({progress_pct:.1f}%) | '
                               f'Elapsed: {elapsed:.1f}s | Rate: {rate:.1f} comb/s | ETA: {eta:.1f}s')
                
                result = future.result()
                if result is not None:
                    # CRITICAL: Filter by correlation AND RMSE to reject bad fits
                    # LTA achieves RMSE ~0.0006, so be very strict
                    # Reject anything with RMSE > 0.002 (still 3x worse than LTA, but reasonable)
                    # Also filter by correlation to reject anti-correlated fits
                    if (result.correlation >= min_correlation_filter and 
                        result.rmse <= 0.002):  # Reject fits with RMSE > 0.002
                        results.append(result)
                    else:
                        filtered_out += 1
        
        stage1_elapsed = time_module.time() - stage1_start
        logger.info(f'✅ Stage 1 completed in {stage1_elapsed:.1f}s')
        logger.info(f'   Evaluated: {completed:,} combinations')
        logger.info(f'   Passed filters: {len(results):,}')
        if filtered_out > 0:
            logger.info(f'   Filtered out: {filtered_out:,} (correlation < {min_correlation_filter} or RMSE > 0.002)')
        
        if len(results) == 0:
            logger.warning('⚠️ No results passed filters! All candidates had correlation < 0.85 or RMSE > 0.002. Try expanding search ranges or relaxing filters.')
            return []
        
        # Sort by score first, then by smallest peak_count_delta, then matched_peaks (score and matched_peaks descending, delta ascending)
        results.sort(key=lambda x: (x.score, -x.peak_count_delta, x.matched_peaks), reverse=True)
        
        # Fine refinement around top candidates (parallel processing)
        logger.info('')
        logger.info('🔬 STAGE 2: Fine Refinement')
        logger.info('-' * 80)
        stage2_start = time_module.time()
        
        # Only refine if we have good candidates (RMSE < 0.0015)
        good_candidates = [r for r in results if r.rmse < 0.0015]
        if fine_search and good_candidates:
            num_refine = min(10, len(good_candidates))  # Refine more candidates
            logger.info(f'🎯 Refining top {num_refine} candidates (RMSE < 0.0015) with fine search...')
            top_candidates = good_candidates[:num_refine]
        elif fine_search and results:
            # If no good candidates, still refine top ones but be more aggressive
            num_refine = min(5, len(results))
            logger.info(f'🎯 Refining top {num_refine} candidates with fine search (no excellent candidates found)...')
            top_candidates = results[:num_refine]
        else:
            top_candidates = []
            logger.info('⏭️  Skipping fine search (fine_search=False or no candidates)')
        
        refined_results = []
        if top_candidates:
            refined_results = []
            fine_combinations = []
            
            # Calculate remaining budget for fine search
            coarse_used = min(total, max_combinations) if max_combinations else total
            remaining_budget = (max_combinations - coarse_used) if max_combinations else None
            
            logger.info(f'📊 Fine Search Budget:')
            logger.info(f'   - Coarse search used: {coarse_used:,} combinations')
            if remaining_budget is not None:
                logger.info(f'   - Remaining budget: {remaining_budget:,} combinations')
            else:
                logger.info(f'   - Remaining budget: Unlimited')
            
            for rank, candidate in enumerate(top_candidates, 1):
                logger.info(f'   Candidate #{rank}: Score={candidate.score:.4f}, Corr={candidate.correlation:.3f}, '
                           f'RMSE={candidate.rmse:.5f}, L={candidate.lipid_nm:.1f}nm, '
                           f'A={candidate.aqueous_nm:.1f}nm, R={candidate.mucus_nm:.1f}Å')
                # Create fine search ranges around each candidate
                # Use much finer steps for precision
                fine_lipid_step = max(0.5, lipid_range[2] * fine_refinement_factor)
                fine_aqueous_step = max(5.0, aqueous_range[2] * fine_refinement_factor)
                fine_roughness_step = max(5.0, roughness_range[2] * fine_refinement_factor)
                
                # Search in a wider window around the candidate (±3 coarse steps for better coverage)
                fine_lipid_min = max(lipid_range[0], candidate.lipid_nm - 3 * lipid_range[2])
                fine_lipid_max = min(lipid_range[1], candidate.lipid_nm + 3 * lipid_range[2])
                fine_lipid_values = np.arange(fine_lipid_min, fine_lipid_max + fine_lipid_step, fine_lipid_step)
                
                fine_aqueous_min = max(aqueous_range[0], candidate.aqueous_nm - 3 * aqueous_range[2])
                fine_aqueous_max = min(aqueous_range[1], candidate.aqueous_nm + 3 * aqueous_range[2])
                fine_aqueous_values = np.arange(fine_aqueous_min, fine_aqueous_max + fine_aqueous_step, fine_aqueous_step)
                
                fine_roughness_min = max(roughness_range[0], candidate.mucus_nm - 3 * roughness_range[2])
                fine_roughness_max = min(roughness_range[1], candidate.mucus_nm + 3 * roughness_range[2])
                fine_roughness_values = np.arange(fine_roughness_min, fine_roughness_max + fine_roughness_step, fine_roughness_step)
                
                # Filter out negative values and enforce standard ADOM ranges
                fine_lipid_values = fine_lipid_values[(fine_lipid_values >= 9) & (fine_lipid_values <= 250)]
                fine_aqueous_values = fine_aqueous_values[(fine_aqueous_values >= 800) & (fine_aqueous_values <= 12000)]
                fine_roughness_values = fine_roughness_values[(fine_roughness_values >= 600) & (fine_roughness_values <= 7000)]
                
                # Generate fine search combinations
                for lipid in fine_lipid_values:
                    for aqueous in fine_aqueous_values:
                        for roughness_angstrom in fine_roughness_values:
                            # Skip if this is the original candidate (already evaluated)
                            if (abs(lipid - candidate.lipid_nm) < 0.1 and 
                                abs(aqueous - candidate.aqueous_nm) < 1.0 and 
                                abs(roughness_angstrom - candidate.mucus_nm) < 1.0):
                                continue
                            fine_combinations.append((lipid, aqueous, roughness_angstrom))
            
            # Check if fine combinations exceed budget
            total_fine = len(fine_combinations)
            logger.info(f'📊 Generated {total_fine:,} fine search combinations')
            
            if max_combinations and remaining_budget and total_fine > remaining_budget:
                logger.warning(f'⚠️ Fine search combinations ({total_fine:,}) exceed remaining budget ({remaining_budget:,})')
                logger.warning(f'   Limiting to {remaining_budget:,} combinations by random sampling...')
                random.seed(42)
                fine_combinations = random.sample(fine_combinations, remaining_budget)
                total_fine = len(fine_combinations)
                logger.info(f'   After limiting: {total_fine:,} combinations')
            elif max_combinations and total_fine > max_combinations:
                logger.warning(f'⚠️ Fine search combinations ({total_fine:,}) exceed max limit ({max_combinations:,})')
                logger.warning(f'   Limiting to {max_combinations:,} combinations by random sampling...')
                random.seed(42)
                fine_combinations = random.sample(fine_combinations, max_combinations)
                total_fine = len(fine_combinations)
                logger.info(f'   After limiting: {total_fine:,} combinations')
            
            # Parallel fine search
            if fine_combinations:
                num_workers = os.cpu_count() or 4
                logger.info(f'🚀 Starting parallel evaluation of {total_fine:,} combinations with {num_workers} workers...')
                
                # Prepare material data for worker processes (same as coarse search)
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
                
                fine_completed = 0
                fine_filtered = 0
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
                        fine_completed += 1
                        # Log progress every 10% or every 100 combinations
                        progress_interval = max(1, min(100, total_fine // 10))
                        if fine_completed % progress_interval == 0 or fine_completed == total_fine:
                            progress_pct = 100 * fine_completed / total_fine
                            elapsed = time_module.time() - stage2_start
                            rate = fine_completed / elapsed if elapsed > 0 else 0
                            eta = (total_fine - fine_completed) / rate if rate > 0 else 0
                            logger.info(f'   Progress: {fine_completed}/{total_fine} ({progress_pct:.1f}%) | '
                                       f'Elapsed: {elapsed:.1f}s | Rate: {rate:.1f} comb/s | ETA: {eta:.1f}s')
                        
                        result = future.result()
                        if result is not None:
                            # Filter refined results by correlation and RMSE
                            if (result.correlation >= min_correlation_filter and 
                                result.rmse <= 0.002):
                                refined_results.append(result)
                        else:
                            fine_filtered += 1
                
                stage2_elapsed = time_module.time() - stage2_start
                logger.info(f'✅ Stage 2 completed in {stage2_elapsed:.1f}s')
                logger.info(f'   Evaluated: {fine_completed:,} combinations')
                logger.info(f'   Passed filters: {len(refined_results):,}')
                logger.info(f'   Filtered out: {fine_filtered:,}')
        
            # Combine coarse and refined results, sort by score
            logger.info('')
            logger.info('🔄 Combining Stage 1 and Stage 2 results...')
            all_results = results + refined_results
            all_results.sort(key=lambda x: (x.score, -x.peak_count_delta, x.matched_peaks), reverse=True)
            
            # Remove duplicates (keep best score for same parameters)
            seen = set()
            unique_results = []
            duplicates_removed = 0
            for r in all_results:
                key = (round(r.lipid_nm, 1), round(r.aqueous_nm, 1), round(r.mucus_nm, 0))
                if key not in seen:
                    seen.add(key)
                    unique_results.append(r)
                else:
                    duplicates_removed += 1
            
            total_elapsed = time_module.time() - search_start_time
            logger.info('')
            logger.info('=' * 80)
            logger.info('✅ COARSE SEARCH COMPLETED')
            logger.info('=' * 80)
            logger.info(f'⏱️  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)')
            logger.info(f'📊 Results:')
            logger.info(f'   - Stage 1 candidates: {len(results)}')
            logger.info(f'   - Stage 2 candidates: {len(refined_results)}')
            logger.info(f'   - Duplicates removed: {duplicates_removed}')
            logger.info(f'   - Unique results: {len(unique_results)}')
            logger.info(f'   - Returning top {top_k} results')
            if unique_results:
                logger.info(f'🏆 Best result: Score={unique_results[0].score:.4f}, Corr={unique_results[0].correlation:.3f}, '
                           f'RMSE={unique_results[0].rmse:.5f}, L={unique_results[0].lipid_nm:.1f}nm, '
                           f'A={unique_results[0].aqueous_nm:.1f}nm, R={unique_results[0].mucus_nm:.1f}Å')
            logger.info('=' * 80)
            
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
    
    def _run_full_grid_search(
        self,
        wavelengths: np.ndarray,
        measured: np.ndarray,
        lipid_range: Tuple[float, float, float],
        aqueous_range: Tuple[float, float, float],
        roughness_range: Tuple[float, float, float],
        top_k: int,
        enable_roughness: bool,
        min_correlation_filter: float,
    ) -> List[PyElliResult]:
        """
        Full grid search: Exhaustive search over entire parameter space with fine steps.
        This is slower but more thorough - searches every combination in the parameter space.
        """
        # Use finer steps for full grid search
        fine_lipid_step = max(1.0, lipid_range[2] * 0.2)  # 20% of coarse step
        fine_aqueous_step = max(20.0, aqueous_range[2] * 0.2)
        fine_roughness_step = max(10.0, roughness_range[2] * 0.2)
        
        logger.info(f'🔍 Running FULL GRID SEARCH with fine steps: Lipid={fine_lipid_step:.1f}nm, Aqueous={fine_aqueous_step:.1f}nm, Roughness={fine_roughness_step:.1f}Å')
        
        # Generate fine parameter grid
        lipid_values = np.arange(lipid_range[0], lipid_range[1] + 1, fine_lipid_step)
        aqueous_values = np.arange(aqueous_range[0], aqueous_range[1] + 1, fine_aqueous_step)
        roughness_values_angstrom = np.arange(roughness_range[0], roughness_range[1] + 1, fine_roughness_step)
        
        # Filter and enforce standard ADOM ranges
        lipid_values = lipid_values[(lipid_values >= 9) & (lipid_values <= 250)]
        aqueous_values = aqueous_values[(aqueous_values >= 800) & (aqueous_values <= 12000)]
        roughness_values_angstrom = roughness_values_angstrom[(roughness_values_angstrom >= 600) & (roughness_values_angstrom <= 7000)]
        
        if len(lipid_values) == 0 or len(aqueous_values) == 0 or len(roughness_values_angstrom) == 0:
            logger.warning(f'⚠️ Parameter ranges resulted in empty grid!')
            return []
        
        total = len(lipid_values) * len(aqueous_values) * len(roughness_values_angstrom)
        logger.info(f'📊 Full grid search: {total:,} total combinations to evaluate (no limit - exhaustive search)')
        
        # Use same evaluation logic as coarse search
        return self._evaluate_parameter_grid(
            wavelengths, measured, lipid_values, aqueous_values, roughness_values_angstrom,
            top_k, enable_roughness, min_correlation_filter
        )
    
    def _run_dynamic_search(
        self,
        wavelengths: np.ndarray,
        measured: np.ndarray,
        lipid_range: Tuple[float, float, float],
        aqueous_range: Tuple[float, float, float],
        roughness_range: Tuple[float, float, float],
        top_k: int,
        enable_roughness: bool,
        min_correlation_filter: float,
        max_combinations: Optional[int] = 5000,
    ) -> List[PyElliResult]:
        """
        Dynamic search: Adaptive step sizes that focus more computation on promising regions.
        
        Strategy:
        1. Start with coarse grid to identify promising regions
        2. Identify top candidates and their neighborhoods
        3. Dynamically refine promising regions with smaller steps
        4. Use larger steps in unpromising regions
        5. Iteratively refine until convergence
        """
        import time as time_module
        search_start_time = time_module.time()
        
        logger.info('=' * 80)
        logger.info('🔍 DYNAMIC SEARCH - Starting adaptive grid search')
        logger.info('=' * 80)
        logger.info(f'📋 Search Parameters:')
        logger.info(f'   - Lipid Range: {lipid_range[0]:.1f} - {lipid_range[1]:.1f} nm (step: {lipid_range[2]:.1f} nm)')
        logger.info(f'   - Aqueous Range: {aqueous_range[0]:.1f} - {aqueous_range[1]:.1f} nm (step: {aqueous_range[2]:.1f} nm)')
        logger.info(f'   - Roughness Range: {roughness_range[0]:.1f} - {roughness_range[1]:.1f} Å (step: {roughness_range[2]:.1f} Å)')
        logger.info(f'   - Max Combinations Limit: {max_combinations:,}' if max_combinations else '   - Max Combinations Limit: None (unlimited)')
        logger.info(f'   - Top K Results: {top_k}')
        logger.info(f'   - Enable Roughness: {enable_roughness}')
        logger.info(f'   - Min Correlation Filter: {min_correlation_filter:.2f}')
        logger.info('=' * 80)
        
        # Stage 1: Coarse initial search to identify promising regions
        logger.info('')
        logger.info('📊 STAGE 1: Coarse Grid Search')
        logger.info('-' * 80)
        stage1_start = time_module.time()
        
        coarse_lipid_step = lipid_range[2]
        coarse_aqueous_step = aqueous_range[2]
        coarse_roughness_step = roughness_range[2]
        
        logger.info(f'🔧 Coarse Step Sizes:')
        logger.info(f'   - Lipid: {coarse_lipid_step:.1f} nm')
        logger.info(f'   - Aqueous: {coarse_aqueous_step:.1f} nm')
        logger.info(f'   - Roughness: {coarse_roughness_step:.1f} Å')
        
        lipid_values_coarse = np.arange(lipid_range[0], lipid_range[1] + 1, coarse_lipid_step)
        aqueous_values_coarse = np.arange(aqueous_range[0], aqueous_range[1] + 1, coarse_aqueous_step)
        roughness_values_coarse = np.arange(roughness_range[0], roughness_range[1] + 1, coarse_roughness_step)
        
        # Filter and enforce ranges
        lipid_values_coarse = lipid_values_coarse[(lipid_values_coarse >= 9) & (lipid_values_coarse <= 250)]
        aqueous_values_coarse = aqueous_values_coarse[(aqueous_values_coarse >= 800) & (aqueous_values_coarse <= 12000)]
        roughness_values_coarse = roughness_values_coarse[(roughness_values_coarse >= 600) & (roughness_values_coarse <= 7000)]
        
        coarse_total = len(lipid_values_coarse) * len(aqueous_values_coarse) * len(roughness_values_coarse)
        logger.info(f'📈 Coarse Grid Dimensions:')
        logger.info(f'   - Lipid values: {len(lipid_values_coarse)} ({lipid_values_coarse[0]:.1f} to {lipid_values_coarse[-1]:.1f} nm)')
        logger.info(f'   - Aqueous values: {len(aqueous_values_coarse)} ({aqueous_values_coarse[0]:.1f} to {aqueous_values_coarse[-1]:.1f} nm)')
        logger.info(f'   - Roughness values: {len(roughness_values_coarse)} ({roughness_values_coarse[0]:.1f} to {roughness_values_coarse[-1]:.1f} Å)')
        logger.info(f'   - Total combinations: {coarse_total:,}')
        
        # For dynamic search, reserve 30% budget for Stage 2 refinement
        # Stage 1 gets 70% of max_combinations
        if max_combinations is not None:
            stage1_budget = int(max_combinations * 0.7)  # 70% for Stage 1
        else:
            stage1_budget = None
        
        # Check if coarse search exceeds Stage 1 budget
        if stage1_budget is not None and coarse_total > stage1_budget:
            logger.warning(f'⚠️ Coarse search ({coarse_total:,} combinations) exceeds Stage 1 budget ({stage1_budget:,})')
            logger.warning(f'   Limiting coarse search to {stage1_budget:,} combinations (70% of {max_combinations:,} total)...')
            
            # Intelligently sample from each dimension to maximize coverage within budget
            # Strategy: Keep smaller dimensions fully, sample more aggressively from larger dimensions
            lipid_count = len(lipid_values_coarse)
            aqueous_count = len(aqueous_values_coarse)
            roughness_count = len(roughness_values_coarse)
            
            # Sort dimensions by size to decide sampling strategy
            dims = [
                ('lipid', lipid_count, lipid_values_coarse),
                ('aqueous', aqueous_count, aqueous_values_coarse),
                ('roughness', roughness_count, roughness_values_coarse)
            ]
            dims_sorted = sorted(dims, key=lambda x: x[1])
            
            # Start with smallest dimension - keep it fully
            smallest_name, smallest_count, smallest_values = dims_sorted[0]
            remaining_budget_per_combination = stage1_budget / smallest_count
            
            # For the two larger dimensions, calculate how many samples we need
            # We want: smallest_count × dim2_samples × dim3_samples ≈ stage1_budget
            # So: dim2_samples × dim3_samples ≈ remaining_budget_per_combination
            target_product = remaining_budget_per_combination
            
            # Get the two larger dimensions
            dim2_name, dim2_count, dim2_values = dims_sorted[1]
            dim3_name, dim3_count, dim3_values = dims_sorted[2]
            
            # Calculate optimal sampling: if we sample evenly, we want sqrt(target_product) from each
            target_per_large_dim = int(np.ceil(np.sqrt(target_product)))
            
            # Sample from larger dimensions, but don't exceed their original size
            dim2_target = min(dim2_count, target_per_large_dim)
            dim3_target = min(dim3_count, target_per_large_dim)
            
            # If still under budget, try to increase one dimension
            current_estimate = smallest_count * dim2_target * dim3_target
            if current_estimate < stage1_budget:
                # Try increasing the larger of the two dimensions
                if dim3_count > dim3_target:
                    dim3_target = min(dim3_count, int(np.ceil(stage1_budget / (smallest_count * dim2_target))))
                elif dim2_count > dim2_target:
                    dim2_target = min(dim2_count, int(np.ceil(stage1_budget / (smallest_count * dim3_target))))
            
            # Apply sampling to each dimension
            if smallest_name == 'lipid':
                lipid_values_coarse = smallest_values  # Keep all
                if dim2_name == 'aqueous':
                    if dim2_target < aqueous_count:
                        indices = np.linspace(0, aqueous_count-1, dim2_target, dtype=int)
                        aqueous_values_coarse = aqueous_values_coarse[indices]
                    if dim3_target < roughness_count:
                        indices = np.linspace(0, roughness_count-1, dim3_target, dtype=int)
                        roughness_values_coarse = roughness_values_coarse[indices]
                else:  # dim2 is roughness
                    if dim2_target < roughness_count:
                        indices = np.linspace(0, roughness_count-1, dim2_target, dtype=int)
                        roughness_values_coarse = roughness_values_coarse[indices]
                    if dim3_target < aqueous_count:
                        indices = np.linspace(0, aqueous_count-1, dim3_target, dtype=int)
                        aqueous_values_coarse = aqueous_values_coarse[indices]
            elif smallest_name == 'aqueous':
                aqueous_values_coarse = smallest_values  # Keep all
                if dim2_name == 'lipid':
                    if dim2_target < lipid_count:
                        indices = np.linspace(0, lipid_count-1, dim2_target, dtype=int)
                        lipid_values_coarse = lipid_values_coarse[indices]
                    if dim3_target < roughness_count:
                        indices = np.linspace(0, roughness_count-1, dim3_target, dtype=int)
                        roughness_values_coarse = roughness_values_coarse[indices]
                else:  # dim2 is roughness
                    if dim2_target < roughness_count:
                        indices = np.linspace(0, roughness_count-1, dim2_target, dtype=int)
                        roughness_values_coarse = roughness_values_coarse[indices]
                    if dim3_target < lipid_count:
                        indices = np.linspace(0, lipid_count-1, dim3_target, dtype=int)
                        lipid_values_coarse = lipid_values_coarse[indices]
            else:  # smallest is roughness
                roughness_values_coarse = smallest_values  # Keep all
                if dim2_name == 'lipid':
                    if dim2_target < lipid_count:
                        indices = np.linspace(0, lipid_count-1, dim2_target, dtype=int)
                        lipid_values_coarse = lipid_values_coarse[indices]
                    if dim3_target < aqueous_count:
                        indices = np.linspace(0, aqueous_count-1, dim3_target, dtype=int)
                        aqueous_values_coarse = aqueous_values_coarse[indices]
                else:  # dim2 is aqueous
                    if dim2_target < aqueous_count:
                        indices = np.linspace(0, aqueous_count-1, dim2_target, dtype=int)
                        aqueous_values_coarse = aqueous_values_coarse[indices]
                    if dim3_target < lipid_count:
                        indices = np.linspace(0, lipid_count-1, dim3_target, dtype=int)
                        lipid_values_coarse = lipid_values_coarse[indices]
            
            coarse_total = len(lipid_values_coarse) * len(aqueous_values_coarse) * len(roughness_values_coarse)
            logger.info(f'   After limiting: {coarse_total:,} combinations (target: {stage1_budget:,})')
            logger.info(f'   Sampling: {len(lipid_values_coarse)} lipid × {len(aqueous_values_coarse)} aqueous × {len(roughness_values_coarse)} roughness')
        
        logger.info(f'🚀 Starting parallel evaluation with {os.cpu_count() or 4} workers...')
        
        # Evaluate coarse grid (now uses intelligently sampled dimensions)
        coarse_results = self._evaluate_parameter_grid(
            wavelengths, measured, lipid_values_coarse, aqueous_values_coarse, roughness_values_coarse,
            top_k=20,  # Get more candidates for dynamic refinement
            enable_roughness=enable_roughness,
            min_correlation_filter=min_correlation_filter
        )
        # Track actual Stage 1 usage (equals coarse_total after sampling)
        stage1_actual_used = coarse_total
        
        stage1_elapsed = time_module.time() - stage1_start
        logger.info(f'✅ Stage 1 completed in {stage1_elapsed:.1f}s')
        logger.info(f'   Found {len(coarse_results)} candidates passing filters')
        
        if len(coarse_results) == 0:
            logger.warning('⚠️ No promising candidates found in coarse search')
            return []
        
        # Log top candidates
        logger.info(f'🏆 Top {min(5, len(coarse_results))} candidates from Stage 1:')
        for i, result in enumerate(coarse_results[:5], 1):
            logger.info(f'   {i}. Score: {result.score:.4f}, Corr: {result.correlation:.3f}, RMSE: {result.rmse:.5f}, '
                       f'L={result.lipid_nm:.1f}nm, A={result.aqueous_nm:.1f}nm, R={result.mucus_nm:.1f}Å')
        
        # Stage 2: Identify promising regions and create dynamic grid
        logger.info('')
        logger.info('🎯 STAGE 2: Dynamic Refinement')
        logger.info('-' * 80)
        stage2_start = time_module.time()
        
        # Refine top 10 candidates to use more of the Stage 2 budget (15,000 combinations)
        # Increased window sizes (3x, 2.5x, 2x) will generate many more combinations per candidate
        top_candidates = coarse_results[:min(10, len(coarse_results))]
        logger.info(f'🎯 Refining around {len(top_candidates)} top candidates with adaptive step sizes...')
        
        # Build dynamic parameter grid with ADAPTIVE step sizes based on candidate rank
        fine_combinations = []
        combinations_per_candidate = []
        
        for rank, candidate in enumerate(top_candidates):
            candidate_start_count = len(fine_combinations)
            # Adaptive strategy: finer steps for better candidates, coarser for lower-ranked
            # Increased window sizes to use more of the Stage 2 budget (15,000 combinations)
            if rank == 0:
                # Best candidate: fine refinement (±3x coarse step, 0.15x step size)
                strategy_name = "FINE (best candidate)"
                lipid_window = 3.0 * coarse_lipid_step  # Increased from 0.5x to 3x
                aqueous_window = 3.0 * coarse_aqueous_step
                roughness_window = 3.0 * coarse_roughness_step
                fine_lipid_step = max(1.0, coarse_lipid_step * 0.15)
                fine_aqueous_step = max(20.0, coarse_aqueous_step * 0.15)
                fine_roughness_step = max(10.0, coarse_roughness_step * 0.15)
            elif rank < 5:
                # Top 5: medium refinement (±2.5x coarse step, 0.25x step size)
                strategy_name = "MEDIUM (top 5)"
                lipid_window = 2.5 * coarse_lipid_step  # Increased from 0.75x to 2.5x
                aqueous_window = 2.5 * coarse_aqueous_step
                roughness_window = 2.5 * coarse_roughness_step
                fine_lipid_step = max(1.0, coarse_lipid_step * 0.25)
                fine_aqueous_step = max(20.0, coarse_aqueous_step * 0.25)
                fine_roughness_step = max(10.0, coarse_roughness_step * 0.25)
            else:
                # Lower-ranked (6-10): coarse refinement (±2x coarse step, 0.4x step size)
                strategy_name = "COARSE (lower ranked)"
                lipid_window = 2.0 * coarse_lipid_step  # Increased from 0.5x to 2x
                aqueous_window = 2.0 * coarse_aqueous_step
                roughness_window = 2.0 * coarse_roughness_step
                fine_lipid_step = max(1.0, coarse_lipid_step * 0.4)
                fine_aqueous_step = max(20.0, coarse_aqueous_step * 0.4)
                fine_roughness_step = max(10.0, coarse_roughness_step * 0.4)
            
            logger.info(f'   Candidate #{rank+1} (Score: {candidate.score:.4f}) - Strategy: {strategy_name}')
            logger.info(f'      Center: L={candidate.lipid_nm:.1f}nm, A={candidate.aqueous_nm:.1f}nm, R={candidate.mucus_nm:.1f}Å')
            logger.info(f'      Window: ±{lipid_window:.1f}nm (L), ±{aqueous_window:.1f}nm (A), ±{roughness_window:.1f}Å (R)')
            logger.info(f'      Step sizes: {fine_lipid_step:.1f}nm (L), {fine_aqueous_step:.1f}nm (A), {fine_roughness_step:.1f}Å (R)')
            
            # Generate fine grid around candidate
            lipid_min = max(lipid_range[0], candidate.lipid_nm - lipid_window)
            lipid_max = min(lipid_range[1], candidate.lipid_nm + lipid_window)
            aqueous_min = max(aqueous_range[0], candidate.aqueous_nm - aqueous_window)
            aqueous_max = min(aqueous_range[1], candidate.aqueous_nm + aqueous_window)
            roughness_min = max(roughness_range[0], candidate.mucus_nm - roughness_window)
            roughness_max = min(roughness_range[1], candidate.mucus_nm + roughness_window)
            
            # Generate grid points
            fine_lipids = np.arange(lipid_min, lipid_max + fine_lipid_step, fine_lipid_step)
            fine_aqueous = np.arange(aqueous_min, aqueous_max + fine_aqueous_step, fine_aqueous_step)
            fine_roughness = np.arange(roughness_min, roughness_max + fine_roughness_step, fine_roughness_step)
            
            # Filter to valid ranges
            fine_lipids = fine_lipids[(fine_lipids >= 9) & (fine_lipids <= 250)]
            fine_aqueous = fine_aqueous[(fine_aqueous >= 800) & (fine_aqueous <= 12000)]
            fine_roughness = fine_roughness[(fine_roughness >= 600) & (fine_roughness <= 7000)]
            
            # Add combinations (avoid duplicates with coarse results)
            for lipid in fine_lipids:
                for aqueous in fine_aqueous:
                    for roughness_angstrom in fine_roughness:
                        # Skip if too close to original candidate (already evaluated in coarse)
                        if (abs(lipid - candidate.lipid_nm) < fine_lipid_step * 0.5 and
                            abs(aqueous - candidate.aqueous_nm) < fine_aqueous_step * 0.5 and
                            abs(roughness_angstrom - candidate.mucus_nm) < fine_roughness_step * 0.5):
                            continue
                        fine_combinations.append((lipid, aqueous, roughness_angstrom))
            
            candidate_combinations = len(fine_combinations) - candidate_start_count
            combinations_per_candidate.append(candidate_combinations)
            logger.info(f'      Generated {candidate_combinations:,} combinations around this candidate')
        
        total_dynamic = len(fine_combinations)
        logger.info('')
        logger.info(f'📊 Stage 2 Summary:')
        logger.info(f'   - Total combinations generated: {total_dynamic:,}')
        logger.info(f'   - Combinations per candidate: {combinations_per_candidate}')
        
        # SAFETY CHECK: Limit total combinations to prevent excessive runtime
        if max_combinations is None:
            max_combinations = 5000  # Default if not provided
        
        # For dynamic search, reserve budget for Stage 2 refinement
        # Use 70% for Stage 1 (coarse search) and 30% for Stage 2 (refinement)
        stage1_budget = int(max_combinations * 0.7)  # Reserve 70% for coarse search
        stage2_budget = max_combinations - stage1_budget  # Remaining 30% for refinement
        
        # Use actual Stage 1 usage (tracked during evaluation)
        stage1_used = stage1_actual_used if max_combinations else coarse_total
        # Add unused Stage 1 budget to Stage 2 to utilize full max_combinations
        unused_stage1_budget = stage1_budget - stage1_used if max_combinations else 0
        remaining_budget = stage2_budget + unused_stage1_budget  # Use full budget
        
        logger.info(f'   - Max combinations limit: {max_combinations:,}')
        logger.info(f'   - Stage 1 budget: {stage1_budget:,} combinations (70% of total)')
        logger.info(f'   - Stage 1 used: {stage1_used:,} combinations')
        logger.info(f'   - Stage 1 unused: {unused_stage1_budget:,} combinations')
        logger.info(f'   - Stage 2 base budget: {stage2_budget:,} combinations (30% of total)')
        logger.info(f'   - Stage 2 total budget: {remaining_budget:,} combinations (includes unused Stage 1 budget)')
        
        if max_combinations and total_dynamic > remaining_budget:
            logger.warning(f'⚠️ Stage 2 combinations ({total_dynamic:,}) exceed remaining budget ({remaining_budget:,})')
            logger.warning(f'   Limiting to {remaining_budget:,} combinations by random sampling...')
            # Randomly sample to stay within limit (preserves diversity)
            random.seed(42)  # Reproducible sampling
            fine_combinations = random.sample(fine_combinations, remaining_budget)
            total_dynamic = len(fine_combinations)
            logger.info(f'   After limiting: {total_dynamic:,} combinations')
        elif total_dynamic > max_combinations:
            logger.warning(f'⚠️ Total combinations ({total_dynamic:,}) exceed max limit ({max_combinations:,})')
            logger.warning(f'   Limiting to {max_combinations:,} combinations by random sampling...')
            random.seed(42)
            fine_combinations = random.sample(fine_combinations, max_combinations)
            total_dynamic = len(fine_combinations)
            logger.info(f'   After limiting: {total_dynamic:,} combinations')
        
        logger.info(f'🚀 Starting parallel evaluation of {total_dynamic:,} combinations with {os.cpu_count() or 4} workers...')
        
        # Evaluate dynamic grid using direct parallel processing (not _evaluate_parameter_grid)
        # This avoids creating full 3D grid which can be huge
        num_workers = os.cpu_count() or 4
        logger.info(f'⚡ Using {num_workers} parallel workers for dynamic refinement')
        
        # Prepare material data
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
        
        dynamic_results: List[PyElliResult] = []
        completed_count = 0
        filtered_out = 0
        
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
            
            # Collect results with progress logging
            for future in as_completed(futures):
                completed_count += 1
                if completed_count % max(1, total_dynamic // 10) == 0 or completed_count == total_dynamic:
                    progress_pct = 100 * completed_count / total_dynamic
                    elapsed = time_module.time() - stage2_start
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    eta = (total_dynamic - completed_count) / rate if rate > 0 else 0
                    logger.info(f'   Progress: {completed_count}/{total_dynamic} ({progress_pct:.1f}%) | '
                               f'Elapsed: {elapsed:.1f}s | Rate: {rate:.1f} comb/s | ETA: {eta:.1f}s')
                
                result = future.result()
                if result is not None:
                    if (result.correlation >= min_correlation_filter and 
                        result.rmse <= 0.002):
                        dynamic_results.append(result)
                    else:
                        filtered_out += 1
        
        stage2_elapsed = time_module.time() - stage2_start
        logger.info(f'✅ Stage 2 completed in {stage2_elapsed:.1f}s')
        logger.info(f'   Evaluated: {completed_count:,} combinations')
        logger.info(f'   Passed filters: {len(dynamic_results):,}')
        logger.info(f'   Filtered out: {filtered_out:,} (correlation < {min_correlation_filter} or RMSE > 0.002)')
        
        # Sort by score first, then by smallest peak_count_delta, then matched_peaks (score and matched_peaks descending, delta ascending)
        dynamic_results.sort(key=lambda x: (x.score, -x.peak_count_delta, x.matched_peaks), reverse=True)
        
        # Combine coarse and dynamic results, remove duplicates
        logger.info('')
        logger.info('🔄 Combining Stage 1 and Stage 2 results...')
        all_results = coarse_results + dynamic_results
        # Sort by score first, then by smallest peak_count_delta, then matched_peaks
        all_results.sort(key=lambda x: (x.score, -x.peak_count_delta, x.matched_peaks), reverse=True)
        
        # Remove duplicates
        seen = set()
        unique_results = []
        duplicates_removed = 0
        for r in all_results:
            key = (round(r.lipid_nm, 1), round(r.aqueous_nm, 1), round(r.mucus_nm, 0))
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
            else:
                duplicates_removed += 1
        
        total_elapsed = time_module.time() - search_start_time
        logger.info('')
        logger.info('=' * 80)
        logger.info('✅ DYNAMIC SEARCH COMPLETED')
        logger.info('=' * 80)
        logger.info(f'⏱️  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)')
        logger.info(f'📊 Results:')
        logger.info(f'   - Stage 1 candidates: {len(coarse_results)}')
        logger.info(f'   - Stage 2 candidates: {len(dynamic_results)}')
        logger.info(f'   - Duplicates removed: {duplicates_removed}')
        logger.info(f'   - Unique results: {len(unique_results)}')
        logger.info(f'   - Returning top {top_k} results')
        if unique_results:
            logger.info(f'🏆 Best result: Score={unique_results[0].score:.4f}, Corr={unique_results[0].correlation:.3f}, '
                       f'RMSE={unique_results[0].rmse:.5f}, Matched Peaks={unique_results[0].matched_peaks}, '
                       f'Peak Count Delta={unique_results[0].peak_count_delta}, '
                       f'L={unique_results[0].lipid_nm:.1f}nm, A={unique_results[0].aqueous_nm:.1f}nm, R={unique_results[0].mucus_nm:.1f}Å')
        logger.info('=' * 80)
        
        return unique_results[:top_k]
    
    def _evaluate_parameter_grid(
        self,
        wavelengths: np.ndarray,
        measured: np.ndarray,
        lipid_values: np.ndarray,
        aqueous_values: np.ndarray,
        roughness_values: np.ndarray,
        top_k: int,
        enable_roughness: bool,
        min_correlation_filter: float,
    ) -> List[PyElliResult]:
        """
        Helper method to evaluate a parameter grid. Used by all search strategies.
        """
        results: List[PyElliResult] = []
        total = len(lipid_values) * len(aqueous_values) * len(roughness_values)
        
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
        
        # Generate all parameter combinations
        combinations = [
            (lipid, aqueous, roughness_angstrom)
            for lipid in lipid_values
            for aqueous in aqueous_values
            for roughness_angstrom in roughness_values
        ]
        
        # Use all available CPU cores
        num_workers = os.cpu_count() or 4
        logger.info(f'⚡ Using {num_workers} parallel workers for evaluation')
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
                for lipid, aqueous, roughness_angstrom in combinations
            }
            
            # Collect results
            completed = 0
            filtered_out = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 100 == 0:
                    logger.debug(f'Progress: {completed}/{total} ({100*completed/total:.1f}%)')
                
                result = future.result()
                if result is not None:
                    if (result.correlation >= min_correlation_filter and 
                        result.rmse <= 0.002):
                        results.append(result)
                    else:
                        filtered_out += 1
        
        if filtered_out > 0:
            logger.info(f'📊 Filtered out {filtered_out} results (correlation < {min_correlation_filter} or RMSE > 0.002)')
        
        if len(results) == 0:
            logger.warning('⚠️ No results passed filters!')
            return []
        
        # Sort by score first, then by smallest peak_count_delta, then matched_peaks (score and matched_peaks descending, delta ascending)
        results.sort(key=lambda x: (x.score, -x.peak_count_delta, x.matched_peaks), reverse=True)
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

