"""
PyElli Grid Search with Peak-Based Scoring

This module integrates pyElli's Transfer Matrix Method with peak count and peak alignment
scoring for finding the best fit.

This demonstrates how to use PyElli for auto-fitting tear film spectra.

Roughness Modeling:
    Uses pyElli's BruggemanEMA + VaryingMixtureLayer for proper interface roughness
    modeling between the mucus layer and corneal epithelium substrate.

Performance Note:
    BLAS thread limiting is applied at module load to prevent thread oversubscription
    when using multiprocessing. This is critical for performance on multi-core systems.
"""

# =============================================================================
# CRITICAL: Set BLAS thread limits BEFORE importing numpy/scipy
# This prevents thread oversubscription when using ProcessPoolExecutor
# Each worker process would otherwise spawn multiple BLAS threads, causing
# massive contention on multi-core systems (e.g., 32 cores x 8 threads = 256 threads)
# =============================================================================
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

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
# Ensure logger is set to INFO level to show parameter logging
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)


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
    mean_delta_nm: float = 0.0  # Mean delta between matched peaks (alignment quality)
    oscillation_ratio: float = 1.0  # Ratio of theoretical to measured oscillation amplitude
    theoretical_peaks: int = 0  # Number of peaks detected in theoretical spectrum
    theoretical_spectrum: np.ndarray = None
    wavelengths: np.ndarray = None
    # Drift metrics (cycle jump proxies) - single-spectrum indicators
    peak_drift_slope: float = 0.0  # Linear trend of Δλ vs wavelength (nm/nm)
    peak_drift_r_squared: float = 0.0  # R² of peak drift fit
    peak_drift_flagged: bool = False  # True if systematic peak drift detected
    amplitude_drift_slope: float = 0.0  # Linear trend of amplitude ratio vs wavelength
    amplitude_drift_r_squared: float = 0.0  # R² of amplitude drift fit
    amplitude_drift_flagged: bool = False  # True if systematic amplitude drift detected
    # Deviation vs measured (same formula as deviation vs LTA); used for ranking when set (lower = better)
    deviation_vs_measured: Optional[float] = None


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
    
    # === DRIFT METRICS (Cycle Jump Proxies) ===
    # Extract matched peak data for drift analysis
    if len(matched_meas) >= 3:
        matched_meas_wavelengths = meas_peaks[matched_meas]
        matched_theo_wavelengths = theo_peaks[matched_theo]
        # Get amplitudes from the peaks DataFrames
        matched_meas_amplitudes = meas_peaks_df['amplitude'].iloc[matched_meas].to_numpy(dtype=float)
        matched_theo_amplitudes = theo_peaks_df['amplitude'].iloc[matched_theo].to_numpy(dtype=float)
        
        drift_metrics = calculate_drift_metrics(
            matched_meas_wavelengths,
            matched_theo_wavelengths,
            matched_meas_amplitudes,
            matched_theo_amplitudes,
        )
    else:
        # Not enough matched peaks for drift analysis
        drift_metrics = {
            'peak_drift_slope': 0.0,
            'peak_drift_r_squared': 0.0,
            'peak_drift_flagged': False,
            'amplitude_drift_slope': 0.0,
            'amplitude_drift_r_squared': 0.0,
            'amplitude_drift_flagged': False,
            'drift_analysis_valid': False,
        }
    
    # Peak count score - CRITICAL for matching peak frequency
    # We need theoretical spectra to have SIMILAR peak counts to measured
    meas_count = len(meas_peaks)
    theo_count = len(theo_peaks)
    matched_count = len(matched_meas)
    
    if meas_count == 0:
        peak_count_score = 1.0 if theo_count == 0 else 0.0
        peak_count_ratio_score = 1.0
    elif matched_count == 0:
        # If no peaks match, give a small score based on how close the counts are
        count_ratio = min(meas_count, theo_count) / max(meas_count, theo_count) if max(meas_count, theo_count) > 0 else 0.0
        peak_count_score = 0.1 * count_ratio
        peak_count_ratio_score = count_ratio
    else:
        # CRITICAL: Reward matching most of the MEASURED peaks
        # This is the key metric - we want to find all the peaks in the measured spectrum
        match_ratio = matched_count / float(meas_count)
        
        # Also reward having similar peak counts (theo should be close to meas)
        # If theo_count << meas_count, we're not generating enough oscillations
        if theo_count > 0:
            count_similarity = min(theo_count, meas_count) / max(theo_count, meas_count)
        else:
            count_similarity = 0.0
        
        # Combined score: 60% match ratio (how many measured peaks we found) + 40% count similarity
        peak_count_score = 0.6 * match_ratio + 0.4 * count_similarity
        peak_count_score = max(0.0, min(1.0, peak_count_score))
        peak_count_ratio_score = count_similarity
    
    # CRITICAL: Penalize peak count mismatches
    # We need theoretical peak count to be close to measured (within +2/-2)
    max_allowed_excess = 2  # At most 2 more peaks than measured
    
    if meas_count > 0:
        peak_excess = theo_count - meas_count
        
        # HARD CONSTRAINT: Too many theoretical peaks invalidates the fit
        # Even if all measured peaks "match", having 13 peaks vs 7 measured is wrong
        if peak_excess > max_allowed_excess:
            # Severe penalty for too many theoretical peaks
            excess_over_limit = peak_excess - max_allowed_excess
            excess_penalty = 0.15 * excess_over_limit  # 0.15 per extra peak over limit
            peak_count_score = max(0.0, peak_count_score - excess_penalty)
        
        # Also penalize too few theoretical peaks
        if theo_count < meas_count * 0.6:
            deficit_ratio = theo_count / float(meas_count)
            peak_deficit_penalty = (0.6 - deficit_ratio) * 0.5
            peak_count_score = max(0.0, peak_count_score - peak_deficit_penalty)
    
    # Peak delta score - more lenient
    unmatched_measurement = len(meas_peaks) - len(matched_meas)
    unmatched_theoretical = len(theo_peaks) - len(matched_theo)
    total_unmatched = unmatched_measurement + unmatched_theoretical
    
    if deltas.size == 0:
        # No matched peaks - use sentinel value to prevent selection over candidates with real matches
        # Only use 0.0 if there are truly no peaks at all (both measured and theoretical have 0 peaks)
        if unmatched_measurement == 0 and unmatched_theoretical == 0:
            # No peaks at all - this is valid, mean_delta = 0.0 is appropriate
            mean_delta = 0.0
            peak_delta_score = 1.0
        else:
            # Peaks exist but none matched - use sentinel value (1000.0) so this candidate
            # is filtered out by mean_delta_nm < 1000.0 and not selected over candidates with real matches
            mean_delta = 1000.0  # Sentinel value - indicates no peak alignment
            # If no matches but similar peak counts, give small score
            total_peaks = len(meas_peaks) + len(theo_peaks)
            if total_peaks > 0:
                unmatched_ratio = total_unmatched / float(total_peaks)
                peak_delta_score = 0.05 * (1.0 - unmatched_ratio)  # Small score for similar counts
            else:
                peak_delta_score = 0.0
    else:
        mean_delta = float(np.mean(deltas))
        # More lenient scoring: use larger tau for exponential decay
        peak_delta_score = float(np.exp(-mean_delta / max(tau_nm, 1e-6)))
    
    # Apply penalty to peak_delta_score (for the 3% component)
    penalty = penalty_unpaired * float(total_unmatched)
    peak_delta_score = max(0.0, min(1.0, peak_delta_score - penalty))
    
    # === CALCULATE RMSE SCORE ===
    # RMSE measures the actual residual magnitude - critical for detecting bad fits!
    # Increased sensitivity: LTA achieves extremely low residuals (< 0.001)
    residual = measured - theoretical
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    
    # === REGIONAL RMSE SCORING ===
    # Weight early wavelengths MORE heavily to fix the start mismatch
    # Split early region into very-early (600-680nm) and early (680-750nm)
    very_early_mask = wavelengths <= 680
    early_mask = (wavelengths > 680) & (wavelengths <= 750)
    mid_mask = (wavelengths > 750) & (wavelengths <= 950)
    late_mask = wavelengths > 950
    
    very_early_rmse = float(np.sqrt(np.mean(residual[very_early_mask] ** 2))) if very_early_mask.any() else rmse
    early_rmse = float(np.sqrt(np.mean(residual[early_mask] ** 2))) if early_mask.any() else rmse
    mid_rmse = float(np.sqrt(np.mean(residual[mid_mask] ** 2))) if mid_mask.any() else rmse
    late_rmse = float(np.sqrt(np.mean(residual[late_mask] ** 2))) if late_mask.any() else rmse
    
    # Weighted RMSE: 30% very-early, 20% early, 25% mid, 25% late
    # Extra weight on early wavelengths to prioritize accuracy in this region
    weighted_rmse = 0.30 * very_early_rmse + 0.20 * early_rmse + 0.25 * mid_rmse + 0.25 * late_rmse
    
    # Normalize RMSE to a score (0-1) using weighted RMSE
    # EXTREMELY tight: LTA achieves RMSE ~0.0006, so we need to heavily penalize anything above 0.001
    # Using weighted_rmse gives more importance to early wavelengths (600-750nm)
    rmse_tau = 0.0008  # Much tighter - heavily penalizes RMSE > 0.001
    rmse_score = float(np.exp(-weighted_rmse / rmse_tau))
    
    # Also track unweighted RMSE for reporting
    rmse_unweighted = rmse
    rmse = weighted_rmse  # Use weighted for all subsequent calculations
    
    # === OSCILLATION AMPLITUDE CHECK ===
    # Penalize if theoretical oscillation amplitude doesn't match measured
    # This catches both "flat line" (too little) and "excessive oscillation" (too much) problems
    meas_oscillation = float(np.std(meas_detrended))
    theo_oscillation = float(np.std(theo_detrended))
    
    # Initialize defaults to avoid NameError
    oscillation_penalty = 1.0
    oscillation_ratio = 1.0
    amplitude_score = 1.0
    
    if meas_oscillation > 1e-8:
        oscillation_ratio = theo_oscillation / meas_oscillation
        
        # CRITICAL: Penalize BOTH too little AND too much oscillation
        # Ideal ratio is 1.0 (theoretical matches measured amplitude)
        # PyElli often produces 2-3x the oscillation amplitude - this is wrong physics!
        
        if oscillation_ratio < 0.5:
            # Too little oscillation (flat line)
            oscillation_penalty = oscillation_ratio * 2  # Scale 0-0.5 to 0-1
            amplitude_score = oscillation_ratio * 2
        elif oscillation_ratio > 2.0:
            # Too much oscillation (excessive interference) - SEVERE penalty
            # Ratio of 2.0 -> penalty 0.5, ratio of 3.0 -> penalty 0.33
            oscillation_penalty = 1.0 / oscillation_ratio
            amplitude_score = 1.0 / oscillation_ratio
        elif oscillation_ratio > 1.5:
            # Moderately too much oscillation - moderate penalty
            # Ratio of 1.5 -> penalty 0.85, ratio of 2.0 -> penalty 0.5
            excess = oscillation_ratio - 1.0
            oscillation_penalty = max(0.5, 1.0 - excess)
            amplitude_score = oscillation_penalty
        else:
            # Good match (0.5 to 1.5) - mild penalty for deviation from 1.0
            deviation = abs(oscillation_ratio - 1.0)
            oscillation_penalty = 1.0 - 0.2 * deviation  # Up to 10% penalty
            amplitude_score = oscillation_penalty
    else:
        oscillation_penalty = 1.0
        oscillation_ratio = 1.0
        amplitude_score = 1.0
    
    # === IMPROVED COMPOSITE SCORE ===
    # Updated weighting to prioritize peak matching (finding all measured peaks)
    # 
    # Weighting:
    # - 25% RMSE score (residual magnitude) - ensures curves are physically close
    # - 20% Amplitude score (oscillation matching) - prevents excessive/insufficient oscillations
    # - 15% Correlation (shape similarity) - important for visual fit quality
    # - 15% Peak delta (alignment quality) - peak positions should align
    # - 25% Peak count (matched peaks ratio) - CRITICAL: ensures we match most measured peaks
    
    # Ensure all components are defined
    c_rmse = float(rmse_score)
    c_corr = float(correlation_score)
    c_delta = float(peak_delta_score)
    c_count = float(peak_count_score)
    c_amplitude = float(amplitude_score)
    
    composite_score = (
        0.25 * c_rmse +
        0.20 * c_amplitude +
        0.15 * c_corr +
        0.15 * c_delta +
        0.25 * c_count     # INCREASED from 5% to 25% - finding peaks is critical
    )
    
    # CRITICAL: Penalty for peak count mismatch
    # Theoretical peaks should be close to measured (within +2/-2)
    max_allowed_excess = 2
    if meas_count > 0:
        peak_excess = theo_count - meas_count
        
        # HARD CONSTRAINT: Too many theoretical peaks is a deal-breaker
        # E.g., 13 theoretical vs 7 measured = 6 excess -> SEVERE penalty
        if peak_excess > max_allowed_excess:
            excess_over_limit = peak_excess - max_allowed_excess
            # Very severe penalty: 0.2 per extra peak over limit
            # 6 excess - 2 allowed = 4 over limit -> 0.8 penalty (basically reject)
            excess_penalty = 0.2 * excess_over_limit
            composite_score = max(0.0, composite_score - excess_penalty)
        
        # Also penalize too few peaks
        peak_coverage = theo_count / float(meas_count)
        if peak_coverage < 0.7:
            coverage_penalty = (0.7 - peak_coverage) * 0.5
            composite_score = max(0.0, composite_score - coverage_penalty)
    
    # Additional penalty for unmatched peaks (both directions)
    if total_unmatched > 0:
        # Base penalty: 0.015 per unpaired peak
        unpaired_penalty = 0.015 * float(total_unmatched)
        # Extra penalty for unmatched MEASURED peaks (we failed to find them)
        if unmatched_measurement > 0:
            unpaired_penalty += 0.02 * float(unmatched_measurement)
        composite_score = max(0.0, composite_score - unpaired_penalty)
    
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
    
    # CRITICAL: Require minimum matched peaks based on measured peaks
    # This ensures we select fits that actually align with the measured spectrum
    if meas_count > 0:
        match_ratio = matched_count / float(meas_count)
        min_match_ratio = 0.5  # Require at least 50% of measured peaks to be matched
        if match_ratio < min_match_ratio:
            # Heavy penalty for insufficient peak matches
            # Penalty scales with how far below the threshold we are
            penalty_factor = 1.0 - (match_ratio / min_match_ratio)
            composite_score = composite_score * (1.0 - 0.6 * penalty_factor)  # Up to 60% penalty
        
        # Bonus for matching most/all peaks with good alignment
        if match_ratio >= 0.9 and matched_count >= 5:
            # Excellent: matched 90%+ of peaks
            composite_score = min(1.0, composite_score * 1.15)  # 15% bonus
        elif match_ratio >= 0.7 and matched_count >= 5:
            # Good: matched 70%+ of peaks
            composite_score = min(1.0, composite_score * 1.08)  # 8% bonus
    elif matched_count >= 5 and correlation >= 0.95 and rmse <= 0.0011:
            # Moderate: at least 5 matches with excellent correlation/RMSE
            composite_score = min(1.0, composite_score * 1.05)  # 5% bonus
    
    # Calculate peak coverage ratio for tracking
    peak_coverage = theo_count / float(meas_count) if meas_count > 0 else 1.0
    
    return {
        "score": float(np.clip(composite_score, 0.0, 1.0)),
        "correlation": correlation,
        "correlation_score": correlation_score,
        "rmse": rmse,
        "rmse_score": rmse_score,
        "oscillation_ratio": oscillation_ratio if meas_oscillation > 1e-8 else 1.0,
        "amplitude_score": amplitude_score,
        "peak_count_score": peak_count_score,
        "peak_count_ratio_score": peak_count_ratio_score,
        "peak_coverage": peak_coverage,  # theo_count / meas_count - closer to 1.0 is better
        "peak_delta_score": peak_delta_score,
        "matched_peaks": float(matched_count),
        "mean_delta_nm": mean_delta,
        "measurement_peaks": float(meas_count),
        "theoretical_peaks": float(len(theo_peaks)),
        "unpaired_measurement": float(unmatched_measurement),
        "unpaired_theoretical": float(unmatched_theoretical),
        "meas_oscillation": meas_oscillation,
        "theo_oscillation": theo_oscillation,
        "start_offset": 0.0,  # Removed start offset penalty - kept for compatibility
        "very_early_rmse": very_early_rmse,
        # Drift metrics (cycle jump proxies)
        "peak_drift_slope": drift_metrics['peak_drift_slope'],
        "peak_drift_r_squared": drift_metrics['peak_drift_r_squared'],
        "peak_drift_flagged": drift_metrics['peak_drift_flagged'],
        "amplitude_drift_slope": drift_metrics['amplitude_drift_slope'],
        "amplitude_drift_r_squared": drift_metrics['amplitude_drift_r_squared'],
        "amplitude_drift_flagged": drift_metrics['amplitude_drift_flagged'],
        "drift_analysis_valid": drift_metrics.get('drift_analysis_valid', False),
    }


def calculate_drift_metrics(
    matched_meas_wavelengths: np.ndarray,
    matched_theo_wavelengths: np.ndarray,
    matched_meas_amplitudes: np.ndarray,
    matched_theo_amplitudes: np.ndarray,
    min_peaks_for_drift: int = 3,
    peak_drift_slope_threshold: float = 0.05,
    amplitude_drift_slope_threshold: float = 0.01,
    r_squared_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate drift metrics as single-spectrum proxies for cycle jump detection.
    
    Drift indicates systematic misalignment that grows across the spectrum,
    suggesting the chosen frequency/thickness solution may be a wrong multiple.
    
    Args:
        matched_meas_wavelengths: Wavelengths of matched measurement peaks
        matched_theo_wavelengths: Wavelengths of matched theoretical peaks
        matched_meas_amplitudes: Amplitudes of matched measurement peaks
        matched_theo_amplitudes: Amplitudes of matched theoretical peaks
        min_peaks_for_drift: Minimum matched peaks required for drift analysis
        peak_drift_slope_threshold: |slope| above this flags peak drift (nm/nm)
        amplitude_drift_slope_threshold: |slope| above this flags amplitude drift
        r_squared_threshold: R² above this indicates systematic (not random) drift
        
    Returns:
        Dictionary with drift metrics:
        - peak_drift_slope: Linear trend of Δλ vs wavelength (nm/nm)
        - peak_drift_r_squared: R² of peak drift fit
        - peak_drift_flagged: True if systematic peak drift detected
        - amplitude_drift_slope: Linear trend of amplitude ratio vs wavelength
        - amplitude_drift_r_squared: R² of amplitude drift fit
        - amplitude_drift_flagged: True if systematic amplitude drift detected
    """
    default_result = {
        'peak_drift_slope': 0.0,
        'peak_drift_r_squared': 0.0,
        'peak_drift_flagged': False,
        'amplitude_drift_slope': 0.0,
        'amplitude_drift_r_squared': 0.0,
        'amplitude_drift_flagged': False,
        'drift_analysis_valid': False,
    }
    
    num_matched = len(matched_meas_wavelengths)
    if num_matched < min_peaks_for_drift:
        logger.debug(f'📊 Drift analysis skipped: only {num_matched} matched peaks (need {min_peaks_for_drift})')
        return default_result
    
    # === PEAK DRIFT: How Δλ evolves across the spectrum ===
    # Δλ = theoretical - measured wavelength at each matched peak
    delta_wavelengths = matched_theo_wavelengths - matched_meas_wavelengths
    
    # Use measurement wavelengths as x-axis (where in spectrum is this peak?)
    x_values = matched_meas_wavelengths
    
    # Linear regression: Δλ = slope * wavelength + intercept
    x_mean = np.mean(x_values)
    y_mean = np.mean(delta_wavelengths)
    
    numerator = np.sum((x_values - x_mean) * (delta_wavelengths - y_mean))
    denominator = np.sum((x_values - x_mean) ** 2)
    
    if abs(denominator) < 1e-10:
        peak_drift_slope = 0.0
        peak_drift_r_squared = 0.0
    else:
        peak_drift_slope = numerator / denominator
        intercept = y_mean - peak_drift_slope * x_mean
        
        # Calculate R²
        y_predicted = peak_drift_slope * x_values + intercept
        ss_residual = np.sum((delta_wavelengths - y_predicted) ** 2)
        ss_total = np.sum((delta_wavelengths - y_mean) ** 2)
        peak_drift_r_squared = 1.0 - (ss_residual / ss_total) if ss_total > 1e-10 else 0.0
        peak_drift_r_squared = max(0.0, peak_drift_r_squared)
    
    # Flag if slope is significant AND systematic (high R²)
    peak_drift_flagged = (
        abs(peak_drift_slope) > peak_drift_slope_threshold and
        peak_drift_r_squared > r_squared_threshold
    )
    
    # === AMPLITUDE DRIFT: How amplitude mismatch evolves across spectrum ===
    # Ratio = theoretical amplitude / measured amplitude at each peak
    # Protect against division by zero
    safe_meas_amps = np.where(np.abs(matched_meas_amplitudes) > 1e-10, 
                               matched_meas_amplitudes, 1e-10)
    amplitude_ratios = matched_theo_amplitudes / safe_meas_amps
    
    # Linear regression: ratio = slope * wavelength + intercept
    y_mean_amp = np.mean(amplitude_ratios)
    
    numerator_amp = np.sum((x_values - x_mean) * (amplitude_ratios - y_mean_amp))
    
    if abs(denominator) < 1e-10:
        amplitude_drift_slope = 0.0
        amplitude_drift_r_squared = 0.0
    else:
        amplitude_drift_slope = numerator_amp / denominator
        intercept_amp = y_mean_amp - amplitude_drift_slope * x_mean
        
        # Calculate R²
        y_predicted_amp = amplitude_drift_slope * x_values + intercept_amp
        ss_residual_amp = np.sum((amplitude_ratios - y_predicted_amp) ** 2)
        ss_total_amp = np.sum((amplitude_ratios - y_mean_amp) ** 2)
        amplitude_drift_r_squared = 1.0 - (ss_residual_amp / ss_total_amp) if ss_total_amp > 1e-10 else 0.0
        amplitude_drift_r_squared = max(0.0, amplitude_drift_r_squared)
    
    # Flag if slope is significant AND systematic
    amplitude_drift_flagged = (
        abs(amplitude_drift_slope) > amplitude_drift_slope_threshold and
        amplitude_drift_r_squared > r_squared_threshold
    )
    
    logger.debug(
        f'📊 Drift analysis: peak_slope={peak_drift_slope:.4f} (R²={peak_drift_r_squared:.2f}), '
        f'amp_slope={amplitude_drift_slope:.4f} (R²={amplitude_drift_r_squared:.2f})'
    )
    
    return {
        'peak_drift_slope': float(peak_drift_slope),
        'peak_drift_r_squared': float(peak_drift_r_squared),
        'peak_drift_flagged': bool(peak_drift_flagged),
        'amplitude_drift_slope': float(amplitude_drift_slope),
        'amplitude_drift_r_squared': float(amplitude_drift_r_squared),
        'amplitude_drift_flagged': bool(amplitude_drift_flagged),
        'drift_analysis_valid': True,
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
# PyElli vs LTA MAPE (normalized for scale-invariant shape comparison)
# =============================================================================

def mape_pyelli_vs_lta_normalized(
    pyelli: np.ndarray,
    lta: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """
    MAPE between PyElli and LTA after normalizing both to unit L2 norm.
    Compares shape (relative profile) rather than absolute amplitude, so the
    deviation score is not dominated when LTA has a different amplitude convention.
    Returns value in percent (0--100+).
    """
    min_len = min(len(pyelli), len(lta))
    if min_len < 10:
        return 0.0
    p = np.asarray(pyelli[:min_len], dtype=float)
    l = np.asarray(lta[:min_len], dtype=float)
    p_norm = np.linalg.norm(p)
    l_norm = np.linalg.norm(l)
    if p_norm < 1e-12:
        p_norm = 1e-12
    if l_norm < 1e-12:
        return 0.0
    p = p / p_norm
    l = l / l_norm
    denom = np.abs(l) + eps
    return float(np.mean(np.abs(p - l) / denom)) * 100.0


# =============================================================================
# Deviation vs measured (same formula as deviation vs LTA, used for ranking)
# =============================================================================

def compute_deviation_vs_measured(
    wavelengths: np.ndarray,
    measured: np.ndarray,
    theoretical: np.ndarray,
    reference_spacing_nm: float = 50.0,
) -> float:
    """
    Composite deviation of theoretical vs measured (reference = measured).
    Same formula as deviation vs LTA: 0.4*MAPE + 0.3*peak_match_dev + 0.3*alignment_dev.
    Lower is better. Used for ranking so that selected fit best matches measured.
    """
    min_len = min(len(wavelengths), len(measured), len(theoretical))
    if min_len < 10:
        return float("inf")
    wl = wavelengths[:min_len]
    meas = measured[:min_len]
    theo = theoretical[:min_len]
    meas_abs = np.abs(meas)
    valid = meas_abs > 1e-10
    if valid.any():
        mape = float(np.mean(np.abs(theo[valid] - meas[valid]) / meas_abs[valid])) * 100.0
    else:
        mape = 0.0
    score_result = calculate_peak_based_score(wl, meas, theo)
    meas_peaks = int(score_result.get("measurement_peaks", 0))
    matched = int(score_result.get("matched_peaks", 0))
    if meas_peaks > 0:
        peak_match_dev = (1.0 - (matched / float(meas_peaks))) * 100.0
        peak_match_dev = max(0.0, peak_match_dev)
    else:
        peak_match_dev = 0.0
    mean_delta_nm = float(score_result.get("mean_delta_nm", 0.0))
    alignment_dev = (mean_delta_nm / reference_spacing_nm) * 100.0
    composite = 0.40 * mape + 0.30 * peak_match_dev + 0.30 * alignment_dev
    return float(composite)


# =============================================================================
# Helper Functions for Candidate Selection
# =============================================================================

def _get_candidate_sort_key(result: PyElliResult, min_matched_peaks: int = 0, meas_peaks_count: int = 0) -> tuple:
    """
    Sort key for candidate selection. When deviation_vs_measured is set, rank by product
    combined = score * (1 - deviation_vs_measured/100) so both high score and low deviation are preferred.
    Otherwise prioritizes score, matched peaks, peak count, oscillation ratio.
    """
    # When deviation_vs_measured is set, rank by product (high score and low deviation)
    if result.deviation_vs_measured is not None:
        inv_dev = max(0.0, 1.0 - result.deviation_vs_measured / 100.0)
        combined = result.score * inv_dev
        osc_deviation = abs(result.oscillation_ratio - 1.0)
        return (combined, -osc_deviation, -result.peak_count_delta, result.matched_peaks)
    
    # Base score with peak_count_delta penalty
    adjusted_score = result.score - (result.peak_count_delta * 0.01)
    
    # CRITICAL: Heavy penalty for too many theoretical peaks
    # Max allowed: measured + 2
    if meas_peaks_count > 0:
        max_allowed = meas_peaks_count + 2
        if result.theoretical_peaks > max_allowed:
            excess = result.theoretical_peaks - max_allowed
            # 0.15 penalty per extra peak beyond the limit
            adjusted_score -= excess * 0.15
        
        # Bonus for matching more peaks (relative to measured)
        match_ratio = result.matched_peaks / float(meas_peaks_count)
        adjusted_score += match_ratio * 0.3
    
    # CRITICAL: Heavy penalty for bad oscillation ratio (amplitude mismatch)
    # Ideal ratio is 1.0, penalize both too low and too high
    osc_ratio = result.oscillation_ratio
    if osc_ratio > 2.0:
        # Excessive oscillation - severe penalty (e.g., 3x amplitude -> 0.33 multiplier)
        adjusted_score = adjusted_score * (1.0 / osc_ratio)
    elif osc_ratio > 1.5:
        # Moderately excessive oscillation - moderate penalty
        adjusted_score = adjusted_score * (1.0 - (osc_ratio - 1.0) * 0.5)
    elif osc_ratio < 0.5:
        # Too little oscillation - penalty
        adjusted_score = adjusted_score * osc_ratio * 2
    
    # Heavy penalty if below minimum matched peaks requirement
    if min_matched_peaks > 0 and result.matched_peaks < min_matched_peaks:
        # Reduce score significantly if below minimum
        adjusted_score = adjusted_score * 0.3
    
    # Oscillation ratio penalty for sorting (how far from 1.0)
    osc_deviation = abs(osc_ratio - 1.0)
    
    return (adjusted_score, -osc_deviation, -result.peak_count_delta, result.matched_peaks)


def _get_measured_peaks_count(wavelengths: np.ndarray, measured: np.ndarray) -> int:
    """
    Get the number of peaks in the measured spectrum for minimum matching requirements.
    """
    try:
        from src.analysis.measurement_utils import detrend_signal, detect_peaks
        meas_detrended = detrend_signal(wavelengths, measured, cutoff_frequency=0.008, filter_order=3)
        meas_peaks_df = detect_peaks(wavelengths, meas_detrended, prominence=0.0001)
        return len(meas_peaks_df)
    except Exception:
        return 0


# =============================================================================
# Parallel Processing Worker Functions
# =============================================================================

# Global worker state - initialized once per worker process to avoid per-task overhead
_worker_state = {}


def _worker_initializer(
    wavelengths: np.ndarray,
    measured: np.ndarray,
    material_data: Dict[str, Any],
    enable_roughness: bool,
) -> None:
    """
    Initialize worker process with shared data.
    
    Called once per worker process (not per task) to:
    1. Set up logging/warning suppression
    2. Import pyElli once per worker (not per task)
    3. Pre-interpolate material data to target wavelengths
    4. Store all shared data in global state
    
    This dramatically reduces per-task overhead by avoiding:
    - Repeated pickling/unpickling of large arrays
    - Repeated module imports
    - Repeated interpolation of the same data
    """
    global _worker_state
    
    # Suppress warnings in worker processes (multiprocessing workers don't have Streamlit context)
    import warnings
    import logging as worker_logging
    
    warnings.filterwarnings('ignore', message='.*ScriptRunContext.*', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*', category=UserWarning)
    
    root_logger = worker_logging.getLogger()
    root_logger.setLevel(worker_logging.ERROR)
    
    for logger_name in ['streamlit', 'streamlit.runtime', 'streamlit.runtime.scriptrunner', 
                        'streamlit.runtime.scriptrunner.script_runner',
                        'streamlit.runtime.scriptrunner_utils', 
                        'streamlit.runtime.scriptrunner_utils.script_run_context']:
        worker_logging.getLogger(logger_name).setLevel(worker_logging.ERROR)
    
    class ScriptRunContextFilter(worker_logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return 'ScriptRunContext' not in msg and 'missing ScriptRunContext' not in msg
    
    for handler in root_logger.handlers:
        handler.addFilter(ScriptRunContextFilter())
    worker_logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(worker_logging.CRITICAL)
    worker_logging.getLogger('streamlit.runtime.scriptrunner_utils').setLevel(worker_logging.CRITICAL)
    
    # Import pyElli once per worker (not per task)
    # This avoids repeated import overhead for each parameter combination
    import elli
    from elli.dispersions.table_index import Table
    from elli.materials import IsotropicMaterial
    
    # Pre-interpolate material data to target wavelengths
    # This is the same for all tasks, so do it once per worker
    lipid_n = np.interp(wavelengths, material_data['lipid']['wavelength_nm'], material_data['lipid']['n'])
    lipid_k = np.interp(wavelengths, material_data['lipid']['wavelength_nm'], material_data['lipid']['k'])
    water_n = np.interp(wavelengths, material_data['water']['wavelength_nm'], material_data['water']['n'])
    water_k = np.interp(wavelengths, material_data['water']['wavelength_nm'], material_data['water']['k'])
    mucus_n = np.interp(wavelengths, material_data['mucus']['wavelength_nm'], material_data['mucus']['n'])
    mucus_k = np.interp(wavelengths, material_data['mucus']['wavelength_nm'], material_data['mucus']['k'])
    substratum_n = np.interp(wavelengths, material_data['substratum']['wavelength_nm'], material_data['substratum']['n'])
    substratum_k = np.interp(wavelengths, material_data['substratum']['wavelength_nm'], material_data['substratum']['k'])
    
    # Pre-compute focus mask (same for all tasks)
    focus_mask = (wavelengths >= 600.0) & (wavelengths <= 1120.0)
    
    # Store all shared data in worker state
    _worker_state.update({
        'wavelengths': wavelengths,
        'measured': measured,
        'enable_roughness': enable_roughness,
        'focus_mask': focus_mask,
        # Pre-interpolated material data
        'lipid_n': lipid_n,
        'lipid_k': lipid_k,
        'water_n': water_n,
        'water_k': water_k,
        'mucus_n': mucus_n,
        'mucus_k': mucus_k,
        'substratum_n': substratum_n,
        'substratum_k': substratum_k,
        # pyElli modules (avoid per-task imports)
        'elli': elli,
        'Table': Table,
        'IsotropicMaterial': IsotropicMaterial,
    })


def _evaluate_combination_fast(args: Tuple[float, float, float]) -> Optional[PyElliResult]:
    """
    Fast worker function using pre-initialized state.
    
    This function only receives the parameter values to evaluate,
    not the large shared data arrays. All shared data is accessed
    from _worker_state which was initialized once per worker.
    
    Args:
        args: Tuple of (lipid_nm, aqueous_nm, roughness_angstrom)
        
    Returns:
        PyElliResult or None if calculation fails
    """
    global _worker_state
    
    lipid, aqueous, roughness_angstrom = args
    
    try:
        # Access pre-initialized data from worker state
        wavelengths = _worker_state['wavelengths']
        measured = _worker_state['measured']
        enable_roughness = _worker_state['enable_roughness']
        focus_mask = _worker_state['focus_mask']
        
        # Use pre-interpolated material data
        lipid_n = _worker_state['lipid_n']
        lipid_k = _worker_state['lipid_k']
        water_n = _worker_state['water_n']
        water_k = _worker_state['water_k']
        mucus_n = _worker_state['mucus_n']
        mucus_k = _worker_state['mucus_k']
        substratum_n = _worker_state['substratum_n']
        substratum_k = _worker_state['substratum_k']
        
        # Calculate theoretical spectrum
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
        
        # Align using simple proportional scaling (focus region 600-1120 nm)
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
        
        # Quick correlation check
        wl_focus = wavelengths[focus_mask] if focus_mask.sum() > 0 else wavelengths
        meas_focus_scaled = measured[focus_mask] if focus_mask.sum() > 0 else measured
        theo_focus_scaled = theoretical_scaled[focus_mask] if focus_mask.sum() > 0 else theoretical_scaled
        
        if focus_mask.sum() > 0:
            if np.std(meas_focus_scaled) > 1e-10 and np.std(theo_focus_scaled) > 1e-10:
                quick_corr = float(np.corrcoef(meas_focus_scaled, theo_focus_scaled)[0, 1])
                if np.isnan(quick_corr):
                    quick_corr = 0.0
            else:
                quick_corr = 0.0
            
            if quick_corr < 0.5:
                return None
        else:
            quick_corr = 0.0
        
        # Fast scoring for moderate correlations
        if quick_corr < 0.7:
            residual = meas_focus_scaled - theo_focus_scaled
            rmse = float(np.sqrt(np.mean(residual ** 2)))
            
            rmse_tau = 0.0008
            rmse_score = float(np.exp(-rmse / rmse_tau))
            correlation_score = max(0.0, quick_corr) if quick_corr > 0 else 0.0
            
            try:
                meas_detrended_quick = detrend_signal(wl_focus, meas_focus_scaled, 0.008, 3)
                theo_detrended_quick = detrend_signal(wl_focus, theo_focus_scaled, 0.008, 3)
                meas_osc = float(np.std(meas_detrended_quick))
                theo_osc = float(np.std(theo_detrended_quick))
                if meas_osc > 1e-8:
                    osc_ratio = theo_osc / meas_osc
                    if osc_ratio < 0.5:
                        amplitude_score = osc_ratio * 2
                    elif osc_ratio > 2.0:
                        amplitude_score = 1.0 / osc_ratio
                    elif osc_ratio > 1.5:
                        excess = osc_ratio - 1.0
                        amplitude_score = max(0.5, 1.0 - excess)
                    else:
                        deviation = abs(osc_ratio - 1.0)
                        amplitude_score = 1.0 - 0.2 * deviation
                else:
                    osc_ratio = 1.0
                    amplitude_score = 1.0
            except Exception:
                osc_ratio = 1.0
                amplitude_score = 1.0
            
            simple_score = 0.35 * rmse_score + 0.35 * amplitude_score + 0.30 * correlation_score
            
            if rmse > 0.002:
                simple_score *= 0.2
            elif rmse > 0.0015:
                simple_score *= 0.5
            if quick_corr < 0.5:
                simple_score *= 0.2
            if osc_ratio > 2.0:
                simple_score *= 0.3
            elif osc_ratio > 1.5:
                simple_score *= 0.6
            
            score_result = {
                "score": float(np.clip(simple_score, 0.0, 1.0)),
                "correlation": quick_corr,
                "oscillation_ratio": osc_ratio,
                "matched_peaks": 0,
                "mean_delta_nm": 1000.0,
                "measurement_peaks": 0.0,
                "theoretical_peaks": 0.0,
                # Drift metrics not available in fast path (insufficient peaks)
                "peak_drift_slope": 0.0,
                "peak_drift_r_squared": 0.0,
                "peak_drift_flagged": False,
                "amplitude_drift_slope": 0.0,
                "amplitude_drift_r_squared": 0.0,
                "amplitude_drift_flagged": False,
            }
            correlation = quick_corr
        else:
            score_result = calculate_peak_based_score(
                wl_focus, meas_focus_scaled, theo_focus_scaled
            )
            residual = meas_focus_scaled - theo_focus_scaled
            rmse = float(np.sqrt(np.mean(residual ** 2)))
            correlation = score_result.get('correlation', quick_corr)
        
        meas_peaks_count = int(score_result.get('measurement_peaks', 0))
        theo_peaks_count = int(score_result.get('theoretical_peaks', 0))
        peak_count_delta = abs(meas_peaks_count - theo_peaks_count)
        mean_delta_nm = float(score_result.get('mean_delta_nm', 0.0))
        
        dev_vs_meas = compute_deviation_vs_measured(wl_focus, meas_focus_scaled, theo_focus_scaled)
        
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
            mean_delta_nm=mean_delta_nm,
            oscillation_ratio=float(score_result.get('oscillation_ratio', 1.0)),
            theoretical_peaks=theo_peaks_count,
            theoretical_spectrum=theoretical_scaled,
            wavelengths=wavelengths,
            # Drift metrics (cycle jump proxies)
            peak_drift_slope=float(score_result.get('peak_drift_slope', 0.0)),
            peak_drift_r_squared=float(score_result.get('peak_drift_r_squared', 0.0)),
            peak_drift_flagged=bool(score_result.get('peak_drift_flagged', False)),
            amplitude_drift_slope=float(score_result.get('amplitude_drift_slope', 0.0)),
            amplitude_drift_r_squared=float(score_result.get('amplitude_drift_r_squared', 0.0)),
            amplitude_drift_flagged=bool(score_result.get('amplitude_drift_flagged', False)),
            deviation_vs_measured=dev_vs_meas,
        )
    except Exception as e:
        return None


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
    
    # Suppress all ScriptRunContext warnings
    warnings.filterwarnings('ignore', message='.*ScriptRunContext.*', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*', category=UserWarning)
    
    # Suppress Streamlit runtime warnings at all levels
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.ERROR)
    
    # Suppress all streamlit-related loggers
    for logger_name in ['streamlit', 'streamlit.runtime', 'streamlit.runtime.scriptrunner', 
                        'streamlit.runtime.scriptrunner.script_runner',
                        'streamlit.runtime.scriptrunner_utils', 
                        'streamlit.runtime.scriptrunner_utils.script_run_context']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
    
    # Add filter to suppress ScriptRunContext messages
    class ScriptRunContextFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return 'ScriptRunContext' not in msg and 'missing ScriptRunContext' not in msg
    
    # Apply filter to all handlers
    for handler in root_logger.handlers:
        handler.addFilter(ScriptRunContextFilter())
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
            
            # Simple score based on RMSE, correlation, AND amplitude matching
            rmse_tau = 0.0008
            rmse_score = float(np.exp(-rmse / rmse_tau))
            correlation_score = max(0.0, quick_corr) if quick_corr > 0 else 0.0
            
            # CRITICAL: Calculate amplitude matching even in quick path
            # This catches excessive oscillation early and prevents bad candidates from passing
            try:
                meas_detrended_quick = detrend_signal(wl_focus, meas_focus_scaled, 0.008, 3)
                theo_detrended_quick = detrend_signal(wl_focus, theo_focus_scaled, 0.008, 3)
                meas_osc = float(np.std(meas_detrended_quick))
                theo_osc = float(np.std(theo_detrended_quick))
                if meas_osc > 1e-8:
                    osc_ratio = theo_osc / meas_osc
                    # Calculate amplitude score (same logic as full path)
                    if osc_ratio < 0.5:
                        amplitude_score = osc_ratio * 2
                    elif osc_ratio > 2.0:
                        amplitude_score = 1.0 / osc_ratio
                    elif osc_ratio > 1.5:
                        excess = osc_ratio - 1.0
                        amplitude_score = max(0.5, 1.0 - excess)
                    else:
                        deviation = abs(osc_ratio - 1.0)
                        amplitude_score = 1.0 - 0.2 * deviation
                else:
                    osc_ratio = 1.0
                    amplitude_score = 1.0
            except Exception:
                osc_ratio = 1.0
                amplitude_score = 1.0
            
            # Simple composite score (35% RMSE, 35% amplitude, 30% correlation)
            simple_score = 0.35 * rmse_score + 0.35 * amplitude_score + 0.30 * correlation_score
            
            # Apply penalties
            if rmse > 0.002:
                simple_score *= 0.2
            elif rmse > 0.0015:
                simple_score *= 0.5
            if quick_corr < 0.5:
                simple_score *= 0.2
            # CRITICAL: Heavy penalty for excessive oscillation
            if osc_ratio > 2.0:
                simple_score *= 0.3  # 70% penalty for 2x+ amplitude
            elif osc_ratio > 1.5:
                simple_score *= 0.6  # 40% penalty for 1.5x+ amplitude
            
            score_result = {
                "score": float(np.clip(simple_score, 0.0, 1.0)),
                "correlation": quick_corr,
                "correlation_score": correlation_score,
                "rmse": rmse,
                "rmse_score": rmse_score,
                "oscillation_ratio": osc_ratio,
                "amplitude_score": amplitude_score,
                "peak_count_score": 0.0,
                "peak_delta_score": 0.0,
                "matched_peaks": 0,
                "mean_delta_nm": 1000.0,  # Use large number so these are sorted last (no peak alignment data)
                "measurement_peaks": 0.0,
                "theoretical_peaks": 0.0,
                "unpaired_measurement": 0.0,
                "unpaired_theoretical": 0.0,
                # Drift metrics not available in fast path (insufficient peaks)
                "peak_drift_slope": 0.0,
                "peak_drift_r_squared": 0.0,
                "peak_drift_flagged": False,
                "amplitude_drift_slope": 0.0,
                "amplitude_drift_r_squared": 0.0,
                "amplitude_drift_flagged": False,
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
        mean_delta_nm = float(score_result.get('mean_delta_nm', 0.0))
        
        dev_vs_meas = compute_deviation_vs_measured(wl_focus, meas_focus_scaled, theo_focus_scaled)
        
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
            mean_delta_nm=mean_delta_nm,
            oscillation_ratio=float(score_result.get('oscillation_ratio', 1.0)),
            theoretical_peaks=theo_peaks_count,
            theoretical_spectrum=theoretical_scaled,
            wavelengths=wavelengths,
            # Drift metrics (cycle jump proxies)
            peak_drift_slope=float(score_result.get('peak_drift_slope', 0.0)),
            peak_drift_r_squared=float(score_result.get('peak_drift_r_squared', 0.0)),
            peak_drift_flagged=bool(score_result.get('peak_drift_flagged', False)),
            amplitude_drift_slope=float(score_result.get('amplitude_drift_slope', 0.0)),
            amplitude_drift_r_squared=float(score_result.get('amplitude_drift_r_squared', 0.0)),
            amplitude_drift_flagged=bool(score_result.get('amplitude_drift_flagged', False)),
            deviation_vs_measured=dev_vs_meas,
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
    
    # Default material files (matching original LTA Stack XML configuration)
    DEFAULT_LIPID_FILE = 'lipid_05-02621extrapolated.csv'
    DEFAULT_WATER_FILE = 'water_Bashkatov1353extrapolated.csv'
    DEFAULT_MUCUS_FILE = 'water_Bashkatov1353extrapolated.csv'
    DEFAULT_SUBSTRATUM_FILE = 'struma_Bashkatov140extrapolated.csv'
    
    def __init__(
        self,
        materials_path: Path,
        lipid_file: Optional[str] = None,
        water_file: Optional[str] = None,
        mucus_file: Optional[str] = None,
        substratum_file: Optional[str] = None,
        custom_materials: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        """
        Initialize with material data.
        
        Args:
            materials_path: Path to Materials directory with CSV files
            lipid_file: Lipid layer material CSV filename (default: lipid_05-02621extrapolated.csv)
            water_file: Aqueous layer material CSV filename (default: water_Bashkatov1353extrapolated.csv)
            mucus_file: Mucus layer material CSV filename (default: water_Bashkatov1353extrapolated.csv)
            substratum_file: Substratum material CSV filename (default: struma_Bashkatov140extrapolated.csv)
            custom_materials: Optional dict of custom material DataFrames (filename -> DataFrame)
        """
        self.materials_path = materials_path
        self.materials = get_available_materials(materials_path)
        self.custom_materials = custom_materials or {}
        
        # Use provided files or fall back to defaults
        lipid_file = lipid_file or self.DEFAULT_LIPID_FILE
        water_file = water_file or self.DEFAULT_WATER_FILE
        mucus_file = mucus_file or self.DEFAULT_MUCUS_FILE
        substratum_file = substratum_file or self.DEFAULT_SUBSTRATUM_FILE
        
        # Store selected material filenames for reference
        self.lipid_file = lipid_file
        self.water_file = water_file
        self.mucus_file = mucus_file
        self.substratum_file = substratum_file
        
        # Load tear film materials (check custom materials first, then load from file)
        self.lipid_df = self._load_material(lipid_file)
        self.water_df = self._load_material(water_file)
        self.mucus_df = self._load_material(mucus_file)
        self.substratum_df = self._load_material(substratum_file)
        
        logger.info(f'✅ Loaded tear film materials: lipid={lipid_file}, water={water_file}, mucus={mucus_file}, substratum={substratum_file}')
    
    def _load_material(self, material_name: str) -> pd.DataFrame:
        """
        Load material data from custom materials dict or from file.
        
        Args:
            material_name: Material filename or custom material name
            
        Returns:
            DataFrame with wavelength_nm, n, k columns
        """
        # Check if it's a custom material
        if material_name in self.custom_materials:
            logger.info(f'📤 Using custom material: {material_name}')
            return self.custom_materials[material_name]
        
        # Load from file
        return load_material_data(self.materials_path / material_name)
    
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
        aqueous_range: Tuple[float, float, float] = (800, 12000, 300),  # Standard range: 800-12000 nm
        roughness_range: Tuple[float, float, float] = (6000, 7000, 100),  # Optimized: good fits at ≥6000 Å
        top_k: int = 20,  # Increased to explore more candidates
        enable_roughness: bool = True,
        fine_search: bool = True,  
        fine_refinement_factor: float = 0.05,  # Much finer: 5% of coarse step for precision
        min_correlation_filter: float = 0.85,  # Keep selective
        search_strategy: str = 'Coarse Search',  # 'Coarse Search', 'Full Grid Search', or 'Dynamic Search'
        max_combinations: Optional[int] = 10000,  # Max combinations for Coarse/Dynamic search
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
                        Default: 9-250nm with 10nm step
            aqueous_range: (min, max, step) for aqueous thickness in nm
                          Default: 800-12000nm with 300nm step
            roughness_range: (min, max, step) for interface roughness in Angstroms (Å)
                            Default: 6000-7000Å with 100Å step (empirically optimized)
                            Note: Good fits consistently occur at ≥6000Å
            top_k: Number of top results to return
            enable_roughness: If True, use Bruggeman EMA roughness modeling
            min_correlation_filter: Minimum correlation for results (default 0.85)
            
        Returns:
            List of top results sorted by score (descending), filtered by correlation
        """
        # Log entry point with all parameters
        logger.info('')
        logger.info('=' * 80)
        logger.info('🚀 GRID SEARCH STARTED')
        logger.info('=' * 80)
        logger.info(f'📋 Input Parameters:')
        logger.info(f'   - Search Strategy: {search_strategy}')
        logger.info(f'   - Lipid Range: {lipid_range[0]:.1f} - {lipid_range[1]:.1f} nm (step: {lipid_range[2]:.1f} nm)')
        logger.info(f'   - Aqueous Range: {aqueous_range[0]:.1f} - {aqueous_range[1]:.1f} nm (step: {aqueous_range[2]:.1f} nm)')
        logger.info(f'   - Roughness Range: {roughness_range[0]:.1f} - {roughness_range[1]:.1f} Å (step: {roughness_range[2]:.1f} Å)')
        logger.info(f'   - Max Combinations: {max_combinations:,}' if max_combinations else '   - Max Combinations: None (unlimited)')
        logger.info(f'   - Top K Results: {top_k}')
        logger.info(f'   - Enable Roughness: {enable_roughness}')
        if search_strategy == 'Coarse Search':
            logger.info(f'   - Fine Search: {fine_search}')
            logger.info(f'   - Fine Refinement Factor: {fine_refinement_factor:.2f}')
        logger.info(f'   - Min Correlation Filter: {min_correlation_filter:.2f}')
        logger.info('=' * 80)
        logger.info('')
        
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
        logger.info('')
        logger.info('🔧 Generating parameter grid from user inputs...')
        logger.info(f'   Input step sizes: Lipid={lipid_range[2]:.1f}nm, Aqueous={aqueous_range[2]:.1f}nm, Roughness={roughness_range[2]:.1f}Å')
        
        lipid_values = np.arange(lipid_range[0], lipid_range[1] + 1, lipid_range[2])
        aqueous_values = np.arange(aqueous_range[0], aqueous_range[1] + 1, aqueous_range[2])
        roughness_values_angstrom = np.arange(roughness_range[0], roughness_range[1] + 1, roughness_range[2])
        
        logger.info(f'   Generated before filtering: Lipid={len(lipid_values)}, Aqueous={len(aqueous_values)}, Roughness={len(roughness_values_angstrom)}')
        
        # Filter to user-specified ranges (already bounded by ADOM limits in UI)
        lipid_values = lipid_values[(lipid_values >= lipid_range[0]) & (lipid_values <= lipid_range[1])]
        aqueous_values = aqueous_values[(aqueous_values >= aqueous_range[0]) & (aqueous_values <= aqueous_range[1])]
        roughness_values_angstrom = roughness_values_angstrom[(roughness_values_angstrom >= roughness_range[0]) & (roughness_values_angstrom <= roughness_range[1])]
        
        logger.info(f'   After range filtering: Lipid={len(lipid_values)}, Aqueous={len(aqueous_values)}, Roughness={len(roughness_values_angstrom)}')
        
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
            logger.info(f'   Before sampling: {len(lipid_values)} lipid × {len(aqueous_values)} aqueous × {len(roughness_values_angstrom)} roughness = {total:,} combinations')
            # Sample evenly from each dimension to stay within limit
            target_per_dim = int(np.ceil(max_combinations ** (1/3)))
            logger.info(f'   Target samples per dimension: {target_per_dim}')
            
            lipid_before = len(lipid_values)
            aqueous_before = len(aqueous_values)
            roughness_before = len(roughness_values_angstrom)
            
            if len(lipid_values) > target_per_dim:
                indices = np.linspace(0, len(lipid_values)-1, target_per_dim, dtype=int)
                lipid_values = lipid_values[indices]
                logger.info(f'   Lipid sampled: {lipid_before} → {len(lipid_values)} values')
            if len(aqueous_values) > target_per_dim:
                indices = np.linspace(0, len(aqueous_values)-1, target_per_dim, dtype=int)
                aqueous_values = aqueous_values[indices]
                logger.info(f'   Aqueous sampled: {aqueous_before} → {len(aqueous_values)} values')
            if len(roughness_values_angstrom) > target_per_dim:
                indices = np.linspace(0, len(roughness_values_angstrom)-1, target_per_dim, dtype=int)
                roughness_values_angstrom = roughness_values_angstrom[indices]
                logger.info(f'   Roughness sampled: {roughness_before} → {len(roughness_values_angstrom)} values')
            
            total = len(lipid_values) * len(aqueous_values) * len(roughness_values_angstrom)
            logger.info(f'   After sampling: {len(lipid_values)} lipid × {len(aqueous_values)} aqueous × {len(roughness_values_angstrom)} roughness = {total:,} combinations')
        
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
        
        # Generate all parameter combinations as tuples (for fast worker function)
        combinations = [
            (lipid, aqueous, roughness_angstrom)
            for lipid in lipid_values
            for aqueous in aqueous_values
            for roughness_angstrom in roughness_values_angstrom
        ]
        
        # Parallel evaluation with worker initializer (avoids per-task serialization overhead)
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_worker_initializer,
            initargs=(wavelengths, measured, material_data, enable_roughness),
        ) as executor:
            # Use map for better efficiency - chunks work automatically
            # Submit all tasks using the fast worker function
            futures = {
                executor.submit(_evaluate_combination_fast, combo): combo
                for combo in combinations
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
        # Note: min_matched_peaks=0 for intermediate sorting (filtering happens later)
        # Note: meas_peaks_count=0 for intermediate sorting (will be calculated in final selection)
        results.sort(key=lambda r: _get_candidate_sort_key(r, 0, 0), reverse=True)
        
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
                
                # Filter to user-specified ranges (already bounded by ADOM limits in UI)
                fine_lipid_values = fine_lipid_values[(fine_lipid_values >= lipid_range[0]) & (fine_lipid_values <= lipid_range[1])]
                fine_aqueous_values = fine_aqueous_values[(fine_aqueous_values >= aqueous_range[0]) & (fine_aqueous_values <= aqueous_range[1])]
                fine_roughness_values = fine_roughness_values[(fine_roughness_values >= roughness_range[0]) & (fine_roughness_values <= roughness_range[1])]
                
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
            
            # Parallel fine search with worker initializer
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
                with ProcessPoolExecutor(
                    max_workers=num_workers,
                    initializer=_worker_initializer,
                    initargs=(wavelengths, measured, material_data, enable_roughness),
                ) as executor:
                    futures = {
                        executor.submit(_evaluate_combination_fast, combo): combo
                        for combo in fine_combinations
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
            # Note: min_matched_peaks=0 for intermediate sorting (filtering happens later)
            # Note: meas_peaks_count=0 for intermediate sorting
            all_results.sort(key=lambda r: _get_candidate_sort_key(r, 0, 0), reverse=True)
            
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
            
            # Get measured peaks count for minimum matching requirement
            wl_mask = (wavelengths >= 600) & (wavelengths <= 1120)
            wl_focus = wavelengths[wl_mask] if wl_mask.any() else wavelengths
            meas_focus = measured[wl_mask] if wl_mask.any() else measured
            meas_peaks_count = _get_measured_peaks_count(wl_focus, meas_focus)
            min_matched_peaks = max(1, int(meas_peaks_count * 0.5))  # Require at least 50% of measured peaks
            
            # Stage 1: Identify top candidates using score, peak_count_delta, and matched_peaks
            # Sort by score (descending), peak_count_delta (ascending), matched_peaks (descending)
            sort_key_func = lambda r: _get_candidate_sort_key(r, min_matched_peaks, meas_peaks_count)
            unique_results.sort(key=sort_key_func, reverse=True)
            
            # Filter candidates that meet minimum matched peaks requirement
            candidates_meeting_min = [r for r in unique_results if r.matched_peaks >= min_matched_peaks]
            
            # Select top candidates (top 20% or at least top 10, whichever is larger)
            # Prefer candidates meeting minimum, but include others if not enough
            if len(candidates_meeting_min) >= 10:
                candidate_pool = candidates_meeting_min
                logger.info(f'📊 Filtering: {len(candidates_meeting_min)} candidates meet minimum matched peaks requirement ({min_matched_peaks} of {meas_peaks_count} measured peaks)')
            else:
                candidate_pool = unique_results
                logger.warning(f'⚠️ Only {len(candidates_meeting_min)} candidates meet minimum matched peaks ({min_matched_peaks} of {meas_peaks_count}), using all candidates')
            
            num_candidates = max(10, len(candidate_pool) // 5)
            top_candidates = candidate_pool[:num_candidates]
            
            # Stage 2: From top candidates, apply strict filters
            # Filter out candidates without peak alignment data (mean_delta_nm >= 1000.0)
            candidates_with_peaks = [r for r in top_candidates if r.mean_delta_nm < 1000.0]
            
            # CRITICAL: Filter by oscillation ratio (prefer 0.7 to 1.5 range)
            candidates_good_amplitude = [r for r in candidates_with_peaks if 0.7 <= r.oscillation_ratio <= 1.5]
            
            # CRITICAL: Filter out candidates with too many theoretical peaks
            # Max allowed: measured peaks + 2
            max_theo_peaks = meas_peaks_count + 2
            candidates_valid_peak_count = [r for r in candidates_good_amplitude if r.theoretical_peaks <= max_theo_peaks]
            
            if not candidates_valid_peak_count and candidates_good_amplitude:
                # Fallback: relax peak count constraint if no candidates pass
                logger.warning(f'⚠️ No candidates with theo_peaks <= {max_theo_peaks}, relaxing constraint')
                candidates_valid_peak_count = candidates_good_amplitude
            
            if candidates_valid_peak_count:
                # Best case: good peaks, good amplitude, valid peak count
                candidates_meeting_all = [r for r in candidates_valid_peak_count if r.matched_peaks >= min_matched_peaks]
                if candidates_meeting_all:
                    # Find max matched peaks among candidates
                    max_matched = max(c.matched_peaks for c in candidates_meeting_all)
                    # Get all candidates within 1 of the max (to allow for tie-breaking)
                    top_matched = [c for c in candidates_meeting_all if c.matched_peaks >= max_matched - 1]
                    # Among those with similar matched peaks, select by SMALLEST mean_delta (best alignment)
                    best_candidate = min(top_matched, key=lambda x: x.mean_delta_nm)
                else:
                    max_matched = max(c.matched_peaks for c in candidates_valid_peak_count)
                    top_matched = [c for c in candidates_valid_peak_count if c.matched_peaks >= max_matched - 1]
                    best_candidate = min(top_matched, key=lambda x: x.mean_delta_nm)
                logger.info(f'✅ Found {len(candidates_valid_peak_count)} candidates with good amplitude AND valid peak count (theo <= {max_theo_peaks})')
            elif candidates_with_peaks:
                # Fallback: has peaks but other filters not met
                # Still try to filter by peak count
                candidates_valid_in_fallback = [r for r in candidates_with_peaks if r.theoretical_peaks <= max_theo_peaks]
                if candidates_valid_in_fallback:
                    best_candidate = max(candidates_valid_in_fallback, key=lambda x: (x.matched_peaks, -x.mean_delta_nm))
                else:
                    best_candidate = max(candidates_with_peaks, key=lambda x: (x.matched_peaks, -x.mean_delta_nm))
                logger.warning(f'⚠️ No candidates with good amplitude, using best peak match (osc_ratio={best_candidate.oscillation_ratio:.2f}, theo_peaks={best_candidate.theoretical_peaks})')
            else:
                # No candidates with peak data
                results = candidate_pool[:top_k]
                logger.info(f'✅ Fine search completed. Best refined score: {results[0].score:.4f} (no peak alignment data available)')
            # Note: If we found candidates with peaks/good amplitude, we'll process them below
            
            if candidates_with_peaks or candidates_good_amplitude:
                # Reorder results to put best candidate first
                results = [best_candidate] + [r for r in candidate_pool if r != best_candidate]
                results = results[:top_k]
                logger.info(f'✅ Fine search completed. Selected best candidate: Score={best_candidate.score:.4f}, Peak Count Delta={best_candidate.peak_count_delta}, Matched Peaks={best_candidate.matched_peaks}, Mean Delta={best_candidate.mean_delta_nm:.2f}nm, Osc Ratio={best_candidate.oscillation_ratio:.2f} (from {len(candidates_with_peaks)} candidates)')
        
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
                f'Score={best.score:.4f}, Matched Peaks={best.matched_peaks:.0f}, '
                f'Mean Delta={best.mean_delta_nm:.2f}nm, '
                f'Meas Peaks={best_score_result.get("measurement_peaks", 0):.0f}, '
                f'Theo Peaks={best_score_result.get("theoretical_peaks", 0):.0f}, '
                f'Correlation={best.correlation:.3f}'
            )
            # Log top 3 results for debugging
            for i, r in enumerate(results[:min(3, len(results))]):
                logger.debug(
                    f'  Rank {i+1}: L={r.lipid_nm:.1f}, A={r.aqueous_nm:.1f}, R={r.mucus_nm:.0f}Å, '
                    f'Score={r.score:.4f}, Matched={r.matched_peaks:.0f}, '
                    f'Delta={r.mean_delta_nm:.2f}nm'
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
        logger.info('')
        logger.info('🔧 Calculating fine step sizes for full grid search...')
        logger.info(f'   User input step sizes: Lipid={lipid_range[2]:.1f}nm, Aqueous={aqueous_range[2]:.1f}nm, Roughness={roughness_range[2]:.1f}Å')
        
        fine_lipid_step = max(1.0, lipid_range[2] * 0.2)  # 20% of coarse step
        fine_aqueous_step = max(20.0, aqueous_range[2] * 0.2)
        fine_roughness_step = max(10.0, roughness_range[2] * 0.2)
        
        logger.info(f'   Calculated fine steps: Lipid={fine_lipid_step:.1f}nm (20% of {lipid_range[2]:.1f}nm), Aqueous={fine_aqueous_step:.1f}nm (20% of {aqueous_range[2]:.1f}nm), Roughness={fine_roughness_step:.1f}Å (20% of {roughness_range[2]:.1f}Å)')
        logger.info(f'🔍 Running FULL GRID SEARCH with fine steps: Lipid={fine_lipid_step:.1f}nm, Aqueous={fine_aqueous_step:.1f}nm, Roughness={fine_roughness_step:.1f}Å')
        
        # Generate fine parameter grid
        logger.info('')
        logger.info('🔧 Generating fine parameter grid...')
        lipid_values = np.arange(lipid_range[0], lipid_range[1] + 1, fine_lipid_step)
        aqueous_values = np.arange(aqueous_range[0], aqueous_range[1] + 1, fine_aqueous_step)
        roughness_values_angstrom = np.arange(roughness_range[0], roughness_range[1] + 1, fine_roughness_step)
        
        logger.info(f'   Generated before filtering: Lipid={len(lipid_values)}, Aqueous={len(aqueous_values)}, Roughness={len(roughness_values_angstrom)}')
        
        # Filter to user-specified ranges (already bounded by ADOM limits in UI)
        lipid_values = lipid_values[(lipid_values >= lipid_range[0]) & (lipid_values <= lipid_range[1])]
        aqueous_values = aqueous_values[(aqueous_values >= aqueous_range[0]) & (aqueous_values <= aqueous_range[1])]
        roughness_values_angstrom = roughness_values_angstrom[(roughness_values_angstrom >= roughness_range[0]) & (roughness_values_angstrom <= roughness_range[1])]
        
        logger.info(f'   After range filtering: Lipid={len(lipid_values)}, Aqueous={len(aqueous_values)}, Roughness={len(roughness_values_angstrom)}')
        
        if len(lipid_values) == 0 or len(aqueous_values) == 0 or len(roughness_values_angstrom) == 0:
            logger.warning(f'⚠️ Parameter ranges resulted in empty grid!')
            return []
        
        total = len(lipid_values) * len(aqueous_values) * len(roughness_values_angstrom)
        logger.info(f'📊 Full grid search: {total:,} total combinations to evaluate (no limit - exhaustive search)')
        logger.info(f'   Final grid: {len(lipid_values)} lipid × {len(aqueous_values)} aqueous × {len(roughness_values_angstrom)} roughness')
        
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
        
        # For dynamic search, reserve 30% budget for Stage 2 refinement
        # Stage 1 gets 70% of max_combinations
        if max_combinations is not None:
            stage1_budget = int(max_combinations * 0.7)  # 70% for Stage 1
        else:
            stage1_budget = None
        
        # Use user's exact step sizes (respect client's UI settings)
        logger.info('')
        logger.info('🔧 Using user-specified step sizes for dynamic search...')
        logger.info(f'   User input step sizes: Lipid={lipid_range[2]:.1f}nm, Aqueous={aqueous_range[2]:.1f}nm, Roughness={roughness_range[2]:.1f}Å')
        
        # Always use the user's step sizes - no optimization/change
        coarse_lipid_step = lipid_range[2]
        coarse_aqueous_step = aqueous_range[2]
        coarse_roughness_step = roughness_range[2]
        
        logger.info(f'🔧 Coarse Step Sizes (using user input):')
        logger.info(f'   - Lipid: {coarse_lipid_step:.1f} nm')
        logger.info(f'   - Aqueous: {coarse_aqueous_step:.1f} nm')
        logger.info(f'   - Roughness: {coarse_roughness_step:.1f} Å')
        
        logger.info('')
        logger.info('🔧 Generating coarse parameter grid for Stage 1 with user step sizes...')
        lipid_values_coarse = np.arange(lipid_range[0], lipid_range[1] + 1, coarse_lipid_step)
        aqueous_values_coarse = np.arange(aqueous_range[0], aqueous_range[1] + 1, coarse_aqueous_step)
        roughness_values_coarse = np.arange(roughness_range[0], roughness_range[1] + 1, coarse_roughness_step)
        
        logger.info(f'   Generated before filtering: Lipid={len(lipid_values_coarse)}, Aqueous={len(aqueous_values_coarse)}, Roughness={len(roughness_values_coarse)}')
        
        # Filter to user-specified ranges (already bounded by ADOM limits in UI)
        lipid_values_coarse = lipid_values_coarse[(lipid_values_coarse >= lipid_range[0]) & (lipid_values_coarse <= lipid_range[1])]
        aqueous_values_coarse = aqueous_values_coarse[(aqueous_values_coarse >= aqueous_range[0]) & (aqueous_values_coarse <= aqueous_range[1])]
        roughness_values_coarse = roughness_values_coarse[(roughness_values_coarse >= roughness_range[0]) & (roughness_values_coarse <= roughness_range[1])]
        
        logger.info(f'   After range filtering: Lipid={len(lipid_values_coarse)}, Aqueous={len(aqueous_values_coarse)}, Roughness={len(roughness_values_coarse)}')
        
        coarse_total = len(lipid_values_coarse) * len(aqueous_values_coarse) * len(roughness_values_coarse)
        logger.info(f'📈 Coarse Grid Dimensions:')
        logger.info(f'   - Lipid values: {len(lipid_values_coarse)} ({lipid_values_coarse[0]:.1f} to {lipid_values_coarse[-1]:.1f} nm)')
        logger.info(f'   - Aqueous values: {len(aqueous_values_coarse)} ({aqueous_values_coarse[0]:.1f} to {aqueous_values_coarse[-1]:.1f} nm)')
        logger.info(f'   - Roughness values: {len(roughness_values_coarse)} ({roughness_values_coarse[0]:.1f} to {roughness_values_coarse[-1]:.1f} Å)')
        logger.info(f'   - Total combinations: {coarse_total:,}')
        
        # Check if coarse search exceeds Stage 1 budget - sample grid to fit budget while respecting step sizes
        if stage1_budget is not None and coarse_total > stage1_budget:
            logger.warning(f'⚠️ Coarse search ({coarse_total:,} combinations) exceeds Stage 1 budget ({stage1_budget:,})')
            logger.warning(f'   Limiting coarse search to {stage1_budget:,} combinations (70% of {max_combinations:,} total)...')
            logger.info(f'   Before sampling: {len(lipid_values_coarse)} lipid × {len(aqueous_values_coarse)} aqueous × {len(roughness_values_coarse)} roughness = {coarse_total:,} combinations')
            
            # Intelligently sample from each dimension to maximize coverage within budget
            # Strategy: Keep smaller dimensions fully, sample more aggressively from larger dimensions
            lipid_count = len(lipid_values_coarse)
            aqueous_count = len(aqueous_values_coarse)
            roughness_count = len(roughness_values_coarse)
            
            lipid_before = lipid_count
            aqueous_before = aqueous_count
            roughness_before = roughness_count
            
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
                        logger.info(f'   Aqueous sampled: {aqueous_before} → {len(aqueous_values_coarse)} values')
                    if dim3_target < roughness_count:
                        indices = np.linspace(0, roughness_count-1, dim3_target, dtype=int)
                        roughness_values_coarse = roughness_values_coarse[indices]
                        logger.info(f'   Roughness sampled: {roughness_before} → {len(roughness_values_coarse)} values')
                else:  # dim2 is roughness
                    if dim2_target < roughness_count:
                        indices = np.linspace(0, roughness_count-1, dim2_target, dtype=int)
                        roughness_values_coarse = roughness_values_coarse[indices]
                        logger.info(f'   Roughness sampled: {roughness_before} → {len(roughness_values_coarse)} values')
                    if dim3_target < aqueous_count:
                        indices = np.linspace(0, aqueous_count-1, dim3_target, dtype=int)
                        aqueous_values_coarse = aqueous_values_coarse[indices]
                        logger.info(f'   Aqueous sampled: {aqueous_before} → {len(aqueous_values_coarse)} values')
            elif smallest_name == 'aqueous':
                aqueous_values_coarse = smallest_values  # Keep all
                if dim2_name == 'lipid':
                    if dim2_target < lipid_count:
                        indices = np.linspace(0, lipid_count-1, dim2_target, dtype=int)
                        lipid_values_coarse = lipid_values_coarse[indices]
                        logger.info(f'   Lipid sampled: {lipid_before} → {len(lipid_values_coarse)} values')
                    if dim3_target < roughness_count:
                        indices = np.linspace(0, roughness_count-1, dim3_target, dtype=int)
                        roughness_values_coarse = roughness_values_coarse[indices]
                        logger.info(f'   Roughness sampled: {roughness_before} → {len(roughness_values_coarse)} values')
                else:  # dim2 is roughness
                    if dim2_target < roughness_count:
                        indices = np.linspace(0, roughness_count-1, dim2_target, dtype=int)
                        roughness_values_coarse = roughness_values_coarse[indices]
                        logger.info(f'   Roughness sampled: {roughness_before} → {len(roughness_values_coarse)} values')
                    if dim3_target < lipid_count:
                        indices = np.linspace(0, lipid_count-1, dim3_target, dtype=int)
                        lipid_values_coarse = lipid_values_coarse[indices]
                        logger.info(f'   Lipid sampled: {lipid_before} → {len(lipid_values_coarse)} values')
            else:  # smallest is roughness
                roughness_values_coarse = smallest_values  # Keep all
                if dim2_name == 'lipid':
                    if dim2_target < lipid_count:
                        indices = np.linspace(0, lipid_count-1, dim2_target, dtype=int)
                        lipid_values_coarse = lipid_values_coarse[indices]
                        logger.info(f'   Lipid sampled: {lipid_before} → {len(lipid_values_coarse)} values')
                    if dim3_target < aqueous_count:
                        indices = np.linspace(0, aqueous_count-1, dim3_target, dtype=int)
                        aqueous_values_coarse = aqueous_values_coarse[indices]
                        logger.info(f'   Aqueous sampled: {aqueous_before} → {len(aqueous_values_coarse)} values')
                else:  # dim2 is aqueous
                    if dim2_target < aqueous_count:
                        indices = np.linspace(0, aqueous_count-1, dim2_target, dtype=int)
                        aqueous_values_coarse = aqueous_values_coarse[indices]
                        logger.info(f'   Aqueous sampled: {aqueous_before} → {len(aqueous_values_coarse)} values')
                    if dim3_target < lipid_count:
                        indices = np.linspace(0, lipid_count-1, dim3_target, dtype=int)
                        lipid_values_coarse = lipid_values_coarse[indices]
                        logger.info(f'   Lipid sampled: {lipid_before} → {len(lipid_values_coarse)} values')
            
            coarse_total = len(lipid_values_coarse) * len(aqueous_values_coarse) * len(roughness_values_coarse)
            logger.info(f'   After sampling: {len(lipid_values_coarse)} lipid × {len(aqueous_values_coarse)} aqueous × {len(roughness_values_coarse)} roughness = {coarse_total:,} combinations (target: {stage1_budget:,})')
        
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
            
            # Filter to user-specified ranges (already bounded by ADOM limits in UI)
            fine_lipids = fine_lipids[(fine_lipids >= lipid_range[0]) & (fine_lipids <= lipid_range[1])]
            fine_aqueous = fine_aqueous[(fine_aqueous >= aqueous_range[0]) & (fine_aqueous <= aqueous_range[1])]
            fine_roughness = fine_roughness[(fine_roughness >= roughness_range[0]) & (fine_roughness <= roughness_range[1])]
            
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
        
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_worker_initializer,
            initargs=(wavelengths, measured, material_data, enable_roughness),
        ) as executor:
            futures = {
                executor.submit(_evaluate_combination_fast, combo): combo
                for combo in fine_combinations
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
        
        # Sort by score first, then peak_count_delta, then matched_peaks (for candidate identification)
        # Note: min_matched_peaks=0 for intermediate sorting (filtering happens later)
        # Note: meas_peaks_count=0 for intermediate sorting
        dynamic_results.sort(key=lambda r: _get_candidate_sort_key(r, 0, 0), reverse=True)
        
        # Combine coarse and dynamic results, remove duplicates
        logger.info('')
        logger.info('🔄 Combining Stage 1 and Stage 2 results...')
        all_results = coarse_results + dynamic_results
        # Sort by score first, then peak_count_delta, then matched_peaks (for candidate identification)
        # Note: min_matched_peaks=0 for intermediate sorting (filtering happens later)
        all_results.sort(key=lambda r: _get_candidate_sort_key(r, 0), reverse=True)
        
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
        # Get measured peaks count for minimum matching requirement
        wl_mask = (wavelengths >= 600) & (wavelengths <= 1120)
        wl_focus = wavelengths[wl_mask] if wl_mask.any() else wavelengths
        meas_focus = measured[wl_mask] if wl_mask.any() else measured
        meas_peaks_count = _get_measured_peaks_count(wl_focus, meas_focus)
        min_matched_peaks = max(1, int(meas_peaks_count * 0.5))  # Require at least 50% of measured peaks
        
        # Stage 1: Identify top candidates using score, peak_count_delta, and matched_peaks
        # Sort by score (descending), peak_count_delta (ascending), matched_peaks (descending)
        sort_key_func = lambda r: _get_candidate_sort_key(r, min_matched_peaks)
        unique_results.sort(key=sort_key_func, reverse=True)
        
        # Filter candidates that meet minimum matched peaks requirement
        candidates_meeting_min = [r for r in unique_results if r.matched_peaks >= min_matched_peaks]
        
        # Select top candidates (top 20% or at least top 10, whichever is larger)
        # Prefer candidates meeting minimum, but include others if not enough
        if len(candidates_meeting_min) >= 10:
            candidate_pool = candidates_meeting_min
            logger.info(f'📊 Filtering: {len(candidates_meeting_min)} candidates meet minimum matched peaks requirement ({min_matched_peaks} of {meas_peaks_count} measured peaks)')
        else:
            candidate_pool = unique_results
            logger.warning(f'⚠️ Only {len(candidates_meeting_min)} candidates meet minimum matched peaks ({min_matched_peaks} of {meas_peaks_count}), using all candidates')
        
        num_candidates = max(10, len(candidate_pool) // 5)
        top_candidates = candidate_pool[:num_candidates]
        
        # Stage 2: From top candidates, select the one with smallest mean_delta_nm (best peak alignment)
        # Filter out candidates without peak alignment data (mean_delta_nm >= 1000.0)
        # AND filter out candidates with bad oscillation ratio (amplitude mismatch)
        candidates_with_peaks = [r for r in top_candidates if r.mean_delta_nm < 1000.0]
        # CRITICAL: Also filter by oscillation ratio (prefer 0.7 to 1.5 range)
        candidates_good_amplitude = [r for r in candidates_with_peaks if 0.7 <= r.oscillation_ratio <= 1.5]
        
        # CRITICAL: Filter out candidates with too many theoretical peaks
        # Max allowed: measured peaks + 2
        max_theo_peaks = meas_peaks_count + 2
        candidates_valid_peak_count = [r for r in candidates_good_amplitude if r.theoretical_peaks <= max_theo_peaks]
        
        if not candidates_valid_peak_count and candidates_good_amplitude:
            logger.warning(f'⚠️ No candidates with theo_peaks <= {max_theo_peaks}, relaxing constraint')
            candidates_valid_peak_count = candidates_good_amplitude
        
        if candidates_valid_peak_count:
            # Best case: good peaks, good amplitude, valid peak count
            candidates_meeting_all = [r for r in candidates_valid_peak_count if r.matched_peaks >= min_matched_peaks]
            if candidates_meeting_all:
                # Find max matched peaks, then select by smallest mean_delta among top matches
                max_matched = max(c.matched_peaks for c in candidates_meeting_all)
                top_matched = [c for c in candidates_meeting_all if c.matched_peaks >= max_matched - 1]
                best_candidate = min(top_matched, key=lambda x: x.mean_delta_nm)
            else:
                max_matched = max(c.matched_peaks for c in candidates_valid_peak_count)
                top_matched = [c for c in candidates_valid_peak_count if c.matched_peaks >= max_matched - 1]
                best_candidate = min(top_matched, key=lambda x: x.mean_delta_nm)
            logger.info(f'✅ Found {len(candidates_valid_peak_count)} candidates with good amplitude AND valid peak count (theo <= {max_theo_peaks})')
            # Reorder results to put best candidate first
            final_results = [best_candidate] + [r for r in candidate_pool if r != best_candidate]
            final_results = final_results[:top_k]
            logger.info(f'✅ Selected best candidate: Score={best_candidate.score:.4f}, Theo Peaks={best_candidate.theoretical_peaks}, Matched Peaks={best_candidate.matched_peaks}, Mean Delta={best_candidate.mean_delta_nm:.2f}nm, Osc Ratio={best_candidate.oscillation_ratio:.2f}')
        elif candidates_with_peaks:
            # Fallback: has peaks but other filters not met
            candidates_valid_in_fallback = [r for r in candidates_with_peaks if r.theoretical_peaks <= max_theo_peaks]
            if candidates_valid_in_fallback:
                best_candidate = max(candidates_valid_in_fallback, key=lambda x: (x.matched_peaks, -x.mean_delta_nm))
            else:
                best_candidate = max(candidates_with_peaks, key=lambda x: (x.matched_peaks, -x.mean_delta_nm))
            logger.warning(f'⚠️ No candidates with good amplitude, using best peak match (theo_peaks={best_candidate.theoretical_peaks}, osc_ratio={best_candidate.oscillation_ratio:.2f})')
            # Reorder results to put best candidate first
            final_results = [best_candidate] + [r for r in candidate_pool if r != best_candidate]
            final_results = final_results[:top_k]
            logger.info(f'✅ Selected best candidate: Score={best_candidate.score:.4f}, Theo Peaks={best_candidate.theoretical_peaks}, Matched Peaks={best_candidate.matched_peaks}, Mean Delta={best_candidate.mean_delta_nm:.2f}nm')
        else:
            # Fallback: use top by score if no candidates have peak data
            final_results = candidate_pool[:top_k]
            logger.info(f'⚠️ No candidates with peak alignment data, using top by score')
        
        logger.info(f'   - Returning top {top_k} results')
        if final_results:
            logger.info(f'🏆 Best result: Score={final_results[0].score:.4f}, Corr={final_results[0].correlation:.3f}, '
                       f'RMSE={final_results[0].rmse:.5f}, Matched Peaks={final_results[0].matched_peaks}, '
                       f'Mean Delta={final_results[0].mean_delta_nm:.2f}nm, '
                       f'L={final_results[0].lipid_nm:.1f}nm, A={final_results[0].aqueous_nm:.1f}nm, R={final_results[0].mucus_nm:.1f}Å')
        logger.info('=' * 80)
        
        # =====================================================================
        # STAGE 3: Ultra-Fine Refinement for better peak alignment
        # =====================================================================
        if final_results and final_results[0].mean_delta_nm > 2.5:
            logger.info('')
            logger.info('🔬 STAGE 3: Ultra-Fine Refinement (improving peak alignment)')
            logger.info('-' * 80)
            
            best = final_results[0]
            # Very small search window with tiny steps - respect user-specified ranges
            ultra_fine_lipid = np.arange(
                max(lipid_range[0], best.lipid_nm - 10), 
                min(lipid_range[1], best.lipid_nm + 10) + 1, 
                2.0  # 2nm step
            )
            ultra_fine_aqueous = np.arange(
                max(aqueous_range[0], best.aqueous_nm - 200),
                min(aqueous_range[1], best.aqueous_nm + 200) + 1,
                20.0  # 20nm step
            )
            ultra_fine_roughness = np.arange(
                max(roughness_range[0], best.mucus_nm - 200),
                min(roughness_range[1], best.mucus_nm + 200) + 1,
                50.0  # 50Å step
            )
            
            ultra_fine_total = len(ultra_fine_lipid) * len(ultra_fine_aqueous) * len(ultra_fine_roughness)
            logger.info(f'📊 Ultra-fine grid: {ultra_fine_total:,} combinations')
            logger.info(f'   - Lipid: {len(ultra_fine_lipid)} values ({ultra_fine_lipid[0]:.1f} to {ultra_fine_lipid[-1]:.1f} nm, step 2nm)')
            logger.info(f'   - Aqueous: {len(ultra_fine_aqueous)} values ({ultra_fine_aqueous[0]:.1f} to {ultra_fine_aqueous[-1]:.1f} nm, step 20nm)')
            logger.info(f'   - Roughness: {len(ultra_fine_roughness)} values ({ultra_fine_roughness[0]:.1f} to {ultra_fine_roughness[-1]:.1f} Å, step 50Å)')
            
            ultra_fine_results = self._evaluate_parameter_grid(
                wavelengths, measured,
                ultra_fine_lipid, ultra_fine_aqueous, ultra_fine_roughness,
                top_k, enable_roughness, min_correlation_filter
            )
            
            if ultra_fine_results:
                # Combine and select best
                all_results = final_results + ultra_fine_results
                
                # Filter by valid peak count and good amplitude
                valid_results = [r for r in all_results 
                                if r.theoretical_peaks <= meas_peaks_count + 2 
                                and 0.7 <= r.oscillation_ratio <= 1.5
                                and r.mean_delta_nm < 1000.0]
                
                if valid_results:
                    # Select by most matched peaks, then smallest mean_delta
                    max_matched = max(r.matched_peaks for r in valid_results)
                    top_matched = [r for r in valid_results if r.matched_peaks >= max_matched - 1]
                    best_ultra = min(top_matched, key=lambda x: x.mean_delta_nm)
                    
                    if best_ultra.mean_delta_nm < best.mean_delta_nm:
                        logger.info(f'✅ Ultra-fine improved: Mean Delta {best.mean_delta_nm:.2f}nm → {best_ultra.mean_delta_nm:.2f}nm')
                        final_results = [best_ultra] + [r for r in valid_results if r != best_ultra][:top_k-1]
                    else:
                        logger.info(f'ℹ️ Ultra-fine did not improve (current: {best.mean_delta_nm:.2f}nm, best found: {best_ultra.mean_delta_nm:.2f}nm)')
        
        return final_results[:top_k]
    
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
        
        # Use all available CPU cores with worker initializer
        num_workers = os.cpu_count() or 4
        logger.info(f'⚡ Using {num_workers} parallel workers for evaluation')
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_worker_initializer,
            initargs=(wavelengths, measured, material_data, enable_roughness),
        ) as executor:
            futures = {
                executor.submit(_evaluate_combination_fast, combo): combo
                for combo in combinations
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
        
        # Get measured peaks count for minimum matching requirement
        # Use focus region (600-1120 nm) to match scoring
        wl_mask = (wavelengths >= 600) & (wavelengths <= 1120)
        wl_focus = wavelengths[wl_mask] if wl_mask.any() else wavelengths
        meas_focus = measured[wl_mask] if wl_mask.any() else measured
        meas_peaks_count = _get_measured_peaks_count(wl_focus, meas_focus)
        min_matched_peaks = max(1, int(meas_peaks_count * 0.5))  # Require at least 50% of measured peaks
        
        # Stage 1: Sort by score, peak_count_delta, matched_peaks (for candidate identification)
        sort_key_func = lambda r: _get_candidate_sort_key(r, min_matched_peaks)
        results.sort(key=sort_key_func, reverse=True)
        
        # Filter candidates that meet minimum matched peaks requirement
        candidates_meeting_min = [r for r in results if r.matched_peaks >= min_matched_peaks]
        
        # Select top candidates (top 20% or at least top 10, whichever is larger)
        # Prefer candidates meeting minimum, but include others if not enough
        if len(candidates_meeting_min) >= 10:
            candidate_pool = candidates_meeting_min
            logger.info(f'📊 Filtering: {len(candidates_meeting_min)} candidates meet minimum matched peaks requirement ({min_matched_peaks} of {meas_peaks_count} measured peaks)')
        else:
            candidate_pool = results  # Use all if not enough meet minimum
            logger.warning(f'⚠️ Only {len(candidates_meeting_min)} candidates meet minimum matched peaks ({min_matched_peaks} of {meas_peaks_count}), using all candidates')
        
        num_candidates = max(10, len(candidate_pool) // 5)
        top_candidates = candidate_pool[:num_candidates]
        
        # Stage 2: From top candidates, select the one with smallest mean_delta_nm (best peak alignment)
        # Filter out candidates without peak alignment data (mean_delta_nm >= 1000.0)
        # AND filter out candidates with bad oscillation ratio (amplitude mismatch)
        candidates_with_peaks = [r for r in top_candidates if r.mean_delta_nm < 1000.0]
        # CRITICAL: Also filter by oscillation ratio (prefer 0.7 to 1.5 range)
        candidates_good_amplitude = [r for r in candidates_with_peaks if 0.7 <= r.oscillation_ratio <= 1.5]
        
        # CRITICAL: Filter out candidates with too many theoretical peaks
        # Max allowed: measured peaks + 2
        max_theo_peaks = meas_peaks_count + 2
        candidates_valid_peak_count = [r for r in candidates_good_amplitude if r.theoretical_peaks <= max_theo_peaks]
        
        if not candidates_valid_peak_count and candidates_good_amplitude:
            logger.warning(f'⚠️ No candidates with theo_peaks <= {max_theo_peaks}, relaxing constraint')
            candidates_valid_peak_count = candidates_good_amplitude
        
        if candidates_valid_peak_count:
            # Best case: good peaks, good amplitude, valid peak count
            candidates_meeting_all = [r for r in candidates_valid_peak_count if r.matched_peaks >= min_matched_peaks]
            if candidates_meeting_all:
                # Find max matched peaks, then select by smallest mean_delta among top matches
                max_matched = max(c.matched_peaks for c in candidates_meeting_all)
                top_matched = [c for c in candidates_meeting_all if c.matched_peaks >= max_matched - 1]
                best_candidate = min(top_matched, key=lambda x: x.mean_delta_nm)
            else:
                max_matched = max(c.matched_peaks for c in candidates_valid_peak_count)
                top_matched = [c for c in candidates_valid_peak_count if c.matched_peaks >= max_matched - 1]
                best_candidate = min(top_matched, key=lambda x: x.mean_delta_nm)
            logger.info(f'✅ Found {len(candidates_valid_peak_count)} candidates with good amplitude AND valid peak count (theo <= {max_theo_peaks})')
            # Reorder results to put best candidate first
            final_results = [best_candidate] + [r for r in candidate_pool if r != best_candidate]
            return final_results[:top_k]
        elif candidates_with_peaks:
            # Fallback: has peaks but other filters not met
            candidates_valid_in_fallback = [r for r in candidates_with_peaks if r.theoretical_peaks <= max_theo_peaks]
            if candidates_valid_in_fallback:
                best_candidate = max(candidates_valid_in_fallback, key=lambda x: (x.matched_peaks, -x.mean_delta_nm))
            else:
                best_candidate = max(candidates_with_peaks, key=lambda x: (x.matched_peaks, -x.mean_delta_nm))
            logger.warning(f'⚠️ No candidates with good amplitude, using best peak match (theo_peaks={best_candidate.theoretical_peaks})')
            # Reorder results to put best candidate first
            final_results = [best_candidate] + [r for r in candidate_pool if r != best_candidate]
            return final_results[:top_k]
        
        # Fallback: use top by score if no candidates have peak data
        return candidate_pool[:top_k]
    
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

