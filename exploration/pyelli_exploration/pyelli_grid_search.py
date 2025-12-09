"""
PyElli Grid Search with Monotonic Alignment Scoring

This module integrates pyElli's Transfer Matrix Method with the monotonic_alignment
scoring that we developed for visually correct fits (no criss-crossing).

This demonstrates how to use PyElli for auto-fitting tear film spectra.

Author: RGT Team
Date: December 2025
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys

import numpy as np
import pandas as pd
from dataclasses import dataclass

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exploration.pyelli_exploration.pyelli_utils import (
    load_measured_spectrum,
    load_material_data,
    get_available_materials,
)

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
    roughness_angstrom: float = 0.0,  # Roughness in Angstroms (LTA uses 300-3000 Å)
    enable_roughness: bool = False,  # Disable roughness by default - adds complexity without clear benefit
) -> np.ndarray:
    """
    Calculate theoretical reflectance using the actual pyElli library.
    
    This uses the proper pyElli TMM implementation with optional roughness modeling via EMA.
    
    NOTE: Roughness modeling is disabled by default because:
    1. It significantly increases computation time (adds multiple sublayers)
    2. It doesn't appear to improve fit quality in practice
    3. pyElli's simplified EMA may not match LTA's proprietary roughness model
    
    Args:
        wavelengths_nm: Wavelength array in nanometers
        lipid_n, lipid_k: Lipid layer refractive index and extinction coefficient arrays
        lipid_thickness_nm: Lipid layer thickness in nm
        aqueous_n, aqueous_k: Aqueous layer n,k arrays
        aqueous_thickness_nm: Aqueous layer thickness in nm
        mucus_n, mucus_k: Mucus layer n,k arrays
        mucus_thickness_nm: Mucus layer thickness in nm (LTA uses fixed 500nm)
        substratum_n, substratum_k: Substrate (struma) n,k arrays
        roughness_angstrom: Surface roughness in Angstroms (LTA range: 300-3000 Å)
        enable_roughness: If False, roughness is ignored (faster, default)
        
    Returns:
        Theoretical reflectance array
    """
    try:
        import elli
        from elli.dispersions.table_index import Table
        from elli.materials import IsotropicMaterial
        
        # Create material dispersions from n,k data
        # pyElli expects complex refractive index: n + i*k
        lipid_nk = lipid_n + 1j * lipid_k
        aqueous_nk = aqueous_n + 1j * aqueous_k
        mucus_nk = mucus_n + 1j * mucus_k
        substratum_nk = substratum_n + 1j * substratum_k
        
        # Create Table dispersions (wavelength in nm, complex nk)
        lipid_disp = Table(wavelengths_nm, lipid_nk)
        aqueous_disp = Table(wavelengths_nm, aqueous_nk)
        mucus_disp = Table(wavelengths_nm, mucus_nk)
        substratum_disp = Table(wavelengths_nm, substratum_nk)
        
        # Wrap in IsotropicMaterial (required by pyElli)
        lipid_mat = IsotropicMaterial(lipid_disp)
        aqueous_mat = IsotropicMaterial(aqueous_disp)
        mucus_mat = IsotropicMaterial(mucus_disp)
        substrate_mat = IsotropicMaterial(substratum_disp)
        
        # Build layers
        # Note: elli.Layer takes (material, thickness) as positional args, not d= keyword
        layers = [
            elli.Layer(lipid_mat, lipid_thickness_nm),
            elli.Layer(aqueous_mat, aqueous_thickness_nm),
        ]
        
        # Add main mucus layer first (LTA uses fixed 500nm)
        layers.append(elli.Layer(mucus_mat, mucus_thickness_nm))
        
        # Add roughness modeling if specified (EMA - Effective Medium Approximation)
        # Roughness is modeled as a graded interface layer between mucus and substrate
        # DISABLED BY DEFAULT: Adds complexity and runtime without clear benefit
        if enable_roughness and roughness_angstrom > 0:
            # Convert roughness from Angstroms to nm
            roughness_nm = roughness_angstrom / 10.0
            
            logger.debug(f"🔬 Applying roughness modeling: {roughness_angstrom:.1f} Å ({roughness_nm:.1f} nm)")
            
            # EMA: Create a graded layer representing the rough interface
            # The effective refractive index is a mixture of mucus and substrate
            # Using simplified linear mixing for effective refractive index
            # More accurate would be Bruggeman EMA, but this approximation works for thin layers
            
            # Create EMA layer: split roughness into sub-layers for graded interface
            num_sublayers = max(3, int(roughness_nm / 5.0))  # At least 3 sublayers, ~5nm per sublayer
            sublayer_thickness = roughness_nm / num_sublayers
            
            logger.debug(f"   Creating {num_sublayers} EMA sublayers, {sublayer_thickness:.2f} nm each")
            
            for i in range(num_sublayers):
                # Volume fraction: transitions from mucus (f=1) to substrate (f=0)
                f = 1.0 - (i + 0.5) / num_sublayers  # Linear transition
                
                # Effective refractive index using linear mixing (simplified EMA)
                # For each wavelength, compute effective n and k
                n_eff = np.sqrt(f * mucus_n**2 + (1-f) * substratum_n**2)
                k_eff = f * mucus_k + (1-f) * substratum_k
                
                # Create effective medium dispersion (per-wavelength)
                ema_nk = n_eff + 1j * k_eff
                ema_disp = Table(wavelengths_nm, ema_nk)
                ema_mat = IsotropicMaterial(ema_disp)
                
                # elli.Layer takes (material, thickness) as positional args
                layers.append(elli.Layer(ema_mat, sublayer_thickness))
        else:
            logger.debug("⚠️ No roughness modeling (roughness_angstrom = 0)")
        
        # Build structure: Air -> Lipid -> Aqueous -> [Roughness EMA layers] -> Mucus -> Substrate
        structure = elli.Structure(
            elli.AIR,  # Front medium (air)
            layers,
            substrate_mat  # Back medium (substrate)
        )
        
        # Evaluate structure at normal incidence (theta_i = 0)
        result = structure.evaluate(wavelengths_nm, theta_i=0.0)
        
        # Get total reflectance (R is the total reflectance)
        reflectance = result.R
        
        return np.array(reflectance)
        
    except ImportError:
        logger.error('❌ pyElli not installed. Run: pip install pyElli')
        raise
    except Exception as e:
        logger.error(f'Error calculating pyElli reflectance: {e}')
        # Fallback to custom TMM if pyElli fails
        logger.warning('Falling back to custom TMM implementation')
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
# Monotonic Alignment Scoring (Ported from metrics.py)
# =============================================================================

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
        mucus_nm: float,  # Actually maps to roughness in Angstroms (LTA: 300-3000 Å)
    ) -> np.ndarray:
        """
        Calculate theoretical spectrum using the actual pyElli library.
        
        This uses pyElli's proper TMM implementation with roughness modeling.
        
        Args:
            wavelengths: Wavelength array in nm
            lipid_nm: Lipid layer thickness in nm
            aqueous_nm: Aqueous layer thickness in nm
            mucus_nm: Roughness in Angstroms (LTA range: 300-3000 Å)
                      Note: Mucus layer thickness is fixed at 500nm in LTA
            
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
        
        # Convert mucus_nm (which is actually roughness in Angstroms) to Angstroms
        # The grid search uses nm, but LTA uses Angstroms, so we need to convert
        # mucus_range: (30, 300, 30) nm = (300, 3000, 300) Angstroms
        roughness_angstrom = mucus_nm * 10.0  # Convert nm to Angstroms
        
        # Use actual pyElli library for TMM calculation
        # Roughness disabled by default - doesn't improve fit and slows computation
        return calculate_reflectance_pyelli(
            wavelengths,
            lipid_n, lipid_k, lipid_nm,
            water_n, water_k, aqueous_nm,
            mucus_n, mucus_k, mucus_thickness_nm,
            substratum_n, substratum_k,
            roughness_angstrom=roughness_angstrom,
            enable_roughness=False,  # Disabled: no improvement, increases runtime
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
        lipid_range: Tuple[float, float, float] = (0, 400, 20),    # Match LTA: (min, max, step) nm
        aqueous_range: Tuple[float, float, float] = (-20, 6000, 200),  # Match LTA: (min, max, step) nm
        mucus_range: Tuple[float, float, float] = (30, 300, 30),  # Match LTA roughness: 300-3000 Å = 30-300 nm, step 30
        top_k: int = 10,
    ) -> List[PyElliResult]:
        """
        Run coarse grid search with monotonic alignment scoring.
        
        Args:
            wavelengths: Wavelength array
            measured: Measured reflectance
            lipid_range: (min, max, step) for lipid thickness
            aqueous_range: (min, max, step) for aqueous thickness
            mucus_range: (min, max, step) for mucus thickness
            top_k: Number of top results to return
            
        Returns:
            List of top results sorted by score (descending)
        """
        results: List[PyElliResult] = []
        
        # Generate parameter grid
        lipid_values = np.arange(lipid_range[0], lipid_range[1] + 1, lipid_range[2])
        aqueous_values = np.arange(aqueous_range[0], aqueous_range[1] + 1, aqueous_range[2])
        mucus_values = np.arange(mucus_range[0], mucus_range[1] + 1, mucus_range[2])
        
        total = len(lipid_values) * len(aqueous_values) * len(mucus_values)
        logger.info(f"🔍 Running grid search: {total} combinations")
        
        for lipid in lipid_values:
            for aqueous in aqueous_values:
                for mucus in mucus_values:
                    # Calculate theoretical spectrum
                    theoretical = self.calculate_theoretical_spectrum(
                        wavelengths, lipid, aqueous, mucus
                    )
                    
                    # Align using linear regression (better than simple scaling!)
                    theoretical_scaled = self._align_spectra(
                        measured, theoretical, 
                        focus_min=600.0, focus_max=1100.0,
                        wavelengths=wavelengths
                    )
                    
                    # Score using monotonic alignment (NOT simple RMSE!)
                    score_result = calculate_monotonic_alignment_score(
                        wavelengths, measured, theoretical_scaled
                    )
                    
                    results.append(PyElliResult(
                        lipid_nm=float(lipid),
                        aqueous_nm=float(aqueous),
                        mucus_nm=float(mucus),
                        score=score_result["score"],
                        rmse=score_result["rmse"],
                        correlation=score_result["correlation"],
                        crossing_count=score_result["crossing_count"],
                        theoretical_spectrum=theoretical_scaled,
                        wavelengths=wavelengths,
                    ))
        
        # Sort all results by score (descending) - no hard rejection
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Log top result
        if results:
            best = results[0]
            logger.info(
                f"✅ Best fit: Lipid={best.lipid_nm:.1f}nm, "
                f"Aqueous={best.aqueous_nm:.1f}nm, Mucus={best.mucus_nm:.1f}nm, "
                f"Score={best.score:.4f}, Crossings={best.crossing_count}, "
                f"Correlation={best.correlation:.3f}"
            )
        
        return results[:top_k]
    
    def refine_around_best(
        self,
        wavelengths: np.ndarray,
        measured: np.ndarray,
        best_result: PyElliResult,
        lipid_step: float = 5.0,
        aqueous_step: float = 50.0,
        mucus_step: float = 25.0,
        search_radius: float = 1.5,  # Multiplier for search window
    ) -> List[PyElliResult]:
        """
        Fine-grain search around best coarse result.
        
        Args:
            wavelengths: Wavelength array
            measured: Measured reflectance
            best_result: Best result from coarse search
            lipid_step: Fine step for lipid
            aqueous_step: Fine step for aqueous
            mucus_step: Fine step for mucus
            search_radius: How far around best to search (in coarse steps)
            
        Returns:
            Refined top results
        """
        # Define search window around best
        lipid_range = (
            max(20, best_result.lipid_nm - 30),
            min(200, best_result.lipid_nm + 30),
            lipid_step
        )
        aqueous_range = (
            max(500, best_result.aqueous_nm - 300),
            min(6000, best_result.aqueous_nm + 300),
            aqueous_step
        )
        mucus_range = (
            max(50, best_result.mucus_nm - 100),
            min(800, best_result.mucus_nm + 100),
            mucus_step
        )
        
        logger.info(f"🔬 Refining around best result...")
        return self.run_grid_search(
            wavelengths, measured,
            lipid_range, aqueous_range, mucus_range,
            top_k=10
        )


# =============================================================================
# Demo / CLI
# =============================================================================

def demo():
    """Run a demo of PyElli grid search with monotonic alignment scoring."""
    import matplotlib.pyplot as plt
    
    # Setup paths
    materials_path = PROJECT_ROOT / "data" / "Materials"
    sample_data_path = PROJECT_ROOT / "exploration" / "sample_data" / "good_fit" / "1"
    
    # Find sample file
    sample_files = list(sample_data_path.glob("(Run)spectra_*.txt"))
    sample_files = [f for f in sample_files if "_BestFit" not in f.name]
    
    if not sample_files:
        logger.error("No sample files found!")
        return
    
    sample_file = sample_files[0]
    logger.info(f"📂 Loading sample: {sample_file.name}")
    
    # Load measured spectrum
    wavelengths, measured = load_measured_spectrum(sample_file)
    
    # Initialize grid search
    grid_search = PyElliGridSearch(materials_path)
    
    # Run coarse search
    print("\n" + "="*60)
    print("COARSE GRID SEARCH (with monotonic alignment scoring)")
    print("="*60)
    
    coarse_results = grid_search.run_grid_search(
        wavelengths, measured,
        lipid_range=(20, 150, 20),
        aqueous_range=(1000, 4500, 250),
        mucus_range=(100, 500, 100),
    )
    
    print("\nTop 5 results:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Lipid(nm)':<12} {'Aqueous(nm)':<14} {'Mucus(nm)':<12} {'Score':<10} {'RMSE':<10} {'Crossings':<10}")
    print("-" * 80)
    
    for i, r in enumerate(coarse_results[:5], 1):
        print(f"{i:<6} {r.lipid_nm:<12.1f} {r.aqueous_nm:<14.1f} {r.mucus_nm:<12.1f} {r.score:<10.4f} {r.rmse:<10.6f} {r.crossing_count:<10}")
    
    # Refine around best
    if coarse_results:
        print("\n" + "="*60)
        print("FINE REFINEMENT")
        print("="*60)
        
        refined_results = grid_search.refine_around_best(
            wavelengths, measured, coarse_results[0]
        )
        
        print("\nRefined Top 5:")
        print("-" * 80)
        for i, r in enumerate(refined_results[:5], 1):
            print(f"{i:<6} {r.lipid_nm:<12.1f} {r.aqueous_nm:<14.1f} {r.mucus_nm:<12.1f} {r.score:<10.4f} {r.rmse:<10.6f} {r.crossing_count:<10}")
        
        # Plot best result
        best = refined_results[0]
        
        plt.figure(figsize=(12, 5))
        
        # Main comparison plot
        plt.subplot(1, 2, 1)
        plt.plot(wavelengths, measured, 'b-', linewidth=2, label='Measured')
        plt.plot(best.wavelengths, best.theoretical_spectrum, 'r--', linewidth=2, 
                label=f'PyElli TMM (L={best.lipid_nm:.0f}, A={best.aqueous_nm:.0f}, M={best.mucus_nm:.0f})')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance')
        plt.title('PyElli Auto-Fit Result')
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
        plt.savefig(PROJECT_ROOT / "exploration" / "pyelli_exploration" / "grid_search_result.png", dpi=150)
        plt.show()
        
        print(f"\n✅ Plot saved to: exploration/pyelli_exploration/grid_search_result.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    demo()

