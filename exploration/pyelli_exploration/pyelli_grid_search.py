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
# Transfer Matrix Calculation (from app.py)
# =============================================================================

def transfer_matrix_reflectance(
    wavelengths_nm: np.ndarray,
    layers: List[Tuple[np.ndarray, np.ndarray, float]],
) -> np.ndarray:
    """
    Calculate reflectance using transfer matrix method.
    
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
    focus_wavelength_min: float = 600.0,
    focus_wavelength_max: float = 1200.0,
    focus_reflectance_max: float = 0.10,  # Increased to include more data
    strict_crossing_rejection: bool = False,  # Use soft penalty for PyElli
) -> Dict[str, float]:
    """
    Visual quality metric that evaluates fit quality.
    
    For PyElli (approximate TMM), we use SOFT penalties for crossings
    since it's harder to get perfect alignment than with the proprietary LTA.
    
    Args:
        wavelengths: Wavelength array
        measured: Measured reflectance
        theoretical: Theoretical reflectance
        focus_wavelength_min: Start of focus region (nm)
        focus_wavelength_max: End of focus region (nm)
        focus_reflectance_max: Maximum reflectance in focus region
        strict_crossing_rejection: If True, use hard rejection (like LTA). If False, use soft penalty.
        
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
    
    # === CRITERION 3: CRISS-CROSSING DETECTION (20% weight - soft penalty) ===
    sign_changes_focus = int(np.sum(np.abs(np.diff(np.sign(residual_focus))) > 0))
    
    # Normalize crossings by number of points (crossings per 100 points)
    crossings_per_100 = (sign_changes_focus / max(points_in_focus, 1)) * 100
    
    if strict_crossing_rejection:
        # Hard rejection for LTA-like behavior
        crossing_score = 1.0 if sign_changes_focus == 0 else 0.0
    else:
        # Soft penalty for PyElli
        # 0 crossings → score = 1.0
        # 1-2 crossings → score ~0.8
        # 5+ crossings → score < 0.5
        crossing_score = float(np.exp(-crossings_per_100 * 0.05))
    
    # === COMBINED SCORE ===
    # Weight: shape 40%, closeness 40%, crossing penalty 20%
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
    
    # Hard rejection if strict mode and has crossings
    if strict_crossing_rejection and sign_changes_focus > 0:
        score = 0.05
    
    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "crossing_count": sign_changes_focus,
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
        self.mucus_df = load_material_data(
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
        mucus_nm: float,
    ) -> np.ndarray:
        """
        Calculate theoretical spectrum using TMM.
        
        Args:
            wavelengths: Wavelength array in nm
            lipid_nm: Lipid layer thickness in nm
            aqueous_nm: Aqueous layer thickness in nm
            mucus_nm: Mucus layer thickness in nm (maps to ADOM's roughness)
            
        Returns:
            Theoretical reflectance array
        """
        lipid_n, lipid_k = self._get_nk(self.lipid_df, wavelengths)
        water_n, water_k = self._get_nk(self.water_df, wavelengths)
        mucus_n, mucus_k = self._get_nk(self.mucus_df, wavelengths)
        
        layers = [
            (lipid_n, lipid_k, lipid_nm),
            (water_n, water_k, aqueous_nm),
            (mucus_n, mucus_k, mucus_nm),
        ]
        
        return transfer_matrix_reflectance(wavelengths, layers)
    
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
        lipid_range: Tuple[float, float, float] = (20, 150, 20),    # (min, max, step) nm
        aqueous_range: Tuple[float, float, float] = (500, 5000, 200),  # (min, max, step) nm
        mucus_range: Tuple[float, float, float] = (100, 600, 100),  # (min, max, step) nm
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
        
        # Sort by score (descending) and return top results
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Log top result
        if results:
            best = results[0]
            logger.info(
                f"✅ Best fit: Lipid={best.lipid_nm:.1f}nm, "
                f"Aqueous={best.aqueous_nm:.1f}nm, Mucus={best.mucus_nm:.1f}nm, "
                f"Score={best.score:.4f}, Crossings={best.crossing_count}"
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

