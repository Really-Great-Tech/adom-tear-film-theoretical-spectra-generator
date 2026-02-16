"""
Evaluate New Spectra: Grid Search and Deviation Score Analysis

This script runs grid search on all spectra in the "new spectra" directory
and calculates the PyElli vs LTA BestFit deviation score to determine if
each spectrum meets the 10% threshold target.

Usage:
    python evaluate_new_spectra.py
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json
from datetime import datetime

import numpy as np
import pandas as pd

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exploration.pyelli_exploration.pyelli_utils import (
    load_measured_spectrum,
    load_bestfit_spectrum,
)
from exploration.pyelli_exploration.pyelli_grid_search import (
    PyElliGridSearch,
    calculate_peak_based_score,
    mape_pyelli_vs_lta_normalized,
)
from src.analysis.measurement_utils import detrend_signal, detect_peaks
from src.analysis.metrics import _match_peaks

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Paths
NEW_SPECTRA_PATH = PROJECT_ROOT / 'exploration' / 'new_spectra'
MATERIALS_PATH = PROJECT_ROOT / 'data' / 'Materials'


@dataclass
class SpectrumResult:
    """Result for a single spectrum."""
    sample_id: str
    measured_file: str
    bestfit_file: str
    
    # PyElli grid search results
    pyelli_lipid: float = 0.0
    pyelli_aqueous: float = 0.0
    pyelli_roughness: float = 0.0
    pyelli_score: float = 0.0
    pyelli_correlation: float = 0.0
    pyelli_rmse: float = 0.0
    pyelli_matched_peaks: int = 0
    pyelli_mean_delta_nm: float = 0.0
    pyelli_meas_peaks: int = 0
    pyelli_theo_peaks: int = 0
    
    # LTA BestFit metrics (for reference)
    lta_matched_peaks: int = 0
    lta_meas_peaks: int = 0
    lta_bestfit_peaks: int = 0
    
    # Deviation score components
    mape: float = 0.0
    peak_match_deviation: float = 0.0
    alignment_deviation: float = 0.0
    composite_deviation: float = 0.0
    
    # Status
    within_threshold: bool = False
    status: str = ""  # "Excellent", "Good", "Moderate", "High"
    
    # Error tracking
    error: Optional[str] = None
    grid_search_duration_seconds: float = 0.0


def calculate_deviation_score(
    wavelengths: np.ndarray,
    pyelli_theo: np.ndarray,
    lta_bestfit: np.ndarray,
    pyelli_peak_result: Dict,
    lta_peak_result: Dict,
) -> Tuple[float, float, float, float]:
    """
    Calculate deviation score components between PyElli and LTA BestFit.
    
    Returns:
        Tuple of (mape, peak_match_deviation, alignment_deviation, composite_deviation)
    """
    # Ensure same length
    min_len = min(len(pyelli_theo), len(lta_bestfit))
    pyelli_aligned = pyelli_theo[:min_len]
    lta_aligned = lta_bestfit[:min_len]
    
    # 1. MAPE between PyElli and LTA (normalize to unit L2 first = shape comparison, scale-invariant)
    mape = mape_pyelli_vs_lta_normalized(pyelli_aligned, lta_aligned)
    
    # 2. Peak match rate deviation
    # Compare PyElli matched peaks to LTA's peak count (as reference)
    pyelli_matched = pyelli_peak_result.get('matched_peaks', 0)
    lta_peaks = lta_peak_result.get('measurement_peaks', 0)  # LTA matched peaks against measured
    if lta_peaks > 0:
        peak_match_deviation = (1.0 - (pyelli_matched / lta_peaks)) * 100
        peak_match_deviation = max(0.0, peak_match_deviation)  # Can't be negative
    else:
        peak_match_deviation = 0.0
    
    # 3. Alignment deviation (mean_delta relative to typical peak spacing ~50nm)
    pyelli_mean_delta = pyelli_peak_result.get('mean_delta_nm', 0.0)
    reference_spacing = 50.0  # Typical peak-to-peak spacing in nm
    alignment_deviation = (pyelli_mean_delta / reference_spacing) * 100
    
    # Composite deviation score (weighted average)
    # MAPE: 40%, Peak Match: 30%, Alignment: 30%
    composite_deviation = 0.40 * mape + 0.30 * peak_match_deviation + 0.30 * alignment_deviation
    
    return mape, peak_match_deviation, alignment_deviation, composite_deviation


def find_spectrum_files(spectrum_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Find measured and bestfit files in a spectrum directory."""
    measured_file = None
    bestfit_file = None
    
    for f in spectrum_dir.glob('*.txt'):
        if '_BestFit' in f.name:
            bestfit_file = f
        elif f.name.startswith('(Run)spectra_') and '_BestFit' not in f.name:
            measured_file = f
    
    return measured_file, bestfit_file


def analyze_single_spectrum(
    sample_id: str,
    measured_file: Path,
    bestfit_file: Path,
    grid_search: PyElliGridSearch,
) -> SpectrumResult:
    """
    Analyze a single spectrum by running grid search and calculating deviation score.
    """
    result = SpectrumResult(
        sample_id=sample_id,
        measured_file=measured_file.name,
        bestfit_file=bestfit_file.name,
    )
    
    import time
    start_time = time.time()
    
    try:
        # Load measured spectrum
        wavelengths, measured = load_measured_spectrum(measured_file)
        
        # Load LTA BestFit spectrum
        lta_wl, lta_refl = load_bestfit_spectrum(bestfit_file)
        
        # Focus on 600-1120 nm range (LTA analysis region)
        wl_mask = (wavelengths >= 600) & (wavelengths <= 1120)
        wl_display = wavelengths[wl_mask]
        meas_display = measured[wl_mask]
        
        # Interpolate LTA BestFit to match wavelengths
        lta_interp = np.interp(wl_display, lta_wl, lta_refl)
        
        # === RUN PYELLI GRID SEARCH ===
        # Optimized ranges based on empirical analysis:
        # - Lipid: 10nm step (was 5nm) - sufficient resolution
        # - Aqueous: 300nm step (was 200nm) - coarser for speed
        # - Roughness: 6000-7000 Å (was 600-7000) - good fits occur at ≥6000 Å
        logger.info(f'  Running grid search for {sample_id}...')
        pyelli_results = grid_search.run_grid_search(
            wavelengths, measured,
            lipid_range=(9, 250, 10),
            aqueous_range=(800, 12000, 300),
            roughness_range=(6000, 7000, 100),
            top_k=1,  # Only need the best result
            enable_roughness=True,
            fine_search=True,
            strategy='Dynamic Search',
            max_combinations=15000,
        )
        
        if not pyelli_results:
            result.error = "No grid search results found"
            return result
        
        best = pyelli_results[0]
        result.pyelli_lipid = best.lipid_nm
        result.pyelli_aqueous = best.aqueous_nm
        result.pyelli_roughness = best.mucus_nm
        result.pyelli_score = best.score
        result.pyelli_correlation = best.correlation
        result.pyelli_rmse = best.rmse
        result.pyelli_matched_peaks = best.matched_peaks
        result.pyelli_mean_delta_nm = best.mean_delta_nm
        
        # Get PyElli theoretical spectrum for display range
        pyelli_theo_full = best.theoretical_spectrum
        if len(pyelli_theo_full) > len(wl_display):
            pyelli_theo_display = pyelli_theo_full[wl_mask]
        else:
            # Interpolate if needed
            pyelli_theo_display = np.interp(wl_display, wavelengths[:len(pyelli_theo_full)], pyelli_theo_full)
        
        # Calculate peak-based scores for both PyElli and LTA BestFit
        pyelli_peak_result = calculate_peak_based_score(
            wl_display, meas_display, pyelli_theo_display
        )
        lta_peak_result = calculate_peak_based_score(
            wl_display, meas_display, lta_interp
        )
        
        result.pyelli_meas_peaks = int(pyelli_peak_result.get('measurement_peaks', 0))
        result.pyelli_theo_peaks = int(pyelli_peak_result.get('theoretical_peaks', 0))
        result.lta_meas_peaks = int(lta_peak_result.get('measurement_peaks', 0))
        result.lta_bestfit_peaks = int(lta_peak_result.get('theoretical_peaks', 0))
        result.lta_matched_peaks = int(lta_peak_result.get('matched_peaks', 0))
        
        # Calculate deviation score
        mape, peak_match_dev, align_dev, composite_dev = calculate_deviation_score(
            wl_display, pyelli_theo_display, lta_interp,
            pyelli_peak_result, lta_peak_result
        )
        
        result.mape = mape
        result.peak_match_deviation = peak_match_dev
        result.alignment_deviation = align_dev
        result.composite_deviation = composite_dev
        
        # Determine status
        if composite_dev <= 10:
            result.status = "Excellent Match"
            result.within_threshold = True
        elif composite_dev <= 15:
            result.status = "Good Match"
            result.within_threshold = False
        elif composite_dev <= 25:
            result.status = "Moderate Deviation"
            result.within_threshold = False
        else:
            result.status = "High Deviation"
            result.within_threshold = False
        
        result.grid_search_duration_seconds = time.time() - start_time
        
    except Exception as e:
        result.error = str(e)
        result.grid_search_duration_seconds = time.time() - start_time
        logger.error(f'  Error analyzing {sample_id}: {e}')
        import traceback
        traceback.print_exc()
    
    return result


def print_results_table(results: List[SpectrumResult]):
    """Print a formatted table of results."""
    print("\n" + "=" * 140)
    print("NEW SPECTRA EVALUATION RESULTS")
    print("=" * 140)
    
    # Header
    header = (
        f"{'ID':<12} "
        f"{'Deviation':<12} {'Status':<20} "
        f"{'PyElli Score':<12} {'PyElli Corr':<12} {'PyElli Peaks':<12} "
        f"{'LTA Peaks':<10} {'MAPE':<8} {'Peak Δ':<8} {'Align Δ':<8} "
        f"{'Time (s)':<10}"
    )
    print(header)
    print("-" * 140)
    
    # Sort by deviation (best first)
    sorted_results = sorted(results, key=lambda x: x.composite_deviation if x.error is None else 999.0)
    
    for r in sorted_results:
        if r.error:
            row = f"{r.sample_id:<12} {'ERROR':<12} {r.error[:18]:<20} {'-':<12} {'-':<12} {'-':<12} {'-':<10} {'-':<8} {'-':<8} {'-':<8} {'-':<10}"
        else:
            row = (
                f"{r.sample_id:<12} "
                f"{r.composite_deviation:<12.1f} {r.status:<20} "
                f"{r.pyelli_score:<12.4f} {r.pyelli_correlation:<12.3f} {r.pyelli_matched_peaks:<12} "
                f"{r.lta_matched_peaks:<10} {r.mape:<8.1f} {r.peak_match_deviation:<8.1f} {r.alignment_deviation:<8.1f} "
                f"{r.grid_search_duration_seconds:<10.1f}"
            )
        print(row)
    
    print("-" * 140)


def print_summary(results: List[SpectrumResult]):
    """Print summary statistics."""
    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal Spectra: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    
    if successful:
        within_threshold = [r for r in successful if r.within_threshold]
        print(f"\nDeviation Score Results:")
        print(f"  Within 10% threshold: {len(within_threshold)} / {len(successful)} ({100*len(within_threshold)/len(successful):.1f}%)")
        
        deviations = [r.composite_deviation for r in successful]
        print(f"  Average deviation: {np.mean(deviations):.1f}%")
        print(f"  Min deviation: {np.min(deviations):.1f}%")
        print(f"  Max deviation: {np.max(deviations):.1f}%")
        print(f"  Median deviation: {np.median(deviations):.1f}%")
        
        # Status breakdown
        excellent = [r for r in successful if r.status == "Excellent Match"]
        good = [r for r in successful if r.status == "Good Match"]
        moderate = [r for r in successful if r.status == "Moderate Deviation"]
        high = [r for r in successful if r.status == "High Deviation"]
        
        print(f"\nStatus Breakdown:")
        print(f"  🟢 Excellent Match (≤10%): {len(excellent)}")
        print(f"  🟡 Good Match (10-15%): {len(good)}")
        print(f"  🟠 Moderate Deviation (15-25%): {len(moderate)}")
        print(f"  🔴 High Deviation (>25%): {len(high)}")
        
        # Average metrics
        print(f"\nAverage Metrics:")
        print(f"  PyElli Score: {np.mean([r.pyelli_score for r in successful]):.4f}")
        print(f"  PyElli Correlation: {np.mean([r.pyelli_correlation for r in successful]):.3f}")
        print(f"  PyElli Matched Peaks: {np.mean([r.pyelli_matched_peaks for r in successful]):.1f}")
        print(f"  LTA Matched Peaks: {np.mean([r.lta_matched_peaks for r in successful]):.1f}")
        
        # Average time
        avg_time = np.mean([r.grid_search_duration_seconds for r in successful])
        print(f"\nPerformance:")
        print(f"  Average grid search time: {avg_time:.1f} seconds ({avg_time/60:.1f} minutes)")
        total_time = sum([r.grid_search_duration_seconds for r in successful])
        print(f"  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    if failed:
        print(f"\nFailed Analyses:")
        for r in failed:
            print(f"  - {r.sample_id}: {r.error}")
    
    print("\n" + "=" * 80)


def save_results(results: List[SpectrumResult], output_path: Path):
    """Save results to JSON for further analysis."""
    results_dict = {
        'timestamp': datetime.now().isoformat(),
        'total_spectra': len(results),
        'results': [
            {
                'sample_id': r.sample_id,
                'measured_file': r.measured_file,
                'bestfit_file': r.bestfit_file,
                'pyelli_lipid': r.pyelli_lipid,
                'pyelli_aqueous': r.pyelli_aqueous,
                'pyelli_roughness': r.pyelli_roughness,
                'pyelli_score': r.pyelli_score,
                'pyelli_correlation': r.pyelli_correlation,
                'pyelli_rmse': r.pyelli_rmse,
                'pyelli_matched_peaks': r.pyelli_matched_peaks,
                'pyelli_mean_delta_nm': r.pyelli_mean_delta_nm,
                'pyelli_meas_peaks': r.pyelli_meas_peaks,
                'pyelli_theo_peaks': r.pyelli_theo_peaks,
                'lta_matched_peaks': r.lta_matched_peaks,
                'lta_meas_peaks': r.lta_meas_peaks,
                'lta_bestfit_peaks': r.lta_bestfit_peaks,
                'mape': r.mape,
                'peak_match_deviation': r.peak_match_deviation,
                'alignment_deviation': r.alignment_deviation,
                'composite_deviation': r.composite_deviation,
                'within_threshold': r.within_threshold,
                'status': r.status,
                'error': r.error,
                'grid_search_duration_seconds': r.grid_search_duration_seconds,
            }
            for r in results
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {output_path}")


def main():
    """Main function to evaluate all new spectra."""
    print("\n" + "=" * 80)
    print("NEW SPECTRA EVALUATION")
    print("Running grid search and calculating deviation scores for all new spectra")
    print("=" * 80 + "\n")
    
    if not NEW_SPECTRA_PATH.exists():
        logger.error(f"❌ New spectra directory not found: {NEW_SPECTRA_PATH}")
        return
    
    # Initialize grid search
    logger.info("Loading materials...")
    grid_search = PyElliGridSearch(MATERIALS_PATH)
    
    results: List[SpectrumResult] = []
    
    # Process all spectrum directories
    spectrum_dirs = sorted([d for d in NEW_SPECTRA_PATH.iterdir() if d.is_dir()])
    logger.info(f"Found {len(spectrum_dirs)} spectrum directories\n")
    
    for i, spectrum_dir in enumerate(spectrum_dirs, 1):
        sample_id = spectrum_dir.name
        measured_file, bestfit_file = find_spectrum_files(spectrum_dir)
        
        if not measured_file:
            logger.warning(f"[{i}/{len(spectrum_dirs)}] Skipping {sample_id}: No measured file found")
            continue
        
        if not bestfit_file:
            logger.warning(f"[{i}/{len(spectrum_dirs)}] Skipping {sample_id}: No BestFit file found")
            continue
        
        logger.info(f"[{i}/{len(spectrum_dirs)}] Analyzing {sample_id}...")
        result = analyze_single_spectrum(
            sample_id=sample_id,
            measured_file=measured_file,
            bestfit_file=bestfit_file,
            grid_search=grid_search,
        )
        results.append(result)
        
        if result.error:
            logger.error(f"  ❌ Failed: {result.error}")
        else:
            logger.info(f"  ✅ Deviation: {result.composite_deviation:.1f}% ({result.status})")
    
    # Print results
    print_results_table(results)
    print_summary(results)
    
    # Save results
    output_path = PROJECT_ROOT / 'exploration' / 'pyelli_exploration' / 'new_spectra_evaluation_results.json'
    save_results(results, output_path)
    
    print("\n✅ Evaluation complete!")
    print(f"   Results saved to: {output_path}")
    print("\nNext steps:")
    print("1. Review the summary to see how many spectra meet the 10% threshold")
    print("2. Analyze spectra that exceed the threshold to identify improvement opportunities")
    print("3. Compare PyElli parameters across successful vs unsuccessful cases")


if __name__ == "__main__":
    main()

