
import sys
import os
from pathlib import Path
import numpy as np

# Add project root for imports
PROJECT_ROOT = Path("/Users/dede/Lab/adom-tear-film-theoretical-spectra-generator")
sys.path.insert(0, str(PROJECT_ROOT))

# Also add exploration path if needed
sys.path.insert(0, str(PROJECT_ROOT / "exploration" / "pyelli_exploration"))

from exploration.pyelli_exploration.batch_grid_search import calculate_deviation_score, load_measured_spectrum, load_bestfit_spectrum
from exploration.pyelli_exploration.pyelli_grid_search import PyElliGridSearch

def test_single_file():
    input_dir = PROJECT_ROOT / "exploration" / "more_good_spectras" / "Corrected_Spectra"
    bestfit_dir = PROJECT_ROOT / "exploration" / "more_good_spectras" / "BestFit"
    file_name = "(Run)spectra_15-13-12-323.txt"
    file_path = input_dir / file_name
    bestfit_path = bestfit_dir / file_name.replace('.txt', '_BestFit.txt')
    
    materials_path = PROJECT_ROOT / "data" / "Materials"
    grid_search = PyElliGridSearch(materials_path)
    
    WL_MIN, WL_MAX = 600.0, 1120.0
    
    wavelengths, measured = load_measured_spectrum(file_path)
    wl_mask = (wavelengths >= WL_MIN) & (wavelengths <= WL_MAX)
    wl_fit, meas_fit = wavelengths[wl_mask], measured[wl_mask]
    
    # Run a quick search with EXACT app params to get the results_obj
    results = grid_search.run_grid_search(
        wl_fit, meas_fit,
        lipid_range=(9.0, 250.0, 5.0),
        aqueous_range=(800.0, 12000.0, 200.0),
        roughness_range=(600.0, 7000.0, 100.0),
        top_k=1, search_strategy='Dynamic Search', max_combinations=4000
    )
    
    if results:
        best = results[0]
        print(f"Fit Result: L={best.lipid_nm:.2f}, A={best.aqueous_nm:.1f}, R={best.mucus_nm:.0f}")
        
        bf_wl, bf_refl = load_bestfit_spectrum(bestfit_path)
        dev_metrics = calculate_deviation_score(best, grid_search, bf_wl, bf_refl, wl_fit, meas_fit)
        
        if dev_metrics:
            print(f"Dev Score: {dev_metrics['composite_dev']:.2f}%")
            print(f"MAPE: {dev_metrics['mape']:.2f}%")
            print(f"Peak Match Dev: {dev_metrics['peak_match_dev']:.2f}%")
            print(f"Alignment Dev: {dev_metrics['alignment_dev']:.2f}%")
        else:
            print("Failed to calculate dev metrics")
    else:
        print("No fit found")

if __name__ == "__main__":
    test_single_file()
