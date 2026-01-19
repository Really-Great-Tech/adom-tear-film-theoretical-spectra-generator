"""Test improved scoring against ground truth spectra in 'new spectra' folder."""
import sys
import pathlib
import numpy as np
import pandas as pd
import yaml
import logging
from typing import Dict, Any, List, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add src to path
src_path = pathlib.Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from analysis.measurement_utils import (
    PreparedMeasurement,
    PreparedTheoreticalSpectrum,
    prepare_measurement,
    prepare_theoretical_spectrum,
)
from analysis.metrics import (
    score_spectrum,
    measurement_quality_score,
)

def load_config() -> Dict[str, Any]:
    """Load config.yaml."""
    config_path = pathlib.Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_measurement_spectrum(file_path: pathlib.Path, config: Dict[str, Any]) -> pd.DataFrame:
    """Load a measurement spectrum file."""
    meas_config = config.get('measurements', {})
    wavelength_col = meas_config.get('wavelength_column', 0)
    reflectance_col = meas_config.get('reflectance_column', 1)
    header_rows = meas_config.get('header_rows', 4)
    
    # Read as fixed-width or space-separated
    df = pd.read_csv(
        file_path,
        sep=r'\s+',
        skiprows=header_rows,
        header=None,
        usecols=[wavelength_col, reflectance_col],
        names=['wavelength', 'reflectance'],
    )
    return df

def parse_ground_truth_from_folder(folder_name: str) -> Tuple[float, float]:
    """Parse ground truth lipid,aqueous from folder name like '21,3745'."""
    parts = folder_name.split(',')
    lipid = float(parts[0])
    aqueous = float(parts[1])
    return lipid, aqueous

def generate_candidate_grid(
    gt_lipid: float,
    gt_aqueous: float,
    config: Dict[str, Any],
    quick_mode: bool = True,
) -> List[Tuple[float, float, float]]:
    """Generate a grid of candidate parameters around ground truth."""
    # Get roughness range from config
    rough_cfg = config.get('parameters', {}).get('roughness', {})
    rough_min = rough_cfg.get('min', 600)
    rough_max = rough_cfg.get('max', 2750)
    
    candidates = []
    
    if quick_mode:
        # Faster test with tighter grid around GT
        lipid_values = np.arange(max(9, gt_lipid - 30), min(250, gt_lipid + 30) + 1, 5)
        aqueous_values = np.arange(max(800, gt_aqueous - 500), min(12000, gt_aqueous + 500) + 1, 50)
        roughness_values = np.arange(rough_min, rough_max + 1, 400)
    else:
        # Full grid search
        lipid_values = np.arange(max(9, gt_lipid - 100), min(250, gt_lipid + 100) + 1, 10)
        aqueous_values = np.arange(max(800, gt_aqueous - 2000), min(12000, gt_aqueous + 2000) + 1, 100)
        roughness_values = np.arange(rough_min, rough_max + 1, 200)
    
    for lipid in lipid_values:
        for aqueous in aqueous_values:
            for roughness in roughness_values:
                candidates.append((float(lipid), float(aqueous), float(roughness)))
    
    return candidates

def run_scoring_test(
    measurement_df: pd.DataFrame,
    candidates: List[Tuple[float, float, float]],
    config: Dict[str, Any],
    calc_spectrum_func,
    theo_wavelengths: np.ndarray,
) -> pd.DataFrame:
    """Run scoring on all candidates and return sorted results."""
    from scipy import interpolate
    
    analysis_cfg = config.get('analysis', {})
    metrics_cfg = analysis_cfg.get('metrics', {})
    
    # Prepare measurement features (takes DataFrame, not arrays)
    measurement_features = prepare_measurement(
        measurement_df,
        analysis_cfg,
    )
    
    # Get wavelengths for theoretical spectra (filtered to analysis range)
    wavelengths = measurement_features.wavelengths
    
    # Quality check
    quality_cfg = analysis_cfg.get('quality_gates', {})
    quality_result, _ = measurement_quality_score(
        measurement_features,
        min_peaks=quality_cfg.get('min_peaks', 3),
        min_signal_amplitude=quality_cfg.get('min_signal_amplitude', 0.02),
        min_wavelength_span_nm=quality_cfg.get('min_wavelength_span_nm', 150.0),
    )
    
    records = []
    debug_count = 0
    skipped_none = 0
    skipped_std = 0
    
    # Debug: Test first candidate before loop
    test_lipid, test_aqueous, test_rough = candidates[0]
    logger.info(f'  🧪 Testing first candidate: L={test_lipid}, A={test_aqueous}, R={test_rough}')
    try:
        test_spectrum = calc_spectrum_func(test_lipid, test_aqueous, test_rough)
        if test_spectrum is not None:
            logger.info(f'     Test spectrum: len={len(test_spectrum)}, min={np.min(test_spectrum):.4f}, max={np.max(test_spectrum):.4f}, std={np.std(test_spectrum):.6f}')
        else:
            logger.info(f'     Test spectrum: None')
    except Exception as e:
        logger.error(f'     Test spectrum FAILED: {e}')
    
    for lipid, aqueous, roughness in candidates:
        try:
            # Generate theoretical spectrum
            spectrum = calc_spectrum_func(lipid, aqueous, roughness)
            
            if spectrum is None or len(spectrum) == 0 or np.all(spectrum == 0):
                skipped_none += 1
                continue
            if np.std(spectrum) < 1e-6:
                skipped_std += 1
                continue
            
            # Interpolate theoretical spectrum to measurement wavelengths
            interp_func = interpolate.interp1d(theo_wavelengths, spectrum, kind='linear', fill_value='extrapolate')
            spectrum_interp = interp_func(wavelengths)
            
            theoretical = prepare_theoretical_spectrum(
                wavelengths,
                spectrum_interp,
                measurement_features,
                analysis_cfg,
            )
            
            # Score
            score_result = score_spectrum(
                measurement_features,
                theoretical,
                metrics_cfg,
                lipid_nm=lipid,
                aqueous_nm=aqueous,
                roughness_A=roughness,
                measurement_quality=quality_result,
            )
            
            record = {
                'lipid_nm': lipid,
                'aqueous_nm': aqueous,
                'roughness_A': roughness,
                'score_composite': score_result.scores.get('composite', 0),
                'score_correlation': score_result.scores.get('correlation', 0),
                'score_residual': score_result.scores.get('residual', 0),
                'score_peak_count': score_result.scores.get('peak_count', 0),
                'score_peak_delta': score_result.scores.get('peak_delta', 0),
                'score_amplitude': score_result.scores.get('amplitude', 0),
                'correlation': score_result.diagnostics.get('correlation', {}).get('correlation', 0),
                'rmse': score_result.diagnostics.get('residual', {}).get('rmse', 0),
                'matched_peaks': score_result.diagnostics.get('peak_count', {}).get('matched_peaks', 0),
            }
            records.append(record)
        except Exception as e:
            # Skip failed candidates silently
            continue
    
    logger.info(f'  📊 Scoring complete: {len(records)} valid, {skipped_none} skipped (None/empty), {skipped_std} skipped (low std)')
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    df = df.sort_values('score_composite', ascending=False).reset_index(drop=True)
    return df

def main():
    logger.info('='*80)
    logger.info('GROUND TRUTH SCORING TEST')
    logger.info('='*80)
    
    # Load config
    config = load_config()
    
    # Find new spectra folder
    new_spectra_dir = pathlib.Path(__file__).parent / 'new spectra'
    if not new_spectra_dir.exists():
        logger.error(f'New spectra folder not found: {new_spectra_dir}')
        return
    
    # Load wavelengths first (needed for both DLL and fallback)
    wavelengths_dir = pathlib.Path(__file__).parent / config['paths']['wavelengths_dir']
    wavelengths_file = config['wavelengths']['file']
    wavelengths_path = wavelengths_dir / wavelengths_file
    theo_wavelengths = np.loadtxt(wavelengths_path, delimiter=',')
    logger.info(f'📊 Loaded {len(theo_wavelengths)} theoretical wavelengths')
    
    # Setup DLL for spectrum generation
    try:
        from tear_film_generator import make_calc_refl, build_stack_xml
        import xml.etree.ElementTree as ET
        import os
        
        dll_path = pathlib.Path(__file__).parent / config['paths']['dll']
        algorithm_path = pathlib.Path(__file__).parent / config['paths']['algorithm']
        materials_dir = pathlib.Path(__file__).parent / config['paths']['materials']
        config_xml = pathlib.Path(__file__).parent / config['paths']['configuration']
        stack_xml_path = pathlib.Path(__file__).parent / config['paths']['stack']
        
        logger.info(f'📁 Loading DLL from: {dll_path}')
        calc_spectrum = make_calc_refl(dll_path, algorithm_path, materials_dir, config_xml)
        
        # Load base stack XML for building temp stacks
        base_stack = ET.parse(stack_xml_path).getroot()
        
        def generate_spectrum(lipid: float, aqueous: float, roughness: float) -> np.ndarray:
            """Generate theoretical spectrum for given parameters."""
            # Build temporary stack XML with parameters
            temp_xml = build_stack_xml(
                lipid, aqueous, 
                mucus_rough=roughness,
                materials_dir=materials_dir,
                base_stack=base_stack
            )
            try:
                result = calc_spectrum(theo_wavelengths.tolist(), temp_xml)
                return np.array(result)
            finally:
                # Clean up temp file
                if os.path.exists(temp_xml):
                    os.remove(temp_xml)
        
    except Exception as e:
        logger.error(f'❌ Failed to setup DLL: {e}')
        import traceback
        traceback.print_exc()
        logger.info('Falling back to mock spectrum generator for testing...')
        
        def generate_spectrum(lipid: float, aqueous: float, roughness: float) -> np.ndarray:
            """Mock spectrum generator for testing."""
            # This is just for testing the scoring logic
            return np.sin(np.linspace(0, 10, len(theo_wavelengths)) * lipid / 100) * 0.1 + 0.5
    
    # Get all ground truth folders
    gt_folders = [
        f for f in new_spectra_dir.iterdir() 
        if f.is_dir() and ',' in f.name
    ]
    
    logger.info(f'\n📂 Found {len(gt_folders)} ground truth folders\n')
    
    results_summary = []
    
    for gt_folder in sorted(gt_folders):
        gt_lipid, gt_aqueous = parse_ground_truth_from_folder(gt_folder.name)
        logger.info(f'\n{"="*60}')
        logger.info(f'📊 Testing: {gt_folder.name} (GT: lipid={gt_lipid}, aqueous={gt_aqueous})')
        logger.info(f'{"="*60}')
        
        # Find measurement file (not BestFit)
        measurement_files = [
            f for f in gt_folder.iterdir()
            if f.is_file() and f.suffix == '.txt' and '_BestFit' not in f.name
        ]
        
        if not measurement_files:
            logger.warning(f'  ⚠️ No measurement file found in {gt_folder.name}')
            continue
        
        measurement_file = measurement_files[0]
        logger.info(f'  📄 Loading: {measurement_file.name}')
        
        try:
            # Load measurement
            meas_df = load_measurement_spectrum(measurement_file, config)
            logger.info(f'  📈 Loaded {len(meas_df)} data points')
            
            # Generate candidates around ground truth
            candidates = generate_candidate_grid(gt_lipid, gt_aqueous, config)
            logger.info(f'  🔍 Evaluating {len(candidates)} candidates...')
            
            # Run scoring
            results_df = run_scoring_test(meas_df, candidates, config, generate_spectrum, theo_wavelengths)
            
            if results_df.empty:
                logger.warning(f'  ⚠️ No valid results for {gt_folder.name}')
                continue
            
            # Find ground truth rank
            gt_tolerance_lipid = 5  # Within 5nm of GT
            gt_tolerance_aqueous = 100  # Within 100nm of GT
            
            gt_mask = (
                (np.abs(results_df['lipid_nm'] - gt_lipid) <= gt_tolerance_lipid) &
                (np.abs(results_df['aqueous_nm'] - gt_aqueous) <= gt_tolerance_aqueous)
            )
            
            if gt_mask.any():
                gt_rank = results_df[gt_mask].index[0]
                gt_row = results_df[gt_mask].iloc[0]
                logger.info(f'\n  ✅ Ground truth found at RANK {gt_rank}')
                logger.info(f'     L={gt_row["lipid_nm"]:.0f}, A={gt_row["aqueous_nm"]:.0f}, R={gt_row["roughness_A"]:.0f}')
                logger.info(f'     composite={gt_row["score_composite"]:.4f}, corr={gt_row["correlation"]:.4f}, rmse={gt_row["rmse"]:.6f}')
            else:
                gt_rank = -1
                logger.warning(f'\n  ⚠️ Ground truth NOT in candidate set!')
            
            # Show top 5 results
            logger.info(f'\n  🏆 Top 5 Results:')
            for i, row in results_df.head(5).iterrows():
                is_gt = ' ⭐ GT' if (
                    abs(row['lipid_nm'] - gt_lipid) <= gt_tolerance_lipid and
                    abs(row['aqueous_nm'] - gt_aqueous) <= gt_tolerance_aqueous
                ) else ''
                logger.info(f'     {i}: L={row["lipid_nm"]:.0f}, A={row["aqueous_nm"]:.0f}, R={row["roughness_A"]:.0f} | '
                           f'comp={row["score_composite"]:.4f} | corr={row["correlation"]:.4f} | rmse={row["rmse"]:.6f}{is_gt}')
            
            # Best result comparison
            best = results_df.iloc[0]
            lipid_error = abs(best['lipid_nm'] - gt_lipid)
            aqueous_error = abs(best['aqueous_nm'] - gt_aqueous)
            
            results_summary.append({
                'folder': gt_folder.name,
                'gt_lipid': gt_lipid,
                'gt_aqueous': gt_aqueous,
                'best_lipid': best['lipid_nm'],
                'best_aqueous': best['aqueous_nm'],
                'lipid_error': lipid_error,
                'aqueous_error': aqueous_error,
                'gt_rank': gt_rank,
                'best_score': best['score_composite'],
                'best_correlation': best['correlation'],
            })
            
        except Exception as e:
            logger.error(f'  ❌ Error processing {gt_folder.name}: {e}')
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    logger.info(f'\n\n{"="*80}')
    logger.info('SUMMARY')
    logger.info('='*80)
    
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        logger.info(f'\n{"Folder":<15} {"GT(L,A)":<15} {"Best(L,A)":<15} {"L_err":<8} {"A_err":<8} {"GT_Rank":<8} {"Score":<8}')
        logger.info('-'*80)
        
        for _, row in summary_df.iterrows():
            gt_str = f'{row["gt_lipid"]:.0f},{row["gt_aqueous"]:.0f}'
            best_str = f'{row["best_lipid"]:.0f},{row["best_aqueous"]:.0f}'
            gt_rank_str = str(int(row['gt_rank'])) if row['gt_rank'] >= 0 else 'N/A'
            logger.info(f'{row["folder"]:<15} {gt_str:<15} {best_str:<15} {row["lipid_error"]:<8.0f} {row["aqueous_error"]:<8.0f} {gt_rank_str:<8} {row["best_score"]:<8.4f}')
        
        # Statistics
        logger.info(f'\n📊 Statistics:')
        logger.info(f'   Mean lipid error: {summary_df["lipid_error"].mean():.1f} nm')
        logger.info(f'   Mean aqueous error: {summary_df["aqueous_error"].mean():.1f} nm')
        
        exact_matches = len(summary_df[(summary_df['lipid_error'] <= 5) & (summary_df['aqueous_error'] <= 100)])
        logger.info(f'   Exact matches (L≤5, A≤100): {exact_matches}/{len(summary_df)}')
        
        top5_rank = len(summary_df[(summary_df['gt_rank'] >= 0) & (summary_df['gt_rank'] < 5)])
        logger.info(f'   GT in top 5: {top5_rank}/{len(summary_df)}')
    else:
        logger.warning('No results to summarize')

if __name__ == '__main__':
    main()
