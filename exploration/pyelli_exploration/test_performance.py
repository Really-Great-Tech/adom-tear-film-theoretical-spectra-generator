"""
Performance Benchmark Test for PyElli Grid Search

This script benchmarks the grid search performance to validate the
parallelization optimizations:
1. BLAS thread limiting (OMP_NUM_THREADS=1, etc.)
2. Worker initializer for shared data (avoids per-task serialization)
3. Pre-computed material interpolation

Run with: python -m exploration.pyelli_exploration.test_performance

Expected improvements:
- On multi-core Linux (32+ cores): ~10-50x faster
- On Windows: ~2-5x faster (spawn overhead still exists but reduced)
- On Apple Silicon: Minimal change (was already fast)
"""

import logging
import os
import sys
import time
from pathlib import Path

# Ensure BLAS thread limiting is set before any numpy import
# This is critical for the test to be valid
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import numpy as np

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exploration.pyelli_exploration.pyelli_grid_search import (
    PyElliGridSearch,
    _worker_initializer,
    _evaluate_combination_fast,
)
from exploration.pyelli_exploration.pyelli_utils import load_measured_spectrum


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def get_sample_spectrum(project_root: Path) -> tuple:
    """Load a sample spectrum for testing."""
    # Try exploration/sample_data first, then fallback to project root
    sample_data_paths = [
        project_root / 'exploration' / 'sample_data' / 'good_fit',
        project_root / 'sample_data' / 'good_fit',
    ]
    
    for sample_data_path in sample_data_paths:
        if sample_data_path.exists():
            # Find first available sample
            for sample_dir in sorted(sample_data_path.iterdir()):
                if sample_dir.is_dir():
                    for file in sample_dir.iterdir():
                        if file.suffix == '.txt' and '_BestFit' not in file.name:
                            logger.info(f'📂 Using sample: {file}')
                            return load_measured_spectrum(file)
    
    raise FileNotFoundError(f'No sample spectrum found. Searched: {sample_data_paths}')


def benchmark_grid_search(
    grid_search: PyElliGridSearch,
    wavelengths: np.ndarray,
    measured: np.ndarray,
    num_combinations: int = 1000,
) -> dict:
    """
    Benchmark the grid search with a fixed number of combinations.
    
    Returns:
        Dictionary with benchmark results
    """
    # Calculate grid dimensions to achieve target combinations
    # target = lipid_count * aqueous_count * roughness_count
    # Use approximately equal dimensions
    dim_size = int(np.ceil(num_combinations ** (1/3)))
    
    # Create parameter arrays
    lipid_values = np.linspace(30, 100, dim_size)  # nm
    aqueous_values = np.linspace(2000, 5000, dim_size)  # nm
    roughness_values = np.linspace(1000, 3000, dim_size)  # Angstroms
    
    actual_combinations = len(lipid_values) * len(aqueous_values) * len(roughness_values)
    
    logger.info(f'📊 Benchmark Parameters:')
    logger.info(f'   - Lipid: {len(lipid_values)} values ({lipid_values[0]:.1f} to {lipid_values[-1]:.1f} nm)')
    logger.info(f'   - Aqueous: {len(aqueous_values)} values ({aqueous_values[0]:.1f} to {aqueous_values[-1]:.1f} nm)')
    logger.info(f'   - Roughness: {len(roughness_values)} values ({roughness_values[0]:.1f} to {roughness_values[-1]:.1f} Å)')
    logger.info(f'   - Total combinations: {actual_combinations:,}')
    logger.info(f'   - CPU cores: {os.cpu_count()}')
    
    # Run the grid search using the internal method
    start_time = time.perf_counter()
    
    results = grid_search._evaluate_parameter_grid(
        wavelengths=wavelengths,
        measured=measured,
        lipid_values=lipid_values,
        aqueous_values=aqueous_values,
        roughness_values=roughness_values,
        top_k=10,
        enable_roughness=True,
        min_correlation_filter=0.5,  # Lower threshold for benchmark
    )
    
    elapsed = time.perf_counter() - start_time
    
    # Calculate metrics
    combinations_per_second = actual_combinations / elapsed
    
    return {
        'combinations': actual_combinations,
        'elapsed_seconds': elapsed,
        'combinations_per_second': combinations_per_second,
        'results_count': len(results),
        'cpu_cores': os.cpu_count(),
    }


def run_benchmark():
    """Run the performance benchmark."""
    logger.info('=' * 80)
    logger.info('🚀 PyElli Grid Search Performance Benchmark')
    logger.info('=' * 80)
    
    # Check BLAS thread settings
    logger.info('')
    logger.info('🔧 BLAS Thread Settings:')
    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        logger.info(f'   - {var}: {os.environ.get(var, "not set")}')
    
    # Load sample data
    logger.info('')
    logger.info('📂 Loading sample data...')
    wavelengths, measured = get_sample_spectrum(PROJECT_ROOT)
    logger.info(f'   - Wavelengths: {len(wavelengths)} points ({wavelengths.min():.1f} to {wavelengths.max():.1f} nm)')
    
    # Initialize grid search
    logger.info('')
    logger.info('🔧 Initializing grid search...')
    # Materials are in data/Materials/ not at project root
    materials_path = PROJECT_ROOT / 'data' / 'Materials'
    if not materials_path.exists():
        # Fallback to project root Materials if data/Materials doesn't exist
        materials_path = PROJECT_ROOT / 'Materials'
    grid_search = PyElliGridSearch(materials_path)
    
    # Run benchmark with different sizes
    benchmark_sizes = [500, 1000, 2000]
    
    logger.info('')
    logger.info('=' * 80)
    logger.info('📊 BENCHMARK RESULTS')
    logger.info('=' * 80)
    
    results = []
    for size in benchmark_sizes:
        logger.info('')
        logger.info(f'🔄 Running benchmark with ~{size} combinations...')
        logger.info('-' * 40)
        
        try:
            result = benchmark_grid_search(grid_search, wavelengths, measured, size)
            results.append(result)
            
            logger.info('')
            logger.info(f'✅ Benchmark Complete:')
            logger.info(f'   - Combinations evaluated: {result["combinations"]:,}')
            logger.info(f'   - Time elapsed: {result["elapsed_seconds"]:.2f} seconds')
            logger.info(f'   - Throughput: {result["combinations_per_second"]:.1f} combinations/second')
            logger.info(f'   - Results found: {result["results_count"]:,}')
            
        except Exception as e:
            logger.error(f'❌ Benchmark failed: {e}')
            import traceback
            traceback.print_exc()
    
    # Summary
    if results:
        logger.info('')
        logger.info('=' * 80)
        logger.info('📈 SUMMARY')
        logger.info('=' * 80)
        avg_throughput = np.mean([r['combinations_per_second'] for r in results])
        logger.info(f'   - Average throughput: {avg_throughput:.1f} combinations/second')
        logger.info(f'   - CPU cores used: {results[0]["cpu_cores"]}')
        logger.info(f'   - Effective parallelism: {avg_throughput / (results[0]["cpu_cores"] or 1):.1f} comb/s per core')
        logger.info('')
        logger.info('💡 Performance Tips:')
        logger.info('   - On 32-core Linux: expect 500-2000 comb/s')
        logger.info('   - On 8-core Mac M1/M4: expect 200-800 comb/s')
        logger.info('   - On 4-8 core Windows: expect 50-200 comb/s')
        logger.info('')
        logger.info('   If throughput is much lower, check:')
        logger.info('   1. BLAS thread settings (should be 1)')
        logger.info('   2. System memory usage (avoid swapping)')
        logger.info('   3. Background processes competing for CPU')
    
    return results


if __name__ == '__main__':
    run_benchmark()
