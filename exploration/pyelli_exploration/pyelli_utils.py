"""
PyElli Exploration Utilities

This module provides utilities for loading sample data and integrating with pyElli
for tear film interferometry analysis.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading Utilities
# =============================================================================

def load_measured_spectrum(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load measured spectrum from ADOM format file.
    
    Args:
        file_path: Path to the measured spectrum file
        
    Returns:
        Tuple of (wavelengths, reflectances) as numpy arrays
    """
    logger.info(f'📂 Loading measured spectrum from {file_path}')
    
    wavelengths = []
    reflectances = []
    data_started = False
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '>>>>>Begin Spectral Data<<<<<' in line:
                data_started = True
                continue
            if data_started and line:
                parts = line.split()
                if len(parts) >= 2:
                    wavelengths.append(float(parts[0]))
                    reflectances.append(float(parts[1]))
    
    logger.debug(f'📊 Loaded {len(wavelengths)} data points')
    return np.array(wavelengths), np.array(reflectances)


def load_bestfit_spectrum(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load BestFit theoretical spectrum file.
    
    Args:
        file_path: Path to the BestFit spectrum file
        
    Returns:
        Tuple of (wavelengths, reflectances) as numpy arrays
    """
    logger.info(f'📂 Loading BestFit spectrum from {file_path}')
    
    wavelengths = []
    reflectances = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip header line
            if line.startswith('BestFit') or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    wavelengths.append(float(parts[0]))
                    reflectances.append(float(parts[1]))
                except ValueError:
                    continue
    
    logger.debug(f'📊 Loaded {len(wavelengths)} BestFit data points')
    return np.array(wavelengths), np.array(reflectances)


def load_material_data(file_path: Path) -> pd.DataFrame:
    """
    Load material optical properties (n, k) from CSV file.
    
    Args:
        file_path: Path to material CSV file
        
    Returns:
        DataFrame with columns: wavelength_nm, n, k
    """
    logger.info(f'📂 Loading material data from {file_path}')
    
    # Read all lines and find where data starts (skip header rows)
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find first line with 3 comma-separated numeric values
    data_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 3:
            try:
                float(parts[0])
                float(parts[1])
                float(parts[2])
                data_lines.append(line)
            except ValueError:
                continue
    
    # Parse the data
    wavelengths = []
    n_values = []
    k_values = []
    
    for line in data_lines:
        parts = line.split(',')
        wavelengths.append(float(parts[0]))
        n_values.append(float(parts[1]))
        k_values.append(float(parts[2]))
    
    df = pd.DataFrame({
        'wavelength_nm': wavelengths,
        'n': n_values,
        'k': k_values
    })
    
    logger.debug(f'📊 Material data: {len(df)} wavelengths, λ range: {df.wavelength_nm.min():.1f}-{df.wavelength_nm.max():.1f} nm')
    
    return df


def get_sample_data_paths(base_path: Path) -> Dict[str, Dict[str, Path]]:
    """
    Discover all sample data files in the good_fit and bad_fit directories.
    
    Args:
        base_path: Base path to sample_data directory
        
    Returns:
        Dictionary with structure: {category: {sample_id: {measured: path, bestfit: path}}}
    """
    logger.info(f'🔍 Discovering sample data in {base_path}')
    
    samples = {'good_fit': {}, 'bad_fit': {}}
    
    for category in ['good_fit', 'bad_fit']:
        category_path = base_path / category
        if not category_path.exists():
            continue
            
        for sample_dir in sorted(category_path.iterdir()):
            if sample_dir.is_dir():
                sample_id = sample_dir.name
                samples[category][sample_id] = {
                    'measured': None,
                    'bestfit': None,
                    'dir': sample_dir
                }
                
                for file in sample_dir.iterdir():
                    if file.suffix == '.txt':
                        if '_BestFit' in file.name:
                            samples[category][sample_id]['bestfit'] = file
                        elif file.name.startswith('(Run)spectra_') and '_BestFit' not in file.name:
                            samples[category][sample_id]['measured'] = file
    
    logger.debug(f'✅ Found {len(samples["good_fit"])} good fits, {len(samples["bad_fit"])} bad fits')
    return samples


def get_available_materials(materials_path: Path) -> Dict[str, Path]:
    """
    Get available material files from the Materials directory.
    
    Args:
        materials_path: Path to Materials directory
        
    Returns:
        Dictionary mapping material names to file paths
    """
    logger.info(f'🔍 Discovering materials in {materials_path}')
    
    materials = {}
    for file in materials_path.glob('*.csv'):
        material_name = file.stem
        materials[material_name] = file
    
    logger.debug(f'✅ Found {len(materials)} material files')
    return materials


# =============================================================================
# PyElli Integration
# =============================================================================

def create_material_from_nk_data(
    wavelengths_nm: np.ndarray,
    n_values: np.ndarray,
    k_values: np.ndarray
):
    """
    Create a pyElli material from n,k dispersion data.
    
    Args:
        wavelengths_nm: Wavelength array in nanometers
        n_values: Real part of refractive index
        k_values: Imaginary part (extinction coefficient)
        
    Returns:
        pyElli DispersionMaterial object
    """
    try:
        import elli
        from elli.dispersions import TableIndex
        
        # pyElli expects wavelength in nm and complex refractive index
        nk_complex = n_values + 1j * k_values
        
        # Create tabulated dispersion
        dispersion = TableIndex(wavelengths_nm, nk_complex)
        
        return dispersion
    except ImportError:
        logger.error('❌ pyElli not installed. Run: pip install pyElli')
        raise


def build_tear_film_structure(
    lipid_thickness_nm: float,
    aqueous_thickness_nm: float,
    mucus_thickness_nm: float,
    wavelengths_nm: np.ndarray,
    materials: Dict[str, pd.DataFrame],
    angle_of_incidence: float = 0.0
):
    """
    Build a tear film multi-layer structure using pyElli.
    
    Structure: Air -> Lipid -> Aqueous -> Mucus -> Substrate (approximated)
    
    Args:
        lipid_thickness_nm: Lipid layer thickness in nm
        aqueous_thickness_nm: Aqueous layer thickness in nm  
        mucus_thickness_nm: Mucus layer thickness in nm
        wavelengths_nm: Array of wavelengths for calculation
        materials: Dict of material DataFrames with n,k data
        angle_of_incidence: Angle of incidence in degrees
        
    Returns:
        pyElli Structure object
    """
    try:
        import elli
        
        # Create dispersions from material data
        def interp_material(mat_df: pd.DataFrame, wavelengths: np.ndarray):
            """Interpolate material properties to target wavelengths."""
            n_interp = np.interp(wavelengths, mat_df['wavelength_nm'].values, mat_df['n'].values)
            k_interp = np.interp(wavelengths, mat_df['wavelength_nm'].values, mat_df['k'].values)
            return n_interp, k_interp
        
        # Get material properties at target wavelengths
        lipid_n, lipid_k = interp_material(materials['lipid'], wavelengths_nm)
        water_n, water_k = interp_material(materials['water'], wavelengths_nm)
        mucus_n, mucus_k = interp_material(materials['mucus'], wavelengths_nm)
        
        # Build the structure layer by layer
        # Note: pyElli uses a specific structure building approach
        
        # For now, return the interpolated data for manual calculation
        # Real pyElli integration will use elli.Structure
        
        return {
            'wavelengths': wavelengths_nm,
            'lipid': {'n': lipid_n, 'k': lipid_k, 'd': lipid_thickness_nm},
            'aqueous': {'n': water_n, 'k': water_k, 'd': aqueous_thickness_nm},
            'mucus': {'n': mucus_n, 'k': mucus_k, 'd': mucus_thickness_nm},
        }
        
    except ImportError:
        logger.error('❌ pyElli not installed')
        raise


def calculate_reflectance_fresnel(
    wavelengths_nm: np.ndarray,
    layers: list,  # List of (n, k, thickness_nm) tuples
    angle_deg: float = 0.0
) -> np.ndarray:
    """
    Calculate reflectance for a multi-layer thin film using Fresnel equations.
    
    This is a simplified implementation for demonstration. PyElli provides
    more sophisticated transfer matrix calculations.
    
    Args:
        wavelengths_nm: Array of wavelengths
        layers: List of (n, k, d) tuples for each layer
        angle_deg: Angle of incidence
        
    Returns:
        Reflectance array
    """
    # Convert angle to radians
    theta = np.deg2rad(angle_deg)
    
    # Start with air (n=1)
    n_air = 1.0
    
    # Simple two-interface approximation for demonstration
    # Real implementation uses transfer matrix method
    
    # For a 3-layer system (lipid/aqueous/mucus), this is complex
    # We'll use a simplified model here
    
    if len(layers) == 0:
        return np.ones_like(wavelengths_nm) * 0.04  # ~4% for air-glass
    
    # Get first layer properties
    n1, k1, d1 = layers[0]
    N1 = n1 + 1j * k1
    
    # Fresnel reflection at air-first layer interface
    r01 = (n_air - N1) / (n_air + N1)
    
    # Phase shift through first layer
    phase = 4 * np.pi * N1 * d1 / wavelengths_nm
    
    if len(layers) >= 2:
        n2, k2, d2 = layers[1]
        N2 = n2 + 1j * k2
        r12 = (N1 - N2) / (N1 + N2)
        
        # Simple two-beam interference
        r_total = r01 + r12 * np.exp(-1j * phase)
    else:
        r_total = r01
    
    # Reflectance is |r|^2
    reflectance = np.abs(r_total) ** 2
    
    return reflectance


# =============================================================================
# Spectrum Analysis Utilities
# =============================================================================

def calculate_residual(
    measured: np.ndarray,
    theoretical: np.ndarray
) -> float:
    """Calculate RMS residual between measured and theoretical spectra."""
    return np.sqrt(np.mean((measured - theoretical) ** 2))


def calculate_correlation(
    measured: np.ndarray,
    theoretical: np.ndarray
) -> float:
    """Calculate Pearson correlation coefficient."""
    return np.corrcoef(measured, theoretical)[0, 1]


def find_spectral_peaks(
    wavelengths: np.ndarray,
    reflectances: np.ndarray,
    prominence: float = 0.001
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find peaks in a reflectance spectrum.
    
    Args:
        wavelengths: Wavelength array
        reflectances: Reflectance array
        prominence: Minimum prominence for peak detection
        
    Returns:
        Tuple of (peak_wavelengths, peak_reflectances)
    """
    from scipy.signal import find_peaks
    
    peak_indices, properties = find_peaks(reflectances, prominence=prominence)
    
    return wavelengths[peak_indices], reflectances[peak_indices]


def interpolate_to_common_wavelengths(
    wavelengths1: np.ndarray,
    values1: np.ndarray,
    wavelengths2: np.ndarray,
    values2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate two spectra to common wavelength range.
    
    Returns:
        Tuple of (common_wavelengths, interpolated_values1, interpolated_values2)
    """
    # Find overlapping range
    min_wl = max(wavelengths1.min(), wavelengths2.min())
    max_wl = min(wavelengths1.max(), wavelengths2.max())
    
    # Create common wavelength grid
    common_wavelengths = np.linspace(min_wl, max_wl, 500)
    
    # Interpolate both spectra
    interp_values1 = np.interp(common_wavelengths, wavelengths1, values1)
    interp_values2 = np.interp(common_wavelengths, wavelengths2, values2)
    
    return common_wavelengths, interp_values1, interp_values2

