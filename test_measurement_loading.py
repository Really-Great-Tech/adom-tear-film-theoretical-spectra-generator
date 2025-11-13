#!/usr/bin/env python3
"""
Test script to verify measurement file loading
"""

import pathlib
import glob

import pytest

pd = pytest.importorskip("pandas")

def test_measurement_loading():
    """Test loading measurement files"""
    
    # Configuration
    measurements_dir = pathlib.Path("data/measurements")
    file_pattern = "*.txt"
    header_rows = 4
    wl_col = 0
    refl_col = 1
    
    print(f"Testing measurement loading from: {measurements_dir.resolve()}")
    print(f"Directory exists: {measurements_dir.exists()}")
    
    if not measurements_dir.exists():
        print("ERROR: Measurements directory not found!")
        return
    
    # Find all matching files
    pattern_path = measurements_dir / file_pattern
    file_paths = glob.glob(str(pattern_path))
    
    print(f"Found {len(file_paths)} measurement files")
    
    if not file_paths:
        print("ERROR: No measurement files found!")
        return
    
    # Test loading first few files
    test_files = file_paths[:3]
    
    for file_path in test_files:
        try:
            file_name = pathlib.Path(file_path).stem
            print(f"\nTesting file: {file_name}")
            
            # Load with space separator and skip headers
            df = pd.read_csv(file_path, sep=r'\s+', skiprows=header_rows, header=None)
            
            print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
            
            if df.shape[1] > max(wl_col, refl_col):
                wavelengths = df.iloc[:, wl_col].values
                reflectances = df.iloc[:, refl_col].values
                
                print(f"  Wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
                print(f"  Reflectance range: {reflectances.min():.6f} - {reflectances.max():.6f}")
                
                # Check for NaN values
                nan_count = pd.isna(wavelengths).sum() + pd.isna(reflectances).sum()
                print(f"  NaN values: {nan_count}")
                
                print("  ✓ File loaded successfully")
            else:
                print(f"  ERROR: Not enough columns. Expected at least {max(wl_col, refl_col)+1}, got {df.shape[1]}")
                
        except Exception as e:
            print(f"  ERROR loading {file_path}: {e}")
    
    print(f"\nTest complete. Found {len(file_paths)} total measurement files.")

if __name__ == "__main__":
    test_measurement_loading()
