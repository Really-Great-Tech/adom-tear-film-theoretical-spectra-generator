"""Test script to demonstrate the automated quality metrics implementation."""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analysis.quality_metrics import (
    assess_spectrum_quality,
    calculate_snr,
    check_peak_quality,
    check_signal_integrity,
    check_spectral_completeness,
)


def create_test_spectrum(quality_level="good"):
    """Create synthetic test spectra with different quality levels."""
    wavelengths = np.linspace(600, 1120, 521)  # 1nm spacing, 520nm span

    if quality_level == "excellent":
        # High SNR, clear fringes, no issues
        base = 0.5
        amplitude = 0.3
        noise_level = 0.001
        num_fringes = 8
    elif quality_level == "good":
        # Good SNR, clear fringes
        base = 0.5
        amplitude = 0.2
        noise_level = 0.01
        num_fringes = 6
    elif quality_level == "marginal":
        # Low SNR, fewer fringes
        base = 0.5
        amplitude = 0.1
        noise_level = 0.05
        num_fringes = 4
    else:  # "reject"
        # Very low SNR, almost no fringes
        base = 0.5
        amplitude = 0.02
        noise_level = 0.1
        num_fringes = 2

    # Create interference pattern
    frequency = num_fringes * 2 * np.pi / (wavelengths[-1] - wavelengths[0])
    signal = base + amplitude * np.sin(frequency * wavelengths)

    # Add noise
    noise = np.random.normal(0, noise_level, len(wavelengths))
    reflectance = signal + noise

    # Ensure valid range
    reflectance = np.clip(reflectance, 0, 1)

    return wavelengths, reflectance


def print_quality_report(report, title):
    """Print a formatted quality report."""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")
    print(f"Overall Quality: {report.overall_quality}")
    print(f"Passed All Checks: {report.passed_all_checks}")
    print(f"\nMetric Results:")
    print(f"{'-' * 70}")

    for name, result in report.metrics.items():
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"\n{name.upper()}: {status}")
        print(f"  Value: {result.value:.4f} (threshold: {result.threshold:.4f})")
        print(f"  Details:")
        for key, value in result.details.items():
            if isinstance(value, (int, float)):
                print(f"    - {key}: {value:.4f}")
            else:
                print(f"    - {key}: {value}")

    if report.warnings:
        print(f"\n⚠ Warnings:")
        for warning in report.warnings:
            print(f"  - {warning}")

    if report.failures:
        print(f"\n✗ Failures:")
        for failure in report.failures:
            print(f"  - {failure}")

    print(f"{'=' * 70}\n")


def test_individual_metrics():
    """Test individual quality metrics."""
    print("\n" + "=" * 70)
    print("TESTING INDIVIDUAL QUALITY METRICS")
    print("=" * 70)

    wavelengths, reflectance = create_test_spectrum("good")

    # Test SNR
    print("\n1. SNR Calculation:")
    snr_result = calculate_snr(wavelengths, reflectance)
    print(f"   SNR: {snr_result.value:.2f}")
    print(f"   Quality Level: {snr_result.details['quality_level']}")
    print(f"   Passed: {snr_result.passed}")

    # Test Peak Quality
    print("\n2. Peak Quality Check:")
    peak_result = check_peak_quality(wavelengths, reflectance)
    print(f"   Peak Count: {int(peak_result.details['peak_count'])}")
    print(f"   Prominence CV: {peak_result.details['prominence_cv']:.3f}")
    print(f"   Spacing CV: {peak_result.details['spacing_cv']:.3f}")
    print(f"   Passed: {peak_result.passed}")

    # Test Signal Integrity
    print("\n3. Signal Integrity Check:")
    integrity_result = check_signal_integrity(reflectance)
    print(f"   Dynamic Range: {integrity_result.details['dynamic_range']:.4f}")
    print(
        f"   Saturation Fraction: {integrity_result.details['saturation_fraction']:.4f}"
    )
    print(
        f"   Baseline Drift: {integrity_result.details['baseline_drift_percent']:.2f}%"
    )
    print(f"   Negative Count: {int(integrity_result.details['negative_count'])}")
    print(f"   Passed: {integrity_result.passed}")

    # Test Spectral Completeness
    print("\n4. Spectral Completeness Check:")
    completeness_result = check_spectral_completeness(wavelengths)
    print(
        f"   Wavelength Span: {completeness_result.details['wavelength_span_nm']:.1f} nm"
    )
    print(
        f"   Point Density: {completeness_result.details['point_density']:.2f} pts/nm"
    )
    print(f"   Max Gap: {completeness_result.details['max_gap_nm']:.2f} nm")
    print(f"   Passed: {completeness_result.passed}")


def test_comprehensive_assessment():
    """Test comprehensive quality assessment for different quality levels."""
    print("\n" + "=" * 70)
    print("TESTING COMPREHENSIVE QUALITY ASSESSMENT")
    print("=" * 70)

    quality_levels = ["excellent", "good", "marginal", "reject"]

    for level in quality_levels:
        wavelengths, reflectance = create_test_spectrum(level)
        report = assess_spectrum_quality(wavelengths, reflectance)
        print_quality_report(report, f"Test Spectrum: {level.upper()}")


def test_with_fit_quality():
    """Test quality assessment including fit residual quality."""
    print("\n" + "=" * 70)
    print("TESTING WITH FIT QUALITY METRICS")
    print("=" * 70)

    wavelengths, measured = create_test_spectrum("good")

    # Create a "fitted" spectrum with small residuals
    fitted = measured + np.random.normal(0, 0.005, len(measured))

    report = assess_spectrum_quality(
        wavelengths,
        measured,
        fitted_spectrum=fitted,
    )

    print_quality_report(report, "Spectrum with Fit Quality Assessment")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("AUTOMATED QUALITY METRICS - PART C IMPLEMENTATION TEST")
    print("=" * 70)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Run tests
    test_individual_metrics()
    test_comprehensive_assessment()
    test_with_fit_quality()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
