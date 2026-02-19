"""Automated Quality Metrics for Single Spectrum Assessment (Part C).

These metrics assess the quality of a single measured spectrum before/during fitting,
determining whether the data is suitable for reliable thickness extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

# Analysis Constants as per Shlomo feedback
ANALYSIS_WAVELENGTH_MIN = 600.0
ANALYSIS_WAVELENGTH_MAX = 1200.0

# Quality tier thresholds: GOOD (SNR ≥ 20), noisy band 5–20 (NOISY_FITTABLE / NOISY_UNFITTABLE), BAD (<5 or <2 peaks or saturated/flat)
SNR_GOOD_MIN = 20.0
SNR_NOISY_LOW = 5.0
MIN_PEAKS_FOR_QUALITY = 2
SMOOTHING_IMPROVEMENT_MIN = 0.1
NOISY_FITTABLE_NRMSE_PERCENT_MAX = 10.0
GOOD_NRMSE_PERCENT_MAX = 5.0
GOOD_R2_MIN = 0.90
GOOD_CHI2_MAX = 1.2


@dataclass
class QualityMetricResult:
    """Result from a single quality metric check."""

    passed: bool
    value: float
    threshold: float
    metric_name: str
    status: str  # "Excellent", "Good", "Marginal", "Reject"
    details: Dict[str, any]


@dataclass
class SpectrumQualityReport:
    """Comprehensive quality assessment report for a spectrum."""

    overall_quality: str  # "GOOD" | "NOISY_FITTABLE" | "NOISY_UNFITTABLE" | "BAD"
    passed_all_checks: bool
    metrics: Dict[str, QualityMetricResult]
    warnings: List[str]
    failures: List[str]

    def get_summary(self) -> Dict[str, any]:
        """Get a summary dictionary of all quality metrics."""
        summary = {
            "overall_quality": self.overall_quality,
            "passed_all_checks": self.passed_all_checks,
            "num_warnings": len(self.warnings),
            "num_failures": len(self.failures),
        }

        for name, result in self.metrics.items():
            summary[f"{name}_passed"] = result.passed
            summary[f"{name}_value"] = result.value

        return summary


def calculate_snr(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    *,
    use_detrended: bool = True,
    cutoff_frequency: float = 0.008,
    filter_order: int = 3,
    signal_center_fraction: float = 0.6,
    baseline_edge_fraction: float = 0.1,
    band_nm: Optional[Tuple[float, float]] = None,
    use_smooth_residual: bool = True,
    boxcar_width_nm: float = 11.0,
    boxcar_repeats: int = 2,
) -> QualityMetricResult:
    """Calculate Signal-to-Noise Ratio (SNR).

    Metric 1: Signal-to-Noise Ratio (SNR)

    Default (use_smooth_residual=True): Detrend → boxcar smooth (11 nm, Shlomo) → residual
    = detrended - smooth. Signal = ptp(smooth), noise = std(residual). More consistent
    for noisy spectra. Set boxcar_width_nm to 17 for previous optimum.

    Legacy (use_smooth_residual=False): Signal = ptp(detrended), noise = first-diff
    over band (no edge baseline).

    Args:
        wavelengths: Wavelength array in nm
        reflectance: Reflectance values
        use_detrended: Whether to detrend before SNR calculation (default True)
        cutoff_frequency: Cutoff frequency for detrending (default 0.008)
        filter_order: Filter order for detrending (default 3)
        signal_center_fraction: Unused (kept for API compatibility)
        baseline_edge_fraction: Unused (kept for API compatibility)
        band_nm: (min, max) wavelength band in nm (default 600–1120)
        use_smooth_residual: Use smooth/residual separation with boxcar (default True)
        boxcar_width_nm: Boxcar width in nm for smooth (default 11, Shlomo)
        boxcar_repeats: Number of boxcar passes (default 2)

    Returns:
        QualityMetricResult with SNR assessment

    Quality Thresholds (Empirically Derived for Detrended Interference Spectra):
        SNR >= 2.5: Excellent
        SNR 1.5-2.5: Good
        SNR 1.0-1.5: Marginal
        SNR < 1.0: Reject
    """
    if len(reflectance) < 2:
        return QualityMetricResult(
            passed=False,
            value=0.0,
            threshold=1.0,
            metric_name="SNR",
            status="Reject",
            details={},
        )

    from analysis.snr import compute_global_snr

    band = band_nm or (ANALYSIS_WAVELENGTH_MIN, 1120.0)
    g = compute_global_snr(
        wavelengths,
        reflectance,
        band_nm=band,
        detrend_cutoff=cutoff_frequency,
        detrend_order=filter_order,
        use_smooth_residual=use_smooth_residual,
        boxcar_width_nm=boxcar_width_nm,
        boxcar_repeats=boxcar_repeats,
    )
    if not use_detrended:
        # Raw: crop to band, signal = ptp, noise = first-diff over band (no detrend)
        wl, refl = np.asarray(wavelengths), np.asarray(reflectance)
        mask = (wl >= band[0]) & (wl <= band[1])
        wl, refl = wl[mask], refl[mask]
        if len(wl) < 3:
            return QualityMetricResult(
                passed=False, value=0.0, threshold=1.0, metric_name="SNR",
                status="Reject", details={"method": "Raw (band cropped)"},
            )
        sig_ptp = float(np.ptp(refl))
        d = np.diff(refl.astype(float))
        noise_std = float(np.std(d, ddof=1)) / np.sqrt(2.0) if len(d) else 0.0
        snr = (sig_ptp / noise_std) if noise_std > 1e-12 else 0.0
        if snr >= 2.5:
            quality = "Excellent"
        elif snr >= 1.5:
            quality = "Good"
        elif snr >= 1.0:
            quality = "Marginal"
        else:
            quality = "Reject"
        return QualityMetricResult(
            passed=snr >= 1.0,
            value=float(snr),
            threshold=1.0,
            metric_name="SNR",
            status=quality,
            details={
                "snr": snr,
                "signal_ptp": sig_ptp,
                "noise_std": noise_std,
                "quality_level": quality,
                "method": "Raw (band 600–1120 nm, first-diff noise)",
                "detrended": False,
            },
        )

    snr = g.snr
    quality = g.quality_band.capitalize()
    passed = snr >= 1.0
    if use_smooth_residual:
        method_str = f"Detrended, band 600–1120 nm, smooth/residual (boxcar {boxcar_width_nm:.0f} nm), noise=std(residual)"
    else:
        method_str = "Detrended, band 600–1120 nm, first-diff noise (no edge baseline)"
    details = {
        "snr": snr,
        "signal_max": g.signal_ptp,
        "signal_ptp": g.signal_ptp,
        "baseline_mean": 0.0,
        "baseline_std": g.noise_std,
        "noise_std": g.noise_std,
        "quality_level": quality,
        "method": method_str,
        "detrended": True,
        "band_used": g.band_used,
        "n_points": g.n_points,
    }
    if use_smooth_residual:
        details["boxcar_width_nm"] = boxcar_width_nm
        details["boxcar_repeats"] = boxcar_repeats
    return QualityMetricResult(
        passed=passed,
        value=float(snr),
        threshold=1.0,
        metric_name="SNR",
        status=quality,
        details=details,
    )


def calculate_sliding_window_snr(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    window_nm: float = 100.0,
    stride_nm: Optional[float] = None,
    boxcar_width_nm: float = 11.0,
    boxcar_repeats: int = 2,
) -> Dict[str, any]:
    """Calculate SNR in sliding windows using same boxcar smooth/residual as global SNR.

    Detrend once → boxcar smooth once (600–1120 nm) → residual; per window:
    signal = ptp(smooth), noise = std(residual), SNR = signal/noise.

    Args:
        wavelengths: Wavelength array (nm)
        reflectance: Reflectance array
        window_nm: Size of the sliding window in nm
        stride_nm: Step size between windows in nm (defaults to window_nm / 2)
        boxcar_width_nm: Boxcar width for smooth (default 11)
        boxcar_repeats: Boxcar passes (default 2)

    Returns:
        Dict with 'centers', 'snr_values', 'ranges', 'min_snr', 'max_snr', 'avg_snr', 'window_nm', 'stride_nm'
    """
    if len(wavelengths) < 2:
        return {"centers": np.array([]), "snr_values": np.array([])}

    stride_nm = stride_nm or (window_nm / 2)

    from analysis.snr import compute_window_snr

    band = (ANALYSIS_WAVELENGTH_MIN, 1120.0)
    w = compute_window_snr(
        wavelengths,
        reflectance,
        band_nm=band,
        window_nm=window_nm,
        stride_nm=stride_nm,
        boxcar_width_nm=boxcar_width_nm,
        boxcar_repeats=boxcar_repeats,
    )

    if len(w.snr_per_window) == 0:
        return {
            "centers": np.array([]),
            "snr_values": np.array([]),
            "ranges": [],
            "min_snr": 0.0,
            "max_snr": 0.0,
            "avg_snr": 0.0,
            "window_nm": window_nm,
            "stride_nm": stride_nm,
        }

    half = window_nm / 2.0
    ranges = [(float(c - half), float(c + half)) for c in w.window_centers_nm]
    return {
        "centers": w.window_centers_nm,
        "snr_values": w.snr_per_window,
        "ranges": ranges,
        "min_snr": w.snr_min,
        "max_snr": w.snr_max,
        "avg_snr": w.snr_mean,
        "window_nm": window_nm,
        "stride_nm": stride_nm,
    }


def check_peak_quality(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    *,
    prominence: float = 0.0001,
    min_peak_count: int = 3,
    max_prominence_cv: float = 1.0,
    max_spacing_cv: float = 0.5,
    detrend: bool = True,
    cutoff_frequency: float = 0.008,
    filter_order: int = 3,
) -> QualityMetricResult:
    """Check peak/fringe detection quality.

    Metric 2: Peak/Fringe Detection Quality

    Sub-metrics:
    - Minimum peak count (>= 3 peaks)
    - Peak prominence consistency (CV < 1.0)
    - Peak spacing regularity (CV < 0.5)

    Args:
        wavelengths: Wavelength array in nm
        reflectance: Reflectance values
        prominence: Minimum prominence for peak detection
        min_peak_count: Minimum number of peaks required (default 3)
        max_prominence_cv: Maximum coefficient of variation for prominence (default 1.0)
        max_spacing_cv: Maximum coefficient of variation for spacing (default 0.5)
        detrend: Whether to detrend the signal before peak detection (default True)
        cutoff_frequency: Cutoff frequency for detrending filter (default 0.008)
        filter_order: Order of the Butterworth filter for detrending (default 3)

    Returns:
        QualityMetricResult with peak quality assessment
    """
    # CRITICAL FIX: Detrend the signal before finding peaks to match amplitude analysis
    # This ensures we detect interference fringes, not the overall spectral shape
    signal_for_peaks = reflectance
    if detrend:
        from analysis.measurement_utils import detrend_signal

        signal_for_peaks = detrend_signal(
            wavelengths, reflectance, cutoff_frequency, filter_order
        )

    # Detect peaks on the (optionally detrended) signal
    peak_indices, properties = find_peaks(signal_for_peaks, prominence=prominence)

    peak_count = len(peak_indices)

    # Initialize metrics
    prominence_cv = 0.0
    spacing_cv = 0.0
    mean_prominence = 0.0
    mean_spacing = 0.0

    if peak_count >= 2:
        # Peak prominence consistency
        prominences = properties["prominences"]
        mean_prominence = float(np.mean(prominences))
        std_prominence = float(np.std(prominences, ddof=1))
        prominence_cv = (
            std_prominence / mean_prominence if mean_prominence > 0 else float("inf")
        )

        # Peak spacing regularity
        peak_wavelengths = wavelengths[peak_indices]
        spacings = np.diff(peak_wavelengths)
        mean_spacing = float(np.mean(spacings))
        std_spacing = float(np.std(spacings, ddof=1))
        spacing_cv = std_spacing / mean_spacing if mean_spacing > 0 else float("inf")

    # Check all criteria
    checks_passed = []
    checks_failed = []

    if peak_count >= min_peak_count:
        checks_passed.append("peak_count")
    else:
        checks_failed.append(f"peak_count ({peak_count} < {min_peak_count})")

    if peak_count >= 2:  # Only check if we have enough peaks
        if prominence_cv <= max_prominence_cv:
            checks_passed.append("prominence_consistency")
        else:
            checks_failed.append(
                f"prominence_cv ({prominence_cv:.2f} > {max_prominence_cv})"
            )

        if spacing_cv <= max_spacing_cv:
            checks_passed.append("spacing_regularity")
        else:
            checks_failed.append(f"spacing_cv ({spacing_cv:.2f} > {max_spacing_cv})")

    passed = len(checks_failed) == 0

    details = {
        "peak_count": float(peak_count),
        "prominence_cv": prominence_cv,
        "spacing_cv": spacing_cv,
        "mean_prominence": mean_prominence,
        "mean_spacing": mean_spacing,
        "checks_passed": len(checks_passed),
        "checks_failed": len(checks_failed),
        "detrended": detrend,
    }

    # Determine status
    if not passed:
        status = "Reject"
    elif len(checks_passed) == 3:
        status = "Excellent"
    elif len(checks_passed) == 2:
        status = "Good"
    else:
        status = "Marginal"

    return QualityMetricResult(
        passed=passed,
        value=float(peak_count),
        threshold=float(min_peak_count),
        metric_name="Peak Quality",
        status=status,
        details=details,
    )


def calculate_fit_residual_quality(
    measured: np.ndarray,
    fitted: np.ndarray,
    *,
    measurement_error: Optional[np.ndarray] = None,
    max_nrmse_percent: float = 5.0,
    min_r_squared: float = 0.90,
    chi_squared_range: Tuple[float, float] = (0.8, 1.2),
) -> QualityMetricResult:
    """Calculate fit residual quality metrics.

    Metric 3: Fit Residual Quality

    Sub-metrics:
    - RMSE (< 0.01 absolute)
    - Normalized RMSE (< 5%)
    - R² (> 0.90)
    - Reduced Chi-squared (0.8 - 1.2)

    Args:
        measured: Measured spectrum values
        fitted: Fitted/theoretical spectrum values
        measurement_error: Optional measurement uncertainties (σ) for chi-squared
        max_nrmse_percent: Maximum acceptable NRMSE in percent (default 5%)
        min_r_squared: Minimum acceptable R² (default 0.90)
        chi_squared_range: Acceptable range for reduced chi-squared (default 0.8-1.2)

    Returns:
        QualityMetricResult with fit quality assessment
    """
    residuals = measured - fitted

    # RMSE
    rmse = float(np.sqrt(np.mean(residuals**2)))

    # Normalized RMSE (NRMSE)
    data_range = float(np.max(measured) - np.min(measured))
    nrmse = (rmse / data_range * 100) if data_range > 0 else float("inf")

    # R² (Coefficient of Determination)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((measured - np.mean(measured)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Reduced Chi-squared (χ²/dof)
    if measurement_error is not None and len(measurement_error) == len(measured):
        # Use provided measurement errors
        chi_squared_values = (residuals**2) / (measurement_error**2 + 1e-10)
        dof = len(measured) - 1  # Degrees of freedom (n - number of fitted parameters)
        reduced_chi_squared = float(np.sum(chi_squared_values) / dof)
    else:
        # Estimate from residuals if no error provided
        variance_estimate = np.var(residuals, ddof=1)
        if variance_estimate > 0:
            reduced_chi_squared = float(np.mean(residuals**2) / variance_estimate)
        else:
            reduced_chi_squared = 1.0

    # Check criteria
    checks_passed = []
    checks_failed = []

    if rmse < 0.01:
        checks_passed.append("rmse")
    else:
        checks_failed.append(f"rmse ({rmse:.4f} >= 0.01)")

    if nrmse < max_nrmse_percent:
        checks_passed.append("nrmse")
    else:
        checks_failed.append(f"nrmse ({nrmse:.2f}% >= {max_nrmse_percent}%)")

    if r_squared > min_r_squared:
        checks_passed.append("r_squared")
    else:
        checks_failed.append(f"r_squared ({r_squared:.3f} <= {min_r_squared})")

    if chi_squared_range[0] <= reduced_chi_squared <= chi_squared_range[1]:
        checks_passed.append("chi_squared")
    else:
        checks_failed.append(
            f"chi_squared ({reduced_chi_squared:.3f} outside [{chi_squared_range[0]}, {chi_squared_range[1]}])"
        )

    passed = len(checks_failed) == 0

    details = {
        "rmse": rmse,
        "nrmse_percent": nrmse,
        "r_squared": r_squared,
        "reduced_chi_squared": reduced_chi_squared,
        "checks_passed": len(checks_passed),
        "checks_failed": len(checks_failed),
    }

    # Determine status
    if not passed:
        status = "Reject"
    elif len(checks_passed) == 4:
        status = "Excellent"
    elif len(checks_passed) == 3:
        status = "Good"
    else:
        status = "Marginal"

    return QualityMetricResult(
        passed=passed,
        value=r_squared,
        threshold=min_r_squared,
        metric_name="Fit Residual Quality",
        status=status,
        details=details,
    )


def check_signal_integrity(
    reflectance: np.ndarray,
    *,
    min_dynamic_range: float = 0.05,
    max_saturation_fraction: float = 0.01,
    saturation_limit: float = 0.99,
    max_baseline_drift_percent: float = 10.0,
    edge_fraction: float = 0.1,
) -> QualityMetricResult:
    """Check signal integrity for hardware/acquisition issues.

    Metric 4: Signal Integrity Checks

    Sub-checks:
    - Dynamic range (> 0.05)
    - Saturation detection (< 1% saturated points)
    - Baseline stability (< 10% drift)
    - Negative value check (0% negative values)

    Args:
        reflectance: Reflectance values
        min_dynamic_range: Minimum (max-min)/mean ratio (default 0.05)
        max_saturation_fraction: Maximum fraction of saturated points (default 0.01 = 1%)
        saturation_limit: Reflectance value considered saturated (default 0.99)
        max_baseline_drift_percent: Maximum baseline drift in percent (default 10%)
        edge_fraction: Fraction of data at edges for baseline stability check (default 0.1 = 10%)

    Returns:
        QualityMetricResult with signal integrity assessment
    """
    n = len(reflectance)
    edge_size = max(1, int(n * edge_fraction))

    # Dynamic range check
    mean_val = float(np.mean(reflectance))
    max_val = float(np.max(reflectance))
    min_val = float(np.min(reflectance))
    dynamic_range = (max_val - min_val) / mean_val if mean_val > 0 else 0.0

    # Saturation detection
    saturated_count = int(np.sum(reflectance >= saturation_limit))
    saturation_fraction = saturated_count / n

    # Baseline stability (drift between first and last 10%)
    first_edge_mean = float(np.mean(reflectance[:edge_size]))
    last_edge_mean = float(np.mean(reflectance[-edge_size:]))
    baseline_drift_percent = (
        abs(first_edge_mean - last_edge_mean) / mean_val * 100 if mean_val > 0 else 0.0
    )

    # Negative value check
    negative_count = int(np.sum(reflectance < 0))

    # Categorize findings per specification names
    critical_errors = []
    indicative_warnings = []

    # 1. Negative value check (= 0%)
    if negative_count > 0:
        critical_errors.append(f"Negative value check ({negative_count} points)")

    # 2. Saturation detection (< 1%)
    if saturation_fraction >= max_saturation_fraction:
        critical_errors.append(
            f"Saturation detection ({saturation_fraction * 100:.1f}%)"
        )

    # 3. Dynamic range (> 0.05)
    if dynamic_range <= min_dynamic_range:
        indicative_warnings.append(f"Dynamic range ({dynamic_range:.3f})")

    # 4. Baseline stability (< 10%)
    if baseline_drift_percent >= max_baseline_drift_percent:
        indicative_warnings.append(
            f"Baseline stability ({baseline_drift_percent:.1f}%)"
        )

    # Final Decision
    if len(critical_errors) > 0:
        status = "Reject"
        passed = False
    elif len(indicative_warnings) > 0:
        status = "Marginal"  # Orange in UI
        passed = True  # Still passes the hard gate
    else:
        status = "Excellent"
        passed = True

    details = {
        "dynamic_range": dynamic_range,
        "saturation_fraction": saturation_fraction,
        "baseline_drift_percent": baseline_drift_percent,
        "negative_count": float(negative_count),
        "critical_errors": critical_errors,
        "indicative_warnings": indicative_warnings,
    }

    return QualityMetricResult(
        passed=passed,
        value=dynamic_range,
        threshold=min_dynamic_range,
        metric_name="Signal Integrity",
        status=status,
        details=details,
    )


def check_spectral_completeness(
    wavelengths: np.ndarray,
    *,
    min_wavelength_span_nm: float = 400.0,
    min_point_density: float = 1.0,
    max_gap_nm: float = 5.0,
) -> QualityMetricResult:
    """Check spectral completeness and coverage.

    Metric 5: Spectral Completeness

    Sub-checks:
    - Wavelength span (>= 400 nm)
    - Data point density (>= 1 point per nm)
    - Gap detection (no gaps > 5 nm)

    Args:
        wavelengths: Wavelength array in nm
        min_wavelength_span_nm: Minimum wavelength coverage (default 400 nm)
        min_point_density: Minimum points per nm (default 1.0)
        max_gap_nm: Maximum allowed gap between consecutive points (default 5 nm)

    Returns:
        QualityMetricResult with spectral completeness assessment
    """
    # Wavelength span
    wavelength_span = float(np.max(wavelengths) - np.min(wavelengths))

    # Data point density
    n_points = len(wavelengths)
    point_density = n_points / wavelength_span if wavelength_span > 0 else 0.0

    # Gap detection
    gaps = np.diff(wavelengths)
    max_gap = float(np.max(gaps)) if len(gaps) > 0 else 0.0
    large_gaps = np.sum(gaps > max_gap_nm)

    # Check all criteria
    checks_passed = []
    checks_failed = []

    if wavelength_span >= min_wavelength_span_nm:
        checks_passed.append("wavelength_span")
    else:
        checks_failed.append(
            f"wavelength_span ({wavelength_span:.1f} < {min_wavelength_span_nm} nm)"
        )

    if point_density >= min_point_density:
        checks_passed.append("point_density")
    else:
        checks_failed.append(
            f"point_density ({point_density:.2f} < {min_point_density} pts/nm)"
        )

    if large_gaps == 0:
        checks_passed.append("no_large_gaps")
    else:
        checks_failed.append(f"large_gaps ({large_gaps} gaps > {max_gap_nm} nm)")

    passed = len(checks_failed) == 0

    details = {
        "wavelength_span_nm": wavelength_span,
        "point_density": point_density,
        "max_gap_nm": max_gap,
        "large_gap_count": float(large_gaps),
        "checks_passed": len(checks_passed),
        "checks_failed": len(checks_failed),
    }

    # Determine status
    if not passed:
        status = "Reject"
    elif len(checks_passed) == 3:
        status = "Excellent"
    else:
        status = "Good"

    return QualityMetricResult(
        passed=passed,
        value=wavelength_span,
        threshold=min_wavelength_span_nm,
        metric_name="Spectral Completeness",
        status=status,
        details=details,
    )


def _compute_quality_tier(
    metrics: Dict[str, QualityMetricResult],
    config: Optional[Dict] = None,
) -> str:
    """
    Map metrics to the four quality tiers:
    GOOD, NOISY_FITTABLE, NOISY_UNFITTABLE, BAD.
    """
    cfg = config or {}
    snr_result = metrics.get("snr")
    peak_result = metrics.get("peak_quality")
    fit_result = metrics.get("fit_quality")
    integrity_result = metrics.get("signal_integrity")

    snr = float(snr_result.value) if snr_result else 0.0
    peak_count = int(peak_result.details.get("peak_count", 0)) if peak_result else 0
    saturated = False
    flat = False
    if integrity_result and integrity_result.details:
        sat_frac = integrity_result.details.get("saturation_fraction", 0)
        dyn_range = integrity_result.details.get("dynamic_range", 0)
        max_sat = cfg.get("max_saturation_fraction", 0.01)
        min_dyn = cfg.get("min_dynamic_range", 0.05)
        saturated = sat_frac >= max_sat
        flat = dyn_range <= min_dyn

    # BAD: SNR < 5, <2 peaks, saturated, or flat
    if snr < SNR_NOISY_LOW:
        return "BAD"
    if peak_count < MIN_PEAKS_FOR_QUALITY:
        return "BAD"
    if saturated or flat:
        return "BAD"

    # Noisy band: 5 <= SNR < 20
    if SNR_NOISY_LOW <= snr < SNR_GOOD_MIN:
        smoothing_improvement = 0.0
        if snr_result and snr_result.details:
            smoothing_improvement = float(
                snr_result.details.get("smoothing_improvement", 0.0)
            )
        nrmse = 100.0
        if fit_result and fit_result.details:
            nrmse = float(fit_result.details.get("nrmse_percent", 100.0))
        if (
            smoothing_improvement > SMOOTHING_IMPROVEMENT_MIN
            and nrmse < NOISY_FITTABLE_NRMSE_PERCENT_MAX
        ):
            return "NOISY_FITTABLE"
        return "NOISY_UNFITTABLE"

    # SNR >= 20: candidate for GOOD
    if peak_count < MIN_PEAKS_FOR_QUALITY:
        return "BAD"
    # GOOD requires fit criteria when fit is present
    if fit_result and fit_result.details:
        nrmse = float(fit_result.details.get("nrmse_percent", 100.0))
        r2 = float(fit_result.details.get("r_squared", 0.0))
        chi2 = float(fit_result.details.get("reduced_chi_squared", 999.0))
        if (
            nrmse >= GOOD_NRMSE_PERCENT_MAX
            or r2 < GOOD_R2_MIN
            or chi2 >= GOOD_CHI2_MAX
        ):
            # Still good SNR/peaks but fit not excellent -> treat as Good for tier
            return "GOOD"
    return "GOOD"


def assess_spectrum_quality(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    *,
    fitted_spectrum: Optional[np.ndarray] = None,
    measurement_error: Optional[np.ndarray] = None,
    prominence: float = 0.0001,
    config: Optional[Dict] = None,
) -> SpectrumQualityReport:
    """Perform comprehensive quality assessment on a spectrum.

    Runs all quality metrics (1-5) and generates a comprehensive report.

    Args:
        wavelengths: Wavelength array in nm
        reflectance: Reflectance values
        fitted_spectrum: Optional fitted/theoretical spectrum for residual analysis
        measurement_error: Optional measurement uncertainties for chi-squared
        prominence: Peak detection prominence threshold
        config: Optional configuration dict to override default thresholds

    Returns:
        SpectrumQualityReport with comprehensive assessment
    """
    config = config or {}

    # CRITICAL: Apply the Shlomo rule - only analyze 600nm - 1200nm
    mask = (wavelengths >= ANALYSIS_WAVELENGTH_MIN) & (
        wavelengths <= ANALYSIS_WAVELENGTH_MAX
    )
    wavelengths = wavelengths[mask]
    reflectance = reflectance[mask]
    if fitted_spectrum is not None and len(fitted_spectrum) == len(mask):
        fitted_spectrum = fitted_spectrum[mask]

    if len(wavelengths) < 10:
        return SpectrumQualityReport(
            overall_quality="BAD",
            passed_all_checks=False,
            metrics={},
            warnings=["Spectrum does not cover the 600-1200nm usable range"],
            failures=["Insufficient wavelength coverage for analysis"],
        )

    metrics = {}
    warnings = []
    failures = []

    # Metric 1: SNR (Part C: smooth/residual pipeline with threshold 20, or legacy)
    use_part_c_snr = config.get("use_part_c_snr", True)
    if use_part_c_snr:
        from .snr import (
            SNR_PASS_THRESHOLD,
            _noise_std_first_difference,
            compute_global_snr,
            get_snr_smooth_residual_curves,
            compute_window_snr,
            window_snr_result_to_dict,
        )

        band_nm = (600.0, 1120.0)
        global_snr = compute_global_snr(
            wavelengths,
            reflectance,
            band_nm=band_nm,
            detrend_cutoff=config.get("snr_cutoff_frequency", 0.008),
            detrend_order=config.get("snr_filter_order", 3),
            boxcar_width_nm=11.0,
            boxcar_repeats=2,
        )
        wl_curves, detrended, smooth, residual = get_snr_smooth_residual_curves(
            wavelengths,
            reflectance,
            band_nm=band_nm,
            detrend_cutoff=config.get("snr_cutoff_frequency", 0.008),
            detrend_order=config.get("snr_filter_order", 3),
            boxcar_width_nm=11.0,
            boxcar_repeats=2,
        )
        # For quality tiers: raw SNR (no smoothing) and smoothing improvement
        raw_noise = _noise_std_first_difference(detrended)
        raw_signal = float(np.ptp(detrended))
        raw_snr = (
            raw_signal / raw_noise
            if (raw_noise and raw_noise > 1e-12)
            else 0.0
        )
        smoothing_improvement = (
            (global_snr.snr - raw_snr) / raw_snr
            if (raw_snr and raw_snr > 1e-12)
            else 0.0
        )
        window_nm = config.get("snr_window_nm", 100.0)
        stride_nm = config.get("snr_stride_nm", 50.0)
        window_snr_result = compute_window_snr(
            wavelengths,
            reflectance,
            band_nm=band_nm,
            window_nm=window_nm,
            stride_nm=stride_nm,
            detrend_cutoff=config.get("snr_cutoff_frequency", 0.008),
            detrend_order=config.get("snr_filter_order", 3),
            boxcar_width_nm=11.0,
            boxcar_repeats=2,
        )
        sw_dict = window_snr_result_to_dict(window_snr_result)

        snr_result = QualityMetricResult(
            passed=global_snr.passed,
            value=global_snr.snr,
            threshold=SNR_PASS_THRESHOLD,
            metric_name="SNR",
            status=global_snr.quality_band.title(),
            details={
                "snr": global_snr.snr,
                "signal_ptp": global_snr.signal_ptp,
                "noise_std": global_snr.noise_std,
                "quality_level": global_snr.quality_band,
                "method": "Part C (smooth/residual, 600-1120 nm, boxcar 11 nm 2 passes)",
                "detrended": True,
                "snr_smooth_residual_curves": (wl_curves, detrended, smooth, residual),
                "sliding_window_snr": sw_dict,
                "raw_snr": raw_snr,
                "smoothing_improvement": smoothing_improvement,
            },
        )
    else:
        snr_result = calculate_snr(
            wavelengths,
            reflectance,
            use_detrended=config.get("use_detrended_snr", True),
            cutoff_frequency=config.get("snr_cutoff_frequency", 0.008),
            filter_order=config.get("snr_filter_order", 3),
        )
        snr_kw = {}
        if "snr_window_nm" in config:
            snr_kw["window_nm"] = config["snr_window_nm"]
        window_snr = calculate_sliding_window_snr(wavelengths, reflectance, **snr_kw)
        snr_result.details["sliding_window_snr"] = window_snr

    metrics["snr"] = snr_result

    if not snr_result.passed:
        failures.append(
            f"SNR too low: {snr_result.value:.2f} < {snr_result.threshold}"
        )
    elif snr_result.details.get("quality_level") == "Marginal":
        warnings.append(f"SNR marginal: {snr_result.value:.2f}")

    # Metric 2: Peak Quality
    peak_result = check_peak_quality(
        wavelengths,
        reflectance,
        prominence=prominence,
        min_peak_count=config.get("min_peak_count", 3),
        max_prominence_cv=config.get("max_prominence_cv", 1.0),
        max_spacing_cv=config.get("max_spacing_cv", 0.5),
    )
    metrics["peak_quality"] = peak_result

    if not peak_result.passed:
        failures.append(
            f"Peak quality insufficient: {peak_result.details.get('checks_failed', 0)} checks failed"
        )

    # Metric 3: Fit Residual Quality (only if fitted spectrum provided)
    if fitted_spectrum is not None:
        fit_result = calculate_fit_residual_quality(
            reflectance,
            fitted_spectrum,
            measurement_error=measurement_error,
            max_nrmse_percent=config.get("max_nrmse_percent", 5.0),
            min_r_squared=config.get("min_r_squared", 0.90),
            chi_squared_range=config.get("chi_squared_range", (0.8, 1.2)),
        )
        metrics["fit_quality"] = fit_result
        # Fit quality is diagnostic only: do not use it for pass/fail or failures.

    # Metric 4: Signal Integrity (Evaluative)
    integrity_result = check_signal_integrity(
        reflectance,
        min_dynamic_range=config.get("min_dynamic_range", 0.05),
        max_saturation_fraction=config.get("max_saturation_fraction", 0.01),
        max_baseline_drift_percent=config.get("max_baseline_drift_percent", 10.0),
    )
    metrics["signal_integrity"] = integrity_result

    # Handle Critical vs Indicative Findings
    if not integrity_result.passed:
        for error in integrity_result.details.get("critical_errors", []):
            failures.append(f"FAILURE: {error}")

    # Always show indicative findings as warnings
    for note in integrity_result.details.get("indicative_warnings", []):
        warnings.append(f"Signal integrity Warning: {note}")

    # Metric 5: Spectral Completeness
    completeness_result = check_spectral_completeness(
        wavelengths,
        min_wavelength_span_nm=config.get("min_wavelength_span_nm", 400.0),
        min_point_density=config.get("min_point_density", 1.0),
        max_gap_nm=config.get("max_gap_nm", 5.0),
    )
    metrics["spectral_completeness"] = completeness_result

    if not completeness_result.passed:
        failures.append(
            f"Spectral completeness insufficient: {completeness_result.details.get('checks_failed', 0)} checks failed"
        )

    # Overall quality: four tiers (GOOD, NOISY_FITTABLE, NOISY_UNFITTABLE, BAD)
    overall_quality = _compute_quality_tier(metrics, config)
    # Passed = usable for fitting (GOOD or NOISY_FITTABLE)
    passed_all_checks = overall_quality in ("GOOD", "NOISY_FITTABLE")

    return SpectrumQualityReport(
        overall_quality=overall_quality,
        passed_all_checks=passed_all_checks,
        metrics=metrics,
        warnings=warnings,
        failures=failures,
    )


__all__ = [
    "QualityMetricResult",
    "SpectrumQualityReport",
    "calculate_snr",
    "calculate_sliding_window_snr",
    "check_peak_quality",
    "calculate_fit_residual_quality",
    "check_signal_integrity",
    "check_spectral_completeness",
    "assess_spectrum_quality",
]
