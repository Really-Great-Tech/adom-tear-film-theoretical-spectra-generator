"""
SNR (signal-to-noise ratio) for tear film spectra — Part C: Automated Quality Metrics.

- Compute on detrended signal; restrict to band 600–1120 nm.
- Global SNR: Detrend → boxcar smooth (11 nm, 2 passes) → residual = detrended - smooth.
  Signal = ptp(smooth), noise = std(residual), SNR = signal/noise. Pass threshold ≥ 20.
- Per-window: signal_global = ptp(smooth), noise_local = std(residual in window),
  SNR_window = signal_global / noise_local; 3-window moving median applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Part C spec: Pass ≥ 20, Reject < 20
SNR_PASS_THRESHOLD = 20.0

# Display bands (for UI)
SNR_EXCELLENT = 2.5
SNR_GOOD = 1.5
SNR_MARGINAL = 1.0


@dataclass
class GlobalSNRResult:
    """Result of global SNR calculation."""

    snr: float
    quality_band: str  # "excellent" | "good" | "marginal" | "reject"
    signal_ptp: float
    noise_std: float
    band_used: Tuple[float, float]
    n_points: int
    passed: bool  # True if snr >= SNR_PASS_THRESHOLD (Part C spec)


@dataclass
class WindowSNRResult:
    """Per-window SNR (from detrended signal, one detrend for full band)."""

    window_centers_nm: np.ndarray
    snr_per_window: np.ndarray
    signal_ptp_per_window: np.ndarray
    noise_std_per_window: np.ndarray
    snr_min: float
    snr_max: float
    snr_mean: float
    window_nm: float
    stride_nm: float


def _crop_to_band(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    band_nm: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (wl, refl) cropped to band_nm (min, max) inclusive."""
    wl = np.asarray(wavelengths)
    refl = np.asarray(reflectance)
    lo, hi = band_nm[0], band_nm[1]
    mask = (wl >= lo) & (wl <= hi)
    return wl[mask], refl[mask]


def _noise_std_first_difference(signal: np.ndarray) -> float:
    """Noise std from first difference: std(diff(signal)) / sqrt(2)."""
    if len(signal) < 2:
        return 0.0
    d = np.diff(signal.astype(float))
    std_d = float(np.std(d, ddof=1))
    return std_d / np.sqrt(2.0) if std_d > 1e-12 else 0.0


def compute_global_snr(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    *,
    band_nm: Tuple[float, float] = (600.0, 1120.0),
    detrend_cutoff: float = 0.008,
    detrend_order: int = 3,
    detrend_signal_fn=None,
    use_smooth_residual: bool = True,
    boxcar_width_nm: float = 11.0,
    boxcar_repeats: int = 2,
) -> GlobalSNRResult:
    """
    Global SNR on detrended signal, restricted to band (Part C).

    use_smooth_residual=True (default): Detrend → boxcar smooth → residual = detrended - smooth.
    Signal = ptp(smooth), noise = std(residual). Pass threshold ≥ 20.
    """
    from .measurement_utils import detrend_signal, boxcar_smooth

    detrend_fn = detrend_signal_fn or detrend_signal
    wl, refl = _crop_to_band(wavelengths, reflectance, band_nm)
    if len(wl) < 3:
        return GlobalSNRResult(
            snr=0.0,
            quality_band="reject",
            signal_ptp=0.0,
            noise_std=0.0,
            band_used=band_nm,
            n_points=len(wl),
            passed=False,
        )

    detrended = detrend_fn(wl, refl, detrend_cutoff, detrend_order)

    if use_smooth_residual:
        smooth = boxcar_smooth(
            detrended, wl, width_nm=boxcar_width_nm, repeats=boxcar_repeats
        )
        residual = detrended - smooth
        signal_ptp = float(np.ptp(smooth))
        noise_std = float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0
    else:
        signal_ptp = float(np.ptp(detrended))
        noise_std = _noise_std_first_difference(detrended)

    if noise_std < 1e-12:
        snr = 0.0
    else:
        snr = signal_ptp / noise_std

    passed = snr >= SNR_PASS_THRESHOLD
    if snr >= SNR_EXCELLENT:
        quality_band = "excellent"
    elif snr >= SNR_GOOD:
        quality_band = "good"
    elif snr >= SNR_MARGINAL:
        quality_band = "marginal"
    else:
        quality_band = "reject"

    return GlobalSNRResult(
        snr=float(snr),
        quality_band=quality_band,
        signal_ptp=signal_ptp,
        noise_std=noise_std,
        band_used=band_nm,
        n_points=len(wl),
        passed=passed,
    )


def get_snr_smooth_residual_curves(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    *,
    band_nm: Tuple[float, float] = (600.0, 1120.0),
    detrend_cutoff: float = 0.008,
    detrend_order: int = 3,
    boxcar_width_nm: float = 11.0,
    boxcar_repeats: int = 2,
    detrend_signal_fn=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (wl, detrended, smooth, residual) in the SNR band for plotting.
    Same pipeline as compute_global_snr.
    """
    from .measurement_utils import detrend_signal, boxcar_smooth

    detrend_fn = detrend_signal_fn or detrend_signal
    wl, refl = _crop_to_band(wavelengths, reflectance, band_nm)
    if len(wl) < 3:
        return np.array([]), np.array([]), np.array([]), np.array([])

    detrended = detrend_fn(wl, refl, detrend_cutoff, detrend_order)
    smooth = boxcar_smooth(
        detrended, wl, width_nm=boxcar_width_nm, repeats=boxcar_repeats
    )
    residual = detrended - smooth
    return wl, detrended, smooth, residual


def _window_slices(
    wavelengths: np.ndarray,
    window_nm: float,
    stride_nm: float,
    band_nm: Tuple[float, float],
) -> list:
    """Return list of (start_idx, end_idx) for windows within band (by wavelength)."""
    wl = np.asarray(wavelengths)
    lo, hi = band_nm[0], band_nm[1]
    in_band = np.where((wl >= lo) & (wl <= hi))[0]
    if len(in_band) == 0:
        return []
    wl_band = wl[in_band]
    win_start_wl = wl_band[0]
    slices = []
    while win_start_wl + window_nm <= wl_band[-1] + 1e-6:
        win_end_wl = win_start_wl + window_nm
        mask = (wl_band >= win_start_wl) & (wl_band < win_end_wl)
        local_idx = np.where(mask)[0]
        if len(local_idx) >= 2:
            start = in_band[local_idx[0]]
            end = in_band[local_idx[-1]] + 1
            slices.append((start, end))
        win_start_wl += stride_nm
    return slices


def compute_window_snr(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    *,
    band_nm: Tuple[float, float] = (600.0, 1120.0),
    window_nm: float = 100.0,
    stride_nm: float = 50.0,
    detrend_cutoff: float = 0.008,
    detrend_order: int = 3,
    detrend_signal_fn=None,
    boxcar_width_nm: float = 11.0,
    boxcar_repeats: int = 2,
) -> WindowSNRResult:
    """
    Per-window SNR: global ptp(smooth) / local std(residual).
    Same detrend → boxcar smooth → residual. 3-window moving median on per-window SNR.
    """
    from .measurement_utils import detrend_signal, boxcar_smooth

    detrend_fn = detrend_signal_fn or detrend_signal
    wl, refl = _crop_to_band(wavelengths, reflectance, band_nm)
    if len(wl) < 3:
        return WindowSNRResult(
            window_centers_nm=np.array([]),
            snr_per_window=np.array([]),
            signal_ptp_per_window=np.array([]),
            noise_std_per_window=np.array([]),
            snr_min=0.0,
            snr_max=0.0,
            snr_mean=0.0,
            window_nm=window_nm,
            stride_nm=stride_nm,
        )

    detrended = detrend_fn(wl, refl, detrend_cutoff, detrend_order)
    smooth = boxcar_smooth(
        detrended, wl, width_nm=boxcar_width_nm, repeats=boxcar_repeats
    )
    residual = detrended - smooth

    global_signal_ptp = float(np.ptp(smooth))

    slices = _window_slices(wl, window_nm, stride_nm, (float(wl.min()), float(wl.max())))
    if not slices:
        return WindowSNRResult(
            window_centers_nm=np.array([]),
            snr_per_window=np.array([]),
            signal_ptp_per_window=np.array([]),
            noise_std_per_window=np.array([]),
            snr_min=0.0,
            snr_max=0.0,
            snr_mean=0.0,
            window_nm=window_nm,
            stride_nm=stride_nm,
        )

    centers = []
    snrs = []
    noises = []
    for start, end in slices:
        res_slice = residual[start:end]
        w = wl[start:end]
        centers.append(float(0.5 * (w[0] + w[-1])))
        noise = float(np.std(res_slice, ddof=1)) if len(res_slice) > 1 else 0.0
        noises.append(noise)
        snrs.append(global_signal_ptp / noise if noise > 1e-12 else 0.0)

    snr_arr = np.array(snrs)
    if len(snr_arr) >= 2:
        from scipy.ndimage import median_filter

        snr_arr = median_filter(snr_arr.astype(float), size=3, mode="nearest")
    ptps = np.full(len(centers), global_signal_ptp)
    return WindowSNRResult(
        window_centers_nm=np.array(centers),
        snr_per_window=snr_arr,
        signal_ptp_per_window=ptps,
        noise_std_per_window=np.array(noises),
        snr_min=float(np.min(snr_arr)) if len(snr_arr) else 0.0,
        snr_max=float(np.max(snr_arr)) if len(snr_arr) else 0.0,
        snr_mean=float(np.mean(snr_arr)) if len(snr_arr) else 0.0,
        window_nm=window_nm,
        stride_nm=stride_nm,
    )


def window_snr_result_to_dict(result: WindowSNRResult) -> dict:
    """Convert WindowSNRResult to dict for display (centers, snr_values, ranges, etc.)."""
    centers = result.window_centers_nm
    half = result.window_nm / 2.0
    ranges = [(float(c - half), float(c + half)) for c in centers]
    return {
        "centers": centers,
        "snr_values": result.snr_per_window,
        "ranges": ranges,
        "min_snr": result.snr_min,
        "max_snr": result.snr_max,
        "avg_snr": result.snr_mean,
        "window_nm": result.window_nm,
        "stride_nm": result.stride_nm,
        "threshold": SNR_PASS_THRESHOLD,  # Part C pass threshold for chart
    }
