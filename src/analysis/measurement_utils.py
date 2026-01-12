"""Shared measurement preprocessing utilities."""

from __future__ import annotations

import csv
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks, get_window

WarnFn = Callable[[str], None]


def _default_warn(message: str) -> None:
    print(f"[measurement_utils] {message}")


@dataclass(frozen=True)
class PreparedMeasurement:
    """Container holding cached features for a measurement spectrum."""

    wavelengths: np.ndarray
    reflectance: np.ndarray
    detrended: np.ndarray
    peaks: pd.DataFrame
    resampled_wavelengths: np.ndarray
    resampled_reflectance: np.ndarray
    resampled_detrended: np.ndarray
    fft_frequencies: np.ndarray
    fft_spectrum: np.ndarray


@dataclass(frozen=True)
class PreparedTheoreticalSpectrum:
    """Cached feature bundle for a theoretical spectrum evaluated on a grid."""

    aligned_reflectance: np.ndarray
    detrended: np.ndarray
    peaks: pd.DataFrame
    resampled_reflectance: np.ndarray
    resampled_detrended: np.ndarray
    fft_spectrum: np.ndarray


@dataclass(frozen=True)
class FFTArtifacts:
    """Legacy FFT container retained for backwards compatibility."""

    freqs: np.ndarray
    spectrum: np.ndarray


def load_txt_file_enhanced(file_path: Path | str) -> pd.DataFrame:
    """Load AdOM TXT exports that contain metadata headers."""

    path = Path(file_path)
    wavelengths: list[float] = []
    intensities: list[float] = []
    data_started = False

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">>>>>Begin Spectral Data<<<<<"):
                data_started = True
                continue

            parts = line.split()
            if len(parts) < 2:
                if data_started:
                    break
                continue

            try:
                wl, intensity = float(parts[0]), float(parts[1])
            except ValueError:
                if data_started:
                    break
                continue

            wavelengths.append(wl)
            intensities.append(intensity)
            data_started = True

    if not wavelengths:
        raise ValueError(f"No spectral data found in {path}")

    df = pd.DataFrame({"wavelength": wavelengths, "reflectance": intensities})
    return df


def _load_text_measurement(path: Path) -> pd.DataFrame:
    try:
        return load_txt_file_enhanced(path)
    except ValueError:
        wavelengths: list[float] = []
        intensities: list[float] = []
        data_started = False
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            reader = csv.reader(handle, delimiter="	")
            for line in reader:
                if not line:
                    continue
                if len(line) == 1:
                    parts = line[0].split()
                else:
                    parts = [token for cell in line for token in cell.split()]
                if len(parts) < 2:
                    if data_started:
                        break
                    continue
                try:
                    wl = float(parts[0])
                    intensity = float(parts[1])
                except ValueError:
                    if data_started:
                        break
                    continue
                wavelengths.append(wl)
                intensities.append(intensity)
                data_started = True
        if not wavelengths:
            raise
        return pd.DataFrame({"wavelength": wavelengths, "reflectance": intensities})


def load_measurement_spectrum(path: Path | str, measurement_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load a measurement spectrum into a dataframe."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Measurement file not found: {path}")

    fmt = (measurement_cfg.get("format") or path.suffix.lstrip(".")).lower()
    header_rows = measurement_cfg.get("header_rows")
    skiprows = int(header_rows) if header_rows is not None else None
    wavelength_col = int(measurement_cfg.get("wavelength_column", 0))
    reflectance_col = int(measurement_cfg.get("reflectance_column", 1))
    usecols = [wavelength_col, reflectance_col]

    if fmt in {"txt", "dat", "tsv"}:
        df = _load_text_measurement(path)
    elif fmt == "csv":
        read_kwargs: Dict[str, Any] = {
            "comment": "#",
            "usecols": usecols,
            "skiprows": skiprows,
            "header": None,
        }
        delimiter = measurement_cfg.get("delimiter")
        if delimiter:
            read_kwargs["sep"] = delimiter
        df = pd.read_csv(path, **read_kwargs)
    elif fmt in {"xls", "xlsx"}:
        df = pd.read_excel(path, usecols=usecols, skiprows=skiprows, header=None)
    else:
        raise ValueError(f"Unsupported measurement format: {fmt}")

    df = df.rename(columns={df.columns[0]: "wavelength", df.columns[1]: "reflectance"})
    df["wavelength"] = pd.to_numeric(df["wavelength"], errors="coerce")
    df["reflectance"] = pd.to_numeric(df["reflectance"], errors="coerce")
    df = df.dropna().sort_values("wavelength").reset_index(drop=True)
    return df


def load_measurement_files(
    measurements_dir: Path,
    config: Dict[str, Any],
    *,
    warn: WarnFn = _default_warn,
) -> Dict[str, pd.DataFrame]:
    """Load all measurement spectra matching the configured file pattern."""

    results: Dict[str, pd.DataFrame] = {}
    if not measurements_dir.exists():
        warn(f"Measurements directory not found: {measurements_dir}")
        return results

    meas_cfg = config.get("measurements", {})
    file_pattern = meas_cfg.get("file_pattern", "*.txt")
    pattern_path = measurements_dir / file_pattern
    for file in sorted(glob.glob(str(pattern_path))):
        path = Path(file)
        try:
            df = load_measurement_spectrum(path, meas_cfg)
        except Exception as exc:
            warn(f"Error loading {path}: {exc}")
            continue
        if df.empty:
            continue
        results[path.stem] = df
    return results


def interpolate_measurement_to_theoretical(
    measured_df: pd.DataFrame,
    theoretical_wavelengths: np.ndarray,
) -> np.ndarray:
    """Interpolate measured reflectance onto the theoretical wavelength grid."""

    return np.interp(
        theoretical_wavelengths,
        measured_df["wavelength"].to_numpy(),
        measured_df["reflectance"].to_numpy(),
    )


def calculate_fit_metrics(measured: np.ndarray, theoretical: np.ndarray) -> Dict[str, float]:
    """Compute standard fit metrics between measured and theoretical spectra."""

    ss_res = float(np.sum((measured - theoretical) ** 2))
    ss_tot = float(np.sum((measured - measured.mean()) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    rmse = float(np.sqrt(np.mean((measured - theoretical) ** 2)))
    mae = float(np.mean(np.abs(measured - theoretical)))
    with np.errstate(divide="ignore", invalid="ignore"):
        perc = np.abs((measured - theoretical) / measured)
        perc = perc[np.isfinite(perc)]
        mape = float(np.mean(perc) * 100) if len(perc) else 0.0
    return {"R²": r_squared, "RMSE": rmse, "MAE": mae, "MAPE (%)": mape}


def detrend_signal(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    cutoff_frequency: float,
    filter_order: int,
) -> np.ndarray:
    """High-pass filter to remove low-frequency drift from spectra."""

    wavelengths = np.asarray(wavelengths)
    reflectance = np.asarray(reflectance)
    if wavelengths.ndim != 1 or reflectance.ndim != 1:
        raise ValueError("wavelengths and reflectance must be 1-D arrays")
    if len(wavelengths) != len(reflectance):
        raise ValueError("wavelength and reflectance arrays must be equal length")

    diffs = np.diff(wavelengths)
    if not np.all(diffs > 0):
        raise ValueError("wavelengths must be strictly increasing for detrending")

    sampling_interval = float(np.mean(diffs))
    sampling_frequency = 1.0 / sampling_interval
    nyquist_freq = 0.5 * sampling_frequency
    normal_cutoff = min(max(cutoff_frequency / nyquist_freq, 1e-6), 0.999)

    b, a = butter(filter_order, normal_cutoff, btype="high", analog=False)
    try:
        return filtfilt(b, a, reflectance)
    except ValueError:
        return reflectance.copy()


def boxcar_smooth(
    intensity: np.ndarray,
    wavelengths_nm: np.ndarray,
    width_nm: float = 17.0,
    repeats: int = 1,
) -> np.ndarray:
    """Apply boxcar (uniform/moving average) smoothing matching TFI/LTA system.

    Args:
        intensity: Intensity/reflectance values to smooth.
        wavelengths_nm: Wavelength array in nanometers.
        width_nm: Smoothing width in nanometers (default 17.0 from TFI config).
        repeats: Number of smoothing passes (default 1; use 2 for peak detection).

    Returns:
        Smoothed intensity values.
    """
    step_nm = float(np.median(np.diff(wavelengths_nm)))
    window_size = max(1, int(round(width_nm / step_nm)))
    smoothed = intensity.copy()
    for _ in range(repeats):
        smoothed = uniform_filter1d(smoothed, size=window_size, mode='nearest')
    return smoothed


def gaussian_smooth(
    intensity: np.ndarray,
    kernel_size: int = 11,
) -> np.ndarray:
    """Apply Gaussian smoothing with sample-based kernel size.

    Args:
        intensity: Intensity/reflectance values to smooth.
        kernel_size: Gaussian kernel size in samples (7, 9, or 11).

    Returns:
        Smoothed intensity values.
    """
    # gaussian_filter1d uses kernel length = 2 * truncate * sigma + 1 (truncate defaults to 4)
    # Solve for sigma to get exact kernel_size: sigma = (kernel_size - 1) / (2 * truncate)
    truncate = 4.0  # scipy default
    sigma = (kernel_size - 1) / (2 * truncate)
    return gaussian_filter1d(intensity.copy(), sigma=sigma, mode='nearest')


def detect_peaks(
    wavelengths: np.ndarray,
    signal: np.ndarray,
    *,
    prominence: float,
    height: Optional[float] = None,
) -> pd.DataFrame:
    """Detect peaks and return dataframe of peak positions."""

    peak_indices, properties = find_peaks(signal, prominence=prominence, height=height)
    peaks_df = pd.DataFrame(
        {
            "wavelength": wavelengths[peak_indices],
            "amplitude": signal[peak_indices],
            "prominence": properties.get("prominences", np.zeros_like(peak_indices, dtype=float)),
        }
    )
    return peaks_df.reset_index(drop=True)


def detect_valleys(
    wavelengths: np.ndarray,
    signal: np.ndarray,
    *,
    prominence: float,
) -> pd.DataFrame:
    """Detect valleys by reusing the peak detector on the inverted signal."""

    valley_indices, properties = find_peaks(-signal, prominence=prominence)
    valleys_df = pd.DataFrame(
        {
            "wavelength": wavelengths[valley_indices],
            "amplitude": signal[valley_indices],
            "prominence": properties.get("prominences", np.zeros_like(valley_indices, dtype=float)),
        }
    )
    return valleys_df.reset_index(drop=True)


def resample_uniform_grid(
    wavelengths: np.ndarray,
    values: np.ndarray,
    *,
    num_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample onto a uniform grid for FFT analysis."""

    wl_min = float(np.min(wavelengths))
    wl_max = float(np.max(wavelengths))
    grid = np.linspace(wl_min, wl_max, num_points)
    resampled = np.interp(grid, wavelengths, values)
    return grid, resampled


def compute_fft(
    wavelengths: np.ndarray,
    signal: np.ndarray,
    *,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the one-sided FFT for the provided signal."""

    wavelengths = np.asarray(wavelengths)
    spacing = float(np.mean(np.diff(wavelengths)))
    if window == "hann":
        window_vals = np.hanning(len(signal))
    else:
        window_vals = np.ones_like(signal)

    windowed = signal * window_vals
    fft_values = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(windowed), d=spacing)
    return freqs, fft_values


def compute_fft_artifacts(signal: np.ndarray, window: str = "hann") -> FFTArtifacts:
    """Legacy helper that mirrors the previous compute_fft API."""

    window_values = get_window(window, len(signal), fftbins=True)
    windowed = signal * window_values
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(windowed))
    norm = np.linalg.norm(spectrum)
    if norm > 0:
        spectrum = spectrum / norm
    return FFTArtifacts(freqs=freqs, spectrum=spectrum)


def prepare_measurement(
    measurement_df: pd.DataFrame,
    analysis_cfg: Dict[str, Any],
) -> PreparedMeasurement:
    """Prepare measurement features shared across metrics."""

    detrend_cfg = analysis_cfg.get("detrending", {})
    peak_cfg = analysis_cfg.get("peak_detection", {})
    metrics_cfg = analysis_cfg.get("metrics", {})
    phase_cfg = metrics_cfg.get("phase_overlap", {})
    wavelength_range_cfg = analysis_cfg.get("wavelength_range", {})

    cutoff = float(detrend_cfg.get("default_cutoff_frequency", 0.008))
    order = int(detrend_cfg.get("filter_order", 3))
    prominence = float(peak_cfg.get("default_prominence", 0.0001))
    height = peak_cfg.get("min_height")
    
    # Filter to wavelength range of interest (default: 600-1120nm)
    wl_min = float(wavelength_range_cfg.get("min", 600))
    wl_max = float(wavelength_range_cfg.get("max", 1120))
    
    # Apply wavelength range filter
    df_filtered = measurement_df[
        (measurement_df["wavelength"] >= wl_min) & 
        (measurement_df["wavelength"] <= wl_max)
    ].reset_index(drop=True)

    wavelengths = df_filtered["wavelength"].to_numpy(dtype=float)
    reflectance = df_filtered["reflectance"].to_numpy(dtype=float)

    detrended = detrend_signal(wavelengths, reflectance, cutoff, order)
    peaks = detect_peaks(wavelengths, detrended, prominence=prominence, height=height)

    num_points = int(phase_cfg.get("resample_points", 1024))
    resample_grid, resampled_reflectance = resample_uniform_grid(
        wavelengths, reflectance, num_points=num_points
    )
    resampled_detrended = detrend_signal(resample_grid, resampled_reflectance, cutoff, order)
    fft_freqs, fft_values = compute_fft(
        resample_grid,
        resampled_detrended,
        window=phase_cfg.get("window", "hann"),
    )

    return PreparedMeasurement(
        wavelengths=wavelengths,
        reflectance=reflectance,
        detrended=detrended,
        peaks=peaks,
        resampled_wavelengths=resample_grid,
        resampled_reflectance=resampled_reflectance,
        resampled_detrended=resampled_detrended,
        fft_frequencies=fft_freqs,
        fft_spectrum=fft_values,
    )


def prepare_theoretical_spectrum(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    measurement: PreparedMeasurement,
    analysis_cfg: Dict[str, Any],
) -> PreparedTheoreticalSpectrum:
    """Compute cached features for a theoretical spectrum."""

    detrend_cfg = analysis_cfg.get("detrending", {})
    peak_cfg = analysis_cfg.get("peak_detection", {})
    metrics_cfg = analysis_cfg.get("metrics", {})
    phase_cfg = metrics_cfg.get("phase_overlap", {})
    wavelength_range_cfg = analysis_cfg.get("wavelength_range", {})

    cutoff = float(detrend_cfg.get("default_cutoff_frequency", 0.008))
    order = int(detrend_cfg.get("filter_order", 3))
    # Use lower prominence for theoretical to catch more peaks (theoretical peaks may be less prominent after detrending)
    theoretical_prominence = float(peak_cfg.get("theoretical_prominence", peak_cfg.get("default_prominence", 0.0001)))
    prominence = float(peak_cfg.get("default_prominence", 0.0001))
    height = peak_cfg.get("min_height")
    
    # Filter theoretical spectrum to wavelength range of interest (default: 600-1120nm)
    wl_min = float(wavelength_range_cfg.get("min", 600))
    wl_max = float(wavelength_range_cfg.get("max", 1120))
    mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    wavelengths = wavelengths[mask]
    reflectance = reflectance[mask]

    # Interpolate theoretical to measurement wavelengths
    interpolated = np.interp(measurement.wavelengths, wavelengths, reflectance)
    
    # Align theoretical to measured using linear regression (handles baseline + amplitude)
    # This makes the theoretical spectrum 'flow' with the measured spectrum visually
    aligned = align_spectrum_linear_regression(
        interpolated,
        measurement.reflectance,
        measurement.wavelengths,
        focus_min=wl_min,
        focus_max=wl_max,
    )
    
    detrended = detrend_signal(measurement.wavelengths, aligned, cutoff, order)
    
    # Try detecting peaks on both detrended and raw signals, then combine
    # Theoretical peaks might be more visible in raw signal or require lower threshold
    peaks_detrended = detect_peaks(measurement.wavelengths, detrended, prominence=theoretical_prominence, height=height)
    
    # Also try raw signal with even lower prominence (catch peaks that detrending might reduce)
    # Use very low threshold (10% of theoretical prominence) to catch all possible peaks
    peaks_raw = detect_peaks(measurement.wavelengths, aligned, prominence=theoretical_prominence * 0.1, height=height)
    
    # Combine peaks from both sources, removing duplicates (within 2nm tolerance)
    peak_list = []
    if len(peaks_detrended) > 0:
        peak_list.append(peaks_detrended)
    if len(peaks_raw) > 0:
        peak_list.append(peaks_raw)
    
    if len(peak_list) > 0:
        all_peaks = pd.concat(peak_list, ignore_index=True)
        # Remove duplicate peaks (peaks within 2nm of each other)
        all_peaks = all_peaks.sort_values("wavelength")
        keep_mask = np.ones(len(all_peaks), dtype=bool)
        for i in range(len(all_peaks)):
            if not keep_mask[i]:
                continue
            # Mark peaks within 2nm as duplicates (keep the one with higher prominence/amplitude)
            distances = np.abs(all_peaks["wavelength"].to_numpy() - all_peaks.iloc[i]["wavelength"])
            duplicates = np.where((distances < 2.0) & (distances > 1e-6))[0]
            if len(duplicates) > 0:
                # Keep the peak with highest amplitude among duplicates
                candidates = [i] + list(duplicates)
                best_idx = all_peaks.iloc[candidates]["amplitude"].idxmax()
                keep_mask[candidates] = False
                keep_mask[best_idx] = True
        
        peaks = all_peaks[keep_mask].reset_index(drop=True)
    else:
        # No peaks found - return empty dataframe with correct structure
        peaks = pd.DataFrame(columns=["wavelength", "amplitude", "prominence"])

    # Also align resampled spectrum for FFT analysis consistency
    resampled_interp = np.interp(measurement.resampled_wavelengths, wavelengths, reflectance)
    resampled = align_spectrum_linear_regression(
        resampled_interp,
        measurement.resampled_reflectance,
        measurement.resampled_wavelengths,
        focus_min=wl_min,
        focus_max=wl_max,
    )
    resampled_detrended = detrend_signal(
        measurement.resampled_wavelengths, resampled, cutoff, order
    )
    _, fft_values = compute_fft(
        measurement.resampled_wavelengths,
        resampled_detrended,
        window=phase_cfg.get("window", "hann"),
    )

    return PreparedTheoreticalSpectrum(
        aligned_reflectance=aligned,
        detrended=detrended,
        peaks=peaks,
        resampled_reflectance=resampled,
        resampled_detrended=resampled_detrended,
        fft_spectrum=fft_values,
    )


def align_spectrum_linear_regression(
    theoretical: np.ndarray,
    measured: np.ndarray,
    wavelengths: np.ndarray,
    focus_min: float = 600.0,
    focus_max: float = 1120.0,
) -> np.ndarray:
    """
    Align theoretical spectrum to measured using linear regression.
    
    Fits: theoretical_aligned = a * theoretical + b
    This handles both amplitude AND baseline differences, making the
    theoretical spectrum 'flow' with the measured spectrum visually.
    
    Args:
        theoretical: Theoretical reflectance values
        measured: Measured reflectance values (same length as theoretical)
        wavelengths: Wavelength array (same length as theoretical/measured)
        focus_min: Minimum wavelength for focus region (nm)
        focus_max: Maximum wavelength for focus region (nm)
        
    Returns:
        Aligned theoretical spectrum (same length as input)
    """
    mask = (wavelengths >= focus_min) & (wavelengths <= focus_max)
    
    if mask.sum() < 10:
        # Fallback to simple mean scaling if insufficient data
        scale = np.mean(measured) / np.mean(theoretical) if np.mean(theoretical) > 0 else 1.0
        return theoretical * scale
    
    meas_fit = measured[mask]
    theo_fit = theoretical[mask]
    
    if np.std(theo_fit) < 1e-10:
        return theoretical
    
    # Linear regression: measured ≈ a * theoretical + b
    design_matrix = np.vstack([theo_fit, np.ones_like(theo_fit)]).T
    try:
        coefficients, _, _, _ = np.linalg.lstsq(design_matrix, meas_fit, rcond=None)
        scale_factor, offset = coefficients
        
        # Apply transformation to full spectrum
        aligned = scale_factor * theoretical + offset
        
        # Ensure non-negative reflectance
        return np.clip(aligned, 0, None)
    except Exception:
        # Fallback to simple scaling
        scale = np.mean(measured) / np.mean(theoretical) if np.mean(theoretical) > 0 else 1.0
        return theoretical * scale


def align_spectrum_proportional(
    theoretical: np.ndarray,
    measured: np.ndarray,
    wavelengths: np.ndarray,
    focus_min: float = 600.0,
    focus_max: float = 1120.0,
) -> np.ndarray:
    """
    Scale theoretical spectrum using optimal least-squares proportional scaling.
    
    Minimizes ||measured - scale * theoretical||^2 in the focus region.
    Simpler than linear regression but doesn't correct baseline offset.
    
    Args:
        theoretical: Theoretical reflectance values
        measured: Measured reflectance values (same length as theoretical)
        wavelengths: Wavelength array (same length as theoretical/measured)
        focus_min: Minimum wavelength for focus region (nm)
        focus_max: Maximum wavelength for focus region (nm)
        
    Returns:
        Scaled theoretical spectrum
    """
    mask = (wavelengths >= focus_min) & (wavelengths <= focus_max)
    
    if mask.sum() < 10:
        scale = np.mean(measured) / np.mean(theoretical) if np.mean(theoretical) > 0 else 1.0
        return theoretical * scale
    
    meas_focus = measured[mask]
    theo_focus = theoretical[mask]
    
    if np.std(theo_focus) < 1e-10:
        return theoretical
    
    # Optimal scale: scale = (measured · theoretical) / (theoretical · theoretical)
    scale = np.dot(meas_focus, theo_focus) / np.dot(theo_focus, theo_focus)
    return theoretical * scale


__all__ = [
    "PreparedMeasurement",
    "PreparedTheoreticalSpectrum",
    "FFTArtifacts",
    "load_txt_file_enhanced",
    "load_measurement_spectrum",
    "load_measurement_files",
    "interpolate_measurement_to_theoretical",
    "calculate_fit_metrics",
    "detrend_signal",
    "boxcar_smooth",
    "gaussian_smooth",
    "detect_peaks",
    "detect_valleys",
    "resample_uniform_grid",
    "compute_fft",
    "compute_fft_artifacts",
    "prepare_measurement",
    "prepare_theoretical_spectrum",
    "align_spectrum_linear_regression",
    "align_spectrum_proportional",
]
