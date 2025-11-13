"""Shared measurement preprocessing utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


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


def load_measurement_spectrum(path: Path | str, measurement_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load a measurement spectrum into a dataframe.

    The loader supports whitespace-delimited text files with optional headers,
    CSV, and XLSX files. A robust fallback parser is included for legacy TXT
    exports that embed metadata blocks.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Measurement file not found: {path}")

    fmt = (measurement_cfg.get("format") or path.suffix.lstrip(".")).lower()
    header_rows = measurement_cfg.get("header_rows")
    skiprows = int(header_rows) if header_rows is not None else None
    col_indices = [
        int(measurement_cfg.get("wavelength_column", 0)),
        int(measurement_cfg.get("reflectance_column", 1)),
    ]

    if fmt in {"txt", "dat", "tsv"}:
        df = _load_text_measurement(path)
    elif fmt == "csv":
        read_kwargs: Dict[str, Any] = {
            "comment": "#",
            "usecols": col_indices,
            "skiprows": skiprows,
            "header": None,
        }
        delimiter = measurement_cfg.get("delimiter")
        if delimiter:
            read_kwargs["sep"] = delimiter
        df = pd.read_csv(path, **read_kwargs)
    elif fmt in {"xls", "xlsx"}:
        df = pd.read_excel(
            path,
            usecols=col_indices,
            skiprows=skiprows,
            header=None,
        )
    else:
        raise ValueError(f"Unsupported measurement format: {fmt}")

    df = df.rename(columns={df.columns[0]: "wavelength", df.columns[1]: "reflectance"})
    df["wavelength"] = pd.to_numeric(df["wavelength"], errors="coerce")
    df["reflectance"] = pd.to_numeric(df["reflectance"], errors="coerce")
    df = df.dropna().sort_values("wavelength").reset_index(drop=True)
    return df


def _load_text_measurement(path: Path) -> pd.DataFrame:
    wavelengths: list[float] = []
    intensities: list[float] = []
    data_started = False

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line in reader:
            if not line:
                continue
            if len(line) == 1:
                parts = line[0].split()
            else:
                parts = [col for token in line for col in token.split()]

            if len(parts) < 2:
                if data_started:
                    break
                continue

            try:
                wavelength = float(parts[0])
                intensity = float(parts[1])
            except ValueError:
                if data_started:
                    break
                continue

            wavelengths.append(wavelength)
            intensities.append(intensity)
            data_started = True

    if not wavelengths:
        raise ValueError(f"No spectral data found in {path}")

    return pd.DataFrame({"wavelength": wavelengths, "reflectance": intensities})


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
        # Fall back to original signal if filtering fails (e.g., insufficient data)
        return reflectance.copy()


def detect_peaks(
    wavelengths: np.ndarray,
    signal: np.ndarray,
    *,
    prominence: float,
    height: Optional[float] = None,
) -> pd.DataFrame:
    """Detect peaks and return a dataframe of peak positions."""

    peak_indices, properties = find_peaks(signal, prominence=prominence, height=height)
    peaks_df = pd.DataFrame({
        "wavelength": wavelengths[peak_indices],
        "amplitude": signal[peak_indices],
        "prominence": properties.get("prominences", np.zeros_like(peak_indices, dtype=float)),
    })
    return peaks_df.reset_index(drop=True)


def detect_valleys(
    wavelengths: np.ndarray,
    signal: np.ndarray,
    *,
    prominence: float,
) -> pd.DataFrame:
    """Detect valleys by reusing the peak detector on the inverted signal."""

    valley_indices, properties = find_peaks(-signal, prominence=prominence)
    valleys_df = pd.DataFrame({
        "wavelength": wavelengths[valley_indices],
        "amplitude": signal[valley_indices],
        "prominence": properties.get("prominences", np.zeros_like(valley_indices, dtype=float)),
    })
    return valleys_df.reset_index(drop=True)


def resample_uniform_grid(
    wavelengths: np.ndarray,
    values: np.ndarray,
    *,
    num_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample onto a uniform grid for FFT analysis."""

    wl_min = float(wavelengths.min())
    wl_max = float(wavelengths.max())
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


def prepare_measurement(
    measurement_df: pd.DataFrame,
    analysis_cfg: Dict[str, Any],
) -> PreparedMeasurement:
    """Prepare measurement features shared across metrics."""

    detrend_cfg = analysis_cfg.get("detrending", {})
    peak_cfg = analysis_cfg.get("peak_detection", {})
    metrics_cfg = analysis_cfg.get("metrics", {})
    phase_cfg = metrics_cfg.get("phase_overlap", {})

    cutoff = float(detrend_cfg.get("default_cutoff_frequency", 0.01))
    order = int(detrend_cfg.get("filter_order", 3))
    prominence = float(peak_cfg.get("default_prominence", 0.005))
    height = peak_cfg.get("min_height")

    wavelengths = measurement_df["wavelength"].to_numpy(dtype=float)
    reflectance = measurement_df["reflectance"].to_numpy(dtype=float)

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

    cutoff = float(detrend_cfg.get("default_cutoff_frequency", 0.01))
    order = int(detrend_cfg.get("filter_order", 3))
    prominence = float(peak_cfg.get("default_prominence", 0.005))
    height = peak_cfg.get("min_height")

    aligned = np.interp(measurement.wavelengths, wavelengths, reflectance)
    detrended = detrend_signal(measurement.wavelengths, aligned, cutoff, order)
    peaks = detect_peaks(measurement.wavelengths, detrended, prominence=prominence, height=height)

    resampled = np.interp(measurement.resampled_wavelengths, wavelengths, reflectance)
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
