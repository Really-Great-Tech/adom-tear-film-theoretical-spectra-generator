"""Utilities for ingesting and preprocessing tear-film measurement spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import glob
import pathlib

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, get_window

WarnFn = Callable[[str], None]


def _default_warn(message: str) -> None:
    print(f"[measurement_utils] {message}")


def load_txt_file_enhanced(file_path: pathlib.Path | str) -> pd.DataFrame:
    """Load spectral data from a text file and return wavelength/reflectance columns."""
    file_path = pathlib.Path(file_path)
    data_started = False
    wavelengths: List[float] = []
    intensities: List[float] = []

    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">>>>>Begin Spectral Data<<<<<"):
                data_started = True
                continue

            parts = line.split()
            if len(parts) == 2:
                try:
                    wavelength, intensity = float(parts[0]), float(parts[1])
                except ValueError:
                    if data_started:
                        break
                    continue
                wavelengths.append(wavelength)
                intensities.append(intensity)
                data_started = True
            elif data_started:
                break

    return pd.DataFrame({"wavelength": wavelengths, "reflectance": intensities})


def detrend_signal(
    df: pd.DataFrame,
    cutoff_frequency: float = 0.01,
    filter_order: int = 3,
    *,
    warn: WarnFn = _default_warn,
) -> pd.DataFrame:
    """Apply a high-pass Butterworth filter to remove slow trends."""
    df = df.sort_values(by="wavelength").reset_index(drop=True)
    result = df.copy()

    sampling_interval = df["wavelength"].diff().mean()
    if sampling_interval is None or sampling_interval <= 0:
        raise ValueError("Invalid wavelength spacing for detrending")

    sampling_frequency = 1.0 / sampling_interval
    nyquist_freq = 0.5 * sampling_frequency
    normal_cutoff = min(cutoff_frequency / nyquist_freq, 0.99)

    try:
        b, a = butter(filter_order, normal_cutoff, btype="high", analog=False)
        detrended = filtfilt(b, a, df["reflectance"].to_numpy())
        result["detrended"] = detrended
    except Exception as exc:  # pragma: no cover - defensive fallback
        warn(f"Detrending failed ({exc}); using original signal.")
        result["detrended"] = result["reflectance"]

    return result


def detect_peaks(
    df: pd.DataFrame,
    column: str = "reflectance",
    prominence: float = 0.005,
    height: Optional[float] = None,
) -> pd.DataFrame:
    """Detect peaks in the specified column."""
    indices, properties = find_peaks(df[column], prominence=prominence, height=height)
    peaks_df = df.iloc[indices].copy()
    peaks_df["peak_prominence"] = properties.get("prominences", [0.0] * len(indices))
    return peaks_df.reset_index(drop=True)


def detect_valleys(
    df: pd.DataFrame,
    column: str = "reflectance",
    prominence: float = 0.005,
) -> pd.DataFrame:
    """Detect valleys by running peak detection on the inverted signal."""
    indices, properties = find_peaks(-df[column], prominence=prominence)
    valleys_df = df.iloc[indices].copy()
    valleys_df["valley_prominence"] = properties.get("prominences", [0.0] * len(indices))
    return valleys_df.reset_index(drop=True)


def load_measurement_files(
    measurements_dir: pathlib.Path,
    config: Dict[str, object],
    *,
    warn: WarnFn = _default_warn,
) -> Dict[str, pd.DataFrame]:
    """Load all measurement spectra matching the configured file pattern."""
    results: Dict[str, pd.DataFrame] = {}

    if not measurements_dir.exists():
        warn(f"Measurements directory not found: {measurements_dir}")
        return results

    meas_config = config.get("measurements", {}) or {}
    file_pattern = meas_config.get("file_pattern", "*.txt")  # type: ignore[arg-type]
    pattern_path = measurements_dir / file_pattern
    matches = glob.glob(str(pattern_path))

    if not matches:
        warn(f"No measurement files found matching {file_pattern} in {measurements_dir}")
        return results

    for file in sorted(matches):
        path = pathlib.Path(file)
        try:
            df = load_txt_file_enhanced(path)
            if df.empty:
                continue
            results[path.stem] = df.dropna()
        except Exception as exc:
            warn(f"Error loading {path}: {exc}")

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


@dataclass(slots=True)
class FFTArtifacts:
    freqs: np.ndarray
    spectrum: np.ndarray


def compute_fft(signal: np.ndarray, window: str = "hann") -> FFTArtifacts:
    """Compute normalized FFT magnitude/phase for a detrended signal."""
    window_values = get_window(window, len(signal), fftbins=True)
    windowed = signal * window_values
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(windowed))
    norm = np.linalg.norm(spectrum)
    if norm > 0:
        spectrum = spectrum / norm
    return FFTArtifacts(freqs=freqs, spectrum=spectrum)
