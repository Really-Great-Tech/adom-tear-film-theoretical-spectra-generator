"""Scoring utilities for comparing theoretical and measured spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import math
import numpy as np
import pandas as pd

from .measurement_utils import (
    FFTArtifacts,
    compute_fft,
    detect_peaks,
    detrend_signal,
    interpolate_measurement_to_theoretical,
)


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values()) or 1.0
    return {key: value / total for key, value in weights.items()}


@dataclass(slots=True)
class MeasurementArtifacts:
    wavelengths: np.ndarray
    original_df: pd.DataFrame
    measured_df: pd.DataFrame
    detrended_df: pd.DataFrame
    peaks_df: pd.DataFrame
    fft: FFTArtifacts
    cutoff_frequency: float
    filter_order: int
    prominence: float


@dataclass(slots=True)
class MetricScores:
    lipid_nm: float
    aqueous_nm: float
    roughness_A: float
    peak_count_score: float
    peak_delta_score: float
    phase_overlap_score: float
    composite_score: float
    matched_peaks: int
    unmatched_measured: int
    unmatched_theoretical: int


def prepare_measurement_artifacts(
    measured_df: pd.DataFrame,
    *,
    wavelengths: np.ndarray,
    analysis_config: Dict[str, object],
) -> MeasurementArtifacts:
    detrending_cfg = analysis_config.get("detrending", {}) or {}
    peak_cfg = analysis_config.get("peak_detection", {}) or {}

    cutoff = float(detrending_cfg.get("default_cutoff_frequency", 0.01))
    order = int(detrending_cfg.get("filter_order", 3))
    prominence = float(peak_cfg.get("default_prominence", 0.005))

    resampled_reflectance = interpolate_measurement_to_theoretical(
        measured_df, wavelengths
    )
    resampled_df = pd.DataFrame(
        {"wavelength": wavelengths, "reflectance": resampled_reflectance}
    )

    detrended_df = detrend_signal(
        resampled_df,
        cutoff_frequency=cutoff,
        filter_order=order,
    )
    peaks_df = detect_peaks(
        detrended_df,
        column="detrended",
        prominence=prominence,
    )
    fft_artifacts = compute_fft(detrended_df["detrended"].to_numpy())

    return MeasurementArtifacts(
        wavelengths=wavelengths,
        original_df=measured_df,
        measured_df=resampled_df,
        detrended_df=detrended_df,
        peaks_df=peaks_df,
        fft=fft_artifacts,
        cutoff_frequency=cutoff,
        filter_order=order,
        prominence=prominence,
    )


def score_spectrum(
    theoretical_wavelengths: np.ndarray,
    theoretical_reflectance: np.ndarray,
    *,
    measurement: MeasurementArtifacts,
    metrics_config: Dict[str, object],
    lipid_nm: float,
    aqueous_nm: float,
    roughness_A: float,
) -> MetricScores:
    detrended_df = detrend_signal(
        pd.DataFrame(
            {
                "wavelength": theoretical_wavelengths,
                "reflectance": theoretical_reflectance,
            }
        ),
        cutoff_frequency=measurement.cutoff_frequency,
        filter_order=measurement.filter_order,
    )

    peaks_df = detect_peaks(
        detrended_df,
        column="detrended",
        prominence=measurement.prominence,
    )

    peak_matching_cfg = metrics_config.get("peak_matching", {}) or {}
    peak_delta_cfg = metrics_config.get("peak_delta", {}) or {}
    phase_cfg = metrics_config.get("phase", {}) or {}
    weight_cfg = _normalize_weights(
        metrics_config.get(
            "weights",
            {"peak_count": 0.4, "peak_delta": 0.4, "phase": 0.2},
        )
    )

    pairs, unmatched_meas, unmatched_theor = _match_peaks(
        measurement.peaks_df,
        peaks_df,
        tolerance_nm=float(peak_matching_cfg.get("wavelength_tolerance_nm", 5.0)),
    )

    peak_count = _peak_count_score(
        len(measurement.peaks_df),
        len(pairs),
        unmatched_theor,
    )

    peak_delta = _peak_delta_score(
        pairs,
        tau=float(peak_delta_cfg.get("wavelength_tau", 5.0)),
        unmatched_penalty=float(peak_delta_cfg.get("unmatched_penalty", 0.1)),
        unmatched_total=unmatched_meas + unmatched_theor,
    )

    phase_score = _phase_overlap_score(
        measurement.fft,
        detrended_df["detrended"].to_numpy(),
        window=phase_cfg.get("window", "hann"),
    )

    composite = (
        weight_cfg.get("peak_count", 0.0) * peak_count
        + weight_cfg.get("peak_delta", 0.0) * peak_delta
        + weight_cfg.get("phase", 0.0) * phase_score
    )

    return MetricScores(
        lipid_nm=lipid_nm,
        aqueous_nm=aqueous_nm,
        roughness_A=roughness_A,
        peak_count_score=peak_count,
        peak_delta_score=peak_delta,
        phase_overlap_score=phase_score,
        composite_score=composite,
        matched_peaks=len(pairs),
        unmatched_measured=unmatched_meas,
        unmatched_theoretical=unmatched_theor,
    )


def _match_peaks(
    measured_peaks: pd.DataFrame,
    theoretical_peaks: pd.DataFrame,
    *,
    tolerance_nm: float,
) -> Tuple[List[Tuple[int, int, float]], int, int]:
    measured_wl = measured_peaks["wavelength"].to_numpy()
    theoretical_wl = theoretical_peaks["wavelength"].to_numpy()

    if len(measured_wl) == 0 or len(theoretical_wl) == 0:
        return [], len(measured_wl), len(theoretical_wl)

    used = np.zeros(len(theoretical_wl), dtype=bool)
    pairs: List[Tuple[int, int, float]] = []

    for idx_meas, wl in enumerate(measured_wl):
        diffs = np.abs(theoretical_wl - wl)
        best_idx = int(np.argmin(diffs))
        if diffs[best_idx] <= tolerance_nm and not used[best_idx]:
            used[best_idx] = True
            pairs.append((idx_meas, best_idx, float(diffs[best_idx])))

    unmatched_theoretical = int((~used).sum())
    unmatched_measured = len(measured_wl) - len(pairs)
    return pairs, unmatched_measured, unmatched_theoretical


def _peak_count_score(
    measured_count: int,
    matched_count: int,
    unmatched_theoretical: int,
) -> float:
    if measured_count == 0:
        return 1.0 if matched_count == 0 else 0.0

    term = abs(measured_count - matched_count) / measured_count
    theoretical_penalty = unmatched_theoretical / max(measured_count, 1)
    return max(0.0, 1.0 - term - theoretical_penalty)


def _peak_delta_score(
    pairs: Iterable[Tuple[int, int, float]],
    *,
    tau: float,
    unmatched_penalty: float,
    unmatched_total: int,
) -> float:
    deltas = [abs(delta) for *_, delta in pairs]
    if not deltas:
        return 0.0

    mean_delta = sum(deltas) / len(deltas)
    score = math.exp(-mean_delta / max(tau, 1e-6))
    penalty = unmatched_penalty * unmatched_total
    return max(0.0, score * math.exp(-penalty))


def _phase_overlap_score(
    measurement_fft: FFTArtifacts,
    theoretical_signal: np.ndarray,
    *,
    window: str = "hann",
) -> float:
    theor_fft = compute_fft(theoretical_signal, window=window)
    numerator = np.vdot(theor_fft.spectrum, measurement_fft.spectrum)
    denominator = np.linalg.norm(theor_fft.spectrum) * np.linalg.norm(
        measurement_fft.spectrum
    )
    if denominator == 0:
        return 0.0
    return float(abs(numerator) / denominator)
