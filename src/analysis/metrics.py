"""Metric implementations for comparing theoretical and measured spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .measurement_utils import PreparedMeasurement, PreparedTheoreticalSpectrum


@dataclass
class MetricResult:
    score: float
    diagnostics: Dict[str, float]


def _match_peaks(
    measurement_peaks: np.ndarray,
    theoretical_peaks: np.ndarray,
    tolerance_nm: float,
) -> Tuple[List[int], List[int], np.ndarray]:
    """Greedy nearest-neighbour matching for peak positions."""

    if measurement_peaks.size == 0 or theoretical_peaks.size == 0:
        return [], [], np.array([], dtype=float)

    matched_measurement: List[int] = []
    matched_theoretical: List[int] = []
    deltas: List[float] = []

    available = set(range(len(theoretical_peaks)))
    for meas_idx, meas_peak in enumerate(measurement_peaks):
        if not available:
            break
        available_indices = sorted(available)
        distances = np.abs(theoretical_peaks[available_indices] - meas_peak)
        best_local_idx = int(np.argmin(distances))
        theo_idx = available_indices[best_local_idx]
        delta = distances[best_local_idx]
        if delta <= tolerance_nm:
            matched_measurement.append(meas_idx)
            matched_theoretical.append(theo_idx)
            deltas.append(float(delta))
            available.remove(theo_idx)

    return matched_measurement, matched_theoretical, np.asarray(deltas, dtype=float)


def peak_count_score(
    measurement: PreparedMeasurement,
    theoretical: PreparedTheoreticalSpectrum,
    *,
    tolerance_nm: float,
) -> MetricResult:
    """Score similarity based on peak counts within a tolerance window."""

    meas_peaks = measurement.peaks["wavelength"].to_numpy(dtype=float)
    theo_peaks = theoretical.peaks["wavelength"].to_numpy(dtype=float)

    matched_meas, matched_theo, _ = _match_peaks(meas_peaks, theo_peaks, tolerance_nm)

    meas_count = len(meas_peaks)
    matched_count = len(matched_meas)

    if meas_count == 0:
        score = 1.0 if len(theo_peaks) == 0 else 0.0
    else:
        score = 1.0 - abs(meas_count - matched_count) / float(meas_count)
        score = max(0.0, min(1.0, score))

    diagnostics = {
        "measurement_peaks": float(meas_count),
        "theoretical_peaks": float(len(theo_peaks)),
        "matched_peaks": float(matched_count),
    }
    return MetricResult(score=score, diagnostics=diagnostics)


def peak_delta_score(
    measurement: PreparedMeasurement,
    theoretical: PreparedTheoreticalSpectrum,
    *,
    tolerance_nm: float,
    tau_nm: float,
    penalty_unpaired: float,
) -> MetricResult:
    """Score similarity based on matched peak wavelength deltas."""

    meas_peaks = measurement.peaks["wavelength"].to_numpy(dtype=float)
    theo_peaks = theoretical.peaks["wavelength"].to_numpy(dtype=float)

    matched_meas, matched_theo, deltas = _match_peaks(meas_peaks, theo_peaks, tolerance_nm)

    if deltas.size == 0:
        score = 0.0
    else:
        mean_delta = float(np.mean(deltas))
        score = float(np.exp(-mean_delta / tau_nm))

    unmatched_measurement = len(meas_peaks) - len(matched_meas)
    unmatched_theoretical = len(theo_peaks) - len(matched_theo)
    penalty = penalty_unpaired * float(unmatched_measurement + unmatched_theoretical)
    score = max(0.0, min(1.0, score - penalty))

    diagnostics = {
        "matched_pairs": float(len(matched_meas)),
        "mean_delta_nm": float(np.mean(deltas)) if deltas.size else 0.0,
        "unpaired_measurement": float(unmatched_measurement),
        "unpaired_theoretical": float(unmatched_theoretical),
    }
    return MetricResult(score=score, diagnostics=diagnostics)


def phase_overlap_score(
    measurement: PreparedMeasurement,
    theoretical: PreparedTheoreticalSpectrum,
) -> MetricResult:
    """Compare FFT overlap between measured and theoretical spectra."""

    reference_fft = measurement.fft_spectrum
    candidate_fft = theoretical.fft_spectrum
    numerator = np.vdot(candidate_fft, reference_fft)
    denom = float(np.linalg.norm(candidate_fft) * np.linalg.norm(reference_fft))
    score = float(abs(numerator) / denom) if denom else 0.0
    diagnostics = {
        "coherence": float(abs(numerator)),
        "norm_reference": float(np.linalg.norm(reference_fft)),
        "norm_candidate": float(np.linalg.norm(candidate_fft)),
    }
    return MetricResult(score=score, diagnostics=diagnostics)


def composite_score(component_scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """Combine component metric scores with configurable weights."""

    total_weight = float(sum(weights.values()))
    if total_weight <= 0:
        if component_scores:
            return float(np.mean(list(component_scores.values())))
        return 0.0

    combined = 0.0
    for key, score in component_scores.items():
        weight = weights.get(key, 0.0)
        combined += weight * score
    return combined / total_weight
