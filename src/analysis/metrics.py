"""Metric implementations for comparing theoretical and measured spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .measurement_utils import PreparedMeasurement, PreparedTheoreticalSpectrum


@dataclass
class MetricResult:
    score: float
    diagnostics: Dict[str, float]


@dataclass
class SpectrumScore:
    lipid_nm: Optional[float]
    aqueous_nm: Optional[float]
    roughness_A: Optional[float]
    scores: Dict[str, float]
    diagnostics: Dict[str, Dict[str, float]]

    @property
    def composite(self) -> float:
        return self.scores.get("composite", 0.0)

    @property
    def peak_count(self) -> float:
        return self.scores.get("peak_count", 0.0)

    @property
    def peak_delta(self) -> float:
        return self.scores.get("peak_delta", 0.0)

    @property
    def phase_overlap(self) -> float:
        return self.scores.get("phase_overlap", 0.0)

    def as_dict(self) -> Dict[str, float]:
        payload: Dict[str, float] = {}
        if self.lipid_nm is not None:
            payload["lipid_nm"] = float(self.lipid_nm)
        if self.aqueous_nm is not None:
            payload["aqueous_nm"] = float(self.aqueous_nm)
        if self.roughness_A is not None:
            payload["roughness_A"] = float(self.roughness_A)
        for key, value in self.scores.items():
            payload[f"{key}_score"] = float(value)
        for metric, diag in self.diagnostics.items():
            for diag_key, diag_val in diag.items():
                payload[f"{metric}_{diag_key}"] = float(diag_val)
        return payload


def _match_peaks(
    measurement_peaks: np.ndarray,
    theoretical_peaks: np.ndarray,
    tolerance_nm: float,
) -> Tuple[List[int], List[int], np.ndarray]:
    if measurement_peaks.size == 0 or theoretical_peaks.size == 0:
        return [], [], np.array([], dtype=float)

    matched_measurement: List[int] = []
    matched_theoretical: List[int] = []
    deltas: List[float] = []

    available = set(range(len(theoretical_peaks)))
    for meas_idx, meas_peak in enumerate(measurement_peaks):
        if not available:
            break
        candidates = sorted(available)
        distances = np.abs(theoretical_peaks[candidates] - meas_peak)
        best_local_idx = int(np.argmin(distances))
        theo_idx = candidates[best_local_idx]
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
    meas_peaks = measurement.peaks["wavelength"].to_numpy(dtype=float)
    theo_peaks = theoretical.peaks["wavelength"].to_numpy(dtype=float)

    matched_meas, matched_theo, deltas = _match_peaks(meas_peaks, theo_peaks, tolerance_nm)

    unmatched_measurement = len(meas_peaks) - len(matched_meas)
    unmatched_theoretical = len(theo_peaks) - len(matched_theo)
    if deltas.size == 0:
        mean_delta = 0.0
        if unmatched_measurement == 0 and unmatched_theoretical == 0:
            score = 1.0
        else:
            score = 0.0
    else:
        mean_delta = float(np.mean(deltas))
        score = float(np.exp(-mean_delta / max(tau_nm, 1e-6)))

    penalty = penalty_unpaired * float(unmatched_measurement + unmatched_theoretical)
    score = max(0.0, min(1.0, score - penalty))

    diagnostics = {
        "matched_pairs": float(len(matched_meas)),
        "mean_delta_nm": mean_delta,
        "unpaired_measurement": float(unmatched_measurement),
        "unpaired_theoretical": float(unmatched_theoretical),
    }
    return MetricResult(score=score, diagnostics=diagnostics)


def phase_overlap_score(
    measurement: PreparedMeasurement,
    theoretical: PreparedTheoreticalSpectrum,
) -> MetricResult:
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


def score_spectrum(
    measurement: PreparedMeasurement,
    theoretical: PreparedTheoreticalSpectrum,
    metrics_cfg: Dict[str, Dict[str, float]],
    *,
    lipid_nm: Optional[float] = None,
    aqueous_nm: Optional[float] = None,
    roughness_A: Optional[float] = None,
) -> SpectrumScore:
    peak_count_cfg = metrics_cfg.get("peak_count", {})
    peak_delta_cfg = metrics_cfg.get("peak_delta", {})
    weights_cfg = metrics_cfg.get("composite", {}).get("weights", {})

    count_result = peak_count_score(
        measurement,
        theoretical,
        tolerance_nm=float(peak_count_cfg.get("wavelength_tolerance_nm", 5.0)),
    )
    delta_result = peak_delta_score(
        measurement,
        theoretical,
        tolerance_nm=float(peak_delta_cfg.get("tolerance_nm", 5.0)),
        tau_nm=float(peak_delta_cfg.get("tau_nm", 15.0)),
        penalty_unpaired=float(peak_delta_cfg.get("penalty_unpaired", 0.05)),
    )
    phase_result = phase_overlap_score(measurement, theoretical)

    component_scores = {
        "peak_count": count_result.score,
        "peak_delta": delta_result.score,
        "phase_overlap": phase_result.score,
    }
    composite = composite_score(component_scores, weights_cfg)
    component_scores["composite"] = composite

    diagnostics = {
        "peak_count": count_result.diagnostics,
        "peak_delta": delta_result.diagnostics,
        "phase_overlap": phase_result.diagnostics,
    }

    return SpectrumScore(
        lipid_nm=lipid_nm,
        aqueous_nm=aqueous_nm,
        roughness_A=roughness_A,
        scores=component_scores,
        diagnostics=diagnostics,
    )


__all__ = [
    "MetricResult",
    "SpectrumScore",
    "peak_count_score",
    "peak_delta_score",
    "phase_overlap_score",
    "composite_score",
    "score_spectrum",
]
