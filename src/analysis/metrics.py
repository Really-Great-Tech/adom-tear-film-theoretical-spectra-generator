"""Metric implementations for comparing theoretical and measured spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .measurement_utils import (
    PreparedMeasurement,
    PreparedTheoreticalSpectrum,
    calculate_fit_metrics,
)


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
    """
    Peak matching using measurement-order matching to maximize cardinality.
    
    Processes measurement peaks sequentially, matching each to the closest
    available theoretical peak within tolerance. This avoids the cardinality
    loss that greedy-by-delta can cause (where picking smallest deltas first
    can block matches and reduce total pairs).
    
    Example: measurement=[0,4], theoretical=[3,8], tolerance=5
      - Greedy-by-delta picks (m1,t0,delta=1) first, blocking m0 → 1 match
      - Measurement-order: m0→t0 (delta=3), m1→t1 (delta=4) → 2 matches
    """
    if measurement_peaks.size == 0 or theoretical_peaks.size == 0:
        return [], [], np.array([], dtype=float)

    matched_measurement: List[int] = []
    matched_theoretical: List[int] = []
    deltas: List[float] = []
    used_theo: set = set()

    for meas_idx, meas_peak in enumerate(measurement_peaks):
        best_theo_idx: Optional[int] = None
        best_delta: float = float('inf')

        for theo_idx, theo_peak in enumerate(theoretical_peaks):
            if theo_idx in used_theo:
                continue
            delta = abs(theo_peak - meas_peak)
            if delta <= tolerance_nm and delta < best_delta:
                best_theo_idx = theo_idx
                best_delta = delta

        if best_theo_idx is not None:
            matched_measurement.append(meas_idx)
            matched_theoretical.append(best_theo_idx)
            deltas.append(best_delta)
            used_theo.add(best_theo_idx)

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


def residual_score(
    measurement: PreparedMeasurement,
    theoretical: PreparedTheoreticalSpectrum,
    *,
    tau_rmse: float,
    max_rmse: Optional[float] = None,
) -> MetricResult:
    # Use RAW (non-detrended) spectra for residual calculation - visual quality depends on raw alignment
    # Detrended spectra remove baseline, which can hide misalignment issues
    fit_metrics = calculate_fit_metrics(measurement.reflectance, theoretical.aligned_reflectance)
    rmse = float(fit_metrics.get("RMSE", 0.0))
    r2 = float(fit_metrics.get("R²", 0.0))
    tau = max(float(tau_rmse), 1e-9)
    
    # Base score from RMSE (lower RMSE = higher score)
    rmse_score = float(np.exp(-rmse / tau))
    if max_rmse is not None and rmse >= float(max_rmse):
        rmse_score = 0.0
    
    # Incorporate R²: Negative R² doesn't necessarily mean bad visual fit
    # Solution: Ignore R² when negative (use RMSE only), since R² is misleading for visual quality
    if r2 < 0:
        # Negative R² - ignore it, use RMSE score only
        score = rmse_score
    else:
        # Positive R² - combine RMSE and R² scores
        r2_score = max(0.3, min(1.0, 0.3 + 0.7 * r2))
        score = 0.6 * rmse_score + 0.4 * (rmse_score * r2_score)
    
    diagnostics = {
        "rmse": rmse,
        "mae": float(fit_metrics.get("MAE", 0.0)),
        "r2": r2,
        "mape_pct": float(fit_metrics.get("MAPE (%)", 0.0)),
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


def measurement_quality_score(
    measurement: PreparedMeasurement,
    *,
    min_peaks: Optional[int] = None,
    min_signal_amplitude: Optional[float] = None,
    min_wavelength_span_nm: Optional[float] = None,
) -> Tuple[MetricResult, List[str]]:
    peak_count = float(len(measurement.peaks))
    signal_amplitude = float(np.ptp(measurement.detrended)) if measurement.detrended.size else 0.0
    if measurement.wavelengths.size:
        span = float(measurement.wavelengths.max() - measurement.wavelengths.min())
    else:
        span = 0.0

    diagnostics: Dict[str, float] = {
        "peak_count": peak_count,
        "signal_amplitude": signal_amplitude,
        "wavelength_span_nm": span,
    }

    failures: List[str] = []
    checks = 0

    if min_peaks is not None:
        checks += 1
        diagnostics["min_peaks"] = float(min_peaks)
        if peak_count < float(min_peaks):
            failures.append("min_peaks")

    if min_signal_amplitude is not None:
        checks += 1
        diagnostics["min_signal_amplitude"] = float(min_signal_amplitude)
        if signal_amplitude < float(min_signal_amplitude):
            failures.append("min_signal_amplitude")

    if min_wavelength_span_nm is not None:
        checks += 1
        diagnostics["min_wavelength_span_nm"] = float(min_wavelength_span_nm)
        if span < float(min_wavelength_span_nm):
            failures.append("min_wavelength_span_nm")

    if checks == 0:
        score = 1.0
    else:
        score = float((checks - len(failures)) / checks)

    diagnostics["failed_checks"] = float(len(failures))
    return MetricResult(score=score, diagnostics=diagnostics), failures


def temporal_continuity_score(
    current_params: Dict[str, Optional[float]],
    previous_params: Optional[Dict[str, float]],
    *,
    tau_lipid_nm: float,
    tau_aqueous_nm: float,
    tau_roughness_A: float,
) -> MetricResult:
    if not previous_params:
        diagnostics = {
            "lipid_jump_nm": 0.0,
            "aqueous_jump_nm": 0.0,
            "roughness_jump_A": 0.0,
        }
        return MetricResult(score=1.0, diagnostics=diagnostics)

    lipid_jump = abs(float(current_params.get("lipid_nm") or 0.0) - previous_params.get("lipid_nm", 0.0))
    aqueous_jump = abs(
        float(current_params.get("aqueous_nm") or 0.0) - previous_params.get("aqueous_nm", 0.0)
    )
    roughness_jump = abs(
        float(current_params.get("roughness_A") or 0.0) - previous_params.get("roughness_A", 0.0)
    )

    tau_lipid = max(float(tau_lipid_nm), 1e-6)
    tau_aqueous = max(float(tau_aqueous_nm), 1e-6)
    tau_roughness = max(float(tau_roughness_A), 1e-6)

    penalty = (lipid_jump / tau_lipid) + (aqueous_jump / tau_aqueous) + (roughness_jump / tau_roughness)
    score = float(np.exp(-penalty))
    diagnostics = {
        "lipid_jump_nm": float(lipid_jump),
        "aqueous_jump_nm": float(aqueous_jump),
        "roughness_jump_A": float(roughness_jump),
    }
    return MetricResult(score=score, diagnostics=diagnostics)


def score_spectrum(
    measurement: PreparedMeasurement,
    theoretical: PreparedTheoreticalSpectrum,
    metrics_cfg: Dict[str, Dict[str, float]],
    *,
    lipid_nm: Optional[float] = None,
    aqueous_nm: Optional[float] = None,
    roughness_A: Optional[float] = None,
    measurement_quality: Optional[MetricResult] = None,
    previous_params: Optional[Dict[str, float]] = None,
) -> SpectrumScore:
    peak_count_cfg = metrics_cfg.get("peak_count", {})
    peak_delta_cfg = metrics_cfg.get("peak_delta", {})
    residual_cfg = metrics_cfg.get("residual", {})
    temporal_cfg = metrics_cfg.get("temporal_continuity", {})
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

    component_scores: Dict[str, float] = {
        "peak_count": count_result.score,
        "peak_delta": delta_result.score,
        "phase_overlap": phase_result.score,
    }
    diagnostics: Dict[str, Dict[str, float]] = {
        "peak_count": count_result.diagnostics,
        "peak_delta": delta_result.diagnostics,
        "phase_overlap": phase_result.diagnostics,
    }

    if residual_cfg:
        residual_result = residual_score(
            measurement,
            theoretical,
            tau_rmse=float(residual_cfg.get("tau_rmse", 0.02)),
            max_rmse=residual_cfg.get("max_rmse"),
        )
        component_scores["residual"] = residual_result.score
        diagnostics["residual"] = residual_result.diagnostics

    if measurement_quality is not None:
        component_scores["quality"] = measurement_quality.score
        diagnostics["quality"] = measurement_quality.diagnostics

    if temporal_cfg.get("enabled") and previous_params is not None:
        temporal_result = temporal_continuity_score(
            {"lipid_nm": lipid_nm, "aqueous_nm": aqueous_nm, "roughness_A": roughness_A},
            previous_params,
            tau_lipid_nm=float(temporal_cfg.get("tau_lipid_nm", 10.0)),
            tau_aqueous_nm=float(temporal_cfg.get("tau_aqueous_nm", 10.0)),
            tau_roughness_A=float(temporal_cfg.get("tau_roughness_A", 50.0)),
        )
        component_scores["temporal_continuity"] = temporal_result.score
        diagnostics["temporal_continuity"] = temporal_result.diagnostics

    composite = composite_score(component_scores, weights_cfg)
    component_scores["composite"] = composite

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
    "residual_score",
    "measurement_quality_score",
    "temporal_continuity_score",
    "composite_score",
    "score_spectrum",
]
