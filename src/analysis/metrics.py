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
    max_allowed_excess: int = 2,
    min_coverage_ratio: float = 0.7,
    excess_penalty_per_peak: float = 0.2,
    coverage_penalty_factor: float = 0.5,
) -> MetricResult:
    """
    Score based on peak count matching with penalties for excess/missing peaks.
    
    Based on PyElli analysis:
    - Theoretical peaks should be close to measured (within +2/-2)
    - Too many theoretical peaks is a deal-breaker (severe penalty)
    - Too few peaks (poor coverage) also penalized
    
    Args:
        measurement: Prepared measurement spectrum
        theoretical: Prepared theoretical spectrum
        tolerance_nm: Peak matching tolerance in nm
        max_allowed_excess: Max extra theoretical peaks before severe penalty
        min_coverage_ratio: Minimum theo_peaks/meas_peaks ratio
        excess_penalty_per_peak: Penalty per excess peak over limit
        coverage_penalty_factor: Multiplier for coverage penalty
    """
    meas_peaks = measurement.peaks["wavelength"].to_numpy(dtype=float)
    theo_peaks = theoretical.peaks["wavelength"].to_numpy(dtype=float)

    matched_meas, matched_theo, _ = _match_peaks(meas_peaks, theo_peaks, tolerance_nm)

    meas_count = len(meas_peaks)
    theo_count = len(theo_peaks)
    matched_count = len(matched_meas)

    if meas_count == 0:
        score = 1.0 if theo_count == 0 else 0.0
        peak_coverage = 1.0
        peak_excess = 0
    else:
        # Base score: ratio of matched to measured peaks
        score = matched_count / float(meas_count)
        score = max(0.0, min(1.0, score))
        
        # Calculate peak excess (theoretical - measured)
        peak_excess = theo_count - meas_count
        peak_coverage = theo_count / float(meas_count)
        
        # PENALTY 1: Too many theoretical peaks (from PyElli)
        # E.g., 13 theoretical vs 7 measured = 6 excess -> SEVERE penalty
        if peak_excess > max_allowed_excess:
            excess_over_limit = peak_excess - max_allowed_excess
            excess_penalty = excess_penalty_per_peak * excess_over_limit
            score = max(0.0, score - excess_penalty)
        
        # PENALTY 2: Too few theoretical peaks (poor coverage)
        if peak_coverage < min_coverage_ratio:
            coverage_penalty = (min_coverage_ratio - peak_coverage) * coverage_penalty_factor
            score = max(0.0, score - coverage_penalty)

    diagnostics = {
        "measurement_peaks": float(meas_count),
        "theoretical_peaks": float(theo_count),
        "matched_peaks": float(matched_count),
        "peak_excess": float(peak_excess),
        "peak_coverage": float(peak_coverage),
    }
    return MetricResult(score=score, diagnostics=diagnostics)


def peak_delta_score(
    measurement: PreparedMeasurement,
    theoretical: PreparedTheoreticalSpectrum,
    *,
    tolerance_nm: float,
    tau_nm: float,
    penalty_unpaired: float,
    extra_penalty_unmatched_measured: float = 0.02,
    use_soft_penalty: bool = True,
) -> MetricResult:
    """
    Score based on peak alignment quality (position delta between matched peaks).
    
    Uses soft multiplicative penalty instead of subtractive to avoid zeroing out
    candidates with good fit quality but imperfect peak matching.
    
    Args:
        measurement: Prepared measurement spectrum
        theoretical: Prepared theoretical spectrum
        tolerance_nm: Peak matching tolerance in nm
        tau_nm: Decay constant for delta scoring (lower = stricter)
        penalty_unpaired: Base penalty per unpaired peak (used in soft mode as dampening)
        extra_penalty_unmatched_measured: Extra penalty for unmatched measured peaks
        use_soft_penalty: If True, use multiplicative dampening; if False, use subtractive
    """
    meas_peaks = measurement.peaks["wavelength"].to_numpy(dtype=float)
    theo_peaks = theoretical.peaks["wavelength"].to_numpy(dtype=float)

    matched_meas, matched_theo, deltas = _match_peaks(meas_peaks, theo_peaks, tolerance_nm)

    unmatched_measurement = len(meas_peaks) - len(matched_meas)
    unmatched_theoretical = len(theo_peaks) - len(matched_theo)
    total_unmatched = unmatched_measurement + unmatched_theoretical
    
    if deltas.size == 0:
        mean_delta = 0.0
        if total_unmatched == 0:
            score = 1.0
        else:
            # No matches but have peaks - still give partial credit for having peaks
            score = 0.2 if use_soft_penalty else 0.0
    else:
        mean_delta = float(np.mean(deltas))
        score = float(np.exp(-mean_delta / max(tau_nm, 1e-6)))

    if use_soft_penalty:
        # Soft penalty: multiplicative dampening based on match ratio
        # This prevents good-fit-but-imperfect-peak-matching candidates from zeroing out
        total_peaks = max(len(meas_peaks), len(theo_peaks))
        if total_peaks > 0:
            match_ratio = len(matched_meas) / total_peaks
            # Dampening factor: ranges from 0.3 (no matches) to 1.0 (all matched)
            dampening = 0.3 + 0.7 * match_ratio
            score *= dampening
            
            # Extra dampening for unmatched measured peaks (more important)
            if unmatched_measurement > 0 and len(meas_peaks) > 0:
                meas_match_ratio = len(matched_meas) / len(meas_peaks)
                extra_dampening = 0.5 + 0.5 * meas_match_ratio
                score *= extra_dampening
    else:
        # Original subtractive penalty (legacy behavior)
        penalty = penalty_unpaired * float(total_unmatched)
        if unmatched_measurement > 0:
            penalty += extra_penalty_unmatched_measured * float(unmatched_measurement)
        score = max(0.0, min(1.0, score - penalty))

    diagnostics = {
        "matched_pairs": float(len(matched_meas)),
        "mean_delta_nm": mean_delta,
        "unpaired_measurement": float(unmatched_measurement),
        "unpaired_theoretical": float(unmatched_theoretical),
        "total_unmatched": float(total_unmatched),
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


def correlation_score(
    measurement: PreparedMeasurement,
    theoretical: PreparedTheoreticalSpectrum,
    *,
    min_correlation: float = 0.85,
    use_direct_mapping: bool = True,
) -> MetricResult:
    """
    Score based on Pearson correlation between measured and theoretical spectra.
    
    Critical for rejecting anti-correlated fits. PyElli analysis showed that
    LTA BestFit achieves 0.99+ correlation on good fits.
    
    Args:
        measurement: Prepared measurement spectrum
        theoretical: Prepared theoretical spectrum
        min_correlation: Minimum acceptable correlation (below this, heavily penalized)
        use_direct_mapping: If True, use correlation directly as score (preserves ranking);
                           if False, use compressed range transformation
        
    Returns:
        MetricResult with correlation-based score
    """
    measured = measurement.reflectance
    theo = theoretical.aligned_reflectance
    
    if np.std(measured) < 1e-10 or np.std(theo) < 1e-10:
        return MetricResult(score=0.0, diagnostics={'correlation': 0.0})
    
    correlation = float(np.corrcoef(measured, theo)[0, 1])
    if np.isnan(correlation):
        correlation = 0.0
    
    if use_direct_mapping:
        # Direct mapping: correlation value IS the score (with floor for negative)
        # This preserves the ranking power of correlation differences
        # 0.94 correlation → 0.94 score, 0.87 correlation → 0.87 score
        if correlation < 0:
            score = 0.0
        elif correlation < min_correlation:
            # Below threshold: dampen but don't collapse completely
            score = correlation * 0.8  # 0.80 → 0.64, 0.70 → 0.56
        else:
            score = correlation
    else:
        # Legacy: compressed range transformation
        # - Negative correlation = 0 (anti-correlated fits rejected)
        # - Below min_correlation = partial score (max 0.3)
        # - Above min_correlation = scales from 0.7 to 1.0
        if correlation < 0:
            score = 0.0
        elif correlation < min_correlation:
            score = (correlation / min_correlation) * 0.3
        else:
            score = 0.7 + 0.3 * ((correlation - min_correlation) / (1.0 - min_correlation))
    
    score = float(np.clip(score, 0.0, 1.0))
    
    diagnostics = {
        'correlation': correlation,
        'min_correlation_threshold': min_correlation,
    }
    return MetricResult(score=score, diagnostics=diagnostics)


def amplitude_score(
    measurement: PreparedMeasurement,
    theoretical: PreparedTheoreticalSpectrum,
    *,
    optimal_ratio: float = 1.0,
    tolerance: float = 0.3,
) -> MetricResult:
    """
    Score based on oscillation amplitude matching between measured and theoretical.
    
    Ensures the theoretical spectrum has similar oscillation magnitude to measured.
    Catches "flat line" theoretical spectra and overly oscillating fits.
    
    Args:
        measurement: Prepared measurement spectrum
        theoretical: Prepared theoretical spectrum
        optimal_ratio: Target ratio of theoretical/measured oscillation (1.0 = same)
        tolerance: Acceptable deviation from optimal ratio
        
    Returns:
        MetricResult with amplitude-based score
    """
    # Calculate oscillation amplitude (peak-to-peak of detrended signal)
    meas_oscillation = float(np.ptp(measurement.detrended)) if measurement.detrended.size else 0.0
    theo_oscillation = float(np.ptp(theoretical.detrended)) if theoretical.detrended.size else 0.0
    
    if meas_oscillation < 1e-8:
        # Measured has no oscillation - can't compare
        oscillation_ratio = 1.0
        score = 0.5  # Neutral score
    else:
        oscillation_ratio = theo_oscillation / meas_oscillation
        
        # Score based on how close ratio is to optimal (1.0)
        # Score = 1.0 when ratio is optimal, decreases as ratio deviates
        deviation = abs(oscillation_ratio - optimal_ratio)
        
        if deviation <= tolerance:
            # Within tolerance: high score
            score = 1.0 - (deviation / tolerance) * 0.3
        else:
            # Outside tolerance: penalized
            score = 0.7 * np.exp(-(deviation - tolerance) / 0.5)
        
        # Additional penalties for extreme cases (from PyElli)
        if oscillation_ratio > 2.0:
            score *= 0.3  # 70% penalty for 2x+ amplitude
        elif oscillation_ratio > 1.5:
            score *= 0.6  # 40% penalty for 1.5x+ amplitude
        elif oscillation_ratio < 0.3:
            score *= 0.5  # 50% penalty for very low amplitude (flat line)
    
    score = float(np.clip(score, 0.0, 1.0))
    
    diagnostics = {
        'measured_oscillation': meas_oscillation,
        'theoretical_oscillation': theo_oscillation,
        'oscillation_ratio': oscillation_ratio,
    }
    return MetricResult(score=score, diagnostics=diagnostics)


def _calculate_regional_rmse(
    residual: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Calculate RMSE for a specific wavelength region."""
    if not mask.any():
        return 0.0
    return float(np.sqrt(np.mean(residual[mask] ** 2)))


def residual_score(
    measurement: PreparedMeasurement,
    theoretical: PreparedTheoreticalSpectrum,
    *,
    tau_rmse: float,
    max_rmse: Optional[float] = None,
    use_regional_weighting: bool = True,
    regional_weights: Optional[Dict[str, float]] = None,
) -> MetricResult:
    """
    Score based on residual (RMSE) between measured and theoretical spectra.
    
    PyElli-inspired improvement: Regional RMSE weighting prioritizes early
    wavelengths (600-750nm) where fit quality is most visually impactful.
    
    Args:
        measurement: Prepared measurement spectrum
        theoretical: Prepared theoretical spectrum
        tau_rmse: Decay constant for RMSE scoring (lower = stricter)
        max_rmse: Hard cutoff - score = 0 if RMSE exceeds this
        use_regional_weighting: If True, weight RMSE by spectral region
        regional_weights: Custom weights for regions (very_early, early, mid, late)
    """
    wavelengths = measurement.wavelengths
    measured = measurement.reflectance
    theo = theoretical.aligned_reflectance
    residual = measured - theo
    
    # Global fit metrics for diagnostics
    fit_metrics = calculate_fit_metrics(measured, theo)
    global_rmse = float(fit_metrics.get("RMSE", 0.0))
    r2 = float(fit_metrics.get("R²", 0.0))
    tau = max(float(tau_rmse), 1e-9)
    
    if use_regional_weighting and len(wavelengths) > 0:
        # Regional RMSE weighting - prioritize early wavelengths (PyElli approach)
        # Early wavelength accuracy is most visually impactful
        weights = regional_weights or {
            'very_early': 0.30,  # 600-680nm - 30% weight (most critical)
            'early': 0.20,      # 680-750nm - 20% weight
            'mid': 0.25,        # 750-950nm - 25% weight
            'late': 0.25,       # >950nm    - 25% weight
        }
        
        very_early_mask = wavelengths <= 680
        early_mask = (wavelengths > 680) & (wavelengths <= 750)
        mid_mask = (wavelengths > 750) & (wavelengths <= 950)
        late_mask = wavelengths > 950
        
        very_early_rmse = _calculate_regional_rmse(residual, very_early_mask)
        early_rmse = _calculate_regional_rmse(residual, early_mask)
        mid_rmse = _calculate_regional_rmse(residual, mid_mask)
        late_rmse = _calculate_regional_rmse(residual, late_mask)
        
        # Weighted RMSE calculation
        weighted_rmse = (
            weights.get('very_early', 0.30) * very_early_rmse +
            weights.get('early', 0.20) * early_rmse +
            weights.get('mid', 0.25) * mid_rmse +
            weights.get('late', 0.25) * late_rmse
        )
        rmse = weighted_rmse
    else:
        rmse = global_rmse
        very_early_rmse = global_rmse
        early_rmse = global_rmse
        mid_rmse = global_rmse
        late_rmse = global_rmse
    
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
        "rmse_global": global_rmse,
        "rmse_very_early": very_early_rmse,
        "rmse_early": early_rmse,
        "rmse_mid": mid_rmse,
        "rmse_late": late_rmse,
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


def apply_bonus_penalty_adjustments(
    base_composite: float,
    rmse: float,
    correlation: float,
    matched_peaks: int,
    meas_peak_count: int,
    *,
    bonus_cfg: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Apply PyElli-style bonus/penalty adjustments to composite score.
    
    Creates strong separation between excellent and mediocre fits by applying
    multiplicative bonuses for excellent metrics and penalties for poor metrics.
    
    Args:
        base_composite: Initial weighted composite score
        rmse: RMSE value from residual scoring
        correlation: Correlation coefficient from correlation scoring
        matched_peaks: Number of matched peaks
        meas_peak_count: Total measured peaks (for match ratio calculation)
        bonus_cfg: Optional config for bonus/penalty thresholds
        
    Returns:
        Tuple of (adjusted_score, adjustment_diagnostics)
    """
    cfg = bonus_cfg or {}
    
    # Configurable thresholds with sensible defaults based on PyElli analysis
    excellent_rmse_threshold = float(cfg.get('excellent_rmse', 0.0015))
    good_rmse_threshold = float(cfg.get('good_rmse', 0.002))
    poor_rmse_threshold = float(cfg.get('poor_rmse', 0.003))
    
    excellent_corr_threshold = float(cfg.get('excellent_correlation', 0.98))
    good_corr_threshold = float(cfg.get('good_correlation', 0.95))
    poor_corr_threshold = float(cfg.get('poor_correlation', 0.5))
    
    min_match_ratio = float(cfg.get('min_match_ratio', 0.5))
    
    adjusted = base_composite
    applied_adjustments: Dict[str, float] = {}
    
    # === BONUSES FOR EXCELLENT FITS ===
    # Use smaller bonuses and don't cap at 1.0 to preserve ranking differentiation
    
    # Bonus for excellent RMSE (LTA achieves ~0.0006-0.0011)
    if rmse <= excellent_rmse_threshold:
        bonus = 1.10  # 10% bonus (reduced from 30%)
        adjusted *= bonus
        applied_adjustments['rmse_excellent_bonus'] = bonus
    elif rmse <= good_rmse_threshold:
        bonus = 1.05  # 5% bonus (reduced from 15%)
        adjusted *= bonus
        applied_adjustments['rmse_good_bonus'] = bonus
    
    # Bonus for excellent correlation (LTA achieves 0.99+)
    if correlation >= excellent_corr_threshold:
        bonus = 1.08  # 8% bonus (reduced from 20%)
        adjusted *= bonus
        applied_adjustments['corr_excellent_bonus'] = bonus
    elif correlation >= good_corr_threshold:
        bonus = 1.04  # 4% bonus (reduced from 10%)
        adjusted *= bonus
        applied_adjustments['corr_good_bonus'] = bonus
    
    # Combined bonus for excellent correlation AND low RMSE
    if correlation >= excellent_corr_threshold and rmse <= excellent_rmse_threshold:
        bonus = 1.05  # 5% bonus (reduced from 15%)
        adjusted *= bonus
        applied_adjustments['combined_excellent_bonus'] = bonus
    
    # Bonus for high peak match ratio
    if meas_peak_count > 0:
        match_ratio = matched_peaks / float(meas_peak_count)
        if match_ratio >= 0.9 and matched_peaks >= 5:
            bonus = 1.03  # 3% bonus (reduced from 10%)
            adjusted *= bonus
            applied_adjustments['peak_match_bonus'] = bonus
    
    # === PENALTIES FOR POOR FITS ===
    
    # Penalty for low/negative correlation
    if correlation < poor_corr_threshold:
        penalty = 0.3  # 70% penalty
        adjusted *= penalty
        applied_adjustments['corr_poor_penalty'] = penalty
    elif correlation < cfg.get('min_correlation', 0.85):
        penalty = 0.6  # 40% penalty
        adjusted *= penalty
        applied_adjustments['corr_below_min_penalty'] = penalty
    
    # Penalty for high RMSE
    if rmse > poor_rmse_threshold:
        penalty = 0.4  # 60% penalty
        adjusted *= penalty
        applied_adjustments['rmse_poor_penalty'] = penalty
    
    # Penalty for insufficient peak matches
    if meas_peak_count > 0:
        match_ratio = matched_peaks / float(meas_peak_count)
        if match_ratio < min_match_ratio:
            # Scale penalty based on how far below threshold
            penalty_factor = 1.0 - (match_ratio / min_match_ratio)
            penalty = 1.0 - 0.5 * penalty_factor  # Up to 50% penalty
            adjusted *= penalty
            applied_adjustments['peak_match_penalty'] = penalty
    
    # Don't cap at 1.0 to preserve ranking differentiation among excellent candidates
    # Final score can exceed 1.0 slightly, which helps rank the best candidates
    adjusted = float(np.clip(adjusted, 0.0, 1.5))
    applied_adjustments['final_adjustment_ratio'] = adjusted / max(base_composite, 1e-9)
    
    return adjusted, applied_adjustments


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
    """
    Score a theoretical spectrum against a measurement using multiple metrics.
    
    Uses PyElli-inspired scoring with:
    - Peak count matching (with excess/coverage penalties)
    - Peak delta (alignment quality)
    - Correlation (shape similarity - rejects anti-correlated fits)
    - Amplitude (oscillation magnitude matching)
    - Residual (RMSE/MAE fit quality)
    """
    peak_count_cfg = metrics_cfg.get("peak_count", {})
    peak_delta_cfg = metrics_cfg.get("peak_delta", {})
    correlation_cfg = metrics_cfg.get("correlation", {})
    amplitude_cfg = metrics_cfg.get("amplitude", {})
    residual_cfg = metrics_cfg.get("residual", {})
    temporal_cfg = metrics_cfg.get("temporal_continuity", {})
    weights_cfg = metrics_cfg.get("composite", {}).get("weights", {})

    # Peak count score with excess/coverage penalties
    count_result = peak_count_score(
        measurement,
        theoretical,
        tolerance_nm=float(peak_count_cfg.get("wavelength_tolerance_nm", 20.0)),
        max_allowed_excess=int(peak_count_cfg.get("max_allowed_excess", 2)),
        min_coverage_ratio=float(peak_count_cfg.get("min_coverage_ratio", 0.7)),
        excess_penalty_per_peak=float(peak_count_cfg.get("excess_penalty_per_peak", 0.2)),
        coverage_penalty_factor=float(peak_count_cfg.get("coverage_penalty_factor", 0.5)),
    )
    
    # Peak delta score with soft penalty (avoids zeroing out good fits)
    delta_result = peak_delta_score(
        measurement,
        theoretical,
        tolerance_nm=float(peak_delta_cfg.get("tolerance_nm", 20.0)),
        tau_nm=float(peak_delta_cfg.get("tau_nm", 15.0)),
        penalty_unpaired=float(peak_delta_cfg.get("penalty_unpaired", 0.04)),
        extra_penalty_unmatched_measured=float(peak_delta_cfg.get("extra_penalty_unmatched_measured", 0.02)),
        use_soft_penalty=bool(peak_delta_cfg.get("use_soft_penalty", True)),
    )
    
    # Correlation score (critical for rejecting anti-correlated fits)
    corr_result = correlation_score(
        measurement,
        theoretical,
        min_correlation=float(correlation_cfg.get("min_correlation", 0.85)),
        use_direct_mapping=bool(correlation_cfg.get("use_direct_mapping", True)),
    )
    
    # Amplitude score (oscillation matching)
    amp_result = amplitude_score(
        measurement,
        theoretical,
        optimal_ratio=float(amplitude_cfg.get("optimal_ratio", 1.0)),
        tolerance=float(amplitude_cfg.get("tolerance", 0.3)),
    )
    
    # Phase overlap score (FFT-based)
    phase_result = phase_overlap_score(measurement, theoretical)
    
    # Residual score with regional weighting (PyElli-inspired)
    residual_result = residual_score(
        measurement,
        theoretical,
        tau_rmse=float(residual_cfg.get("tau_rmse", 0.015)),
        max_rmse=residual_cfg.get("max_rmse"),
        use_regional_weighting=bool(residual_cfg.get("use_regional_weighting", True)),
    )

    component_scores: Dict[str, float] = {
        "peak_count": count_result.score,
        "peak_delta": delta_result.score,
        "correlation": corr_result.score,
        "amplitude": amp_result.score,
        "residual": residual_result.score,
        "phase_overlap": phase_result.score,
    }
    diagnostics: Dict[str, Dict[str, float]] = {
        "peak_count": count_result.diagnostics,
        "peak_delta": delta_result.diagnostics,
        "correlation": corr_result.diagnostics,
        "amplitude": amp_result.diagnostics,
        "residual": residual_result.diagnostics,
        "phase_overlap": phase_result.diagnostics,
    }

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

    # Calculate base composite score
    base_composite = composite_score(component_scores, weights_cfg)
    
    # Apply PyElli-style bonus/penalty adjustments
    bonus_cfg = metrics_cfg.get("bonus_penalty", {})
    rmse = residual_result.diagnostics.get("rmse", 0.0)
    correlation = corr_result.diagnostics.get("correlation", 0.0)
    matched_peaks = int(count_result.diagnostics.get("matched_peaks", 0))
    meas_peak_count = int(count_result.diagnostics.get("measurement_peaks", 0))
    
    if bonus_cfg.get("enabled", True):
        adjusted_composite, adjustment_diagnostics = apply_bonus_penalty_adjustments(
            base_composite,
            rmse,
            correlation,
            matched_peaks,
            meas_peak_count,
            bonus_cfg=bonus_cfg,
        )
        diagnostics["bonus_penalty"] = adjustment_diagnostics
    else:
        adjusted_composite = base_composite
    
    component_scores["composite"] = adjusted_composite
    component_scores["composite_base"] = base_composite  # Store pre-adjustment score

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
    "correlation_score",
    "amplitude_score",
    "residual_score",
    "measurement_quality_score",
    "temporal_continuity_score",
    "composite_score",
    "apply_bonus_penalty_adjustments",
    "score_spectrum",
]