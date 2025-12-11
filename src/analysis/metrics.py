"""Metric implementations for comparing theoretical and measured spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy import signal

try:
    import emd
    EMD_AVAILABLE = True
except ImportError:
    EMD_AVAILABLE = False

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


def cross_correlation_score(
    measurement: PreparedMeasurement,
    theoretical: PreparedTheoreticalSpectrum,
) -> MetricResult:
    """
    Calculate normalized cross-correlation to measure spectral alignment.
    
    Based on Shlomo's feedback: "peaks perfectly aligned → Correct physics"
    
    CRITICAL: Use ZERO-LAG correlation, not max correlation.
    - Zero-lag measures actual visual alignment (what you see)
    - Max correlation finds best offset anywhere (misleading - signals could be shifted far apart)
    """
    # Use detrended signals for cross-correlation (removes baseline effects)
    meas_signal = measurement.detrended
    theo_signal = theoretical.detrended
    
    # Ensure same length
    min_len = min(len(meas_signal), len(theo_signal))
    if min_len == 0:
        return MetricResult(score=0.0, diagnostics={})
    
    meas_signal = meas_signal[:min_len]
    theo_signal = theo_signal[:min_len]
    
    # Normalize signals to focus on shape, not amplitude
    meas_norm = (meas_signal - np.mean(meas_signal)) / (np.std(meas_signal) + 1e-10)
    theo_norm = (theo_signal - np.mean(theo_signal)) / (np.std(theo_signal) + 1e-10)
    
    # Calculate zero-lag correlation (Pearson correlation at current alignment)
    # This is what actually matters for visual match - are the peaks aligned NOW?
    zero_lag_corr = float(np.corrcoef(meas_norm, theo_norm)[0, 1])
    if np.isnan(zero_lag_corr):
        zero_lag_corr = 0.0
    
    # Convert to score: correlation ranges from -1 to 1
    # -1 = perfectly anti-correlated (peaks where valleys should be)
    # 0 = no correlation
    # +1 = perfectly correlated (peaks aligned)
    # Score: map [-1, 1] to [0, 1], with higher weight on positive correlation
    if zero_lag_corr >= 0:
        # Positive correlation: map [0, 1] to [0.5, 1.0]
        score = 0.5 + 0.5 * zero_lag_corr
    else:
        # Negative correlation: map [-1, 0] to [0, 0.5] - heavily penalize
        score = 0.5 + 0.5 * zero_lag_corr  # e.g., -0.5 -> 0.25
    
    score = float(np.clip(score, 0.0, 1.0))
    
    # Also calculate max correlation for diagnostics only
    correlation = signal.correlate(meas_norm, theo_norm, mode='same', method='auto')
    correlation_normalized = correlation / min_len
    max_corr = float(np.max(np.abs(correlation_normalized)))
    
    diagnostics = {
        "max_correlation": max_corr,
        "zero_lag_correlation": zero_lag_corr,
    }
    return MetricResult(score=score, diagnostics=diagnostics)


def peak_spacing_consistency_score(
    measurement: PreparedMeasurement,
    theoretical: PreparedTheoreticalSpectrum,
) -> MetricResult:
    """
    Measure consistency of peak spacing patterns.
    
    Based on Shlomo's feedback: "suddenly there are loads packed close together,
    then a gap... So there isn't any frequency here that's relatively steady, normal, and consistent"
    
    Good fits have regular, consistent peak spacing. Bad fits have irregular patterns.
    """
    meas_peaks = measurement.peaks["wavelength"].to_numpy(dtype=float)
    theo_peaks = theoretical.peaks["wavelength"].to_numpy(dtype=float)
    
    scores = []
    spacing_cvs = []
    
    # Check measurement peak spacing consistency
    if len(meas_peaks) >= 3:
        meas_spacings = np.diff(np.sort(meas_peaks))
        if len(meas_spacings) > 0 and np.mean(meas_spacings) > 0:
            meas_cv = float(np.std(meas_spacings) / np.mean(meas_spacings))  # Coefficient of variation
            # Lower CV = more consistent spacing (better)
            meas_score = float(np.exp(-meas_cv))  # Score: 1.0 for perfect consistency, decays with CV
            scores.append(meas_score)
            spacing_cvs.append(meas_cv)
    
    # Check theoretical peak spacing consistency
    if len(theo_peaks) >= 3:
        theo_spacings = np.diff(np.sort(theo_peaks))
        if len(theo_spacings) > 0 and np.mean(theo_spacings) > 0:
            theo_cv = float(np.std(theo_spacings) / np.mean(theo_spacings))
            theo_score = float(np.exp(-theo_cv))
            scores.append(theo_score)
            spacing_cvs.append(theo_cv)
    
    # Average score if both exist, otherwise use single value
    if scores:
        score = float(np.mean(scores))
    else:
        score = 1.0  # If too few peaks to assess, assume consistent
    
    diagnostics = {
        "measurement_peaks": float(len(meas_peaks)),
        "theoretical_peaks": float(len(theo_peaks)),
        "spacing_cv_mean": float(np.mean(spacing_cvs)) if spacing_cvs else 0.0,
    }
    return MetricResult(score=score, diagnostics=diagnostics)


def oscillation_match_score(
    measurement: PreparedMeasurement,
    theoretical: PreparedTheoreticalSpectrum,
) -> MetricResult:
    """
    CRITICAL METRIC: Measures if theoretical has similar oscillation patterns to measured.
    
    This metric detects:
    1. Flat theoretical when measured has oscillations (BAD)
    2. Similar oscillation amplitude (variance) between signals
    3. Oscillation FREQUENCY matching (critical for aqueous thickness detection)
    4. Direct visual similarity via point-by-point comparison
    
    Key insight: High aqueous (~3000nm) produces high-frequency oscillations,
    low aqueous (~100nm) produces low-frequency or nearly flat spectra.
    """
    # Use DETRENDED signals for oscillation analysis (removes baseline, reveals oscillations)
    meas_signal = measurement.detrended
    theo_signal = theoretical.detrended
    
    # Ensure same length
    min_len = min(len(meas_signal), len(theo_signal))
    if min_len == 0:
        return MetricResult(score=0.0, diagnostics={})
    
    meas_signal = meas_signal[:min_len]
    theo_signal = theo_signal[:min_len]
    
    # === Part 1: Oscillation Amplitude Match ===
    # Compare variance (oscillation amplitude) between signals
    meas_var = float(np.var(meas_signal))
    theo_var = float(np.var(theo_signal))
    
    if meas_var > 1e-12:
        var_ratio = theo_var / meas_var
        if var_ratio < 1.0:
            # Theoretical is flatter than measured - penalize
            var_score = float(np.power(max(var_ratio, 0.01), 0.25))  # More aggressive penalty
        else:
            var_score = float(np.exp(-(var_ratio - 1.0) * 0.5))
    else:
        var_score = 0.5
        var_ratio = 0.0
    
    # === Part 2: Oscillation FREQUENCY Match (Zero-crossing count) ===
    # Count zero crossings - this directly measures oscillation frequency
    # High aqueous → many zero crossings, Low aqueous → few zero crossings
    meas_zero_crossings = np.sum(np.abs(np.diff(np.sign(meas_signal))) > 0)
    theo_zero_crossings = np.sum(np.abs(np.diff(np.sign(theo_signal))) > 0)
    
    # Frequency ratio: theoretical oscillations / measured oscillations
    # Should be close to 1.0 for a good fit
    if meas_zero_crossings > 2:
        freq_ratio = theo_zero_crossings / max(meas_zero_crossings, 1)
        # Penalize significantly if frequency is very different
        # freq_ratio = 1.0 → perfect
        # freq_ratio = 0.5 → half the frequency → bad
        # freq_ratio = 0.1 → nearly flat theoretical → very bad
        if freq_ratio < 1.0:
            # Theoretical has fewer oscillations - heavily penalize
            freq_score = float(np.power(max(freq_ratio, 0.05), 0.5))
        else:
            # Theoretical has more oscillations - penalize less
            freq_score = float(np.exp(-(freq_ratio - 1.0) * 0.3))
    else:
        # Measured has few zero crossings (might be noisy or flat measurement)
        freq_score = 0.7  # Neutral-ish score
        freq_ratio = 0.0
    
    # === Part 3: Peak Count Ratio ===
    # Direct comparison of detected peaks (more robust than zero crossings)
    meas_peaks = len(measurement.peaks)
    theo_peaks = len(theoretical.peaks)
    
    if meas_peaks > 0:
        peak_ratio = theo_peaks / max(meas_peaks, 1)
        if peak_ratio < 1.0:
            # Theoretical has fewer peaks - penalize based on how few
            peak_score = float(np.power(max(peak_ratio, 0.1), 0.4))
        else:
            # Theoretical has more peaks - penalize slightly
            peak_score = float(np.exp(-(peak_ratio - 1.0) * 0.2))
    else:
        peak_score = 0.5
        peak_ratio = 0.0
    
    # === Part 4: Shape Similarity (normalized correlation) ===
    # Use Pearson correlation on detrended signals for shape matching
    if np.std(meas_signal) > 1e-10 and np.std(theo_signal) > 1e-10:
        correlation = float(np.corrcoef(meas_signal, theo_signal)[0, 1])
        if np.isnan(correlation):
            correlation = 0.0
        # Map correlation [-1, 1] to score [0, 1]
        # Positive correlation is good, negative is very bad
        shape_score = float(0.5 + 0.5 * correlation)
    else:
        shape_score = 0.3
        correlation = 0.0
    
    # === Combined Score ===
    # Emphasize frequency matching - this is the key discriminator for aqueous thickness
    # freq_score: 40% - Critical for distinguishing high vs low aqueous
    # peak_score: 25% - Validates oscillation count
    # shape_score: 25% - Overall pattern similarity  
    # var_score: 10% - Amplitude matching
    score = float(
        0.40 * freq_score +
        0.25 * peak_score +
        0.25 * shape_score +
        0.10 * var_score
    )
    
    diagnostics = {
        "measurement_variance": meas_var,
        "theoretical_variance": theo_var,
        "variance_ratio": float(var_ratio),
        "variance_score": var_score,
        "measurement_zero_crossings": float(meas_zero_crossings),
        "theoretical_zero_crossings": float(theo_zero_crossings),
        "frequency_ratio": float(freq_ratio) if meas_zero_crossings > 2 else 0.0,
        "frequency_score": freq_score,
        "measurement_peaks": float(meas_peaks),
        "theoretical_peaks": float(theo_peaks),
        "peak_ratio": float(peak_ratio) if meas_peaks > 0 else 0.0,
        "peak_score": peak_score,
        "shape_correlation": correlation if np.std(meas_signal) > 1e-10 else 0.0,
        "shape_score": shape_score,
    }
    return MetricResult(score=float(np.clip(score, 0.0, 1.0)), diagnostics=diagnostics)


def amplitude_alignment_score(
    measurement: PreparedMeasurement,
    theoretical: PreparedTheoreticalSpectrum,
) -> MetricResult:
    """
    Measure amplitude/peak height alignment.
    
    Based on Shlomo's feedback: "the peak is really high, and then they're small. It's not like they all dropped to the same level"
    
    Focuses on whether theoretical peaks have internally consistent amplitudes (not too variable).
    Does NOT penalize based on measurement's amplitude consistency (that's a measurement property, not fit quality).
    """
    meas_peaks = measurement.peaks
    theo_peaks = theoretical.peaks
    
    diagnostics = {}
    diagnostics["measurement_peak_count"] = float(len(meas_peaks))
    diagnostics["theoretical_peak_count"] = float(len(theo_peaks))
    
    # Don't penalize if we have too few peaks to assess
    if len(theo_peaks) < 2:
        # Can't assess amplitude consistency with < 2 peaks
        return MetricResult(score=1.0, diagnostics=diagnostics)
    
    # Check theoretical peak amplitude consistency
    # Shlomo: "the peak is really high, and then they're small" = bad theoretical fit
    theo_amplitudes = theo_peaks["amplitude"].to_numpy(dtype=float)
    theo_mean_amp = np.mean(theo_amplitudes)
    theo_std_amp = np.std(theo_amplitudes)
    
    if theo_mean_amp > 1e-10:
        theo_amplitude_cv = float(theo_std_amp / theo_mean_amp)
        diagnostics["theoretical_amplitude_cv"] = theo_amplitude_cv
        
        # Score based on consistency: lower CV = more consistent = better
        # CV = 0.0 → perfect consistency → score = 1.0
        # CV = 0.5 → moderate variation → score ≈ 0.6
        # CV = 1.0 → high variation → score ≈ 0.4
        # CV > 2.0 → very high variation (one peak much higher) → score ≈ 0.2
        score = float(np.exp(-theo_amplitude_cv * 0.8))  # Exponential decay with CV
        
        # Additional check: if theoretical has extremely high single peaks relative to others
        if len(theo_amplitudes) >= 3:
            sorted_amps = np.sort(theo_amplitudes)
            # Check if max peak is > 3x the median (one extremely dominant peak)
            if sorted_amps[-1] > 3.0 * sorted_amps[len(sorted_amps)//2]:
                score *= 0.7  # Penalize extreme outlier peaks
    else:
        # Zero or near-zero amplitudes
        score = 0.5
        diagnostics["theoretical_amplitude_cv"] = 0.0
    
    # Store measurement CV for diagnostics only (not used in scoring)
    if len(meas_peaks) >= 3:
        meas_amplitudes = meas_peaks["amplitude"].to_numpy(dtype=float)
        meas_mean = np.mean(meas_amplitudes)
        if meas_mean > 1e-10:
            meas_amplitude_cv = float(np.std(meas_amplitudes) / meas_mean)
            diagnostics["measurement_amplitude_cv"] = meas_amplitude_cv
    
    return MetricResult(score=float(np.clip(score, 0.0, 1.0)), diagnostics=diagnostics)


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


def _count_cycles_emd(signal_data: np.ndarray, sample_rate: float = 1.0) -> int:
    """
    Count oscillatory cycles in a signal using EMD (Empirical Mode Decomposition).
    
    This is more robust than simple peak detection because EMD decomposes the signal
    into intrinsic mode functions (IMFs) and then detects cycles based on instantaneous phase.
    
    Args:
        signal_data: 1D array of signal values
        sample_rate: Sampling rate (for frequency transform). Default 1.0 means unit spacing.
        
    Returns:
        Number of good cycles detected, or 0 if EMD is unavailable or fails.
    """
    if not EMD_AVAILABLE or len(signal_data) < 20:
        return 0
    
    try:
        # Detrend the signal first to focus on oscillations
        signal_detrended = signal.detrend(signal_data)
        
        # Decompose signal into IMFs using mask sift
        # This extracts oscillatory components from the signal
        # Use fewer IMFs for faster processing and focus on main oscillations
        imf = emd.sift.mask_sift(signal_detrended, max_imfs=4)
        
        if imf.size == 0 or imf.shape[0] < 20:
            return 0
        
        # Find the IMF with the most energy (likely contains the main oscillation)
        # Sum of squares of each IMF
        imf_energy = np.sum(imf ** 2, axis=0)
        if len(imf_energy) == 0:
            return 0
        
        # Use the IMF with highest energy (excluding residual if present)
        # Check if we have multiple IMFs
        if imf.shape[1] > 1:
            # Exclude the last IMF (usually residual/trend)
            main_imf_idx = np.argmax(imf_energy[:-1])
        else:
            main_imf_idx = 0
        
        main_imf = imf[:, main_imf_idx]
        
        # Skip if IMF is too flat (no oscillations)
        if np.std(main_imf) < 1e-6:
            return 0
        
        # Compute instantaneous phase using normalized Hilbert transform
        IP, IF, IA = emd.spectra.frequency_transform(
            main_imf.reshape(-1, 1), 
            sample_rate, 
            'nht'
        )
        
        # Extract cycle locations from instantaneous phase
        # return_good=True filters out bad cycles (incomplete, distorted, etc.)
        # Use smaller phase_step (pi instead of 1.5*pi) to catch more cycles
        good_cycles = emd.cycles.get_cycle_vector(
            IP, 
            return_good=True, 
            phase_step=np.pi  # More sensitive to detect cycles
        )
        
        # Count number of unique cycles (non-zero values in cycle vector)
        if good_cycles.size > 0:
            cycle_count = int(np.max(good_cycles))
            return max(0, cycle_count)
        else:
            return 0
            
    except Exception as e:
        # If EMD fails for any reason, return 0 (will fall back to other methods)
        return 0


def monotonic_alignment_score(
    measurement: PreparedMeasurement,
    theoretical: PreparedTheoreticalSpectrum,
    *,
    focus_wavelength_min: float = 600.0,
    focus_wavelength_max: float = 1200.0,
    focus_reflectance_max: float = 0.06,
    prefer_theoretical_below: bool = True,
) -> MetricResult:
    """
    Visual quality metric: theoretical should LOOK LIKE measured and be CLOSE to it.
    
    Key criteria (in focus region 600-1200nm, reflectance ≤ 0.06):
    1. SHAPE SIMILARITY - theoretical must follow the same pattern as measured (PRIMARY)
    2. CLOSENESS - theoretical should be close to measured (not too far in either direction)
    3. PREFER BELOW - theoretical slightly below measured is OK, but not too far below
    4. NO EXCESSIVE CROSSING - minimal criss-crossing is acceptable if shape matches
    
    This balances visual quality with the client's typical fit patterns.
    """
    wavelengths = measurement.wavelengths
    meas_raw = measurement.reflectance
    theo_raw = theoretical.aligned_reflectance
    
    min_len = min(len(wavelengths), len(meas_raw), len(theo_raw))
    if min_len < 10:
        return MetricResult(score=0.5, diagnostics={"zero_crossings_focus": 0.0})
    
    wavelengths = wavelengths[:min_len]
    meas_raw = meas_raw[:min_len]
    theo_raw = theo_raw[:min_len]
    
    # Calculate residual (measured - theoretical)
    residual = meas_raw - theo_raw
    
    # Find indices in focus region
    focus_mask = (
        (wavelengths >= focus_wavelength_min) & 
        (wavelengths <= focus_wavelength_max) &
        (meas_raw <= focus_reflectance_max)
    )
    
    points_in_focus = int(np.sum(focus_mask))
    
    if points_in_focus < 10:
        residual_focus = residual
        meas_focus = meas_raw
        theo_focus = theo_raw
        points_in_focus = len(residual)
    else:
        residual_focus = residual[focus_mask]
        meas_focus = meas_raw[focus_mask]
        theo_focus = theo_raw[focus_mask]
    
    # === CRITERION 1: SHAPE SIMILARITY (PRIMARY - 50% weight) ===
    # Theoretical must follow the same pattern as measured
    if np.std(meas_focus) > 1e-10 and np.std(theo_focus) > 1e-10:
        correlation = float(np.corrcoef(meas_focus, theo_focus)[0, 1])
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0
    
    # Convert correlation to score
    # correlation = 1.0 → perfect shape match → score = 1.0
    # correlation = 0.9 → good match → score = 0.9
    # correlation = 0.5 → poor match → score = 0.5
    # correlation < 0 → inverse pattern → score = 0
    shape_score = max(0.0, correlation)
    
    # === OSCILLATION MATCHING ANALYSIS ===
    # Critical: Detect if theoretical lacks oscillations that measured has
    # Use multiple methods: EMD cycles, variance ratio, and detrended correlation
    
    # Detrend both signals to focus on oscillations (not just overall trend)
    meas_detrended = signal.detrend(meas_focus)
    theo_detrended = signal.detrend(theo_focus)
    
    # Method 1: Variance ratio (oscillation amplitude)
    meas_var = float(np.var(meas_detrended))
    theo_var = float(np.var(theo_detrended))
    var_ratio = 1.0
    var_penalty = 1.0
    
    if meas_var > 1e-12:  # Measured has oscillations
        var_ratio = theo_var / meas_var
        if var_ratio < 0.5:  # Theoretical has < 50% of measured variance
            # Heavy penalty for smooth/flat theoretical
            var_penalty = float(np.power(max(var_ratio, 0.01), 0.3))  # Very aggressive
        elif var_ratio < 0.7:
            var_penalty = 0.5 + 0.3 * (var_ratio - 0.5) / 0.2  # 0.5-0.8
        elif var_ratio < 0.9:
            var_penalty = 0.8 + 0.2 * (var_ratio - 0.7) / 0.2  # 0.8-1.0
        elif var_ratio <= 1.5:
            # Theoretical has 90-150% of measured variance - ideal range
            var_penalty = 1.0  # No penalty, this is good
        else:
            # Theoretical has > 150% of measured variance - too oscillatory
            # Slight penalty, but not as severe as too smooth
            var_penalty = float(np.exp(-(var_ratio - 1.5) * 0.3))  # Decay from 1.0
    
    # Method 2: EMD cycle counting
    meas_cycles = 0
    theo_cycles = 0
    cycle_ratio = 1.0
    cycle_penalty = 1.0
    
    if EMD_AVAILABLE and len(meas_focus) >= 20:
        # Calculate sample rate (wavelength spacing)
        if points_in_focus >= 10:
            wl_focus = wavelengths[focus_mask]
        else:
            wl_focus = wavelengths
        
        if len(wl_focus) > 1:
            sample_rate = 1.0 / float(np.mean(np.diff(wl_focus)))
        else:
            sample_rate = 1.0
        
        # Count cycles using EMD on detrended signals
        meas_cycles = _count_cycles_emd(meas_detrended, sample_rate)
        theo_cycles = _count_cycles_emd(theo_detrended, sample_rate)
        
        # Compare cycle counts
        if meas_cycles > 0:
            cycle_ratio = theo_cycles / max(meas_cycles, 1)
            if cycle_ratio < 0.5:
                cycle_penalty = float(np.power(max(cycle_ratio, 0.1), 0.5))
            elif cycle_ratio < 0.7:
                cycle_penalty = 0.6 + 0.2 * (cycle_ratio - 0.5) / 0.2  # 0.6-0.8
            elif cycle_ratio < 0.9:
                cycle_penalty = 0.8 + 0.2 * (cycle_ratio - 0.7) / 0.2  # 0.8-1.0
    
    # Method 3: Detrended correlation (oscillation pattern matching)
    osc_correlation = 0.0
    osc_corr_penalty = 1.0
    
    if np.std(meas_detrended) > 1e-10 and np.std(theo_detrended) > 1e-10:
        osc_correlation = float(np.corrcoef(meas_detrended, theo_detrended)[0, 1])
        if np.isnan(osc_correlation):
            osc_correlation = 0.0
        # If detrended correlation is low, theoretical lacks oscillation pattern
        if osc_correlation < 0.5:
            osc_corr_penalty = max(0.3, osc_correlation)  # Heavy penalty
        elif osc_correlation < 0.7:
            osc_corr_penalty = 0.5 + 0.3 * (osc_correlation - 0.5) / 0.2  # 0.5-0.8
        elif osc_correlation < 0.9:
            osc_corr_penalty = 0.8 + 0.2 * (osc_correlation - 0.7) / 0.2  # 0.8-1.0
    
    # Combine all oscillation penalties (use minimum = most aggressive)
    # This ensures we catch smooth theoretical spectra
    oscillation_penalty = min(var_penalty, cycle_penalty, osc_corr_penalty)
    
    # Apply oscillation penalty to shape score
    # This is critical - smooth theoretical should get heavily penalized
    shape_score = shape_score * oscillation_penalty
    
    # === CRITERION 2: CLOSENESS (25% weight) ===
    # Theoretical should be close to measured - penalize large gaps in EITHER direction
    # Use mean absolute residual normalized by measurement range
    mean_abs_residual = float(np.mean(np.abs(residual_focus)))
    meas_range = float(np.max(meas_focus) - np.min(meas_focus)) if len(meas_focus) > 0 else 0.01
    meas_range = max(meas_range, 0.001)  # Avoid division by zero
    
    # Normalized gap: how far is theoretical from measured relative to measurement range
    normalized_gap = mean_abs_residual / meas_range
    
    # Score: small gap = high score, large gap = low score
    # gap = 0 → score = 1.0
    # gap = 0.5 (50% of range) → score ≈ 0.37
    # gap = 1.0 (100% of range) → score ≈ 0.14
    closeness_score = float(np.exp(-normalized_gap * 2))
    
    # === DIAGNOSTIC: Mean residual (for info only, not used in scoring) ===
    mean_residual = float(np.mean(residual_focus))  # Positive = theo below, Negative = theo above
    
    # === CRITERION 3: NO CRISS-CROSSING - HARD REJECTION ===
    # ANY crossing in the focus region = REJECTED
    sign_changes_focus = int(np.sum(np.abs(np.diff(np.sign(residual_focus))) > 0))
    
    # HARD RULE: Zero crossings allowed
    if sign_changes_focus == 0:
        crossing_score = 1.0  # Perfect - no crossings, this is what we want
        is_valid_fit = True
    else:
        crossing_score = 0.0  # ANY crossing = REJECTED
        is_valid_fit = False
    
    # === COMBINED SCORE ===
    if is_valid_fit:
        # Only score fits with NO criss-crossing
        score = (
            0.50 * shape_score +      # Shape is most important
            0.50 * closeness_score    # Must be close to measured
        )
        
        # Bonus: if shape is excellent AND close, boost score
        if shape_score > 0.95 and closeness_score > 0.6:
            score = min(1.0, score * 1.1)
        
        # Penalty: if theoretical is TOO FAR from measured, cap score
        if closeness_score < 0.3:
            score = min(score, 0.4)  # Hard cap when too far away
    else:
        # REJECTED: criss-crossing fit
        # Give a very low score so it never gets selected
        score = 0.05
    
    # Penalty: if shape is poor (<0.7), cap the score
    if shape_score < 0.7:
        score = min(score, 0.5)
    
    diagnostics = {
        "shape_correlation": float(correlation),
        "shape_score": float(shape_score),
        "mean_abs_residual": float(mean_abs_residual),
        "normalized_gap": float(normalized_gap),
        "closeness_score": float(closeness_score),
        "mean_residual": float(mean_residual),
        "zero_crossings_focus": float(sign_changes_focus),
        "crossing_score": float(crossing_score),
        "points_in_focus": float(points_in_focus),
        "meas_cycles_emd": float(meas_cycles),
        "theo_cycles_emd": float(theo_cycles),
        "cycle_ratio_emd": float(cycle_ratio),
        "oscillation_penalty": float(oscillation_penalty),
        "meas_var": meas_var,
        "theo_var": theo_var,
        "var_ratio": float(var_ratio),
        "var_penalty": float(var_penalty),
        "osc_correlation": float(osc_correlation),
        "osc_corr_penalty": float(osc_corr_penalty),
    }
    
    return MetricResult(score=float(np.clip(score, 0.0, 1.0)), diagnostics=diagnostics)


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

    # Optional metrics based on Shlomo's feedback (for visual quality assessment)
    cross_corr_cfg = metrics_cfg.get("cross_correlation", {})
    if cross_corr_cfg and cross_corr_cfg.get("enabled", False):
        cross_corr_result = cross_correlation_score(measurement, theoretical)
        component_scores["cross_correlation"] = cross_corr_result.score
        diagnostics["cross_correlation"] = cross_corr_result.diagnostics
    
    spacing_cfg = metrics_cfg.get("peak_spacing_consistency", {})
    if spacing_cfg and spacing_cfg.get("enabled", False):
        spacing_result = peak_spacing_consistency_score(measurement, theoretical)
        component_scores["peak_spacing_consistency"] = spacing_result.score
        diagnostics["peak_spacing_consistency"] = spacing_result.diagnostics
    
    amplitude_cfg = metrics_cfg.get("amplitude_alignment", {})
    if amplitude_cfg and amplitude_cfg.get("enabled", False):
        amplitude_result = amplitude_alignment_score(measurement, theoretical)
        component_scores["amplitude_alignment"] = amplitude_result.score
        diagnostics["amplitude_alignment"] = amplitude_result.diagnostics

    # CRITICAL: Oscillation match - detects flat theoretical vs oscillating measured
    oscillation_cfg = metrics_cfg.get("oscillation_match", {})
    if oscillation_cfg and oscillation_cfg.get("enabled", False):
        oscillation_result = oscillation_match_score(measurement, theoretical)
        component_scores["oscillation_match"] = oscillation_result.score
        diagnostics["oscillation_match"] = oscillation_result.diagnostics

    # Monotonic alignment - penalizes criss-crossing between theoretical and measured
    monotonic_cfg = metrics_cfg.get("monotonic_alignment", {})
    if monotonic_cfg and monotonic_cfg.get("enabled", False):
        monotonic_result = monotonic_alignment_score(
            measurement,
            theoretical,
            focus_wavelength_min=float(monotonic_cfg.get("focus_wavelength_min", 600.0)),
            focus_wavelength_max=float(monotonic_cfg.get("focus_wavelength_max", 1200.0)),
            focus_reflectance_max=float(monotonic_cfg.get("focus_reflectance_max", 0.06)),
            prefer_theoretical_below=bool(monotonic_cfg.get("prefer_theoretical_below", True)),
        )
        component_scores["monotonic_alignment"] = monotonic_result.score
        diagnostics["monotonic_alignment"] = monotonic_result.diagnostics

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
    "cross_correlation_score",
    "peak_spacing_consistency_score",
    "oscillation_match_score",
    "amplitude_alignment_score",
    "monotonic_alignment_score",
    "measurement_quality_score",
    "temporal_continuity_score",
    "composite_score",
    "score_spectrum",
]
