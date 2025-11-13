"""Analysis utilities for tear film spectrum alignment."""

from .measurement_utils import (
    PreparedMeasurement,
    PreparedTheoreticalSpectrum,
    load_measurement_spectrum,
    prepare_measurement,
    prepare_theoretical_spectrum,
    detrend_signal,
    detect_peaks,
    detect_valleys,
)
from .metrics import (
    peak_count_score,
    peak_delta_score,
    phase_overlap_score,
    composite_score,
)

__all__ = [
    "PreparedMeasurement",
    "PreparedTheoreticalSpectrum",
    "load_measurement_spectrum",
    "prepare_measurement",
    "prepare_theoretical_spectrum",
    "detrend_signal",
    "detect_peaks",
    "detect_valleys",
    "peak_count_score",
    "peak_delta_score",
    "phase_overlap_score",
    "composite_score",
]
