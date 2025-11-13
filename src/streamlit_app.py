"""
Interactive Streamlit + Plotly app for exploring tear film reflectance spectra.

Features:
- Sliders for lipid thickness, aqueous thickness, and roughness
- Measurement spectra overlay from specified directory
- Side-by-side comparison of measured vs theoretical spectra
- Fit metrics calculation
"""

from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any, Dict, List, Optional, Tuple

from tear_film_generator import (
    load_config,
    validate_config,
    make_single_spectrum_calculator,
    PROJECT_ROOT,
    get_project_path,
)

from analysis.measurement_utils import (
    calculate_fit_metrics,
    detrend_signal,
    detect_peaks,
    detect_valleys,
    interpolate_measurement_to_theoretical,
    load_measurement_files,
    load_txt_file_enhanced,
)

def clamp_to_step(value: float, min_val: float, step: float) -> float:
    """Snap a value to the nearest step from min_val."""
    return min_val + round((value - min_val) / step) * step


