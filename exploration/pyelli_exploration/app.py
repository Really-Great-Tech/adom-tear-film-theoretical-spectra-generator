"""
PyElli Exploration App

A Streamlit application demonstrating pyElli's capabilities for tear film
interferometry analysis using ADOM sample data.

This is an exploration tool for evaluating pyElli's potential integration
into the AdOM-TFI workflow.

Run with: streamlit run app.py
"""

import logging
import sys
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Suppress Streamlit widget warning about default value + session state
warnings.filterwarnings(
    "ignore",
    message=".*widget.*key.*was created with a default value but also had its value set via the Session State API.*",
    category=UserWarning,
)
# Suppress ScriptRunContext warnings from multiprocessing (these are harmless)
warnings.filterwarnings(
    "ignore", message=".*missing ScriptRunContext.*", category=UserWarning
)
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*", category=UserWarning)
# Also suppress via Streamlit logger if it's logged there
streamlit_logger = logging.getLogger("streamlit")
streamlit_logger.setLevel(logging.ERROR)  # Only show errors, not warnings
# Suppress ScriptRunContext warnings from streamlit runtime
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
    logging.ERROR
)
logging.getLogger("streamlit.runtime.scriptrunner_utils").setLevel(logging.ERROR)
# Suppress all streamlit runtime loggers that might produce ScriptRunContext warnings
logging.getLogger("streamlit.runtime").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(
    logging.ERROR
)


# Add a custom filter to suppress ScriptRunContext warnings at the handler level
class ScriptRunContextFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return "ScriptRunContext" not in msg and "missing ScriptRunContext" not in msg


# Apply filter to root logger and all streamlit loggers
root_logger = logging.getLogger()
root_logger.setLevel(logging.ERROR)  # Suppress all warnings at root level
for handler in root_logger.handlers:
    handler.addFilter(ScriptRunContextFilter())

# Suppress all streamlit-related loggers
for logger_name in [
    "streamlit",
    "streamlit.runtime",
    "streamlit.runtime.scriptrunner",
    "streamlit.runtime.scriptrunner.script_runner",
    "streamlit.runtime.scriptrunner_utils",
    "streamlit.runtime.scriptrunner_utils.script_run_context",
]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)
    for handler in logger.handlers:
        handler.addFilter(ScriptRunContextFilter())

# Also add filter to any future handlers
logging.getLogger().addFilter(ScriptRunContextFilter())

# Add parent paths for imports
# Resolve PROJECT_ROOT: try from __file__ first, then fallback to current working directory
_candidate_root = Path(__file__).resolve().parent.parent.parent
if (_candidate_root / "exploration" / "pyelli_exploration" / "app.py").exists():
    PROJECT_ROOT = _candidate_root
else:
    # Fallback: try resolving from current working directory
    _cwd_root = Path.cwd()
    if (_cwd_root / "exploration" / "pyelli_exploration" / "app.py").exists():
        PROJECT_ROOT = _cwd_root
    else:
        # Last resort: use __file__ resolution even if structure doesn't match perfectly
        PROJECT_ROOT = _candidate_root
        logger.warning(f"⚠️ PROJECT_ROOT resolution ambiguous. Using: {PROJECT_ROOT}")
        logger.warning(f"   Current working directory: {Path.cwd()}")
        logger.warning(f"   __file__ path: {Path(__file__).resolve()}")
sys.path.insert(0, str(PROJECT_ROOT))
logger.debug(f"📁 PROJECT_ROOT: {PROJECT_ROOT}")

from src.pdf_report import generate_pyelli_pdf_report

from exploration.pyelli_exploration.pyelli_utils import (
    load_measured_spectrum,
    load_bestfit_spectrum,
    load_material_data,
    get_sample_data_paths,
    get_new_spectra_paths,
    get_available_materials,
    calculate_residual,
    calculate_correlation,
    interpolate_to_common_wavelengths,
)
from exploration.pyelli_exploration.pyelli_grid_search import (
    PyElliGridSearch,
    calculate_peak_based_score,
)
from src.analysis.measurement_utils import (
    detrend_signal,
    boxcar_smooth,
    gaussian_smooth,
    detect_peaks,
    detect_valleys,
)
from src.analysis.metrics import _match_peaks
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="PyElli Exploration | AdOM-TFI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state early (before sidebar code accesses it)
if "pending_update" not in st.session_state:
    st.session_state.pending_update = None
if "autofit_results" not in st.session_state:
    st.session_state.autofit_results = None
if "last_run_elapsed_s" not in st.session_state:
    st.session_state.last_run_elapsed_s = None
if "selected_rank" not in st.session_state:
    st.session_state.selected_rank = 1  # Default to rank 1
# Widget key version - increment to force slider reset
if "widget_key_version" not in st.session_state:
    st.session_state.widget_key_version = 0
# Forced parameter values from grid search or rank selection
if "forced_lipid" not in st.session_state:
    st.session_state.forced_lipid = None
if "forced_aqueous" not in st.session_state:
    st.session_state.forced_aqueous = None
if "forced_mucus" not in st.session_state:
    st.session_state.forced_mucus = None
# Smoothing parameters
if "smoothing_type" not in st.session_state:
    st.session_state.smoothing_type = "none"
if "boxcar_width_nm" not in st.session_state:
    st.session_state.boxcar_width_nm = 17.0
if "boxcar_passes" not in st.session_state:
    st.session_state.boxcar_passes = 1
if "gaussian_kernel" not in st.session_state:
    st.session_state.gaussian_kernel = 11

# Material selection (defaults match LTA Stack XML configuration)
if "selected_lipid_material" not in st.session_state:
    st.session_state.selected_lipid_material = "lipid_05-02621extrapolated.csv"
if "selected_water_material" not in st.session_state:
    st.session_state.selected_water_material = "water_Bashkatov1353extrapolated.csv"
if "selected_mucus_material" not in st.session_state:
    st.session_state.selected_mucus_material = "water_Bashkatov1353extrapolated.csv"
if "selected_substratum_material" not in st.session_state:
    st.session_state.selected_substratum_material = (
        "struma_Bashkatov140extrapolated.csv"
    )
# Custom uploaded materials: dict mapping filename -> DataFrame with wavelength_nm, n, k
if "custom_materials" not in st.session_state:
    st.session_state.custom_materials = {}

# Grid search range settings (match ADOM standard ranges)
if "grid_lipid_min" not in st.session_state:
    st.session_state.grid_lipid_min = 9.0
if "grid_lipid_max" not in st.session_state:
    st.session_state.grid_lipid_max = 250.0
if "grid_lipid_step" not in st.session_state:
    st.session_state.grid_lipid_step = 5.0
if "grid_aq_min" not in st.session_state:
    st.session_state.grid_aq_min = 800.0
if "grid_aq_max" not in st.session_state:
    st.session_state.grid_aq_max = 12000.0
if "grid_aq_step" not in st.session_state:
    st.session_state.grid_aq_step = 200.0
if "grid_mu_min" not in st.session_state:
    st.session_state.grid_mu_min = 600.0
if "grid_mu_max" not in st.session_state:
    st.session_state.grid_mu_max = 7000.0
if "grid_mu_step" not in st.session_state:
    st.session_state.grid_mu_step = 100.0


def validate_material_file(uploaded_file) -> tuple[bool, str, pd.DataFrame | None]:
    """
    Validate an uploaded material CSV file.

    Expected format: CSV with 3 columns (wavelength_nm, n, k)
    - wavelength_nm: wavelength in nanometers (typically 200-2000 nm range)
    - n: refractive index (typically 1.0-3.0)
    - k: extinction coefficient (typically 0-1)

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Tuple of (is_valid, message, dataframe or None)
    """
    try:
        # Read file content
        content = uploaded_file.getvalue().decode("utf-8")
        lines = content.strip().split("\n")

        if len(lines) < 3:
            return (
                False,
                "❌ File too short. Need at least a header + 2 data rows.",
                None,
            )

        # Parse data lines (skip potential header)
        data_rows = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    wavelength = float(parts[0])
                    n_value = float(parts[1])
                    k_value = float(parts[2])
                    data_rows.append((wavelength, n_value, k_value))
                except ValueError:
                    # Skip header or malformed rows
                    continue

        if len(data_rows) < 2:
            return (
                False,
                "❌ Not enough valid data rows. Need at least 2 rows with 3 numeric columns.",
                None,
            )

        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=["wavelength_nm", "n", "k"])

        # Sanity checks
        wavelength_min, wavelength_max = (
            df["wavelength_nm"].min(),
            df["wavelength_nm"].max(),
        )
        n_min, n_max = df["n"].min(), df["n"].max()
        k_min, k_max = df["k"].min(), df["k"].max()

        warnings = []

        # Check wavelength range (should cover visible/NIR)
        if wavelength_min > 400:
            warnings.append(f"⚠️ Min wavelength ({wavelength_min:.0f} nm) > 400 nm")
        if wavelength_max < 1000:
            warnings.append(f"⚠️ Max wavelength ({wavelength_max:.0f} nm) < 1000 nm")

        # Check refractive index range
        if n_min < 0.5 or n_max > 5.0:
            warnings.append(
                f"⚠️ Unusual n range: {n_min:.3f} - {n_max:.3f} (expected 0.5-5.0)"
            )

        # Check extinction coefficient
        if k_min < 0:
            return (
                False,
                f"❌ Negative k values found (min: {k_min:.6f}). k must be >= 0.",
                None,
            )
        if k_max > 10:
            warnings.append(f"⚠️ High k values (max: {k_max:.3f})")

        # Check data is sorted by wavelength
        if not df["wavelength_nm"].is_monotonic_increasing:
            warnings.append("⚠️ Wavelengths not sorted. Will be sorted automatically.")
            df = df.sort_values("wavelength_nm").reset_index(drop=True)

        # Build success message
        msg = f"✅ Valid material file: {len(df)} points, λ: {wavelength_min:.0f}-{wavelength_max:.0f} nm"
        if warnings:
            msg += "\n" + "\n".join(warnings)

        return True, msg, df

    except Exception as e:
        return False, f"❌ Error parsing file: {str(e)}", None


def apply_smoothing_to_signal(
    signal: np.ndarray,
    wavelengths: np.ndarray,
    smoothing_type: str,
    boxcar_width_nm: float = 17.0,
    boxcar_passes: int = 1,
    gaussian_kernel: int = 11,
) -> np.ndarray:
    """Apply smoothing to a signal array.

    Args:
        signal: Signal values to smooth.
        wavelengths: Wavelength array in nanometers.
        smoothing_type: One of 'none', 'boxcar', or 'gaussian'.
        boxcar_width_nm: Boxcar smoothing width in nanometers.
        boxcar_passes: Number of boxcar smoothing passes.
        gaussian_kernel: Gaussian kernel size in samples.

    Returns:
        Smoothed signal values.
    """
    if smoothing_type == "none":
        return signal

    if smoothing_type == "boxcar":
        return boxcar_smooth(signal, wavelengths, boxcar_width_nm, boxcar_passes)
    elif smoothing_type == "gaussian":
        return gaussian_smooth(signal, gaussian_kernel)
    else:
        return signal


# Clean Light Theme CSS
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Main app styling */
    .stApp {
        background: #ffffff;
    }
    
    /* Make sidebar wider */
    section[data-testid="stSidebar"] {
        min-width: 400px !important;
        max-width: 450px !important;
    }
    
    /* Hide sidebar collapse button and hover key icon */
    button[data-testid="baseButton-header"],
    [data-testid="stSidebar"] button[title*="Close"],
    [data-testid="stSidebar"] button[title*="Open"],
    [data-testid="stSidebar"] button[aria-label*="Close"],
    [data-testid="stSidebar"] button[aria-label*="Open"],
    [data-testid="stSidebar"] [class*="keyboard"],
    [data-testid="stSidebar"] [class*="double"],
    [data-testid="stSidebar"] svg[viewBox*="24"],
    /* Hide the "key" tooltip/indicator that appears on sidebar hover */
    [data-testid="stSidebar"] [data-testid*="key"],
    [data-testid="stSidebar"] [title="key"],
    .stTooltipIcon,
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    /* Hide keyboard shortcut hints */
    [data-testid="stSidebar"] > div:first-child > div:first-child > span,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] + span,
    section[data-testid="stSidebar"] > div > div > span:first-child {
        display: none !important;
    }
    
    /* Hide the floating "key" text at top of sidebar */
    section[data-testid="stSidebar"] > div:first-child > span {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Hide Material Icons text when font fails to load (shows "keyboard_arrow_right" etc.) */
    /* Target the Material Icon element by its data-testid - hide it completely */
    [data-testid="stIconMaterial"] {
        position: absolute !important;
        left: -9999px !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        clip: rect(0,0,0,0) !important;
    }
    
    /* Style the expander header container */
    [data-testid="stExpander"] summary > span {
        display: flex !important;
        align-items: center !important;
    }
    
    /* Add a CSS-based arrow icon before the expander label */
    [data-testid="stExpander"] summary > span::before {
        content: "▶" !important;
        font-size: 12px !important;
        margin-right: 6px !important;
        font-family: sans-serif !important;
    }
    [data-testid="stExpander"][open] summary > span::before {
        content: "▼" !important;
    }
    
    /* Fallback: hide keyboard-related elements */
    [data-testid="stSidebar"] span[class*="keyboard"],
    [data-testid="stSidebar"] [class*="StyledKeyboardShortcut"],
    [data-testid="stExpander"] [class*="StyledKeyboardShortcut"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Ensure regular buttons in sidebar are visible */
    [data-testid="stSidebar"] .stButton button {
        display: block !important;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Typography */
    html, body, [class*="st-"] {
        font-family: 'DM Sans', sans-serif;
        color: #1e293b;
    }
    
    h1, h2, h3, h4 {
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        color: #0f172a;
    }
    
    h1 { font-size: 2.25rem !important; letter-spacing: -0.02em; }
    h2 { font-size: 1.5rem !important; color: #1e40af; }
    h3 { font-size: 1.15rem !important; color: #334155; }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #1e40af !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #475569 !important;
    }
    
    /* Custom tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f1f5f9;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 10px 18px;
        color: #64748b;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #fff;
        color: #1e40af;
    }
    
    .stTabs [aria-selected="true"] {
        background: #fff !important;
        color: #1e40af !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        font-weight: 600;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem !important;
        font-weight: 600;
        color: #1e40af;
    }
    
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 0.7rem !important;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #1e40af 100%) !important;
    }
    
    .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: #1e40af;
        font-weight: 600;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: #f8fafc;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    
    .stRadio label {
        color: #334155 !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: #fff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    
    /* Checkboxes */
    .stCheckbox label {
        color: #334155 !important;
    }
    
    /* Buttons - Force white text */
    .stButton > button,
    .stButton button,
    button[kind="primary"],
    button[data-testid="baseButton-primary"],
    [data-testid="stBaseButton-primary"],
    [data-testid="baseButton-primary"] {
        background: #1e40af !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(30, 64, 175, 0.2);
    }
    
    .stButton > button:hover,
    .stButton button:hover,
    button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    [data-testid="stBaseButton-primary"]:hover,
    [data-testid="baseButton-primary"]:hover {
        background: #1e3a8a !important;
        color: #ffffff !important;
        box-shadow: 0 4px 8px rgba(30, 64, 175, 0.3);
    }
    
    /* Button text - ensure visibility */
    .stButton > button p,
    .stButton > button span,
    button[kind="primary"] p,
    button[kind="primary"] span {
        color: #ffffff !important;
    }
    
    /* Section headers */
    .section-header {
        font-size: 0.9rem;
        font-weight: 600;
        color: #475569;
        margin-top: 12px;
        margin-bottom: 8px;
    }
    
    /* Success alerts */
    .stAlert {
        border-radius: 8px !important;
    }
    
    /* Hide Streamlit widget warnings about session state - target warning alerts */
    div[data-testid="stAlert"]:has(> div:contains("widget")),
    div[data-testid="stAlert"]:has(> div:contains("param_lipid")),
    div[data-testid="stAlert"]:has(> div:contains("Session State API")),
    /* Also target by alert type (warning) */
    div[data-testid="stAlert"][data-baseweb="notification"] {
        /* Check if it contains widget warning text */
    }
    
    /* More aggressive: Hide all warning alerts that might contain widget warnings */
    div[data-testid="stAlert"] {
        /* We'll use JavaScript to check content and hide */
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }
    
    /* Separators */
    hr {
        border-color: #e2e8f0 !important;
    }
    
    /* Custom classes */
    .hero-text {
        font-size: 1.05rem;
        color: #64748b;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    
    .gradient-text {
        color: #1e40af;
        font-weight: 700;
    }
    
    .card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
    }
    
    .stat-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    
    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 2rem 0;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
        font-size: 0.9rem;
    }
</style>
<script>
    // Hide Streamlit widget warnings about session state
    function hideWidgetWarnings() {
        const alerts = document.querySelectorAll('[data-testid="stAlert"]');
        alerts.forEach(alert => {
            const text = alert.textContent || alert.innerText || '';
            // Check if this is the widget warning we want to hide
            if (text.includes('widget') && text.includes('key') && 
                (text.includes('param_lipid') || text.includes('param_aqueous') || text.includes('param_mucus')) &&
                text.includes('Session State API')) {
                alert.style.display = 'none';
                alert.style.visibility = 'hidden';
                alert.style.height = '0';
                alert.style.margin = '0';
                alert.style.padding = '0';
            }
        });
    }
    
    // Run immediately
    hideWidgetWarnings();
    
    // Run on page load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', hideWidgetWarnings);
    } else {
        hideWidgetWarnings();
    }
    
    // Also hide after Streamlit reruns (observe DOM changes)
    const observer = new MutationObserver(function(mutations) {
        hideWidgetWarnings();
    });
    observer.observe(document.body, { 
        childList: true, 
        subtree: true,
        attributes: false
    });
    
    // Also run periodically as a fallback
    setInterval(hideWidgetWarnings, 500);
</script>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Path Configuration
# =============================================================================

SAMPLE_DATA_PATH = PROJECT_ROOT / "exploration" / "sample_data"
MATERIALS_PATH = PROJECT_ROOT / "data" / "Materials"


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown(
        """
    <div style="text-align: center; padding: 16px 0;">
        <div style="font-size: 2.5rem; margin-bottom: 8px;">🔬</div>
        <h2 style="margin: 0; font-size: 1.3rem; color: #1e40af;">PyElli Explorer</h2>
        <p style="color: #64748b; font-size: 0.8rem; margin-top: 4px;">Thin Film Optics Analysis</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<hr style="margin: 12px 0; border-color: #e2e8f0;">', unsafe_allow_html=True
    )

    st.markdown(
        """
    <div class="card" style="padding: 14px;">
        <p style="color: #1e40af; font-weight: 600; margin-bottom: 10px; font-size: 0.8rem;">✨ KEY CAPABILITIES</p>
        <div style="display: flex; flex-direction: column; gap: 6px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1rem;">📐</span>
                <span style="color: #475569; font-size: 0.82rem;">Transfer Matrix Method</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1rem;">🌈</span>
                <span style="color: #475569; font-size: 0.82rem;">Dispersion Modeling</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1rem;">📊</span>
                <span style="color: #475569; font-size: 0.82rem;">Multi-Layer Optics</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1rem;">⚡</span>
                <span style="color: #475569; font-size: 0.82rem;">Spectral Fitting</span>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<hr style="margin: 12px 0; border-color: #e2e8f0;">', unsafe_allow_html=True
    )

    # Check data availability
    samples = get_sample_data_paths(SAMPLE_DATA_PATH)
    materials = get_available_materials(MATERIALS_PATH)
    new_spectra_path = PROJECT_ROOT / "new spectra"
    new_spectra = (
        get_new_spectra_paths(new_spectra_path) if new_spectra_path.exists() else {}
    )

    more_good_path = (
        PROJECT_ROOT / "exploration" / "more_good_spectras" / "Corrected_Spectra"
    )
    more_good_spectra = (
        sorted(list(more_good_path.glob("(Run)spectra_*.txt")))
        if more_good_path.exists()
        else []
    )

    st.markdown(
        """
    <p style="color: #1e40af; font-weight: 600; margin-bottom: 10px; font-size: 0.8rem;">📂 DATA STATUS</p>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 10px; text-align: center;">
            <div style="font-size: 1.3rem; font-weight: 700; color: #16a34a;">10</div>
            <div style="color: #64748b; font-size: 0.65rem; text-transform: uppercase;">Good Fits</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
        <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 10px; text-align: center;">
            <div style="font-size: 1.3rem; font-weight: 700; color: #dc2626;">10</div>
            <div style="color: #64748b; font-size: 0.65rem; text-transform: uppercase;">Bad Fits</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
        <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 10px; text-align: center;">
            <div style="font-size: 1.3rem; font-weight: 700; color: #1e40af;">{len(more_good_spectra)}</div>
            <div style="color: #64748b; font-size: 0.65rem; text-transform: uppercase;">More Good</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
        <div style="background: #fef3c7; border: 1px solid #fde68a; border-radius: 8px; padding: 10px; text-align: center;">
            <div style="font-size: 1.3rem; font-weight: 700; color: #d97706;">{len(new_spectra)}</div>
            <div style="color: #64748b; font-size: 0.65rem; text-transform: uppercase;">New Spectra</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<hr style="margin: 12px 0; border-color: #e2e8f0;">', unsafe_allow_html=True
    )

    # =============================================================================
    # Sidebar Controls (Constant across all tabs)
    # =============================================================================

    st.markdown("### 📂 Spectrum Source")

    spectrum_sources = {
        "More Good Spectras": PROJECT_ROOT
        / "exploration"
        / "more_good_spectras"
        / "Corrected_Spectra",
        "New Spectra": PROJECT_ROOT / "exploration" / "new_spectra",
        "Shlomo Raw Spectra": PROJECT_ROOT / "exploration" / "spectra_from_shlomo",
        "Sample Data (Good Fit)": PROJECT_ROOT
        / "exploration"
        / "sample_data"
        / "good_fit",
        "Sample Data (Bad Fit)": PROJECT_ROOT
        / "exploration"
        / "sample_data"
        / "bad_fit",
        "Exploration Measurements": PROJECT_ROOT / "exploration" / "measurements",
    }

    selected_source = st.selectbox(
        "Select Source", list(spectrum_sources.keys()), key="autofit_source_main"
    )

    source_path = spectrum_sources[selected_source]

    # Log path resolution for debugging
    logger.debug(f"📂 Selected source: {selected_source}")
    logger.debug(f"   Resolved path: {source_path}")
    logger.debug(f"   Path exists: {source_path.exists()}")
    logger.debug(f"   PROJECT_ROOT: {PROJECT_ROOT}")

    # Fallback: if path doesn't exist, try resolving from current working directory
    if not source_path.exists():
        # Try alternative resolution if primary path doesn't exist
        # This handles cases where working directory differs in production
        cwd_path = (
            Path.cwd() / source_path.relative_to(PROJECT_ROOT)
            if source_path.is_relative_to(PROJECT_ROOT)
            else source_path
        )
        if cwd_path != source_path and cwd_path.exists():
            logger.warning(f"⚠️ Primary path not found, using fallback: {cwd_path}")
            source_path = cwd_path
        # Also try if it's a relative path from PROJECT_ROOT that might resolve differently
        elif not source_path.is_absolute():
            abs_path = PROJECT_ROOT / source_path
            if abs_path.exists():
                source_path = abs_path

    if source_path.exists():
        if selected_source in ["Sample Data (Good Fit)", "Sample Data (Bad Fit)"]:
            spectrum_files = []
            for subdir in sorted(source_path.iterdir()):
                if subdir.is_dir():
                    for f in subdir.glob("(Run)spectra_*.txt"):
                        if "_BestFit" not in f.name:
                            spectrum_files.append(f)
        elif selected_source == "New Spectra":
            # New spectra are organized in subfolders, similar to good/bad fit
            spectrum_files = []
            for subdir in sorted(source_path.iterdir()):
                if subdir.is_dir():
                    for f in subdir.glob("(Run)spectra_*.txt"):
                        if "_BestFit" not in f.name:
                            spectrum_files.append(f)
        else:
            spectrum_files = sorted(
                [
                    f
                    for f in source_path.glob("(Run)spectra_*.txt")
                    if "_BestFit" not in f.name
                ]
            )
    else:
        spectrum_files = []
        st.warning(f"⚠️ Source path not found: `{source_path}`")
        st.info(
            f"💡 **Debug Info:**\n- PROJECT_ROOT: `{PROJECT_ROOT}`\n- Working Directory: `{Path.cwd()}`\n- Path exists: `{source_path.exists()}`"
        )
        logger.warning(f"⚠️ Source path not found: {source_path}")
        logger.warning(f"   PROJECT_ROOT: {PROJECT_ROOT}")
        logger.warning(f"   Working directory: {Path.cwd()}")

    if spectrum_files:
        selected_file = st.selectbox(
            f"Select Spectrum ({len(spectrum_files)} files)",
            spectrum_files,
            format_func=lambda x: x.name,
            key="autofit_file_main",
        )
        # Store in session state for access in main area
        st.session_state.selected_file = selected_file

        st.markdown("---")
        st.markdown("### 📏 Display Settings")

        wavelength_range = st.slider(
            "Wavelength Range (nm)",
            400,
            1200,
            (600, 1120),  # Default matches LTA analysis region (600-1120 nm)
            step=10,
            key="wl_range_main",
        )
        # Store in session state
        st.session_state.wavelength_range = wavelength_range
        st.session_state.run_autofit = False  # Reset on file change
        # Store in session state
        st.session_state.wavelength_range = wavelength_range

        # Vertical stretch control to make waves more visible
        zoom_percentage = st.slider(
            "Zoom Level (%)",
            0,
            100,
            0,  # Default to 0 (full view, no zoom)
            step=1,
            key="zoom_percentage_main",
            help="Zoom in to make wave variations more visible. 0% = full view (no zoom), 100% = maximum zoom (waves most visible). Sliding right zooms in more.",
        )
        # Convert percentage to stretch factor: 0% = full view (1.2), 100% = max zoom
        # Higher percentage = more zoom = smaller stretch factor
        # We'll calculate the actual max zoom dynamically based on data, but use a very small
        # minimum stretch to allow aggressive zooming
        min_stretch = 0.001  # Very small stretch for maximum zoom (allows zoom to continue working)
        vertical_stretch = 1.2 - (zoom_percentage / 100) * (1.2 - min_stretch)
        st.session_state.vertical_stretch = vertical_stretch
        st.session_state.zoom_percentage = zoom_percentage
        # Display current zoom level
        if zoom_percentage == 0:
            st.caption(f"📊 **Full View** (no zoom)")
        elif zoom_percentage == 100:
            st.caption(f"🔍 **Maximum Zoom** (waves most visible)")
        else:
            st.caption(f"🔍 **Zoom: {zoom_percentage}%**")

        show_residual = st.checkbox(
            "Show Residual Plot", value=True, key="show_res_main"
        )

        # Check for corresponding BestFit file
        bestfit_file = None
        selected_path = Path(selected_file)
        if "_BestFit" not in selected_path.name:
            # Try to find corresponding BestFit file
            bestfit_name = selected_path.name.replace(".txt", "_BestFit.txt")

            # 1. Try to find corresponding BestFit file in the same directory
            bestfit_path = selected_path.parent / bestfit_name
            if bestfit_path.exists():
                bestfit_file = bestfit_path
                st.session_state.bestfit_file = str(bestfit_file)

            # 2. Also check in a sibling 'BestFit' directory (common in 'More Good Spectras' structure)
            if not bestfit_file:
                sibling_bestfit_path = (
                    selected_path.parent.parent / "BestFit" / bestfit_name
                )
                if sibling_bestfit_path.exists():
                    bestfit_file = sibling_bestfit_path
                    st.session_state.bestfit_file = str(bestfit_file)

            # 3. Also check if we're in a new spectra folder - BestFit might have slightly different naming
            if not bestfit_file and selected_source == "New Spectra":
                # Try alternative patterns for BestFit files
                for alt_file in selected_path.parent.glob("*_BestFit.txt"):
                    # Check if the base name matches (before timestamp)
                    base_name = selected_path.stem
                    alt_base = alt_file.stem.replace("_BestFit", "")
                    # Extract the timestamp part (e.g., "21-47-05-763")
                    base_match = re.search(r"(\d{2}-\d{2}-\d{2}-\d+)", base_name)
                    alt_match = re.search(r"(\d{2}-\d{2}-\d{2}-\d+)", alt_base)
                    if (
                        base_match
                        and alt_match
                        and base_match.group(1) == alt_match.group(1)
                    ):
                        bestfit_file = alt_file
                        st.session_state.bestfit_file = str(bestfit_file)
                        break

        # Toggle to show LTA BestFit
        show_bestfit = False
        show_both_theoretical = False
        if bestfit_file:
            st.markdown("---")
            st.markdown("### 🔬 LTA BestFit Comparison")
            # Default to "Both" when a BestFit file exists
            # Reset view mode to "Both" when switching to a new spectrum with BestFit
            current_selected_file = str(selected_path)
            if (
                "last_selected_file" not in st.session_state
                or st.session_state.last_selected_file != current_selected_file
            ):
                # File has changed - reset view mode to "Both" for new spectrum with BestFit
                st.session_state.theoretical_view_mode = "Both (PyElli + BestFit)"
            st.session_state.last_selected_file = current_selected_file
            view_mode = st.radio(
                "Theoretical Spectrum View",
                [
                    "PyElli Theoretical Only",
                    "LTA BestFit Only",
                    "Both (PyElli + BestFit)",
                ],
                key="theoretical_view_mode",
                help="Compare PyElli-generated theoretical spectra with LTA BestFit results",
            )
            show_bestfit = view_mode in ["LTA BestFit Only", "Both (PyElli + BestFit)"]
            show_both_theoretical = view_mode == "Both (PyElli + BestFit)"
            st.session_state.show_bestfit = show_bestfit
            st.session_state.show_both_theoretical = show_both_theoretical
        else:
            st.session_state.bestfit_file = None
            st.session_state.show_bestfit = False
            st.session_state.show_both_theoretical = False

        st.markdown("---")
        st.markdown("### 🧪 Materials Configuration")

        # Get all available material files (CSV only) + custom uploaded materials
        builtin_material_files = sorted([f.name for f in MATERIALS_PATH.glob("*.csv")])
        custom_material_names = [
            f"📤 {name}" for name in st.session_state.custom_materials.keys()
        ]
        material_files = builtin_material_files + custom_material_names

        # Guard against empty material_files to prevent Streamlit crash
        if not material_files:
            st.warning(
                f"⚠️ No material files found in {MATERIALS_PATH}. Please upload a custom material file or ensure CSV files are present in the Materials directory."
            )
            # Use defaults and allow file upload
            st.session_state.selected_lipid_material = (
                PyElliGridSearch.DEFAULT_LIPID_FILE
            )
            st.session_state.selected_water_material = (
                PyElliGridSearch.DEFAULT_WATER_FILE
            )
            st.session_state.selected_mucus_material = (
                PyElliGridSearch.DEFAULT_MUCUS_FILE
            )
            st.session_state.selected_substratum_material = (
                PyElliGridSearch.DEFAULT_SUBSTRATUM_FILE
            )

            # Still allow file upload even when directory is empty
            st.markdown("**📤 Upload Custom Material**")
            st.caption(
                "CSV format: wavelength_nm, n (refractive index), k (extinction)"
            )

            uploaded_file = st.file_uploader(
                "Upload Material CSV",
                type=["csv"],
                key="material_file_upload",
                help="Upload a CSV file with columns: wavelength (nm), n, k",
            )

            if uploaded_file is not None:
                is_valid, message, df = validate_material_file(uploaded_file)

                if is_valid:
                    st.success(message)

                    # Show preview
                    with st.expander("Preview Data", expanded=False):
                        st.dataframe(df.head(10), use_container_width=True)
                        st.caption(f"Showing first 10 of {len(df)} rows")

                    # Add to custom materials
                    if st.button("✅ Add Material", key="add_custom_material_btn"):
                        material_name = uploaded_file.name
                        st.session_state.custom_materials[material_name] = df
                        st.success(f'Added "{material_name}" to available materials!')
                        st.rerun()
                else:
                    st.error(message)

            # Show currently loaded custom materials
            if st.session_state.custom_materials:
                st.markdown("---")
                st.markdown("**Custom Materials Loaded:**")
                for name in st.session_state.custom_materials.keys():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        df = st.session_state.custom_materials[name]
                        st.caption(f"📤 {name} ({len(df)} pts)")
                    with col2:
                        if st.button("🗑️", key=f"remove_{name}", help=f"Remove {name}"):
                            del st.session_state.custom_materials[name]
                            st.rerun()
        else:
            with st.expander("Configure Layer Materials", expanded=False):
                st.caption("Select optical property files for each tear film layer")

                # Helper to get current selection index
                def get_material_index(selected: str, options: list) -> int:
                    if selected in options:
                        return options.index(selected)
                    # Check if it's a custom material
                    custom_name = f"📤 {selected}"
                    if custom_name in options:
                        return options.index(custom_name)
                    return 0

                # Lipid material selection
                lipid_idx = get_material_index(
                    st.session_state.selected_lipid_material, material_files
                )
                selected_lipid = st.selectbox(
                    "Lipid Layer",
                    material_files,
                    index=lipid_idx,
                    key="material_lipid_select",
                    help="Material for the outermost lipid layer",
                )
                # Strip emoji prefix for custom materials when storing
                st.session_state.selected_lipid_material = (
                    selected_lipid.replace("📤 ", "")
                    if selected_lipid.startswith("📤")
                    else selected_lipid
                )

                # Aqueous material selection
                water_idx = get_material_index(
                    st.session_state.selected_water_material, material_files
                )
                selected_water = st.selectbox(
                    "Aqueous Layer",
                    material_files,
                    index=water_idx,
                    key="material_water_select",
                    help="Material for the aqueous (water) layer",
                )
                st.session_state.selected_water_material = (
                    selected_water.replace("📤 ", "")
                    if selected_water.startswith("📤")
                    else selected_water
                )

                # Mucus material selection
                mucus_idx = get_material_index(
                    st.session_state.selected_mucus_material, material_files
                )
                selected_mucus = st.selectbox(
                    "Mucus Layer",
                    material_files,
                    index=mucus_idx,
                    key="material_mucus_select",
                    help="Material for the mucus layer",
                )
                st.session_state.selected_mucus_material = (
                    selected_mucus.replace("📤 ", "")
                    if selected_mucus.startswith("📤")
                    else selected_mucus
                )

                # Substratum material selection
                substratum_idx = get_material_index(
                    st.session_state.selected_substratum_material, material_files
                )
                selected_substratum = st.selectbox(
                    "Substratum (Cornea)",
                    material_files,
                    index=substratum_idx,
                    key="material_substratum_select",
                    help="Material for the corneal epithelium substrate",
                )
                st.session_state.selected_substratum_material = (
                    selected_substratum.replace("📤 ", "")
                    if selected_substratum.startswith("📤")
                    else selected_substratum
                )

                # File upload section
                st.markdown("---")
                st.markdown("**📤 Upload Custom Material**")
                st.caption(
                    "CSV format: wavelength_nm, n (refractive index), k (extinction)"
                )

                uploaded_file = st.file_uploader(
                    "Upload Material CSV",
                    type=["csv"],
                    key="material_file_upload",
                    help="Upload a CSV file with columns: wavelength (nm), n, k",
                )

                if uploaded_file is not None:
                    is_valid, message, df = validate_material_file(uploaded_file)

                    if is_valid:
                        st.success(message)

                        # Show preview
                        with st.expander("Preview Data", expanded=False):
                            st.dataframe(df.head(10), use_container_width=True)
                            st.caption(f"Showing first 10 of {len(df)} rows")

                        # Add to custom materials
                        if st.button("✅ Add Material", key="add_custom_material_btn"):
                            material_name = uploaded_file.name
                            st.session_state.custom_materials[material_name] = df
                            st.success(
                                f'Added "{material_name}" to available materials!'
                            )
                            st.rerun()
                    else:
                        st.error(message)

                # Show currently loaded custom materials
                if st.session_state.custom_materials:
                    st.markdown("---")
                    st.markdown("**Custom Materials Loaded:**")
                    for name in st.session_state.custom_materials.keys():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            df = st.session_state.custom_materials[name]
                            st.caption(f"📤 {name} ({len(df)} pts)")
                        with col2:
                            if st.button(
                                "🗑️", key=f"remove_{name}", help=f"Remove {name}"
                            ):
                                del st.session_state.custom_materials[name]
                                st.rerun()

        st.markdown("---")
        st.markdown("### 🔧 Parameters")
        st.caption("Adjust manually or use Grid Search to find best values")

        # Handle pending_update: set forced values and increment widget key version
        if st.session_state.pending_update is not None:
            # Set forced values from grid search
            st.session_state.forced_lipid = float(
                st.session_state.pending_update["lipid"]
            )
            st.session_state.forced_aqueous = float(
                st.session_state.pending_update["aqueous"]
            )
            st.session_state.forced_mucus = float(
                st.session_state.pending_update["mucus"]
            )
            st.session_state.autofit_results = st.session_state.pending_update[
                "results"
            ]
            # Increment widget key version to force sliders to reset
            st.session_state.widget_key_version += 1
            # Clear pending_update
            st.session_state.pending_update = None

        # Get slider initial values - use forced values if set, otherwise use defaults
        # Defaults match standard ADOM ranges
        if st.session_state.forced_lipid is not None:
            lipid_init = int(round(np.clip(st.session_state.forced_lipid, 9, 250)))
        else:
            lipid_init = 150  # Default within 9-250 range

        if st.session_state.forced_aqueous is not None:
            aqueous_init = int(
                round(np.clip(st.session_state.forced_aqueous, 800, 12000))
            )
        else:
            aqueous_init = 2000  # Default within 800-12000 range

        if st.session_state.forced_mucus is not None:
            mucus_init = int(round(np.clip(st.session_state.forced_mucus, 600, 7000)))
        else:
            mucus_init = 2000  # Default within 600-7000 range

        # Use versioned keys so Streamlit treats these as new widgets when version changes
        # This forces the slider to use the value parameter instead of cached session state
        key_version = st.session_state.widget_key_version

        # Slider ranges match standard ADOM ranges
        current_lipid = st.slider(
            "Lipid (nm)",
            9,
            250,
            value=lipid_init,
            key=f"param_lipid_v{key_version}",
            step=5,
        )
        current_aqueous = st.slider(
            "Aqueous (nm)",
            800,
            12000,
            value=aqueous_init,
            key=f"param_aqueous_v{key_version}",
            step=50,
        )
        # Interface roughness in Angstroms (extended range: 600-7000)
        current_mucus = st.slider(
            "Interface Roughness (Å)",
            600,
            7000,
            value=mucus_init,
            key=f"param_mucus_v{key_version}",
            step=50,
        )

        # Store current slider values in session state for access elsewhere
        st.session_state.current_lipid = current_lipid
        st.session_state.current_aqueous = current_aqueous
        st.session_state.current_mucus = current_mucus

        st.markdown("---")
        st.markdown(
            '<p class="section-header">📊 Analysis Parameters</p>',
            unsafe_allow_html=True,
        )
        st.caption("Signal processing for spectrum analysis")

        # Safe index lookup helper (returns 0 if value not in options)
        def safe_index(options: list, value, default_idx: int = 0) -> int:
            try:
                return options.index(value)
            except ValueError:
                return default_idx

        # Smoothing controls
        smoothing_type_options = ["none", "boxcar", "gaussian"]
        smoothing_type = st.radio(
            "Smoothing Type",
            options=smoothing_type_options,
            index=safe_index(
                smoothing_type_options, st.session_state.get("smoothing_type", "none")
            ),
            horizontal=True,
            key="smoothing_type_radio",
        )
        st.session_state.smoothing_type = smoothing_type

        # Conditional smoothing parameters based on type
        if smoothing_type == "boxcar":
            boxcar_width_nm = st.slider(
                "Boxcar Width (nm)",
                min_value=5.0,
                max_value=50.0,
                value=st.session_state.get("boxcar_width_nm", 17.0),
                step=1.0,
                format="%.0f",
                key="boxcar_width_slider",
            )
            st.session_state.boxcar_width_nm = boxcar_width_nm

            boxcar_passes_options = [1, 2, 3]
            boxcar_passes = st.selectbox(
                "Boxcar Passes",
                options=boxcar_passes_options,
                index=safe_index(
                    boxcar_passes_options, st.session_state.get("boxcar_passes", 1)
                ),
                key="boxcar_passes_select",
            )
            st.session_state.boxcar_passes = boxcar_passes
        elif smoothing_type == "gaussian":
            gaussian_kernel_options = [7, 9, 11]
            gaussian_kernel = st.selectbox(
                "Gaussian Kernel Size",
                options=gaussian_kernel_options,
                index=safe_index(
                    gaussian_kernel_options, st.session_state.get("gaussian_kernel", 11)
                ),
                key="gaussian_kernel_select",
            )
            st.session_state.gaussian_kernel = gaussian_kernel

        st.markdown("---")
        st.markdown(
            '<p class="section-header">⚙️ Search Range Settings</p>',
            unsafe_allow_html=True,
        )
        st.caption("Optimized ranges based on reverse engineering analysis")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            # Lipid: Standard ADOM range 9-250nm
            lipid_min = st.number_input(
                "Lipid Min",
                min_value=9.0,
                max_value=249.0,
                step=1.0,
                value=st.session_state.get("grid_lipid_min", 9.0),
                key="grid_lipid_min",
            )
            lipid_max = st.number_input(
                "Lipid Max",
                min_value=10.0,
                max_value=250.0,
                step=1.0,
                value=st.session_state.get("grid_lipid_max", 250.0),
                key="grid_lipid_max",
            )
            lipid_step = st.number_input(
                "Lipid Step",
                min_value=1.0,
                max_value=50.0,
                step=1.0,
                value=st.session_state.get("grid_lipid_step", 5.0),
                key="grid_lipid_step",
            )
        with col_b:
            # Aqueous: Standard ADOM range 800-12000nm
            aqueous_min = st.number_input(
                "Aqueous Min",
                min_value=800.0,
                max_value=11999.0,
                step=10.0,
                value=st.session_state.get("grid_aq_min", 800.0),
                key="grid_aq_min",
            )
            aqueous_max = st.number_input(
                "Aqueous Max",
                min_value=801.0,
                max_value=12000.0,
                step=10.0,
                value=st.session_state.get("grid_aq_max", 12000.0),
                key="grid_aq_max",
            )
            aqueous_step = st.number_input(
                "Aqueous Step",
                min_value=10.0,
                max_value=1000.0,
                step=10.0,
                value=st.session_state.get("grid_aq_step", 200.0),
                key="grid_aq_step",
            )
        with col_c:
            # Interface roughness: Standard ADOM range 600-7000 Å
            mucus_min = st.number_input(
                "Roughness Min (Å)",
                min_value=600.0,
                max_value=6999.0,
                step=10.0,
                value=st.session_state.get("grid_mu_min", 600.0),
                key="grid_mu_min",
            )
            mucus_max = st.number_input(
                "Roughness Max (Å)",
                min_value=601.0,
                max_value=7000.0,
                step=10.0,
                value=st.session_state.get("grid_mu_max", 7000.0),
                key="grid_mu_max",
            )
            mucus_step = st.number_input(
                "Roughness Step (Å)",
                min_value=10.0,
                max_value=500.0,
                step=10.0,
                value=st.session_state.get("grid_mu_step", 100.0),
                key="grid_mu_step",
            )

        # Cycle jump detection thresholds (used in Amplitude Analysis display)
        st.markdown("---")
        st.markdown(
            '<p class="section-header">⚠️ Cycle Jump Thresholds</p>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Thresholds for flagging systematic drift (cycle jump candidates). Used in Amplitude Analysis."
        )
        drift_th_peak = st.number_input(
            "Peak drift slope threshold",
            min_value=0.0,
            max_value=0.5,
            step=0.01,
            format="%.2f",
            value=float(st.session_state.get("drift_peak_slope_threshold", 0.05)),
            key="drift_peak_slope_threshold",
            help="|slope| above this (nm/nm) flags peak position drift",
        )
        drift_th_amp = st.number_input(
            "Amplitude drift slope threshold",
            min_value=0.0,
            max_value=0.5,
            step=0.005,
            format="%.3f",
            value=float(st.session_state.get("drift_amplitude_slope_threshold", 0.01)),
            key="drift_amplitude_slope_threshold",
            help="|slope| above this flags amplitude ratio drift",
        )
        drift_th_r2 = st.number_input(
            "R² threshold (drift)",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            format="%.2f",
            value=float(st.session_state.get("drift_r_squared_threshold", 0.5)),
            key="drift_r_squared_threshold",
            help="R² above this indicates systematic (not random) drift",
        )

        # Grid Search settings and button
        st.markdown("---")
        st.markdown("### 🔍 Grid Search")

        run_autofit = st.button(
            "🚀 Run Grid Search",
            width="stretch",
            type="primary",
            key="run_grid_search_btn",
        )
        # Store button state in session state
        if run_autofit:
            st.session_state.run_autofit = True
        else:
            st.session_state.run_autofit = False
    else:
        st.session_state.selected_file = None
        st.session_state.run_autofit = False
        st.session_state.wavelength_range = (
            600,
            1120,
        )  # Default matches LTA analysis region (600-1120 nm)
        st.info("👈 No spectrum files found")


# =============================================================================
# Main Content
# =============================================================================

st.markdown(
    """
<div style="text-align: center; padding: 16px 0 32px 0;"></div>
""",
    unsafe_allow_html=True,
)

# Custom Plotly theme - Clean white
PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "font": {"family": "DM Sans, sans-serif", "color": "#334155"},
        "title": {"font": {"family": "DM Sans, sans-serif", "color": "#1e40af"}},
        "xaxis": {
            "gridcolor": "#e5e7eb",
            "linecolor": "#d1d5db",
            "tickfont": {"color": "#6b7280"},
            "title": {"font": {"color": "#374151"}},
        },
        "yaxis": {
            "gridcolor": "#e5e7eb",
            "linecolor": "#d1d5db",
            "tickfont": {"color": "#6b7280"},
            "title": {"font": {"color": "#374151"}},
        },
        "legend": {
            "bgcolor": "#ffffff",
            "bordercolor": "#e5e7eb",
            "font": {"color": "#374151"},
        },
    }
}

# Color palette - Light theme friendly
COLORS = {
    "primary": "#1e40af",  # Deep blue
    "secondary": "#7c3aed",  # Purple
    "accent": "#0891b2",  # Cyan
    "success": "#059669",  # Green
    "warning": "#d97706",  # Amber
    "measured": "#2563eb",  # Blue for measured data
    "theoretical": "#7c3aed",  # Purple for theoretical
    "bestfit": "#db2777",  # Pink for bestfit
    "residual": "#d97706",  # Amber for residual
}

# Get sidebar values from session state (set in sidebar section above)
selected_file = st.session_state.get("selected_file", None)
run_autofit = st.session_state.get("run_autofit", False)
wavelength_range = st.session_state.get("wavelength_range", (600, 1120))
# Get current parameter values from the sliders (stored in session state by sidebar)
current_lipid = st.session_state.get("current_lipid", 150)
current_aqueous = st.session_state.get("current_aqueous", 2000)
current_mucus = st.session_state.get("current_mucus", 2000)
show_residual = st.session_state.get("show_res_main", True)
show_bestfit = st.session_state.get("show_bestfit", False)
show_both_theoretical = st.session_state.get("show_both_theoretical", False)

# Three tabs: Spectrum Comparison, Amplitude Analysis, and Quality Metrics
tabs = st.tabs(
    [
        "📊 Spectrum Comparison",
        "📈 Amplitude Analysis",
        "🔍 Quality Metrics",
    ]
)

# Create placeholder INSIDE the first tab - this appears right below tab headers
with tabs[0]:
    progress_placeholder = st.empty()
    # Show completion message if grid search was completed
    if st.session_state.get(
        "last_run_elapsed_s"
    ) is not None and not st.session_state.get("run_autofit", False):
        mins, secs = divmod(int(st.session_state.last_run_elapsed_s), 60)
        progress_placeholder.success(
            f"✅ Grid search completed in **{mins:02d}:{secs:02d}** ({st.session_state.last_run_elapsed_s:.1f}s)"
        )

# =============================================================================
# Shared Data Loading and Processing (runs once, used by both tabs)
# =============================================================================

if selected_file and Path(selected_file).exists():
    try:
        wavelengths, measured = load_measured_spectrum(selected_file)

        wl_mask = (wavelengths >= wavelength_range[0]) & (
            wavelengths <= wavelength_range[1]
        )
        wl_display = wavelengths[wl_mask]
        meas_display = measured[wl_mask]

        grid_search = PyElliGridSearch(
            MATERIALS_PATH,
            lipid_file=st.session_state.selected_lipid_material,
            water_file=st.session_state.selected_water_material,
            mucus_file=st.session_state.selected_mucus_material,
            substratum_file=st.session_state.selected_substratum_material,
            custom_materials=st.session_state.custom_materials,
        )

        # Run grid search if button pressed
        if run_autofit:
            start_time = time.perf_counter()

            # Capture session state values BEFORE entering thread (main thread)
            # This ensures we use the correct values even if session_state isn't thread-safe
            captured_lipid_range = (
                st.session_state.get("grid_lipid_min", 9.0),
                st.session_state.get("grid_lipid_max", 250.0),
                st.session_state.get("grid_lipid_step", 5.0),
            )
            captured_aqueous_range = (
                st.session_state.get("grid_aq_min", 800.0),
                st.session_state.get("grid_aq_max", 12000.0),
                st.session_state.get("grid_aq_step", 200.0),
            )
            captured_roughness_range = (
                st.session_state.get("grid_mu_min", 600.0),
                st.session_state.get("grid_mu_max", 7000.0),
                st.session_state.get("grid_mu_step", 100.0),
            )

            def _run_search():
                return grid_search.run_grid_search(
                    wavelengths,
                    measured,
                    lipid_range=captured_lipid_range,
                    aqueous_range=captured_aqueous_range,
                    roughness_range=captured_roughness_range,
                    top_k=10,
                    enable_roughness=True,
                    search_strategy="Dynamic Search",
                    max_combinations=30000,
                )

            # Show progress in the placeholder above the tabs
            progress_placeholder.progress(0, text="⏳ Starting grid search...")

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_search)
                # Update progress bar and timer while running
                while future.running():
                    elapsed = time.perf_counter() - start_time
                    mins, secs = divmod(int(elapsed), 60)
                    # Update progress bar (estimate ~5 min max)
                    progress_pct = min(0.99, elapsed / 300)
                    progress_placeholder.progress(
                        progress_pct,
                        text=f"⏳ Running grid search... {mins:02d}:{secs:02d} elapsed",
                    )
                    time.sleep(0.5)
                try:
                    results = future.result()
                except Exception as exc:
                    progress_placeholder.empty()
                    st.error(f"Grid search failed: {exc}")
                    raise

            elapsed = time.perf_counter() - start_time
            mins, secs = divmod(int(elapsed), 60)
            # Clear progress bar and show completion time
            progress_placeholder.empty()
            st.session_state.last_run_elapsed_s = elapsed

            if results:
                best = results[0]
                # Store results immediately so plots can use them
                st.session_state.autofit_results = results
                # Set selected rank to 1 (best result) when grid search completes
                st.session_state.selected_rank = 1
                # Store the update to be applied before widgets are created on next run
                st.session_state.pending_update = {
                    "lipid": best.lipid_nm,
                    "aqueous": best.aqueous_nm,
                    "mucus": best.mucus_nm,
                    "results": results,
                }
                # Rerun to update sliders and plots
                st.rerun()
            else:
                st.warning("⚠️ No valid fits found")

        # Use current slider values for display
        # The sliders are now set from forced values when grid search completes or rank changes
        display_lipid = current_lipid
        display_aqueous = current_aqueous
        display_mucus = current_mucus  # Already in Angstroms from slider

        # Load BestFit spectrum if available
        bestfit_wl = None
        bestfit_refl = None
        bestfit_display = None
        bestfit_score_result = None
        bestfit_correlation = 0.0

        bestfit_file_path = st.session_state.get("bestfit_file", None)
        show_bestfit = st.session_state.get("show_bestfit", False)
        show_both_theoretical = st.session_state.get("show_both_theoretical", False)

        if bestfit_file_path and Path(bestfit_file_path).exists():
            try:
                bestfit_wl, bestfit_refl = load_bestfit_spectrum(
                    Path(bestfit_file_path)
                )
                # Interpolate BestFit to match wavelengths
                bestfit_interp = np.interp(wavelengths, bestfit_wl, bestfit_refl)
                # Align BestFit with measured
                bestfit_aligned = grid_search._align_spectra(
                    measured,
                    bestfit_interp,
                    focus_min=600.0,
                    focus_max=1120.0,
                    wavelengths=wavelengths,
                )
                bestfit_display = bestfit_aligned[wl_mask]

                # Calculate metrics for BestFit
                bestfit_score_result = calculate_peak_based_score(
                    wl_display, meas_display, bestfit_display
                )

                if np.std(meas_display) > 1e-10 and np.std(bestfit_display) > 1e-10:
                    bestfit_correlation = float(
                        np.corrcoef(meas_display, bestfit_display)[0, 1]
                    )
                    if np.isnan(bestfit_correlation):
                        bestfit_correlation = 0.0
            except Exception as e:
                logger.warning(f"Error loading BestFit spectrum: {e}")
                bestfit_display = None

        # Calculate theoretical spectrum with current params
        # Convert roughness from Angstroms to nm for calculate_theoretical_spectrum
        roughness_nm = display_mucus / 10.0
        theoretical = grid_search.calculate_theoretical_spectrum(
            wavelengths, display_lipid, display_aqueous, roughness_nm
        )

        # Use simple proportional scaling for alignment (same as grid search worker)
        # Align using the wavelength range from the slider
        wl_min, wl_max = wavelength_range
        focus_mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
        if focus_mask.sum() > 0:
            meas_focus = measured[focus_mask]
            theo_focus = theoretical[focus_mask]
            if np.std(theo_focus) > 1e-10:
                # Simple proportional scaling (same as grid search worker)
                scale = np.dot(meas_focus, theo_focus) / np.dot(theo_focus, theo_focus)
                theoretical_aligned = theoretical * scale
            else:
                theoretical_aligned = theoretical
        else:
            theoretical_aligned = theoretical

        theoretical_display = theoretical_aligned[wl_mask]

        # Score on the SLIDER wavelength range (600-1120 nm by default)
        score_result = calculate_peak_based_score(
            wl_display, meas_display, theoretical_display
        )

        # Calculate correlation on the slider wavelength range
        if np.std(meas_display) > 1e-10 and np.std(theoretical_display) > 1e-10:
            correlation = float(np.corrcoef(meas_display, theoretical_display)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        # Determine which theoretical to show based on toggle
        if show_bestfit and not show_both_theoretical and bestfit_display is not None:
            # Show only BestFit, replace theoretical
            display_theoretical = bestfit_display
            display_score_result = bestfit_score_result
            display_correlation = bestfit_correlation
            theoretical_label = "LTA BestFit"
        else:
            # Show PyElli theoretical (or both will be handled in plot)
            display_theoretical = theoretical_display
            display_score_result = score_result
            display_correlation = correlation
            theoretical_label = f"Theoretical (L={display_lipid}, A={display_aqueous}, R={display_mucus:.0f}Å)"

        # Store computed values for use in tabs
        computed_data = {
            "wavelengths": wavelengths,
            "measured": measured,
            "wl_display": wl_display,
            "meas_display": meas_display,
            "theoretical_display": theoretical_display,  # PyElli theoretical
            "bestfit_display": bestfit_display,  # LTA BestFit
            "display_theoretical": display_theoretical,  # What to actually display
            "score_result": display_score_result,
            "pyelli_score_result": score_result,  # Keep PyElli score
            "bestfit_score_result": bestfit_score_result,  # Keep BestFit score
            "correlation": display_correlation,
            "pyelli_correlation": correlation,  # Keep PyElli correlation
            "bestfit_correlation": bestfit_correlation,  # Keep BestFit correlation
            "display_lipid": display_lipid,
            "display_aqueous": display_aqueous,
            "display_mucus": display_mucus,
            "grid_search": grid_search,
            "theoretical_label": theoretical_label,
            "show_bestfit": show_bestfit,
            "show_both_theoretical": show_both_theoretical,
            "has_bestfit": bestfit_display is not None,
        }
    except Exception as e:
        st.error(f"Error loading spectrum: {e}")
        computed_data = None
else:
    computed_data = None

# =============================================================================
# Tab 1: Spectrum Comparison
# =============================================================================

with tabs[0]:
    if computed_data:
        # Display filename at the top
        st.markdown(f"### 📈 `{Path(selected_file).name}`")

        # Spectrum comparison plot (before metrics)
        has_bestfit_plot = (
            computed_data.get("has_bestfit", False)
            and computed_data.get("bestfit_display") is not None
        )
        show_both_plot = (
            computed_data.get("show_both_theoretical", False) and has_bestfit_plot
        )

        if show_residual:
            fig = make_subplots(
                rows=2,
                cols=1,
                row_heights=[0.7, 0.3],
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=["Spectrum Comparison", "Residual"],
            )

            # Always show measured
            fig.add_trace(
                go.Scatter(
                    x=computed_data["wl_display"],
                    y=computed_data["meas_display"],
                    mode="lines",
                    name="Measured",
                    line=dict(color="#2563eb", width=2.5),
                ),
                row=1,
                col=1,
            )

            # Show theoretical based on toggle
            if show_both_plot and has_bestfit_plot:
                # Show both PyElli and BestFit
                fig.add_trace(
                    go.Scatter(
                        x=computed_data["wl_display"],
                        y=computed_data["theoretical_display"],
                        mode="lines",
                        name=f"PyElli (L={computed_data['display_lipid']}, A={computed_data['display_aqueous']}, R={computed_data['display_mucus']:.0f}Å)",
                        line=dict(color="#059669", width=2.5, dash="dash"),
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=computed_data["wl_display"],
                        y=computed_data["bestfit_display"],
                        mode="lines",
                        name="LTA BestFit",
                        line=dict(color="#db2777", width=2.5, dash="dot"),
                    ),
                    row=1,
                    col=1,
                )
                # Residual is measured - displayed theoretical (PyElli if only one shown)
                residual = (
                    computed_data["meas_display"] - computed_data["theoretical_display"]
                )
            else:
                # Show the selected one (either PyElli or BestFit)
                fig.add_trace(
                    go.Scatter(
                        x=computed_data["wl_display"],
                        y=computed_data["display_theoretical"],
                        mode="lines",
                        name=computed_data["theoretical_label"],
                        line=dict(
                            color="#059669"
                            if not computed_data.get("show_bestfit", False)
                            else "#db2777",
                            width=2.5,
                            dash="dash",
                        ),
                    ),
                    row=1,
                    col=1,
                )
                residual = (
                    computed_data["meas_display"] - computed_data["display_theoretical"]
                )

            fig.add_trace(
                go.Scatter(
                    x=computed_data["wl_display"],
                    y=residual,
                    mode="lines",
                    name="Residual",
                    line=dict(color="#d97706", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(251, 191, 36, 0.15)",
                ),
                row=2,
                col=1,
            )
            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

            fig.update_xaxes(title_text="Wavelength (nm)", row=2, col=1)
            fig.update_yaxes(title_text="Reflectance", row=1, col=1)
            fig.update_yaxes(title_text="Δ", row=2, col=1)
            height = 500

            # Apply vertical stretch to y-axis for main plot (row 1)
            vertical_stretch = st.session_state.get(
                "vertical_stretch", 1.2
            )  # Default to full view
            if vertical_stretch != 1.2 and computed_data:
                # Collect all y-values from visible traces
                all_y_values = []
                all_y_values.extend(computed_data["meas_display"])
                if show_both_plot and has_bestfit_plot:
                    all_y_values.extend(computed_data["theoretical_display"])
                    all_y_values.extend(computed_data["bestfit_display"])
                else:
                    all_y_values.extend(computed_data["display_theoretical"])

                if len(all_y_values) > 0:
                    y_min = np.min(all_y_values)
                    y_max = np.max(all_y_values)
                    y_center = (y_min + y_max) / 2
                    y_range = y_max - y_min

                    # Calculate the range based on zoom percentage
                    # At 0%: full view with 20% padding (range = y_range * 1.2)
                    # At 100%: maximum zoom (range = y_range * 1.002, just data + minimal padding)
                    min_padding = y_range * 0.001  # 0.1% padding
                    min_range = y_range * 1.002  # Minimum range (data + padding)
                    max_range = y_range * 1.2  # Maximum range (full view)

                    # Interpolate range based on zoom percentage (0% = max_range, 100% = min_range)
                    # Higher percentage = smaller range = more zoom
                    target_range = max_range - (zoom_percentage / 100) * (
                        max_range - min_range
                    )

                    # Calculate centered range
                    y_axis_min = y_center - target_range / 2
                    y_axis_max = y_center + target_range / 2

                    # Ensure all data is visible - only expand if we would cut off data
                    # Check if we're cutting off data from below
                    if y_axis_min > y_min - min_padding:
                        # We need to expand downward
                        y_axis_min = y_min - min_padding
                    # Check if we're cutting off data from above
                    if y_axis_max < y_max + min_padding:
                        # We need to expand upward
                        y_axis_max = y_max + min_padding

                    # If we had to expand, the range is now larger than target
                    # This is expected when zooming in - we can't zoom beyond the data bounds
                    # The zoom will naturally stop when we hit the minimum range

                    fig.update_yaxes(range=[y_axis_min, y_axis_max], row=1, col=1)
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=computed_data["wl_display"],
                    y=computed_data["meas_display"],
                    mode="lines",
                    name="Measured",
                    line=dict(color="#2563eb", width=2.5),
                )
            )

            # Show theoretical based on toggle
            if show_both_plot and has_bestfit_plot:
                # Show both PyElli and BestFit
                fig.add_trace(
                    go.Scatter(
                        x=computed_data["wl_display"],
                        y=computed_data["theoretical_display"],
                        mode="lines",
                        name=f"PyElli (L={computed_data['display_lipid']}, A={computed_data['display_aqueous']}, R={computed_data['display_mucus']:.0f}Å)",
                        line=dict(color="#059669", width=2.5, dash="dash"),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=computed_data["wl_display"],
                        y=computed_data["bestfit_display"],
                        mode="lines",
                        name="LTA BestFit",
                        line=dict(color="#db2777", width=2.5, dash="dot"),
                    )
                )
            else:
                # Show the selected one (either PyElli or BestFit)
                fig.add_trace(
                    go.Scatter(
                        x=computed_data["wl_display"],
                        y=computed_data["display_theoretical"],
                        mode="lines",
                        name=computed_data["theoretical_label"],
                        line=dict(
                            color="#059669"
                            if not computed_data.get("show_bestfit", False)
                            else "#db2777",
                            width=2.5,
                            dash="dash",
                        ),
                    )
                )

            fig.update_xaxes(title_text="Wavelength (nm)")
            fig.update_yaxes(title_text="Reflectance")
            height = 350

        # Apply vertical stretch to y-axis to make waves more visible
        vertical_stretch = st.session_state.get(
            "vertical_stretch", 1.2
        )  # Default to full view
        if vertical_stretch != 1.2 and computed_data:
            # Collect all y-values from visible traces
            all_y_values = []
            all_y_values.extend(computed_data["meas_display"])
            if show_both_plot and has_bestfit_plot:
                all_y_values.extend(computed_data["theoretical_display"])
                all_y_values.extend(computed_data["bestfit_display"])
            else:
                all_y_values.extend(computed_data["display_theoretical"])

            if len(all_y_values) > 0:
                y_min = np.min(all_y_values)
                y_max = np.max(all_y_values)
                y_center = (y_min + y_max) / 2
                y_range = y_max - y_min

                # Calculate the range based on zoom percentage
                # At 0%: full view with 20% padding (range = y_range * 1.2)
                # At 100%: maximum zoom (range = y_range * 1.002, just data + minimal padding)
                min_padding = y_range * 0.001  # 0.1% padding
                min_range = y_range * 1.002  # Minimum range (data + padding)
                max_range = y_range * 1.2  # Maximum range (full view)

                # Interpolate range based on zoom percentage (0% = max_range, 100% = min_range)
                # Higher percentage = smaller range = more zoom
                target_range = max_range - (zoom_percentage / 100) * (
                    max_range - min_range
                )

                # Calculate centered range
                y_axis_min = y_center - target_range / 2
                y_axis_max = y_center + target_range / 2

                # Ensure all data is visible - only expand if we would cut off data
                # Check if we're cutting off data from below
                if y_axis_min > y_min - min_padding:
                    # We need to expand downward
                    y_axis_min = y_min - min_padding
                # Check if we're cutting off data from above
                if y_axis_max < y_max + min_padding:
                    # We need to expand upward
                    y_axis_max = y_max + min_padding

                # If we had to expand, the range is now larger than target
                # This is expected when zooming in - we can't zoom beyond the data bounds
                # The zoom will naturally stop when we hit the minimum range

                # Update y-axis range for main plot
                if show_residual:
                    fig.update_yaxes(range=[y_axis_min, y_axis_max], row=1, col=1)
                else:
                    fig.update_yaxes(range=[y_axis_min, y_axis_max])

        fig.update_layout(
            height=height,
            margin=dict(t=40, b=40, l=60, r=30),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        st.plotly_chart(fig, width="stretch")

        # Metrics below the plot
        st.markdown("---")
        st.markdown("### 📊 Fit Parameters & Metrics")

        # Determine which metrics to show
        has_bestfit_metrics = (
            computed_data.get("has_bestfit", False)
            and computed_data.get("bestfit_score_result") is not None
        )
        show_both_metrics = (
            computed_data.get("show_both_theoretical", False) and has_bestfit_metrics
        )

        if show_both_metrics:
            # Show both PyElli and BestFit metrics side by side
            st.markdown("#### PyElli Theoretical")
            mcols_pyelli = st.columns(5)
            with mcols_pyelli[0]:
                score_icon = (
                    "🟢"
                    if computed_data["pyelli_score_result"]["score"] >= 0.7
                    else (
                        "🟡"
                        if computed_data["pyelli_score_result"]["score"] >= 0.5
                        else "🔴"
                    )
                )
                st.metric(
                    "Score",
                    f"{score_icon} {computed_data['pyelli_score_result']['score']:.3f}",
                )
            with mcols_pyelli[1]:
                st.metric("Correlation", f"{computed_data['pyelli_correlation']:.3f}")
            with mcols_pyelli[2]:
                rmse_val = computed_data["pyelli_score_result"].get("rmse", 0.0)
                st.metric("RMSE", f"{rmse_val:.5f}")
            with mcols_pyelli[3]:
                matched_peaks = int(
                    computed_data["pyelli_score_result"].get("matched_peaks", 0)
                )
                st.metric("Matched Peaks", f"{matched_peaks}")
            with mcols_pyelli[4]:
                st.metric(
                    "Parameters",
                    f"L={computed_data['display_lipid']}, A={computed_data['display_aqueous']}, R={computed_data['display_mucus']:.0f}Å",
                )

            st.markdown("#### LTA BestFit")
            mcols_bestfit = st.columns(4)
            with mcols_bestfit[0]:
                score_icon = (
                    "🟢"
                    if computed_data["bestfit_score_result"]["score"] >= 0.7
                    else (
                        "🟡"
                        if computed_data["bestfit_score_result"]["score"] >= 0.5
                        else "🔴"
                    )
                )
                st.metric(
                    "Score",
                    f"{score_icon} {computed_data['bestfit_score_result']['score']:.3f}",
                )
            with mcols_bestfit[1]:
                st.metric("Correlation", f"{computed_data['bestfit_correlation']:.3f}")
            with mcols_bestfit[2]:
                rmse_val = computed_data["bestfit_score_result"].get("rmse", 0.0)
                st.metric("RMSE", f"{rmse_val:.5f}")
            with mcols_bestfit[3]:
                matched_peaks = int(
                    computed_data["bestfit_score_result"].get("matched_peaks", 0)
                )
                st.metric("Matched Peaks", f"{matched_peaks}")

            # === DEVIATION SCORE: PyElli vs LTA Comparison ===
            st.markdown("---")
            st.markdown("#### 📐 PyElli vs LTA Deviation Score")

            # Calculate deviation components
            pyelli_theo = computed_data.get("theoretical_display")
            lta_bestfit = computed_data.get("bestfit_display")

            if (
                pyelli_theo is not None
                and lta_bestfit is not None
                and len(pyelli_theo) > 0
                and len(lta_bestfit) > 0
            ):
                # Ensure same length
                min_len = min(len(pyelli_theo), len(lta_bestfit))
                pyelli_theo_aligned = pyelli_theo[:min_len]
                lta_bestfit_aligned = lta_bestfit[:min_len]

                # 1. MAPE between PyElli and LTA spectra (avoid division by zero)
                lta_abs = np.abs(lta_bestfit_aligned)
                valid_mask = lta_abs > 1e-10
                if valid_mask.any():
                    mape = (
                        float(
                            np.mean(
                                np.abs(
                                    pyelli_theo_aligned[valid_mask]
                                    - lta_bestfit_aligned[valid_mask]
                                )
                                / lta_abs[valid_mask]
                            )
                        )
                        * 100
                    )
                else:
                    mape = 0.0

                # 2. Peak match rate deviation
                # Compare PyElli matched peaks to LTA's peak count (as reference)
                pyelli_matched = computed_data["pyelli_score_result"].get(
                    "matched_peaks", 0
                )
                lta_peaks = computed_data["bestfit_score_result"].get(
                    "measurement_peaks", 0
                )  # LTA matched peaks against measured
                if lta_peaks > 0:
                    peak_match_deviation = (1.0 - (pyelli_matched / lta_peaks)) * 100
                    peak_match_deviation = max(
                        0.0, peak_match_deviation
                    )  # Can't be negative
                else:
                    peak_match_deviation = 0.0

                # 3. Alignment deviation (mean_delta relative to typical peak spacing ~50nm)
                pyelli_mean_delta = computed_data["pyelli_score_result"].get(
                    "mean_delta_nm", 0.0
                )
                reference_spacing = 50.0  # Typical peak-to-peak spacing in nm
                alignment_deviation = (pyelli_mean_delta / reference_spacing) * 100

                # Composite deviation score (weighted average)
                # MAPE: 40%, Peak Match: 30%, Alignment: 30%
                composite_deviation = (
                    0.40 * mape
                    + 0.30 * peak_match_deviation
                    + 0.30 * alignment_deviation
                )

                # Determine status icon and color
                if composite_deviation <= 10:
                    status_icon = "🟢"
                    status_color = "#16a34a"
                    status_text = "Excellent Match"
                elif composite_deviation <= 15:
                    status_icon = "🟡"
                    status_color = "#ca8a04"
                    status_text = "Good Match"
                elif composite_deviation <= 25:
                    status_icon = "🟠"
                    status_color = "#ea580c"
                    status_text = "Moderate Deviation"
                else:
                    status_icon = "🔴"
                    status_color = "#dc2626"
                    status_text = "High Deviation"

                # Display composite deviation prominently
                dev_cols = st.columns([2, 1, 1, 1])
                with dev_cols[0]:
                    st.markdown(
                        f"""
                    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                                border: 2px solid {status_color}; 
                                border-radius: 12px; 
                                padding: 16px; 
                                text-align: center;">
                        <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">
                            PyElli vs LTA Deviation
                        </div>
                        <div style="font-size: 2.2rem; font-weight: 700; color: {status_color}; margin: 4px 0;">
                            {status_icon} {composite_deviation:.1f}%
                        </div>
                        <div style="font-size: 0.85rem; color: {status_color}; font-weight: 500;">
                            {status_text}
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                with dev_cols[1]:
                    st.metric(
                        "MAPE",
                        f"{mape:.1f}%",
                        help="Mean Absolute Percentage Error between PyElli and LTA spectra",
                    )
                with dev_cols[2]:
                    st.metric(
                        "Peak Match Δ",
                        f"{peak_match_deviation:.1f}%",
                        help="Deviation in matched peak count vs LTA",
                    )
                with dev_cols[3]:
                    st.metric(
                        "Alignment Δ",
                        f"{alignment_deviation:.1f}%",
                        help="Peak alignment deviation (mean delta / 50nm reference)",
                    )

                # Explanation
                st.markdown(
                    f"""
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-top: 12px; font-size: 0.85rem; color: #64748b;">
                    <strong>Target:</strong> ≤10% deviation indicates PyElli's best fit closely matches LTA's best fit.<br>
                    <strong>Formula:</strong> 40% × MAPE + 30% × Peak Match Δ + 30% × Alignment Δ
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.info(
                    "Deviation score requires both PyElli theoretical and LTA BestFit spectra to be available."
                )
        else:
            # Show single set of metrics
            mcols = st.columns(7)
            with mcols[0]:
                st.metric("Lipid", f"{computed_data['display_lipid']} nm")
            with mcols[1]:
                st.metric("Aqueous", f"{computed_data['display_aqueous']} nm")
            with mcols[2]:
                st.metric("Roughness", f"{computed_data['display_mucus']:.0f} Å")
            with mcols[3]:
                score_result = computed_data["score_result"]
                score_icon = (
                    "🟢"
                    if score_result["score"] >= 0.7
                    else ("🟡" if score_result["score"] >= 0.5 else "🔴")
                )
                st.metric("Score", f"{score_icon} {score_result['score']:.3f}")
            with mcols[4]:
                st.metric("Correlation", f"{computed_data['correlation']:.3f}")
            with mcols[5]:
                rmse_val = score_result.get("rmse", 0.0)
                st.metric("RMSE", f"{rmse_val:.5f}")
            with mcols[6]:
                matched_peaks = int(score_result.get("matched_peaks", 0))
                st.metric("Matched Peaks", f"{matched_peaks}")

        # === DRIFT ANALYSIS (Cycle Jump Indicators) ===
        # Get drift metrics from the appropriate score result
        drift_score_result = computed_data.get(
            "pyelli_score_result"
        ) or computed_data.get("score_result")
        if drift_score_result and drift_score_result.get("drift_analysis_valid", False):
            with st.expander(
                "🔄 Drift Analysis (Cycle Jump Indicators)", expanded=False
            ):
                st.markdown(
                    """
                <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 12px; margin-bottom: 16px; font-size: 0.85rem; color: #92400e;">
                    <strong>Note:</strong> These single-spectrum metrics are <em>proxies</em> for cycle jump detection. 
                    True cycle-jump detection requires adjacent measurements/trends. Flagged fits may indicate 
                    the chosen frequency multiple is incorrect.
                </div>
                """,
                    unsafe_allow_html=True,
                )

                drift_cols = st.columns(2)

                with drift_cols[0]:
                    st.markdown("**Peak Drift**")
                    peak_slope = drift_score_result.get("peak_drift_slope", 0.0)
                    peak_r2 = drift_score_result.get("peak_drift_r_squared", 0.0)
                    peak_flagged = drift_score_result.get("peak_drift_flagged", False)

                    slope_icon = "⚠️" if peak_flagged else "✅"
                    st.metric(
                        "Slope",
                        f"{slope_icon} {peak_slope:.4f} nm/nm",
                        help="Linear trend of wavelength error (Δλ) across matched peaks. Large |slope| with high R² indicates systematic misalignment.",
                    )
                    st.metric(
                        "R²",
                        f"{peak_r2:.3f}",
                        help="Goodness of fit for peak drift trend. High R² (>0.5) with significant slope indicates systematic, not random, drift.",
                    )

                    if peak_flagged:
                        st.warning(
                            "Systematic peak misalignment detected - possible cycle jump candidate"
                        )

                with drift_cols[1]:
                    st.markdown("**Amplitude Drift**")
                    amp_slope = drift_score_result.get("amplitude_drift_slope", 0.0)
                    amp_r2 = drift_score_result.get("amplitude_drift_r_squared", 0.0)
                    amp_flagged = drift_score_result.get(
                        "amplitude_drift_flagged", False
                    )

                    slope_icon = "⚠️" if amp_flagged else "✅"
                    st.metric(
                        "Slope",
                        f"{slope_icon} {amp_slope:.4f}",
                        help="Linear trend of amplitude ratio (theo/meas) across matched peaks. Growing mismatch suggests wrong fit.",
                    )
                    st.metric(
                        "R²",
                        f"{amp_r2:.3f}",
                        help="Goodness of fit for amplitude drift trend. High R² (>0.5) with significant slope indicates systematic mismatch.",
                    )

                    if amp_flagged:
                        st.warning(
                            "Amplitude mismatch trend detected - possible cycle jump candidate"
                        )
        elif drift_score_result:
            # Drift analysis not valid (likely insufficient peaks)
            with st.expander(
                "🔄 Drift Analysis (Cycle Jump Indicators)", expanded=False
            ):
                st.info(
                    "Drift analysis requires at least 3 matched peaks. Insufficient peaks for analysis."
                )

        # Show results table if grid search was run
        if st.session_state.autofit_results:
            st.markdown("---")
            st.markdown("### 📊 Grid Search Results (Top 10)")

            def get_quality(score):
                if score >= 0.7:
                    return "🟢 Excellent"
                elif score >= 0.5:
                    return "🟡 Good"
                elif score >= 0.3:
                    return "🟠 Fair"
                else:
                    return "🔴 Poor"

            # Use sidebar cycle jump thresholds for drift indicator
            th_peak = st.session_state.get("drift_peak_slope_threshold", 0.05)
            th_amp = st.session_state.get("drift_amplitude_slope_threshold", 0.01)
            th_r2 = st.session_state.get("drift_r_squared_threshold", 0.5)

            def get_drift_flag(result):
                """Return drift indicator: ⚠️ if flagged, ✅ if not, - if not analyzed. Uses sidebar thresholds."""
                if not hasattr(result, "peak_drift_slope"):
                    return "-"
                peak_flagged = (
                    abs(getattr(result, "peak_drift_slope", 0)) > th_peak
                    and getattr(result, "peak_drift_r_squared", 0) > th_r2
                )
                amp_flagged = (
                    abs(getattr(result, "amplitude_drift_slope", 0)) > th_amp
                    and getattr(result, "amplitude_drift_r_squared", 0) > th_r2
                )
                return "⚠️" if (peak_flagged or amp_flagged) else "✅"

            results_df = pd.DataFrame(
                [
                    {
                        "Rank": i + 1,
                        "Lipid (nm)": r.lipid_nm,
                        "Aqueous (nm)": r.aqueous_nm,
                        "Roughness (Å)": r.mucus_nm,
                        "Score": f"{r.score:.3f}",
                        "Corr": f"{r.correlation:.3f}",
                        "RMSE": f"{r.rmse:.5f}",
                        "Osc Ratio": f"{r.oscillation_ratio:.2f}",  # CRITICAL: Show amplitude match
                        "Matched": r.matched_peaks,
                        "Drift": get_drift_flag(r),  # Cycle jump indicator
                        "Quality": get_quality(r.score),
                    }
                    for i, r in enumerate(st.session_state.autofit_results)
                ]
            )

            st.dataframe(results_df, use_container_width=True, hide_index=True)

            # Dropdown to select which rank to display in plots
            st.markdown("---")
            st.markdown("### 🎯 Select Result to Display")

            # Get current selected rank from session state, default to 1
            current_selected_rank = st.session_state.get("selected_rank", 1)
            max_rank = min(len(st.session_state.autofit_results), 10)
            rank_options = list(range(1, max_rank + 1))

            # Sync dropdown key with selected_rank before creating the widget (e.g. after "Select Rank N" in Amplitude tab).
            # Setting the key after the widget is instantiated causes a Streamlit error, so we set it here.
            if (
                "selected_rank_dropdown" not in st.session_state
                or st.session_state.selected_rank_dropdown != current_selected_rank
            ):
                st.session_state.selected_rank_dropdown = current_selected_rank

            # Find index for current selection (default to 0 if not in range)
            default_index = (
                min(current_selected_rank - 1, len(rank_options) - 1)
                if current_selected_rank in rank_options
                else 0
            )

            selected_rank = st.selectbox(
                "Choose a rank to display in plots:",
                options=rank_options,
                index=default_index,
                key="selected_rank_dropdown",
                help="Select which grid search result to display in the Spectrum Comparison and Amplitude Analysis plots. The parameter sliders and plots will update to match the selected rank.",
            )

            # Update parameters when rank is selected
            if 1 <= selected_rank <= len(st.session_state.autofit_results):
                selected_result = st.session_state.autofit_results[selected_rank - 1]

                # Check if rank changed
                if selected_rank != current_selected_rank:
                    # Update selected rank in session state
                    st.session_state.selected_rank = selected_rank
                    # Set forced values for the sliders
                    st.session_state.forced_lipid = float(selected_result.lipid_nm)
                    st.session_state.forced_aqueous = float(selected_result.aqueous_nm)
                    st.session_state.forced_mucus = float(selected_result.mucus_nm)
                    # Increment widget key version to force sliders to reset
                    st.session_state.widget_key_version += 1
                    # Trigger rerun to update plots and sliders
                    st.rerun()

            # PDF Export button
            st.markdown("---")
            pdf_cols = st.columns([2, 1])
            with pdf_cols[0]:
                pdf_top_n = st.number_input(
                    "Top N for PDF",
                    min_value=1,
                    max_value=10,
                    value=10,
                    step=1,
                    help="Number of top fits to include in PDF report",
                    key="pyelli_pdf_top_n",
                )
            with pdf_cols[1]:
                if st.button(
                    "📄 Export PDF Report",
                    key="pyelli_export_pdf",
                    use_container_width=True,
                ):
                    with st.spinner(
                        f"Generating PDF report for top {pdf_top_n} fits..."
                    ):
                        try:
                            pdf_bytes = generate_pyelli_pdf_report(
                                autofit_results=st.session_state.autofit_results,
                                measurement_file=Path(selected_file).name,
                                measured_wavelengths=computed_data["wavelengths"],
                                measured_spectrum=computed_data["measured"],
                                wl_min=wavelength_range[0],
                                wl_max=wavelength_range[1],
                                top_n=int(pdf_top_n),
                            )

                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            pdf_filename = f"pyelli_report_{Path(selected_file).stem}_{timestamp}.pdf"

                            st.download_button(
                                label="⬇️ Download PDF Report",
                                data=pdf_bytes,
                                file_name=pdf_filename,
                                mime="application/pdf",
                                key="pyelli_download_pdf",
                            )
                            st.success(
                                "✅ PDF report generated! Click above to download."
                            )
                        except Exception as pdf_err:
                            st.error(f"❌ Error generating PDF: {str(pdf_err)}")
                            import traceback

                            st.code(traceback.format_exc())
    else:
        st.info("👈 Please select a spectrum file from the sidebar")

# =============================================================================
# Tab 2: Amplitude Analysis
# =============================================================================

with tabs[1]:
    if computed_data:
        st.markdown("### 📊 Amplitude Analysis")
        st.markdown(
            '<p style="color: #94a3b8; margin-bottom: 1rem;">Detrended amplitude signals with peak alignment to verify fit quality</p>',
            unsafe_allow_html=True,
        )

        # Get default values for settings (will be used for initial plot)
        # Use correct defaults: cutoff_frequency=0.008, peak_prominence=0.0001
        # Only migrate old defaults (0.01 for cutoff), allow other values to persist

        # Check if keys exist in session state
        cutoff_freq_key_exists = "cutoff_freq_amp" in st.session_state
        peak_prom_key_exists = "peak_prom_amp" in st.session_state

        # Get current values (use defaults if keys don't exist)
        cutoff_freq_raw = st.session_state.get("cutoff_freq_amp", 0.008)
        peak_prominence_raw = st.session_state.get("peak_prom_amp", 0.0001)

        # Migrate cutoff frequency: only reset if it's the old default (0.01) or key doesn't exist
        if not cutoff_freq_key_exists:
            # Key doesn't exist, use default
            cutoff_freq = 0.008
            st.session_state.cutoff_freq_amp = 0.008
        elif abs(cutoff_freq_raw - 0.01) < 0.0001:  # Only migrate old default 0.01
            cutoff_freq = 0.008
            st.session_state.cutoff_freq_amp = 0.008
        else:
            cutoff_freq = (
                cutoff_freq_raw  # Allow any other value (including user-selected)
            )

        # Peak prominence: use default if key doesn't exist, otherwise use stored value
        if not peak_prom_key_exists:
            # Key doesn't exist, use default
            peak_prominence = 0.0001
            st.session_state.peak_prom_amp = 0.0001
        else:
            peak_prominence = (
                peak_prominence_raw  # Allow any value (including user-selected)
            )

        # Get smoothing parameters from session state
        smoothing_type = st.session_state.get("smoothing_type", "none")
        boxcar_width_nm = st.session_state.get("boxcar_width_nm", 17.0)
        boxcar_passes = st.session_state.get("boxcar_passes", 1)
        gaussian_kernel = st.session_state.get("gaussian_kernel", 11)

        # Get toggle state from session state
        show_bestfit_amp = computed_data.get("show_bestfit", False)
        show_both_amp = computed_data.get("show_both_theoretical", False)
        has_bestfit_amp = (
            computed_data.get("has_bestfit", False)
            and computed_data.get("bestfit_display") is not None
        )

        try:
            # Detrend measured signal, then apply smoothing
            meas_detrended = detrend_signal(
                computed_data["wl_display"],
                computed_data["meas_display"],
                cutoff_freq,
                filter_order=3,
            )
            meas_detrended = apply_smoothing_to_signal(
                meas_detrended,
                computed_data["wl_display"],
                smoothing_type,
                boxcar_width_nm,
                boxcar_passes,
                gaussian_kernel,
            )

            # Determine which theoretical(s) to analyze based on toggle
            # Note: Smoothing is only applied to measured spectrum, not theoretical
            if show_both_amp and has_bestfit_amp:
                # Detrend both PyElli and BestFit (no smoothing for theoretical)
                theo_pyelli_detrended = detrend_signal(
                    computed_data["wl_display"],
                    computed_data["theoretical_display"],
                    cutoff_freq,
                    filter_order=3,
                )
                theo_bestfit_detrended = detrend_signal(
                    computed_data["wl_display"],
                    computed_data["bestfit_display"],
                    cutoff_freq,
                    filter_order=3,
                )

                # Detect peaks and valleys for both
                meas_peaks_df = detect_peaks(
                    computed_data["wl_display"],
                    meas_detrended,
                    prominence=peak_prominence,
                )
                theo_pyelli_peaks_df = detect_peaks(
                    computed_data["wl_display"],
                    theo_pyelli_detrended,
                    prominence=peak_prominence,
                )
                theo_bestfit_peaks_df = detect_peaks(
                    computed_data["wl_display"],
                    theo_bestfit_detrended,
                    prominence=peak_prominence,
                )

                meas_valleys_df = detect_valleys(
                    computed_data["wl_display"],
                    meas_detrended,
                    prominence=peak_prominence,
                )
                theo_pyelli_valleys_df = detect_valleys(
                    computed_data["wl_display"],
                    theo_pyelli_detrended,
                    prominence=peak_prominence,
                )
                theo_bestfit_valleys_df = detect_valleys(
                    computed_data["wl_display"],
                    theo_bestfit_detrended,
                    prominence=peak_prominence,
                )

                # Calculate scores for both
                score_result_pyelli = calculate_peak_based_score(
                    computed_data["wl_display"],
                    computed_data["meas_display"],
                    computed_data["theoretical_display"],
                    cutoff_frequency=cutoff_freq,
                    peak_prominence=peak_prominence,
                )
                score_result_bestfit = calculate_peak_based_score(
                    computed_data["wl_display"],
                    computed_data["meas_display"],
                    computed_data["bestfit_display"],
                    cutoff_frequency=cutoff_freq,
                    peak_prominence=peak_prominence,
                )
                score_result_amp = score_result_pyelli  # Use PyElli for main display
            elif show_bestfit_amp and has_bestfit_amp:
                # Use BestFit only - detrend only (no smoothing for theoretical)
                theo_detrended = detrend_signal(
                    computed_data["wl_display"],
                    computed_data["bestfit_display"],
                    cutoff_freq,
                    filter_order=3,
                )

                meas_peaks_df = detect_peaks(
                    computed_data["wl_display"],
                    meas_detrended,
                    prominence=peak_prominence,
                )
                theo_peaks_df = detect_peaks(
                    computed_data["wl_display"],
                    theo_detrended,
                    prominence=peak_prominence,
                )

                meas_valleys_df = detect_valleys(
                    computed_data["wl_display"],
                    meas_detrended,
                    prominence=peak_prominence,
                )
                theo_valleys_df = detect_valleys(
                    computed_data["wl_display"],
                    theo_detrended,
                    prominence=peak_prominence,
                )

                score_result_amp = calculate_peak_based_score(
                    computed_data["wl_display"],
                    computed_data["meas_display"],
                    computed_data["bestfit_display"],
                    cutoff_frequency=cutoff_freq,
                    peak_prominence=peak_prominence,
                )
            else:
                # Use PyElli only (default) - detrend only (no smoothing for theoretical)
                theo_detrended = detrend_signal(
                    computed_data["wl_display"],
                    computed_data["theoretical_display"],
                    cutoff_freq,
                    filter_order=3,
                )

                meas_peaks_df = detect_peaks(
                    computed_data["wl_display"],
                    meas_detrended,
                    prominence=peak_prominence,
                )
                theo_peaks_df = detect_peaks(
                    computed_data["wl_display"],
                    theo_detrended,
                    prominence=peak_prominence,
                )

                meas_valleys_df = detect_valleys(
                    computed_data["wl_display"],
                    meas_detrended,
                    prominence=peak_prominence,
                )
                theo_valleys_df = detect_valleys(
                    computed_data["wl_display"],
                    theo_detrended,
                    prominence=peak_prominence,
                )

                score_result_amp = calculate_peak_based_score(
                    computed_data["wl_display"],
                    computed_data["meas_display"],
                    computed_data["theoretical_display"],
                    cutoff_frequency=cutoff_freq,
                    peak_prominence=peak_prominence,
                )

            # Create amplitude plot (before settings and metrics)
            st.markdown("**Detrended Amplitude Signals with Peaks and Valleys**")
            fig_amp = go.Figure()

            # Plot measured (always shown)
            fig_amp.add_trace(
                go.Scatter(
                    x=computed_data["wl_display"],
                    y=meas_detrended,
                    mode="lines",
                    name="Measured (Detrended)",
                    line=dict(color="#2563eb", width=2.5),
                )
            )

            # Plot theoretical based on toggle selection
            if show_both_amp and has_bestfit_amp:
                # Show both PyElli and BestFit
                fig_amp.add_trace(
                    go.Scatter(
                        x=computed_data["wl_display"],
                        y=theo_pyelli_detrended,
                        mode="lines",
                        name="PyElli (Detrended)",
                        line=dict(color="#059669", width=2.5, dash="dash"),
                    )
                )
                fig_amp.add_trace(
                    go.Scatter(
                        x=computed_data["wl_display"],
                        y=theo_bestfit_detrended,
                        mode="lines",
                        name="LTA BestFit (Detrended)",
                        line=dict(color="#db2777", width=2.5, dash="dot"),
                    )
                )
            else:
                # Show single theoretical (PyElli or BestFit)
                theo_label = (
                    "LTA BestFit (Detrended)"
                    if (show_bestfit_amp and has_bestfit_amp)
                    else "PyElli (Detrended)"
                )
                theo_color = (
                    "#db2777" if (show_bestfit_amp and has_bestfit_amp) else "#059669"
                )
                fig_amp.add_trace(
                    go.Scatter(
                        x=computed_data["wl_display"],
                        y=theo_detrended,
                        mode="lines",
                        name=theo_label,
                        line=dict(color=theo_color, width=2.5, dash="dash"),
                    )
                )

            # Plot peaks
            if len(meas_peaks_df) > 0:
                fig_amp.add_trace(
                    go.Scatter(
                        x=meas_peaks_df["wavelength"],
                        y=meas_peaks_df["amplitude"],
                        mode="markers",
                        name="Measured Peaks",
                        marker=dict(size=8, symbol="circle", color="blue"),
                    )
                )

            if show_both_amp and has_bestfit_amp:
                # Plot peaks for both PyElli and BestFit
                if len(theo_pyelli_peaks_df) > 0:
                    fig_amp.add_trace(
                        go.Scatter(
                            x=theo_pyelli_peaks_df["wavelength"],
                            y=theo_pyelli_peaks_df["amplitude"],
                            mode="markers",
                            name="PyElli Peaks",
                            marker=dict(size=8, symbol="circle", color="green"),
                        )
                    )
                if len(theo_bestfit_peaks_df) > 0:
                    fig_amp.add_trace(
                        go.Scatter(
                            x=theo_bestfit_peaks_df["wavelength"],
                            y=theo_bestfit_peaks_df["amplitude"],
                            mode="markers",
                            name="BestFit Peaks",
                            marker=dict(size=8, symbol="circle", color="red"),
                        )
                    )
            else:
                # Plot peaks for single theoretical
                if len(theo_peaks_df) > 0:
                    peak_name = (
                        "BestFit Peaks"
                        if (show_bestfit_amp and has_bestfit_amp)
                        else "PyElli Peaks"
                    )
                    peak_color = (
                        "red" if (show_bestfit_amp and has_bestfit_amp) else "green"
                    )
                    fig_amp.add_trace(
                        go.Scatter(
                            x=theo_peaks_df["wavelength"],
                            y=theo_peaks_df["amplitude"],
                            mode="markers",
                            name=peak_name,
                            marker=dict(size=8, symbol="circle", color=peak_color),
                        )
                    )

            # Plot valleys
            if len(meas_valleys_df) > 0:
                fig_amp.add_trace(
                    go.Scatter(
                        x=meas_valleys_df["wavelength"],
                        y=meas_valleys_df["amplitude"],
                        mode="markers",
                        name="Measured Valleys",
                        marker=dict(size=8, symbol="circle", color="cyan"),
                    )
                )

            if show_both_amp and has_bestfit_amp:
                # Plot valleys for both PyElli and BestFit
                if len(theo_pyelli_valleys_df) > 0:
                    fig_amp.add_trace(
                        go.Scatter(
                            x=theo_pyelli_valleys_df["wavelength"],
                            y=theo_pyelli_valleys_df["amplitude"],
                            mode="markers",
                            name="PyElli Valleys",
                            marker=dict(size=8, symbol="circle", color="lightgreen"),
                        )
                    )
                if len(theo_bestfit_valleys_df) > 0:
                    fig_amp.add_trace(
                        go.Scatter(
                            x=theo_bestfit_valleys_df["wavelength"],
                            y=theo_bestfit_valleys_df["amplitude"],
                            mode="markers",
                            name="BestFit Valleys",
                            marker=dict(size=8, symbol="circle", color="pink"),
                        )
                    )
            else:
                # Plot valleys for single theoretical
                if len(theo_valleys_df) > 0:
                    valley_name = (
                        "BestFit Valleys"
                        if (show_bestfit_amp and has_bestfit_amp)
                        else "PyElli Valleys"
                    )
                    valley_color = (
                        "pink"
                        if (show_bestfit_amp and has_bestfit_amp)
                        else "lightgreen"
                    )
                    fig_amp.add_trace(
                        go.Scatter(
                            x=theo_valleys_df["wavelength"],
                            y=theo_valleys_df["amplitude"],
                            mode="markers",
                            name=valley_name,
                            marker=dict(size=8, symbol="circle", color=valley_color),
                        )
                    )

            # === DRIFT VISUALIZATION: Connect matched peaks with color-coded lines ===
            # Use the appropriate theoretical peaks based on toggle selection
            if show_both_amp and has_bestfit_amp:
                drift_theo_peaks_df = theo_pyelli_peaks_df  # Use PyElli for drift viz
            else:
                drift_theo_peaks_df = theo_peaks_df

            if len(meas_peaks_df) > 0 and len(drift_theo_peaks_df) > 0:
                meas_peak_wavelengths = meas_peaks_df["wavelength"].to_numpy()
                theo_peak_wavelengths = drift_theo_peaks_df["wavelength"].to_numpy()
                matched_meas_idx, matched_theo_idx, deltas = _match_peaks(
                    meas_peak_wavelengths, theo_peak_wavelengths, tolerance_nm=20.0
                )

                # Draw drift lines between matched peaks
                for i, (m_idx, t_idx, delta) in enumerate(
                    zip(matched_meas_idx, matched_theo_idx, deltas)
                ):
                    # Color by drift magnitude: green (<5nm), yellow (5-10nm), red (>10nm)
                    if delta < 5:
                        color = "#22c55e"  # Green - good alignment
                    elif delta < 10:
                        color = "#eab308"  # Yellow - moderate drift
                    else:
                        color = "#ef4444"  # Red - significant drift

                    fig_amp.add_trace(
                        go.Scatter(
                            x=[
                                meas_peak_wavelengths[m_idx],
                                theo_peak_wavelengths[t_idx],
                            ],
                            y=[
                                meas_peaks_df["amplitude"].iloc[m_idx],
                                drift_theo_peaks_df["amplitude"].iloc[t_idx],
                            ],
                            mode="lines",
                            line=dict(color=color, width=1.5, dash="dot"),
                            showlegend=(i == 0),
                            name="Peak Drift" if i == 0 else None,
                            legendgroup="drift",
                            hovertemplate=f"Δλ = {delta:.1f} nm<extra></extra>",
                        )
                    )

            fig_amp.update_layout(
                height=600,
                title="",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Amplitude",
                xaxis_range=[wavelength_range[0], wavelength_range[1]],
                hovermode="x unified",
                margin=dict(t=60, b=40, l=60, r=30),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )

            st.plotly_chart(fig_amp, width="stretch")

            # === DRIFT ANALYSIS PLOT ===
            # Show detailed drift analysis if we have matched peaks
            if (
                len(meas_peaks_df) > 0
                and len(drift_theo_peaks_df) > 0
                and len(matched_meas_idx) >= 3
            ):
                with st.expander(
                    "🔄 Drift Analysis Plot (Cycle Jump Indicators)", expanded=False
                ):
                    st.markdown(
                        """
                    <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 12px; margin-bottom: 16px; font-size: 0.85rem; color: #92400e;">
                        <strong>Note:</strong> These plots show how peak position error (Δλ) and amplitude ratio evolve across the spectrum.
                        A systematic trend (high R²) suggests the chosen thickness/frequency may be a wrong multiple (cycle jump candidate).
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Get matched peak data for drift plots
                    matched_meas_wl = meas_peak_wavelengths[matched_meas_idx]
                    matched_theo_wl = theo_peak_wavelengths[matched_theo_idx]
                    delta_wavelengths = matched_theo_wl - matched_meas_wl  # Signed Δλ

                    # Get matched amplitudes
                    matched_meas_amp = (
                        meas_peaks_df["amplitude"].iloc[matched_meas_idx].to_numpy()
                    )
                    matched_theo_amp = (
                        drift_theo_peaks_df["amplitude"]
                        .iloc[matched_theo_idx]
                        .to_numpy()
                    )

                    # Calculate amplitude ratios (protect against division by zero)
                    safe_meas_amp = np.where(
                        np.abs(matched_meas_amp) > 1e-10, matched_meas_amp, 1e-10
                    )
                    amplitude_ratios = matched_theo_amp / safe_meas_amp

                    # Get drift metrics from score_result_amp; flag using sidebar thresholds
                    peak_drift_slope = score_result_amp.get("peak_drift_slope", 0.0)
                    peak_drift_r2 = score_result_amp.get("peak_drift_r_squared", 0.0)
                    amp_drift_slope = score_result_amp.get("amplitude_drift_slope", 0.0)
                    amp_drift_r2 = score_result_amp.get(
                        "amplitude_drift_r_squared", 0.0
                    )
                    th_peak_amp = st.session_state.get(
                        "drift_peak_slope_threshold", 0.05
                    )
                    th_amp_amp = st.session_state.get(
                        "drift_amplitude_slope_threshold", 0.01
                    )
                    th_r2_amp = st.session_state.get("drift_r_squared_threshold", 0.5)
                    peak_drift_flagged = (
                        abs(peak_drift_slope) > th_peak_amp
                        and peak_drift_r2 > th_r2_amp
                    )
                    amp_drift_flagged = (
                        abs(amp_drift_slope) > th_amp_amp and amp_drift_r2 > th_r2_amp
                    )

                    # Create subplot with 2 rows
                    fig_drift = make_subplots(
                        rows=2,
                        cols=1,
                        subplot_titles=(
                            f"Peak Position Drift (Δλ vs Wavelength) — Slope: {peak_drift_slope:.4f} nm/nm, R²: {peak_drift_r2:.3f}",
                            f"Amplitude Ratio Drift — Slope: {amp_drift_slope:.4f}, R²: {amp_drift_r2:.3f}",
                        ),
                        vertical_spacing=0.18,
                    )

                    # Row 1: Peak drift scatter + regression line
                    scatter_color = "#ef4444" if peak_drift_flagged else "#3b82f6"
                    fig_drift.add_trace(
                        go.Scatter(
                            x=matched_meas_wl,
                            y=delta_wavelengths,
                            mode="markers",
                            name="Δλ (theo - meas)",
                            marker=dict(size=10, color=scatter_color, symbol="circle"),
                            hovertemplate="λ: %{x:.1f} nm<br>Δλ: %{y:.2f} nm<extra></extra>",
                        ),
                        row=1,
                        col=1,
                    )

                    # Add regression line for peak drift
                    if len(matched_meas_wl) >= 2:
                        x_range = np.array(
                            [matched_meas_wl.min(), matched_meas_wl.max()]
                        )
                        y_mean = np.mean(delta_wavelengths)
                        x_mean = np.mean(matched_meas_wl)
                        y_regression = peak_drift_slope * (x_range - x_mean) + y_mean
                        line_color = "#ef4444" if peak_drift_flagged else "#64748b"
                        fig_drift.add_trace(
                            go.Scatter(
                                x=x_range,
                                y=y_regression,
                                mode="lines",
                                name="Trend line",
                                line=dict(color=line_color, width=2, dash="dash"),
                                hoverinfo="skip",
                            ),
                            row=1,
                            col=1,
                        )

                    # Row 2: Amplitude ratio scatter + regression line
                    scatter_color_amp = "#ef4444" if amp_drift_flagged else "#3b82f6"
                    fig_drift.add_trace(
                        go.Scatter(
                            x=matched_meas_wl,
                            y=amplitude_ratios,
                            mode="markers",
                            name="Amp Ratio (theo/meas)",
                            marker=dict(
                                size=10, color=scatter_color_amp, symbol="diamond"
                            ),
                            hovertemplate="λ: %{x:.1f} nm<br>Ratio: %{y:.3f}<extra></extra>",
                        ),
                        row=2,
                        col=1,
                    )

                    # Add regression line for amplitude drift
                    if len(matched_meas_wl) >= 2:
                        y_mean_amp = np.mean(amplitude_ratios)
                        y_regression_amp = (
                            amp_drift_slope * (x_range - x_mean) + y_mean_amp
                        )
                        line_color_amp = "#ef4444" if amp_drift_flagged else "#64748b"
                        fig_drift.add_trace(
                            go.Scatter(
                                x=x_range,
                                y=y_regression_amp,
                                mode="lines",
                                name="Trend line",
                                line=dict(color=line_color_amp, width=2, dash="dash"),
                                showlegend=False,
                                hoverinfo="skip",
                            ),
                            row=2,
                            col=1,
                        )

                    # Add horizontal reference lines
                    fig_drift.add_hline(
                        y=0, line_dash="dot", line_color="#94a3b8", row=1, col=1
                    )
                    fig_drift.add_hline(
                        y=1.0, line_dash="dot", line_color="#94a3b8", row=2, col=1
                    )

                    fig_drift.update_layout(
                        height=500,
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.08,
                            xanchor="right",
                            x=1,
                        ),
                        margin=dict(t=80, b=40, l=60, r=30),
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                    )
                    fig_drift.update_xaxes(title_text="Wavelength (nm)", row=2, col=1)
                    fig_drift.update_yaxes(title_text="Δλ (nm)", row=1, col=1)
                    fig_drift.update_yaxes(title_text="Amplitude Ratio", row=2, col=1)

                    st.plotly_chart(fig_drift, use_container_width=True)

                    # Gate warning + alternatives on rank 1 being flagged (not current rank), so the section
                    # stays visible when user selects an alternative rank and they can switch between them.
                    rank1 = (
                        st.session_state.autofit_results[0]
                        if st.session_state.autofit_results
                        else None
                    )
                    if rank1 is not None and hasattr(rank1, "peak_drift_slope"):
                        rank1_peak_flagged = (
                            abs(rank1.peak_drift_slope) > th_peak_amp
                            and rank1.peak_drift_r_squared > th_r2_amp
                        )
                        rank1_amp_flagged = (
                            abs(rank1.amplitude_drift_slope) > th_amp_amp
                            and rank1.amplitude_drift_r_squared > th_r2_amp
                        )
                        rank1_any_flagged = rank1_peak_flagged or rank1_amp_flagged
                    else:
                        rank1_any_flagged = False

                    if rank1_any_flagged:
                        flagged_types = []
                        if rank1_peak_flagged:
                            flagged_types.append("peak position drift")
                        if rank1_amp_flagged:
                            flagged_types.append("amplitude drift")
                        st.warning(
                            f"⚠️ **Cycle Jump Warning:** Rank 1 has systematic {' and '.join(flagged_types)}. It may be a wrong frequency multiple."
                        )

                        # === ALTERNATIVE OPTIONS: Show top unflagged results (shortcut to ranks in table); always visible when rank 1 is flagged ===
                        if (
                            st.session_state.autofit_results
                            and len(st.session_state.autofit_results) > 1
                        ):
                            # Find alternatives without drift flags (skip rank 1), using sidebar thresholds
                            th_peak_alt = st.session_state.get(
                                "drift_peak_slope_threshold", 0.05
                            )
                            th_amp_alt = st.session_state.get(
                                "drift_amplitude_slope_threshold", 0.01
                            )
                            th_r2_alt = st.session_state.get(
                                "drift_r_squared_threshold", 0.5
                            )

                            def _is_drift_flagged(r):
                                if not hasattr(r, "peak_drift_slope"):
                                    return False
                                p = (
                                    abs(getattr(r, "peak_drift_slope", 0)) > th_peak_alt
                                    and getattr(r, "peak_drift_r_squared", 0)
                                    > th_r2_alt
                                )
                                a = (
                                    abs(getattr(r, "amplitude_drift_slope", 0))
                                    > th_amp_alt
                                    and getattr(r, "amplitude_drift_r_squared", 0)
                                    > th_r2_alt
                                )
                                return p or a

                            unflagged_alternatives = []
                            for i, r in enumerate(
                                st.session_state.autofit_results[1:10], start=2
                            ):  # Ranks 2-10
                                if not _is_drift_flagged(r):
                                    unflagged_alternatives.append((i, r))
                                if len(unflagged_alternatives) >= 4:
                                    break

                            if unflagged_alternatives:
                                st.markdown("---")
                                st.markdown(
                                    "**🔄 Alternative Options Without Drift Flags:**"
                                )
                                st.markdown(
                                    '<p style="color: #64748b; font-size: 0.85rem;">These results may be better fits if rank 1 is a cycle jump.</p>',
                                    unsafe_allow_html=True,
                                )

                                # Create columns for alternative cards
                                alt_cols = st.columns(len(unflagged_alternatives))
                                for col, (rank, result) in zip(
                                    alt_cols, unflagged_alternatives
                                ):
                                    with col:
                                        # Score color
                                        score_color = (
                                            "#22c55e"
                                            if result.score >= 0.7
                                            else (
                                                "#eab308"
                                                if result.score >= 0.5
                                                else "#ef4444"
                                            )
                                        )
                                        st.markdown(
                                            f"""
                                        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center;">
                                            <div style="font-weight: 600; color: #334155; margin-bottom: 8px;">Rank {rank}</div>
                                            <div style="font-size: 1.2rem; font-weight: 700; color: {score_color};">{result.score:.3f}</div>
                                            <div style="font-size: 0.75rem; color: #64748b; margin-top: 8px;">
                                                L={result.lipid_nm:.0f} A={result.aqueous_nm:.0f}<br>
                                                R={result.mucus_nm:.0f}Å
                                            </div>
                                        </div>
                                        """,
                                            unsafe_allow_html=True,
                                        )
                                        if st.button(
                                            f"Select Rank {rank}",
                                            key=f"select_alt_{rank}",
                                            use_container_width=True,
                                        ):
                                            st.session_state.selected_rank = rank
                                            st.session_state.forced_lipid = (
                                                result.lipid_nm
                                            )
                                            st.session_state.forced_aqueous = (
                                                result.aqueous_nm
                                            )
                                            st.session_state.forced_mucus = (
                                                result.mucus_nm
                                            )
                                            st.session_state.widget_key_version += 1
                                            st.rerun()
                            else:
                                st.info(
                                    "No unflagged alternatives found in top 10 results."
                                )

            # Settings and metrics below the plot
            st.markdown("---")
            st.markdown("### ⚙️ Analysis Settings")

            # Add reset button to restore defaults
            if st.button(
                "🔄 Reset to Defaults",
                help="Reset cutoff frequency to 0.008 and peak prominence to 0.0001",
            ):
                # Delete keys first to ensure clean reset
                if "cutoff_freq_amp" in st.session_state:
                    del st.session_state.cutoff_freq_amp
                if "peak_prom_amp" in st.session_state:
                    del st.session_state.peak_prom_amp
                # Set correct defaults
                st.session_state.cutoff_freq_amp = 0.008
                st.session_state.peak_prom_amp = 0.0001
                st.rerun()

            # Amplitude analysis settings
            col_amp_settings = st.columns(2)
            with col_amp_settings[0]:
                # Use the value from session state (already migrated above if needed)
                cutoff_freq_new = st.slider(
                    "Detrending Cutoff Frequency",
                    0.001,
                    0.02,
                    value=cutoff_freq,
                    step=0.001,
                    format="%.3f",  # Show 3 decimal places to see 0.008 clearly
                    key="cutoff_freq_amp",
                )
            with col_amp_settings[1]:
                peak_prominence_new = st.slider(
                    "Peak Prominence",
                    0.00001,
                    0.01,
                    value=peak_prominence,
                    step=0.00001,
                    format="%.5f",
                    key="peak_prom_amp",
                )

            # Recalculate if settings changed
            if cutoff_freq_new != cutoff_freq or peak_prominence_new != peak_prominence:
                st.rerun()

            st.markdown("---")
            st.markdown("### 📈 Peak Analysis Metrics")

            # Show metrics based on toggle selection
            if show_both_amp and has_bestfit_amp:
                # Show metrics for both PyElli and BestFit
                st.markdown("#### PyElli Theoretical")
                mcols_pyelli_amp = st.columns(6)
                with mcols_pyelli_amp[0]:
                    st.metric(
                        "Measured Peaks",
                        f"{int(score_result_pyelli.get('measurement_peaks', 0))}",
                    )
                with mcols_pyelli_amp[1]:
                    st.metric(
                        "Theoretical Peaks",
                        f"{int(score_result_pyelli.get('theoretical_peaks', 0))}",
                    )
                with mcols_pyelli_amp[2]:
                    st.metric(
                        "Matched Peaks",
                        f"{int(score_result_pyelli.get('matched_peaks', 0))}",
                    )
                with mcols_pyelli_amp[3]:
                    st.metric(
                        "Mean Delta",
                        f"{score_result_pyelli.get('mean_delta_nm', 0):.2f} nm",
                    )
                with mcols_pyelli_amp[4]:
                    # CRITICAL: Show oscillation ratio to debug amplitude issues
                    osc_ratio = score_result_pyelli.get("oscillation_ratio", 1.0)
                    osc_icon = (
                        "🟢"
                        if 0.7 <= osc_ratio <= 1.5
                        else ("🟡" if 0.5 <= osc_ratio <= 2.0 else "🔴")
                    )
                    st.metric("Osc Ratio", f"{osc_icon} {osc_ratio:.2f}")
                with mcols_pyelli_amp[5]:
                    score_icon_amp = (
                        "🟢"
                        if score_result_pyelli["score"] >= 0.7
                        else ("🟡" if score_result_pyelli["score"] >= 0.5 else "🔴")
                    )
                    st.metric(
                        "Peak Score",
                        f"{score_icon_amp} {score_result_pyelli['score']:.3f}",
                    )

                st.markdown("#### LTA BestFit")
                mcols_bestfit_amp = st.columns(6)
                with mcols_bestfit_amp[0]:
                    st.metric(
                        "Measured Peaks",
                        f"{int(score_result_bestfit.get('measurement_peaks', 0))}",
                    )
                with mcols_bestfit_amp[1]:
                    st.metric(
                        "Theoretical Peaks",
                        f"{int(score_result_bestfit.get('theoretical_peaks', 0))}",
                    )
                with mcols_bestfit_amp[2]:
                    st.metric(
                        "Matched Peaks",
                        f"{int(score_result_bestfit.get('matched_peaks', 0))}",
                    )
                with mcols_bestfit_amp[3]:
                    st.metric(
                        "Mean Delta",
                        f"{score_result_bestfit.get('mean_delta_nm', 0):.2f} nm",
                    )
                with mcols_bestfit_amp[4]:
                    osc_ratio_bf = score_result_bestfit.get("oscillation_ratio", 1.0)
                    osc_icon_bf = (
                        "🟢"
                        if 0.7 <= osc_ratio_bf <= 1.5
                        else ("🟡" if 0.5 <= osc_ratio_bf <= 2.0 else "🔴")
                    )
                    st.metric("Osc Ratio", f"{osc_icon_bf} {osc_ratio_bf:.2f}")
                with mcols_bestfit_amp[5]:
                    score_icon_amp = (
                        "🟢"
                        if score_result_bestfit["score"] >= 0.7
                        else ("🟡" if score_result_bestfit["score"] >= 0.5 else "🔴")
                    )
                    st.metric(
                        "Peak Score",
                        f"{score_icon_amp} {score_result_bestfit['score']:.3f}",
                    )
            else:
                # Show single set of metrics
                mcols_amp = st.columns(6)
                with mcols_amp[0]:
                    st.metric(
                        "Measured Peaks",
                        f"{int(score_result_amp.get('measurement_peaks', 0))}",
                    )
                with mcols_amp[1]:
                    st.metric(
                        "Theoretical Peaks",
                        f"{int(score_result_amp.get('theoretical_peaks', 0))}",
                    )
                with mcols_amp[2]:
                    st.metric(
                        "Matched Peaks",
                        f"{int(score_result_amp.get('matched_peaks', 0))}",
                    )
                with mcols_amp[3]:
                    st.metric(
                        "Mean Delta",
                        f"{score_result_amp.get('mean_delta_nm', 0):.2f} nm",
                    )
                with mcols_amp[4]:
                    osc_ratio_amp = score_result_amp.get("oscillation_ratio", 1.0)
                    osc_icon_amp = (
                        "🟢"
                        if 0.7 <= osc_ratio_amp <= 1.5
                        else ("🟡" if 0.5 <= osc_ratio_amp <= 2.0 else "🔴")
                    )
                    st.metric("Osc Ratio", f"{osc_icon_amp} {osc_ratio_amp:.2f}")
                with mcols_amp[5]:
                    score_icon_amp = (
                        "🟢"
                        if score_result_amp["score"] >= 0.7
                        else ("🟡" if score_result_amp["score"] >= 0.5 else "🔴")
                    )
                    st.metric(
                        "Peak Score",
                        f"{score_icon_amp} {score_result_amp['score']:.3f}",
                    )

        except Exception as e:
            st.warning(f"Could not generate amplitude analysis: {e}")
    else:
        st.info("👈 Please select a spectrum file from the sidebar")

# =============================================================================
# Tab 3: Quality Metrics
# =============================================================================

with tabs[2]:
    if selected_file and Path(selected_file).exists():
        try:
            # Import quality metrics functions
            import sys
            from pathlib import Path as PathLib

            sys.path.insert(0, str(PathLib(__file__).parent.parent.parent / "src"))

            from analysis.quality_metrics import assess_spectrum_quality

            # Display function inline since we have the module
            def display_quality_card(
                wavelengths, reflectance, fitted_spectrum=None, prominence=0.0001
            ):
                report = assess_spectrum_quality(
                    wavelengths,
                    reflectance,
                    fitted_spectrum=fitted_spectrum,
                    prominence=prominence,
                )

                # Display overall quality
                quality_colors = {
                    "Excellent": "#16a34a",
                    "Good": "#2563eb",
                    "Marginal": "#d97706",
                    "Reject": "#dc2626",
                }
                quality_icons = {
                    "Excellent": "✅",
                    "Good": "✓",
                    "Marginal": "⚠️",
                    "Reject": "❌",
                }

                color = quality_colors.get(report.overall_quality, "#64748b")
                icon = quality_icons.get(report.overall_quality, "")

                st.markdown(
                    f"""
                <div style="background: {color}15; border: 2px solid {color}; border-radius: 12px; padding: 16px; margin-bottom: 16px;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
                                Overall Quality
                            </div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: {color};">
                                {icon} {report.overall_quality}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.75rem; color: #64748b;">
                                {"All checks passed" if report.passed_all_checks else f"{len(report.failures)} checks failed"}
                            </div>
                        </div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Display metric details
                with st.expander("📊 Quality Metrics Details", expanded=False):
                    cols = st.columns(2)
                    metric_idx = 0
                    for name, result in report.metrics.items():
                        col = cols[metric_idx % 2]
                        metric_idx += 1
                        with col:
                            status_icon = "✅" if result.passed else "❌"
                            st.markdown(
                                f"""
                            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                                <div style="font-weight: 600; color: #1e40af; margin-bottom: 8px;">
                                    {status_icon} {name.replace("_", " ").title()}
                                </div>
                                <div style="font-size: 0.85rem; color: #64748b;">
                                    Value: <span style="font-weight: 600;">{result.value:.4f}</span>
                                    <br>Threshold: {result.threshold:.4f}
                                </div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            if name == "snr":
                                st.caption(
                                    f"SNR: {result.details.get('snr', 0):.2f} ({result.details.get('quality_level', 'N/A')})"
                                )
                            elif name == "peak_quality":
                                st.caption(
                                    f"Peaks: {int(result.details.get('peak_count', 0))}, Prom CV: {result.details.get('prominence_cv', 0):.3f}"
                                )
                            elif name == "fit_quality":
                                st.caption(
                                    f"RMSE: {result.details.get('rmse', 0):.4f}, R²: {result.details.get('r_squared', 0):.4f}"
                                )

                if report.warnings:
                    with st.expander("⚠️ Warnings", expanded=False):
                        for warning in report.warnings:
                            st.warning(warning)

                if report.failures:
                    with st.expander("❌ Failures", expanded=len(report.failures) > 0):
                        for failure in report.failures:
                            st.error(failure)

                return report

            st.markdown("### 🔍 Spectrum Quality Assessment")
            st.markdown(
                '<p style="color: #94a3b8; margin-bottom: 1.5rem;">Automated quality metrics assess whether the measured spectrum is suitable for reliable thickness extraction.</p>',
                unsafe_allow_html=True,
            )

            # Get fitted spectrum if available for fit quality metrics
            fitted_spectrum = None
            if computed_data and "pyelli_reflectance" in computed_data:
                fitted_spectrum = computed_data["pyelli_reflectance"]

            # Display quality metrics
            prominence = st.session_state.get("peak_prom_amp", 0.0001)

            report = display_quality_card(
                wavelengths,
                measured,
                fitted_spectrum=fitted_spectrum,
                prominence=prominence,
            )

            # Add explanation section
            with st.expander("📚 About Quality Metrics", expanded=False):
                st.markdown("""
                ### Metric 1: Signal-to-Noise Ratio (SNR)
                Quantifies measurement quality by comparing signal strength to baseline noise.
                - **Formula (HORIBA FSD Standard)**: `SNR = (max(signal) - mean(baseline)) / std(baseline)`
                - **Implementation**: Signal is detrended first to separate interference fringes from spectral envelope
                - **Signal region**: Center 60% of wavelength range (e.g., 700-1000nm for 600-1120nm range)
                - **Baseline region**: First and last 10% of wavelength range (edges)
                - **Thresholds (Calibrated for Detrended Interference Spectra)**:
                  - SNR ≥ 2.5: **Excellent** (Top ~15%)
                  - SNR 1.5-2.5: **Good** (Top ~50%)
                  - SNR 1.0-1.5: **Marginal** (Top ~80%)
                  - SNR < 1.0: **Reject** (Bottom ~20%)
                
                ### Metric 2: Peak/Fringe Detection Quality
                Verifies sufficient interference fringes exist for reliable thickness extraction.
                - **Implementation**: Peaks detected on detrended signal (matches amplitude analysis)
                - Minimum peak count (≥ 3 peaks)
                - Peak prominence consistency (CV < 1.0)
                - Peak spacing regularity (CV < 0.5)
                
                ### Metric 3: Fit Residual Quality
                Quantifies how well the theoretical model matches measured data (only shown when fit is available).
                - RMSE (< 0.01 absolute)
                - Normalized RMSE (< 5%)
                - R² (> 0.90)
                - Reduced Chi-squared (0.8 - 1.2)
                
                ### Metric 4: Signal Integrity Checks
                Detects hardware/acquisition issues that invalidate the spectrum.
                - Dynamic range (> 0.05)
                - Saturation detection (< 1% saturated points)
                - Baseline stability (< 10% drift)
                - Negative value check (0% negative values)
                
                ### Metric 5: Spectral Completeness
                Ensures adequate wavelength coverage for fitting.
                - Wavelength span (≥ 400 nm coverage)
                - Data point density (≥ 1 point per nm)
                - Gap detection (no gaps > 5 nm)

                ---
                **Note on Sliding Window SNR:**
                The local SNR chart below uses a **'Robust' high-frequency noise method** (residue of signal differences). This calculation focuses on detecting hardware artifacts or noise floor variations in small regions. It differs from the **'Global' detrended SNR** above, as detrending inside small windows is unstable.
                """)

            # Show spectrum plot with quality overlay
            st.markdown("### 📈 Spectrum with Quality Regions")

            fig = go.Figure()

            # Add measured spectrum
            fig.add_trace(
                go.Scatter(
                    x=wavelengths,
                    y=measured,
                    mode="lines",
                    name="Measured Spectrum",
                    line=dict(color="#1e40af", width=2),
                )
            )

            # Add fitted spectrum if available
            if fitted_spectrum is not None:
                fig.add_trace(
                    go.Scatter(
                        x=wavelengths,
                        y=fitted_spectrum,
                        mode="lines",
                        name="Fitted Spectrum",
                        line=dict(color="#dc2626", width=2, dash="dash"),
                    )
                )

            # Highlight SNR regions
            snr_metric = report.metrics.get("snr")
            if snr_metric:
                n = len(wavelengths)
                edge_size = int(n * 0.1)
                center_start = int(n * 0.2)
                center_end = n - center_start

                # Baseline regions (edges)
                fig.add_vrect(
                    x0=wavelengths[0],
                    x1=wavelengths[edge_size],
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    layer="below",
                    line_width=0,
                    annotation_text="Baseline",
                    annotation_position="top left",
                )
                fig.add_vrect(
                    x0=wavelengths[-edge_size],
                    x1=wavelengths[-1],
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    layer="below",
                    line_width=0,
                    annotation_text="Baseline",
                    annotation_position="top right",
                )

                # Signal region (center)
                fig.add_vrect(
                    x0=wavelengths[center_start],
                    x1=wavelengths[center_end],
                    fillcolor="rgba(0, 255, 0, 0.05)",
                    layer="below",
                    line_width=0,
                    annotation_text="Signal Region",
                    annotation_position="top",
                )

            fig.update_layout(
                title="Measured Spectrum with Quality Assessment Regions",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Reflectance",
                hovermode="x unified",
                template="plotly_white",
                height=500,
                showlegend=True,
            )

            st.plotly_chart(fig, use_container_width=True)

            # --- SLIDING WINDOW SNR CHART ---
            snr_metric = report.metrics.get("snr")
            if snr_metric and "sliding_window_snr" in snr_metric.details:
                sw_data = snr_metric.details["sliding_window_snr"]

                if len(sw_data.get("centers", [])) > 0:
                    sw_window = sw_data.get("window_nm", 50.0)
                    st.markdown("---")
                    st.markdown("### 🏁 Local Signal Quality")
                    st.info(
                        f"**Sliding Window SNR**: Calculated using a {sw_window:.0f}nm sliding window. "
                        "This helps identify specific wavelength regions where the signal is degraded by hardware noise or sensor saturation. "
                        "*(Note: This local SNR uses a 'Robust' high-frequency noise method, which differs from the 'Global' detrended SNR above. Detrending inside small windows is avoided for stability.)*"
                    )

                    fig_sw = go.Figure()

                    # 1. The Intensity Shade (Bar underlay)
                    # Use a bar chart with bargap=0 to create a continuous intensity area
                    fig_sw.add_trace(
                        go.Bar(
                            x=sw_data["centers"],
                            y=sw_data["snr_values"],
                            marker=dict(
                                color=sw_data["snr_values"],
                                colorscale="Blues",
                                cmin=0,
                                showscale=True,
                                colorbar=dict(
                                    title="SNR Intensity",
                                    thickness=15,
                                    len=0.8,
                                    y=0.5,
                                    x=1.05,
                                ),
                            ),
                            width=sw_data.get("stride_nm", 25.0),
                            opacity=0.5,
                            showlegend=False,
                            name="Intensity",
                            hoverinfo="skip",
                        )
                    )

                    # 2. The Line (Trend overlay)
                    fig_sw.add_trace(
                        go.Scatter(
                            x=sw_data["centers"],
                            y=sw_data["snr_values"],
                            mode="lines+markers",
                            name="SNR Profile",
                            line=dict(color="#1e40af", width=2),
                            marker=dict(size=4, color="#1e40af"),
                            hovertemplate=(
                                "<b>Wavelength Range</b>: %{customdata[0]:.1f} - %{customdata[1]:.1f} nm<br>"
                                + "<b>SNR</b>: %{y:.1f}<br>"
                                + "<extra></extra>"
                            ),
                            customdata=sw_data["ranges"],
                        )
                    )

                    # Add threshold line (Robust Gate: 150.0 is Marginal)
                    fig_sw.add_hline(
                        y=150.0,
                        line_dash="dash",
                        line_color="#dc2626",
                        line_width=2,
                        annotation_text="Threshold (150.0)",
                        annotation_position="bottom right",
                    )

                    fig_sw.update_layout(
                        title=f"SNR vs Wavelength ({sw_window:.0f}nm Sliding Windows)",
                        xaxis_title="Wavelength (nm)",
                        yaxis_title="SNR Ratio (Variance)",
                        template="plotly_white",
                        height=450,
                        hovermode="closest",
                        bargap=0,  # Makes bars touch for continuous look
                        margin=dict(l=20, r=80, t=50, b=20),
                    )

                    st.plotly_chart(fig_sw, use_container_width=True)

                    # Metrics beneath the chart
                    m_cols = st.columns(3)
                    with m_cols[0]:
                        st.markdown(
                            f"**MIN SNR IN WINDOW**\n### {sw_data['min_snr']:.1f}"
                        )
                    with m_cols[1]:
                        st.markdown(f"**AVG SNR**\n### {sw_data['avg_snr']:.1f}")
                    with m_cols[2]:
                        st.markdown(
                            f"**MAX SNR IN WINDOW**\n### {sw_data['max_snr']:.1f}"
                        )

        except ImportError as e:
            st.error(f"❌ Quality metrics module not available: {e}")
            st.info(
                "The quality metrics feature requires the `quality_display` module."
            )
        except Exception as e:
            st.error(f"❌ Error displaying quality metrics: {e}")
            import traceback

            st.code(traceback.format_exc())
    else:
        st.info(
            "👈 Please select a spectrum file from the sidebar to view quality metrics"
        )


# =============================================================================
# Tab 2: Sample Data Viewer (Hidden)
# =============================================================================
# HIDDEN FOR CLIENT EXPERIMENTS

# with tabs[1]:
if False:  # Disabled for client
    st.markdown(
        """
    <div style="margin-bottom: 24px;">
        <h2>📊 Sample Data Viewer</h2>
        <p style="color: #94a3b8;">Explore measured spectra and their corresponding BestFit theoretical matches from ADOM's LTA software.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1_sdv, col2_sdv = st.columns([1, 3])

    with col1_sdv:
        st.markdown("### Sample Selection")

        fit_category = st.radio(
            "Fit Quality",
            ["good_fit", "bad_fit"],
            format_func=lambda x: "✅ Good Fits" if x == "good_fit" else "❌ Bad Fits",
            key="sdv_category",
        )

        available_samples = list(samples[fit_category].keys())
        selected_sample = st.selectbox(
            "Sample ID",
            available_samples,
            format_func=lambda x: f"Sample {x}",
            key="sdv_sample",
        )

        show_residual_sdv = st.checkbox(
            "Show Residual Plot", value=True, key="sdv_residual"
        )
        wavelength_range_sdv = st.slider(
            "Wavelength Range (nm)", 400, 1200, (600, 1100), step=10, key="sdv_wl"
        )

    with col2_sdv:
        if selected_sample:
            sample_info = samples[fit_category][selected_sample]

            if sample_info["measured"] and sample_info["bestfit"]:
                measured_wl, measured_refl = load_measured_spectrum(
                    sample_info["measured"]
                )
                bestfit_wl, bestfit_refl = load_bestfit_spectrum(sample_info["bestfit"])

                common_wl, meas_interp, best_interp = interpolate_to_common_wavelengths(
                    measured_wl, measured_refl, bestfit_wl, bestfit_refl
                )

                mask = (common_wl >= wavelength_range_sdv[0]) & (
                    common_wl <= wavelength_range_sdv[1]
                )
                common_wl_filtered = common_wl[mask]
                meas_filtered = meas_interp[mask]
                best_filtered = best_interp[mask]

                residual_sdv = calculate_residual(meas_filtered, best_filtered)
                correlation_sdv = calculate_correlation(meas_filtered, best_filtered)

                if show_residual_sdv:
                    fig_sdv = make_subplots(
                        rows=2,
                        cols=1,
                        row_heights=[0.7, 0.3],
                        shared_xaxes=True,
                        vertical_spacing=0.08,
                    )
                    fig_sdv.add_trace(
                        go.Scatter(
                            x=common_wl_filtered,
                            y=meas_filtered,
                            mode="lines",
                            name="Measured",
                            line=dict(color="#2563eb", width=2.5),
                        ),
                        row=1,
                        col=1,
                    )
                    fig_sdv.add_trace(
                        go.Scatter(
                            x=common_wl_filtered,
                            y=best_filtered,
                            mode="lines",
                            name="BestFit (LTA)",
                            line=dict(color="#db2777", width=2.5, dash="dash"),
                        ),
                        row=1,
                        col=1,
                    )
                    fig_sdv.add_trace(
                        go.Scatter(
                            x=common_wl_filtered,
                            y=meas_filtered - best_filtered,
                            mode="lines",
                            name="Residual",
                            line=dict(color="#d97706", width=1.5),
                            fill="tozeroy",
                            fillcolor="rgba(251, 191, 36, 0.15)",
                        ),
                        row=2,
                        col=1,
                    )
                    fig_sdv.add_hline(
                        y=0, line_dash="dot", line_color="gray", row=2, col=1
                    )
                    fig_sdv.update_xaxes(title_text="Wavelength (nm)", row=2, col=1)
                    fig_sdv.update_yaxes(title_text="Reflectance", row=1, col=1)
                    fig_sdv.update_yaxes(title_text="Δ", row=2, col=1)
                    height_sdv = 550
                else:
                    fig_sdv = go.Figure()
                    fig_sdv.add_trace(
                        go.Scatter(
                            x=common_wl_filtered,
                            y=meas_filtered,
                            mode="lines",
                            name="Measured",
                            line=dict(color="#2563eb", width=2.5),
                        )
                    )
                    fig_sdv.add_trace(
                        go.Scatter(
                            x=common_wl_filtered,
                            y=best_filtered,
                            mode="lines",
                            name="BestFit (LTA)",
                            line=dict(color="#db2777", width=2.5, dash="dash"),
                        )
                    )
                    fig_sdv.update_xaxes(title_text="Wavelength (nm)")
                    fig_sdv.update_yaxes(title_text="Reflectance")
                    height_sdv = 400

                fig_sdv.update_layout(
                    height=height_sdv,
                    margin=dict(t=30, b=30, l=60, r=30),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                )
                st.plotly_chart(fig_sdv, width="stretch")

                mcols_sdv = st.columns(4)
                with mcols_sdv[0]:
                    st.metric("RMS Residual", f"{residual_sdv:.6f}")
                with mcols_sdv[1]:
                    st.metric("Correlation", f"{correlation_sdv:.4f}")
                with mcols_sdv[2]:
                    st.metric("Data Points", len(common_wl_filtered))
                with mcols_sdv[3]:
                    quality_text = (
                        "✅ Good" if fit_category == "good_fit" else "❌ Poor"
                    )
                    st.metric("Fit Quality", quality_text)
            else:
                st.warning("⚠️ Missing spectrum files for this sample")


# =============================================================================
# Tab 3: Material Properties
# =============================================================================
# HIDDEN FOR CLIENT EXPERIMENTS

# with tabs[2]:
if False:  # Disabled for client
    st.markdown(
        """
    <div style="margin-bottom: 24px;">
        <h2>🌈 Material Optical Properties</h2>
        <p style="color: #94a3b8;">Visualize refractive index (n) and extinction coefficient (k) dispersion data for tear film materials.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Material selection
    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### Material Selection")

        # Categorize materials
        tear_film_materials = {
            "lipid_05-02621extrapolated": "Lipid Layer",
            "water_Bashkatov1353extrapolated": "Aqueous (Water)",
            "struma_Bashkatov140extrapolated": "Mucus/Stroma",
        }

        substrate_materials = {
            "BK7": "BK7 Glass",
            "SiO2": "Silicon Dioxide",
        }

        material_category = st.radio(
            "Category", ["Tear Film", "Substrates", "All Materials"]
        )

        if material_category == "Tear Film":
            available_mats = [m for m in tear_film_materials.keys() if m in materials]
            selected_materials = st.multiselect(
                "Select Materials",
                available_mats,
                default=available_mats[:3],
                format_func=lambda x: tear_film_materials.get(x, x),
            )
        elif material_category == "Substrates":
            available_mats = [m for m in substrate_materials.keys() if m in materials]
            selected_materials = st.multiselect(
                "Select Materials",
                available_mats,
                default=available_mats[:2] if available_mats else [],
                format_func=lambda x: substrate_materials.get(x, x),
            )
        else:
            selected_materials = st.multiselect(
                "Select Materials",
                list(materials.keys()),
                default=list(tear_film_materials.keys())[:3],
            )

        show_extinction = st.checkbox("Show Extinction (k)", value=False)

        wl_range = st.slider("Wavelength Range (nm)", 200, 1200, (400, 1100), step=10)

    with col2:
        if selected_materials:
            fig = make_subplots(
                rows=2 if show_extinction else 1,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=["Refractive Index (n)", "Extinction Coefficient (k)"]
                if show_extinction
                else ["Refractive Index (n)"],
            )

            colors = [
                "#2563eb",
                "#db2777",
                "#059669",
                "#d97706",
                "#7c3aed",
                "#dc2626",
            ]  # Light-theme friendly palette

            for i, mat_name in enumerate(selected_materials):
                mat_df = load_material_data(materials[mat_name])

                # Filter wavelength range
                mask = (mat_df["wavelength_nm"] >= wl_range[0]) & (
                    mat_df["wavelength_nm"] <= wl_range[1]
                )
                mat_df = mat_df[mask]

                display_name = tear_film_materials.get(mat_name, mat_name)

                # Add n trace
                fig.add_trace(
                    go.Scatter(
                        x=mat_df["wavelength_nm"],
                        y=mat_df["n"],
                        mode="lines",
                        name=f"{display_name}",
                        line=dict(color=colors[i % len(colors)], width=2),
                    ),
                    row=1,
                    col=1,
                )

                # Add k trace if enabled
                if show_extinction:
                    fig.add_trace(
                        go.Scatter(
                            x=mat_df["wavelength_nm"],
                            y=mat_df["k"],
                            mode="lines",
                            name=f"{display_name} (k)",
                            line=dict(
                                color=colors[i % len(colors)], width=2, dash="dot"
                            ),
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )

            fig.update_xaxes(
                title_text="Wavelength (nm)",
                row=2 if show_extinction else 1,
                col=1,
                **PLOTLY_TEMPLATE["layout"]["xaxis"],
            )
            fig.update_yaxes(
                title_text="Refractive Index (n)",
                row=1,
                col=1,
                **PLOTLY_TEMPLATE["layout"]["yaxis"],
            )
            if show_extinction:
                fig.update_yaxes(
                    title_text="Extinction (k)",
                    row=2,
                    col=1,
                    **PLOTLY_TEMPLATE["layout"]["yaxis"],
                )

            fig.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(family="DM Sans, sans-serif", color="#374151"),
                height=550 if show_extinction else 400,
                margin=dict(t=40, b=40, l=60, r=30),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor="#ffffff",
                    bordercolor="#e5e7eb",
                    font=dict(color="#374151"),
                ),
            )

            st.plotly_chart(fig, width="stretch")

            # Material info table
            st.markdown("### Material Summary")
            summary_data = []
            for mat_name in selected_materials:
                mat_df = load_material_data(materials[mat_name])
                summary_data.append(
                    {
                        "Material": tear_film_materials.get(mat_name, mat_name),
                        "λ Min (nm)": f"{mat_df['wavelength_nm'].min():.1f}",
                        "λ Max (nm)": f"{mat_df['wavelength_nm'].max():.1f}",
                        "n @ 550nm": f"{np.interp(550, mat_df['wavelength_nm'], mat_df['n']):.4f}",
                        "n @ 800nm": f"{np.interp(800, mat_df['wavelength_nm'], mat_df['n']):.4f}",
                    }
                )

            st.dataframe(pd.DataFrame(summary_data), width="stretch")
        else:
            st.info(
                "👈 Select materials from the sidebar to visualize their optical properties"
            )


# =============================================================================
# Tab 4: PyElli Structure Demo
# =============================================================================
# HIDDEN FOR CLIENT EXPERIMENTS

# with tabs[3]:
if False:  # Disabled for client
    st.markdown(
        """
    <div style="margin-bottom: 24px;">
        <h2>🔧 PyElli Structure Builder</h2>
        <p style="color: #94a3b8;">Build multi-layer thin film structures and calculate optical responses using the Transfer Matrix Method.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Check if pyElli is available
    try:
        import elli

        pyelli_available = True
    except ImportError:
        pyelli_available = False

    if not pyelli_available:
        st.warning("""
        ⚠️ **pyElli not installed**
        
        To run the full pyElli demos, install it with:
        ```bash
        pip install pyElli
        ```
        
        The demo below shows the conceptual approach using a simplified model.
        """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Layer Configuration")

        st.markdown("#### Tear Film Structure")
        st.markdown("`Air → Lipid → Aqueous → Mucus → Eye`")

        lipid_thickness = st.slider(
            "Lipid Thickness (nm)",
            9,
            250,
            80,
            help="Accepted range: 9-250 nm",
            key="demo_lipid",
        )

        aqueous_thickness = st.slider(
            "Aqueous Thickness (nm)",
            800,
            12000,
            2337,
            step=50,
            help="Accepted range: 800-12000 nm",
            key="demo_aqueous",
        )

        mucus_thickness = st.slider(
            "Mucus Thickness (nm)",
            500,
            500,
            500,
            step=50,
            help="Fixed at 500 nm",
            key="demo_mucus",
            disabled=True,
        )

        st.markdown("#### Calculation Parameters")

        calc_wavelength_range = st.slider(
            "Wavelength Range (nm)",
            400,
            1200,
            (600, 1100),
            step=10,
            key="demo_wavelength",
        )

        num_points = st.slider(
            "Number of Points", 100, 1000, 500, step=50, key="demo_num_points"
        )

    with col2:
        st.markdown("### Theoretical Spectrum")

        # Load material data
        lipid_df = load_material_data(materials["lipid_05-02621extrapolated"])
        water_df = load_material_data(materials["water_Bashkatov1353extrapolated"])
        mucus_df = load_material_data(materials["struma_Bashkatov140extrapolated"])

        # Create wavelength array
        wavelengths = np.linspace(
            calc_wavelength_range[0], calc_wavelength_range[1], num_points
        )

        # Interpolate material properties
        def get_nk(mat_df, wavelengths):
            n = np.interp(wavelengths, mat_df["wavelength_nm"], mat_df["n"])
            k = np.interp(wavelengths, mat_df["wavelength_nm"], mat_df["k"])
            return n, k

        lipid_n, lipid_k = get_nk(lipid_df, wavelengths)
        water_n, water_k = get_nk(water_df, wavelengths)
        mucus_n, mucus_k = get_nk(mucus_df, wavelengths)

        # Simple multi-layer reflectance calculation
        # Using transfer matrix method approximation
        def transfer_matrix_reflectance(wavelengths, layers):
            """
            Calculate reflectance using transfer matrix method.
            Each layer is (n_array, k_array, thickness_nm)
            """
            n_air = 1.0
            reflectance = np.zeros_like(wavelengths)

            for i, wl in enumerate(wavelengths):
                # Build complex refractive indices
                N = [n_air]  # Start with air
                d = [0]  # Air has no thickness

                for n_arr, k_arr, thickness in layers:
                    N.append(n_arr[i] + 1j * k_arr[i])
                    d.append(thickness)

                # Substrate (approximate as semi-infinite mucus)
                N.append(N[-1])

                # Transfer matrix calculation
                M = np.eye(2, dtype=complex)

                for j in range(1, len(N) - 1):
                    # Interface matrix
                    r_jk = (N[j - 1] - N[j]) / (N[j - 1] + N[j])
                    t_jk = 2 * N[j - 1] / (N[j - 1] + N[j])

                    I_jk = np.array([[1, r_jk], [r_jk, 1]], dtype=complex) / t_jk

                    # Propagation matrix
                    delta = 2 * np.pi * N[j] * d[j] / wl
                    L_j = np.array(
                        [[np.exp(-1j * delta), 0], [0, np.exp(1j * delta)]],
                        dtype=complex,
                    )

                    M = M @ I_jk @ L_j

                # Final interface
                r_final = (N[-2] - N[-1]) / (N[-2] + N[-1])
                t_final = 2 * N[-2] / (N[-2] + N[-1])
                I_final = (
                    np.array([[1, r_final], [r_final, 1]], dtype=complex) / t_final
                )

                M = M @ I_final

                # Reflectance from M matrix
                r = M[1, 0] / M[0, 0]
                reflectance[i] = np.abs(r) ** 2

            return reflectance

        # Calculate reflectance
        layers = [
            (lipid_n, lipid_k, lipid_thickness),
            (water_n, water_k, aqueous_thickness),
            (mucus_n, mucus_k, mucus_thickness),
        ]

        theoretical_reflectance = transfer_matrix_reflectance(wavelengths, layers)

        # Create figure
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=wavelengths,
                y=theoretical_reflectance,
                mode="lines",
                name="Theoretical (TMM)",
                line=dict(color=COLORS["theoretical"], width=2.5),
                fill="tozeroy",
                fillcolor="rgba(124, 58, 237, 0.1)",
            )
        )

        fig.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="DM Sans, sans-serif", color="#374151"),
            height=400,
            margin=dict(t=60, b=40, l=60, r=30),
            title=dict(
                text=f'<b>Theoretical Reflectance</b><br><span style="font-size:12px;color:#64748b">Lipid: {lipid_thickness}nm | Aqueous: {aqueous_thickness}nm | Mucus: {mucus_thickness}nm</span>',
                font=dict(size=16, color="#1e40af"),
            ),
            xaxis=dict(
                gridcolor="#e5e7eb",
                linecolor="#d1d5db",
                tickfont=dict(color="#6b7280"),
                title=dict(text="Wavelength (nm)", font=dict(color="#374151")),
            ),
            yaxis=dict(
                gridcolor="#e5e7eb",
                linecolor="#d1d5db",
                tickfont=dict(color="#6b7280"),
                title=dict(text="Reflectance", font=dict(color="#374151")),
            ),
        )

        st.plotly_chart(fig, width="stretch")

        # Structure visualization
        st.markdown("### Layer Structure Diagram")

        # Create a simple bar chart to visualize layer thicknesses
        fig_structure = go.Figure()

        layer_names = ["Air", "Lipid", "Aqueous", "Mucus", "Eye"]
        layer_thicknesses = [0, lipid_thickness, aqueous_thickness, mucus_thickness, 0]
        layer_colors = ["#475569", "#fbbf24", "#60a5fa", "#34d399", "#f472b6"]

        # Vertical bar representation
        cumulative = 0
        for name, thickness, color in zip(layer_names, layer_thicknesses, layer_colors):
            if thickness > 0:
                fig_structure.add_trace(
                    go.Bar(
                        x=[name],
                        y=[thickness],
                        name=name,
                        marker_color=color,
                        text=f"{thickness} nm",
                        textposition="inside",
                    )
                )

        fig_structure.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="DM Sans, sans-serif", color="#374151"),
            height=300,
            margin=dict(t=50, b=40, l=60, r=30),
            showlegend=False,
            title=dict(
                text="Layer Thickness Comparison", font=dict(size=14, color="#1e40af")
            ),
            xaxis=dict(
                gridcolor="rgba(139, 92, 246, 0.1)",
                linecolor="rgba(139, 92, 246, 0.3)",
                tickfont=dict(color="#94a3b8"),
            ),
            yaxis=dict(
                gridcolor="#e5e7eb",
                linecolor="#d1d5db",
                tickfont=dict(color="#6b7280"),
                title=dict(text="Thickness (nm)", font=dict(color="#374151")),
            ),
        )

        st.plotly_chart(fig_structure, width="stretch")


# =============================================================================
# Tab 5: Fitting Comparison
# =============================================================================
# HIDDEN FOR CLIENT EXPERIMENTS

# with tabs[4]:
if False:  # Disabled for client
    st.markdown(
        """
    <div style="margin-bottom: 24px;">
        <h2>📈 Fitting Comparison</h2>
        <p style="color: #94a3b8;">Compare pyElli-generated theoretical spectra with measured data and LTA BestFit results.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### Sample Selection")

        compare_category = st.radio(
            "Fit Category",
            ["good_fit", "bad_fit"],
            format_func=lambda x: "✅ Good Fits" if x == "good_fit" else "❌ Bad Fits",
            key="compare_category",
        )

        compare_sample = st.selectbox(
            "Sample",
            list(samples[compare_category].keys()),
            format_func=lambda x: f"Sample {x}",
            key="compare_sample",
        )

        st.markdown("---")
        st.markdown("### Thickness Parameters")

        fit_lipid = st.slider("Lipid (nm)", 9, 250, 80, key="fit_lipid")

        fit_aqueous = st.slider(
            "Aqueous (nm)", 800, 12000, 2337, step=25, key="fit_aqueous"
        )

        fit_mucus = st.slider(
            "Mucus (nm)", 500, 500, 500, step=25, key="fit_mucus", disabled=True
        )

        st.markdown("---")
        auto_fit = st.button("🔍 Grid Search (Simple)", width="stretch")

    with col2:
        sample_info = samples[compare_category][compare_sample]

        if sample_info["measured"] and sample_info["bestfit"]:
            # Load measured and bestfit spectra
            measured_wl, measured_refl = load_measured_spectrum(sample_info["measured"])
            bestfit_wl, bestfit_refl = load_bestfit_spectrum(sample_info["bestfit"])

            # Common wavelength range
            wl_min = max(measured_wl.min(), bestfit_wl.min(), 600)
            wl_max = min(measured_wl.max(), bestfit_wl.max(), 1100)

            common_wavelengths = np.linspace(wl_min, wl_max, 500)
            measured_interp = np.interp(common_wavelengths, measured_wl, measured_refl)
            bestfit_interp = np.interp(common_wavelengths, bestfit_wl, bestfit_refl)

            # Calculate pyElli theoretical
            lipid_df = load_material_data(materials["lipid_05-02621extrapolated"])
            water_df = load_material_data(materials["water_Bashkatov1353extrapolated"])
            mucus_df = load_material_data(materials["struma_Bashkatov140extrapolated"])

            lipid_n, lipid_k = get_nk(lipid_df, common_wavelengths)
            water_n, water_k = get_nk(water_df, common_wavelengths)
            mucus_n, mucus_k = get_nk(mucus_df, common_wavelengths)

            layers = [
                (lipid_n, lipid_k, fit_lipid),
                (water_n, water_k, fit_aqueous),
                (mucus_n, mucus_k, fit_mucus),
            ]

            pyelli_refl = transfer_matrix_reflectance(common_wavelengths, layers)

            # Scale pyElli to match measured amplitude
            scale_factor = np.mean(measured_interp) / np.mean(pyelli_refl)
            pyelli_scaled = pyelli_refl * scale_factor

            # Calculate metrics
            residual_lta = calculate_residual(measured_interp, bestfit_interp)
            residual_pyelli = calculate_residual(measured_interp, pyelli_scaled)
            corr_lta = calculate_correlation(measured_interp, bestfit_interp)
            corr_pyelli = calculate_correlation(measured_interp, pyelli_scaled)

            # Create comparison figure
            fig = make_subplots(
                rows=2,
                cols=1,
                row_heights=[0.65, 0.35],
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=["Spectrum Comparison", "Residuals"],
            )

            # Spectra
            fig.add_trace(
                go.Scatter(
                    x=common_wavelengths,
                    y=measured_interp,
                    mode="lines",
                    name="Measured",
                    line=dict(color=COLORS["measured"], width=2.5),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=common_wavelengths,
                    y=bestfit_interp,
                    mode="lines",
                    name="LTA BestFit",
                    line=dict(color=COLORS["bestfit"], width=2.5, dash="dash"),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=common_wavelengths,
                    y=pyelli_scaled,
                    mode="lines",
                    name="TMM Theoretical",
                    line=dict(color=COLORS["theoretical"], width=2.5, dash="dot"),
                ),
                row=1,
                col=1,
            )

            # Residuals
            fig.add_trace(
                go.Scatter(
                    x=common_wavelengths,
                    y=measured_interp - bestfit_interp,
                    mode="lines",
                    name="LTA Residual",
                    line=dict(color=COLORS["bestfit"], width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(244, 114, 182, 0.1)",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=common_wavelengths,
                    y=measured_interp - pyelli_scaled,
                    mode="lines",
                    name="TMM Residual",
                    line=dict(color=COLORS["theoretical"], width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(167, 139, 250, 0.1)",
                ),
                row=2,
                col=1,
            )

            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

            fig.update_xaxes(
                title_text="Wavelength (nm)",
                row=2,
                col=1,
                **PLOTLY_TEMPLATE["layout"]["xaxis"],
            )
            fig.update_yaxes(
                title_text="Reflectance",
                row=1,
                col=1,
                **PLOTLY_TEMPLATE["layout"]["yaxis"],
            )
            fig.update_yaxes(
                title_text="Residual",
                row=2,
                col=1,
                **PLOTLY_TEMPLATE["layout"]["yaxis"],
            )

            fig.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(family="DM Sans, sans-serif", color="#374151"),
                height=550,
                margin=dict(t=40, b=40, l=60, r=30),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor="#ffffff",
                    bordercolor="#e5e7eb",
                    font=dict(color="#374151"),
                ),
            )

            st.plotly_chart(fig, width="stretch")

            # Metrics comparison
            st.markdown("### Fit Quality Metrics")

            metric_cols = st.columns(4)

            with metric_cols[0]:
                st.metric("LTA Residual", f"{residual_lta:.6f}", delta=None)

            with metric_cols[1]:
                st.metric(
                    "TMM Residual",
                    f"{residual_pyelli:.6f}",
                    delta=f"{(residual_pyelli - residual_lta) / residual_lta * 100:.1f}%"
                    if residual_lta > 0
                    else None,
                    delta_color="inverse",
                )

            with metric_cols[2]:
                st.metric("LTA Correlation", f"{corr_lta:.4f}")

            with metric_cols[3]:
                st.metric(
                    "TMM Correlation",
                    f"{corr_pyelli:.4f}",
                    delta=f"{(corr_pyelli - corr_lta) * 100:.2f}%"
                    if corr_lta > 0
                    else None,
                )

            # Grid search result
            if auto_fit:
                st.markdown("### 🔍 Grid Search Results")
                with st.spinner("Running grid search..."):
                    best_residual = float("inf")
                    best_params = {}

                    # Coarse grid search
                    for lipid in range(20, 150, 20):
                        for aqueous in range(1500, 4000, 250):
                            for mucus in range(200, 600, 100):
                                layers_test = [
                                    (lipid_n, lipid_k, lipid),
                                    (water_n, water_k, aqueous),
                                    (mucus_n, mucus_k, mucus),
                                ]
                                test_refl = transfer_matrix_reflectance(
                                    common_wavelengths, layers_test
                                )
                                test_scaled = test_refl * (
                                    np.mean(measured_interp) / np.mean(test_refl)
                                )
                                test_residual = calculate_residual(
                                    measured_interp, test_scaled
                                )

                                if test_residual < best_residual:
                                    best_residual = test_residual
                                    best_params = {
                                        "lipid": lipid,
                                        "aqueous": aqueous,
                                        "mucus": mucus,
                                    }

                    st.success(f"""
                    **Best Parameters Found:**
                    - Lipid: {best_params["lipid"]} nm
                    - Aqueous: {best_params["aqueous"]} nm
                    - Mucus: {best_params["mucus"]} nm
                    - Residual: {best_residual:.6f}
                    """)


# =============================================================================
# Tab 6: Integration Guide
# =============================================================================
# HIDDEN FOR CLIENT EXPERIMENTS

# with tabs[5]:
if False:  # Disabled for client
    st.markdown(
        """
    <div style="margin-bottom: 24px;">
        <h2>📚 Integration Guide</h2>
        <p style="color: #94a3b8;">Comprehensive guide for integrating pyElli into the AdOM-TFI workflow.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("""
    ### What is PyElli?
    
    **pyElli** is an open-source Python library for spectroscopic ellipsometry and
    thin film optics calculations. Key features include:
    
    - 🔢 **Transfer Matrix Method (TMM)** - Standard approach for multi-layer thin film calculations
    - 🌈 **Dispersion Models** - Cauchy, Sellmeier, Lorentz, Drude, and tabulated data support
    - 📐 **Ellipsometry Fitting** - Psi/Delta fitting with various optimizers
    - 🔄 **Wavelength-by-wavelength** or **Global** fitting modes
    
    ### Applicability to AdOM-TFI
    
    Based on the codebase analysis, here's how pyElli could enhance the workflow:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### ✅ High-Value Applications
        
        1. **Replace/Validate SpAnalizer.dll**
           - Open-source alternative to proprietary DLL
           - Cross-platform (no Windows dependency)
           - Full control over calculation parameters
        
        2. **Dispersion Model Flexibility**
           - Use tabulated n,k data directly
           - Fit Cauchy/Sellmeier models to material data
           - Model temperature-dependent properties
        
        3. **Advanced Fitting Algorithms**
           - scipy.optimize integration
           - Global optimization methods
           - Constrained fitting
        
        4. **Uncertainty Quantification**
           - Parameter covariance estimation
           - Confidence intervals on thicknesses
        """)

    with col2:
        st.markdown("""
        #### 🔄 Integration Considerations
        
        1. **Material Data Conversion**
           ```python
           # Convert existing CSV to pyElli format
           from elli.dispersions import TableIndex
           
           # Load your material data
           mat = TableIndex.from_csv('lipid.csv')
           ```
        
        2. **Structure Building**
           ```python
           import elli
           
           # Build tear film structure
           structure = elli.Structure(
               elli.AIR,
               elli.Layer(lipid_mat, d_lipid),
               elli.Layer(aqueous_mat, d_aqueous),
               elli.Layer(mucus_mat, d_mucus),
               substrate
           )
           ```
        
        3. **Reflectance Calculation**
           ```python
           # Calculate reflectance
           result = structure.evaluate(
               wavelengths, 
               angle=0,
               polarization='s'
           )
           R = result.R_s
           ```
        """)

    st.markdown("---")

    st.markdown("""
    ### Recommended Integration Path
    
    ```mermaid
    graph LR
        A[Current: SpAnalizer.dll] --> B[Phase 1: Validation]
        B --> C[Phase 2: Parallel Implementation]
        C --> D[Phase 3: Full Migration]
        
        B --> B1[Compare TMM vs DLL output]
        B --> B2[Verify material interpolation]
        B --> B3[Benchmark performance]
        
        C --> C1[pyElli grid search module]
        C --> C2[Side-by-side results]
        C --> C3[Regression testing]
        
        D --> D1[Replace DLL dependency]
        D --> D2[Extend with new features]
        D --> D3[Integrate fitting algorithms]
    ```
    """)

    st.markdown("---")

    st.markdown("### Code Example: Full Tear Film Analysis")

    st.code(
        """
import numpy as np
import elli
from elli.dispersions import TableIndex, Cauchy
from scipy.optimize import minimize

# 1. Load material data
lipid = TableIndex.from_file('data/Materials/lipid_extrapolated.csv')
water = TableIndex.from_file('data/Materials/water_extrapolated.csv')  
mucus = TableIndex.from_file('data/Materials/mucus_extrapolated.csv')

# 2. Define structure function
def build_tear_film(d_lipid, d_aqueous, d_mucus):
    return elli.Structure(
        elli.AIR,
        elli.Layer(lipid, d_lipid),
        elli.Layer(water, d_aqueous),
        elli.Layer(mucus, d_mucus),
        elli.IsotropicMaterial(n=1.38)  # Simplified eye substrate
    )

# 3. Define objective function
def residual(params, wavelengths, measured_R):
    d_lipid, d_aqueous, d_mucus = params
    structure = build_tear_film(d_lipid, d_aqueous, d_mucus)
    
    result = structure.evaluate(wavelengths, angle=0)
    theoretical_R = result.R_s
    
    return np.sum((measured_R - theoretical_R)**2)

# 4. Run optimization
initial_guess = [80, 2500, 500]  # nm
bounds = [(20, 200), (500, 5000), (100, 1000)]

result = minimize(
    residual, 
    initial_guess,
    args=(wavelengths, measured_reflectance),
    bounds=bounds,
    method='L-BFGS-B'
)

best_lipid, best_aqueous, best_mucus = result.x
print(f"Best fit: Lipid={best_lipid:.1f}nm, Aqueous={best_aqueous:.1f}nm, Mucus={best_mucus:.1f}nm")
""",
        language="python",
    )

    st.markdown("---")

    st.info("""
    **Next Steps for Team Discussion:**
    1. Run validation tests comparing pyElli TMM vs current SpAnalizer.dll output
    2. Quantify computational performance differences  
    3. Evaluate fitting convergence on good_fit vs bad_fit samples
    4. Decide on phased integration timeline
    """)


# =============================================================================
# Footer
# =============================================================================

st.markdown(
    """
<div class="footer">
    <p style="font-size: 0.85rem; margin-bottom: 4px; color: #64748b;">
        <span style="color: #1e40af; font-weight: 600;">PyElli Exploration Tool</span> | AdOM Tear Film Interferometry Project
    </p>
    <p style="font-size: 0.75rem; color: #94a3b8;">For research and development purposes only</p>
</div>
""",
    unsafe_allow_html=True,
)
