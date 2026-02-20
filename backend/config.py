"""Backend config: single source for paths and constants."""
import logging
import os
import time
from pathlib import Path
from typing import Dict

_here = Path(__file__).resolve().parent
PROJECT_ROOT = _here.parent

# Materials path: env override, or project-relative (works local and in Docker: /app/data/Materials)
_env_path = os.getenv("BACKEND_MATERIALS_PATH")
MATERIALS_PATH = Path(_env_path) if _env_path else (PROJECT_ROOT / "data" / "Materials")

# Full test cycles: directory containing run folders (e.g. "Full test - 0007_2025-12-30_15-12-20")
_env_full_test = os.getenv("BACKEND_FULL_TEST_CYCLES_DIR")
FULL_TEST_CYCLES_DIR = Path(_env_full_test) if _env_full_test else (PROJECT_ROOT / "exploration" / "full_test_cycles")

# Last-activity timestamp (epoch seconds), updated by middleware on every request.
_last_activity: float = time.time()


def touch_activity() -> None:
    global _last_activity
    _last_activity = time.time()


def get_last_activity() -> float:
    return _last_activity


# Spectrum sources available on the backend filesystem.
# Keys are display names matching the Streamlit sidebar; values are paths relative to PROJECT_ROOT.
SPECTRUM_SOURCES: Dict[str, str] = {
    "More Good Spectras": "exploration/more_good_spectras/Corrected_Spectra",
    "New Spectra": "exploration/new_spectra",
    "Shlomo Raw Spectra": "exploration/spectra_from_shlomo",
    "Sample Data (Good Fit)": "exploration/sample_data/good_fit",
    "Sample Data (Bad Fit)": "exploration/sample_data/bad_fit",
    "Exploration Measurements": "exploration/measurements",
}

# Sources whose files are in nested subdirectories (subdir/*.txt) rather than flat.
NESTED_SPECTRUM_SOURCES = {"Sample Data (Good Fit)", "Sample Data (Bad Fit)", "New Spectra"}

_logger = logging.getLogger(__name__)
if not MATERIALS_PATH.exists():
    _logger.warning("Materials path not found: %s (set BACKEND_MATERIALS_PATH or mount data/Materials)", MATERIALS_PATH)
