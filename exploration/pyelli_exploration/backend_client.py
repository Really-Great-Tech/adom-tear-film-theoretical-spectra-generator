"""
HTTP client for PyElli backend API. The Streamlit app requires the backend at all times.
Single responsibility: talk to backend; no business logic.
Loads BACKEND_URL from .env in project root if set.
"""
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests
from dotenv import load_dotenv

_logger = logging.getLogger(__name__)

# Load .env from project root so BACKEND_URL is set without exporting manually
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")

# POST returns 202 immediately; client polls progress
GRID_SEARCH_START_TIMEOUT_SECONDS = 30
# POST returns 202 immediately; client polls progress. Short timeout for the POST only.
LAST_TWO_CYCLES_START_TIMEOUT_SECONDS = 30
THEORETICAL_TIMEOUT_SECONDS = 30
ALIGN_TIMEOUT_SECONDS = 10

# Cold-start: max time to wait for the backend to become healthy after Lambda wake-up
COLD_START_TIMEOUT_SECONDS = 180
COLD_START_POLL_INTERVAL_SECONDS = 5


BACKEND_REQUIRED_MESSAGE = (
    "BACKEND_URL is not set. Add it to a .env file in the project root "
    "(see .env.example), e.g. BACKEND_URL=http://127.0.0.1:8000"
)


def get_backend_url() -> str:
    """Backend base URL from env. Raises RuntimeError if not set (app requires backend)."""
    url = os.getenv("BACKEND_URL", "").strip()
    if not url:
        raise RuntimeError(BACKEND_REQUIRED_MESSAGE)
    return url


def _api(path: str) -> str:
    base = get_backend_url()
    return f"{base.rstrip('/')}/api{path}"


# ---------------------------------------------------------------------------
# Cold-start / auto-scaling helpers (production only, gated by env var)
# ---------------------------------------------------------------------------

def _get_lambda_url() -> Optional[str]:
    """Return the Lambda orchestrator URL if configured, else None (local dev)."""
    url = os.getenv("BACKEND_START_LAMBDA_URL", "").strip()
    return url or None


def _backend_is_healthy() -> bool:
    """Quick health check against the backend."""
    try:
        resp = requests.get(f"{get_backend_url().rstrip('/')}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def ensure_backend_running(
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """If BACKEND_START_LAMBDA_URL is set, invoke the Lambda to start the EC2 backend and wait until healthy.

    In local dev (no Lambda URL), this is a no-op.
    ``progress_callback`` receives short status strings like "Starting backend..." for UI display.
    """
    lambda_url = _get_lambda_url()
    if not lambda_url:
        return  # local dev — backend assumed running

    if _backend_is_healthy():
        return  # already warm

    if progress_callback:
        progress_callback("Starting backend (cold start)...")

    _logger.info("Backend not healthy; invoking Lambda to start EC2: %s", lambda_url)
    try:
        resp = requests.post(lambda_url, json={"action": "start"}, timeout=30)
        resp.raise_for_status()
        _logger.info("Lambda response: %s", resp.json())
    except Exception as exc:
        _logger.error("Failed to invoke start-backend Lambda: %s", exc)
        raise RuntimeError(f"Could not start backend: {exc}") from exc

    deadline = time.time() + COLD_START_TIMEOUT_SECONDS
    while time.time() < deadline:
        time.sleep(COLD_START_POLL_INTERVAL_SECONDS)
        if _backend_is_healthy():
            if progress_callback:
                progress_callback("Backend ready!")
            _logger.info("Backend is now healthy after cold start")
            return
        if progress_callback:
            remaining = int(deadline - time.time())
            progress_callback(f"Waiting for backend... ({remaining}s remaining)")

    raise RuntimeError(
        f"Backend did not become healthy within {COLD_START_TIMEOUT_SECONDS}s after Lambda invocation"
    )


def get_grid_search_progress() -> Dict[str, Any]:
    """Call GET /api/grid-search-progress. Returns dict with stage, result (when done), error (when failed)."""
    url = _api("/grid-search-progress")
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def post_grid_search(
    wavelengths: List[float],
    measured: List[float],
    lipid_range: tuple,
    aqueous_range: tuple,
    roughness_range: tuple,
    top_k: int = 10,
    search_strategy: str = "Dynamic Search",
    max_combinations: int = 30000,
    lipid_file: Optional[str] = None,
    water_file: Optional[str] = None,
    mucus_file: Optional[str] = None,
    substratum_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Call POST /api/grid-search. Backend returns 202 and runs job in background. Returns 202 body (no results)."""
    url = _api("/grid-search")
    payload = {
        "wavelengths": wavelengths,
        "measured": measured,
        "ranges": {
            "lipid": list(lipid_range),
            "aqueous": list(aqueous_range),
            "roughness_angstrom": list(roughness_range),
        },
        "top_k": top_k,
        "enable_roughness": True,
        "search_strategy": search_strategy,
        "max_combinations": max_combinations,
        "lipid_file": lipid_file,
        "water_file": water_file,
        "mucus_file": mucus_file,
        "substratum_file": substratum_file,
    }
    resp = requests.post(url, json=payload, timeout=GRID_SEARCH_START_TIMEOUT_SECONDS)
    if resp.status_code == 202:
        return resp.json()
    resp.raise_for_status()
    data = resp.json()  # legacy 200 with results
    return data


def post_theoretical(
    wavelengths: List[float],
    lipid_nm: float,
    aqueous_nm: float,
    roughness_angstrom: float,
    lipid_file: Optional[str] = None,
    water_file: Optional[str] = None,
    mucus_file: Optional[str] = None,
    substratum_file: Optional[str] = None,
) -> List[float]:
    """Call POST /api/theoretical. Returns spectrum as list."""
    url = _api("/theoretical")
    payload = {
        "wavelengths": wavelengths,
        "lipid_nm": lipid_nm,
        "aqueous_nm": aqueous_nm,
        "roughness_angstrom": roughness_angstrom,
        "enable_roughness": True,
        "lipid_file": lipid_file,
        "water_file": water_file,
        "mucus_file": mucus_file,
        "substratum_file": substratum_file,
    }
    resp = requests.post(url, json=payload, timeout=THEORETICAL_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()["spectrum"]


def post_align_spectra(
    measured: List[float],
    theoretical: List[float],
    focus_min_nm: float = 600.0,
    focus_max_nm: float = 1120.0,
    wavelengths: Optional[List[float]] = None,
) -> List[float]:
    """Call POST /api/align-spectra. Returns aligned theoretical as list."""
    url = _api("/align-spectra")
    payload = {
        "measured": measured,
        "theoretical": theoretical,
        "focus_min_nm": focus_min_nm,
        "focus_max_nm": focus_max_nm,
        "wavelengths": wavelengths,
    }
    resp = requests.post(url, json=payload, timeout=ALIGN_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()["aligned"]


def get_last_two_cycles_progress() -> Dict[str, Any]:
    """Call GET /api/last-two-cycles-progress. Returns dict with run_folder, processed, total, stage."""
    url = _api("/last-two-cycles-progress")
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def post_last_two_cycles_seed_tune(run_folder_name: str) -> Dict[str, Any]:
    """Call POST /api/last-two-cycles-seed-tune. Backend returns 202 and runs job in background. Returns {'job_id': ...} or full result if 200 (legacy)."""
    url = _api("/last-two-cycles-seed-tune")
    resp = requests.post(
        url,
        json={"run_folder_name": run_folder_name},
        timeout=LAST_TWO_CYCLES_START_TIMEOUT_SECONDS,
    )
    if resp.status_code == 202:
        return resp.json()  # {"job_id": ..., "message": ...}
    resp.raise_for_status()
    return resp.json()  # 200: legacy sync response with summary, seed_result, results


def result_dict_to_view(d: Dict[str, Any]) -> Any:
    """Convert API result dict to an object with attribute access and numpy arrays where needed."""
    import numpy as np

    class ResultView:
        pass

    v = ResultView()
    for key, val in d.items():
        if key in ("theoretical_spectrum", "wavelengths") and isinstance(val, list):
            setattr(v, key, np.asarray(val))
        else:
            setattr(v, key, val)
    return v


# ---------------------------------------------------------------------------
# Spectrum source browsing (production: Streamlit calls these when local data is missing)
# ---------------------------------------------------------------------------

def get_spectrum_sources() -> List[Dict[str, Any]]:
    """Fetch available spectrum sources from the backend."""
    url = _api("/spectrum-sources")
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_spectrum_files(source_name: str) -> List[Dict[str, str]]:
    """Fetch spectrum file list for a source from the backend."""
    url = _api("/spectrum-files")
    resp = requests.get(url, params={"source": source_name}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_spectrum_content(source_name: str, filename: str) -> Dict[str, Any]:
    """Fetch parsed spectrum content (wavelengths + measured) from the backend."""
    url = _api("/spectrum-content")
    resp = requests.get(url, params={"source": source_name, "filename": filename}, timeout=30)
    resp.raise_for_status()
    return resp.json()
