"""
HTTP client for PyElli backend API. The Streamlit app requires the backend at all times.
Single responsibility: talk to backend; no business logic.
Loads BACKEND_URL from .env in project root if set.
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# Load .env from project root so BACKEND_URL is set without exporting manually
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")

# Default timeout for grid search (long-running)
GRID_SEARCH_TIMEOUT_SECONDS = 600
# Last-two-cycles seed+tune can take many minutes
LAST_TWO_CYCLES_TIMEOUT_SECONDS = 3600
THEORETICAL_TIMEOUT_SECONDS = 30
ALIGN_TIMEOUT_SECONDS = 10


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
) -> List[Dict[str, Any]]:
    """Call POST /api/grid-search. Returns list of result dicts."""
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
    resp = requests.post(url, json=payload, timeout=GRID_SEARCH_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()
    return data["results"]


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


def post_last_two_cycles_seed_tune(run_folder_name: str) -> Dict[str, Any]:
    """Call POST /api/last-two-cycles-seed-tune. Runs grid search on backend. Returns dict with summary, seed_result, results."""
    url = _api("/last-two-cycles-seed-tune")
    resp = requests.post(
        url,
        json={"run_folder_name": run_folder_name},
        timeout=LAST_TWO_CYCLES_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()


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
