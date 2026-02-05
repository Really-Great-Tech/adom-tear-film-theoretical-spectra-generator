"""
Shlomo ground truth loader for More Good Spectras.

Provides Shlomo's lipid and aqueous (Mucus-Aqueous_Height) values from
exploration/more_good_spectras/shlomo_ground_truth.csv, which is derived from
Lipid and Mucus-Aqueous_Height.xlsx. Used to:
- Compute loss at Shlomo's values (vs at our best-fit) for deviation analysis.
- Report lipid and aqueous error vs Shlomo: |ours - shlomo| per parameter.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Path to CSV (same repo: exploration/more_good_spectras/shlomo_ground_truth.csv)
_THIS_DIR = Path(__file__).resolve().parent
_MORE_GOOD_DIR = _THIS_DIR.parent / "more_good_spectras"
_CSV_PATH = _MORE_GOOD_DIR / "shlomo_ground_truth.csv"

_df_cache: Optional[pd.DataFrame] = None


def _load_df() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        if not _CSV_PATH.exists():
            return pd.DataFrame()
        _df_cache = pd.read_csv(_CSV_PATH)
    return _df_cache


def _normalize_spectrum_key(name_or_path: str) -> str:
    """Normalize to spectrum base name for lookup: (Run)spectra_15-13-12-259 or .txt path."""
    s = str(name_or_path).strip()
    # Path: .../Corrected_Spectra/(Run)spectra_15-13-12-259.txt -> (Run)spectra_15-13-12-259
    if s.endswith(".txt"):
        s = s[:-4]
    if "_BestFit" in s:
        s = s.replace("_BestFit", "")
    return s


def get_shlomo_params(spectrum_name_or_path: str) -> Optional[Dict[str, float]]:
    """
    Return Shlomo's lipid and aqueous for a spectrum in More Good Spectras.

    Args:
        spectrum_name_or_path: Spectrum filename (e.g. "(Run)spectra_15-13-12-259.txt")
            or path, or base name "(Run)spectra_15-13-12-259".

    Returns:
        Dict with keys: lipid_shlomo_nm, aqueous_shlomo_nm, absolute_time;
        or None if not found.
    """
    df = _load_df()
    if df.empty:
        return None
    key = _normalize_spectrum_key(spectrum_name_or_path)
    row = df.loc[df["spectrum_base"] == key]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "lipid_shlomo_nm": float(r["lipid_shlomo_nm"]),
        "aqueous_shlomo_nm": float(r["aqueous_shlomo_nm"]),
        "absolute_time": str(r["absolute_time"]),
    }


def get_all_shlomo_ground_truth() -> pd.DataFrame:
    """Return full table of Shlomo ground truth (spectrum_base, lipid_shlomo_nm, aqueous_shlomo_nm)."""
    return _load_df().copy()


def csv_path() -> Path:
    """Return path to shlomo_ground_truth.csv for reference."""
    return _CSV_PATH
