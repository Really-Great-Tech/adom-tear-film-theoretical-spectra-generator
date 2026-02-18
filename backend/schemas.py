"""
Pydantic schemas for API request/response. Single source of truth for API contracts.
"""
from typing import Any, List, Optional, Tuple

from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Grid search
# -----------------------------------------------------------------------------

class GridRanges(BaseModel):
    """Grid search parameter ranges."""
    lipid: Tuple[float, float, float] = Field(..., description="(min_nm, max_nm, step_nm)")
    aqueous: Tuple[float, float, float] = Field(..., description="(min_nm, max_nm, step_nm)")
    roughness_angstrom: Tuple[float, float, float] = Field(..., description="(min_Å, max_Å, step_Å)")


class GridSearchRequest(BaseModel):
    """Request body for POST /grid-search."""
    wavelengths: List[float] = Field(..., min_length=10, description="Wavelength array (nm)")
    measured: List[float] = Field(..., min_length=10, description="Measured reflectance")
    ranges: GridRanges
    top_k: int = Field(10, ge=1, le=100)
    enable_roughness: bool = True
    search_strategy: str = Field("Dynamic Search", description="Coarse Search | Full Grid Search | Dynamic Search")
    max_combinations: Optional[int] = Field(30000, ge=1, le=100_000)
    lipid_file: Optional[str] = None
    water_file: Optional[str] = None
    mucus_file: Optional[str] = None
    substratum_file: Optional[str] = None


def _serialize_result(r: Any) -> dict:
    """Convert PyElliResult to JSON-serializable dict. Single place for result shape."""
    return {
        "lipid_nm": float(r.lipid_nm),
        "aqueous_nm": float(r.aqueous_nm),
        "mucus_nm": float(r.mucus_nm),
        "score": float(r.score),
        "rmse": float(r.rmse),
        "correlation": float(r.correlation),
        "crossing_count": int(r.crossing_count),
        "matched_peaks": int(r.matched_peaks),
        "peak_count_delta": int(r.peak_count_delta),
        "mean_delta_nm": float(r.mean_delta_nm),
        "oscillation_ratio": float(r.oscillation_ratio),
        "theoretical_peaks": int(r.theoretical_peaks),
        "theoretical_spectrum": r.theoretical_spectrum.tolist() if r.theoretical_spectrum is not None else None,
        "wavelengths": r.wavelengths.tolist() if r.wavelengths is not None else None,
        "peak_drift_slope": float(getattr(r, "peak_drift_slope", 0.0)),
        "peak_drift_r_squared": float(getattr(r, "peak_drift_r_squared", 0.0)),
        "peak_drift_flagged": bool(getattr(r, "peak_drift_flagged", False)),
        "amplitude_drift_slope": float(getattr(r, "amplitude_drift_slope", 0.0)),
        "amplitude_drift_r_squared": float(getattr(r, "amplitude_drift_r_squared", 0.0)),
        "amplitude_drift_flagged": bool(getattr(r, "amplitude_drift_flagged", False)),
        "deviation_vs_measured": float(x) if (x := getattr(r, "deviation_vs_measured", None)) is not None else None,
    }


class GridSearchResponse(BaseModel):
    """Response for POST /grid-search."""
    results: List[dict] = Field(..., description="Top-k results (each with keys matching PyElliResult)")
    count: int = Field(..., description="Number of results returned")

    @classmethod
    def from_results(cls, results: List[Any]) -> "GridSearchResponse":
        return cls(
            results=[_serialize_result(r) for r in results],
            count=len(results),
        )


# -----------------------------------------------------------------------------
# Single theoretical spectrum
# -----------------------------------------------------------------------------

class TheoreticalRequest(BaseModel):
    """Request body for POST /theoretical."""
    wavelengths: List[float] = Field(..., min_length=10)
    lipid_nm: float = Field(..., gt=0)
    aqueous_nm: float = Field(..., gt=0)
    roughness_angstrom: float = Field(..., ge=0, description="Interface roughness in Å (slider value)")
    enable_roughness: bool = True
    lipid_file: Optional[str] = None
    water_file: Optional[str] = None
    mucus_file: Optional[str] = None
    substratum_file: Optional[str] = None


class TheoreticalResponse(BaseModel):
    """Response for POST /theoretical."""
    spectrum: List[float] = Field(..., description="Theoretical reflectance array")


# -----------------------------------------------------------------------------
# Align spectra (for BestFit alignment in UI)
# -----------------------------------------------------------------------------

class AlignSpectraRequest(BaseModel):
    """Request body for POST /align-spectra."""
    measured: List[float] = Field(..., min_length=10)
    theoretical: List[float] = Field(..., min_length=10)
    focus_min_nm: float = 600.0
    focus_max_nm: float = 1120.0
    wavelengths: Optional[List[float]] = None


class AlignSpectraResponse(BaseModel):
    """Response for POST /align-spectra."""
    aligned: List[float] = Field(..., description="Aligned theoretical spectrum")
