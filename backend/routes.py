"""
FastAPI route handlers. HTTP only; delegate to service layer.
"""
import logging
import threading
import time

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from backend import config
from backend.schemas import (
    AlignSpectraRequest,
    AlignSpectraResponse,
    GridSearchRequest,
    GridSearchResponse,
    LastTwoCyclesRequest,
    LastTwoCyclesResponse,
    TheoreticalRequest,
    TheoreticalResponse,
)
from backend.service import (
    align_spectra,
    calculate_theoretical,
    get_grid_search_progress,
    get_last_two_cycles_progress,
    list_spectrum_files,
    list_spectrum_sources,
    read_spectrum_content,
    run_grid_search,
    run_last_two_cycles_seed_tune,
    set_grid_search_failed,
    set_grid_search_result,
    set_grid_search_running,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/grid-search-progress")
def get_grid_search_progress_route():
    """Return current grid search progress (stage, result when done, error when failed)."""
    return get_grid_search_progress()


@router.post("/grid-search")
def post_grid_search(body: GridSearchRequest):
    """Start grid search in background; return 202 Accepted. Poll GET /api/grid-search-progress for result."""
    logger.info(
        "grid-search start (async) strategy=%s top_k=%s max_combinations=%s ranges=lipid=%s aqueous=%s roughness_angstrom=%s",
        body.search_strategy,
        body.top_k,
        body.max_combinations,
        body.ranges.lipid,
        body.ranges.aqueous,
        body.ranges.roughness_angstrom,
    )

    def _run():
        try:
            set_grid_search_running()
            results = run_grid_search(
                materials_path=config.MATERIALS_PATH,
                wavelengths=body.wavelengths,
                measured=body.measured,
                lipid_range=tuple(body.ranges.lipid),
                aqueous_range=tuple(body.ranges.aqueous),
                roughness_range=tuple(body.ranges.roughness_angstrom),
                top_k=body.top_k,
                enable_roughness=body.enable_roughness,
                search_strategy=body.search_strategy,
                max_combinations=body.max_combinations,
                lipid_file=body.lipid_file,
                water_file=body.water_file,
                mucus_file=body.mucus_file,
                substratum_file=body.substratum_file,
            )
            set_grid_search_result(results)
            logger.info("grid-search done count=%s", len(results))
        except FileNotFoundError as e:
            set_grid_search_failed(str(e))
            logger.warning("grid-search FileNotFoundError: %s", e)
        except ValueError as e:
            set_grid_search_failed(str(e))
            logger.warning("grid-search ValueError: %s", e)
        except Exception as e:
            set_grid_search_failed(str(e))
            logger.exception("grid-search failed: %s", e)

    thread = threading.Thread(target=_run, daemon=False)
    thread.start()
    return JSONResponse(
        status_code=202,
        content={"message": "Grid search started. Poll GET /api/grid-search-progress for result."},
    )


@router.post("/theoretical", response_model=TheoreticalResponse)
def post_theoretical(body: TheoreticalRequest):
    """Compute single theoretical spectrum for given parameters."""
    logger.info(
        "theoretical start lipid_nm=%.1f aqueous_nm=%.1f roughness_angstrom=%.0f",
        body.lipid_nm,
        body.aqueous_nm,
        body.roughness_angstrom,
    )
    t0 = time.perf_counter()
    try:
        spectrum = calculate_theoretical(
            materials_path=config.MATERIALS_PATH,
            wavelengths=body.wavelengths,
            lipid_nm=body.lipid_nm,
            aqueous_nm=body.aqueous_nm,
            roughness_angstrom=body.roughness_angstrom,
            enable_roughness=body.enable_roughness,
            lipid_file=body.lipid_file,
            water_file=body.water_file,
            mucus_file=body.mucus_file,
            substratum_file=body.substratum_file,
        )
        elapsed = time.perf_counter() - t0
        logger.info("theoretical done len=%s elapsed_sec=%.3f", len(spectrum), elapsed)
        return TheoreticalResponse(spectrum=spectrum)
    except FileNotFoundError as e:
        logger.warning("theoretical FileNotFoundError: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        logger.warning("theoretical ValueError: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/align-spectra", response_model=AlignSpectraResponse)
def post_align_spectra(body: AlignSpectraRequest):
    """Align theoretical spectrum to measured (linear regression)."""
    try:
        aligned = align_spectra(
            measured=body.measured,
            theoretical=body.theoretical,
            focus_min_nm=body.focus_min_nm,
            focus_max_nm=body.focus_max_nm,
            wavelengths=body.wavelengths,
        )
        logger.info("align-spectra done len=%s", len(aligned))
        return AlignSpectraResponse(aligned=aligned)
    except ValueError as e:
        logger.warning("align-spectra ValueError: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/last-two-cycles-progress")
def get_last_two_cycles_progress_route():
    """Return current progress of last-two-cycles full test (run_folder, processed, total, stage)."""
    return get_last_two_cycles_progress()


@router.post("/last-two-cycles-seed-tune")
def post_last_two_cycles_seed_tune(body: LastTwoCyclesRequest):
    """Start last-two-cycles seed+tune in background; return 202 Accepted. Poll GET /last-two-cycles-progress for progress and result."""
    logger.info("last-two-cycles-seed-tune start run_folder=%s (async)", body.run_folder_name)

    def _run():
        try:
            run_last_two_cycles_seed_tune(
                run_folder_name=body.run_folder_name,
                full_test_cycles_dir=config.FULL_TEST_CYCLES_DIR,
                materials_path=config.MATERIALS_PATH,
            )
        except FileNotFoundError:
            pass  # service already set stage=failed and error message
        except Exception:
            pass  # service already set stage=failed and error message

    thread = threading.Thread(target=_run, daemon=False)
    thread.start()
    return JSONResponse(
        status_code=202,
        content={
            "job_id": body.run_folder_name,
            "message": "Job started. Poll GET /api/last-two-cycles-progress for progress; when stage=done, result is in response.",
        },
    )


# ---------------------------------------------------------------------------
# Auto-scaling: last-activity
# ---------------------------------------------------------------------------

@router.get("/last-activity")
def get_last_activity():
    """Return epoch timestamp of the last inbound request. Used by Lambda/EventBridge to decide auto-shutdown."""
    return {"last_activity": config.get_last_activity()}


# ---------------------------------------------------------------------------
# Spectrum source browsing (production fallback when Streamlit has no local data)
# ---------------------------------------------------------------------------

@router.get("/spectrum-sources")
def get_spectrum_sources():
    """Return available spectrum sources that exist on the backend filesystem."""
    return list_spectrum_sources()


@router.get("/spectrum-files")
def get_spectrum_files(source: str = Query(..., description="Source name from /spectrum-sources")):
    """Return list of spectrum file names for a given source."""
    try:
        return list_spectrum_files(source)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/spectrum-content")
def get_spectrum_content(
    source: str = Query(..., description="Source name"),
    filename: str = Query(..., description="Spectrum filename"),
):
    """Return parsed spectrum content (wavelengths + measured) for a file."""
    try:
        return read_spectrum_content(source, filename)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
