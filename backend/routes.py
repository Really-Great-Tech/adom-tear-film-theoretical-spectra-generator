"""
FastAPI route handlers. HTTP only; delegate to service layer.
"""
import logging
import time

from fastapi import APIRouter, HTTPException

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
    run_grid_search,
    run_last_two_cycles_seed_tune,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/grid-search", response_model=GridSearchResponse)
def post_grid_search(body: GridSearchRequest):
    """Run grid search; returns top-k results sorted by combined score."""
    logger.info(
        "grid-search start strategy=%s top_k=%s max_combinations=%s ranges=lipid=%s aqueous=%s roughness_angstrom=%s",
        body.search_strategy,
        body.top_k,
        body.max_combinations,
        body.ranges.lipid,
        body.ranges.aqueous,
        body.ranges.roughness_angstrom,
    )
    t0 = time.perf_counter()
    try:
        results = run_grid_search(
            materials_path=config.MATERIALS_PATH,
            wavelengths=body.wavelengths,
            measured=body.measured,
            lipid_range=body.ranges.lipid,
            aqueous_range=body.ranges.aqueous,
            roughness_range=body.ranges.roughness_angstrom,
            top_k=body.top_k,
            enable_roughness=body.enable_roughness,
            search_strategy=body.search_strategy,
            max_combinations=body.max_combinations,
            lipid_file=body.lipid_file,
            water_file=body.water_file,
            mucus_file=body.mucus_file,
            substratum_file=body.substratum_file,
        )
        elapsed = time.perf_counter() - t0
        logger.info("grid-search done count=%s elapsed_sec=%.2f", len(results), elapsed)
        return GridSearchResponse.from_results(results)
    except FileNotFoundError as e:
        logger.warning("grid-search FileNotFoundError: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        logger.warning("grid-search ValueError: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("grid-search failed: %s", e)
        raise


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


@router.post("/last-two-cycles-seed-tune", response_model=LastTwoCyclesResponse)
def post_last_two_cycles_seed_tune(body: LastTwoCyclesRequest):
    """Run last-two-cycles seed+tune (full grid on seed, refine on rest); grid search runs on backend."""
    logger.info("last-two-cycles-seed-tune start run_folder=%s", body.run_folder_name)
    t0 = time.perf_counter()
    try:
        out = run_last_two_cycles_seed_tune(
            run_folder_name=body.run_folder_name,
            full_test_cycles_dir=config.FULL_TEST_CYCLES_DIR,
            materials_path=config.MATERIALS_PATH,
        )
        elapsed = time.perf_counter() - t0
        logger.info("last-two-cycles-seed-tune done elapsed_sec=%.2f", elapsed)
        return LastTwoCyclesResponse(
            summary=out["summary"],
            seed_result=out.get("seed_result"),
            results=out.get("results", []),
        )
    except FileNotFoundError as e:
        logger.warning("last-two-cycles-seed-tune FileNotFoundError: %s", e)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("last-two-cycles-seed-tune failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
