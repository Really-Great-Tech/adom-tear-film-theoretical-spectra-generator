"""
PyElli backend API entrypoint. Run with: python server.py or uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""
import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from backend import config
from backend.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ActivityTrackingMiddleware(BaseHTTPMiddleware):
    """Update last-activity timestamp on every inbound request (used for auto-shutdown)."""

    async def dispatch(self, request: Request, call_next):
        config.touch_activity()
        return await call_next(request)


app = FastAPI(
    title="PyElli Backend",
    description="Grid search and theoretical spectrum computation for tear film analysis",
    version="1.0.0",
)
app.add_middleware(ActivityTrackingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api", tags=["pyelli"])


@app.on_event("startup")
def startup_log():
    """Log config and env on startup so deployment vs local is visible."""
    logger.info("PyElli backend starting")
    logger.info("BACKEND_MATERIALS_PATH=%s (exists=%s)", config.MATERIALS_PATH, config.MATERIALS_PATH.exists())
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        val = os.environ.get(name, "<unset>")
        logger.info("ENV %s=%s", name, val)
    if not config.MATERIALS_PATH.exists():
        logger.warning("Materials path does not exist; /api/grid-search and /api/theoretical will return 503 until mounted or BACKEND_MATERIALS_PATH is set correctly")


@app.get("/health")
def health():
    """Liveness/readiness."""
    return {"status": "ok"}
