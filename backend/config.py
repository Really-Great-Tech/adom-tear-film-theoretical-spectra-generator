"""Backend config: single source for paths and constants."""
import logging
import os
from pathlib import Path

# Materials path: env override, or project-relative (works local and in Docker: /app/data/Materials)
_here = Path(__file__).resolve().parent
_PROJECT_ROOT = _here.parent
_env_path = os.getenv("BACKEND_MATERIALS_PATH")
MATERIALS_PATH = Path(_env_path) if _env_path else (_PROJECT_ROOT / "data" / "Materials")

_logger = logging.getLogger(__name__)
if not MATERIALS_PATH.exists():
    _logger.warning("Materials path not found: %s (set BACKEND_MATERIALS_PATH or mount data/Materials)", MATERIALS_PATH)
