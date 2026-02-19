"""Backend config: single source for paths and constants."""
import logging
import os
from pathlib import Path

# Materials path: env override, or project-relative (works local and in Docker: /app/data/Materials)
_here = Path(__file__).resolve().parent
_PROJECT_ROOT = _here.parent
_env_path = os.getenv("BACKEND_MATERIALS_PATH")
MATERIALS_PATH = Path(_env_path) if _env_path else (_PROJECT_ROOT / "data" / "Materials")

# Full test cycles: directory containing run folders (e.g. "Full test - 0007_2025-12-30_15-12-20")
_env_full_test = os.getenv("BACKEND_FULL_TEST_CYCLES_DIR")
FULL_TEST_CYCLES_DIR = Path(_env_full_test) if _env_full_test else (_PROJECT_ROOT / "exploration" / "full_test_cycles")

_logger = logging.getLogger(__name__)
if not MATERIALS_PATH.exists():
    _logger.warning("Materials path not found: %s (set BACKEND_MATERIALS_PATH or mount data/Materials)", MATERIALS_PATH)
