#!/usr/bin/env python3
"""Start the PyElli backend API. Run from project root: python server.py"""
import sys
from pathlib import Path

# Add project root so backend and exploration are importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
