# PyElli Exploration Streamlit App
# Multi-stage build for optimized image size

# =============================================================================
# Stage 1: Builder - Install dependencies with build tools
# =============================================================================
FROM python:3.11-slim AS builder

# Install build dependencies for scipy/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies (excluding pythonnet)
COPY requirements.txt /tmp/requirements.txt

# Filter out pythonnet from requirements and install
RUN grep -v 'pythonnet' /tmp/requirements.txt > /tmp/requirements_filtered.txt && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements_filtered.txt

# =============================================================================
# Stage 2: Runtime - Slim final image
# =============================================================================
FROM python:3.11-slim AS runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create directory structure for volume mounts
# All data directories the app expects:
#   - data/Materials: material optical properties (CSV files)
#   - exploration/sample_data: good_fit and bad_fit samples
#   - exploration/new_spectra: new spectra to analyze
#   - exploration/spectra_from_shlomo: raw spectra from Shlomo
#   - exploration/measurements: exploration measurements
RUN mkdir -p /app/data/Materials \
    /app/exploration/sample_data \
    /app/exploration/new_spectra \
    /app/exploration/spectra_from_shlomo \
    /app/exploration/measurements

# Copy application code
COPY .streamlit/ /app/.streamlit/
COPY src/ /app/src/
COPY exploration/pyelli_exploration/ /app/exploration/pyelli_exploration/

# Set Python path for imports
ENV PYTHONPATH="/app:${PYTHONPATH}"

# =============================================================================
# BLAS/OpenMP Thread Limiting (CRITICAL for multiprocessing performance)
# =============================================================================
# These MUST be set BEFORE Python starts to prevent thread oversubscription
# when using ProcessPoolExecutor. Without these, each worker process spawns
# multiple BLAS threads (e.g., 32 cores × 8 threads = 256 threads competing).
# Setting to 1 ensures each Python process uses single-threaded BLAS,
# allowing ProcessPoolExecutor to efficiently parallelize at the process level.
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1

# Streamlit configuration
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the app
ENTRYPOINT ["streamlit", "run", "/app/exploration/pyelli_exploration/app.py"]
