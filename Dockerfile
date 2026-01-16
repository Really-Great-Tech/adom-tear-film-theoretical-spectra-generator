FROM python:3.13-slim

WORKDIR /app

# Install curl (for healthcheck) and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install dependencies globally
COPY requirements-linux.txt .
RUN pip install --no-cache-dir -r requirements-linux.txt

# Copy app code
COPY . .

# Create non-root user and fix permissions
RUN useradd -m appuser && chown -R appuser:appuser /app

USER appuser

# Configuration
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "exploration/pyelli_exploration/app.py"]
