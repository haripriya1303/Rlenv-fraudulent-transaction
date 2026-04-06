# Start from openenv base image (or python slim for standalone builds)
FROM python:3.11-slim

LABEL maintainer="openenv-fraud"
LABEL description="Fraud Detection & Transaction Risk Review OpenEnv Environment"
LABEL org.opencontainers.image.source="https://github.com/openenv/fraud-env"
LABEL space_tag="openenv"

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    FRAUD_TASK=medium \
    LOG_DIR=/app/outputs/logs \
    ENABLE_WEB_INTERFACE=true \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install openenv-core first (pinned for reproducibility)
RUN pip install --no-cache-dir openenv-core==0.2.1

# Copy and install package requirements
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy all project files
COPY . /app/

# Create output directories
RUN mkdir -p /app/outputs/logs /app/outputs/evals

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
