# Electricity Price Forecast - Production Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install package and dependencies
RUN pip install --no-cache-dir -e .

# Copy application code
COPY src/ src/
COPY configs/ configs/

# Default: run API
EXPOSE 8000
CMD ["uvicorn", "electricity_forecast.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
