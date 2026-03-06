# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=../mlflow/mlflow.db

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api/main.py .
COPY api/feature_columns.json .
COPY api/median_values.json .
COPY mlruns/ ./mlruns/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]