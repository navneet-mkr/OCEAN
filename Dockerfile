FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data experiments logs

# Set environment variables
ENV OCEAN_DATA_DIR=/app/data \
    OCEAN_OUTPUT_DIR=/app/experiments \
    OCEAN_LOG_DIR=/app/logs \
    OCEAN_DEVICE=cuda \
    PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 8000

# Run with gunicorn
CMD ["gunicorn", "ocean.api.service:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"] 