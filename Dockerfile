# Dockerfile for Aqua-Swachh Web Dashboard & ML API
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Generate dataset and train model if not pre-built
RUN python data/generate_dataset.py --samples 1500
RUN python -m model.train

# Expose port (default 8000)
EXPOSE 8000

# Start Uvicorn production server
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
