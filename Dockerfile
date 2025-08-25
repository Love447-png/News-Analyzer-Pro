# Multi-stage build for production
FROM node:16-alpine AS frontend-build

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM python:3.9-slim AS backend

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY python/ ./python/
COPY --from=frontend-build /app/dist ./static/
COPY index.html ./templates/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Expose port
EXPOSE 5000

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "python.app:app"]
