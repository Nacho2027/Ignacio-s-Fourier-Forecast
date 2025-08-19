# Multi-stage Dockerfile for Ignacio's Fourier Forecast Newsletter Service
# Optimized for production deployment with proper timezone and security configuration

# Build stage: Install dependencies and prepare application
FROM python:3.11-slim as builder

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage: Minimal runtime image
FROM python:3.11-slim as production

# Install runtime dependencies and timezone data
RUN apt-get update && apt-get install -y \
    sqlite3 \
    tzdata \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set timezone to Eastern Time for proper scheduling
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create non-root user for security
RUN groupadd -r newsletter && useradd -r -g newsletter -d /app -s /bin/bash newsletter

# Set up application directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Copy application code
COPY src/ ./src/
COPY templates/ ./templates/
COPY config/ ./config/
COPY run ./run

# Create directories for persistent data with proper permissions
RUN mkdir -p data logs deployment_reports && \
    chown -R newsletter:newsletter /app

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch to non-root user
USER newsletter

# Expose health check port (optional - for future HTTP health endpoint)
EXPOSE 8080

# Health check to ensure service is running properly
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python src/main.py --health || exit 1

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command: run the newsletter service in continuous mode
CMD ["python", "src/main.py"]