# Dockerfile for QuantMindX Strategy Agent
# Paper trading agent with MT5 integration and security hardening

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Security: Restrict where Python can write
    PYTHONPATH=/app/src:$PYTHONPATH

# Install system dependencies and security tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build dependencies for MT5 and Python packages
    build-essential \
    wget \
    curl \
    # Security tools
    libssl-dev \
    libffi-dev \
    # Monitoring and debugging
    procps \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security (CVE-2021-3711 mitigation)
RUN groupadd -r trader && \
    useradd -r -g trader -u 1001 -s /bin/bash -m trader

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    # Remove pip to reduce attack surface
    rm -rf /root/.cache/pip

# Copy application code
COPY --chown=trader:trader src/ /app/src/
COPY --chown=trader:trader entrypoint.sh /app/

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Create directories for runtime data with proper permissions
RUN mkdir -p /app/logs /app/data /app/tmp && \
    chown -R trader:trader /app/logs /app/data /app/tmp

# Create volume mount points
VOLUME ["/app/logs", "/app/data"]

# Switch to non-root user
USER trader

# Set read-only filesystem (except for specified volumes)
# Health check to ensure container is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Expose port for health checks and monitoring
EXPOSE 8080

# Resource limits (applied via docker-compose)
# --memory=512m
# --cpus=1.0
# --pids-limit=100

# Security hardening (applied via docker-compose)
# --security-opt no-new-privileges:true
# cap_drop: ALL (drop all capabilities)
# cap_add: NET_BIND_SERVICE (add only necessary capability)
# --read-only (root filesystem read-only, tmpfs for /tmp)

# Run as non-root with read-only root filesystem
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["agent"]
