FROM python:3.13-slim

# Install curl for healthcheck (--no-install-recommends for minimal footprint)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution and upgrade pip for security
RUN pip install --upgrade "pip>=25.3" && \
    pip install uv

# Copy ALL wheel files (including dependencies)
COPY dist/*.whl /app/
# Copy the workflow folder recursively from the source directory
COPY workflow/ /app/workflow/
# Copy health check script
COPY check_mcp_health.py /app/

WORKDIR /app

# Install wheels WITHOUT dependencies first, then install all deps
# Use uv for resolution to avoid pip's resolution-too-deep errors with loose NAT constraints
RUN uv pip install --system --no-deps usd_code_aiq-*.whl usd_code_mcp-*.whl && \
    uv pip install --system --prerelease=allow \
    "nvidia-nat>=1.4.0a0,<2.0.0" \
    "nvidia-nat-langchain>=1.4.0a0,<2.0.0" \
    "nvidia-nat-opentelemetry>=1.4.0a0,<2.0.0" \
    "nvidia-nat-mcp>=1.3.1" \
    "langchain>=1.2.4" \
    "langchain-core>=1.2.22" \
    "langchain-community>=0.4.1" \
    "urllib3>=2.6.3" \
    "mcp>=1.25.0" \
    "fastapi>=0.119.0" \
    "faiss-cpu>=1.9.0.post1" \
    "pyyaml>=6.0" \
    "aiohttp-sse==2.2.0" \
    "starlette>=0.49.1,<0.51.0" \
    "authlib>=1.6.9" \
    "cryptography>=46.0.5" \
    "litellm>=1.83.0" \
    "orjson>=3.11.6" \
    "protobuf>=6.33.5" \
    "pyjwt>=2.12.0" \
    "ragas>=0.3.0rc1"

# Verify installations
RUN python -c "import omni_aiq_usd_code; print(f'omni_aiq_usd_code version: {omni_aiq_usd_code.__version__}')" && \
    python -c "import usd_code_mcp; print(f'usd_code_mcp version: {usd_code_mcp.__version__}')"

# Create non-root user for security
RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Set environment variables
ENV PYTHONPATH=/app
# Ensure print() output (including the [mcp-startup] banner before the
# nat-mcp-serve subprocess) appears in `docker logs` in real time
# rather than being block-buffered until process exit.
ENV PYTHONUNBUFFERED=1
ENV MCP_PORT=9903
ENV MCP_HOST=0.0.0.0
ENV HOME=/app

# Set ownership and permissions
RUN chown -R appuser:appgroup /app \
    && chgrp -R 0 /app \
    && chmod -R g=u /app

EXPOSE 9903
# Create Docker environment marker
RUN touch /.dockerenv

# Health check against the streamable-http /mcp endpoint. The Accept header is
# REQUIRED — NAT 1.4 returns 406 Not Acceptable without it, and the previous
# version of this line (no Accept, trailing slash) was a silent false-positive
# that passed via the 307 redirect curl ate.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fL -X POST http://localhost:9903/mcp \
        -H "Content-Type: application/json" \
        -H "Accept: application/json, text/event-stream" \
        -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"healthcheck","version":"1.0.0"}}}' || exit 1

# Switch to non-root user
USER appuser

# Set the command to run the application
# Using the console script defined in pyproject.toml
CMD ["usd-code-mcp"]
