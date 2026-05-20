FROM python:3.13-slim

# Install curl for healthcheck (--no-install-recommends for minimal footprint)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution and upgrade pip for security (GHSA-4xh5-x5gv-qwph)
RUN pip install --upgrade "pip>=25.3" && \
    pip install uv

# Copy ALL wheel files (including dependencies)
COPY dist/*.whl /app/
# Copy health check script
COPY check_mcp_health.py /app/
# Copy workflows configuration
COPY workflows/ /app/workflows/

WORKDIR /app

# Install wheels WITHOUT dependencies first, then resolve deps via uv.
# Switched from vanilla pip to uv to match the 3 sibling MCP Dockerfiles
# (kit_mcp, omni_ui_mcp, usd_code_mcp): vanilla pip's backtracking explodes
# when nvidia-nat is loose-pinned (the 1.5.0a20260120 alpha exact-pin used to
# collide with the ragas>=0.3.0rc1 CVE override which transitively required
# ragas~=0.2.14 from the older NAT). uv's resolver picks a compatible newer
# NAT alpha + ragas 0.3.x in seconds rather than hours of backtracking.
RUN uv pip install --system --no-deps isaacsim_fns-*.whl isaacsim_mcp-*.whl && \
    uv pip install --system --prerelease=allow \
    "nvidia-nat>=1.4.0a0,<2.0.0" \
    "nvidia-nat-langchain>=1.4.0a0,<2.0.0" \
    "nvidia-nat-opentelemetry>=1.4.0a0,<2.0.0" \
    "nvidia-nat-mcp>=1.3.1" \
    "langchain>=1.2.7,<2.0.0" \
    "langchain-core>=1.2.22,<2.0.0" \
    "langchain-community==0.4.1" \
    "langchain-nvidia-ai-endpoints==1.0.2" \
    "langchain-tavily==0.2.17" \
    "urllib3==2.6.3" \
    "mcp==1.25.0" \
    "fastapi==0.119.1" \
    "faiss-cpu==1.13.2" \
    "pyyaml==6.0.3" \
    "httpx==0.28.1" \
    "requests==2.32.5" \
    "redis[hiredis]==7.1.0" \
    "authlib>=1.6.9" \
    "cryptography>=46.0.5" \
    "litellm>=1.83.0" \
    "orjson>=3.11.6" \
    "protobuf>=6.33.5" \
    "pyjwt>=2.12.0" \
    "ragas>=0.3.0rc1" && \
    uv pip install --system --reinstall --no-deps "starlette==0.51.0"

# Verify installations
RUN python -c "import isaacsim_fns; print(f'isaacsim_fns version: {isaacsim_fns.__version__}')" && \
    python -c "import isaacsim_mcp; print(f'isaacsim_mcp version: {isaacsim_mcp._get_version()}')"

# Create non-root user for security
RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Set environment variables
ENV PYTHONPATH=/app
# Ensure print() output (including the [mcp-startup] banner before the
# nat-mcp-serve subprocess) appears in `docker logs` in real time
# rather than being block-buffered until process exit.
ENV PYTHONUNBUFFERED=1
ENV ISAACSIM_MCP_DISABLE_USAGE_LOGGING=false
ENV MCP_PORT=9904
ENV MCP_HOST=0.0.0.0
ENV HOME=/app

# Set ownership and permissions
RUN chown -R appuser:appgroup /app \
    && chgrp -R 0 /app \
    && chmod -R g=u /app

EXPOSE 9904
# Create Docker environment marker
RUN touch /.dockerenv

# Health check against the streamable-http /mcp endpoint. The Accept header is
# REQUIRED — NAT 1.4 returns 406 Not Acceptable without it, and the previous
# version of this line (no Accept, trailing slash) was a silent false-positive
# that passed via the 307 redirect curl ate.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fL -X POST http://localhost:9904/mcp \
        -H "Content-Type: application/json" \
        -H "Accept: application/json, text/event-stream" \
        -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"healthcheck","version":"1.0.0"}}}' || exit 1

# Switch to non-root user
USER appuser

# Set the command to run the application
# Using the console script defined in pyproject.toml
CMD ["isaacsim-mcp"]
