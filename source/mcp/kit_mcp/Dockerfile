FROM python:3.13-alpine3.23

# Patch base-image packages before installing anything else. python:alpine
# ships whatever apk had when the base was cut, so CVE fixes Alpine has already
# published (e.g. libuuid 2.41.6-r1 for CVE-2026-76642/78408/78410) do not
# reach us until the base is rebuilt upstream. Upgrading here closes that gap.
RUN apk upgrade --no-cache

# Install the health-check client and native runtimes used by binary wheels.
RUN apk add --no-cache curl libstdc++ libgomp

# Install uv for fast dependency resolution and upgrade pip for security
RUN pip install --upgrade "pip>=26.2" && \
    pip install uv

# Copy ALL wheel files (including dependencies)
COPY dist/*.whl /app/
# Copy health check script
COPY check_mcp_health.py /app/
# Copy workflows configuration
COPY workflows/ /app/workflows/

WORKDIR /app

# Install wheels WITHOUT dependencies first, then install all deps
# Use uv for resolution to avoid pip's resolution-too-deep errors with loose NAT constraints
RUN uv pip install --system --no-deps kit_fns-*.whl kit_mcp-*.whl ovgenai_retrieval-*.whl && \
    uv pip install --system --prerelease=if-necessary-or-explicit \
    "nvidia-nat==1.8.0a20260514" \
    "nvidia-nat-langchain==1.8.0a20260514" \
    "nvidia-nat-opentelemetry==1.8.0a20260514" \
    "nvidia-nat-mcp==1.8.0a20260514" \
    "langchain>=1.3.9" \
    "langchain-core>=1.4.7,<2" \
    "langchain-openai>=1.1.14" \
    "langsmith>=0.8.18" \
    "langchain-community>=0.4.1" \
    "langchain-classic==1.0.7" \
    "langchain-text-splitters>=0.3.9" \
    "langchain-nvidia-ai-endpoints>=1.4.3,<2" \
    "urllib3==2.7.0" \
    "yarl<1.24" \
    "langgraph>=1.0.10" \
    "marshmallow>=3.26.2" \
    "mcp>=1.28.1,<2" \
    "python-multipart==0.0.31" \
    "fastapi>=0.136.1" \
    "faiss-cpu>=1.9.0.post1" \
    "rank_bm25>=0.2.2" \
    "pyyaml>=6.0" \
    "httpx>=0.24.0" \
    "requests>=2.33.0" \
    "redis[hiredis]>=4.5.0" \
    "aiohttp-sse==2.2.0" \
    "aiohttp>=3.14.3,<4" \
    "starlette>=1.3.1,<2.0" \
    "authlib>=1.7.1" \
    "click>=8.3.3,<9" \
    "cryptography>=50.0.0,<51" \
    "idna>=3.15" \
    "joserfc>=1.6.8,<2" \
    "litellm==1.84.0" \
    "Mako==1.3.12" \
    "orjson>=3.11.6" \
    "protobuf>=6.33.5" \
    "pyarrow==23.0.1" \
    "pydantic-settings>=2.14.2" \
    "pygments>=2.20.0" \
    "pyjwt>=2.13.0" \
    "python-dotenv>=1.2.2"

# RAGAS, diskcache, and Pillow are not used by this MCP server. Fail if any
# return transitively; RAGAS and diskcache have no safe release for their findings.
RUN uv pip check --system && \
    python -c "import importlib.util; assert importlib.util.find_spec('ragas') is None; assert importlib.util.find_spec('diskcache') is None; assert importlib.util.find_spec('PIL') is None" && \
    rm -rf /root/.cache /app/.cache \
        /usr/local/lib/python3.13/site-packages/pip \
        /usr/local/lib/python3.13/site-packages/pip-*.dist-info \
        /usr/local/lib/python3.13/site-packages/uv-*.dist-info && \
    rm -f /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.13 \
        /usr/local/bin/uv /usr/local/bin/uvx

# Verify installations
RUN python -c "import kit_fns; print(f'kit_fns version: {kit_fns.__version__}')" && \
    python -c "import kit_mcp; print(f'kit_mcp version: {kit_mcp._get_version()}')" && \
    python -c "import ovgenai_retrieval; print(f'ovgenai_retrieval version: {ovgenai_retrieval.__version__}')"

# Create non-root user for security
RUN addgroup -g 1000 -S appgroup \
    && adduser -u 1000 -S -D -H -s /sbin/nologin -G appgroup appuser

# Set environment variables
ENV PYTHONPATH=/app
ENV KIT_MCP_DISABLE_USAGE_LOGGING=false
ENV MCP_PORT=9902
ENV MCP_HOST=0.0.0.0
ENV OVAI_RERANK_BACKEND=nvidia
ENV OVAI_RERANK_MODEL=nvidia/llama-nemotron-rerank-vl-1b-v2
ENV HOME=/app

# Set ownership and permissions
RUN chown -R appuser:appgroup /app \
    && chgrp -R 0 /app \
    && chmod -R g=u /app

EXPOSE 9902
# Create Docker environment marker
RUN touch /.dockerenv

# Health check against the streamable-http /mcp endpoint. The Accept header is
# REQUIRED — NAT 1.4 returns ``406 Not Acceptable`` without it, and the
# previous version of this line (no Accept, trailing slash) could pass the
# redirect check while missing the actual streamable-http requirement.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fL -X POST http://localhost:9902/mcp \
        -H "Content-Type: application/json" \
        -H "Accept: application/json, text/event-stream" \
        -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"healthcheck","version":"1.0.0"}}}' || exit 1

# Switch to non-root user
USER appuser

# Set the command to run the application
# Using the console script defined in pyproject.toml
CMD ["kit-mcp"]
