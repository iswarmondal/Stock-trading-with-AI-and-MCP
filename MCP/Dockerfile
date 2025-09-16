# syntax=docker/dockerfile:1.7
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (curl for healthcheck/debug)
RUN apt-get update -y && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md /app/
COPY stock_mcp_server /app/stock_mcp_server

# Install
RUN pip install --upgrade pip && pip install .

# Env configuration (overridable by Coolify)
ENV FASTMCP_HOST=0.0.0.0 \
    FASTMCP_PORT=8765

# Expose SSE port
EXPOSE 8765

# Run via mcp (SSE transport)
CMD ["mcp", "run", "stock_mcp_server/server/app.py:app", "--transport", "sse"]
