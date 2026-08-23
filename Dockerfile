# syntax=docker/dockerfile:1

# Multi-stage: build wheels once, ship a slim runtime.
FROM python:3.12-slim AS builder
WORKDIR /build
RUN pip install --no-cache-dir hatchling
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --no-cache-dir --wheel-dir /wheels . \
 && pip wheel --no-cache-dir --wheel-dir /wheels asyncpg anthropic

FROM python:3.12-slim AS runtime

# Run as a non-root user. An agent that fetches arbitrary URLs should have the
# least privilege we can give it.
RUN useradd --create-home --uid 10001 groundwork
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

COPY static ./static
COPY src ./src

USER groundwork
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "groundwork.main:app", "--host", "0.0.0.0", "--port", "8000"]
