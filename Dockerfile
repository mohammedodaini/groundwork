# syntax=docker/dockerfile:1

# Multi-stage: build wheels once, ship a slim runtime.
FROM python:3.12-slim AS builder
WORKDIR /build
RUN pip install --no-cache-dir hatchling
COPY pyproject.toml README.md requirements.lock ./
COPY src ./src
# Constrained by requirements.lock, so an image rebuilt from the same commit gets
# the same dependency versions. The extras are what the image ships with:
# Postgres support and the default LLM provider.
RUN pip wheel --no-cache-dir --wheel-dir /wheels -c requirements.lock ".[anthropic,postgres]"

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

# SQLite is the default database. /app belongs to root and the app runs as an
# unprivileged user, so the relative default (./groundwork.db) could not be
# created and the container exited at startup. Give the app one directory it
# owns. Mount a volume there to keep data across containers; compose sets
# DATABASE_URL to Postgres, which overrides this.
RUN mkdir -p /app/data && chown groundwork:groundwork /app/data
ENV DATABASE_URL=sqlite+aiosqlite:////app/data/groundwork.db

USER groundwork
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "groundwork.main:app", "--host", "0.0.0.0", "--port", "8000"]
