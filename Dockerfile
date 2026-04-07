# ============================================================
# Dockerfile — Email Triage OpenEnv v2
# ============================================================
# Build:  docker build -t email-triage-env .
# Run:    docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... email-triage-env
# ============================================================

FROM python:3.11-slim

# Metadata — FIX #2: version unified to 2.0.0
LABEL maintainer="email-triage-env"
LABEL version="2.0.0"
LABEL description="Email Triage OpenEnv v2 — Real-world AI email triage environment"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root user (required for HF Spaces)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
