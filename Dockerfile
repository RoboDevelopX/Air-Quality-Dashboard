# Small runtime image: the app only needs runtime deps and the committed model artifact.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AQD_CACHE_PATH=/tmp/aqd/cache.sqlite3 \
    AQD_MODEL_DIR=/app/models

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so this layer is cached across code changes.
COPY requirements.lock .
RUN pip install -r requirements.lock

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

COPY app.py ./
COPY .streamlit ./.streamlit
COPY models ./models

RUN useradd --create-home --uid 10001 app && mkdir -p /tmp/aqd && chown app /tmp/aqd
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
