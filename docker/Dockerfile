FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    QE_ARTIFACTS_DIR=/app/artifacts

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements/ requirements/
RUN pip install --upgrade pip \
    && pip install -r requirements/base.txt -r requirements/api.txt -r requirements/quantum.txt

COPY pyproject.toml README.md ./
COPY src/ src/
COPY classical-baseline/README.md classical-baseline/README.md

CMD ["python", "-m", "quantumedge.pipelines.classical_baselines"]
