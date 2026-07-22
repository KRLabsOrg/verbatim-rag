FROM python:3.11-slim

WORKDIR /app

# build-essential is only needed for compilation; purge it in the same layer.
# Pre-install CPU-only torch + torchvision from the same index so their
# compiled operators match (mismatched builds break torchvision::nms, which
# in turn breaks the transformers/sentence-transformers import chain).
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY packages/core packages/core
RUN pip install --no-cache-dir ./packages/core

COPY pyproject.toml README.md ./
COPY verbatim_rag verbatim_rag
RUN pip install --no-cache-dir .

COPY api api

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
