FROM python:3.11-slim

WORKDIR /app

# Every install below is pinned by docker/constraints.txt: the library keeps its
# broad ranges in pyproject.toml, the container gets one known-working set.
# See docker/overrides.txt for how it is regenerated.
COPY docker/constraints.txt docker/constraints.txt

# build-essential is only needed for compilation; purge it in the same layer.
# Pre-install CPU-only torch + torchvision from the same index so their
# compiled operators match (mismatched builds break torchvision::nms, which
# in turn breaks the transformers/sentence-transformers import chain).
# PyPI stays as a secondary index because the PyTorch index does not carry
# torch's own dependencies; the CPU wheels still win because a local version
# (2.6.0+cpu) sorts above the plain release of the same version.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir -c docker/constraints.txt torch torchvision \
       --index-url https://download.pytorch.org/whl/cpu \
       --extra-index-url https://pypi.org/simple \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY packages/core packages/core
RUN pip install --no-cache-dir -c docker/constraints.txt ./packages/core

COPY pyproject.toml README.md ./
COPY verbatim_rag verbatim_rag
RUN pip install --no-cache-dir -c docker/constraints.txt .

COPY api api

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
