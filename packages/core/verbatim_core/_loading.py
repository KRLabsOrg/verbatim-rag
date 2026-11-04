"""Loading models: the errors of a failed load, and the record of a successful one."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

from .errors import ConfigurationError


@contextmanager
def _load_errors_translated(model_id: str) -> Iterator[None]:
    """Turn a missing package, a failed download or a missing model file into a
    ConfigurationError."""
    try:
        yield
    except ImportError as e:
        package = e.name or "a package that is not installed"
        raise ConfigurationError(
            f"The model {model_id} needs {package}. Install it with pip install {package}.",
            code="missing_dependency",
        ) from e
    except OSError as e:
        raise ConfigurationError(
            f"Could not load the model {model_id}. Check the model id or path, and the "
            "connection to Hugging Face.",
            code="model_not_loaded",
        ) from e


def _log_model_loaded(logger: logging.Logger, model_id: str, device: str, started: float) -> None:
    """Log model_loaded on the logger of the module that loaded the model."""
    duration_ms = round((time.perf_counter() - started) * 1000)
    logger.info(
        "Loaded %s on %s in %d ms",
        model_id,
        device,
        duration_ms,
        extra={
            "event": "model_loaded",
            "model": model_id,
            "device": device,
            "duration_ms": duration_ms,
        },
    )
