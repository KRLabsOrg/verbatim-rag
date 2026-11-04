"""Checks of the arguments that the stateless interfaces share."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from verbatim_core.errors import ConfigurationError, InvalidInputError
from verbatim_core.types import Chunk

_ConfigT = TypeVar("_ConfigT", bound=BaseModel)


def _default_config(config: _ConfigT | None, config_class: type[_ConfigT]) -> _ConfigT:
    """The default config a constructor was given, or the default of its class."""
    if config is None:
        return config_class()
    if not isinstance(config, config_class):
        msg = f"config must be a {config_class.__name__}."
        raise ConfigurationError(msg, code="invalid_value")
    return config


def _call_config(config: _ConfigT | None, default: _ConfigT, config_class: type[_ConfigT]) -> _ConfigT:
    """The config of one call: the one the call passed, else the default."""
    if config is None:
        return default
    if not isinstance(config, config_class):
        msg = f"config must be a {config_class.__name__}."
        raise InvalidInputError(msg, code="invalid_value", param="config")
    return config


def _config_from_settings(config_class: type[_ConfigT], settings: dict[str, Any]) -> _ConfigT:
    """The config that settings describe, such as the deprecated setting arguments of a
    constructor or a deprecated setter.
    """
    try:
        return config_class(**settings)
    except ValidationError as e:
        names = ", ".join(str(error["loc"][0]) for error in e.errors())
        msg = f"Invalid value for {names}. See {config_class.__name__} for the valid values."
        raise ConfigurationError(
            msg,
            code="invalid_value",
        ) from e


def _check_unique_ids(chunks: Sequence[Chunk]) -> None:
    positions: dict[str, int] = {}
    for position, chunk in enumerate(chunks):
        if chunk.id in positions:
            msg = (
                f"chunks[{position}] has the same id as chunks[{positions[chunk.id]}]. "
                "Chunk ids must be unique within a call."
            )
            raise InvalidInputError(
                msg,
                code="invalid_value",
                param=f"chunks[{position}].id",
            )
        positions[chunk.id] = position
