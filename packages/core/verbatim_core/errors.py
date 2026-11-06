"""Errors raised by verbatim_core and verbatim_rag."""

from __future__ import annotations

from typing import Any

from typing_extensions import override


class VerbatimError(Exception):
    """Base class of every error the library raises on purpose."""

    code = "verbatim_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        """Create the error.

        Args:
            message: What the problem is and how to fix it.
            code: A stable snake_case code, in place of the default code of the class.

        """
        super().__init__(message)
        if code is not None:
            self.code = code

    @override
    def __reduce__(self) -> tuple[type[VerbatimError], tuple[Any, ...], dict[str, Any]]:
        """Keep the keyword attributes when the error crosses a process boundary."""
        return type(self), self.args, self.__dict__


class InvalidInputError(VerbatimError, ValueError):
    """The caller sent something wrong. Only a different input fixes it."""

    code = "invalid_input"

    def __init__(self, message: str, *, code: str | None = None, param: str | None = None) -> None:
        """Create the error.

        Args:
            message: What the problem is and how to fix it.
            code: A stable snake_case code, in place of the default code of the class.
            param: The argument that holds the bad value, for example `chunks[3].id`.

        """
        super().__init__(message, code=code)
        self.param = param


class ConfigurationError(VerbatimError):
    """The deployment is wrong. Only the operator can fix it."""

    code = "configuration_error"


class UpstreamError(VerbatimError):
    """A service the library calls failed or did not answer in time."""

    code = "upstream_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        service: str = "unknown",
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        """Create the error.

        Args:
            message: What the problem is and how to fix it.
            code: A stable snake_case code, in place of the default code of the class.
            service: The service that failed, for example `llm`.
            retryable: Whether the same call may succeed when retried.
            retry_after: Seconds to wait before retrying, when the service said so.

        """
        super().__init__(message, code=code)
        self.service = service
        self.retryable = retryable
        self.retry_after = retry_after
