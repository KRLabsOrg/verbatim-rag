"""Errors raised by verbatim_core and verbatim_rag."""

from __future__ import annotations


class VerbatimError(Exception):
    """Base class of every error the library raises on purpose."""

    code = "verbatim_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code

    def __reduce__(self):
        # Keeps the keyword attributes when the error crosses a process boundary.
        return type(self), self.args, self.__dict__


class InvalidInputError(VerbatimError, ValueError):
    """The caller sent something wrong. Only a different input fixes it."""

    code = "invalid_input"

    def __init__(self, message: str, *, code: str | None = None, param: str | None = None) -> None:
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
        super().__init__(message, code=code)
        self.service = service
        self.retryable = retryable
        self.retry_after = retry_after
