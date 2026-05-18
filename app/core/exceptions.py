from __future__ import annotations

from typing import Optional


class OpenAIException(Exception):
    """Base exception for API errors that must match OpenAI-style envelopes."""

    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str,
        code: Optional[str] = None,
        param: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.code = code
        self.param = param


class UpstreamModelError(OpenAIException):
    """Raised when the upstream model/graph returns invalid or malformed output."""

    def __init__(self, message: str = "The upstream model returned an invalid response.") -> None:
        super().__init__(
            status_code=502,
            message=message,
            error_type="upstream_model_error",
            code="upstream_model_error",
        )
