"""Server-specific logging utilities."""

import uuid

import structlog
from starlette.requests import Request

from langgate.core.logging import StructLogger, get_logger, is_debug


def set_structlog_request_context(request: Request) -> None:
    """Set request context for structlog."""
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        method=request.method,
        request_path=request.url.path,
        request_client=str(request.client),
        request_user_agent=request.headers.get("User-Agent", None),
        request_id=request.headers.get("X-Request-ID", str(uuid.uuid4())),
    )


async def debug_request(logger: StructLogger, request: Request) -> None:
    """Log request details."""
    body = None
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.json()
        except Exception as e:
            body = f"Error reading body: {str(e)}"
    await logger.adebug(
        "debug_request",
        method=request.method,
        url=request.url,
        body=body,
        headers=request.headers,
    )


# Re-export these functions from core for convenience
__all__ = ["set_structlog_request_context", "debug_request", "get_logger", "is_debug"]
