import random
import string
from collections.abc import Callable, Generator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import structlog
from fastapi import FastAPI
from httpx import AsyncClient as AsyncTestClient
from structlog.testing import LogCapture
from structlog.typing import EventDict


def random_lower_string() -> str:
    return "".join(random.choices(string.ascii_lowercase, k=32))


def raise_after_n_calls(
    original_func: Callable[..., Any], n: int, error: Exception | None = None
) -> tuple[Callable[..., Any], list[int]]:
    if error is None:
        error = Exception("Error")
    call_count = [0]  # mutable to make it nonlocal when returned

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count[0] += 1
        if call_count[0] > n:
            raise error
        return original_func(*args, **kwargs)

    return side_effect, call_count


@contextmanager
def patch_dependency(
    client: AsyncTestClient, dependency: Callable, replacement: Callable
):
    """Replace a FastAPI dependency (Depends) with another. This acts as a
    contextmanager, restoring the dependency overrides on __exit__().
    """
    app: FastAPI = client._transport.app  # type:ignore[attr-defined]
    original_overrides = deepcopy(app.dependency_overrides)
    try:
        app.dependency_overrides[dependency] = replacement
        yield
    finally:
        app.dependency_overrides = original_overrides


@contextmanager
def capture_logs() -> Generator[list[EventDict]]:
    """
    Context manager that appends all logging statements to its yielded list
    while it is active. Disables all configured processors for the duration
    of the context manager.
    Warning: this is **not** thread-safe!

    Returns:
        A list of captured log entries (as event dictionaries)

    Example:
        with capture_logs() as caplogs:
            logger.info("test message")
        assert caplogs[-1]["event"] == "test message"
    """
    cap = LogCapture()
    # Save original processors configured with structlog.configure()
    config = structlog.get_config()
    processors = config.get("processors", [])
    original_processors = processors.copy()

    try:
        # Clear processors and use LogCapture for testing
        processors.clear()
        processors.append(cap)
        structlog.configure(processors=processors)
        yield cap.entries
    finally:
        # Remove LogCapture and restore original processors
        processors.clear()
        processors.extend(original_processors)
        structlog.configure(processors=processors)
