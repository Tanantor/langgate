from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response

from langgate.core.logging import configure_logger
from langgate.server.api.api_v1.api import registry_router
from langgate.server.core.config import settings
from langgate.server.core.logging import (
    debug_request,
    get_logger,
    is_debug,
    set_structlog_request_context,
)

configure_logger(json_logs=settings.JSON_LOGS)

logger = get_logger(__name__)
IS_DEBUG = is_debug()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Handle application startup and shutdown events."""
    logger.info("application_startup")
    yield
    logger.info("application_shutdown")


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)

if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def logging_middleware(request: Request, call_next) -> Response:
    set_structlog_request_context(request)
    if IS_DEBUG:
        await debug_request(logger, request)
    response: Response = await call_next(request)
    return response


app.include_router(registry_router, prefix=settings.API_V1_STR)
