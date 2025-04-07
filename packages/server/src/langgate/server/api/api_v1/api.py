from fastapi import APIRouter

from langgate.server.api.api_v1.endpoints import models_ep

registry_router = APIRouter()


@registry_router.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}


registry_router.include_router(models_ep.router, prefix="/models", tags=["models"])
