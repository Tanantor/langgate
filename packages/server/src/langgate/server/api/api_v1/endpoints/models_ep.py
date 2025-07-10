from fastapi import APIRouter

from langgate.core.logging import get_logger
from langgate.registry.models import ImageModelInfo, LLMInfo
from langgate.server.api.services.registry_api import ModelRegistryAPI

logger = get_logger(__name__)
router = APIRouter()
registry_service = ModelRegistryAPI()


# LLM endpoints
@router.get("/llms", response_model=list[LLMInfo])
async def list_llms():
    return await registry_service.list_llms()


@router.get("/llms/{model_id:path}", response_model=LLMInfo)
async def get_llm_info(
    *,
    model_id: str,
):
    return await registry_service.get_llm_info(model_id)


# Image model endpoints
@router.get("/images", response_model=list[ImageModelInfo])
async def list_image_models():
    return await registry_service.list_image_models()


@router.get("/images/{model_id:path}", response_model=ImageModelInfo)
async def get_image_model_info(
    *,
    model_id: str,
):
    return await registry_service.get_image_model_info(model_id)
