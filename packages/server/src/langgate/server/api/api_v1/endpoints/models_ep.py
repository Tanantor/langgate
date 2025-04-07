from fastapi import APIRouter

from langgate.core.logging import get_logger
from langgate.registry.models import LLMInfo
from langgate.server.api.services.registry_api import ModelRegistryAPI

logger = get_logger(__name__)
router = APIRouter()
registry_service = ModelRegistryAPI()


@router.get("", response_model=list[LLMInfo])
async def list_models():
    return await registry_service.list_models()


@router.get("/{model_id:path}", response_model=LLMInfo)
async def get_model_info(
    *,
    model_id: str,
):
    model = model_id
    return await registry_service.get_model_info(model)
