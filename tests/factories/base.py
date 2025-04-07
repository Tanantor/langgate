from typing import Any, Generic, TypeVar

from factory.base import DictFactory
from pydantic import BaseModel

PydanticModelType = TypeVar("PydanticModelType", bound=BaseModel)


class BasePydanticFactory(DictFactory, Generic[PydanticModelType]):
    @classmethod
    def create(cls, **kwargs: Any) -> PydanticModelType:
        return super().create(**kwargs)

    @classmethod
    def create_batch(cls, size: int, **kwargs: Any) -> list[PydanticModelType]:
        return super().create_batch(size, **kwargs)

    @classmethod
    def create_dump(cls, **kwargs: Any) -> dict[str, Any]:
        instance: PydanticModelType = super().create(**kwargs)
        return instance.model_dump(mode="json")

    @classmethod
    def create_batch_dumps(cls, size: int, **kwargs: Any) -> list[dict[str, Any]]:
        instances: list[PydanticModelType] = super().create_batch(size, **kwargs)
        return [instance.model_dump(mode="json") for instance in instances]
